"""
Evolutionary search for candidate attention normalization formulas.

This stays outside the production package on purpose: the goal is to explore
formula ideas quickly, then promote only the winners into Triton.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smooth_attn import kernels as base


SUPPORTED_SEARCH_SPACES = ("broad", "squareplus_local")
SUPPORTED_CENTERS = ("fixed_theta", "row_mean_visible")
SUPPORTED_ACTIVATIONS = ("squareplus", "softplus", "relu")
DEFAULT_BASELINE = base.FamilyConfig(alpha=2.0, theta=0.5, power=2)
EPS = 1e-12


@dataclass(frozen=True)
class FormulaCandidate:
    center: str = "fixed_theta"
    activation: str = "squareplus"
    alpha: float = 2.0
    theta: float = 0.5
    power: int = 2

    def __post_init__(self):
        if self.center not in SUPPORTED_CENTERS:
            raise ValueError(f"center must be one of {SUPPORTED_CENTERS}, got {self.center}")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {SUPPORTED_ACTIVATIONS}, got {self.activation}"
            )
        if self.power not in base.SUPPORTED_POWERS:
            raise ValueError(f"power must be one of {base.SUPPORTED_POWERS}, got {self.power}")

    @property
    def cache_key(self):
        return (
            self.center,
            self.activation,
            round(float(self.alpha), 4),
            round(float(self.theta), 4),
            int(self.power),
        )

    @property
    def label(self):
        if self.center == "fixed_theta":
            center = f"(x - {self.theta:.3g})"
        else:
            center = "(x - row_mean_visible)"
        return f"{self.activation}({self.alpha:.3g} * {center})^{self.power} / sum"


@dataclass(frozen=True)
class CandidateScore:
    candidate: FormulaCandidate
    score: float
    cand_vs_base: float
    cand_vs_torch: float
    dead_row_frac: float
    grad_nonzero_frac: float
    grad_finite: bool
    sums_ok: bool
    mean_entropy: float
    mean_max_prob: float


def _parse_dtype_list(text):
    return base._parse_dtype_list(text)


def _parse_int_list(text):
    return base._parse_int_list(text)


def _parse_seed_list(text):
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def _geo_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _apply_activation(arg, activation):
    if activation == "squareplus":
        return 0.5 * (arg + torch.sqrt(arg * arg + 4.0))
    if activation == "softplus":
        return F.softplus(arg)
    return torch.relu(arg)


def candidate_causal_eager(x, candidate, *, return_aux=False):
    x_attn = base._prepare_attention_input(x)
    mask = base._causal_mask(x_attn.shape[-1], x_attn.device)
    visible_count = (~mask).sum(dim=-1, keepdim=True).clamp_min(1)

    if candidate.center == "fixed_theta":
        centered = x_attn - candidate.theta
    else:
        visible = x_attn.masked_fill(mask, 0.0)
        row_mean = visible.sum(dim=-1, keepdim=True) / visible_count.to(visible.dtype)
        centered = x_attn - row_mean

    y = _apply_activation(candidate.alpha * centered, candidate.activation)
    if candidate.power == 2:
        y = y * y
    elif candidate.power == 3:
        y = y * y * y
    y = y.masked_fill(mask, 0.0)

    row_sum = y.sum(dim=-1, keepdim=True)
    uniform = (~mask).to(y.dtype) / visible_count.to(y.dtype)
    out = torch.where(row_sum > EPS, y / (row_sum + EPS), uniform)

    if return_aux:
        return out, {"row_sum": row_sum}
    return out


def _make_eval_inputs(dtypes, lengths, batch, heads, seeds):
    inputs = {}
    for seed in seeds:
        torch.manual_seed(seed)
        for dtype in dtypes:
            for seq_len in lengths:
                inputs[(seed, dtype, seq_len)] = torch.randn(
                    batch,
                    heads,
                    seq_len,
                    seq_len,
                    device="cuda",
                    dtype=dtype,
                )
    return inputs


def _softmax_timing_cache(inputs, warmup, iters):
    cache = {}
    for key, x in inputs.items():
        cache[key] = base.bench(base.softmax_causal_eager, x, warmup=warmup, iters=iters)
    return cache


def _baseline_timing_cache(inputs, warmup, iters):
    cache = {}
    for key, x in inputs.items():
        cache[key] = base.bench(
            lambda z: base.softplus_norm_causal_eager(z, family=DEFAULT_BASELINE),
            x,
            warmup=warmup,
            iters=iters,
        )
    return cache


def evaluate_candidate(candidate, *, inputs, softmax_us, baseline_us, warmup, iters):
    cand_times = []
    torch_times = []
    base_times = []
    dead_row_fracs = []
    sum_checks = []
    entropies = []
    max_probs = []

    for (seed, dtype, seq_len), x in inputs.items():
        cand_us = base.bench(
            lambda z: candidate_causal_eager(z, candidate),
            x,
            warmup=warmup,
            iters=iters,
        )
        cand_times.append(cand_us)
        torch_times.append(softmax_us[(seed, dtype, seq_len)])
        base_times.append(baseline_us[(seed, dtype, seq_len)])

        ref_x = x.float()
        y, aux = candidate_causal_eager(ref_x, candidate, return_aux=True)
        row_sum = aux["row_sum"]
        dead_row_fracs.append((row_sum <= EPS).float().mean().item())
        sum_checks.append(
            torch.allclose(
                y.sum(dim=-1),
                torch.ones_like(y.sum(dim=-1)),
                atol=1e-4 if dtype == torch.float32 else 5e-3,
            )
        )
        probs = y.clamp_min(1e-9)
        entropies.append((-(probs * probs.log()).sum(dim=-1)).mean().item())
        max_probs.append(y.max(dim=-1).values.mean().item())

    first_seed, grad_dtype, grad_len = next(iter(inputs.keys()))
    x_grad = torch.randn(
        1,
        2,
        grad_len,
        grad_len,
        device="cuda",
        dtype=grad_dtype,
        requires_grad=True,
    )
    upstream = torch.randn(
        1,
        2,
        grad_len,
        grad_len,
        device="cuda",
        dtype=torch.float32,
    )
    y_grad = candidate_causal_eager(x_grad.float(), candidate)
    loss = (y_grad * upstream).sum()
    loss.backward()
    grad = x_grad.grad.detach().float()
    grad_nonzero_frac = (grad.abs() > 1e-8).float().mean().item()
    grad_finite = bool(torch.isfinite(grad).all().item())

    cand_vs_base = _geo_mean([cand / ref for cand, ref in zip(cand_times, base_times)])
    cand_vs_torch = _geo_mean([cand / ref for cand, ref in zip(cand_times, torch_times)])
    dead_row_frac = sum(dead_row_fracs) / len(dead_row_fracs)
    sums_ok = all(sum_checks)
    mean_entropy = sum(entropies) / len(entropies)
    mean_max_prob = sum(max_probs) / len(max_probs)

    score = 0.0
    score += 1.5 / cand_vs_base
    score += 1.0 / cand_vs_torch
    score += 0.15 * grad_nonzero_frac
    score -= 4.0 * dead_row_frac
    if not sums_ok:
        score -= 100.0
    if not grad_finite:
        score -= 100.0

    return CandidateScore(
        candidate=candidate,
        score=score,
        cand_vs_base=cand_vs_base,
        cand_vs_torch=cand_vs_torch,
        dead_row_frac=dead_row_frac,
        grad_nonzero_frac=grad_nonzero_frac,
        grad_finite=grad_finite,
        sums_ok=sums_ok,
        mean_entropy=mean_entropy,
        mean_max_prob=mean_max_prob,
    )


def _seed_population():
    return [
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=1.0, theta=0.0, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=4.0, theta=0.5, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.25, power=2),
        FormulaCandidate(center="fixed_theta", activation="softplus", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="fixed_theta", activation="relu", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="row_mean_visible", activation="squareplus", alpha=2.0, theta=0.0, power=2),
        FormulaCandidate(center="row_mean_visible", activation="softplus", alpha=2.0, theta=0.0, power=2),
        FormulaCandidate(center="row_mean_visible", activation="relu", alpha=2.0, theta=0.0, power=2),
    ]


def _seed_population_squareplus_local():
    return [
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=1.5, theta=0.5, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.5, theta=0.5, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.375, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.625, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.5, power=1),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.5, power=3),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=1.25, theta=0.25, power=2),
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=3.0, theta=0.75, power=2),
    ]


def _random_candidate_broad(rng):
    center = rng.choice(SUPPORTED_CENTERS)
    activation = rng.choice(SUPPORTED_ACTIVATIONS)
    alpha = rng.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    theta = rng.choice([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0])
    power = rng.choice(base.SUPPORTED_POWERS)
    if center == "row_mean_visible":
        theta = 0.0
    return FormulaCandidate(
        center=center,
        activation=activation,
        alpha=alpha,
        theta=theta,
        power=power,
    )


def _random_candidate_squareplus_local(rng):
    return FormulaCandidate(
        center="fixed_theta",
        activation="squareplus",
        alpha=rng.choice([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]),
        theta=rng.choice([-0.25, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25]),
        power=rng.choice(base.SUPPORTED_POWERS),
    )


def _random_candidate(rng, search_space):
    if search_space == "squareplus_local":
        return _random_candidate_squareplus_local(rng)
    return _random_candidate_broad(rng)


def _mutate_broad(candidate, rng):
    center = candidate.center
    activation = candidate.activation
    alpha = candidate.alpha
    theta = candidate.theta
    power = candidate.power

    if rng.random() < 0.18:
        center = rng.choice(SUPPORTED_CENTERS)
    if rng.random() < 0.22:
        activation = rng.choice(SUPPORTED_ACTIVATIONS)
    if rng.random() < 0.60:
        alpha = max(0.25, min(8.0, alpha * (2.0 ** rng.uniform(-0.75, 0.75))))
    if rng.random() < 0.45:
        theta = max(-2.0, min(2.0, theta + rng.choice([-0.5, -0.25, 0.25, 0.5])))
    if rng.random() < 0.25:
        power = rng.choice(base.SUPPORTED_POWERS)

    if center == "row_mean_visible":
        theta = 0.0

    return FormulaCandidate(
        center=center,
        activation=activation,
        alpha=round(alpha, 4),
        theta=round(theta, 4),
        power=power,
    )


def _mutate_squareplus_local(candidate, rng):
    alpha = candidate.alpha
    theta = candidate.theta
    power = candidate.power

    if rng.random() < 0.75:
        alpha = max(0.25, min(8.0, alpha * (2.0 ** rng.uniform(-0.35, 0.35))))
    if rng.random() < 0.65:
        theta = max(-1.0, min(2.0, theta + rng.choice([-0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25])))
    if rng.random() < 0.20:
        power = rng.choice(base.SUPPORTED_POWERS)

    return FormulaCandidate(
        center="fixed_theta",
        activation="squareplus",
        alpha=round(alpha, 4),
        theta=round(theta, 4),
        power=power,
    )


def _mutate(candidate, rng, search_space):
    if search_space == "squareplus_local":
        return _mutate_squareplus_local(candidate, rng)
    return _mutate_broad(candidate, rng)


def _crossover_broad(left, right, rng):
    center = rng.choice([left.center, right.center])
    activation = rng.choice([left.activation, right.activation])
    alpha = (left.alpha + right.alpha) / 2.0
    theta = (left.theta + right.theta) / 2.0
    power = rng.choice([left.power, right.power])

    if center == "row_mean_visible":
        theta = 0.0

    child = FormulaCandidate(
        center=center,
        activation=activation,
        alpha=round(alpha, 4),
        theta=round(theta, 4),
        power=power,
    )
    return _mutate_broad(child, rng)


def _crossover_squareplus_local(left, right, rng):
    child = FormulaCandidate(
        center="fixed_theta",
        activation="squareplus",
        alpha=round((left.alpha + right.alpha) / 2.0, 4),
        theta=round((left.theta + right.theta) / 2.0, 4),
        power=rng.choice([left.power, right.power]),
    )
    return _mutate_squareplus_local(child, rng)


def _crossover(left, right, rng, search_space):
    if search_space == "squareplus_local":
        return _crossover_squareplus_local(left, right, rng)
    return _crossover_broad(left, right, rng)


def _fill_population(rng, population, target_size, search_space):
    seen = {cand.cache_key for cand in population}
    while len(population) < target_size:
        candidate = _random_candidate(rng, search_space)
        if candidate.cache_key in seen:
            continue
        population.append(candidate)
        seen.add(candidate.cache_key)
    return population


def run_search(args):
    rng = random.Random(args.seed)
    dtypes = _parse_dtype_list(args.dtypes)
    lengths = _parse_int_list(args.lengths)
    seeds = _parse_seed_list(args.seeds) if args.seeds else [args.seed]
    inputs = _make_eval_inputs(dtypes, lengths, args.batch, args.heads, seeds)
    softmax_us = _softmax_timing_cache(inputs, args.warmup, args.iters)
    baseline_us = _baseline_timing_cache(inputs, args.warmup, args.iters)

    if args.search_space == "squareplus_local":
        initial_population = _seed_population_squareplus_local()
    else:
        initial_population = _seed_population()
    population = _fill_population(rng, initial_population, args.population, args.search_space)
    score_cache = {}
    best_scores = []

    print("=" * 80)
    print("FORMULA EVOLUTION SEARCH")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(base._current_device_index())}")
    print(f"Baseline champion: {DEFAULT_BASELINE.label}")
    print(f"Search lengths: {','.join(str(x) for x in lengths)}")
    print(f"Seeds: {','.join(str(seed) for seed in seeds)}")
    print(f"Search space: {args.search_space}")
    print(f"Population={args.population} Generations={args.generations} Elite={args.elite}")

    for generation in range(1, args.generations + 1):
        scored = []
        for candidate in population:
            if candidate.cache_key not in score_cache:
                score_cache[candidate.cache_key] = evaluate_candidate(
                    candidate,
                    inputs=inputs,
                    softmax_us=softmax_us,
                    baseline_us=baseline_us,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            scored.append(score_cache[candidate.cache_key])

        scored.sort(key=lambda item: item.score, reverse=True)
        elites = scored[: args.elite]
        best_scores.append(elites[0])

        print(f"\n--- Generation {generation} ---")
        print(
            f"  {'rank':>4s} {'score':>8s} {'cand/base':>10s} {'cand/torch':>11s}"
            f" {'dead_rows':>10s} {'grad_nonzero':>12s} {'label':>0s}"
        )
        print("  " + "-" * 110)
        for rank, item in enumerate(elites[: min(5, len(elites))], start=1):
            print(
                f"  {rank:4d} {item.score:8.3f} {item.cand_vs_base:9.3f}x"
                f" {item.cand_vs_torch:10.3f}x {item.dead_row_frac:10.3f}"
                f" {item.grad_nonzero_frac:11.3f} {item.candidate.label}"
            )

        next_population = [item.candidate for item in elites]
        seen = {cand.cache_key for cand in next_population}
        while len(next_population) < args.population:
            if rng.random() < 0.55 and len(elites) >= 2:
                left, right = rng.sample(elites, 2)
                child = _crossover(left.candidate, right.candidate, rng, args.search_space)
            else:
                parent = rng.choice(elites).candidate
                child = _mutate(parent, rng, args.search_space)
            if child.cache_key in seen:
                continue
            next_population.append(child)
            seen.add(child.cache_key)

        population = _fill_population(rng, next_population, args.population, args.search_space)

    final_best = max(best_scores, key=lambda item: item.score)
    final_pool = sorted(
        {score_cache[key].candidate.cache_key: score_cache[key] for key in score_cache}.values(),
        key=lambda item: item.score,
        reverse=True,
    )

    print("\n=== Best Found ===")
    print(f"  Formula:      {final_best.candidate.label}")
    print(f"  Score:        {final_best.score:.3f}")
    print(f"  cand/base:    {final_best.cand_vs_base:.3f}x")
    print(f"  cand/torch:   {final_best.cand_vs_torch:.3f}x")
    print(f"  dead_rows:    {final_best.dead_row_frac:.3f}")
    print(f"  grad_nonzero: {final_best.grad_nonzero_frac:.3f}")
    print(f"  grad_finite:  {final_best.grad_finite}")
    print(f"  sums_ok:      {final_best.sums_ok}")
    print(f"  mean_entropy: {final_best.mean_entropy:.3f}")
    print(f"  mean_maxprob: {final_best.mean_max_prob:.3f}")
    print("\n=== Top 10 Overall ===")
    for rank, item in enumerate(final_pool[:10], start=1):
        print(
            f"  {rank:2d}. score={item.score:7.3f}"
            f" cand/base={item.cand_vs_base:6.3f}x"
            f" cand/torch={item.cand_vs_torch:6.3f}x"
            f" dead={item.dead_row_frac:5.3f}"
            f" grad={item.grad_nonzero_frac:5.3f}"
            f" :: {item.candidate.label}"
        )
    print("\n" + "=" * 80)

    if args.save_json:
        payload = []
        for item in final_pool:
            record = asdict(item)
            record["candidate"] = asdict(item.candidate)
            payload.append(record)
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evolutionary search for attention formulas")
    parser.add_argument(
        "--search-space",
        choices=SUPPORTED_SEARCH_SPACES,
        default="broad",
        help="search over all supported families or only squareplus around the current champion",
    )
    parser.add_argument("--dtypes", default="bf16", help="comma-separated: fp32,fp16,bf16")
    parser.add_argument("--lengths", default="128,256,512", help="comma-separated causal sequence lengths")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--seeds",
        default="",
        help="comma-separated evaluation seeds; overrides --seed when provided",
    )
    parser.add_argument("--save-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.elite <= 0 or args.elite > args.population:
        raise ValueError("--elite must be in [1, population]")
    run_search(args)


if __name__ == "__main__":
    main()
