"""
Benchmark mutation ideas against the current squareplus winner.

This sandbox keeps the production path untouched while we compare:
- normalized shifted ReLU-squared
- depth-evolving thresholds
- denominator-free squareplus
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smooth_attn import kernels as base


EPS = 1e-12
DEFAULT_THETA = 0.5
DEFAULT_GAMMA = 0.05
DEFAULT_LAYERS = 12


@dataclass(frozen=True)
class NormalizedMutationRow:
    dtype: torch.dtype
    batch: int
    heads: int
    seq_len: int
    torch_softmax_us: float
    baseline_us: float
    relu_us: float
    baseline_err: float
    relu_err: float
    baseline_sum_ok: bool
    relu_sum_ok: bool
    baseline_meta: dict
    relu_meta: dict


@dataclass(frozen=True)
class DepthScheduleRow:
    dtype: torch.dtype
    batch: int
    heads: int
    seq_len: int
    fixed_avg_us: float
    scheduled_avg_us: float
    theta_first: float
    theta_last: float
    entropy_first: float
    entropy_last: float
    maxprob_first: float
    maxprob_last: float


@dataclass(frozen=True)
class RawMutationRow:
    dtype: torch.dtype
    batch: int
    heads: int
    seq_len: int
    baseline_us: float
    raw_us: float
    raw_err: float
    raw_mean_abs: float
    raw_mean_max: float
    raw_meta: dict


def _squareplus_p2_eager(x, theta=DEFAULT_THETA):
    family = base.FamilyConfig(alpha=2.0, theta=theta, power=2)
    return base.softplus_norm_causal_eager(x, family=family)


def _relu_sq_norm_eager(x, theta=DEFAULT_THETA):
    x_attn = base._prepare_attention_input(x)
    mask = base._causal_mask(x_attn.shape[-1], x_attn.device)
    y = torch.relu(x_attn - theta)
    y = y * y
    y = y.masked_fill(mask, 0.0)
    return y / (y.sum(dim=-1, keepdim=True) + EPS)


def _raw_squareplus_p2_eager(x, theta=DEFAULT_THETA):
    x_attn = base._prepare_attention_input(x)
    mask = base._causal_mask(x_attn.shape[-1], x_attn.device)
    arg = 2.0 * (x_attn - theta)
    y = 0.5 * (arg + torch.sqrt(arg * arg + 4.0))
    y = y * y
    return y.masked_fill(mask, 0.0)


@triton.jit
def _relu_sq_causal_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    THETA: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)

        y = tl.maximum(x - THETA, 0.0)
        y = y * y
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        out = tl.fdiv(y, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


@triton.jit
def _raw_squareplus_p2_causal_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    THETA: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)

        arg = x + x - (2.0 * THETA)
        tmp = arg + tl.sqrt(arg * arg + 4.0)
        y = 0.25 * tmp * tmp
        y = tl.where(causal_mask, y, 0.0)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, y, mask=mask)


def _launch_mutation_out(name, kernel, out, x, *, theta=DEFAULT_THETA, return_meta=False):
    x_attn = base._prepare_attention_input(x)
    kernel_out, user_out = base._prepare_output(out, x_attn)
    result, meta = base._launch_attention_kernel_out(
        name,
        kernel,
        x_attn,
        kernel_out,
        return_meta=True,
        extra_cache_key=(float(theta),),
        kernel_kwargs={"THETA": float(theta)},
    )
    result = base._finalize_output(result, user_out)
    if return_meta:
        return result, meta
    return result


def relu_sq_causal_triton_out(out, x, *, theta=DEFAULT_THETA, return_meta=False):
    return _launch_mutation_out(
        "relu_sq_causal",
        _relu_sq_causal_fwd,
        out,
        x,
        theta=theta,
        return_meta=return_meta,
    )


def raw_squareplus_p2_causal_triton_out(out, x, *, theta=DEFAULT_THETA, return_meta=False):
    return _launch_mutation_out(
        "raw_squareplus_p2_causal",
        _raw_squareplus_p2_causal_fwd,
        out,
        x,
        theta=theta,
        return_meta=return_meta,
    )


def _entropy(y):
    probs = y.clamp_min(1e-9)
    return (-(probs * probs.log()).sum(dim=-1)).mean().item()


def benchmark_normalized_showdown(dtypes, lengths, *, batch, heads, warmup, iters, theta):
    results = []
    for dtype in dtypes:
        for seq_len in lengths:
            x = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=dtype)
            x_ref = x.float()

            baseline_ref = _squareplus_p2_eager(x_ref, theta=theta)
            relu_ref = _relu_sq_norm_eager(x_ref, theta=theta)

            out_base = torch.empty_like(x)
            out_relu = torch.empty_like(x)

            our_base, base_meta = base.softplus_norm_causal_triton_out(
                out_base,
                x,
                family=base.FamilyConfig(alpha=2.0, theta=theta, power=2),
                return_meta=True,
            )
            our_relu, relu_meta = relu_sq_causal_triton_out(
                out_relu,
                x,
                theta=theta,
                return_meta=True,
            )

            atol = 1e-4 if dtype == torch.float32 else 5e-3
            torch_us = base.bench(base.softmax_causal_eager, x, warmup=warmup, iters=iters)
            base_us = base.bench_out(
                lambda out, z: base.softplus_norm_causal_triton_out(
                    out,
                    z,
                    family=base.FamilyConfig(alpha=2.0, theta=theta, power=2),
                ),
                out_base,
                x,
                warmup=warmup,
                iters=iters,
            )
            relu_us = base.bench_out(
                lambda out, z: relu_sq_causal_triton_out(out, z, theta=theta),
                out_relu,
                x,
                warmup=warmup,
                iters=iters,
            )

            results.append(
                NormalizedMutationRow(
                    dtype=dtype,
                    batch=batch,
                    heads=heads,
                    seq_len=seq_len,
                    torch_softmax_us=torch_us,
                    baseline_us=base_us,
                    relu_us=relu_us,
                    baseline_err=(baseline_ref - our_base.float()).abs().max().item(),
                    relu_err=(relu_ref - our_relu.float()).abs().max().item(),
                    baseline_sum_ok=torch.allclose(
                        our_base.float().sum(-1),
                        torch.ones_like(baseline_ref.sum(-1)),
                        atol=atol,
                    ),
                    relu_sum_ok=torch.allclose(
                        our_relu.float().sum(-1),
                        torch.ones_like(relu_ref.sum(-1)),
                        atol=atol,
                    ),
                    baseline_meta=base_meta,
                    relu_meta=relu_meta,
                )
            )
    return results


def benchmark_depth_schedule(dtypes, lengths, *, batch, heads, warmup, iters, theta, gamma, layers):
    results = []
    families = [
        base.FamilyConfig(alpha=2.0, theta=theta + gamma * layer_idx, power=2)
        for layer_idx in range(layers)
    ]
    fixed_family = base.FamilyConfig(alpha=2.0, theta=theta, power=2)

    for dtype in dtypes:
        for seq_len in lengths:
            x = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=dtype)
            out = torch.empty_like(x)

            # Warm each scheduled kernel once so timing measures the steady state.
            for family in families:
                base.softplus_norm_causal_triton_out(out, x, family=family)

            fixed_total = base.bench(
                lambda: [base.softplus_norm_causal_triton_out(out, x, family=fixed_family) for _ in families],
                warmup=warmup,
                iters=iters,
            )
            sched_total = base.bench(
                lambda: [base.softplus_norm_causal_triton_out(out, x, family=family) for family in families],
                warmup=warmup,
                iters=iters,
            )

            x_ref = x.float()
            first = base.softplus_norm_causal_eager(x_ref, family=families[0])
            last = base.softplus_norm_causal_eager(x_ref, family=families[-1])

            results.append(
                DepthScheduleRow(
                    dtype=dtype,
                    batch=batch,
                    heads=heads,
                    seq_len=seq_len,
                    fixed_avg_us=fixed_total / layers,
                    scheduled_avg_us=sched_total / layers,
                    theta_first=families[0].theta,
                    theta_last=families[-1].theta,
                    entropy_first=_entropy(first),
                    entropy_last=_entropy(last),
                    maxprob_first=first.max(dim=-1).values.mean().item(),
                    maxprob_last=last.max(dim=-1).values.mean().item(),
                )
            )
    return results


def benchmark_denominator_free(dtypes, lengths, *, batch, heads, warmup, iters, theta):
    results = []
    for dtype in dtypes:
        for seq_len in lengths:
            x = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=dtype)
            x_ref = x.float()

            raw_ref = _raw_squareplus_p2_eager(x_ref, theta=theta)
            out_base = torch.empty_like(x)
            out_raw = torch.empty_like(x)

            _, _ = base.softplus_norm_causal_triton_out(
                out_base,
                x,
                family=base.FamilyConfig(alpha=2.0, theta=theta, power=2),
                return_meta=True,
            )
            our_raw, raw_meta = raw_squareplus_p2_causal_triton_out(
                out_raw,
                x,
                theta=theta,
                return_meta=True,
            )

            base_us = base.bench_out(
                lambda out, z: base.softplus_norm_causal_triton_out(
                    out,
                    z,
                    family=base.FamilyConfig(alpha=2.0, theta=theta, power=2),
                ),
                out_base,
                x,
                warmup=warmup,
                iters=iters,
            )
            raw_us = base.bench_out(
                lambda out, z: raw_squareplus_p2_causal_triton_out(out, z, theta=theta),
                out_raw,
                x,
                warmup=warmup,
                iters=iters,
            )

            results.append(
                RawMutationRow(
                    dtype=dtype,
                    batch=batch,
                    heads=heads,
                    seq_len=seq_len,
                    baseline_us=base_us,
                    raw_us=raw_us,
                    raw_err=(raw_ref - our_raw.float()).abs().max().item(),
                    raw_mean_abs=our_raw.float().abs().mean().item(),
                    raw_mean_max=our_raw.float().max(dim=-1).values.mean().item(),
                    raw_meta=raw_meta,
                )
            )
    return results


def gradient_health(theta, *, seq_len=128, batch=2, heads=4):
    x = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=torch.float32)

    variants = {
        "baseline": lambda t: _squareplus_p2_eager(t.float(), theta=theta),
        "relu_sq": lambda t: _relu_sq_norm_eager(t.float(), theta=theta),
        "raw_sq": lambda t: _raw_squareplus_p2_eager(t.float(), theta=theta),
    }

    rows = []
    for name, fn in variants.items():
        x_var = x.detach().clone().requires_grad_(True)
        y = fn(x_var)
        loss = (y * upstream).sum()
        loss.backward()
        grad = x_var.grad.detach().float()
        rows.append(
            {
                "name": name,
                "grad_nonzero": (grad.abs() > 1e-8).float().mean().item(),
                "grad_finite": bool(torch.isfinite(grad).all().item()),
                "grad_mean_abs": grad.abs().mean().item(),
            }
        )
    return rows


def _geo_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _print_normalized(results):
    print("\n--- Normalized Showdown ---")
    print("  Compares shifted ReLU-squared against the current squareplus winner.")
    print(
        f"  {'dtype':>6s} {'B,H,T':>12s} {'torch.sm':>10s} {'base_sq':>10s}"
        f" {'relu_sq':>10s} {'relu/base':>10s} {'relu/torch':>11s}"
        f" {'base_err':>10s} {'relu_err':>10s}"
    )
    print("  " + "-" * 106)

    grouped = {}
    for row in results:
        label = base._dtype_label(row.dtype)
        grouped.setdefault(label, {"relu_vs_base": [], "relu_vs_torch": []})
        grouped[label]["relu_vs_base"].append(row.relu_us / row.baseline_us)
        grouped[label]["relu_vs_torch"].append(row.relu_us / row.torch_softmax_us)

        shape = f"{row.batch}x{row.heads}x{row.seq_len}"
        print(
            f"  {label:>6s} {shape:>12s} {row.torch_softmax_us:8.1f}us"
            f" {row.baseline_us:8.1f}us {row.relu_us:8.1f}us"
            f" {row.relu_us / row.baseline_us:8.2f}x"
            f" {row.relu_us / row.torch_softmax_us:9.2f}x"
            f" {row.baseline_err:10.2e} {row.relu_err:10.2e}"
        )

    print("\n  Correctness:")
    for row in results:
        print(
            f"  {base._dtype_label(row.dtype):>6s} T={row.seq_len:<4d}"
            f" base_sum={row.baseline_sum_ok}"
            f" relu_sum={row.relu_sum_ok}"
        )

    print("\n  Geo means by dtype:")
    for label, values in grouped.items():
        print(
            f"  {label:>6s}"
            f" relu/base={_geo_mean(values['relu_vs_base']):.3f}x"
            f" relu/torch={_geo_mean(values['relu_vs_torch']):.3f}x"
        )


def _print_depth_schedule(results):
    print("\n--- Depth Schedule Probe ---")
    print("  Compares fixed-theta squareplus against a depth-evolving theta schedule.")
    print(
        f"  {'dtype':>6s} {'B,H,T':>12s} {'fixed/layer':>12s} {'sched/layer':>12s}"
        f" {'sched/fixed':>11s} {'theta0':>8s} {'thetaL':>8s}"
        f" {'H0':>8s} {'HL':>8s} {'p0':>8s} {'pL':>8s}"
    )
    print("  " + "-" * 112)
    for row in results:
        shape = f"{row.batch}x{row.heads}x{row.seq_len}"
        print(
            f"  {base._dtype_label(row.dtype):>6s} {shape:>12s}"
            f" {row.fixed_avg_us:10.1f}us {row.scheduled_avg_us:10.1f}us"
            f" {row.scheduled_avg_us / row.fixed_avg_us:9.2f}x"
            f" {row.theta_first:8.2f} {row.theta_last:8.2f}"
            f" {row.entropy_first:8.3f} {row.entropy_last:8.3f}"
            f" {row.maxprob_first:8.3f} {row.maxprob_last:8.3f}"
        )


def _print_raw(results):
    print("\n--- Denominator-Free Probe ---")
    print("  This is not normalized attention anymore; compare speed and scale only.")
    print(
        f"  {'dtype':>6s} {'B,H,T':>12s} {'base_sq':>10s} {'raw_sq':>10s}"
        f" {'raw/base':>10s} {'raw_err':>10s} {'raw_mean':>10s} {'raw_max':>10s}"
    )
    print("  " + "-" * 98)

    grouped = {}
    for row in results:
        label = base._dtype_label(row.dtype)
        grouped.setdefault(label, [])
        grouped[label].append(row.raw_us / row.baseline_us)
        shape = f"{row.batch}x{row.heads}x{row.seq_len}"
        print(
            f"  {label:>6s} {shape:>12s} {row.baseline_us:8.1f}us"
            f" {row.raw_us:8.1f}us {row.raw_us / row.baseline_us:8.2f}x"
            f" {row.raw_err:10.2e} {row.raw_mean_abs:10.3e} {row.raw_mean_max:10.3e}"
        )

    print("\n  Geo means by dtype:")
    for label, values in grouped.items():
        print(f"  {label:>6s} raw/base={_geo_mean(values):.3f}x")


def _print_grad(rows):
    print("\n--- Gradient Health ---")
    for row in rows:
        print(
            f"  {row['name']:>8s}"
            f" grad_nonzero={row['grad_nonzero']:.3f}"
            f" grad_finite={row['grad_finite']}"
            f" grad_mean_abs={row['grad_mean_abs']:.3e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare mutation ideas against the squareplus winner")
    parser.add_argument("--dtypes", default="bf16", help="comma-separated: fp32,fp16,bf16")
    parser.add_argument("--lengths", default="128,256,512,1024", help="comma-separated causal sequence lengths")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--quick", action="store_true", help="smaller sweep for iteration speed")
    parser.add_argument("--check-grad", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dtypes = base._parse_dtype_list(args.dtypes)
    lengths = base._parse_int_list(args.lengths)

    if args.quick:
        lengths = lengths[: min(2, len(lengths))]
        args.warmup = min(args.warmup, 20)
        args.iters = min(args.iters, 100)

    base._print_header("MUTATION SHOWDOWN")
    print(f"GPU: {torch.cuda.get_device_name(base._current_device_index())}")
    print(f"Baseline: {base.DEFAULT_FAMILY.label}")
    print(f"Shifted ReLU theta: {args.theta:g}")
    print(f"Depth schedule: theta_l = {args.theta:g} + {args.gamma:g} * l, layers={args.layers}")

    probe = torch.randn(
        args.batch,
        args.heads,
        lengths[0],
        lengths[0],
        device="cuda",
        dtype=dtypes[0],
    )
    _, base_meta = base.softplus_norm_causal_triton_out(
        torch.empty_like(probe),
        probe,
        family=base.FamilyConfig(alpha=2.0, theta=args.theta, power=2),
        return_meta=True,
    )
    _, relu_meta = relu_sq_causal_triton_out(
        torch.empty_like(probe),
        probe,
        theta=args.theta,
        return_meta=True,
    )
    _, raw_meta = raw_squareplus_p2_causal_triton_out(
        torch.empty_like(probe),
        probe,
        theta=args.theta,
        return_meta=True,
    )
    print("\n--- Launch Meta (first case) ---")
    base._print_launch_meta("baseline", base_meta)
    base._print_launch_meta("relu_sq", relu_meta)
    base._print_launch_meta("raw_sq", raw_meta)

    normalized = benchmark_normalized_showdown(
        dtypes,
        lengths,
        batch=args.batch,
        heads=args.heads,
        warmup=args.warmup,
        iters=args.iters,
        theta=args.theta,
    )
    depth_rows = benchmark_depth_schedule(
        dtypes,
        lengths,
        batch=args.batch,
        heads=args.heads,
        warmup=args.warmup,
        iters=args.iters,
        theta=args.theta,
        gamma=args.gamma,
        layers=args.layers,
    )
    raw_rows = benchmark_denominator_free(
        dtypes,
        lengths,
        batch=args.batch,
        heads=args.heads,
        warmup=args.warmup,
        iters=args.iters,
        theta=args.theta,
    )

    _print_normalized(normalized)
    _print_depth_schedule(depth_rows)
    _print_raw(raw_rows)

    if args.check_grad:
        _print_grad(gradient_health(args.theta, seq_len=lengths[0], batch=args.batch, heads=min(args.heads, 4)))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
