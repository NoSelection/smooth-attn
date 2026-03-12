"""
Long-context training sweep for invented row-adaptive attention formulas.

This keeps the production kernel untouched and evaluates new eager formulas
head-to-head against bf16 softmax in the same GPT training loop.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.stdout.reconfigure(line_buffering=True)


def _find_repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(here, "pyproject.toml")) and os.path.isdir(os.path.join(here, "src")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError("Could not locate repo root")
        here = parent


ROOT = _find_repo_root()
SRC = os.path.join(ROOT, "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from experiments.formula_evolution.search import FormulaCandidate, candidate_causal_eager


DEVICE = "cuda"
RESID_DROPOUT = 0.1


with open(os.path.join(ROOT, "shakespeare.txt"), "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]


def get_batch(split, block_size, batch_size):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix]).to(DEVICE)
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix]).to(DEVICE)
    return x, y


class MHA_SoftmaxBF16(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(RESID_DROPOUT)

    def forward(self, x):
        batch, time_steps, channels = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qkv = self.qkv(x).reshape(batch, time_steps, 3, self.n_head, self.hs)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            out = out.transpose(1, 2).reshape(batch, time_steps, channels)
            out = self.proj(out)
        return self.proj_drop(out)


class MHA_FormulaEager(nn.Module):
    def __init__(self, n_embd, n_head, candidate):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.candidate = candidate
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(RESID_DROPOUT)

    def forward(self, x):
        batch, time_steps, channels = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qkv = self.qkv(x).reshape(batch, time_steps, 3, self.n_head, self.hs)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            scores = (q.float() @ k.float().transpose(-2, -1)) * (self.hs ** -0.5)
            attn = candidate_causal_eager(scores, self.candidate)
            out = (attn @ v.float()).to(q.dtype)
            out = out.transpose(1, 2).reshape(batch, time_steps, channels)
            out = self.proj(out)
        return self.proj_drop(out)


def build_model(variant, n_embd, n_head, n_layer, max_ctx, candidate=None):
    if variant == "softmax_bf16":
        mha_fn = lambda: MHA_SoftmaxBF16(n_embd, n_head)
    else:
        mha_fn = lambda: MHA_FormulaEager(n_embd, n_head, candidate)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.sa = mha_fn()
            self.ffwd = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(RESID_DROPOUT),
            )
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB_SIZE, n_embd)
            self.pos = nn.Embedding(max_ctx, n_embd)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, VOCAB_SIZE)

        def forward(self, idx, targets=None):
            batch, time_steps = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(time_steps, device=DEVICE))
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln_f(x))
            if targets is None:
                return logits, None
            return logits, F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

    return GPT()


@torch.no_grad()
def eval_loss(model, block_size, batch_size, eval_batches=30):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch("val", block_size, batch_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _eval_points(steps, interval):
    points = list(range(interval, steps + 1, interval))
    if not points or points[-1] != steps:
        points.append(steps)
    return points


def train_one(variant, seed, cfg, *, candidate=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_model(
        variant,
        cfg["n_embd"],
        cfg["n_head"],
        cfg["n_layer"],
        cfg["ctx"],
        candidate=candidate,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    eval_points = _eval_points(cfg["steps"], cfg["eval_interval"])
    history = []
    next_eval = 0

    start = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        x, y = get_batch("train", cfg["ctx"], cfg["batch"])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if next_eval < len(eval_points) and step == eval_points[next_eval]:
            val = eval_loss(model, cfg["ctx"], cfg["batch"])
            elapsed = time.perf_counter() - start
            ppl = math.exp(val)
            print(f"      step {step:5d} | val {val:.4f} ppl {ppl:.2f} | {elapsed:.0f}s", flush=True)
            history.append({"step": step, "val": val, "ppl": ppl})
            next_eval += 1

    total = time.perf_counter() - start
    final_val = history[-1]["val"] if history else float("inf")
    print(f"  Final: {final_val:.4f} in {total:.0f}s", flush=True)

    del model, opt
    torch.cuda.empty_cache()
    return final_val, total, history


def default_candidates():
    return [
        FormulaCandidate(center="fixed_theta", activation="squareplus", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="row_max_visible", activation="squareplus", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="row_max_visible", activation="squareplus", alpha=3.0, theta=0.75, power=2),
        FormulaCandidate(center="row_mean_std_visible", activation="squareplus", alpha=1.5, theta=0.0, power=2),
        FormulaCandidate(center="row_mean_std_visible", activation="squareplus", alpha=2.0, theta=0.5, power=2),
        FormulaCandidate(center="row_mean_std_visible", activation="squareplus", alpha=2.0, theta=1.0, power=3),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Train invented long-context attention formulas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {
        "seed": args.seed,
        "ctx": args.ctx,
        "batch": args.batch,
        "steps": args.steps,
        "lr": args.lr,
        "eval_interval": args.eval_interval,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
    }

    print("=" * 80, flush=True)
    print("INVENTED FORMULA SWEEP @ LONG CONTEXT", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    print(
        f"Config: {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}d,"
        f" ctx={cfg['ctx']}, batch={cfg['batch']}, steps={cfg['steps']}",
        flush=True,
    )
    print(f"Seed: {cfg['seed']}", flush=True)
    print("Using eager attention for flexible invented formulas", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  BASELINE: softmax_bf16", flush=True)
    print("=" * 60, flush=True)
    softmax_val, softmax_time, _ = train_one("softmax_bf16", cfg["seed"], cfg)

    results = []
    for candidate in default_candidates():
        print("\n" + "=" * 60, flush=True)
        print(f"  {candidate.label}", flush=True)
        print("=" * 60, flush=True)
        val, total, _ = train_one("formula_eager", cfg["seed"], cfg, candidate=candidate)
        gap = (val - softmax_val) / softmax_val * 100.0
        results.append((candidate.label, val, gap, total))

    results.sort(key=lambda item: item[1])
    print("\n" + "=" * 80, flush=True)
    print("RESULTS - RANKED BY VAL LOSS", flush=True)
    print("=" * 80, flush=True)
    print(f"\n  Softmax baseline: {softmax_val:.4f}\n", flush=True)
    print(f"  {'Config':>42s} {'Val':>9s} {'vs SM':>9s} {'Time':>8s}", flush=True)
    print("  " + "-" * 74, flush=True)
    for label, val, gap, total in results:
        print(f"  {label:>42s} {val:9.4f} {gap:+8.2f}% {total:7.0f}s", flush=True)

    best_label, best_val, best_gap, _ = results[0]
    print(f"\n  BEST: {best_label}", flush=True)
    print(f"  Closest gap: {best_gap:+.2f}% vs softmax", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
