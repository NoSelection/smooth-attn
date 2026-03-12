"""
LONG-CONTEXT FAMILY SWEEP at ctx=2048
======================================
Find which (alpha, theta, power) beats softmax at long context.
Uses eager attention (configurable family) — slower but parameterizable.
Single seed, 4000 steps, batch=4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys
import os

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
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from smooth_attn import FamilyConfig, softplus_norm_causal_eager

DEVICE = 'cuda'
RESID_DROPOUT = 0.1

# ============================================================
# Data
# ============================================================

with open(os.path.join(ROOT, 'shakespeare.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split, block_size, batch_size):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix]).to(DEVICE)
    y = torch.stack([d[i+1:i+block_size+1] for i in ix]).to(DEVICE)
    return x, y


# ============================================================
# Attention modules
# ============================================================

class MHA_SoftmaxBF16(nn.Module):
    def __init__(self, n_embd, n_head, max_ctx):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(RESID_DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            out = out.transpose(1, 2).reshape(B, T, C)
            out = self.proj(out)
        return self.proj_drop(out)


class MHA_SP2NormEager(nn.Module):
    def __init__(self, n_embd, n_head, max_ctx, family):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.family = family
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(RESID_DROPOUT)
        self.register_buffer('tril', torch.tril(torch.ones(max_ctx, max_ctx)))

    def forward(self, x):
        B, T, C = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            # Eager attention with configurable family
            scores = (q.float() @ k.float().transpose(-2, -1)) * (self.hs ** -0.5)
            attn = softplus_norm_causal_eager(scores, family=self.family)
            out = (attn @ v.float()).to(q.dtype)
            out = out.transpose(1, 2).reshape(B, T, C)
            out = self.proj(out)
        return self.proj_drop(out)


# ============================================================
# Model
# ============================================================

def build_model(variant, n_embd, n_head, n_layer, max_ctx, family=None):
    if variant == 'softmax_bf16':
        mha_fn = lambda: MHA_SoftmaxBF16(n_embd, n_head, max_ctx)
    else:
        mha_fn = lambda: MHA_SP2NormEager(n_embd, n_head, max_ctx, family)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.sa = mha_fn()
            self.ffwd = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd), nn.Dropout(RESID_DROPOUT),
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
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=DEVICE))
            for b in self.blocks:
                x = b(x)
            logits = self.head(self.ln_f(x))
            if targets is None:
                return logits, None
            return logits, F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

    return GPT()


# ============================================================
# Eval & Train
# ============================================================

@torch.no_grad()
def eval_loss(model, block_size, batch_size, n=30):
    model.eval()
    losses = []
    for _ in range(n):
        X, Y = get_batch('val', block_size, batch_size)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(variant, seed, n_embd, n_head, n_layer, max_ctx, batch_size, steps, family=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = build_model(variant, n_embd, n_head, n_layer, max_ctx, family).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    eval_points = [500, 1000, 1500, 2000, 2500, 3000]
    history = []
    next_eval = 0

    start = time.perf_counter()
    for step in range(1, steps + 1):
        X, Y = get_batch('train', max_ctx, batch_size)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if next_eval < len(eval_points) and step == eval_points[next_eval]:
            val = eval_loss(model, max_ctx, batch_size)
            t = time.perf_counter() - start
            ppl = math.exp(val)
            print(f"      step {step:5d} | val {val:.4f} ppl {ppl:.2f} | {t:.0f}s", flush=True)
            history.append({'step': step, 'val': val, 'ppl': ppl})
            next_eval += 1

    total = time.perf_counter() - start
    final_val = history[-1]['val'] if history else float('inf')
    del model, opt
    torch.cuda.empty_cache()
    return final_val, total, history


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("FAMILY SWEEP @ CTX=1024 — FINDING THE LONG-CONTEXT WINNER", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    SEED = 42
    N_EMBD, N_HEAD, N_LAYER = 192, 6, 6
    CTX, BATCH, STEPS = 1024, 8, 3000

    print(f"Config: {N_LAYER}L/{N_HEAD}H/{N_EMBD}d, ctx={CTX}, batch={BATCH}, steps={STEPS}")
    print(f"Seed: {SEED}")
    print(f"Using eager attention (configurable family)")
    print()

    # Softmax baseline
    print("=" * 60, flush=True)
    print("  BASELINE: softmax_bf16", flush=True)
    print("=" * 60, flush=True)
    sm_val, sm_time, sm_hist = train_one(
        'softmax_bf16', SEED, N_EMBD, N_HEAD, N_LAYER, CTX, BATCH, STEPS
    )
    print(f"  Final: {sm_val:.4f} in {sm_time:.0f}s\n", flush=True)

    # Family sweep
    configs = [
        # (alpha, theta, power) — focused on sharper variants for long context
        (2.0, 0.5, 2),   # current (baseline sp2norm)
        (3.0, 0.5, 2),   # sharper
        (4.0, 0.5, 2),   # much sharper
        (3.0, 0.3, 2),   # sharp + low threshold
        (3.0, 0.5, 3),   # sharp + cubic
        (4.0, 0.5, 3),   # very sharp + cubic
    ]

    results = []
    for alpha, theta, power in configs:
        family = FamilyConfig(alpha=alpha, theta=theta, power=power)
        label = f"a={alpha}, t={theta}, p={power}"
        print("=" * 60, flush=True)
        print(f"  {label}", flush=True)
        print("=" * 60, flush=True)
        val, t, hist = train_one(
            'sp2norm_eager', SEED, N_EMBD, N_HEAD, N_LAYER, CTX, BATCH, STEPS, family=family
        )
        gap = (val - sm_val) / sm_val * 100
        print(f"  Final: {val:.4f} ({gap:+.2f}% vs softmax) in {t:.0f}s\n", flush=True)
        results.append((label, val, gap, t, hist))

    # Sort by val loss
    results.sort(key=lambda x: x[1])

    print("\n" + "=" * 80, flush=True)
    print("SWEEP RESULTS — RANKED BY VAL LOSS", flush=True)
    print("=" * 80, flush=True)
    print(f"\n  Softmax baseline: {sm_val:.4f}\n")
    print(f"  {'Config':>25s}  {'Val':>8s}  {'vs SM':>8s}  {'Time':>6s}  {'Beat?':>6s}")
    print(f"  {'-'*60}")
    for label, val, gap, t, hist in results:
        beat = "YES" if val < sm_val else "no"
        marker = " <<<" if val < sm_val else ""
        print(f"  {label:>25s}  {val:8.4f}  {gap:+7.2f}%  {t:5.0f}s  {beat:>6s}{marker}")

    best_label, best_val, best_gap, _, _ = results[0]
    print(f"\n  BEST: {best_label} at {best_val:.4f} ({best_gap:+.2f}% vs softmax)")
    if best_val < sm_val:
        print(f"  >>> SP2NORM BEATS SOFTMAX AT CTX=1024! <<<")
    else:
        print(f"  Closest gap: {best_gap:+.2f}%")

    print(f"\n{'='*80}", flush=True)
    print("DONE", flush=True)
    print("=" * 80, flush=True)
