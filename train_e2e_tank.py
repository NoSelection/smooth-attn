"""
TANK vs TANK — FAIR BF16 COMPARISON
=====================================
Both softmax AND sp2norm in bf16. No precision advantage either way.

Variants:
  - softmax_bf16: F.scaled_dot_product_attention (cuDNN flash), bf16
  - sp2norm_fused: Our Triton kernel, bf16
  - softmax_fp32: Original fp32 softmax (reference only)
  - sp2norm_eager: Original fp32 eager (reference only)

10 seeds x 5000 steps. Tank vs tank. No excuses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")

from smooth_attn import DEFAULT_FAMILY, softplus_norm_causal_eager, sp2norm_flash_attention

DEVICE = 'cuda'
ATTN_DROPOUT = 0.0
RESID_DROPOUT = 0.1

# ============================================================
# Data
# ============================================================

_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, 'shakespeare.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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
# Attention functions (eager, fp32)
# ============================================================

def attn_softmax(wei, mask):
    wei = wei.masked_fill(mask, float('-inf'))
    return F.softmax(wei, dim=-1)


def attn_sp2norm_eager(wei, mask):
    del mask
    return softplus_norm_causal_eager(wei.float(), family=DEFAULT_FAMILY)


# ============================================================
# MHA modules
# ============================================================

class MHA_Eager(nn.Module):
    """Materialized T×T attention (fp32)."""
    def __init__(
        self,
        n_embd,
        n_head,
        max_ctx,
        attn_fn,
        *,
        attn_dropout=ATTN_DROPOUT,
        proj_dropout=RESID_DROPOUT,
    ):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.attn_fn = attn_fn
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.register_buffer('tril', torch.tril(torch.ones(max_ctx, max_ctx)))
        self.last_attn_weights = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (self.hs ** -0.5)
        mask = self.tril[:T, :T] == 0
        wei = self.attn_fn(wei, mask)

        if capture:
            self.last_attn_weights = wei.detach()

        wei = self.attn_drop(wei)
        out = (wei @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class MHA_SoftmaxBF16(nn.Module):
    """PyTorch SDPA (cuDNN FlashAttention) in bf16. The production baseline."""
    def __init__(
        self,
        n_embd,
        n_head,
        max_ctx,
        *,
        attn_dropout=ATTN_DROPOUT,
        proj_dropout=RESID_DROPOUT,
    ):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop_p = attn_dropout
        self.proj_drop = nn.Dropout(proj_dropout)
        self.register_buffer('tril', torch.tril(torch.ones(max_ctx, max_ctx)))
        self.last_attn_weights = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Cast to bf16 — same precision as sp2norm_fused
        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        v_bf = v.to(torch.bfloat16)

        # PyTorch SDPA with cuDNN flash attention backend
        drop_p = self.attn_drop_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_bf, k_bf, v_bf, is_causal=True, dropout_p=drop_p
        )
        out = out.to(x.dtype)

        if capture:
            # Compute weights eagerly for diagnostics
            with torch.no_grad():
                wei = (q @ k.transpose(-2, -1)) * (self.hs ** -0.5)
                mask = self.tril[:T, :T] == 0
                wei = wei.masked_fill(mask, float('-inf'))
                self.last_attn_weights = F.softmax(wei, dim=-1).detach()

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class MHA_SP2NormFused(nn.Module):
    """Our sp2norm Triton kernel in bf16."""
    def __init__(
        self,
        n_embd,
        n_head,
        max_ctx,
        *,
        attn_dropout=ATTN_DROPOUT,
        proj_dropout=RESID_DROPOUT,
    ):
        super().__init__()
        if attn_dropout != 0.0:
            raise ValueError("sp2norm_fused comparison expects attention dropout to be disabled")
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.register_buffer('tril', torch.tril(torch.ones(max_ctx, max_ctx)))
        self.last_attn_weights = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        v_bf = v.to(torch.bfloat16)

        out = sp2norm_flash_attention(q_bf, k_bf, v_bf)
        out = out.to(x.dtype)

        if capture:
            with torch.no_grad():
                wei = (q @ k.transpose(-2, -1)) * (self.hs ** -0.5)
                mask = self.tril[:T, :T] == 0
                self.last_attn_weights = attn_sp2norm_eager(wei, mask).detach()

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


# ============================================================
# Model builder
# ============================================================

def build_model(variant, n_embd, n_head, n_layer, max_ctx):
    mha_kwargs = {"attn_dropout": ATTN_DROPOUT, "proj_dropout": RESID_DROPOUT}
    mha_map = {
        'softmax_fp32': lambda: MHA_Eager(n_embd, n_head, max_ctx, attn_softmax, **mha_kwargs),
        'softmax_bf16': lambda: MHA_SoftmaxBF16(n_embd, n_head, max_ctx, **mha_kwargs),
        'sp2norm_eager': lambda: MHA_Eager(n_embd, n_head, max_ctx, attn_sp2norm_eager, **mha_kwargs),
        'sp2norm_fused': lambda: MHA_SP2NormFused(n_embd, n_head, max_ctx, **mha_kwargs),
    }
    mha_fn = mha_map[variant]

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

        def forward(self, x, capture=False):
            x = x + self.sa(self.ln1(x), capture=capture)
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

        def forward(self, idx, targets=None, capture=False):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=DEVICE))
            for b in self.blocks:
                x = b(x, capture=capture)
            logits = self.head(self.ln_f(x))
            if targets is None:
                return logits, None
            return logits, F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        def get_all_heads(self):
            for b in self.blocks:
                yield b.sa

        def generate(self, idx, n_tok, temp=0.8):
            for _ in range(n_tok):
                idx_c = idx[:, -max_ctx:]
                logits, _ = self(idx_c)
                logits = logits[:, -1, :] / temp
                probs = F.softmax(logits, dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
            return idx

    return GPT()


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(model, block_size, batch_size, n_batches=10):
    model.eval()
    sinks, pos0s, sparses = [], [], []

    for _ in range(n_batches):
        X, _ = get_batch('val', block_size, batch_size)
        model(X, capture=True)
        for mha in model.get_all_heads():
            w = mha.last_attn_weights
            if w is None:
                continue
            start = min(16, w.shape[-1] - 1)
            a0 = w[:, :, start:, 0]
            sinks.append((a0 > 0.30).float().mean().item())
            pos0s.append(a0.mean().item())
            T = w.shape[-1]
            tril = torch.tril(torch.ones(T, T, device=DEVICE))
            n_unmask = tril.sum()
            sp = ((w < 0.01) & tril.unsqueeze(0).unsqueeze(0).bool()).float().sum() / (w.shape[0] * w.shape[1] * n_unmask)
            sparses.append(sp.item())

    model.train()
    if not sinks:
        return {'sink': 0, 'pos0': 0, 'sparse': 0}
    return {
        'sink': sum(sinks) / len(sinks),
        'pos0': sum(pos0s) / len(pos0s),
        'sparse': sum(sparses) / len(sparses),
    }


@torch.no_grad()
def eval_loss(model, block_size, batch_size, n=50):
    model.eval()
    losses = []
    for _ in range(n):
        X, Y = get_batch('val', block_size, batch_size)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ============================================================
# Training
# ============================================================

def train_variant(variant, seed, n_embd, n_head, n_layer, max_ctx, batch_size, steps):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = build_model(variant, n_embd, n_head, n_layer, max_ctx).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup for fused variants (Triton compilation)
    if 'fused' in variant or 'bf16' in variant:
        for _ in range(3):
            X, Y = get_batch('train', max_ctx, batch_size)
            _, loss = model(X, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = build_model(variant, n_embd, n_head, n_layer, max_ctx).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    eval_points = [500, 1000, 2000, 3000, 4000, 5000]
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
            m = compute_metrics(model, max_ctx, batch_size)
            t = time.perf_counter() - start
            ppl = math.exp(val)
            print(f"      step {step:5d} | val {val:.4f} ppl {ppl:.2f} | "
                  f"sink {m['sink']:.4f} pos0 {m['pos0']:.4f} sparse {m['sparse']:.3f} | {t:.0f}s",
                  flush=True)
            history.append({
                'step': step, 'val': val, 'ppl': ppl,
                'sink': m['sink'], 'pos0': m['pos0'], 'sparse': m['sparse'],
            })
            next_eval += 1

    total = time.perf_counter() - start

    ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample = decode(model.generate(ctx, 200)[0].tolist())
    print(f"      Sample: {sample[:150].replace(chr(10), ' ')}", flush=True)

    del model, opt
    torch.cuda.empty_cache()

    return history, total, n_params


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("TANK vs TANK: FAIR BF16 COMPARISON", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    N_EMBD, N_HEAD, N_LAYER, CTX = 192, 6, 6, 256
    BATCH = 32
    STEPS = 5000
    SEEDS = [42, 137, 2024, 7, 256, 1337, 9999, 314, 555, 808]

    print(f"Config: {N_LAYER}L/{N_HEAD}H/{N_EMBD}d, ctx={CTX}, batch={BATCH}, steps={STEPS}")
    print(f"Seeds: {SEEDS}")
    print()

    # The 4 variants: fp32 references + bf16 head-to-head
    variants = ['softmax_fp32', 'softmax_bf16', 'sp2norm_eager', 'sp2norm_fused']

    all_results = {v: {} for v in variants}

    for seed in SEEDS:
        print(f"\n{'#'*80}", flush=True)
        print(f"  SEED = {seed}", flush=True)
        print(f"{'#'*80}", flush=True)

        for variant in variants:
            print(f"\n  {'='*60}", flush=True)
            print(f"    {variant} (seed={seed})", flush=True)
            print(f"  {'='*60}", flush=True)
            hist, total, n_params = train_variant(
                variant, seed, N_EMBD, N_HEAD, N_LAYER, CTX, BATCH, STEPS
            )
            all_results[variant][seed] = (hist, total, n_params)
            print(f"    Total time: {total:.1f}s\n", flush=True)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print(f"\n{'='*80}", flush=True)
    print("FINAL REPORT — TANK vs TANK (10 SEEDS)", flush=True)
    print("=" * 80, flush=True)

    # Per-seed table
    print(f"\n  --- Per-Seed Val Loss @ step {STEPS} ---", flush=True)
    print(f"  {'Variant':>18s}", end="")
    for seed in SEEDS:
        print(f"  seed={seed:>5d}", end="")
    print(f"    {'mean':>8s}  {'std':>8s}")
    print("  " + "-" * 80)

    avg_val = {}
    avg_time = {}
    avg_metrics = {}

    for v in variants:
        vals, times, sinks, sparses = [], [], [], []
        for seed in SEEDS:
            h = all_results[v][seed][0][-1]
            t = all_results[v][seed][1]
            vals.append(h['val'])
            times.append(t)
            sinks.append(h['sink'])
            sparses.append(h['sparse'])

        mean_val = sum(vals) / len(vals)
        std_val = (sum((x - mean_val)**2 for x in vals) / len(vals)) ** 0.5
        avg_val[v] = (mean_val, std_val)
        avg_time[v] = sum(times) / len(times)
        avg_metrics[v] = {'sink': sum(sinks)/len(sinks), 'sparse': sum(sparses)/len(sparses)}

        print(f"  {v:>18s}", end="")
        for val in vals:
            print(f"  {val:>10.4f}", end="")
        print(f"    {mean_val:8.4f}  {std_val:8.4f}")

    # --- THE MAIN EVENT: bf16 head-to-head ---
    print(f"\n  {'='*60}", flush=True)
    print(f"  THE MAIN EVENT: bf16 vs bf16 (fair fight)", flush=True)
    print(f"  {'='*60}", flush=True)

    bf16_variants = ['softmax_bf16', 'sp2norm_fused']
    sm_bf16_mean = avg_val['softmax_bf16'][0]

    print(f"  {'Variant':>18s} {'Val':>8s} {'±std':>8s} {'PPL':>8s} {'Sparse':>8s} {'Time':>8s} {'vs SM_bf16':>12s} {'Speedup':>8s}", flush=True)
    print("  " + "-" * 90)

    for v in bf16_variants:
        mean_v, std_v = avg_val[v]
        ppl = math.exp(mean_v)
        gap = (mean_v - sm_bf16_mean) / sm_bf16_mean * 100
        gap_str = "BASELINE" if v == 'softmax_bf16' else f"{gap:+.2f}%"
        speedup = avg_time['softmax_bf16'] / avg_time[v]
        speed_str = "1.00x" if v == 'softmax_bf16' else f"{speedup:.2f}x"
        m = avg_metrics[v]
        print(f"  {v:>18s} {mean_v:8.4f} {std_v:8.4f} {ppl:8.2f} {m['sparse']:8.3f} {avg_time[v]:6.0f}s  {gap_str:>12s} {speed_str:>8s}", flush=True)

    # --- Full comparison ---
    print(f"\n  --- Full Comparison (all variants) ---", flush=True)
    sm_fp32_mean = avg_val['softmax_fp32'][0]
    print(f"  {'Variant':>18s} {'Val':>8s} {'±std':>8s} {'PPL':>8s} {'Sparse':>8s} {'Time':>8s} {'vs SM_fp32':>12s}", flush=True)
    print("  " + "-" * 82)

    for v in variants:
        mean_v, std_v = avg_val[v]
        ppl = math.exp(mean_v)
        gap = (mean_v - sm_fp32_mean) / sm_fp32_mean * 100
        gap_str = "BASELINE" if v == 'softmax_fp32' else f"{gap:+.2f}%"
        m = avg_metrics[v]
        print(f"  {v:>18s} {mean_v:8.4f} {std_v:8.4f} {ppl:8.2f} {m['sparse']:8.3f} {avg_time[v]:6.0f}s  {gap_str:>12s}", flush=True)

    # --- Loss curves ---
    print(f"\n  --- Averaged Val Loss Curve ---", flush=True)
    eval_points = [500, 1000, 2000, 3000, 4000, 5000]
    print(f"  {'Step':>6s}", end="")
    for v in variants:
        print(f"  {v:>18s}", end="")
    print()
    print("  " + "-" * (8 + 20 * len(variants)))

    for i, step in enumerate(eval_points):
        print(f"  {step:6d}", end="")
        for v in variants:
            vals_at_step = [all_results[v][s][0][i]['val'] for s in SEEDS]
            mean = sum(vals_at_step) / len(vals_at_step)
            print(f"  {mean:18.4f}", end="")
        print()

    # --- Significance ---
    print(f"\n  --- Significance (per-seed wins) ---", flush=True)
    sm_bf16_vals = [all_results['softmax_bf16'][s][0][-1]['val'] for s in SEEDS]
    sm_fp32_vals = [all_results['softmax_fp32'][s][0][-1]['val'] for s in SEEDS]
    for v in ['sp2norm_fused', 'sp2norm_eager']:
        v_vals = [all_results[v][s][0][-1]['val'] for s in SEEDS]
        wins_bf16 = sum(1 for sv, vv in zip(sm_bf16_vals, v_vals) if vv < sv)
        wins_fp32 = sum(1 for sv, vv in zip(sm_fp32_vals, v_vals) if vv < sv)
        print(f"  {v} beats softmax_bf16 in {wins_bf16}/{len(SEEDS)} seeds", flush=True)
        print(f"  {v} beats softmax_fp32 in {wins_fp32}/{len(SEEDS)} seeds", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("DONE", flush=True)
    print("=" * 80, flush=True)
