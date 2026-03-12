"""
END-TO-END TRAINING VALIDATION
================================
Head-to-head: softmax vs sp2norm_eager vs sp2norm_fused (Triton kernel)

Proves:
1. The 1.7% val loss win holds with fused kernel
2. Wall-clock training speedup from kernel fusion
3. No quality degradation from Triton vs eager

Model: 6L/6H/192d GPT on Shakespeare, 2000 steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "src")

from smooth_attn import DEFAULT_FAMILY, softplus_norm_causal_eager, sp2norm_flash_attention

DEVICE = 'cuda'
ATTN_DROPOUT = 0.0
RESID_DROPOUT = 0.1

# ============================================================
# Data
# ============================================================

import os
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
# Attention variants
# ============================================================

def attn_softmax(wei, mask):
    """Standard softmax attention (baseline)."""
    wei = wei.masked_fill(mask, float('-inf'))
    return F.softmax(wei, dim=-1)


def attn_sp2norm_eager(wei, mask):
    """sp2norm: squareplus(2*scores - 1)^2 / sum. Eager (materialized T×T)."""
    del mask
    return softplus_norm_causal_eager(wei.float(), family=DEFAULT_FAMILY)


# ============================================================
# Model — supports both eager (wei,mask) and fused (q,k,v) paths
# ============================================================

class MHA_Eager(nn.Module):
    """Multi-head attention with materialized score matrix."""
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
        # q,k,v: [B, T, H, hs] -> [B, H, T, hs]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (self.hs ** -0.5)
        mask = self.tril[:T, :T] == 0
        wei = self.attn_fn(wei, mask)

        if capture:
            self.last_attn_weights = wei.detach()

        wei = self.attn_drop(wei)
        out = wei @ v  # [B, H, T, hs]
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class MHA_Fused(nn.Module):
    """Multi-head attention using sp2norm_flash_attention Triton kernel.
    No T×T matrix ever materialized. Causal mask built into kernel."""
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
        self.last_attn_weights = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # [B, T, H, hs] -> [B, H, T, hs]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Cast to bf16 for Triton kernel
        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        v_bf = v.to(torch.bfloat16)

        out = sp2norm_flash_attention(q_bf, k_bf, v_bf)  # [B, H, T, hs]
        out = out.to(x.dtype)

        # capture not supported for fused (no materialized weights)
        if capture:
            self.last_attn_weights = None

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


def build_model(variant, n_embd, n_head, n_layer, max_ctx):
    """Build GPT with specified attention variant."""
    mha_kwargs = {"attn_dropout": ATTN_DROPOUT, "proj_dropout": RESID_DROPOUT}

    if variant == 'softmax':
        mha_fn = lambda: MHA_Eager(n_embd, n_head, max_ctx, attn_softmax, **mha_kwargs)
    elif variant == 'sp2norm_eager':
        mha_fn = lambda: MHA_Eager(n_embd, n_head, max_ctx, attn_sp2norm_eager, **mha_kwargs)
    elif variant == 'sp2norm_fused':
        mha_fn = lambda: MHA_Fused(n_embd, n_head, max_ctx, **mha_kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")

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
            # w: [B, H, T, T]
            start = min(16, w.shape[-1] - 1)
            a0 = w[:, :, start:, 0]  # attention to position 0
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
def eval_loss(model, block_size, batch_size, n=30):
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

def train_variant(variant, n_embd, n_head, n_layer, max_ctx, batch_size, steps):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = build_model(variant, n_embd, n_head, n_layer, max_ctx).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"    params={n_params:,}", flush=True)

    # Warmup: 5 steps to trigger Triton compilation before timing
    for _ in range(5):
        X, Y = get_batch('train', max_ctx, batch_size)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Reset seed for fair comparison
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = build_model(variant, n_embd, n_head, n_layer, max_ctx).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    start = time.perf_counter()
    eval_points = [500, 1000, 1500, 2000]
    history = []
    next_eval = 0

    for step in range(1, steps + 1):
        X, Y = get_batch('train', max_ctx, batch_size)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if next_eval < len(eval_points) and step == eval_points[next_eval]:
            val = eval_loss(model, max_ctx, batch_size, n=50)
            # Only compute attention metrics for eager variants (fused has no materialized weights)
            if variant != 'sp2norm_fused':
                m = compute_metrics(model, max_ctx, batch_size)
            else:
                m = {'sink': -1, 'pos0': -1, 'sparse': -1}
            t = time.perf_counter() - start
            ppl = math.exp(val)
            sink_str = f"{m['sink']:.4f}" if m['sink'] >= 0 else "  N/A "
            pos0_str = f"{m['pos0']:.4f}" if m['pos0'] >= 0 else "  N/A "
            sparse_str = f"{m['sparse']:.3f}" if m['sparse'] >= 0 else " N/A "
            print(f"      step {step:5d} | val {val:.4f} ppl {ppl:.2f} | "
                  f"sink {sink_str} pos0 {pos0_str} sparse {sparse_str} | {t:.0f}s",
                  flush=True)
            history.append({
                'step': step, 'val': val, 'ppl': ppl,
                'sink': m['sink'], 'pos0': m['pos0'], 'sparse': m['sparse'],
            })
            next_eval += 1

    total = time.perf_counter() - start

    # Generate sample
    ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample = decode(model.generate(ctx, 200)[0].tolist())
    print(f"      Sample: {sample[:150].replace(chr(10), ' ')}", flush=True)

    del model, opt
    torch.cuda.empty_cache()

    return history, total


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("END-TO-END TRAINING: SOFTMAX vs SP2NORM_EAGER vs SP2NORM_FUSED", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    N_EMBD, N_HEAD, N_LAYER, CTX = 192, 6, 6, 256
    BATCH = 32
    STEPS = 2000

    print(f"Config: {N_LAYER}L/{N_HEAD}H/{N_EMBD}d, ctx={CTX}, batch={BATCH}, steps={STEPS}")
    print()

    variants = ['softmax', 'sp2norm_eager', 'sp2norm_fused']
    results = {}

    for variant in variants:
        print(f"{'='*60}", flush=True)
        print(f"  TRAINING: {variant}", flush=True)
        print(f"{'='*60}", flush=True)
        hist, total = train_variant(variant, N_EMBD, N_HEAD, N_LAYER, CTX, BATCH, STEPS)
        results[variant] = (hist, total)
        print(f"    Total time: {total:.1f}s\n", flush=True)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print(f"\n{'='*80}", flush=True)
    print("FINAL REPORT", flush=True)
    print("=" * 80, flush=True)

    # Loss curve
    print(f"\n  --- Val Loss Curve ---", flush=True)
    print(f"  {'Step':>6s}", end="")
    for v in variants:
        print(f"  {v:>18s}", end="")
    print()
    print("  " + "-" * (8 + 20 * len(variants)))

    for i, step in enumerate([500, 1000, 1500, 2000]):
        print(f"  {step:6d}", end="")
        for v in variants:
            print(f"  {results[v][0][i]['val']:18.4f}", end="")
        print()

    # Final comparison
    print(f"\n  --- Final (step 2000) ---", flush=True)
    print(f"  {'Variant':>18s} {'Val':>8s} {'PPL':>8s} {'Time':>8s} {'vs softmax':>12s} {'Speedup':>10s}", flush=True)
    print("  " + "-" * 70)

    sm_val = results['softmax'][0][-1]['val']
    sm_time = results['softmax'][1]

    for v in variants:
        h = results[v][0][-1]
        t = results[v][1]
        gap = (h['val'] - sm_val) / sm_val * 100
        gap_str = "BASELINE" if v == 'softmax' else f"{gap:+.2f}%"
        speedup = sm_time / t
        speed_str = "1.00x" if v == 'softmax' else f"{speedup:.2f}x"
        print(f"  {v:>18s} {h['val']:8.4f} {h['ppl']:8.2f} {t:6.0f}s  {gap_str:>12s} {speed_str:>10s}", flush=True)

    # Fused vs eager consistency check
    fused_val = results['sp2norm_fused'][0][-1]['val']
    eager_val = results['sp2norm_eager'][0][-1]['val']
    drift = abs(fused_val - eager_val) / eager_val * 100
    print(f"\n  Fused vs Eager drift: {drift:.2f}% (should be <1%)", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("DONE", flush=True)
    print("=" * 80, flush=True)
