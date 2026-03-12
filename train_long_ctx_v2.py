"""
LONG-CONTEXT v2: Give sp2norm the runway it needs.
====================================================
At ctx=256 (5000 steps), sp2norm crosses over around step 1500-2000.
At ctx=512 (3000 steps), gap was still closing at +0.78%.
At ctx=1024 (3000 steps), gap closing fast at ~2%/500 steps.

sp2norm has smoother gradients (squareplus vs exp) → slower warmup, better asymptote.
Longer context = more tokens = needs more steps to sharpen attention.

Fix: more steps. ctx=512 → 6000 steps, ctx=1024 → 8000 steps.
3 seeds, bf16 only (the fair fight).
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
SP2_EXACT_MATH = True
BASE_LR = 3e-4
MIN_LR_RATIO = 0.1
WARMUP_RATIO = 0.05
GRAD_CLIP_NORM = 1.0

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
# Attention
# ============================================================

def attn_sp2norm_eager(wei, mask):
    del mask
    return softplus_norm_causal_eager(wei.float(), family=DEFAULT_FAMILY)


# ============================================================
# MHA modules (bf16 only — the fair fight)
# ============================================================

class MHA_SoftmaxBF16(nn.Module):
    def __init__(self, n_embd, n_head, max_ctx):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(RESID_DROPOUT)
        self.register_buffer('tril', torch.tril(torch.ones(max_ctx, max_ctx)))
        self.last_attn_weights = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            out = out.transpose(1, 2).reshape(B, T, C)
            out = self.proj(out)
        if capture:
            with torch.no_grad():
                wei = (q.float() @ k.float().transpose(-2, -1)) * (self.hs ** -0.5)
                mask = self.tril[:T, :T] == 0
                wei = wei.masked_fill(mask, float('-inf'))
                self.last_attn_weights = F.softmax(wei, dim=-1).detach()
        return self.proj_drop(out)


class MHA_SP2NormFused(nn.Module):
    def __init__(self, n_embd, n_head, max_ctx):
        super().__init__()
        self.n_head = n_head
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(RESID_DROPOUT)
        self.register_buffer('tril', torch.tril(torch.ones(max_ctx, max_ctx)))
        self.last_attn_weights = None

    def forward(self, x, capture=False):
        B, T, C = x.shape
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.hs)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = sp2norm_flash_attention(q, k, v, exact_math=SP2_EXACT_MATH)
            out = out.transpose(1, 2).reshape(B, T, C)
            out = self.proj(out)
        if capture:
            with torch.no_grad():
                wei = (q.float() @ k.float().transpose(-2, -1)) * (self.hs ** -0.5)
                mask = self.tril[:T, :T] == 0
                self.last_attn_weights = attn_sp2norm_eager(wei, mask).detach()
        return self.proj_drop(out)


# ============================================================
# Model
# ============================================================

def build_model(variant, n_embd, n_head, n_layer, max_ctx):
    mha_map = {
        'softmax_bf16': lambda: MHA_SoftmaxBF16(n_embd, n_head, max_ctx),
        'sp2norm_fused': lambda: MHA_SP2NormFused(n_embd, n_head, max_ctx),
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

    return GPT()


# ============================================================
# Eval
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


# ============================================================
# Training
# ============================================================

def make_adamw(params, lr):
    kwargs = {"lr": lr}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)


def lr_at_step(step, total_steps, base_lr):
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    min_lr = base_lr * MIN_LR_RATIO
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


def train_variant(variant, seed, n_embd, n_head, n_layer, max_ctx, batch_size, steps, eval_every):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = build_model(variant, n_embd, n_head, n_layer, max_ctx).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_adamw(model.parameters(), lr=BASE_LR)

    # Warmup (Triton compilation)
    for _ in range(3):
        X, Y = get_batch('train', max_ctx, batch_size)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = build_model(variant, n_embd, n_head, n_layer, max_ctx).to(DEVICE)
    opt = make_adamw(model.parameters(), lr=BASE_LR)

    eval_points = list(range(eval_every, steps + 1, eval_every))
    if steps not in eval_points:
        eval_points.append(steps)
    history = []
    next_eval = 0

    start = time.perf_counter()

    for step in range(1, steps + 1):
        lr = lr_at_step(step, steps, BASE_LR)
        set_lr(opt, lr)
        X, Y = get_batch('train', max_ctx, batch_size)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()

        if next_eval < len(eval_points) and step == eval_points[next_eval]:
            val = eval_loss(model, max_ctx, batch_size)
            t = time.perf_counter() - start
            ppl = math.exp(val)
            print(f"      step {step:5d} | val {val:.4f} ppl {ppl:.2f} | lr {lr:.2e} | {t:.0f}s",
                  flush=True)
            history.append({'step': step, 'val': val, 'ppl': ppl, 'lr': lr})
            next_eval += 1

    total = time.perf_counter() - start
    del model, opt
    torch.cuda.empty_cache()
    return history, total, n_params


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("LONG-CONTEXT v2: GIVING SP2NORM THE RUNWAY", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    SEEDS = [42, 137, 2024]
    N_EMBD, N_HEAD, N_LAYER = 192, 6, 6

    # More steps for longer contexts — sp2norm needs runway to converge
    ctx_configs = [
        # (ctx, batch, steps, eval_every)
        (512,  16, 6000, 500),    # 2x more steps than v1
        (1024,  8, 8000, 500),    # 2.7x more steps than v1
    ]

    variants = ['softmax_bf16', 'sp2norm_fused']

    for CTX, BATCH, STEPS, EVAL_EVERY in ctx_configs:
        print(f"\n{'#'*80}", flush=True)
        print(f"  CTX = {CTX}, BATCH = {BATCH}, STEPS = {STEPS}", flush=True)
        print(f"  Config: {N_LAYER}L/{N_HEAD}H/{N_EMBD}d, seeds={SEEDS}", flush=True)
        print(f"{'#'*80}", flush=True)

        all_results = {v: {} for v in variants}

        for seed in SEEDS:
            for variant in variants:
                print(f"\n  {variant} (seed={seed}, ctx={CTX})", flush=True)
                hist, total, n_params = train_variant(
                    variant, seed, N_EMBD, N_HEAD, N_LAYER, CTX, BATCH, STEPS, EVAL_EVERY
                )
                all_results[variant][seed] = (hist, total, n_params)
                print(f"    Total: {total:.1f}s", flush=True)

        # Report for this context length
        print(f"\n  {'='*60}", flush=True)
        print(f"  RESULTS: CTX={CTX}, STEPS={STEPS}", flush=True)
        print(f"  {'='*60}", flush=True)

        # Print full learning curve comparison
        print(f"\n  --- Learning Curve (mean over seeds) ---")
        sm_hists = [all_results['softmax_bf16'][s][0] for s in SEEDS]
        sp_hists = [all_results['sp2norm_fused'][s][0] for s in SEEDS]
        n_evals = min(len(h) for h in sm_hists + sp_hists)
        print(f"  {'Step':>8s}  {'softmax_bf16':>14s}  {'sp2norm_fused':>14s}  {'gap':>8s}")
        print(f"  {'-'*50}")
        for i in range(n_evals):
            step = sm_hists[0][i]['step']
            sm_mean = sum(h[i]['val'] for h in sm_hists) / len(sm_hists)
            sp_mean = sum(h[i]['val'] for h in sp_hists) / len(sp_hists)
            gap = (sp_mean - sm_mean) / sm_mean * 100
            marker = " <-- CROSSOVER" if i > 0 and gap <= 0 else ""
            print(f"  {step:8d}  {sm_mean:14.4f}  {sp_mean:14.4f}  {gap:+7.2f}%{marker}")

        print(f"\n  --- Final Results ---")
        print(f"\n  {'Variant':>18s}", end="")
        for seed in SEEDS:
            print(f"  seed={seed:>5d}", end="")
        print(f"    {'mean':>8s}  {'time':>6s}  {'speedup':>8s}")
        print("  " + "-" * 80)

        avg_time = {}
        avg_val = {}
        for v in variants:
            vals = [all_results[v][s][0][-1]['val'] for s in SEEDS]
            times = [all_results[v][s][1] for s in SEEDS]
            mean_val = sum(vals) / len(vals)
            mean_time = sum(times) / len(times)
            avg_val[v] = mean_val
            avg_time[v] = mean_time

            print(f"  {v:>18s}", end="")
            for val in vals:
                print(f"  {val:>10.4f}", end="")
            speedup = avg_time['softmax_bf16'] / mean_time if v != 'softmax_bf16' and 'softmax_bf16' in avg_time else 1.0
            print(f"    {mean_val:8.4f}  {mean_time:5.0f}s  {speedup:7.2f}x")

        # Per-seed wins
        sm_vals = [all_results['softmax_bf16'][s][0][-1]['val'] for s in SEEDS]
        sp_vals = [all_results['sp2norm_fused'][s][0][-1]['val'] for s in SEEDS]
        wins = sum(1 for sv, vv in zip(sm_vals, sp_vals) if vv < sv)
        print(f"\n  sp2norm_fused beats softmax_bf16: {wins}/{len(SEEDS)} seeds")

        gap = (avg_val['sp2norm_fused'] - avg_val['softmax_bf16']) / avg_val['softmax_bf16'] * 100
        print(f"  Quality gap: {gap:+.2f}%")
        speedup = avg_time['softmax_bf16'] / avg_time['sp2norm_fused']
        print(f"  Speed: {speedup:.2f}x")

    print(f"\n{'='*80}", flush=True)
    print("DONE", flush=True)
    print("=" * 80, flush=True)
