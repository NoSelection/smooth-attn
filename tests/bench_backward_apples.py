"""Apples-to-apples: flash vs materialized training step, identical loss."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import sp2norm_flash_attention, softplus_norm_causal


def bench_us(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return times[len(times) // 2]


def main():
    B, H, D = 2, 6, 32
    scale = 1.0 / math.sqrt(D)

    print("=== TRAINING STEP: FLASH vs MATERIALIZED (fwd+bwd, same loss) ===")
    print(f"{'T':>6s} {'flash_us':>10s} {'mat_us':>10s} {'speedup':>8s} {'mem_flash':>10s} {'mem_mat':>10s}")
    print("-" * 60)

    for T in [64, 128, 256, 512, 1024]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run_flash(q=q, k=k, v=v):
            q.grad = k.grad = v.grad = None
            out = sp2norm_flash_attention(q, k, v)
            out.sum().backward()

        def run_mat(q=q, k=k, v=v):
            q.grad = k.grad = v.grad = None
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = softplus_norm_causal(scores, implementation="triton")
            out = torch.matmul(attn, v)
            out.sum().backward()

        # Memory measurement
        torch.cuda.reset_peak_memory_stats()
        run_flash()
        mem_flash = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        run_mat()
        mem_mat = torch.cuda.max_memory_allocated() / 1024**2

        flash_us = bench_us(run_flash)
        mat_us = bench_us(run_mat)
        speedup = mat_us / flash_us
        print(f"{T:6d} {flash_us:8.1f}us {mat_us:8.1f}us {speedup:7.2f}x {mem_flash:8.1f}MB {mem_mat:8.1f}MB")

    # Larger sizes where flash really shines
    print(f"\n=== LARGER SIZES (B=4, H=8, D=64) ===")
    B2, H2, D2 = 4, 8, 64
    scale2 = 1.0 / math.sqrt(D2)
    print(f"{'T':>6s} {'flash_us':>10s} {'mat_us':>10s} {'speedup':>8s} {'mem_flash':>10s} {'mem_mat':>10s}")
    print("-" * 60)

    for T in [128, 256, 512, 1024]:
        q = torch.randn(B2, H2, T, D2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B2, H2, T, D2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B2, H2, T, D2, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run_flash(q=q, k=k, v=v):
            q.grad = k.grad = v.grad = None
            out = sp2norm_flash_attention(q, k, v)
            out.sum().backward()

        def run_mat(q=q, k=k, v=v):
            q.grad = k.grad = v.grad = None
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale2
            attn = softplus_norm_causal(scores, implementation="triton")
            out = torch.matmul(attn, v)
            out.sum().backward()

        torch.cuda.reset_peak_memory_stats()
        run_flash()
        mem_flash = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        run_mat()
        mem_mat = torch.cuda.max_memory_allocated() / 1024**2

        flash_us = bench_us(run_flash)
        mat_us = bench_us(run_mat)
        speedup = mat_us / flash_us
        print(f"{T:6d} {flash_us:8.1f}us {mat_us:8.1f}us {speedup:7.2f}x {mem_flash:8.1f}MB {mem_mat:8.1f}MB")


if __name__ == "__main__":
    main()
