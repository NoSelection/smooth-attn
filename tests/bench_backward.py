"""Benchmark backward tuning for sp2norm flash attention."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import sp2norm_flash_attention, softplus_norm_causal_eager


def bench_us(fn, warmup=20, iters=100):
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

    print("=== FLASH ATTENTION FWD vs BWD LATENCY ===")
    print(f"{'T':>6s} {'fwd_us':>10s} {'bwd_us':>10s} {'bwd/fwd':>10s} {'total_us':>10s}")
    print("-" * 50)

    for T in [64, 128, 256, 512, 1024]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        upstream = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

        def run_fwd():
            return sp2norm_flash_attention(q, k, v)

        def run_bwd():
            q.grad = k.grad = v.grad = None
            out = sp2norm_flash_attention(q, k, v)
            (out.float() * upstream).sum().backward()

        def run_fwd_only():
            with torch.no_grad():
                return sp2norm_flash_attention(q, k, v)

        fwd_us = bench_us(run_fwd_only)
        total_us = bench_us(run_bwd)
        bwd_us = total_us - fwd_us
        ratio = bwd_us / fwd_us if fwd_us > 0 else float('inf')
        print(f"{T:6d} {fwd_us:8.1f}us {bwd_us:8.1f}us {ratio:8.1f}x {total_us:8.1f}us")

    # Also benchmark materialized reference for comparison
    print("\n=== VS MATERIALIZED BWD ===")
    scale = 1.0 / math.sqrt(D)
    print(f"{'T':>6s} {'flash_bwd':>12s} {'mat_bwd':>12s} {'speedup':>10s}")
    print("-" * 40)
    for T in [64, 128, 256, 512]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        upstream = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

        def run_flash(q=q, k=k, v=v, up=upstream):
            q.grad = k.grad = v.grad = None
            out = sp2norm_flash_attention(q, k, v)
            (out.float() * up).sum().backward()

        def run_mat(q=q, k=k, v=v, up=upstream):
            q.grad = k.grad = v.grad = None
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = softplus_norm_causal_eager(scores.float())
            out = torch.matmul(attn.to(v.dtype), v)
            (out.float() * up).sum().backward()

        flash_us = bench_us(run_flash)
        mat_us = bench_us(run_mat)
        speedup = mat_us / flash_us
        print(f"{T:6d} {flash_us:10.1f}us {mat_us:10.1f}us {speedup:8.2f}x")


if __name__ == "__main__":
    main()
