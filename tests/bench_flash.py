"""Benchmark fwd+bwd: flash vs materialized attention, including GQA and sliding window."""
import math
import torch
import sys
sys.path.insert(0, "src")

from smooth_attn import sp2norm_flash_attention, softplus_norm_causal


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
    B, D = 2, 32

    # --- Standard MHA fwd+bwd ---
    print("=== FWD ONLY BENCHMARK ===")
    H = 6
    print(f"{'T':>6s} {'window':>7s} {'flash_us':>10s} {'ratio_vs_full':>14s}")
    print("-" * 42)
    for T in [128, 256, 512, 1024]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

        full_us = bench_us(lambda: sp2norm_flash_attention(q, k, v))
        for W in [0, 64, 128]:
            if W >= T:
                continue
            t_us = bench_us(lambda w=W: sp2norm_flash_attention(q, k, v, window_size=w))
            label = "full" if W == 0 else f"W={W}"
            ratio = f"{t_us/full_us:.2f}x" if W > 0 else "-"
            print(f"{T:6d} {label:>7s} {t_us:8.1f}us {ratio:>14s}")

    # --- GQA benchmark ---
    print("\n=== GQA FWD BENCHMARK ===")
    T = 256
    print(f"{'config':>12s} {'flash_us':>10s}")
    print("-" * 24)
    for H_q, H_kv in [(8, 8), (8, 4), (8, 2), (8, 1)]:
        q = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
        t_us = bench_us(lambda: sp2norm_flash_attention(q, k, v))
        label = {8: "MHA", 4: "GQA-2", 2: "GQA-4", 1: "MQA"}[H_kv]
        print(f"{label:>12s} {t_us:8.1f}us")

    # --- Full fwd+bwd with memory ---
    print("\n=== FWD+BWD BENCHMARK (training) ===")
    H = 6
    scale = 1.0 / math.sqrt(D)
    print(f"{'T':>6s} {'flash':>12s} {'materialized':>14s} {'ratio':>8s} {'mem_flash':>10s} {'mem_mat':>10s}")
    print("-" * 66)
    for T in [64, 128, 256, 512]:
        def run_flash(t=T):
            q = torch.randn(B, H, t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(B, H, t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            out = sp2norm_flash_attention(q, k, v)
            out.float().sum().backward()

        def run_mat(t=T):
            q = torch.randn(B, H, t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(B, H, t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(B, H, t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = softplus_norm_causal(scores, implementation="triton")
            out = torch.matmul(attn, v)
            out.float().sum().backward()

        torch.cuda.reset_peak_memory_stats()
        run_flash()
        mem_flash = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        run_mat()
        mem_mat = torch.cuda.max_memory_allocated() / 1024**2

        flash_us = bench_us(run_flash)
        mat_us = bench_us(run_mat)
        print(
            f"{T:6d} {flash_us:10.1f}us {mat_us:12.1f}us {flash_us/mat_us:7.2f}x"
            f" {mem_flash:8.1f}MB {mem_mat:8.1f}MB"
        )


if __name__ == "__main__":
    main()
