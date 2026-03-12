"""Test FP8 flash attention correctness and benchmark vs BF16."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import sp2norm_fp8_flash_attention, sp2norm_flash_attention, sp2norm_flash_attention_eager


def test_fp8_matches_eager():
    """FP8 flash attention should match eager reference within quantization tolerance."""
    B, H, T, D = 2, 4, 128, 32
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert out_fp8.shape == out_eager.shape
    assert out_fp8.dtype == torch.bfloat16
    assert not out_fp8.isnan().any(), "NaN in FP8 output"

    # FP8 has lower precision — tolerance is wider than BF16
    err = (out_fp8.float() - out_eager.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"  FP8 vs eager: max_err={max_err:.4f}, mean_err={mean_err:.6f}")
    assert max_err < 0.3, f"FP8 max error too high: {max_err}"
    assert mean_err < 0.02, f"FP8 mean error too high: {mean_err}"
    print("  PASSED")


def test_fp8_matches_bf16_flash():
    """FP8 should be reasonably close to BF16 flash."""
    B, H, T, D = 2, 4, 256, 32
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v)
    out_bf16 = sp2norm_flash_attention(q, k, v)

    err = (out_fp8.float() - out_bf16.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"  FP8 vs BF16 flash: max_err={max_err:.4f}, mean_err={mean_err:.6f}")
    # Comparing two approximate paths — max error is wider but mean stays tight
    assert max_err < 0.5, f"FP8 vs BF16 max error too high: {max_err}"
    assert mean_err < 0.02, f"FP8 vs BF16 mean error too high: {mean_err}"
    print("  PASSED")


def test_fp8_various_sizes():
    """Test FP8 at various sequence lengths."""
    B, H, D = 2, 4, 32
    for T in [64, 128, 256, 512]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

        out_fp8 = sp2norm_fp8_flash_attention(q, k, v)
        out_eager = sp2norm_flash_attention_eager(q, k, v)

        assert not out_fp8.isnan().any(), f"NaN at T={T}"
        max_err = (out_fp8.float() - out_eager.float()).abs().max().item()
        print(f"  T={T}: max_err={max_err:.4f}")
        assert max_err < 0.3, f"FP8 error too high at T={T}: {max_err}"
    print("  PASSED")


def test_fp8_gqa():
    """FP8 with GQA (H_q=8, H_kv=2)."""
    B, H_q, H_kv, T, D = 2, 8, 2, 128, 32
    q = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert out_fp8.shape == (B, H_q, T, D)
    max_err = (out_fp8.float() - out_eager.float()).abs().max().item()
    print(f"  GQA FP8: max_err={max_err:.4f}")
    assert max_err < 0.3
    print("  PASSED")


def test_fp8_window():
    """FP8 with sliding window."""
    B, H, T, D, W = 2, 4, 128, 32, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v, window_size=W)
    out_eager = sp2norm_flash_attention_eager(q, k, v, window_size=W)

    max_err = (out_fp8.float() - out_eager.float()).abs().max().item()
    print(f"  Window FP8: max_err={max_err:.4f}")
    assert max_err < 0.3
    print("  PASSED")


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


def benchmark_fp8_vs_bf16():
    """Benchmark FP8 vs BF16 flash forward (no grad)."""
    print("\n=== FP8 vs BF16 FLASH FWD BENCHMARK ===")
    print(f"{'config':>25s} {'bf16_us':>8s} {'fp8_us':>8s} {'speedup':>8s} {'mem_bf16':>10s} {'mem_fp8':>10s}")
    print("-" * 70)

    for B, H, T, D in [(2,6,128,32), (2,6,256,32), (2,6,512,32), (2,6,1024,32),
                         (4,8,256,64), (4,8,512,64), (4,8,1024,64)]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

        def run_bf16(q=q, k=k, v=v):
            with torch.no_grad():
                return sp2norm_flash_attention(q, k, v)

        def run_fp8(q=q, k=k, v=v):
            with torch.no_grad():
                return sp2norm_fp8_flash_attention(q, k, v)

        # Memory
        torch.cuda.reset_peak_memory_stats()
        run_bf16()
        mem_bf16 = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        run_fp8()
        mem_fp8 = torch.cuda.max_memory_allocated() / 1024**2

        bf16_us = bench_us(run_bf16)
        fp8_us = bench_us(run_fp8)
        speedup = bf16_us / fp8_us
        label = f"B={B} H={H} T={T} D={D}"
        print(f"{label:>25s} {bf16_us:6.1f}us {fp8_us:6.1f}us {speedup:7.2f}x {mem_bf16:8.1f}MB {mem_fp8:8.1f}MB")


def main():
    print("=== FP8 FLASH ATTENTION TESTS ===\n")

    print("1. FP8 vs eager reference:")
    test_fp8_matches_eager()

    print("\n2. FP8 vs BF16 flash:")
    test_fp8_matches_bf16_flash()

    print("\n3. Various sizes:")
    test_fp8_various_sizes()

    print("\n4. GQA:")
    test_fp8_gqa()

    print("\n5. Sliding window:")
    test_fp8_window()

    benchmark_fp8_vs_bf16()


if __name__ == "__main__":
    main()
