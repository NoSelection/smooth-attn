"""Benchmark FP8 kernel ALONE (no quantization overhead) vs BF16 kernel."""
import math
import sys
sys.path.insert(0, "src")

import torch
import triton
from smooth_attn.kernels import (
    _sp2norm_flash_fwd,
    _sp2norm_fp8_flash_fwd,
    _quantize_to_fp8,
)


def bench_us(fn, warmup=30, iters=200):
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
    print("=== FP8 vs BF16 KERNEL-ONLY BENCHMARK (pre-quantized, no Python overhead) ===")
    print(f"{'config':>25s} {'bf16_us':>8s} {'fp8_us':>8s} {'ratio':>8s} {'quant_us':>10s}")
    print("-" * 65)

    for B, H, T, D in [(2,6,64,32), (2,6,128,32), (2,6,256,32), (2,6,512,32),
                         (2,6,1024,32), (4,8,256,64), (4,8,512,64), (4,8,1024,64)]:
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()

        # Pre-quantize (done once, amortized over many forward calls)
        q_fp8, q_s = _quantize_to_fp8(q)
        k_fp8, k_s = _quantize_to_fp8(k)
        v_fp8, v_s = _quantize_to_fp8(v)
        q_fp8 = q_fp8.contiguous()
        k_fp8 = k_fp8.contiguous()
        v_fp8 = v_fp8.contiguous()

        def run_bf16(q=q, k=k, v=v, s=scale):
            return _sp2norm_flash_fwd(q, k, v, s, 1, 0)

        def run_fp8(q=q_fp8, k=k_fp8, v=v_fp8, qs=q_s, ks=k_s, vs=v_s, s=scale):
            return _sp2norm_fp8_flash_fwd(q, k, v, qs, ks, vs, s, 1, 0)

        def run_quant(q=q, k=k, v=v):
            _quantize_to_fp8(q)
            _quantize_to_fp8(k)
            _quantize_to_fp8(v)

        bf16_us = bench_us(run_bf16)
        fp8_us = bench_us(run_fp8)
        quant_us = bench_us(run_quant)
        ratio = fp8_us / bf16_us
        label = f"B={B} H={H} T={T} D={D}"
        print(f"{label:>25s} {bf16_us:6.1f}us {fp8_us:6.1f}us {ratio:6.2f}x {quant_us:8.1f}us")

    # Memory comparison
    print(f"\n=== MEMORY: FP8 Q/K/V vs BF16 Q/K/V ===")
    for B, H, T, D in [(4, 8, 1024, 64), (4, 8, 2048, 128), (8, 32, 2048, 128)]:
        bf16_bytes = B * H * T * D * 2 * 3  # 3 tensors, 2 bytes each
        fp8_bytes = B * H * T * D * 1 * 3   # 3 tensors, 1 byte each
        print(f"  B={B} H={H} T={T} D={D}: BF16={bf16_bytes/1024**2:.1f}MB, FP8={fp8_bytes/1024**2:.1f}MB, savings={bf16_bytes/fp8_bytes:.1f}x")


if __name__ == "__main__":
    main()
