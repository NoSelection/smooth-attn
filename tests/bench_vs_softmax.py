"""Head-to-head: sp2norm flash attention vs PyTorch native softmax attention (cuDNN FA)."""
import math
import sys
sys.path.insert(0, "src")

import torch
import torch.nn.functional as F
from smooth_attn import sp2norm_flash_attention, sp2norm_fp8_flash_attention
from smooth_attn.kernels import _sp2norm_flash_fwd, _sp2norm_fp8_flash_fwd, _quantize_to_fp8


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
    print("=" * 80)
    print("SP2NORM vs SOFTMAX — HEAD-TO-HEAD BENCHMARK")
    print("=" * 80)

    # =========================================================================
    # FORWARD ONLY (inference)
    # =========================================================================
    print("\n=== FORWARD ONLY (no grad, inference mode) ===")
    print(f"{'config':>30s} {'softmax':>10s} {'sp2norm':>10s} {'fp8':>10s} {'sp2/sm':>8s} {'fp8/sm':>8s}")
    print("-" * 80)

    configs = [
        (1, 6, 128, 64, "B=1 H=6 T=128 D=64"),
        (1, 6, 256, 64, "B=1 H=6 T=256 D=64"),
        (1, 6, 512, 64, "B=1 H=6 T=512 D=64"),
        (1, 6, 1024, 64, "B=1 H=6 T=1024 D=64"),
        (2, 8, 256, 64, "B=2 H=8 T=256 D=64"),
        (2, 8, 512, 64, "B=2 H=8 T=512 D=64"),
        (2, 8, 1024, 64, "B=2 H=8 T=1024 D=64"),
        (4, 12, 256, 64, "B=4 H=12 T=256 D=64"),
        (4, 12, 512, 64, "B=4 H=12 T=512 D=64"),
        (4, 12, 1024, 64, "B=4 H=12 T=1024 D=64"),
        (2, 32, 512, 128, "B=2 H=32 T=512 D=128"),
        (2, 32, 1024, 128, "B=2 H=32 T=1024 D=128"),
        (2, 32, 2048, 128, "B=2 H=32 T=2048 D=128"),
    ]

    for B, H, T, D, label in configs:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        scale = 1.0 / math.sqrt(D)

        # Pre-quantize for FP8
        q_fp8, q_s = _quantize_to_fp8(q)
        k_fp8, k_s = _quantize_to_fp8(k)
        v_fp8, v_s = _quantize_to_fp8(v)
        q_fp8, k_fp8, v_fp8 = q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous()

        # PyTorch native SDPA (cuDNN FlashAttention)
        def run_softmax(q=q, k=k, v=v):
            with torch.no_grad():
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Our sp2norm BF16
        def run_sp2norm(q=q, k=k, v=v, s=scale):
            with torch.no_grad():
                return _sp2norm_flash_fwd(q, k, v, s, 1, 0)

        # Our sp2norm FP8 (pre-quantized)
        def run_fp8(q=q_fp8, k=k_fp8, v=v_fp8, qs=q_s, ks=k_s, vs=v_s, s=scale):
            return _sp2norm_fp8_flash_fwd(q, k, v, qs, ks, vs, s, 1, 0)

        sm_us = bench_us(run_softmax)
        sp_us = bench_us(run_sp2norm)
        fp8_us = bench_us(run_fp8)

        sp_ratio = sp_us / sm_us
        fp8_ratio = fp8_us / sm_us
        print(f"{label:>30s} {sm_us:8.1f}us {sp_us:8.1f}us {fp8_us:8.1f}us {sp_ratio:7.2f}x {fp8_ratio:7.2f}x")

    # =========================================================================
    # FORWARD + BACKWARD (training)
    # =========================================================================
    print(f"\n=== FORWARD + BACKWARD (training step) ===")
    print(f"{'config':>30s} {'softmax':>10s} {'sp2norm':>10s} {'sp2/sm':>8s} {'mem_sm':>10s} {'mem_sp2':>10s}")
    print("-" * 80)

    train_configs = [
        (2, 8, 256, 64, "B=2 H=8 T=256 D=64"),
        (2, 8, 512, 64, "B=2 H=8 T=512 D=64"),
        (2, 8, 1024, 64, "B=2 H=8 T=1024 D=64"),
        (4, 12, 256, 64, "B=4 H=12 T=256 D=64"),
        (4, 12, 512, 64, "B=4 H=12 T=512 D=64"),
        (4, 12, 1024, 64, "B=4 H=12 T=1024 D=64"),
        (2, 32, 512, 128, "B=2 H=32 T=512 D=128"),
        (2, 32, 1024, 128, "B=2 H=32 T=1024 D=128"),
    ]

    for B, H, T, D, label in train_configs:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def run_softmax_train(q=q, k=k, v=v):
            q.grad = k.grad = v.grad = None
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out.sum().backward()

        def run_sp2norm_train(q=q, k=k, v=v):
            q.grad = k.grad = v.grad = None
            out = sp2norm_flash_attention(q, k, v)
            out.sum().backward()

        torch.cuda.reset_peak_memory_stats()
        run_softmax_train()
        mem_sm = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        run_sp2norm_train()
        mem_sp2 = torch.cuda.max_memory_allocated() / 1024**2

        sm_us = bench_us(run_softmax_train)
        sp_us = bench_us(run_sp2norm_train)
        ratio = sp_us / sm_us

        print(f"{label:>30s} {sm_us:8.1f}us {sp_us:8.1f}us {ratio:7.2f}x {mem_sm:8.1f}MB {mem_sp2:8.1f}MB")


if __name__ == "__main__":
    main()
