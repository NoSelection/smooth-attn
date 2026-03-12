"""Detailed backward profiling - per-kernel timing."""
import math
import sys
sys.path.insert(0, "src")

import torch
import triton
from smooth_attn import sp2norm_flash_attention
from smooth_attn.kernels import (
    _sp2norm_flash_fwd,
    _sp2norm_flash_bwd_precompute_delta,
    _sp2norm_flash_bwd_dq_kernel,
    _sp2norm_flash_bwd_dkv_kernel,
)


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
    scale = 1.0 / math.sqrt(D)

    for T in [64, 128, 256, 512, 1024]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()

        # Run forward to get saved tensors
        o, row_sums = _sp2norm_flash_fwd(q, k, v, scale, 1, 0)
        do = torch.randn_like(o)

        D_HEAD = triton.next_power_of_2(D)
        BLOCK_DELTA = min(64, triton.next_power_of_2(T))

        # Delta precompute
        delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        def run_delta():
            grid_delta = (triton.cdiv(T, BLOCK_DELTA), B * H)
            _sp2norm_flash_bwd_precompute_delta[grid_delta](
                o, do, delta, T,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                delta.stride(0), delta.stride(1), delta.stride(2),
                BLOCK_Q=BLOCK_DELTA, D_HEAD=D_HEAD,
            )
        delta_us = bench_us(run_delta)

        # dQ kernel
        dq = torch.empty_like(q)
        def run_dq():
            grid_q = lambda meta: (triton.cdiv(T, meta["BLOCK_Q"]), B * H)
            _sp2norm_flash_bwd_dq_kernel[grid_q](
                q, k, v, do, dq, row_sums, delta,
                T, scale, 1, 0,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                D_HEAD=D_HEAD,
            )
        dq_us = bench_us(run_dq)

        # dKV kernel
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        def run_dkv():
            grid_kv = lambda meta: (triton.cdiv(T, meta["BLOCK_KV"]), B * H)
            _sp2norm_flash_bwd_dkv_kernel[grid_kv](
                q, k, v, do, dk, dv, row_sums, delta,
                T, scale, 1, 0, H,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                D_HEAD=D_HEAD,
            )
        dkv_us = bench_us(run_dkv)

        total = delta_us + dq_us + dkv_us
        print(f"T={T:4d}  delta={delta_us:6.1f}us  dQ={dq_us:6.1f}us  dKV={dkv_us:6.1f}us  total={total:6.1f}us  dQ_grid=({triton.cdiv(T,64)},{B*H})  dKV_grid=({triton.cdiv(T,64)},{B*H})")


if __name__ == "__main__":
    main()
