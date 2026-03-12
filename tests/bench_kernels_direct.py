"""Direct kernel timing — bypasses autograd entirely."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn.kernels import _sp2norm_flash_fwd, _sp2norm_flash_bwd


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
    print("=== DIRECT KERNEL TIMING (no autograd) ===")
    print(f"{'config':>25s} {'fwd_us':>8s} {'bwd_us':>8s} {'total':>8s} {'ratio':>6s}")
    print("-" * 60)

    for B, H, T, D in [(2,6,64,32), (2,6,128,32), (2,6,256,32), (2,6,512,32), (2,6,1024,32),
                         (4,8,256,64), (4,8,512,64), (4,8,1024,64), (4,8,2048,64)]:
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16).contiguous()

        o, row_sums = _sp2norm_flash_fwd(q, k, v, scale, 1, 0)
        do = torch.randn_like(o)

        def run_fwd(q=q, k=k, v=v, s=scale):
            return _sp2norm_flash_fwd(q, k, v, s, 1, 0)

        def run_bwd(q=q, k=k, v=v, o=o, do=do, rs=row_sums, s=scale):
            return _sp2norm_flash_bwd(q, k, v, o, do, rs, s, 1, 0)

        fwd_us = bench_us(run_fwd)
        bwd_us = bench_us(run_bwd)
        label = f"B={B} H={H} T={T} D={D}"
        print(f"{label:>25s} {fwd_us:6.1f}us {bwd_us:6.1f}us {fwd_us+bwd_us:6.1f}us {bwd_us/fwd_us:5.1f}x")


if __name__ == "__main__":
    main()
