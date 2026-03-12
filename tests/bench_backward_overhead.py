"""Isolate where backward time actually goes."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import sp2norm_flash_attention
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
    B, H, D = 2, 6, 32
    scale = 1.0 / math.sqrt(D)

    for T in [64, 128, 256, 512, 1024]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        upstream = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

        # 1) Full autograd backward (what user measures)
        def run_full():
            q.grad = k.grad = v.grad = None
            out = sp2norm_flash_attention(q, k, v)
            (out.float() * upstream).sum().backward()
        full_us = bench_us(run_full)

        # 2) Direct _sp2norm_flash_bwd call (bypass autograd)
        q_c = q.detach().contiguous()
        k_c = k.detach().contiguous()
        v_c = v.detach().contiguous()
        o, row_sums = _sp2norm_flash_fwd(q_c, k_c, v_c, scale, 1, 0)
        do = torch.randn_like(o)

        def run_bwd_only():
            return _sp2norm_flash_bwd(q_c, k_c, v_c, o, do, row_sums, scale, 1, 0)
        bwd_us = bench_us(run_bwd_only)

        # 3) The float()+mul+sum part
        out_det = o.detach().clone().requires_grad_(True)
        def run_loss():
            return (out_det.float() * upstream).sum()
        loss_us = bench_us(run_loss)

        # 4) Forward only (no grad)
        def run_fwd():
            with torch.no_grad():
                return sp2norm_flash_attention(q, k, v)
        fwd_us = bench_us(run_fwd)

        # 5) Forward with grad tracking
        def run_fwd_grad():
            return sp2norm_flash_attention(q, k, v)
        fwd_grad_us = bench_us(run_fwd_grad)

        print(f"T={T:4d}  full={full_us:6.1f}  fwd_nograd={fwd_us:5.1f}  fwd_grad={fwd_grad_us:5.1f}  "
              f"bwd_direct={bwd_us:6.1f}  loss_compute={loss_us:5.1f}  "
              f"overhead={full_us - fwd_us - bwd_us - loss_us:6.1f}")


if __name__ == "__main__":
    main()
