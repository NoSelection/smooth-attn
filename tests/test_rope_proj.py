"""Test fused RoPE + flash attention and fused output projection."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import (
    sp2norm_flash_attention_eager,
    sp2norm_rope_flash_attention,
    sp2norm_rope_flash_attention_proj,
    precompute_rope_cos_sin,
)


def apply_rope_eager(x, cos, sin):
    """Reference RoPE: rotate_half method."""
    T, D = x.shape[-2], x.shape[-1]
    cos = cos[:T].to(x.dtype)  # [T, D/2]
    sin = sin[:T].to(x.dtype)

    # Expand cos/sin for broadcasting: [1, 1, T, D/2] -> duplicate to [1, 1, T, D]
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)

    # rotate_half: [-x_second, x_first]
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    x_rot = torch.cat([-x2, x1], dim=-1)

    return x * cos_full + x_rot * sin_full


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


def test_rope_flash_correctness():
    """Fused RoPE + flash should match separate RoPE + eager attention."""
    print("=== TEST: Fused RoPE + Flash vs Separate RoPE + Eager ===")
    for B, H, T, D in [(1, 4, 64, 32), (2, 4, 128, 32), (2, 8, 256, 64)]:
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = precompute_rope_cos_sin(D, T, device="cuda", dtype=torch.float32)

        # Fused path
        out_fused = sp2norm_rope_flash_attention(q, k, v, cos, sin)

        # Reference: apply RoPE separately, then eager attention
        q_rot = apply_rope_eager(q.float(), cos, sin).to(torch.bfloat16)
        k_rot = apply_rope_eager(k.float(), cos, sin).to(torch.bfloat16)
        out_ref = sp2norm_flash_attention_eager(q_rot, k_rot, v)

        err = (out_fused.float() - out_ref.float()).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        print(f"  B={B} H={H} T={T} D={D}: max_err={max_err:.4f}, mean_err={mean_err:.6f}")
        assert max_err < 0.05, f"Fused RoPE max error too high: {max_err}"
        assert not out_fused.isnan().any()
    print("  PASSED\n")


def test_rope_flash_gqa():
    """Fused RoPE + flash with GQA."""
    print("=== TEST: Fused RoPE + Flash + GQA ===")
    B, H_q, H_kv, T, D = 2, 8, 2, 128, 32
    q = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    cos, sin = precompute_rope_cos_sin(D, T, device="cuda", dtype=torch.float32)

    out_fused = sp2norm_rope_flash_attention(q, k, v, cos, sin)

    # Reference
    q_rot = apply_rope_eager(q.float(), cos, sin).to(torch.bfloat16)
    k_rot = apply_rope_eager(k.float(), cos, sin).to(torch.bfloat16)
    out_ref = sp2norm_flash_attention_eager(q_rot, k_rot, v)

    max_err = (out_fused.float() - out_ref.float()).abs().max().item()
    print(f"  GQA H_q={H_q} H_kv={H_kv}: max_err={max_err:.4f}")
    assert max_err < 0.05
    assert out_fused.shape == (B, H_q, T, D)
    print("  PASSED\n")


def test_rope_flash_window():
    """Fused RoPE + flash with sliding window."""
    print("=== TEST: Fused RoPE + Flash + Window ===")
    B, H, T, D, W = 2, 4, 128, 32, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    cos, sin = precompute_rope_cos_sin(D, T, device="cuda", dtype=torch.float32)

    out_fused = sp2norm_rope_flash_attention(q, k, v, cos, sin, window_size=W)

    q_rot = apply_rope_eager(q.float(), cos, sin).to(torch.bfloat16)
    k_rot = apply_rope_eager(k.float(), cos, sin).to(torch.bfloat16)
    out_ref = sp2norm_flash_attention_eager(q_rot, k_rot, v, window_size=W)

    max_err = (out_fused.float() - out_ref.float()).abs().max().item()
    print(f"  Window W={W}: max_err={max_err:.4f}")
    assert max_err < 0.05
    print("  PASSED\n")


def test_proj_correctness():
    """Fused output projection should match separate attention + matmul."""
    print("=== TEST: Fused Attention + Output Projection ===")
    B, H, T, D, D_model = 2, 4, 64, 32, 128
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    cos, sin = precompute_rope_cos_sin(D, T, device="cuda", dtype=torch.float32)

    # W_o: [H, D, D_model]
    w_o = torch.randn(H, D, D_model, device="cuda", dtype=torch.bfloat16) * 0.02

    # Fused path
    out_fused = sp2norm_rope_flash_attention_proj(q, k, v, w_o, cos, sin)

    # Reference: separate RoPE, attention, then projection
    q_rot = apply_rope_eager(q.float(), cos, sin).to(torch.bfloat16)
    k_rot = apply_rope_eager(k.float(), cos, sin).to(torch.bfloat16)
    attn_out = sp2norm_flash_attention_eager(q_rot, k_rot, v)  # [B, H, T, D]

    # Manual projection: sum over heads of attn_out[h] @ W_o[h]
    out_ref = torch.zeros(B, T, D_model, device="cuda", dtype=torch.float32)
    for h in range(H):
        out_ref += torch.matmul(attn_out[:, h, :, :].float(), w_o[h].float())
    out_ref = out_ref.to(torch.bfloat16)

    assert out_fused.shape == (B, T, D_model)
    err = (out_fused.float() - out_ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f"  B={B} H={H} T={T} D={D} D_model={D_model}: max_err={max_err:.4f}, mean_err={mean_err:.6f}")
    assert max_err < 0.1, f"Proj max error too high: {max_err}"
    assert not out_fused.isnan().any()
    print("  PASSED\n")


def benchmark_rope_fusion():
    """Benchmark fused vs separate RoPE + attention."""
    print("=== BENCHMARK: Fused RoPE vs Separate RoPE + Attention ===")
    from smooth_attn import sp2norm_flash_attention
    from smooth_attn.kernels import _sp2norm_flash_fwd

    print(f"{'config':>25s} {'separate':>10s} {'fused':>10s} {'speedup':>8s}")
    print("-" * 58)

    for B, H, T, D in [(2, 8, 256, 64), (2, 8, 512, 64), (2, 8, 1024, 64),
                         (4, 12, 512, 64), (2, 32, 1024, 128)]:
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = precompute_rope_cos_sin(D, T, device="cuda", dtype=torch.float32)

        def run_separate(q=q, k=k, v=v, cos=cos, sin=sin, s=scale):
            with torch.no_grad():
                q_rot = apply_rope_eager(q.float(), cos, sin).to(torch.bfloat16)
                k_rot = apply_rope_eager(k.float(), cos, sin).to(torch.bfloat16)
                return _sp2norm_flash_fwd(q_rot, k_rot, v, s, 1, 0)

        def run_fused(q=q, k=k, v=v, cos=cos, sin=sin):
            with torch.no_grad():
                return sp2norm_rope_flash_attention(q, k, v, cos, sin)

        sep_us = bench_us(run_separate)
        fused_us = bench_us(run_fused)
        speedup = sep_us / fused_us
        label = f"B={B} H={H} T={T} D={D}"
        print(f"{label:>25s} {sep_us:8.1f}us {fused_us:8.1f}us {speedup:7.2f}x")


def main():
    test_rope_flash_correctness()
    test_rope_flash_gqa()
    test_rope_flash_window()
    test_proj_correctness()
    benchmark_rope_fusion()


if __name__ == "__main__":
    main()
