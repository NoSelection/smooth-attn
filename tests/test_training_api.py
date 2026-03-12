import pytest


triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from smooth_attn import (
    FamilyConfig,
    SoftplusNormCausal,
    softplus_norm_causal,
    softplus_norm_causal_eager,
    softplus_norm_causal_triton_out,
    sp2norm_flash_attention,
    sp2norm_flash_attention_eager,
    sp2norm_fp8_flash_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_module_forward_shape():
    op = SoftplusNormCausal(family=FamilyConfig())
    x = torch.randn(1, 2, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = op(x)
    assert y.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_function_backward_runs():
    x = torch.randn(1, 2, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = softplus_norm_causal(x)
    loss = y.float().mean()
    loss.backward()
    assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("seq_len", "family", "atol"),
    [
        (64, FamilyConfig(), 5e-3),
        (512, FamilyConfig(), 5e-3),
        (128, FamilyConfig(alpha=1.5, theta=0.25, power=3), 7e-3),
    ],
)
def test_causal_triton_matches_eager(seq_len, family, atol):
    x = torch.randn(1, 2, seq_len, seq_len, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    y, meta = softplus_norm_causal_triton_out(out, x, family=family, return_meta=True)
    ref = softplus_norm_causal_eager(x.float(), family=family)

    torch.testing.assert_close(y.float(), ref, atol=atol, rtol=5e-2)
    torch.testing.assert_close(
        y.float().sum(dim=-1),
        torch.ones_like(ref.sum(dim=-1)),
        atol=atol,
        rtol=0.0,
    )
    if family == FamilyConfig():
        assert meta["variant"].startswith("squareplus_winner")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backward_matches_eager_reference():
    family = FamilyConfig()
    x = torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float32)

    y_triton = softplus_norm_causal(x, family=family, implementation="triton")
    (y_triton.float() * upstream).sum().backward()
    grad_triton = x.grad.detach().clone()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = softplus_norm_causal_eager(x_ref.float(), family=family)
    (y_ref * upstream).sum().backward()
    grad_ref = x_ref.grad.detach()

    torch.testing.assert_close(grad_triton.float(), grad_ref.float(), atol=5e-3, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
def test_flash_attention_matches_eager(seq_len):
    B, H, D = 2, 4, 32
    q = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)

    out_flash = sp2norm_flash_attention(q, k, v)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert not out_flash.isnan().any(), "NaN in flash output"
    torch.testing.assert_close(out_flash.float(), out_eager.float(), atol=2e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_exact_math_matches_eager():
    B, H, T, D = 2, 4, 256, 32
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_flash = sp2norm_flash_attention(q, k, v, exact_math=True)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert not out_flash.isnan().any(), "NaN in exact-math flash output"
    torch.testing.assert_close(out_flash.float(), out_eager.float(), atol=2e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_noncontiguous_matches_eager():
    B, T, H_q, H_kv, D, W = 2, 128, 8, 2, 32, 32
    q = torch.randn(B, T, H_q, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(B, T, H_kv, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn(B, T, H_kv, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2)

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    out_flash = sp2norm_flash_attention(q, k, v, window_size=W)
    out_eager = sp2norm_flash_attention_eager(q, k, v, window_size=W)

    assert not out_flash.isnan().any(), "NaN in non-contiguous flash output"
    torch.testing.assert_close(out_flash.float(), out_eager.float(), atol=2e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_backward():
    import math
    from smooth_attn import softplus_norm_causal_eager

    B, H, T, D = 1, 2, 64, 32
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

    out = sp2norm_flash_attention(q, k, v)
    (out.float() * upstream).sum().backward()
    dq_flash, dk_flash, dv_flash = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    attn = softplus_norm_causal_eager(scores)
    (torch.matmul(attn, v_ref) * upstream).sum().backward()

    torch.testing.assert_close(dq_flash.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dk_flash.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dv_flash.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_exact_math_backward():
    B, H, T, D = 1, 2, 64, 32
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

    out = sp2norm_flash_attention(q, k, v, exact_math=True)
    (out.float() * upstream).sum().backward()
    dq_flash, dk_flash, dv_flash = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    out_eager = sp2norm_flash_attention_eager(q_ref, k_ref, v_ref)
    (out_eager.float() * upstream).sum().backward()

    torch.testing.assert_close(dq_flash.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dk_flash.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dv_flash.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_noncontiguous_backward():
    B, T, H_q, H_kv, D, W = 1, 64, 8, 2, 32, 32
    q = torch.randn(B, T, H_q, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2).detach().requires_grad_(True)
    k = torch.randn(B, T, H_kv, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2).detach().requires_grad_(True)
    v = torch.randn(B, T, H_kv, D, device="cuda", dtype=torch.bfloat16).transpose(1, 2).detach().requires_grad_(True)
    upstream = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.float32)

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    out = sp2norm_flash_attention(q, k, v, window_size=W)
    (out.float() * upstream).sum().backward()
    dq_flash, dk_flash, dv_flash = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    out_eager = sp2norm_flash_attention_eager(q_ref, k_ref, v_ref, window_size=W)
    (out_eager.float() * upstream).sum().backward()

    torch.testing.assert_close(dq_flash.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dk_flash.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dv_flash.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_flash_attention_gqa_forward(seq_len):
    """GQA forward: H_q=8, H_kv=2 (4 query heads per KV head)."""
    B, H_q, H_kv, D = 2, 8, 2, 32
    q = torch.randn(B, H_q, seq_len, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, seq_len, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, seq_len, D, device="cuda", dtype=torch.bfloat16)

    out_flash = sp2norm_flash_attention(q, k, v)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert out_flash.shape == (B, H_q, seq_len, D)
    assert not out_flash.isnan().any(), "NaN in GQA flash output"
    torch.testing.assert_close(out_flash.float(), out_eager.float(), atol=2e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_flash_attention_window_forward(seq_len):
    """Sliding window forward: window_size=32."""
    B, H, D, W = 2, 4, 32, 32
    q = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)

    out_flash = sp2norm_flash_attention(q, k, v, window_size=W)
    out_eager = sp2norm_flash_attention_eager(q, k, v, window_size=W)

    assert not out_flash.isnan().any(), "NaN in window flash output"
    torch.testing.assert_close(out_flash.float(), out_eager.float(), atol=2e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("H_q", "H_kv"),
    [(8, 2), (8, 4), (4, 1)],
)
def test_flash_attention_gqa_backward(H_q, H_kv):
    """GQA backward: verify dQ, dK, dV match eager reference."""
    B, T, D = 1, 64, 32

    q = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.float32)

    out = sp2norm_flash_attention(q, k, v)
    (out.float() * upstream).sum().backward()
    dq_flash, dk_flash, dv_flash = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    out_eager = sp2norm_flash_attention_eager(q_ref, k_ref, v_ref)
    (out_eager.float() * upstream).sum().backward()

    torch.testing.assert_close(dq_flash.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dk_flash.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dv_flash.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("window_size", [32, 64])
def test_flash_attention_window_backward(window_size):
    """Sliding window backward: verify gradients match eager reference."""
    B, H, T, D = 1, 2, 128, 32

    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

    out = sp2norm_flash_attention(q, k, v, window_size=window_size)
    (out.float() * upstream).sum().backward()
    dq_flash, dk_flash, dv_flash = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    out_eager = sp2norm_flash_attention_eager(q_ref, k_ref, v_ref, window_size=window_size)
    (out_eager.float() * upstream).sum().backward()

    torch.testing.assert_close(dq_flash.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dk_flash.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dv_flash.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)


# NOTE: test_flash_attention_gqa_window_backward is the LAST bwd test before FP8 tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_attention_gqa_window_backward():
    """Combined GQA + sliding window backward."""
    B, H_q, H_kv, T, D, W = 1, 8, 2, 64, 32, 32

    q = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.float32)

    out = sp2norm_flash_attention(q, k, v, window_size=W)
    (out.float() * upstream).sum().backward()
    dq_flash, dk_flash, dv_flash = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    out_eager = sp2norm_flash_attention_eager(q_ref, k_ref, v_ref, window_size=W)
    (out_eager.float() * upstream).sum().backward()

    torch.testing.assert_close(dq_flash.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dk_flash.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    torch.testing.assert_close(dv_flash.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)


# ---------------------------------------------------------------------------
# FP8 Flash Attention Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
def test_fp8_flash_attention_matches_eager(seq_len):
    """FP8 flash attention forward matches eager reference within FP8 tolerance."""
    B, H, D = 2, 4, 32
    q = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert not out_fp8.isnan().any(), "NaN in FP8 output"
    err = (out_fp8.float() - out_eager.float()).abs()
    assert err.max().item() < 0.4, f"FP8 max error too high: {err.max().item()}"
    assert err.mean().item() < 0.02, f"FP8 mean error too high: {err.mean().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_flash_attention_gqa():
    """FP8 with GQA (H_q=8, H_kv=2)."""
    B, H_q, H_kv, T, D = 2, 8, 2, 128, 32
    q = torch.randn(B, H_q, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v)
    out_eager = sp2norm_flash_attention_eager(q, k, v)

    assert out_fp8.shape == (B, H_q, T, D)
    assert not out_fp8.isnan().any()
    assert (out_fp8.float() - out_eager.float()).abs().max().item() < 0.3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_flash_attention_window():
    """FP8 with sliding window."""
    B, H, T, D, W = 2, 4, 128, 32, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_fp8 = sp2norm_fp8_flash_attention(q, k, v, window_size=W)
    out_eager = sp2norm_flash_attention_eager(q, k, v, window_size=W)

    assert not out_fp8.isnan().any()
    assert (out_fp8.float() - out_eager.float()).abs().max().item() < 0.3
