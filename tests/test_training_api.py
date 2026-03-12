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
