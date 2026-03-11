import pytest


triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from smooth_attn import FamilyConfig, SoftplusNormCausal, softplus_norm_causal


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
