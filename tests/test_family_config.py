import pytest


def test_family_config_rejects_bad_power():
    triton = pytest.importorskip("triton")
    assert triton is not None
    from smooth_attn import FamilyConfig

    with pytest.raises(ValueError):
        FamilyConfig(power=4)
