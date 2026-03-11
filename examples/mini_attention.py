from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smooth_attn import FamilyConfig, SoftplusNormCausal


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example.")

    family = FamilyConfig(alpha=2.0, theta=0.5, power=2)
    op = SoftplusNormCausal(family=family)

    scores = torch.randn(
        2,
        8,
        512,
        512,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    values = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)

    attn = op(scores)
    out = attn @ values
    loss = out.float().pow(2).mean()
    loss.backward()

    print("attn shape:", tuple(attn.shape))
    print("out shape:", tuple(out.shape))
    print("loss:", float(loss))


if __name__ == "__main__":
    main()
