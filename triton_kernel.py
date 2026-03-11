"""
Compatibility entrypoint for the smooth-attn package.

This preserves the original workflow:
    python3 triton_kernel.py ...
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smooth_attn.kernels import *  # noqa: F401,F403
from smooth_attn.kernels import main


if __name__ == "__main__":
    main()
