# smooth-attn

Smooth squareplus attention normalization kernels for PyTorch + Triton.

`smooth-attn` packages the prototype you built here into a cleaner repo shape:
- Triton forward kernels for row-wise and causal normalization
- a production-focused causal benchmark centered on reusable output buffers
- a differentiable training API with Triton forward and safe eager backward
- the winning squareplus family `squareplus(alpha * (x - theta))^power / sum`

## Why this exists

Softmax is not the only useful normalization for attention. This repo explores a smooth thresholded family that keeps the operator normalized while changing the geometry of the scores:

```text
squareplus(alpha * (x - theta))^power / sum
```

In the current prototype, the best regime is causal attention in mixed precision. On the local RTX 4090 runs from this workspace, the `bf16` causal family kernel reached a geometric mean of roughly `0.81x` vs PyTorch masked softmax timing, meaning about `1.24x` faster overall in that benchmark configuration. Results will vary by GPU, Triton version, and shape mix.

## Status

- Forward Triton kernels: implemented
- Causal mixed-precision benchmark: implemented
- Public training API: implemented
- Backward: correct eager fallback, not fused Triton yet

## Install

Editable install:

```bash
pip install -e .[dev]
```

CLI entrypoint after install:

```bash
smooth-attn --mode causal --dtypes bf16 --lengths 128,256,512,1024
```

You can also keep using the compatibility script from the repo root:

```bash
python3 triton_kernel.py --mode causal --dtypes bf16 --lengths 128,256,512,1024
```

## Quickstart

```python
import torch
from smooth_attn import FamilyConfig, SoftplusNormCausal

family = FamilyConfig(alpha=2.0, theta=0.5, power=2)
op = SoftplusNormCausal(family=family)

scores = torch.randn(
    2, 8, 512, 512,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)

attn = op(scores)
loss = attn.float().mean()
loss.backward()
```

If you want the raw Triton out-buffer path:

```python
import torch
from smooth_attn import FamilyConfig, softplus_norm_causal_triton_out

family = FamilyConfig(alpha=2.0, theta=0.5, power=2)
scores = torch.randn(2, 8, 512, 512, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(scores)

softplus_norm_causal_triton_out(out, scores, family=family)
```

## Benchmarking

Production-oriented causal sweep:

```bash
python3 triton_kernel.py --mode causal --dtypes bf16 --lengths 128,256,512,1024
```

Forward + backward validation:

```bash
python3 triton_kernel.py --mode causal --dtypes bf16 --lengths 128 --quick --check-backward
```

Legacy row-wise microbenchmarks:

```bash
python3 triton_kernel.py --mode rowwise --dtypes fp32
```

## Repo layout

```text
src/smooth_attn/
  __init__.py
  __main__.py
  kernels.py
benchmarks/
  causal_benchmark.py
experiments/
  formula_evolution/
  long_context/
  mutation_showdown/
  training/
examples/
  mini_attention.py
results/
  long_context/
  training/
tests/
  test_family_config.py
  test_training_api.py
triton_kernel.py
```

Research scripts and raw result logs are kept out of the package code on purpose:
- `experiments/` contains one-off training runs, sweeps, and formula probes.
- `results/` contains the text logs that back those experiments.

## Notes

- This project currently targets NVIDIA GPUs with CUDA and Triton.
- The backward pass is intentionally conservative for now: Triton forward, eager backward.
- The mainline keeps the winning fixed-theta squareplus family and drops the losing formula branches from the public CLI.
- The next major systems win is a fused Triton backward for the causal family path.

## License

MIT
