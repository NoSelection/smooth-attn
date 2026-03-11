# Mutation Showdown Sandbox

This folder compares three new mutation ideas against the current production
winner:

`squareplus(2 * (x - 0.5))^2 / sum`

## What it tests

1. `Shifted ReLU-Squared`

   ```text
   relu(x - theta)^2 / sum
   ```

   This is the "true zero" variant. It is directly comparable to the current
   normalized attention operator.

2. `Depth-Evolving Thresholds`

   ```text
   squareplus(2 * (x - theta_l))^2 / sum
   theta_l = theta_base + gamma * l
   ```

   This reuses the current winning family, but sweeps `theta` across layers to
   see whether the mechanism can tighten its focus with depth at little systems
   cost.

3. `Denominator-Free Squareplus`

   ```text
   squareplus(2 * (x - theta))^2
   ```

   This is not apples-to-apples normalized attention anymore. It is benchmarked
   separately as a risky throughput probe, and the report focuses on scale and
   gradient health rather than row sums.

## Run

From the repo root:

```bash
python3 experiments/mutation_showdown/benchmark.py --dtypes bf16 --lengths 128,256,512,1024
python3 experiments/mutation_showdown/benchmark.py --dtypes bf16 --lengths 128 --quick --check-grad
```

## Read the sections

- `Normalized Showdown`
  - `relu/base < 1.0x` means shifted ReLU-squared beats the current winner.

- `Depth Schedule Probe`
  - `sched/fixed < 1.0x` means varying `theta` across layers is not adding much overhead.
  - `entropy_last < entropy_first` usually means the schedule is tightening attention depth-wise.

- `Denominator-Free Probe`
  - `raw/base < 1.0x` means the raw elementwise variant is faster than the normalized winner.
  - `raw_mean` and `raw_max` tell you whether the output scale looks sane enough to be fed into the next block with extra normalization.

- `Gradient Health`
  - `grad_nonzero` and `grad_finite` help catch dead or unstable variants quickly.
