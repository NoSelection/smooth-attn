# Formula Evolution Sandbox

This folder is a contained search harness for exploring attention normalization
formulas without touching the production `smooth_attn` path.

## Purpose

The current production winner is:

`squareplus(2 * (x - 0.5))^2 / sum`

Before we spend more time hand-designing formulas, this sandbox lets us run a
small evolutionary search and rank candidates by:

- speed proxy vs the current champion
- speed proxy vs PyTorch masked softmax
- dead-row behavior
- gradient health

## Search Spaces

`broad`

The original search mutates candidates over:

- center mode: fixed `theta` or visible-prefix row mean
- activation: `squareplus`, `softplus`, or `relu`
- `alpha`
- `theta` for fixed-center formulas
- power `p in {1, 2, 3}`

`squareplus_local`

The tighter local search keeps the family fixed to the current winner:

- center mode: fixed `theta`
- activation: `squareplus`
- mutate only `alpha`, `theta`, and `power`
- bias the initial population around `squareplus(2 * (x - 0.5))^2 / sum`

All candidates are evaluated in eager mode on CUDA so the search stays generic.
The top candidates can then be ported into Triton if they look worthwhile.

## Run

From the repo root:

```bash
python3 experiments/formula_evolution/search.py --dtypes bf16 --lengths 128,256,512 --generations 6 --population 18
python3 experiments/formula_evolution/search.py --dtypes bf16 --lengths 128,256,512,1024 --generations 8 --population 24 --save-json experiments/formula_evolution/latest.json
python3 experiments/formula_evolution/search.py --search-space squareplus_local --dtypes bf16 --lengths 128,256,512,1024 --population 48 --generations 16 --warmup 12 --iters 30 --seeds 1234,2027,4099
```

## Read the Output

- `cand/base`: lower is better; below `1.0x` beats the current squareplus champion.
- `cand/torch`: lower is better; below `1.0x` beats PyTorch masked softmax in the proxy benchmark.
- `dead_rows`: lower is better; nonzero values usually mean the formula is brittle.
- `grad_nonzero`: higher is usually better; it indicates how much of the input still gets gradient signal.

## Note

This is a proxy search, not a final verdict. A candidate still needs a real
Triton kernel and the production causal benchmark before it can replace the
mainline formula.

For longer runs, prefer adding more seeds instead of only more generations. That
reduces overfitting to one random score sample.
