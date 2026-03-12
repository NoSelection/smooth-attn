"""
Production-ready Triton kernels for row-wise and causal squareplus normalization.

The main benchmark entrypoint is centered on the regime that currently looks best:
causal attention with reusable output buffers and mixed precision.

Legacy `softplus_norm_*` API names are kept for compatibility, but the mainline
implementation is now the winning squareplus family.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver


SUPPORTED_POWERS = (1, 2, 3)
SUPPORTED_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DEFAULT_ROWWISE_SHAPES = (
    (256, 256),
    (256, 512),
    (256, 1024),
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (4096, 128),
    (4096, 256),
    (4096, 512),
    (4096, 1024),
    (8192, 256),
    (8192, 512),
    (8192, 1024),
    (16384, 128),
    (16384, 256),
    (16384, 512),
    (32768, 256),
    (32768, 512),
    (32, 512),
    (32, 1024),
    (32, 2048),
    (128, 512),
    (128, 1024),
    (128, 2048),
)
DEFAULT_CAUSAL_LENGTHS = (128, 256, 512, 1024)
EPS = 1e-12

_DEVICE_PROP_CACHE = {}
_CAUSAL_MASK_CACHE = {}
_LAUNCH_META_CACHE = {}
_TUNED_CONFIG_CACHE = {}


@dataclass(frozen=True)
class FamilyConfig:
    alpha: float = 2.0
    theta: float = 0.5
    power: int = 2

    def __post_init__(self):
        if self.power not in SUPPORTED_POWERS:
            raise ValueError(f"power must be one of {SUPPORTED_POWERS}, got {self.power}")

    @property
    def cache_key(self):
        return (float(self.alpha), float(self.theta), int(self.power))

    @property
    def kernel_kwargs(self):
        return {
            "ALPHA": float(self.alpha),
            "THETA": float(self.theta),
            "POWER": int(self.power),
        }

    @property
    def label(self):
        return f"squareplus({self.alpha:g} * (x - {self.theta:g}))^{self.power} / sum"


@dataclass(frozen=True)
class RowwiseBenchmarkResult:
    shape: tuple[int, int]
    torch_softmax_us: float
    triton_softmax_us: float
    family_us: float


@dataclass(frozen=True)
class CausalBenchmarkResult:
    dtype: torch.dtype
    batch: int
    heads: int
    seq_len: int
    torch_softmax_us: float
    triton_softmax_us: float
    family_us: float
    softmax_err: float
    family_err: float
    softmax_sums_ok: bool
    family_sums_ok: bool
    softmax_meta: dict
    family_meta: dict


DEFAULT_FAMILY = FamilyConfig()

# Backward-compatible constants for ad-hoc probes.
BENCH_ALPHA = DEFAULT_FAMILY.alpha
BENCH_THETA = DEFAULT_FAMILY.theta
BENCH_POWER = DEFAULT_FAMILY.power
def _current_device_index():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these Triton kernels")
    return torch.cuda.current_device()


def _device_props():
    device_index = _current_device_index()
    if device_index not in _DEVICE_PROP_CACHE:
        device = driver.active.get_active_torch_device()
        props = driver.active.utils.get_device_properties(device.index)
        _DEVICE_PROP_CACHE[device_index] = {
            "num_sm": props["multiprocessor_count"],
            "num_regs": props["max_num_regs"],
            "size_smem": props["max_shared_mem"],
            "warp_size": props["warpSize"],
        }
    return _DEVICE_PROP_CACHE[device_index]


def _dtype_label(dtype):
    for label, candidate in SUPPORTED_DTYPES.items():
        if candidate == dtype:
            return label
    return str(dtype).replace("torch.", "")


def _parse_dtype_list(text):
    labels = [token.strip().lower() for token in text.split(",") if token.strip()]
    if not labels:
        raise ValueError("at least one dtype must be provided")
    try:
        return [SUPPORTED_DTYPES[label] for label in labels]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype label: {exc.args[0]}") from exc


def _parse_int_list(text):
    values = [token.strip() for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("at least one integer must be provided")
    return [int(token) for token in values]


def _resolve_family(alpha, theta, power, family):
    if family is None:
        return FamilyConfig(alpha=alpha, theta=theta, power=power)
    if (alpha, theta, power) != (
        DEFAULT_FAMILY.alpha,
        DEFAULT_FAMILY.theta,
        DEFAULT_FAMILY.power,
    ):
        raise ValueError("pass either family=... or alpha/theta/power overrides, not both")
    return family


def _validate_cuda_tensor(x, *, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype not in SUPPORTED_DTYPES.values():
        raise ValueError(
            f"{name} dtype must be one of {tuple(SUPPORTED_DTYPES.values())}, got {x.dtype}"
        )


def _prepare_rowwise_input(x):
    _validate_cuda_tensor(x, name="x")
    if x.ndim != 2:
        raise ValueError(f"row-wise kernels expect a 2D tensor, got shape {tuple(x.shape)}")
    return x if x.is_contiguous() else x.contiguous()


def _prepare_attention_input(x):
    _validate_cuda_tensor(x, name="x")
    if x.ndim < 2:
        raise ValueError("attention scores must have at least 2 dimensions")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("causal attention expects square score matrices on the last two dims")
    return x if x.is_contiguous() else x.contiguous()


def _prepare_output(out, reference):
    _validate_cuda_tensor(out, name="out")
    if out.shape != reference.shape:
        raise ValueError(f"out shape {tuple(out.shape)} must match input shape {tuple(reference.shape)}")
    if out.dtype != reference.dtype:
        raise ValueError(f"out dtype {out.dtype} must match input dtype {reference.dtype}")
    if out.device != reference.device:
        raise ValueError("out and input must live on the same CUDA device")
    if out.is_contiguous():
        return out, None
    return torch.empty_like(reference), out


def _finalize_output(kernel_out, user_out):
    if user_out is None:
        return kernel_out
    user_out.copy_(kernel_out)
    return user_out


def _causal_mask(seq_len, device):
    key = (device.index, seq_len)
    if key not in _CAUSAL_MASK_CACHE:
        _CAUSAL_MASK_CACHE[key] = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
    return _CAUSAL_MASK_CACHE[key]


def _pick_block_size(n_cols):
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 4096:
        raise ValueError(f"n_cols={n_cols} is too large for this fused row kernel")
    return block_size


def _candidate_configs(block_size):
    if block_size <= 256:
        return [(2, 2), (4, 2), (4, 4), (8, 2)]
    if block_size <= 512:
        return [(4, 2), (4, 4), (8, 2), (8, 4)]
    return [(4, 2), (8, 2), (8, 4)]


def _winner_candidate_configs(seq_len):
    if seq_len <= 256:
        return [(2, 2), (2, 4), (4, 2), (4, 4), (8, 2), (8, 4)]
    return [(2, 2), (2, 4), (4, 2), (4, 4), (8, 2), (8, 4)]


def _default_config(block_size):
    if block_size <= 256:
        return 4, 2
    return 8, 2


def _get_launch_meta(
    name,
    kernel,
    x,
    out,
    n_rows,
    n_cols,
    block_size,
    num_warps,
    num_stages,
    extra_cache_key=(),
    kernel_kwargs=None,
):
    device_props = _device_props()
    key = (
        name,
        _current_device_index(),
        str(x.dtype),
        block_size,
        num_warps,
        num_stages,
        *extra_cache_key,
    )
    if key in _LAUNCH_META_CACHE:
        return _LAUNCH_META_CACHE[key]

    compiled = kernel.warmup(
        x,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        grid=(1,),
        **(kernel_kwargs or {}),
    )
    compiled._init_handles()

    n_regs = max(1, compiled.n_regs)
    size_smem = compiled.metadata.shared
    occupancy = max(1, device_props["num_regs"] // (n_regs * device_props["warp_size"] * num_warps))
    if size_smem > 0:
        occupancy = min(occupancy, max(1, device_props["size_smem"] // size_smem))

    meta = {
        "occupancy": occupancy,
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "n_regs": n_regs,
        "shared_mem": size_smem,
    }
    _LAUNCH_META_CACHE[key] = meta
    return meta


def _median_cuda_us(callback, warmup, iters):
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        callback()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(start.elapsed_time(end) * 1000 for start, end in zip(starts, ends))
    return times[len(times) // 2]


def _measure_kernel_us(
    kernel,
    x,
    out,
    n_rows,
    n_cols,
    block_size,
    num_warps,
    num_stages,
    num_programs,
    kernel_kwargs=None,
    warmup=10,
    iters=40,
):
    return _median_cuda_us(
        lambda: kernel[(num_programs,)](
            x,
            out,
            n_rows,
            n_cols,
            x.stride(0),
            out.stride(0),
            BLOCK_SIZE=block_size,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
            **(kernel_kwargs or {}),
        ),
        warmup=warmup,
        iters=iters,
    )


def _tune_launch_config(
    name,
    kernel,
    x,
    n_rows,
    n_cols,
    block_size,
    extra_cache_key=(),
    kernel_kwargs=None,
    candidate_configs=None,
):
    device_props = _device_props()
    key = (
        name,
        _current_device_index(),
        str(x.dtype),
        n_rows,
        n_cols,
        *extra_cache_key,
    )
    if key in _TUNED_CONFIG_CACHE:
        return _TUNED_CONFIG_CACHE[key]

    out = torch.empty_like(x)
    best = None
    for num_warps, num_stages in (candidate_configs or _candidate_configs(block_size)):
        meta = _get_launch_meta(
            name,
            kernel,
            x,
            out,
            n_rows,
            n_cols,
            block_size,
            num_warps,
            num_stages,
            extra_cache_key=extra_cache_key,
            kernel_kwargs=kernel_kwargs,
        )
        num_programs = max(1, min(device_props["num_sm"] * meta["occupancy"], n_rows))
        t_us = _measure_kernel_us(
            kernel,
            x,
            out,
            n_rows,
            n_cols,
            block_size,
            num_warps,
            num_stages,
            num_programs,
            kernel_kwargs=kernel_kwargs,
        )
        candidate = {
            **meta,
            "num_programs": num_programs,
            "time_us": t_us,
        }
        if best is None or candidate["time_us"] < best["time_us"]:
            best = candidate

    _TUNED_CONFIG_CACHE[key] = best
    return best


def _heuristic_launch_config(
    name,
    kernel,
    x,
    n_rows,
    n_cols,
    block_size,
    extra_cache_key=(),
    kernel_kwargs=None,
):
    device_props = _device_props()
    out = torch.empty_like(x)
    num_warps, num_stages = _default_config(block_size)
    meta = _get_launch_meta(
        name,
        kernel,
        x,
        out,
        n_rows,
        n_cols,
        block_size,
        num_warps,
        num_stages,
        extra_cache_key=extra_cache_key,
        kernel_kwargs=kernel_kwargs,
    )
    return {
        **meta,
        "num_programs": max(1, min(device_props["num_sm"] * meta["occupancy"], n_rows)),
    }


@triton.jit
def _fast_sqrt(x):
    # rsqrt is a single-cycle SFU instruction on NVIDIA; sqrt(x) = x * rsqrt(x)
    return x * tl.rsqrt(x)


@triton.jit
def _apply_family_activation(arg):
    s = arg * arg + 4.0
    return 0.5 * (arg + _fast_sqrt(s))


@triton.jit
def _apply_winner_squareplus_p2(arg):
    # squareplus(arg)^2 = 0.25 * (arg + sqrt(arg^2 + 4))^2
    s = arg * arg + 4.0
    tmp = arg + _fast_sqrt(s)
    return 0.25 * tmp * tmp


@triton.jit
def _softplus_norm_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    ALPHA: tl.constexpr,
    THETA: tl.constexpr,
    POWER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

        arg = ALPHA * (x - THETA)
        sp = _apply_family_activation(arg)
        if POWER == 1:
            y = sp
        elif POWER == 2:
            y = sp * sp
        else:
            y = sp * sp * sp
        y = tl.where(mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        out = tl.fdiv(y, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


@triton.jit
def _softmax_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=mask, other=-float("inf")).to(tl.float32)

        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        e = tl.where(mask, e, 0.0)

        row_sum = tl.sum(e, axis=0)
        out = tl.fdiv(e, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


@triton.jit
def _softplus_norm_causal_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    ALPHA: tl.constexpr,
    THETA: tl.constexpr,
    POWER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=causal_mask, other=0.0).to(tl.float32)

        arg = ALPHA * (x - THETA)
        sp = _apply_family_activation(arg)
        if POWER == 1:
            y = sp
        elif POWER == 2:
            y = sp * sp
        else:
            y = sp * sp * sp
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        out = tl.fdiv(y, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


@triton.jit
def _winner_squareplus_p2_causal_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=causal_mask, other=0.0).to(tl.float32)

        arg = x + x - 1.0
        y = _apply_winner_squareplus_p2(arg)
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        out = tl.fdiv(y, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


@triton.jit
def _winner_squareplus_p2_causal_tiered_fwd(
    input_ptr,
    output_ptr,
    n_tier_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    row_start,
    n_matrices,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    total_work = n_tier_rows * n_matrices
    for work_idx in tl.range(pid, total_work, row_step, num_stages=NUM_STAGES):
        mat_idx = work_idx // n_tier_rows
        local_row = work_idx % n_tier_rows
        actual_row = mat_idx * n_cols + row_start + local_row
        q_pos = row_start + local_row

        causal_mask = col_offsets <= q_pos
        row_ptr = input_ptr + actual_row * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=causal_mask, other=0.0).to(tl.float32)

        arg = x + x - 1.0
        y = _apply_winner_squareplus_p2(arg)
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        out = tl.fdiv(y, row_sum)

        out_ptr = output_ptr + actual_row * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=causal_mask)


@triton.jit
def _softplus_norm_causal_bwd(
    input_ptr,
    grad_out_ptr,
    grad_in_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    grad_out_row_stride,
    grad_in_row_stride,
    BLOCK_SIZE: tl.constexpr,
    ALPHA: tl.constexpr,
    THETA: tl.constexpr,
    POWER: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)

        # Load x and grad_output
        x_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(x_ptr, mask=causal_mask, other=0.0).to(tl.float32)
        g_ptr = grad_out_ptr + row_idx * grad_out_row_stride + col_offsets
        g = tl.load(g_ptr, mask=causal_mask, other=0.0).to(tl.float32)

        # Forward recomputation: y_i = sp(alpha*(x-theta))^p
        arg = ALPHA * (x - THETA)
        s = arg * arg + 4.0
        sqrt_s = _fast_sqrt(s)
        sp = 0.5 * (arg + sqrt_s)
        if POWER == 1:
            y = sp
        elif POWER == 2:
            y = sp * sp
        else:
            y = sp * sp * sp
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        p = tl.fdiv(y, row_sum)

        # sp'(arg) = 0.5 * (1 + arg / sqrt(arg^2 + 4))
        sp_prime = 0.5 * (1.0 + arg * tl.rsqrt(s))
        # dy/dx = ALPHA * POWER * sp^(POWER-1) * sp'
        if POWER == 1:
            dydx = ALPHA * sp_prime
        elif POWER == 2:
            dydx = ALPHA * 2.0 * sp * sp_prime
        else:
            dydx = ALPHA * 3.0 * sp * sp * sp_prime
        dydx = tl.where(causal_mask, dydx, 0.0)

        # grad_x_i = (dydx_i / S) * (g_i - dot(g, p))
        dot_gp = tl.sum(g * p, axis=0)
        grad_x = tl.fdiv(dydx, row_sum) * (g - dot_gp)
        grad_x = tl.where(causal_mask, grad_x, 0.0)

        out_ptr = grad_in_ptr + row_idx * grad_in_row_stride + col_offsets
        tl.store(out_ptr, grad_x, mask=mask)


@triton.jit
def _winner_squareplus_p2_causal_bwd(
    input_ptr,
    grad_out_ptr,
    grad_in_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    grad_out_row_stride,
    grad_in_row_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)

        x_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(x_ptr, mask=causal_mask, other=0.0).to(tl.float32)
        g_ptr = grad_out_ptr + row_idx * grad_out_row_stride + col_offsets
        g = tl.load(g_ptr, mask=causal_mask, other=0.0).to(tl.float32)

        # a = 2x - 1, y = sp(a)^2
        arg = x + x - 1.0
        s = arg * arg + 4.0
        sqrt_s = _fast_sqrt(s)
        sp = 0.5 * (arg + sqrt_s)
        y = sp * sp
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        p = tl.fdiv(y, row_sum)

        # sp'(a) = 0.5*(1 + a*rsqrt(a^2+4)), dy/dx = 4*sp*sp'
        sp_prime = 0.5 * (1.0 + arg * tl.rsqrt(s))
        dydx = 4.0 * sp * sp_prime
        dydx = tl.where(causal_mask, dydx, 0.0)

        dot_gp = tl.sum(g * p, axis=0)
        grad_x = tl.fdiv(dydx, row_sum) * (g - dot_gp)
        grad_x = tl.where(causal_mask, grad_x, 0.0)

        out_ptr = grad_in_ptr + row_idx * grad_in_row_stride + col_offsets
        tl.store(out_ptr, grad_x, mask=mask)


@triton.jit
def _winner_squareplus_p2_causal_tiered_bwd(
    input_ptr,
    grad_out_ptr,
    grad_in_ptr,
    n_tier_rows,
    n_cols,
    input_row_stride,
    grad_out_row_stride,
    grad_in_row_stride,
    row_start,
    n_matrices,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    total_work = n_tier_rows * n_matrices
    for work_idx in tl.range(pid, total_work, row_step, num_stages=NUM_STAGES):
        mat_idx = work_idx // n_tier_rows
        local_row = work_idx % n_tier_rows
        actual_row = mat_idx * n_cols + row_start + local_row
        q_pos = row_start + local_row

        causal_mask = col_offsets <= q_pos

        x_ptr = input_ptr + actual_row * input_row_stride + col_offsets
        x = tl.load(x_ptr, mask=causal_mask, other=0.0).to(tl.float32)
        g_ptr = grad_out_ptr + actual_row * grad_out_row_stride + col_offsets
        g = tl.load(g_ptr, mask=causal_mask, other=0.0).to(tl.float32)

        arg = x + x - 1.0
        s = arg * arg + 4.0
        sqrt_s = _fast_sqrt(s)
        sp = 0.5 * (arg + sqrt_s)
        y = sp * sp
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        p = tl.fdiv(y, row_sum)

        sp_prime = 0.5 * (1.0 + arg * tl.rsqrt(s))
        dydx = 4.0 * sp * sp_prime
        dydx = tl.where(causal_mask, dydx, 0.0)

        dot_gp = tl.sum(g * p, axis=0)
        grad_x = tl.fdiv(dydx, row_sum) * (g - dot_gp)
        grad_x = tl.where(causal_mask, grad_x, 0.0)

        out_ptr = grad_in_ptr + actual_row * grad_in_row_stride + col_offsets
        tl.store(out_ptr, grad_x, mask=causal_mask)


@triton.jit
def _softmax_causal_fwd(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    for row_idx in tl.range(pid, n_rows, row_step, num_stages=NUM_STAGES):
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=mask, other=-float("inf")).to(tl.float32)

        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)
        x = tl.where(causal_mask, x, -float("inf"))

        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        e = tl.where(causal_mask, e, 0.0)

        row_sum = tl.sum(e, axis=0)
        out = tl.fdiv(e, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


def _launch_row_kernel_out(
    name,
    kernel,
    x,
    out,
    *,
    return_meta=False,
    extra_cache_key=(),
    kernel_kwargs=None,
    tune_configs=None,
):
    n_rows, n_cols = x.shape
    block_size = _pick_block_size(n_cols)
    if name == "softmax":
        launch_meta = _heuristic_launch_config(
            name,
            kernel,
            x,
            n_rows,
            n_cols,
            block_size,
            extra_cache_key=extra_cache_key,
            kernel_kwargs=kernel_kwargs,
        )
    else:
        launch_meta = _tune_launch_config(
            name,
            kernel,
            x,
            n_rows,
            n_cols,
            block_size,
            extra_cache_key=extra_cache_key,
            kernel_kwargs=kernel_kwargs,
            candidate_configs=tune_configs,
        )

    kernel[(launch_meta["num_programs"],)](
        x,
        out,
        n_rows,
        n_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=block_size,
        NUM_STAGES=launch_meta["num_stages"],
        num_warps=launch_meta["num_warps"],
        **(kernel_kwargs or {}),
    )

    if return_meta:
        return out, launch_meta
    return out


def _launch_row_kernel(name, kernel, x, *, return_meta=False, extra_cache_key=(), kernel_kwargs=None):
    out = torch.empty_like(x)
    return _launch_row_kernel_out(
        name,
        kernel,
        x,
        out,
        return_meta=return_meta,
        extra_cache_key=extra_cache_key,
        kernel_kwargs=kernel_kwargs,
    )


def _launch_attention_kernel_out(
    name,
    kernel,
    x,
    out,
    *,
    return_meta=False,
    extra_cache_key=(),
    kernel_kwargs=None,
    tune_configs=None,
):
    seq_len = x.shape[-1]
    x_2d = x.view(-1, seq_len)
    out_2d = out.view(-1, seq_len)
    result_2d, meta = _launch_row_kernel_out(
        name,
        kernel,
        x_2d,
        out_2d,
        return_meta=True,
        extra_cache_key=extra_cache_key,
        kernel_kwargs=kernel_kwargs,
        tune_configs=tune_configs,
    )
    result = result_2d.view_as(out)
    if return_meta:
        return result, meta
    return result


def softplus_norm_triton(
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
    return_meta=False,
):
    family = _resolve_family(alpha, theta, power, family)
    x_row = _prepare_rowwise_input(x)
    return _launch_row_kernel(
        "softplus_norm",
        _softplus_norm_fwd,
        x_row,
        return_meta=return_meta,
        extra_cache_key=family.cache_key,
        kernel_kwargs=family.kernel_kwargs,
    )


def softplus_norm_triton_out(
    out,
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
    return_meta=False,
):
    family = _resolve_family(alpha, theta, power, family)
    x_row = _prepare_rowwise_input(x)
    kernel_out, user_out = _prepare_output(out, x_row)
    result, meta = _launch_row_kernel_out(
        "softplus_norm",
        _softplus_norm_fwd,
        x_row,
        kernel_out,
        return_meta=True,
        extra_cache_key=family.cache_key,
        kernel_kwargs=family.kernel_kwargs,
    )
    result = _finalize_output(result, user_out)
    if return_meta:
        return result, meta
    return result


def sp2norm_triton(x, return_meta=False):
    return softplus_norm_triton(x, family=DEFAULT_FAMILY, return_meta=return_meta)


def _is_winner_family(family):
    return family.cache_key == DEFAULT_FAMILY.cache_key


_MIN_TIER_BLOCK = 128
# Tiered dispatch only pays off when T is large enough that SIMD savings
# exceed the zero_() + multi-launch overhead. On RTX 4090, ~2048+ rows.
_TIER_THRESHOLD = 2048


def _compute_tiers(seq_len):
    """Split [0, seq_len) into tiers where each tier's BLOCK_SIZE = next_pow2(tier_end).

    Returns list of (row_start, n_tier_rows, block_size).
    Only tiers when seq_len > threshold and there are at least 2 tiers.
    """
    full_block = triton.next_power_of_2(seq_len)
    if full_block <= _MIN_TIER_BLOCK or seq_len < _TIER_THRESHOLD:
        return None
    tiers = []
    pos = 0
    block = _MIN_TIER_BLOCK
    while pos < seq_len and block < full_block:
        tier_end = min(block, seq_len)
        if tier_end > pos:
            tiers.append((pos, tier_end - pos, block))
        pos = tier_end
        block *= 2
    if pos < seq_len:
        tiers.append((pos, seq_len - pos, full_block))
    return tiers if len(tiers) > 1 else None


def _launch_tiered_fwd(x_2d, out_2d, seq_len, n_matrices):
    device_props = _device_props()
    tiers = _compute_tiers(seq_len)
    if tiers is None:
        return False
    out_2d.zero_()
    for row_start, n_tier_rows, block_size in tiers:
        num_warps, num_stages = _default_config(block_size)
        total_work = n_tier_rows * n_matrices
        num_programs = max(1, min(device_props["num_sm"] * 2, total_work))
        _winner_squareplus_p2_causal_tiered_fwd[(num_programs,)](
            x_2d,
            out_2d,
            n_tier_rows,
            seq_len,
            x_2d.stride(0),
            out_2d.stride(0),
            row_start,
            n_matrices,
            BLOCK_SIZE=block_size,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
        )
    return True


def _launch_tiered_bwd(x_2d, g_2d, grad_in, seq_len, n_matrices):
    device_props = _device_props()
    tiers = _compute_tiers(seq_len)
    if tiers is None:
        return False
    for row_start, n_tier_rows, block_size in tiers:
        num_warps, num_stages = _default_config(block_size)
        total_work = n_tier_rows * n_matrices
        num_programs = max(1, min(device_props["num_sm"] * 2, total_work))
        _winner_squareplus_p2_causal_tiered_bwd[(num_programs,)](
            x_2d,
            g_2d,
            grad_in,
            n_tier_rows,
            seq_len,
            x_2d.stride(0),
            g_2d.stride(0),
            grad_in.stride(0),
            row_start,
            n_matrices,
            BLOCK_SIZE=block_size,
            NUM_STAGES=num_stages,
            num_warps=num_warps,
        )
    return True


def _generic_family_causal_triton_out(out, x, family, *, return_meta=False):
    x_attn = _prepare_attention_input(x)
    kernel_out, user_out = _prepare_output(out, x_attn)
    result, meta = _launch_attention_kernel_out(
        "softplus_norm_causal",
        _softplus_norm_causal_fwd,
        x_attn,
        kernel_out,
        return_meta=True,
        extra_cache_key=family.cache_key,
        kernel_kwargs=family.kernel_kwargs,
    )
    result = _finalize_output(result, user_out)
    if return_meta:
        return result, meta
    return result


def _winner_squareplus_causal_triton_out(out, x, *, return_meta=False):
    x_attn = _prepare_attention_input(x)
    kernel_out, user_out = _prepare_output(out, x_attn)
    seq_len = x_attn.shape[-1]
    x_2d = x_attn.view(-1, seq_len)
    out_2d = kernel_out.view(-1, seq_len)
    n_matrices = x_2d.shape[0] // seq_len

    tiered = _launch_tiered_fwd(x_2d, out_2d, seq_len, n_matrices)
    if tiered:
        result = out_2d.view_as(kernel_out)
        result = _finalize_output(result, user_out)
        meta = {"variant": "squareplus_winner_tiered", "tiers": _compute_tiers(seq_len)}
        if return_meta:
            return result, meta
        return result

    result, meta = _launch_attention_kernel_out(
        "squareplus_winner",
        _winner_squareplus_p2_causal_fwd,
        x_attn,
        kernel_out,
        return_meta=True,
        extra_cache_key=("winner_squareplus",),
        tune_configs=_winner_candidate_configs(seq_len),
    )
    result = _finalize_output(result, user_out)
    meta = {**meta, "variant": "squareplus_winner"}
    if return_meta:
        return result, meta
    return result


def _launch_bwd_row_kernel(name, kernel, x, grad_out, n_rows, n_cols, block_size, kernel_kwargs=None):
    grad_in = torch.empty_like(x)
    device_props = _device_props()
    num_warps, num_stages = _default_config(block_size)
    num_programs = max(1, min(device_props["num_sm"] * 2, n_rows))
    kernel[(num_programs,)](
        x,
        grad_out,
        grad_in,
        n_rows,
        n_cols,
        x.stride(0),
        grad_out.stride(0),
        grad_in.stride(0),
        BLOCK_SIZE=block_size,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        **(kernel_kwargs or {}),
    )
    return grad_in


def _softplus_norm_causal_backward(x, grad_output, family):
    seq_len = x.shape[-1]
    x_2d = x.reshape(-1, seq_len)
    g_2d = grad_output.reshape(-1, seq_len)
    n_rows, n_cols = x_2d.shape
    block_size = _pick_block_size(n_cols)

    if _is_winner_family(family):
        n_matrices = n_rows // n_cols
        grad_in = torch.empty_like(x_2d)
        tiered = _launch_tiered_bwd(x_2d, g_2d, grad_in, n_cols, n_matrices)
        if tiered:
            return grad_in.view_as(x)
        grad_in_2d = _launch_bwd_row_kernel(
            "winner_bwd",
            _winner_squareplus_p2_causal_bwd,
            x_2d,
            g_2d,
            n_rows,
            n_cols,
            block_size,
        )
    else:
        grad_in_2d = _launch_bwd_row_kernel(
            "generic_bwd",
            _softplus_norm_causal_bwd,
            x_2d,
            g_2d,
            n_rows,
            n_cols,
            block_size,
            kernel_kwargs=family.kernel_kwargs,
        )
    return grad_in_2d.view_as(x)


def softmax_triton(x, return_meta=False):
    x_row = _prepare_rowwise_input(x)
    return _launch_row_kernel("softmax", _softmax_fwd, x_row, return_meta=return_meta)


def softmax_triton_out(out, x, return_meta=False):
    x_row = _prepare_rowwise_input(x)
    kernel_out, user_out = _prepare_output(out, x_row)
    result, meta = _launch_row_kernel_out(
        "softmax",
        _softmax_fwd,
        x_row,
        kernel_out,
        return_meta=True,
    )
    result = _finalize_output(result, user_out)
    if return_meta:
        return result, meta
    return result


def softplus_norm_causal_triton(
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
    return_meta=False,
):
    family = _resolve_family(alpha, theta, power, family)
    x_attn = _prepare_attention_input(x)
    out = torch.empty_like(x_attn)
    return softplus_norm_causal_triton_out(
        out,
        x_attn,
        family=family,
        return_meta=return_meta,
    )


def softplus_norm_causal_triton_out(
    out,
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
    return_meta=False,
):
    family = _resolve_family(alpha, theta, power, family)
    x_attn = _prepare_attention_input(x)
    if _is_winner_family(family):
        return _winner_squareplus_causal_triton_out(
            out,
            x_attn,
            return_meta=return_meta,
        )
    return _generic_family_causal_triton_out(
        out,
        x_attn,
        family,
        return_meta=return_meta,
    )


def softmax_causal_triton(x, return_meta=False):
    x_attn = _prepare_attention_input(x)
    out = torch.empty_like(x_attn)
    return _launch_attention_kernel_out(
        "softmax_causal",
        _softmax_causal_fwd,
        x_attn,
        out,
        return_meta=return_meta,
    )


def softmax_causal_triton_out(out, x, return_meta=False):
    x_attn = _prepare_attention_input(x)
    kernel_out, user_out = _prepare_output(out, x_attn)
    result, meta = _launch_attention_kernel_out(
        "softmax_causal",
        _softmax_causal_fwd,
        x_attn,
        kernel_out,
        return_meta=True,
    )
    result = _finalize_output(result, user_out)
    if return_meta:
        return result, meta
    return result


def _apply_family_activation_eager(arg):
    return 0.5 * (arg + torch.sqrt(arg * arg + 4.0))


def softplus_norm_eager(
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
):
    family = _resolve_family(alpha, theta, power, family)
    y = _apply_family_activation_eager(family.alpha * (x - family.theta))
    if family.power == 2:
        y = y * y
    elif family.power == 3:
        y = y * y * y
    return y / (y.sum(dim=-1, keepdim=True) + EPS)


def softplus_norm_causal_eager(
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
):
    family = _resolve_family(alpha, theta, power, family)
    mask = _causal_mask(x.shape[-1], x.device)
    y = _apply_family_activation_eager(family.alpha * (x - family.theta))
    if family.power == 2:
        y = y * y
    elif family.power == 3:
        y = y * y * y
    y = y.masked_fill(mask, 0.0)
    return y / (y.sum(dim=-1, keepdim=True) + EPS)


def softmax_causal_eager(x):
    mask = _causal_mask(x.shape[-1], x.device)
    return F.softmax(x.masked_fill(mask, float("-inf")), dim=-1)


class _SoftplusNormCausalAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, theta, power):
        family = FamilyConfig(
            alpha=float(alpha),
            theta=float(theta),
            power=int(power),
        )
        x_attn = _prepare_attention_input(x)
        out = torch.empty_like(x_attn)
        result = softplus_norm_causal_triton_out(out, x_attn, family=family)

        ctx.family = family
        ctx.save_for_backward(x_attn)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None

        (saved_x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = _softplus_norm_causal_backward(saved_x, grad_output.float(), ctx.family)

        return grad_x, None, None, None


def softplus_norm_causal(
    x,
    alpha=DEFAULT_FAMILY.alpha,
    theta=DEFAULT_FAMILY.theta,
    power=DEFAULT_FAMILY.power,
    *,
    family=None,
    implementation="triton",
):
    family = _resolve_family(alpha, theta, power, family)
    if implementation == "triton":
        return _SoftplusNormCausalAutogradFn.apply(
            x,
            family.alpha,
            family.theta,
            family.power,
        )
    if implementation == "eager":
        return softplus_norm_causal_eager(x.float(), family=family).to(x.dtype)
    raise ValueError("implementation must be 'triton' or 'eager'")


class SoftplusNormCausal(torch.nn.Module):
    def __init__(
        self,
        alpha=DEFAULT_FAMILY.alpha,
        theta=DEFAULT_FAMILY.theta,
        power=DEFAULT_FAMILY.power,
        *,
        family=None,
        implementation="triton",
    ):
        super().__init__()
        self.family = _resolve_family(alpha, theta, power, family)
        self.implementation = implementation

    def forward(self, x):
        return softplus_norm_causal(
            x,
            family=self.family,
            implementation=self.implementation,
        )


# ---------------------------------------------------------------------------
# Flash-style fused QK^T -> sp2norm -> V kernel
# ---------------------------------------------------------------------------
# Key insight: sp2norm has no exp() so no overflow risk and no running-max
# tracking. We can simply accumulate unnormalized output and divide at the end.


@triton.jit
def _sp2norm_flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len, scale,
    n_kv_groups,  # q_heads_per_kv_head (1 = MHA, >1 = GQA/MQA)
    window_size,  # 0 = full causal, >0 = sliding window
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    # GQA: map query head to KV head
    kv_bh_idx = bh_idx // n_kv_groups

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, D_HEAD)
    kv_offs_base = tl.arange(0, BLOCK_KV)

    q_base = Q_ptr + bh_idx * stride_qh
    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh

    q_ptrs = q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = q_offs[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    o_acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)
    s_acc = tl.full([BLOCK_Q], 1e-12, dtype=tl.float32)

    max_kv_block = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
    end_kv = tl.minimum(max_kv_block, num_kv_blocks)

    # Sliding window: start from max(0, (q_start - window_size) // BLOCK_KV)
    if window_size > 0:
        window_start_pos = tl.maximum(0, q_start - window_size + 1)
        start_kv = window_start_pos // BLOCK_KV
    else:
        start_kv = 0

    for kv_block_idx in tl.range(start_kv, end_kv):
        kv_start = kv_block_idx * BLOCK_KV
        kv_offs = kv_start + kv_offs_base

        k_ptrs = k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k_mask = kv_offs[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale

        # Causal mask: k <= q, plus sliding window: k > q - window_size
        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        if window_size > 0:
            window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
            causal_mask = causal_mask & window_mask
        valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

        arg = scores + scores - 1.0
        s = arg * arg + 4.0
        tmp = arg + _fast_sqrt(s)
        y = 0.25 * tmp * tmp
        y = tl.where(valid_mask, y, 0.0)

        s_acc += tl.sum(y, axis=1)

        v_ptrs = v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        o_acc += tl.dot(y.to(v.dtype), v).to(tl.float32)

    o = o_acc / s_acc[:, None]

    o_base = O_ptr + bh_idx * stride_oh
    o_ptrs = o_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    o_mask = q_offs[:, None] < seq_len
    tl.store(o_ptrs, o.to(q.dtype), mask=o_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
    ],
    key=["seq_len", "D_HEAD"],
)
@triton.jit
def _sp2norm_flash_fwd_with_rowsum_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, RowSum_ptr,
    seq_len, scale,
    n_kv_groups, window_size,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_rsz, stride_rsh, stride_rss,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    kv_bh_idx = bh_idx // n_kv_groups

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, D_HEAD)
    kv_offs_base = tl.arange(0, BLOCK_KV)

    q_base = Q_ptr + bh_idx * stride_qh
    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh

    q_ptrs = q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = q_offs[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    o_acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)
    s_acc = tl.full([BLOCK_Q], 1e-12, dtype=tl.float32)

    max_kv_block = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
    end_kv = tl.minimum(max_kv_block, num_kv_blocks)

    if window_size > 0:
        window_start_pos = tl.maximum(0, q_start - window_size + 1)
        start_kv = window_start_pos // BLOCK_KV
    else:
        start_kv = 0

    for kv_block_idx in tl.range(start_kv, end_kv):
        kv_start = kv_block_idx * BLOCK_KV
        kv_offs = kv_start + kv_offs_base

        k_ptrs = k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k_mask = kv_offs[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        if window_size > 0:
            window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
            causal_mask = causal_mask & window_mask
        valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

        arg = scores + scores - 1.0
        s = arg * arg + 4.0
        tmp = arg + _fast_sqrt(s)
        y = 0.25 * tmp * tmp
        y = tl.where(valid_mask, y, 0.0)

        s_acc += tl.sum(y, axis=1)

        v_ptrs = v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        o_acc += tl.dot(y.to(v.dtype), v).to(tl.float32)

    o = o_acc / s_acc[:, None]

    o_base = O_ptr + bh_idx * stride_oh
    o_ptrs = o_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    o_mask = q_offs[:, None] < seq_len
    tl.store(o_ptrs, o.to(q.dtype), mask=o_mask)

    rs_base = RowSum_ptr + bh_idx * stride_rsh
    tl.store(rs_base + q_offs * stride_rss, s_acc, mask=q_offs < seq_len)


@triton.jit
def _sp2norm_flash_bwd_precompute_delta(
    O_ptr, DO_ptr, Delta_ptr,
    seq_len,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_doz, stride_doh, stride_dos, stride_dod,
    stride_dz, stride_dh, stride_ds,
    BLOCK_Q: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """Precompute delta = rowwise sum(do * o). Grid: (q_blocks, B * H_q)."""
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, D_HEAD)
    q_mask = q_offs[:, None] < seq_len

    o_base = O_ptr + bh_idx * stride_oh
    do_base = DO_ptr + bh_idx * stride_doh

    o = tl.load(o_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
                mask=q_mask, other=0.0).to(tl.float32)
    do = tl.load(do_base + q_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod,
                 mask=q_mask, other=0.0).to(tl.float32)

    delta = tl.sum(do * o, axis=1)

    d_base = Delta_ptr + bh_idx * stride_dh
    tl.store(d_base + q_offs * stride_ds, delta, mask=q_offs < seq_len)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=8, num_stages=1),
    ],
    key=["seq_len", "D_HEAD"],
)
@triton.jit
def _sp2norm_flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, DO_ptr, DQ_ptr, RowSum_ptr, Delta_ptr,
    seq_len, scale,
    n_kv_groups, window_size,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_doz, stride_doh, stride_dos, stride_dod,
    stride_dqz, stride_dqh, stride_dqs, stride_dqd,
    stride_rsz, stride_rsh, stride_rss,
    stride_dtz, stride_dth, stride_dts,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """Backward pass: compute dQ + precompute delta. Grid: (q_blocks, B * H_q).
    Also writes delta for dKV kernel to consume."""
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    kv_bh_idx = bh_idx // n_kv_groups

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, D_HEAD)
    kv_offs_base = tl.arange(0, BLOCK_KV)

    q_base = Q_ptr + bh_idx * stride_qh
    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh
    do_base = DO_ptr + bh_idx * stride_doh

    q_mask = q_offs[:, None] < seq_len

    q = tl.load(q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
                mask=q_mask, other=0.0)
    do = tl.load(do_base + q_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod,
                 mask=q_mask, other=0.0).to(tl.float32)

    # Compute delta inline (fused, no separate kernel)
    o_base = O_ptr + bh_idx * stride_oh
    o = tl.load(o_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
                mask=q_mask, other=0.0).to(tl.float32)
    delta = tl.sum(do * o, axis=1)

    # Write delta out for the dKV kernel
    dt_base = Delta_ptr + bh_idx * stride_dth
    tl.store(dt_base + q_offs * stride_dts, delta, mask=q_offs < seq_len)

    rs_base = RowSum_ptr + bh_idx * stride_rsh
    row_sum = tl.load(rs_base + q_offs * stride_rss, mask=q_offs < seq_len, other=1.0)

    dq_acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

    max_kv_block = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
    end_kv = tl.minimum(max_kv_block, num_kv_blocks)

    if window_size > 0:
        window_start_pos = tl.maximum(0, q_start - window_size + 1)
        start_kv = window_start_pos // BLOCK_KV
    else:
        start_kv = 0

    for kv_block_idx in tl.range(start_kv, end_kv):
        kv_start = kv_block_idx * BLOCK_KV
        kv_offs = kv_start + kv_offs_base
        kv_mask = kv_offs[:, None] < seq_len

        k = tl.load(k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                     mask=kv_mask, other=0.0)
        v = tl.load(v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                     mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        if window_size > 0:
            window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
            causal_mask = causal_mask & window_mask
        valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

        arg = scores + scores - 1.0
        s = arg * arg + 4.0
        sqrt_s = _fast_sqrt(s)
        sp = 0.5 * (arg + sqrt_s)
        y = sp * sp
        y = tl.where(valid_mask, y, 0.0)

        dp = tl.dot(do.to(q.dtype), tl.trans(v).to(q.dtype)).to(tl.float32)
        dy = (dp - delta[:, None]) / row_sum[:, None]
        dy = tl.where(valid_mask, dy, 0.0)

        sp_prime = 0.5 * (1.0 + arg * tl.rsqrt(s))
        ds = dy * 4.0 * sp * sp_prime * scale
        ds = tl.where(valid_mask, ds, 0.0)

        dq_acc += tl.dot(ds.to(k.dtype), k).to(tl.float32)

    dq_base = DQ_ptr + bh_idx * stride_dqh
    dq_ptrs = dq_base + q_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(q.dtype), mask=q_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 32}, num_warps=8, num_stages=1),
    ],
    key=["seq_len", "D_HEAD"],
)
@triton.jit
def _sp2norm_flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, DK_ptr, DV_ptr, RowSum_ptr, Delta_ptr,
    seq_len, scale,
    n_kv_groups, window_size, n_q_heads,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_doz, stride_doh, stride_dos, stride_dod,
    stride_dkz, stride_dkh, stride_dks, stride_dkd,
    stride_dvz, stride_dvh, stride_dvs, stride_dvd,
    stride_rsz, stride_rsh, stride_rss,
    stride_dtz, stride_dth, stride_dts,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """Backward pass: compute dK, dV. Grid: (kv_blocks, B * H_kv).
    Accumulates gradients from ALL query heads in the GQA group."""
    kv_block_idx = tl.program_id(0)
    kv_bh_idx = tl.program_id(1)

    kv_start = kv_block_idx * BLOCK_KV
    kv_offs = kv_start + tl.arange(0, BLOCK_KV)
    d_offs = tl.arange(0, D_HEAD)
    q_offs_base = tl.arange(0, BLOCK_Q)

    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh

    kv_mask = kv_offs[:, None] < seq_len

    k = tl.load(k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                mask=kv_mask, other=0.0)
    v = tl.load(v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                mask=kv_mask, other=0.0)

    dk_acc = tl.zeros([BLOCK_KV, D_HEAD], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, D_HEAD], dtype=tl.float32)

    first_q_block = kv_start // BLOCK_Q
    num_q_blocks = tl.cdiv(seq_len, BLOCK_Q)

    if window_size > 0:
        last_q_pos = kv_start + BLOCK_KV - 1 + window_size
        last_q_block = tl.minimum(tl.cdiv(last_q_pos + 1, BLOCK_Q), num_q_blocks)
    else:
        last_q_block = num_q_blocks

    q_bh_start = kv_bh_idx * n_kv_groups

    for group_offset in tl.range(0, n_kv_groups):
        q_bh_idx = q_bh_start + group_offset

        q_base = Q_ptr + q_bh_idx * stride_qh
        do_base = DO_ptr + q_bh_idx * stride_doh
        rs_base = RowSum_ptr + q_bh_idx * stride_rsh
        dt_base = Delta_ptr + q_bh_idx * stride_dth

        for q_block_idx in tl.range(first_q_block, last_q_block):
            q_start = q_block_idx * BLOCK_Q
            q_offs = q_start + q_offs_base
            q_mask = q_offs[:, None] < seq_len

            q = tl.load(q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
                         mask=q_mask, other=0.0)
            do = tl.load(do_base + q_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod,
                          mask=q_mask, other=0.0).to(tl.float32)
            row_sum = tl.load(rs_base + q_offs * stride_rss, mask=q_offs < seq_len, other=1.0)
            delta = tl.load(dt_base + q_offs * stride_dts, mask=q_offs < seq_len, other=0.0)

            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
            causal_mask = kv_offs[None, :] <= q_offs[:, None]
            if window_size > 0:
                window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
                causal_mask = causal_mask & window_mask
            valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

            arg = scores + scores - 1.0
            s = arg * arg + 4.0
            sqrt_s = _fast_sqrt(s)
            sp = 0.5 * (arg + sqrt_s)
            y = sp * sp
            y = tl.where(valid_mask, y, 0.0)
            p = y / row_sum[:, None]

            dp = tl.dot(do.to(q.dtype), tl.trans(v).to(q.dtype)).to(tl.float32)
            dy = (dp - delta[:, None]) / row_sum[:, None]
            dy = tl.where(valid_mask, dy, 0.0)

            sp_prime = 0.5 * (1.0 + arg * tl.rsqrt(s))
            ds = dy * 4.0 * sp * sp_prime * scale
            ds = tl.where(valid_mask, ds, 0.0)

            ds_cast = ds.to(q.dtype)
            dk_acc += tl.dot(tl.trans(ds_cast), q).to(tl.float32)
            p_cast = p.to(q.dtype)
            do_cast = do.to(q.dtype)
            dv_acc += tl.dot(tl.trans(p_cast), do_cast).to(tl.float32)

    dk_base = DK_ptr + kv_bh_idx * stride_dkh
    dv_base = DV_ptr + kv_bh_idx * stride_dvh
    tl.store(dk_base + kv_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd,
             dk_acc.to(k.dtype), mask=kv_mask)
    tl.store(dv_base + kv_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd,
             dv_acc.to(v.dtype), mask=kv_mask)


def _sp2norm_flash_fwd(q, k, v, scale, n_kv_groups=1, window_size=0):
    """Run flash forward and return (output, row_sums) for backward."""
    B, H_q, T, D = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    row_sums = torch.empty(B, H_q, T, device=q.device, dtype=torch.float32)

    D_HEAD = triton.next_power_of_2(D)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_Q"]), B * H_q)

    _sp2norm_flash_fwd_with_rowsum_kernel[grid](
        q, k, v, o, row_sums,
        T, scale, n_kv_groups, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
        D_HEAD=D_HEAD,
    )
    return o, row_sums


def _sp2norm_flash_bwd(q, k, v, o, do, row_sums, scale, n_kv_groups=1, window_size=0):
    """Run flash backward: compute dQ, dK, dV. Supports GQA and sliding window."""
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    D_HEAD = triton.next_power_of_2(D)

    # Delta buffer: computed inline by dQ kernel, consumed by dKV kernel
    delta = torch.empty(B, H_q, T, device=q.device, dtype=torch.float32)

    # dQ kernel (autotuned) — also computes delta inline
    grid_q = lambda meta: (triton.cdiv(T, meta["BLOCK_Q"]), B * H_q)
    _sp2norm_flash_bwd_dq_kernel[grid_q](
        q, k, v, o, do, dq, row_sums, delta,
        T, scale, n_kv_groups, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        D_HEAD=D_HEAD,
    )

    # dK/dV kernel (autotuned) — reads precomputed delta
    grid_kv = lambda meta: (triton.cdiv(T, meta["BLOCK_KV"]), B * H_kv)
    _sp2norm_flash_bwd_dkv_kernel[grid_kv](
        q, k, v, do, dk, dv, row_sums, delta,
        T, scale, n_kv_groups, window_size, H_q,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        D_HEAD=D_HEAD,
    )
    return dq, dk, dv


class _SP2NormFlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, n_kv_groups, window_size):
        o, row_sums = _sp2norm_flash_fwd(q, k, v, scale, n_kv_groups, window_size)
        ctx.save_for_backward(q, k, v, o, row_sums)
        ctx.scale = scale
        ctx.n_kv_groups = n_kv_groups
        ctx.window_size = window_size
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, row_sums = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dq, dk, dv = _sp2norm_flash_bwd(
            q, k, v, o, grad_output, row_sums,
            ctx.scale, ctx.n_kv_groups, ctx.window_size,
        )
        return dq, dk, dv, None, None, None


def sp2norm_flash_attention(q, k, v, *, scale=None, window_size=0):
    """Fused QK^T -> sp2norm -> V without materializing the T x T score matrix.

    Supports autograd backward, GQA/MQA, and sliding window attention.

    Args:
        q: [B, H_q, T, D] query tensor
        k: [B, H_kv, T, D] key tensor (H_kv <= H_q, H_q must be divisible by H_kv)
        v: [B, H_kv, T, D] value tensor
        scale: score scaling factor (default: 1/sqrt(D))
        window_size: sliding window size (0 = full causal, >0 = attend to last W tokens)

    Returns:
        output: [B, H_q, T, D] attention output
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    assert k.shape == (B, H_kv, T, D) and v.shape == (B, H_kv, T, D)
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_kv_groups = H_q // H_kv
    return _SP2NormFlashAttentionFn.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), scale, n_kv_groups, window_size,
    )


def sp2norm_flash_attention_eager(q, k, v, *, scale=None, window_size=0):
    """Eager reference for sp2norm flash attention (supports GQA + sliding window)."""
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Expand KV heads for GQA
    if H_kv < H_q:
        n_groups = H_q // H_kv
        k = k.unsqueeze(2).expand(B, H_kv, n_groups, T, D).reshape(B, H_q, T, D)
        v = v.unsqueeze(2).expand(B, H_kv, n_groups, T, D).reshape(B, H_q, T, D)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    # Compute activation
    arg = 2.0 * scores - 1.0
    sp = 0.5 * (arg + torch.sqrt(arg * arg + 4.0))
    y = sp * sp

    # Causal mask
    causal = _causal_mask(T, q.device)
    y = y.masked_fill(causal, 0.0)

    # Sliding window: zero out positions where distance > window_size
    if window_size > 0:
        positions = torch.arange(T, device=q.device)
        window_mask = (positions[:, None] - positions[None, :]) >= window_size
        y = y.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), 0.0)

    attn = y / (y.sum(dim=-1, keepdim=True) + EPS)
    return torch.matmul(attn, v.float()).to(q.dtype)


# ---------------------------------------------------------------------------
# Fused RoPE + Flash Attention — apply rotary embeddings in-register.
# Eliminates a full Q/K global memory round-trip.
# ---------------------------------------------------------------------------


def precompute_rope_cos_sin(dim, seq_len, base=10000.0, device="cuda", dtype=torch.float32):
    """Precompute cos/sin tables for RoPE. Shape: [T, D/2] each.

    Usage: call once at model init, reuse across all forward passes.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, D/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


@triton.jit
def _sp2norm_rope_flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, RowSum_ptr,
    Cos_ptr, Sin_ptr,
    seq_len, scale,
    n_kv_groups, window_size,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_rsz, stride_rsh, stride_rss,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_HALF: tl.constexpr,
):
    """Flash attention forward with fused RoPE. No separate RoPE kernel needed.

    Q/K are loaded un-rotated, RoPE is applied in-register using the identity:
    Q_rot · K_rot^T = dot(q_a_rot, k_a_rot^T) + dot(q_b_rot, k_b_rot^T)
    where a = first half, b = second half of head dimension.
    """
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    kv_bh_idx = bh_idx // n_kv_groups

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_first = tl.arange(0, D_HALF)          # [0, ..., D/2-1]
    d_second = d_first + D_HALF              # [D/2, ..., D-1]
    d_full = tl.arange(0, D_HEAD)
    kv_offs_base = tl.arange(0, BLOCK_KV)

    q_base = Q_ptr + bh_idx * stride_qh
    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh

    q_mask_1d = q_offs < seq_len
    q_mask = q_offs[:, None] < seq_len

    # Load Q as two halves (un-rotated)
    q_a = tl.load(q_base + q_offs[:, None] * stride_qs + d_first[None, :] * stride_qd,
                  mask=q_mask, other=0.0).to(tl.float32)
    q_b = tl.load(q_base + q_offs[:, None] * stride_qs + d_second[None, :] * stride_qd,
                  mask=q_mask, other=0.0).to(tl.float32)

    # Load cos/sin for Q positions — shape [BLOCK_Q, D/2]
    cos_q = tl.load(Cos_ptr + q_offs[:, None] * D_HALF + d_first[None, :],
                    mask=q_mask, other=1.0).to(tl.float32)
    sin_q = tl.load(Sin_ptr + q_offs[:, None] * D_HALF + d_first[None, :],
                    mask=q_mask, other=0.0).to(tl.float32)

    # Apply RoPE to Q in-register
    q_ra = (q_a * cos_q - q_b * sin_q).to(tl.bfloat16)
    q_rb = (q_b * cos_q + q_a * sin_q).to(tl.bfloat16)

    o_acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)
    s_acc = tl.full([BLOCK_Q], 1e-12, dtype=tl.float32)

    max_kv_block = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
    end_kv = tl.minimum(max_kv_block, num_kv_blocks)

    if window_size > 0:
        window_start_pos = tl.maximum(0, q_start - window_size + 1)
        start_kv = window_start_pos // BLOCK_KV
    else:
        start_kv = 0

    for kv_block_idx in tl.range(start_kv, end_kv):
        kv_start = kv_block_idx * BLOCK_KV
        kv_offs = kv_start + kv_offs_base
        kv_mask = kv_offs[:, None] < seq_len

        # Load K as two halves (un-rotated)
        k_a = tl.load(k_base + kv_offs[:, None] * stride_ks + d_first[None, :] * stride_kd,
                      mask=kv_mask, other=0.0).to(tl.float32)
        k_b = tl.load(k_base + kv_offs[:, None] * stride_ks + d_second[None, :] * stride_kd,
                      mask=kv_mask, other=0.0).to(tl.float32)

        # Load cos/sin for K positions
        cos_k = tl.load(Cos_ptr + kv_offs[:, None] * D_HALF + d_first[None, :],
                        mask=kv_mask, other=1.0).to(tl.float32)
        sin_k = tl.load(Sin_ptr + kv_offs[:, None] * D_HALF + d_first[None, :],
                        mask=kv_mask, other=0.0).to(tl.float32)

        # Apply RoPE to K in-register
        k_ra = (k_a * cos_k - k_b * sin_k).to(tl.bfloat16)
        k_rb = (k_b * cos_k + k_a * sin_k).to(tl.bfloat16)

        # QK^T via split-dimension dot: Q_rot · K_rot^T = qa·ka^T + qb·kb^T
        scores = (tl.dot(q_ra, tl.trans(k_ra)) + tl.dot(q_rb, tl.trans(k_rb))).to(tl.float32) * scale

        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        if window_size > 0:
            window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
            causal_mask = causal_mask & window_mask
        valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

        # sp2norm activation in FP32
        arg = scores + scores - 1.0
        s = arg * arg + 4.0
        tmp = arg + _fast_sqrt(s)
        y = 0.25 * tmp * tmp
        y = tl.where(valid_mask, y, 0.0)

        s_acc += tl.sum(y, axis=1)

        # V does NOT get RoPE — load full D_HEAD
        v_ptrs = v_base + kv_offs[:, None] * stride_vs + d_full[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        o_acc += tl.dot(y.to(v.dtype), v).to(tl.float32)

    o = o_acc / s_acc[:, None]

    o_base = O_ptr + bh_idx * stride_oh
    o_ptrs = o_base + q_offs[:, None] * stride_os + d_full[None, :] * stride_od
    tl.store(o_ptrs, o.to(tl.bfloat16), mask=q_mask)

    rs_base = RowSum_ptr + bh_idx * stride_rsh
    tl.store(rs_base + q_offs * stride_rss, s_acc, mask=q_mask_1d)


# ---------------------------------------------------------------------------
# Fused Attention + Output Projection — skip writing [B,H,T,D], write
# [B,T,D_model] directly by multiplying by W_o inside the kernel.
# ---------------------------------------------------------------------------


@triton.jit
def _sp2norm_rope_flash_proj_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Wo_ptr, Out_ptr, RowSum_ptr,
    Cos_ptr, Sin_ptr,
    seq_len, scale,
    n_kv_groups, window_size,
    n_heads, d_model,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_wh, stride_wd, stride_wm,
    stride_oz, stride_os, stride_om,
    stride_rsz, stride_rsh, stride_rss,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_HALF: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Flash attention + RoPE + output projection.

    Instead of writing [B,H,T,D] attention output, multiplies by W_o[h]
    and atomically accumulates into [B,T,D_model] directly.
    W_o shape: [H, D, D_model] (pre-reshaped from [H*D, D_model]).
    """
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    kv_bh_idx = bh_idx // n_kv_groups

    # Extract batch and head indices
    batch_idx = bh_idx // n_heads
    head_idx = bh_idx % n_heads

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_first = tl.arange(0, D_HALF)
    d_second = d_first + D_HALF
    d_full = tl.arange(0, D_HEAD)
    kv_offs_base = tl.arange(0, BLOCK_KV)

    q_base = Q_ptr + bh_idx * stride_qh
    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh

    q_mask_1d = q_offs < seq_len
    q_mask = q_offs[:, None] < seq_len

    # Load Q halves + RoPE
    q_a = tl.load(q_base + q_offs[:, None] * stride_qs + d_first[None, :] * stride_qd,
                  mask=q_mask, other=0.0).to(tl.float32)
    q_b = tl.load(q_base + q_offs[:, None] * stride_qs + d_second[None, :] * stride_qd,
                  mask=q_mask, other=0.0).to(tl.float32)
    cos_q = tl.load(Cos_ptr + q_offs[:, None] * D_HALF + d_first[None, :],
                    mask=q_mask, other=1.0).to(tl.float32)
    sin_q = tl.load(Sin_ptr + q_offs[:, None] * D_HALF + d_first[None, :],
                    mask=q_mask, other=0.0).to(tl.float32)
    q_ra = (q_a * cos_q - q_b * sin_q).to(tl.bfloat16)
    q_rb = (q_b * cos_q + q_a * sin_q).to(tl.bfloat16)

    o_acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)
    s_acc = tl.full([BLOCK_Q], 1e-12, dtype=tl.float32)

    max_kv_block = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
    end_kv = tl.minimum(max_kv_block, num_kv_blocks)

    if window_size > 0:
        window_start_pos = tl.maximum(0, q_start - window_size + 1)
        start_kv = window_start_pos // BLOCK_KV
    else:
        start_kv = 0

    for kv_block_idx in tl.range(start_kv, end_kv):
        kv_start = kv_block_idx * BLOCK_KV
        kv_offs = kv_start + kv_offs_base
        kv_mask = kv_offs[:, None] < seq_len

        k_a = tl.load(k_base + kv_offs[:, None] * stride_ks + d_first[None, :] * stride_kd,
                      mask=kv_mask, other=0.0).to(tl.float32)
        k_b = tl.load(k_base + kv_offs[:, None] * stride_ks + d_second[None, :] * stride_kd,
                      mask=kv_mask, other=0.0).to(tl.float32)
        cos_k = tl.load(Cos_ptr + kv_offs[:, None] * D_HALF + d_first[None, :],
                        mask=kv_mask, other=1.0).to(tl.float32)
        sin_k = tl.load(Sin_ptr + kv_offs[:, None] * D_HALF + d_first[None, :],
                        mask=kv_mask, other=0.0).to(tl.float32)
        k_ra = (k_a * cos_k - k_b * sin_k).to(tl.bfloat16)
        k_rb = (k_b * cos_k + k_a * sin_k).to(tl.bfloat16)

        scores = (tl.dot(q_ra, tl.trans(k_ra)) + tl.dot(q_rb, tl.trans(k_rb))).to(tl.float32) * scale

        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        if window_size > 0:
            window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
            causal_mask = causal_mask & window_mask
        valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

        arg = scores + scores - 1.0
        s = arg * arg + 4.0
        tmp = arg + _fast_sqrt(s)
        y = 0.25 * tmp * tmp
        y = tl.where(valid_mask, y, 0.0)
        s_acc += tl.sum(y, axis=1)

        v_ptrs = v_base + kv_offs[:, None] * stride_vs + d_full[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        o_acc += tl.dot(y.to(v.dtype), v).to(tl.float32)

    # Normalize: [BLOCK_Q, D_HEAD]
    o_head = o_acc / s_acc[:, None]

    # Save row_sums for potential backward use
    rs_base = RowSum_ptr + bh_idx * stride_rsh
    tl.store(rs_base + q_offs * stride_rss, s_acc, mask=q_mask_1d)

    # Output projection: o_head [BLOCK_Q, D] @ W_o[head] [D, D_model] → [BLOCK_Q, D_model]
    # Then atomically accumulate into Out[batch, q_pos, :d_model]
    # W_o layout: [H, D, D_model]
    w_base = Wo_ptr + head_idx * stride_wh
    dm_offs = tl.arange(0, BLOCK_DMODEL)

    # Tile over D_model in blocks
    o_head_bf16 = o_head.to(tl.bfloat16)
    for dm_start in tl.range(0, d_model, BLOCK_DMODEL):
        dm = dm_start + dm_offs
        dm_mask = dm[None, :] < d_model

        # Load W_o slice: [D_HEAD, BLOCK_DMODEL]
        w = tl.load(w_base + d_full[:, None] * stride_wd + dm[None, :] * stride_wm,
                    mask=dm_mask, other=0.0)

        # Projected output: [BLOCK_Q, BLOCK_DMODEL]
        proj = tl.dot(o_head_bf16, w).to(tl.float32)

        # Atomic accumulate into Out[batch, q_pos, dm]
        out_base = Out_ptr + batch_idx * stride_oz
        out_ptrs = out_base + q_offs[:, None] * stride_os + dm[None, :] * stride_om
        out_mask = q_mask & dm_mask
        tl.atomic_add(out_ptrs, proj.to(tl.bfloat16), mask=out_mask)


def sp2norm_rope_flash_attention(q, k, v, cos, sin, *, scale=None, window_size=0):
    """Fused RoPE + sp2norm flash attention — no separate RoPE kernel needed.

    Q and K are loaded un-rotated. RoPE is applied in-register, saving a full
    global memory round-trip of Q and K.

    Args:
        q: [B, H_q, T, D] query tensor (un-rotated BF16)
        k: [B, H_kv, T, D] key tensor (un-rotated BF16)
        v: [B, H_kv, T, D] value tensor (BF16)
        cos: [T, D/2] cosine table from precompute_rope_cos_sin
        sin: [T, D/2] sine table from precompute_rope_cos_sin
        scale: score scaling factor (default: 1/sqrt(D))
        window_size: sliding window size (0 = full causal)

    Returns:
        output: [B, H_q, T, D] attention output (BF16)
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    assert H_q % H_kv == 0
    assert D % 2 == 0
    assert cos.shape == (T, D // 2) and sin.shape == (T, D // 2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_kv_groups = H_q // H_kv
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()

    o = torch.empty_like(q)
    row_sums = torch.empty(B, H_q, T, device=q.device, dtype=torch.float32)

    D_HEAD = triton.next_power_of_2(D)
    D_HALF = D_HEAD // 2
    # Reduce block sizes for large D to avoid shared memory overflow
    max_block = 64 if D_HEAD <= 64 else 32
    BLOCK_Q = min(max_block, triton.next_power_of_2(T))
    BLOCK_KV = min(max_block, triton.next_power_of_2(T))
    grid = (triton.cdiv(T, BLOCK_Q), B * H_q)

    _sp2norm_rope_flash_fwd_kernel[grid](
        q, k, v, o, row_sums, cos, sin,
        T, scale, n_kv_groups, window_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D_HEAD=D_HEAD, D_HALF=D_HALF,
    )
    return o


def sp2norm_rope_flash_attention_proj(q, k, v, w_o, cos, sin, *, scale=None, window_size=0):
    """Fused RoPE + sp2norm flash attention + output projection.

    Combines three operations into one kernel launch:
    1. RoPE on Q/K (in-register)
    2. sp2norm flash attention
    3. Output projection by W_o — writes [B, T, D_model] directly

    Args:
        q: [B, H_q, T, D] query tensor (un-rotated BF16)
        k: [B, H_kv, T, D] key tensor (un-rotated BF16)
        v: [B, H_kv, T, D] value tensor (BF16)
        w_o: [H_q, D, D_model] output projection weight (pre-reshaped)
        cos: [T, D/2] cosine table
        sin: [T, D/2] sine table
        scale: score scaling factor (default: 1/sqrt(D))
        window_size: sliding window size (0 = full causal)

    Returns:
        output: [B, T, D_model] projected output (BF16)
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    assert H_q % H_kv == 0
    assert D % 2 == 0
    assert w_o.shape[0] == H_q and w_o.shape[1] == D
    D_model = w_o.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_kv_groups = H_q // H_kv
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    w_o = w_o.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()

    # Output: [B, T, D_model] — zero-init for atomic accumulation across heads
    out = torch.zeros(B, T, D_model, device=q.device, dtype=torch.bfloat16)
    row_sums = torch.empty(B, H_q, T, device=q.device, dtype=torch.float32)

    D_HEAD = triton.next_power_of_2(D)
    D_HALF = D_HEAD // 2
    max_block = 64 if D_HEAD <= 64 else 32
    BLOCK_Q = min(max_block, triton.next_power_of_2(T))
    BLOCK_KV = min(max_block, triton.next_power_of_2(T))
    BLOCK_DMODEL = min(64, triton.next_power_of_2(D_model))
    grid = (triton.cdiv(T, BLOCK_Q), B * H_q)

    _sp2norm_rope_flash_proj_fwd_kernel[grid](
        q, k, v, w_o, out, row_sums, cos, sin,
        T, scale, n_kv_groups, window_size, H_q, D_model,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        w_o.stride(0), w_o.stride(1), w_o.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D_HEAD=D_HEAD,
        D_HALF=D_HALF, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out


# ---------------------------------------------------------------------------
# FP8 Flash Attention — only possible with sp2norm (polynomial, bounded).
# Softmax's exp() would overflow in FP8 E4M3 range [-448, 448].
# ---------------------------------------------------------------------------

@triton.jit
def _sp2norm_fp8_flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, RowSum_ptr,
    q_scale, k_scale, v_scale,
    seq_len, attn_scale,
    n_kv_groups, window_size,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_rsz, stride_rsh, stride_rss,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """FP8 flash attention forward. Q/K/V stored as float8_e4m3fn.
    Score computation: (Q_fp8 @ K_fp8^T) * q_scale * k_scale * attn_scale.
    Activation (sp2norm) computed in FP32. PV matmul uses bf16 cast of activation."""
    q_block_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    kv_bh_idx = bh_idx // n_kv_groups

    q_start = q_block_idx * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, D_HEAD)
    kv_offs_base = tl.arange(0, BLOCK_KV)

    q_base = Q_ptr + bh_idx * stride_qh
    k_base = K_ptr + kv_bh_idx * stride_kh
    v_base = V_ptr + kv_bh_idx * stride_vh

    q_ptrs = q_base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = q_offs[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # FP8

    o_acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)
    s_acc = tl.full([BLOCK_Q], 1e-12, dtype=tl.float32)

    # Combined scale: dequant Q * dequant K * attention scale
    score_scale = q_scale * k_scale * attn_scale

    max_kv_block = tl.cdiv(q_start + BLOCK_Q, BLOCK_KV)
    num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
    end_kv = tl.minimum(max_kv_block, num_kv_blocks)

    if window_size > 0:
        window_start_pos = tl.maximum(0, q_start - window_size + 1)
        start_kv = window_start_pos // BLOCK_KV
    else:
        start_kv = 0

    for kv_block_idx in tl.range(start_kv, end_kv):
        kv_start = kv_block_idx * BLOCK_KV
        kv_offs = kv_start + kv_offs_base

        k_ptrs = k_base + kv_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k_mask = kv_offs[:, None] < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)  # FP8

        # FP8 dot: Q_fp8 @ K_fp8^T → FP32, then apply combined scale
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * score_scale

        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        if window_size > 0:
            window_mask = kv_offs[None, :] > (q_offs[:, None] - window_size)
            causal_mask = causal_mask & window_mask
        valid_mask = causal_mask & (kv_offs[None, :] < seq_len) & (q_offs[:, None] < seq_len)

        # sp2norm activation — FULLY in FP32 (no overflow risk, polynomial)
        arg = scores + scores - 1.0
        s = arg * arg + 4.0
        tmp = arg + _fast_sqrt(s)
        y = 0.25 * tmp * tmp
        y = tl.where(valid_mask, y, 0.0)

        s_acc += tl.sum(y, axis=1)

        # PV matmul: load V as FP8, cast activation to bf16 for dot
        v_ptrs = v_base + kv_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)  # FP8
        # Cast both to bf16 for the PV dot (FP8 y values may exceed E4M3 range)
        o_acc += tl.dot(y.to(tl.bfloat16), v.to(tl.bfloat16)).to(tl.float32) * v_scale

    o = o_acc / s_acc[:, None]

    o_base = O_ptr + bh_idx * stride_oh
    o_ptrs = o_base + q_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    o_mask = q_offs[:, None] < seq_len
    tl.store(o_ptrs, o.to(tl.bfloat16), mask=o_mask)

    rs_base = RowSum_ptr + bh_idx * stride_rsh
    tl.store(rs_base + q_offs * stride_rss, s_acc, mask=q_offs < seq_len)


def _quantize_to_fp8(x):
    """Quantize BF16/FP32 tensor to FP8 E4M3 with per-tensor absmax scaling.

    Returns (x_fp8, scale) where x ≈ x_fp8 * scale.
    """
    FP8_MAX = 448.0  # E4M3 max
    amax = x.detach().float().abs().amax()
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    x_scaled = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    return x_scaled.to(torch.float8_e4m3fn), scale.item()


def _sp2norm_fp8_flash_fwd(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale,
                            attn_scale, n_kv_groups=1, window_size=0):
    """Run FP8 flash forward. Inputs are pre-quantized FP8 E4M3 tensors."""
    B, H_q, T, D = q_fp8.shape
    o = torch.empty(B, H_q, T, D, device=q_fp8.device, dtype=torch.bfloat16)
    row_sums = torch.empty(B, H_q, T, device=q_fp8.device, dtype=torch.float32)

    D_HEAD = triton.next_power_of_2(D)
    BLOCK_Q = min(64, triton.next_power_of_2(T))
    BLOCK_KV = min(64, triton.next_power_of_2(T))
    grid = (triton.cdiv(T, BLOCK_Q), B * H_q)

    _sp2norm_fp8_flash_fwd_kernel[grid](
        q_fp8, k_fp8, v_fp8, o, row_sums,
        q_scale, k_scale, v_scale,
        T, attn_scale, n_kv_groups, window_size,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2), q_fp8.stride(3),
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2), k_fp8.stride(3),
        v_fp8.stride(0), v_fp8.stride(1), v_fp8.stride(2), v_fp8.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        row_sums.stride(0), row_sums.stride(1), row_sums.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D_HEAD=D_HEAD,
    )
    return o, row_sums


def sp2norm_fp8_flash_attention(q, k, v, *, scale=None, window_size=0):
    """FP8 fused flash attention — only possible with sp2norm.

    Quantizes BF16 Q/K/V to FP8 E4M3 with per-tensor scaling, runs
    flash attention with FP8 tensor core matmuls for QK^T, and returns
    BF16 output. The sp2norm activation is computed in full FP32.

    Why this works with sp2norm but NOT softmax:
    - sp2norm is polynomial: squareplus(2x-1)^2. Bounded for bounded input.
    - softmax uses exp(x), which for x > 4.5 exceeds E4M3 max (448) → overflow.
    - sp2norm at x=4.5: squareplus(8)^2 ≈ 65. Well within FP8 range.

    Args:
        q: [B, H_q, T, D] query tensor (BF16)
        k: [B, H_kv, T, D] key tensor (BF16)
        v: [B, H_kv, T, D] value tensor (BF16)
        scale: score scaling factor (default: 1/sqrt(D))
        window_size: sliding window size (0 = full causal)

    Returns:
        output: [B, H_q, T, D] attention output (BF16)
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H_q, T, D = q.shape
    H_kv = k.shape[1]
    assert k.shape == (B, H_kv, T, D) and v.shape == (B, H_kv, T, D)
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_kv_groups = H_q // H_kv

    # Quantize to FP8 E4M3 with per-tensor scaling
    q_fp8, q_scale = _quantize_to_fp8(q)
    k_fp8, k_scale = _quantize_to_fp8(k)
    v_fp8, v_scale = _quantize_to_fp8(v)

    o, _row_sums = _sp2norm_fp8_flash_fwd(
        q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous(),
        q_scale, k_scale, v_scale,
        scale, n_kv_groups, window_size,
    )
    return o


# ---------------------------------------------------------------------------
# Paged KV-cache Attention — vLLM-style inference with non-contiguous pages.
# Decode phase: single query token per sequence attending to all cached KV.
# ---------------------------------------------------------------------------

@triton.jit
def _sp2norm_paged_attention_kernel(
    Q_ptr,          # [batch, H_q, 1, D]
    K_cache_ptr,    # [num_blocks, H_kv, block_size, D]
    V_cache_ptr,    # [num_blocks, H_kv, block_size, D]
    O_ptr,          # [batch, H_q, D]
    BlockTable_ptr, # [batch, max_blocks_per_seq] — logical-to-physical
    CtxLens_ptr,    # [batch] — actual KV length per sequence
    scale,
    n_kv_groups,
    block_size,     # tokens per KV page
    max_blocks_per_seq,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_btb, stride_bts,
    D_HEAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # must match block_size
):
    """Paged sp2norm attention for decode (single query token).

    Each program handles one (batch, head) pair. Iterates over all KV cache
    pages for this sequence using the block table for indirection.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // n_kv_groups

    # Load context length for this sequence
    ctx_len = tl.load(CtxLens_ptr + batch_idx)

    # Load query: [1, D] -> [D]
    d_offs = tl.arange(0, D_HEAD)
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    q = tl.load(q_base + 0 * stride_qs + d_offs * stride_qd,
                mask=d_offs < D_HEAD, other=0.0).to(tl.float32)  # [D]

    # Accumulators
    o_acc = tl.zeros([D_HEAD], dtype=tl.float32)
    s_acc = tl.full([1], 1e-12, dtype=tl.float32)

    # Number of blocks this sequence uses
    num_blocks = (ctx_len + block_size - 1) // block_size

    # Iterate over KV cache pages
    s_offs = tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]

    for block_idx in range(0, max_blocks_per_seq):
        # Skip blocks beyond this sequence's actual length
        if block_idx < num_blocks:
            # Look up physical block from block table
            phys_block = tl.load(BlockTable_ptr + batch_idx * stride_btb + block_idx * stride_bts)

            # Token positions within this block
            token_start = block_idx * block_size
            token_offs = token_start + s_offs  # [BLOCK_SIZE]
            token_mask = token_offs < ctx_len  # [BLOCK_SIZE]

            # Load K from paged cache: [BLOCK_SIZE, D]
            k_base = K_cache_ptr + phys_block * stride_kb + kv_head_idx * stride_kh
            k_ptrs = k_base + s_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=token_mask[:, None], other=0.0)  # [BLOCK_SIZE, D]

            # Score: q @ k^T -> [BLOCK_SIZE]
            scores = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * scale  # [BLOCK_SIZE]

            # sp2norm activation: squareplus(2*scores - 1)^2
            x = 2.0 * scores - 1.0
            sp = 0.5 * (x + tl.sqrt(x * x + 4.0))  # softplus / squareplus
            y = sp * sp  # ^2

            # Mask out padding tokens
            y = tl.where(token_mask, y, 0.0)

            # Accumulate row sum
            s_acc += tl.sum(y, axis=0)[None]

            # Load V from paged cache: [BLOCK_SIZE, D]
            v_base = V_cache_ptr + phys_block * stride_vb + kv_head_idx * stride_vh
            v_ptrs = v_base + s_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=token_mask[:, None], other=0.0)  # [BLOCK_SIZE, D]

            # Weighted V accumulation
            o_acc += tl.sum(y[:, None] * v.to(tl.float32), axis=0)  # [D]

    # Normalize
    o = o_acc / s_acc

    # Store output: [D]
    o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = o_base + d_offs * stride_od
    tl.store(o_ptrs, o.to(tl.bfloat16), mask=d_offs < D_HEAD)


def sp2norm_paged_attention(q, k_cache, v_cache, block_table, context_lens,
                            *, scale=None, block_size=16):
    """Paged KV-cache sp2norm attention for autoregressive decode.

    This kernel supports vLLM-style non-contiguous KV cache pages, enabling
    efficient memory management for inference serving with dynamic batching.

    Args:
        q: [batch, H_q, 1, D] — single query token per sequence
        k_cache: [num_blocks, H_kv, block_size, D] — paged key cache
        v_cache: [num_blocks, H_kv, block_size, D] — paged value cache
        block_table: [batch, max_blocks_per_seq] — logical-to-physical block mapping (int32)
        context_lens: [batch] — actual KV length per sequence (int32)
        scale: score scaling (default: 1/sqrt(D))
        block_size: tokens per cache page (default: 16, must match k_cache/v_cache dim 2)

    Returns:
        output: [batch, H_q, D] — attention output (BF16)
    """
    assert q.ndim == 4 and q.shape[2] == 1, "Decode: q must be [B, H, 1, D]"
    B, H_q, _, D = q.shape
    H_kv = k_cache.shape[1]
    assert H_q % H_kv == 0
    assert k_cache.shape[2] == block_size and v_cache.shape[2] == block_size
    max_blocks_per_seq = block_table.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    n_kv_groups = H_q // H_kv
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.contiguous()

    out = torch.empty(B, H_q, D, device=q.device, dtype=torch.bfloat16)

    D_HEAD = triton.next_power_of_2(D)
    BLOCK_SIZE = triton.next_power_of_2(block_size)

    grid = (B, H_q)

    _sp2norm_paged_attention_kernel[grid](
        q, k_cache, v_cache, out, block_table, context_lens,
        scale, n_kv_groups, block_size, max_blocks_per_seq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_table.stride(0), block_table.stride(1),
        D_HEAD=D_HEAD, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def bench(fn, x=None, warmup=300, iters=1000):
    callback = (lambda: fn(x)) if x is not None else fn
    return _median_cuda_us(callback, warmup=warmup, iters=iters)


def bench_out(fn, out, x, warmup=300, iters=1000):
    return _median_cuda_us(lambda: fn(out, x), warmup=warmup, iters=iters)


def benchmark_rowwise_suite(family, shapes=None, warmup=50, iters=200):
    shapes = tuple(shapes or DEFAULT_ROWWISE_SHAPES)
    results = []
    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="cuda")
        out_sm = torch.empty_like(x)
        out_fam = torch.empty_like(x)

        torch_us = bench(lambda z: F.softmax(z, dim=-1), x, warmup=warmup, iters=iters)
        triton_sm_us = bench_out(softmax_triton_out, out_sm, x, warmup=warmup, iters=iters)
        family_us = bench_out(
            lambda out, z: softplus_norm_triton_out(out, z, family=family),
            out_fam,
            x,
            warmup=warmup,
            iters=iters,
        )
        results.append(
            RowwiseBenchmarkResult(
                shape=(rows, cols),
                torch_softmax_us=torch_us,
                triton_softmax_us=triton_sm_us,
                family_us=family_us,
            )
        )
    return results


def benchmark_causal_suite(family, dtypes, lengths, batch=2, heads=8, warmup=50, iters=200):
    results = []
    for dtype in dtypes:
        for seq_len in lengths:
            x = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=dtype)
            x_ref = x.float()
            ref_sm = softmax_causal_eager(x_ref)
            ref_fam = softplus_norm_causal_eager(x_ref, family=family)

            out_sm = torch.empty_like(x)
            out_fam = torch.empty_like(x)

            our_sm, sm_meta = softmax_causal_triton_out(out_sm, x, return_meta=True)
            our_fam, fam_meta = softplus_norm_causal_triton_out(
                out_fam,
                x,
                family=family,
                return_meta=True,
            )

            atol = 1e-4 if dtype == torch.float32 else 5e-3
            torch_sm_us = bench(softmax_causal_eager, x, warmup=warmup, iters=iters)
            triton_sm_us = bench_out(softmax_causal_triton_out, out_sm, x, warmup=warmup, iters=iters)
            family_us = bench_out(
                lambda out, z: softplus_norm_causal_triton_out(out, z, family=family),
                out_fam,
                x,
                warmup=warmup,
                iters=iters,
            )

            results.append(
                CausalBenchmarkResult(
                    dtype=dtype,
                    batch=batch,
                    heads=heads,
                    seq_len=seq_len,
                    torch_softmax_us=torch_sm_us,
                    triton_softmax_us=triton_sm_us,
                    family_us=family_us,
                    softmax_err=(ref_sm - our_sm.float()).abs().max().item(),
                    family_err=(ref_fam - our_fam.float()).abs().max().item(),
                    softmax_sums_ok=torch.allclose(
                        our_sm.float().sum(-1),
                        torch.ones_like(ref_sm.sum(-1)),
                        atol=atol,
                    ),
                    family_sums_ok=torch.allclose(
                        our_fam.float().sum(-1),
                        torch.ones_like(ref_fam.sum(-1)),
                        atol=atol,
                    ),
                    softmax_meta=sm_meta,
                    family_meta=fam_meta,
                )
            )
    return results


def validate_causal_backward(family, dtypes, seq_len=128, batch=2, heads=4):
    results = []
    for dtype in dtypes:
        x = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=dtype, requires_grad=True)
        upstream = torch.randn(batch, heads, seq_len, seq_len, device="cuda", dtype=torch.float32)

        y_triton = softplus_norm_causal(x, family=family, implementation="triton")
        loss_triton = (y_triton.float() * upstream).sum()
        loss_triton.backward()
        grad_triton = x.grad.detach().clone()

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = softplus_norm_causal_eager(x_ref.float(), family=family)
        loss_ref = (y_ref * upstream).sum()
        loss_ref.backward()
        grad_ref = x_ref.grad.detach()

        results.append(
            {
                "dtype": dtype,
                "max_abs_grad_err": (grad_triton.float() - grad_ref.float()).abs().max().item(),
                "grad_close": torch.allclose(
                    grad_triton.float(),
                    grad_ref.float(),
                    atol=1e-4 if dtype == torch.float32 else 5e-3,
                    rtol=1e-3 if dtype == torch.float32 else 5e-2,
                ),
            }
        )
    return results


def _geo_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _print_header(title):
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_launch_meta(label, meta):
    parts = [
        f"  {label}:",
        f"BLOCK={meta['block_size']}",
        f"warps={meta['num_warps']}",
        f"stages={meta['num_stages']}",
        f"progs={meta['num_programs']}",
    ]
    if "variant" in meta:
        parts.append(f"variant={meta['variant']}")
    print(" ".join(parts))


def print_rowwise_report(results):
    print("\n--- Legacy Rowwise Benchmark ---")
    print("  Triton columns use reusable output buffers.")
    print(
        f"  {'Shape':>18s} {'F.softmax':>10s} {'sm_triton':>10s}"
        f" {'fam_triton':>11s} {'fam/F.sm':>9s} {'fam/sm_tr':>10s}"
    )
    print("  " + "-" * 74)

    fam_vs_torch = []
    fam_vs_sm = []
    for row in results:
        fam_vs_torch.append(row.family_us / row.torch_softmax_us)
        fam_vs_sm.append(row.family_us / row.triton_softmax_us)
        print(
            f"  {str(row.shape):>18s} {row.torch_softmax_us:8.1f}us"
            f" {row.triton_softmax_us:8.1f}us {row.family_us:9.1f}us"
            f" {row.family_us / row.torch_softmax_us:7.2f}x"
            f" {row.family_us / row.triton_softmax_us:8.2f}x"
        )

    print(f"\n  Geo mean fam_triton / F.softmax:      {_geo_mean(fam_vs_torch):.3f}x")
    print(f"  Geo mean fam_triton / softmax_triton: {_geo_mean(fam_vs_sm):.3f}x")


def print_causal_report(results, family):
    print("\n--- Production Causal Benchmark ---")
    print("  Triton columns use reusable output buffers and fp32 accumulation.")
    print(f"  Family: {family.label}")
    print(
        f"  {'dtype':>6s} {'B,H,T':>12s} {'torch.sm':>10s} {'sm_tr':>10s}"
        f" {'fam_tr':>10s} {'fam/torch':>10s} {'fam/sm':>8s}"
        f" {'sm_err':>10s} {'fam_err':>10s}"
    )
    print("  " + "-" * 100)

    grouped = {}
    for row in results:
        label = _dtype_label(row.dtype)
        grouped.setdefault(label, {"fam_vs_torch": [], "fam_vs_sm": []})
        grouped[label]["fam_vs_torch"].append(row.family_us / row.torch_softmax_us)
        grouped[label]["fam_vs_sm"].append(row.family_us / row.triton_softmax_us)

        shape_label = f"{row.batch}x{row.heads}x{row.seq_len}"
        print(
            f"  {label:>6s} {shape_label:>12s} {row.torch_softmax_us:8.1f}us"
            f" {row.triton_softmax_us:8.1f}us {row.family_us:8.1f}us"
            f" {row.family_us / row.torch_softmax_us:8.2f}x"
            f" {row.family_us / row.triton_softmax_us:7.2f}x"
            f" {row.softmax_err:10.2e} {row.family_err:10.2e}"
        )

    print("\n  Correctness:")
    for row in results:
        print(
            f"  {_dtype_label(row.dtype):>6s} T={row.seq_len:<4d}"
            f" softmax_sum={row.softmax_sums_ok}"
            f" family_sum={row.family_sums_ok}"
        )

    print("\n  Geo means by dtype:")
    for label, values in grouped.items():
        print(
            f"  {label:>6s}"
            f" fam/torch={_geo_mean(values['fam_vs_torch']):.3f}x"
            f" fam/sm_tr={_geo_mean(values['fam_vs_sm']):.3f}x"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Triton squareplus benchmark harness")
    parser.add_argument("--mode", choices=("causal", "rowwise", "all"), default="causal")
    parser.add_argument("--dtypes", default="bf16", help="comma-separated: fp32,fp16,bf16")
    parser.add_argument("--lengths", default="128,256,512,1024", help="comma-separated causal sequence lengths")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=DEFAULT_FAMILY.alpha)
    parser.add_argument("--theta", type=float, default=DEFAULT_FAMILY.theta)
    parser.add_argument("--power", type=int, default=DEFAULT_FAMILY.power)
    parser.add_argument("--quick", action="store_true", help="smaller sweep for iteration speed")
    parser.add_argument("--check-backward", action="store_true", help="validate custom autograd backward")
    return parser.parse_args()


def main():
    args = parse_args()
    family = FamilyConfig(
        alpha=args.alpha,
        theta=args.theta,
        power=args.power,
    )
    dtypes = _parse_dtype_list(args.dtypes)
    lengths = _parse_int_list(args.lengths)

    if args.quick:
        lengths = lengths[: min(2, len(lengths))]
        args.warmup = min(args.warmup, 20)
        args.iters = min(args.iters, 100)

    _print_header("TRITON SQUAREPLUS BENCHMARK")
    print(f"GPU: {torch.cuda.get_device_name(_current_device_index())}")
    print(f"Family: {family.label}")

    if args.mode in ("rowwise", "all"):
        rowwise_results = benchmark_rowwise_suite(
            family=family,
            warmup=args.warmup,
            iters=args.iters,
        )
        print_rowwise_report(rowwise_results)

    if args.mode in ("causal", "all"):
        causal_results = benchmark_causal_suite(
            family=family,
            dtypes=dtypes,
            lengths=lengths,
            batch=args.batch,
            heads=args.heads,
            warmup=args.warmup,
            iters=args.iters,
        )
        if causal_results:
            print("\n--- Causal Launch Meta (first case) ---")
            _print_launch_meta("softmax", causal_results[0].softmax_meta)
            _print_launch_meta("family", causal_results[0].family_meta)
        print_causal_report(causal_results, family)

    if args.check_backward:
        backward_results = validate_causal_backward(
            family=family,
            dtypes=dtypes,
            seq_len=lengths[0],
            batch=min(args.batch, 2),
            heads=min(args.heads, 4),
        )
        print("\n--- Backward Validation ---")
        for row in backward_results:
            print(
                f"  {_dtype_label(row['dtype']):>6s}"
                f" grad_close={row['grad_close']}"
                f" max_abs_grad_err={row['max_abs_grad_err']:.2e}"
            )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
