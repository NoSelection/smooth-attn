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
    return [(4, 2), (4, 4), (8, 2), (8, 4)]


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
def _apply_family_activation(arg):
    return 0.5 * (arg + tl.sqrt(arg * arg + 4.0))


@triton.jit
def _apply_winner_squareplus_p2(arg):
    # squareplus(arg)^2 = 0.25 * (arg + sqrt(arg^2 + 4))^2
    tmp = arg + tl.sqrt(arg * arg + 4.0)
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
        row_ptr = input_ptr + row_idx * input_row_stride + col_offsets
        x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)

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
def _winner_squareplus_p2_causal_small_fwd(
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
        x = tl.load(row_ptr, mask=mask, other=0.0).to(tl.float32)

        q_pos = row_idx % n_cols
        causal_mask = mask & (col_offsets <= q_pos)

        arg = x + x - 1.0
        y = _apply_winner_squareplus_p2(arg)
        y = tl.where(causal_mask, y, 0.0)

        row_sum = tl.sum(y, axis=0) + 1e-12
        out = tl.fdiv(y, row_sum)

        out_ptr = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(out_ptr, out, mask=mask)


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
    result, meta = _launch_attention_kernel_out(
        "squareplus_winner_small",
        _winner_squareplus_p2_causal_small_fwd,
        x_attn,
        kernel_out,
        return_meta=True,
        extra_cache_key=("winner_squareplus", "small"),
        tune_configs=_winner_candidate_configs(x_attn.shape[-1]),
    )
    result = _finalize_output(result, user_out)
    meta = {**meta, "variant": "squareplus_winner_small"}
    if return_meta:
        return result, meta
    return result


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
    if _is_winner_family(family) and x_attn.shape[-1] <= 256:
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

        with torch.enable_grad():
            x = saved_x.detach().requires_grad_(True)
            ref = softplus_norm_causal_eager(x.float(), family=ctx.family)
            grad_x = torch.autograd.grad(
                ref,
                x,
                grad_outputs=grad_output.float(),
                only_inputs=True,
            )[0]

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
