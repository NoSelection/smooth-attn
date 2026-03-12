"""Long-context stress test for paged KV-cache sp2norm attention.
Tests 8K, 16K, 24K, 32K context lengths — the regime paged attention was built for."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import sp2norm_paged_attention


def _build_paged_kv_cache_shuffled(k_contiguous, v_contiguous, block_size=16):
    """Convert contiguous KV into paged format with shuffled physical blocks."""
    B, H_kv, T, D = k_contiguous.shape
    num_blocks_per_seq = (T + block_size - 1) // block_size
    total_blocks = B * num_blocks_per_seq
    device = k_contiguous.device

    k_cache = torch.zeros(total_blocks, H_kv, block_size, D, device=device, dtype=k_contiguous.dtype)
    v_cache = torch.zeros(total_blocks, H_kv, block_size, D, device=device, dtype=v_contiguous.dtype)
    block_table = torch.zeros(B, num_blocks_per_seq, device=device, dtype=torch.int32)
    context_lens = torch.full((B,), T, device=device, dtype=torch.int32)

    perm = torch.randperm(total_blocks, device=device)

    for b in range(B):
        for blk in range(num_blocks_per_seq):
            logical_idx = b * num_blocks_per_seq + blk
            phys_idx = perm[logical_idx].item()
            block_table[b, blk] = phys_idx
            start = blk * block_size
            end = min(start + block_size, T)
            length = end - start
            k_cache[phys_idx, :, :length, :] = k_contiguous[b, :, start:end, :]
            v_cache[phys_idx, :, :length, :] = v_contiguous[b, :, start:end, :]

    return k_cache, v_cache, block_table, context_lens


def reference_decode_sp2norm(q, k, v, scale):
    """Reference decode attention in pure PyTorch (batched, no loops over heads)."""
    B, H_q, _, D = q.shape
    H_kv = k.shape[1]
    n_groups = H_q // H_kv

    # Expand KV heads to match Q heads
    if n_groups > 1:
        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)

    # q: [B, H, 1, D], k: [B, H, T, D]
    # scores: [B, H, T]
    scores = torch.bmm(
        q.view(B * H_q, 1, D).float(),
        k.view(B * H_q, -1, D).float().transpose(1, 2)
    ).squeeze(1) * scale  # [B*H, T]

    x = 2.0 * scores - 1.0
    sp = 0.5 * (x + torch.sqrt(x * x + 4.0))
    y = sp * sp
    row_sum = y.sum(dim=-1, keepdim=True) + 1e-12
    attn = y / row_sum  # [B*H, T]

    T = k.shape[2]
    out = torch.bmm(attn.unsqueeze(1), v.view(B * H_q, T, D).float()).squeeze(1)  # [B*H, D]
    return out.view(B, H_q, D).to(torch.bfloat16)


def bench_us(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return times[len(times) // 2]


def test_long_context_correctness():
    """Verify paged attention is correct at long context lengths."""
    print("=" * 70)
    print("LONG-CONTEXT PAGED ATTENTION — CORRECTNESS")
    print("=" * 70)

    # Test at moderate lengths first (full reference comparison)
    configs = [
        (1, 8, 2, 1024, 64, 16, "1K warmup"),
        (1, 8, 2, 2048, 64, 16, "2K"),
        (1, 8, 2, 4096, 64, 16, "4K"),
        (1, 8, 2, 8192, 64, 16, "8K"),
    ]

    for B, H_q, H_kv, T, D, BS, label in configs:
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)

        k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache_shuffled(k, v, block_size=BS)

        out_paged = sp2norm_paged_attention(q, k_cache, v_cache, block_table, ctx_lens,
                                            block_size=BS)
        out_ref = reference_decode_sp2norm(q, k, v, scale)

        max_err = (out_paged.float() - out_ref.float()).abs().max().item()
        mean_err = (out_paged.float() - out_ref.float()).abs().mean().item()
        has_nan = out_paged.isnan().any().item()
        print(f"  {label:>12s} T={T:>5d}: max_err={max_err:.6f} mean_err={mean_err:.8f} nan={has_nan}")
        assert max_err < 0.1, f"Error too high at T={T}: {max_err}"
        assert not has_nan

    print("  PASSED\n")


def test_very_long_context_no_crash():
    """Verify kernel doesn't crash at 16K/24K/32K (sanity + NaN check only)."""
    print("=" * 70)
    print("VERY LONG CONTEXT — STABILITY TEST (no NaN, no crash)")
    print("=" * 70)

    configs = [
        (1, 32, 8, 16384, 128, 16, "16K"),
        (1, 32, 8, 24576, 128, 16, "24K"),
        (1, 32, 8, 32768, 128, 16, "32K"),
    ]

    for B, H_q, H_kv, T, D, BS, label in configs:
        torch.cuda.empty_cache()
        # KV cache memory: 2 * num_blocks * H_kv * BS * D * 2 bytes
        num_blocks = (T + BS - 1) // BS
        mem_mb = 2 * num_blocks * H_kv * BS * D * 2 / 1024**2
        print(f"  {label}: T={T}, KV cache ~{mem_mb:.0f} MB ...", end=" ", flush=True)

        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)

        k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache_shuffled(k, v, block_size=BS)

        # Free contiguous KV to save memory
        del k, v
        torch.cuda.empty_cache()

        out = sp2norm_paged_attention(q, k_cache, v_cache, block_table, ctx_lens,
                                       block_size=BS)

        has_nan = out.isnan().any().item()
        has_inf = out.isinf().any().item()
        out_norm = out.float().norm().item()
        print(f"nan={has_nan} inf={has_inf} norm={out_norm:.2f}")
        assert not has_nan, f"NaN at T={T}!"
        assert not has_inf, f"Inf at T={T}!"

        del k_cache, v_cache, block_table, ctx_lens, out, q
        torch.cuda.empty_cache()

    print("  PASSED\n")


def benchmark_long_context():
    """Benchmark paged attention across context lengths."""
    print("=" * 70)
    print("PAGED ATTENTION BENCHMARK — LONG CONTEXT")
    print("=" * 70)

    # Header
    print(f"\n{'config':>40s} {'paged_us':>10s} {'tokens/s':>12s} {'GB/s':>8s}")
    print("-" * 75)

    configs = [
        # Single sequence, production-like (Llama-style: H=32, D=128)
        (1, 32, 8, 256, 128, 16),
        (1, 32, 8, 512, 128, 16),
        (1, 32, 8, 1024, 128, 16),
        (1, 32, 8, 2048, 128, 16),
        (1, 32, 8, 4096, 128, 16),
        (1, 32, 8, 8192, 128, 16),
        (1, 32, 8, 16384, 128, 16),
        (1, 32, 8, 24576, 128, 16),
        (1, 32, 8, 32768, 128, 16),
        # Batched decode (inference serving)
        (8, 32, 8, 2048, 128, 16),
        (8, 32, 8, 4096, 128, 16),
        (8, 32, 8, 8192, 128, 16),
        (16, 32, 8, 2048, 128, 16),
        (32, 32, 8, 1024, 128, 16),
        # Smaller model (GPT-2 like: H=12, D=64)
        (1, 12, 12, 8192, 64, 16),
        (1, 12, 12, 16384, 64, 16),
        (1, 12, 12, 32768, 64, 16),
    ]

    for B, H_q, H_kv, T, D, BS in configs:
        torch.cuda.empty_cache()
        scale = 1.0 / math.sqrt(D)

        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        # Build paged cache directly to save memory
        num_blocks_per_seq = (T + BS - 1) // BS
        total_blocks = B * num_blocks_per_seq

        k_cache = torch.randn(total_blocks, H_kv, BS, D, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.randn(total_blocks, H_kv, BS, D, device="cuda", dtype=torch.bfloat16)
        block_table = torch.arange(total_blocks, device="cuda", dtype=torch.int32).view(B, num_blocks_per_seq)
        ctx_lens = torch.full((B,), T, device="cuda", dtype=torch.int32)

        def run(q=q, kc=k_cache, vc=v_cache, bt=block_table, cl=ctx_lens, bs=BS):
            return sp2norm_paged_attention(q, kc, vc, bt, cl, block_size=bs)

        try:
            us = bench_us(run, warmup=5, iters=30)
            # Throughput: how many KV tokens processed per second
            total_kv_tokens = B * T
            tokens_per_sec = total_kv_tokens / (us * 1e-6)
            # Memory bandwidth: bytes of KV loaded
            kv_bytes = B * T * H_kv * D * 2 * 2  # K + V, bf16
            gbps = kv_bytes / (us * 1e-6) / 1e9

            label = f"B={B} H={H_q}/{H_kv} T={T} D={D}"
            print(f"{label:>40s} {us:8.1f}us {tokens_per_sec:10.0f} {gbps:6.1f}")
        except Exception as e:
            label = f"B={B} H={H_q}/{H_kv} T={T} D={D}"
            print(f"{label:>40s}  FAILED: {e}")

        del k_cache, v_cache, block_table, ctx_lens, q
        torch.cuda.empty_cache()


def benchmark_block_sizes():
    """Compare different page sizes at long context."""
    print(f"\n{'='*70}")
    print("BLOCK SIZE COMPARISON at T=8192")
    print("=" * 70)
    print(f"{'block_size':>12s} {'us':>10s} {'pages':>8s}")
    print("-" * 35)

    B, H_q, H_kv, T, D = 1, 32, 8, 8192, 128

    for BS in [8, 16, 32, 64, 128]:
        torch.cuda.empty_cache()
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        num_blocks = (T + BS - 1) // BS
        total_blocks = B * num_blocks

        k_cache = torch.randn(total_blocks, H_kv, BS, D, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.randn(total_blocks, H_kv, BS, D, device="cuda", dtype=torch.bfloat16)
        block_table = torch.arange(total_blocks, device="cuda", dtype=torch.int32).view(B, num_blocks)
        ctx_lens = torch.full((B,), T, device="cuda", dtype=torch.int32)

        def run(q=q, kc=k_cache, vc=v_cache, bt=block_table, cl=ctx_lens, bs=BS):
            return sp2norm_paged_attention(q, kc, vc, bt, cl, block_size=bs)

        us = bench_us(run, warmup=5, iters=30)
        print(f"{BS:>12d} {us:8.1f}us {num_blocks:>8d}")

        del k_cache, v_cache, q
        torch.cuda.empty_cache()


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    test_long_context_correctness()
    test_very_long_context_no_crash()
    benchmark_long_context()
    benchmark_block_sizes()

    print("\n" + "=" * 70)
    print("ALL LONG-CONTEXT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
