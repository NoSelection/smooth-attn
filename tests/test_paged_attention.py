"""Test paged KV-cache sp2norm attention for vLLM-style inference."""
import math
import sys
sys.path.insert(0, "src")

import torch
from smooth_attn import sp2norm_paged_attention, sp2norm_flash_attention_eager


def _build_paged_kv_cache(k_contiguous, v_contiguous, block_size=16):
    """Convert contiguous [B, H_kv, T, D] KV into paged format.

    Returns:
        k_cache: [num_blocks, H_kv, block_size, D]
        v_cache: [num_blocks, H_kv, block_size, D]
        block_table: [B, max_blocks_per_seq] (int32)
        context_lens: [B] (int32)
    """
    B, H_kv, T, D = k_contiguous.shape
    num_blocks_per_seq = (T + block_size - 1) // block_size
    total_blocks = B * num_blocks_per_seq
    device = k_contiguous.device

    k_cache = torch.zeros(total_blocks, H_kv, block_size, D, device=device, dtype=k_contiguous.dtype)
    v_cache = torch.zeros(total_blocks, H_kv, block_size, D, device=device, dtype=v_contiguous.dtype)
    block_table = torch.zeros(B, num_blocks_per_seq, device=device, dtype=torch.int32)
    context_lens = torch.full((B,), T, device=device, dtype=torch.int32)

    # Assign physical blocks (sequential for simplicity, but non-contiguous in memory)
    for b in range(B):
        for blk in range(num_blocks_per_seq):
            phys_idx = b * num_blocks_per_seq + blk
            block_table[b, blk] = phys_idx
            start = blk * block_size
            end = min(start + block_size, T)
            length = end - start
            k_cache[phys_idx, :, :length, :] = k_contiguous[b, :, start:end, :]
            v_cache[phys_idx, :, :length, :] = v_contiguous[b, :, start:end, :]

    return k_cache, v_cache, block_table, context_lens


def _build_paged_kv_cache_shuffled(k_contiguous, v_contiguous, block_size=16):
    """Like _build_paged_kv_cache but with shuffled physical blocks.

    This tests that the kernel correctly handles non-sequential block mappings,
    which is the whole point of paged attention.
    """
    B, H_kv, T, D = k_contiguous.shape
    num_blocks_per_seq = (T + block_size - 1) // block_size
    total_blocks = B * num_blocks_per_seq
    device = k_contiguous.device

    k_cache = torch.zeros(total_blocks, H_kv, block_size, D, device=device, dtype=k_contiguous.dtype)
    v_cache = torch.zeros(total_blocks, H_kv, block_size, D, device=device, dtype=v_contiguous.dtype)
    block_table = torch.zeros(B, num_blocks_per_seq, device=device, dtype=torch.int32)
    context_lens = torch.full((B,), T, device=device, dtype=torch.int32)

    # Shuffle physical block indices
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


def reference_decode_attention(q, k, v):
    """Reference: use eager attention for last token only.

    q: [B, H, 1, D], k: [B, H_kv, T, D], v: [B, H_kv, T, D]
    Returns: [B, H, D]
    """
    # sp2norm_flash_attention_eager expects causal mask, but for decode
    # the query is at position T (last), so ALL KV positions are visible.
    # We compute manually.
    B, H_q, _, D = q.shape
    H_kv = k.shape[1]
    n_groups = H_q // H_kv
    scale = 1.0 / math.sqrt(D)

    out = torch.zeros(B, H_q, D, device=q.device, dtype=torch.float32)

    for h in range(H_q):
        kv_h = h // n_groups
        q_h = q[:, h, 0, :].float()  # [B, D]
        k_h = k[:, kv_h, :, :].float()  # [B, T, D]
        v_h = v[:, kv_h, :, :].float()  # [B, T, D]
        # scores: [B, T]
        scores = torch.bmm(q_h.unsqueeze(1), k_h.transpose(1, 2)).squeeze(1) * scale
        # sp2norm: squareplus(2*scores - 1)^2
        x = 2.0 * scores - 1.0
        sp = 0.5 * (x + torch.sqrt(x * x + 4.0))
        y = sp * sp
        # normalize
        row_sum = y.sum(dim=-1, keepdim=True) + 1e-12
        attn = y / row_sum  # [B, T]
        # weighted sum: [B, 1, T] @ [B, T, D] -> [B, 1, D] -> [B, D]
        out[:, h, :] = torch.bmm(attn.unsqueeze(1), v_h).squeeze(1)

    return out.to(torch.bfloat16)


def test_paged_basic():
    """Basic paged attention vs contiguous reference."""
    print("=== TEST: Paged Attention Basic ===")
    for B, H, T, D, BS in [(1, 4, 64, 32, 16), (2, 4, 128, 32, 16),
                             (2, 8, 256, 64, 16), (1, 4, 48, 32, 16)]:
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

        k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache(k, v, block_size=BS)

        out_paged = sp2norm_paged_attention(q, k_cache, v_cache, block_table, ctx_lens,
                                            block_size=BS)
        out_ref = reference_decode_attention(q, k, v)

        err = (out_paged.float() - out_ref.float()).abs()
        max_err = err.max().item()
        mean_err = err.mean().item()
        print(f"  B={B} H={H} T={T} D={D} BS={BS}: max_err={max_err:.4f}, mean_err={mean_err:.6f}")
        assert max_err < 0.05, f"Paged attention max error too high: {max_err}"
        assert not out_paged.isnan().any()
    print("  PASSED\n")


def test_paged_shuffled_blocks():
    """Paged attention with shuffled physical blocks (the real test)."""
    print("=== TEST: Paged Attention Shuffled Blocks ===")
    for B, H, T, D, BS in [(2, 4, 128, 32, 16), (4, 8, 256, 64, 16)]:
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

        k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache_shuffled(k, v, block_size=BS)

        out_paged = sp2norm_paged_attention(q, k_cache, v_cache, block_table, ctx_lens,
                                            block_size=BS)
        out_ref = reference_decode_attention(q, k, v)

        max_err = (out_paged.float() - out_ref.float()).abs().max().item()
        print(f"  B={B} H={H} T={T} D={D} BS={BS} (shuffled): max_err={max_err:.4f}")
        assert max_err < 0.05, f"Shuffled paged attention max error too high: {max_err}"
    print("  PASSED\n")


def test_paged_gqa():
    """Paged attention with GQA (H_q != H_kv)."""
    print("=== TEST: Paged Attention GQA ===")
    B, H_q, H_kv, T, D, BS = 2, 8, 2, 128, 32, 16
    q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)

    k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache(k, v, block_size=BS)

    out_paged = sp2norm_paged_attention(q, k_cache, v_cache, block_table, ctx_lens,
                                        block_size=BS)
    out_ref = reference_decode_attention(q, k, v)

    max_err = (out_paged.float() - out_ref.float()).abs().max().item()
    print(f"  GQA H_q={H_q} H_kv={H_kv}: max_err={max_err:.4f}")
    assert max_err < 0.05
    assert out_paged.shape == (B, H_q, D)
    print("  PASSED\n")


def test_paged_variable_context_lengths():
    """Sequences with different context lengths in the same batch."""
    print("=== TEST: Paged Attention Variable Context Lengths ===")
    B, H, D, BS = 4, 4, 32, 16
    max_T = 128  # max context length
    # Variable lengths
    true_lens = [32, 64, 48, 128]

    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)

    # Build KV cache for max length, but use variable context_lens
    num_blocks_per_seq = (max_T + BS - 1) // BS
    total_blocks = B * num_blocks_per_seq

    k_cache = torch.zeros(total_blocks, H, BS, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros(total_blocks, H, BS, D, device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros(B, num_blocks_per_seq, device="cuda", dtype=torch.int32)
    context_lens = torch.tensor(true_lens, device="cuda", dtype=torch.int32)

    # Fill KV cache with random data for each sequence's actual length
    k_full = torch.randn(B, H, max_T, D, device="cuda", dtype=torch.bfloat16)
    v_full = torch.randn(B, H, max_T, D, device="cuda", dtype=torch.bfloat16)

    for b in range(B):
        for blk in range(num_blocks_per_seq):
            phys_idx = b * num_blocks_per_seq + blk
            block_table[b, blk] = phys_idx
            start = blk * BS
            end = min(start + BS, true_lens[b])
            if start < true_lens[b]:
                length = end - start
                k_cache[phys_idx, :, :length, :] = k_full[b, :, start:end, :]
                v_cache[phys_idx, :, :length, :] = v_full[b, :, start:end, :]

    out_paged = sp2norm_paged_attention(q, k_cache, v_cache, block_table, context_lens,
                                        block_size=BS)

    # Reference: compute each sequence separately with its true length
    for b in range(B):
        T_b = true_lens[b]
        q_b = q[b:b+1]
        k_b = k_full[b:b+1, :, :T_b, :]
        v_b = v_full[b:b+1, :, :T_b, :]
        out_ref_b = reference_decode_attention(q_b, k_b, v_b)
        max_err = (out_paged[b].float() - out_ref_b[0].float()).abs().max().item()
        print(f"  Batch {b}, ctx_len={T_b}: max_err={max_err:.4f}")
        assert max_err < 0.05, f"Variable ctx_len error too high for batch {b}: {max_err}"
    print("  PASSED\n")


def test_paged_block_sizes():
    """Test different block sizes."""
    print("=== TEST: Paged Attention Block Sizes ===")
    B, H, T, D = 2, 4, 128, 32

    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    out_ref = reference_decode_attention(q, k, v)

    for BS in [8, 16, 32, 64]:
        k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache(k, v, block_size=BS)
        out_paged = sp2norm_paged_attention(q, k_cache, v_cache, block_table, ctx_lens,
                                            block_size=BS)
        max_err = (out_paged.float() - out_ref.float()).abs().max().item()
        print(f"  block_size={BS}: max_err={max_err:.4f}")
        assert max_err < 0.05, f"Block size {BS} error too high: {max_err}"
    print("  PASSED\n")


def bench_us(fn, warmup=25, iters=100):
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


def benchmark_paged():
    """Benchmark paged attention at various context lengths."""
    print("=== BENCHMARK: Paged KV-cache Attention ===")
    print(f"{'config':>35s} {'paged':>10s} {'contiguous':>10s} {'ratio':>8s}")
    print("-" * 68)

    for B, H, T, D, BS in [(1, 32, 256, 128, 16),
                             (1, 32, 512, 128, 16),
                             (1, 32, 1024, 128, 16),
                             (1, 32, 2048, 128, 16),
                             (8, 32, 512, 128, 16),
                             (8, 32, 1024, 128, 16),
                             (32, 32, 256, 128, 16)]:
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

        k_cache, v_cache, block_table, ctx_lens = _build_paged_kv_cache_shuffled(k, v, block_size=BS)

        def run_paged(q=q, kc=k_cache, vc=v_cache, bt=block_table, cl=ctx_lens, bs=BS):
            with torch.no_grad():
                return sp2norm_paged_attention(q, kc, vc, bt, cl, block_size=bs)

        def run_contiguous(q=q, k=k, v=v):
            with torch.no_grad():
                return reference_decode_attention(q, k, v)

        paged_us = bench_us(run_paged)
        contig_us = bench_us(run_contiguous)
        ratio = paged_us / contig_us
        label = f"B={B} H={H} T={T} D={D}"
        print(f"{label:>35s} {paged_us:8.1f}us {contig_us:8.1f}us {ratio:7.2f}x")


def main():
    test_paged_basic()
    test_paged_shuffled_blocks()
    test_paged_gqa()
    test_paged_variable_context_lengths()
    test_paged_block_sizes()
    benchmark_paged()


if __name__ == "__main__":
    main()
