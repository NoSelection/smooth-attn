"""Test if FP8 tl.dot works on RTX 4090."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_dot_test(A_ptr, B_ptr, C_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    a = tl.load(A_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(B_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    c = tl.dot(a, b)
    tl.store(C_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)


@triton.jit
def _bf16_dot_test(A_ptr, B_ptr, C_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    a = tl.load(A_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(B_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    c = tl.dot(a, b)
    tl.store(C_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)


def bench_us(fn, warmup=50, iters=200):
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


def main():
    M, K, N = 64, 64, 64

    # Test FP8 E4M3
    print("=== FP8 tl.dot test ===")
    a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    try:
        a_fp8 = a_bf16.to(torch.float8_e4m3fn)
        b_fp8 = b_bf16.to(torch.float8_e4m3fn)
        c_fp8 = torch.empty(M, N, device="cuda", dtype=torch.float32)
        _fp8_dot_test[(1,)](a_fp8, b_fp8, c_fp8, M=M, K=K, N=N)
        torch.cuda.synchronize()

        c_ref = a_fp8.float() @ b_fp8.float()
        err = (c_fp8 - c_ref).abs().max().item()
        print(f"FP8 E4M3 tl.dot: WORKS! Max error: {err:.6f}")
    except Exception as e:
        print(f"FP8 E4M3 tl.dot FAILED: {e}")

    # Test FP8 E5M2
    try:
        a_e5 = a_bf16.to(torch.float8_e5m2)
        b_e5 = b_bf16.to(torch.float8_e5m2)
        c_e5 = torch.empty(M, N, device="cuda", dtype=torch.float32)
        _fp8_dot_test[(1,)](a_e5, b_e5, c_e5, M=M, K=K, N=N)
        torch.cuda.synchronize()
        print(f"FP8 E5M2 tl.dot: WORKS!")
    except Exception as e:
        print(f"FP8 E5M2 tl.dot FAILED: {e}")

    # Benchmark FP8 vs BF16 dot
    print("\n=== FP8 vs BF16 tl.dot throughput ===")
    c_bf16 = torch.empty(M, N, device="cuda", dtype=torch.float32)

    def run_bf16():
        _bf16_dot_test[(1,)](a_bf16, b_bf16, c_bf16, M=M, K=K, N=N)

    def run_fp8():
        _fp8_dot_test[(1,)](a_fp8, b_fp8, c_fp8, M=M, K=K, N=N)

    try:
        bf16_us = bench_us(run_bf16)
        fp8_us = bench_us(run_fp8)
        print(f"BF16 dot: {bf16_us:.1f}us")
        print(f"FP8  dot: {fp8_us:.1f}us")
        print(f"FP8/BF16 ratio: {fp8_us/bf16_us:.2f}x")
    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
