#!/usr/bin/env python3
"""Test: K as non-constexpr + tl.range for proper loop pipelining."""
import torch
import triton
import triton.language as tl
from flash_kmeans.assign_euclid_triton import compute_sq_norms, _ceil_div

@triton.jit
def _euclid_assign_kernel_v2(
    x_ptr, c_ptr, x_sq_ptr, c_sq_ptr, out_ptr,
    B: tl.constexpr, N: tl.constexpr, K, D: tl.constexpr,
    stride_x_b: tl.constexpr, stride_x_n: tl.constexpr, stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr, stride_c_k: tl.constexpr, stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr, stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr, stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    offs_d = tl.arange(0, D).to(tl.int64)
    x_ptrs = x_ptr + pid_b * stride_x_b + n_offsets[:, None] * stride_x_n + offs_d[None, :] * stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0, eviction_policy='evict_first')

    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

    best_dist = tl.full((BLOCK_N,), float('inf'), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        c_ptrs = c_ptr + pid_b * stride_c_b + k_offsets[None, :] * stride_c_k + offs_d[:, None] * stride_c_d
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0, eviction_policy='evict_last')

        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        cross = tl.dot(x_tile, c_tile, out_dtype=tl.float32, max_num_imprecise_acc=D)
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.where(k_mask[None, :], dist, float('inf'))

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)


def bench_v2(wl, config, num_warmup=5, num_timed=20):
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    try:
        for _ in range(num_warmup):
            _euclid_assign_kernel_v2[grid](
                x, centroids, x_sq, c_sq, out,
                B, N, K, D,
                x.stride(0), x.stride(1), x.stride(2),
                centroids.stride(0), centroids.stride(1), centroids.stride(2),
                x_sq.stride(0), x_sq.stride(1),
                c_sq.stride(0), c_sq.stride(1),
                out.stride(0), out.stride(1),
                **config,
            )
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(num_timed):
            _euclid_assign_kernel_v2[grid](
                x, centroids, x_sq, c_sq, out,
                B, N, K, D,
                x.stride(0), x.stride(1), x.stride(2),
                centroids.stride(0), centroids.stride(1), centroids.stride(2),
                x_sq.stride(0), x_sq.stride(1),
                c_sq.stride(0), c_sq.stride(1),
                out.stride(0), out.stride(1),
                **config,
            )
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / num_timed
    except Exception as ex:
        print(f"  ERROR: {ex}")
        return None

WORKLOADS = [
    {"label": "stress",      "B": 4,  "N": 262144,"D": 128, "K": 8192},
    {"label": "large-dense", "B": 32, "N": 65536, "D": 128, "K": 4096},
    {"label": "medium-std",  "B": 32, "N": 32768, "D": 128, "K": 1000},
    {"label": "large-scale", "B": 8,  "N": 131072,"D": 128, "K": 1000},
]

CONFIGS = [
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 3},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 2},
]

for wl in WORKLOADS:
    print(f"\n=== {wl['label']} (B={wl['B']}, N={wl['N']}, D={wl['D']}, K={wl['K']}) ===")
    results = []
    for cfg in CONFIGS:
        ms = bench_v2(wl, cfg)
        if ms is not None:
            results.append((ms, cfg))
    results.sort(key=lambda x: x[0])
    for ms, cfg in results[:5]:
        print(f"  {ms:8.3f} ms  BN={cfg['BLOCK_N']:3d} BK={cfg['BLOCK_K']:3d} warps={cfg['num_warps']} stages={cfg['num_stages']}")
    torch.cuda.empty_cache()
