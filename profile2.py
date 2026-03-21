#!/usr/bin/env python3
"""Profile time breakdown with counting sort."""
import torch
import triton
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, compute_sq_norms, _heuristic_euclid_config
from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    _histogram_kernel, _counting_scatter_kernel,
)

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "D": 128, "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

NUM_WARMUP = 5
NUM_TIMED = 30

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    label = wl["label"]
    print(f"\n{'='*60}")
    print(f"{label}: B={B}, N={N}, D={D}, K={K}")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty(B, N, device="cuda", dtype=torch.int32)
    centroid_sums = torch.zeros(B, K, D, device="cuda", dtype=torch.float32)
    centroid_cnts = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    sort_vals = torch.empty(B, N, device="cuda", dtype=torch.int16)
    sort_idx = torch.empty(B, N, device="cuda", dtype=torch.int64)
    hist_buf = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    offsets_buf = torch.empty(B, K, device="cuda", dtype=torch.int32)
    centroids_out = torch.empty_like(centroids)

    cached_config = _heuristic_euclid_config(N, K, D, device=x.device)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Time assign
    for _ in range(NUM_WARMUP):
        euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=cached_config, use_heuristic=False)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=cached_config, use_heuristic=False)
    e.record(); torch.cuda.synchronize()
    assign_ms = s.elapsed_time(e) / NUM_TIMED

    # Time histogram
    SORT_BN = 128
    sort_grid = (triton.cdiv(N, SORT_BN), B)
    for _ in range(NUM_WARMUP):
        hist_buf.zero_()
        _histogram_kernel[sort_grid](out, hist_buf, N=N, K=K, BLOCK_N=SORT_BN, num_warps=1)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        hist_buf.zero_()
        _histogram_kernel[sort_grid](out, hist_buf, N=N, K=K, BLOCK_N=SORT_BN, num_warps=1)
    e.record(); torch.cuda.synchronize()
    hist_ms = s.elapsed_time(e) / NUM_TIMED

    # Time cumsum
    for _ in range(NUM_WARMUP):
        torch.cumsum(hist_buf, dim=1, out=offsets_buf)
        offsets_buf -= hist_buf
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        torch.cumsum(hist_buf, dim=1, out=offsets_buf)
        offsets_buf -= hist_buf
    e.record(); torch.cuda.synchronize()
    cumsum_ms = s.elapsed_time(e) / NUM_TIMED

    # Time scatter
    for _ in range(NUM_WARMUP):
        torch.cumsum(hist_buf, dim=1, out=offsets_buf)
        offsets_buf -= hist_buf
        _counting_scatter_kernel[sort_grid](out, offsets_buf, sort_vals, sort_idx, N=N, K=K, BLOCK_N=SORT_BN, num_warps=1)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        torch.cumsum(hist_buf, dim=1, out=offsets_buf)
        offsets_buf -= hist_buf
        _counting_scatter_kernel[sort_grid](out, offsets_buf, sort_vals, sort_idx, N=N, K=K, BLOCK_N=SORT_BN, num_warps=1)
    e.record(); torch.cuda.synchronize()
    scatter_ms = s.elapsed_time(e) / NUM_TIMED - cumsum_ms

    # Time full update (counting sort + chunk + finalize)
    for _ in range(NUM_WARMUP):
        triton_centroid_update_sorted_euclid(x, out, centroids, BLOCK_N=32,
            centroid_sums=centroid_sums, centroid_cnts=centroid_cnts, c_sq_out=c_sq,
            sort_vals_buf=sort_vals, sort_idx_buf=sort_idx,
            hist_buf=hist_buf, offsets_buf=offsets_buf, centroids_out=centroids_out)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        triton_centroid_update_sorted_euclid(x, out, centroids, BLOCK_N=32,
            centroid_sums=centroid_sums, centroid_cnts=centroid_cnts, c_sq_out=c_sq,
            sort_vals_buf=sort_vals, sort_idx_buf=sort_idx,
            hist_buf=hist_buf, offsets_buf=offsets_buf, centroids_out=centroids_out)
    e.record(); torch.cuda.synchronize()
    update_ms = s.elapsed_time(e) / NUM_TIMED

    total = assign_ms + update_ms
    sort_total = hist_ms + cumsum_ms + scatter_ms
    print(f"  Assign:    {assign_ms:.3f} ms ({assign_ms/total*100:.1f}%)")
    print(f"  Update:    {update_ms:.3f} ms ({update_ms/total*100:.1f}%)")
    print(f"    Hist:    {hist_ms:.3f} ms")
    print(f"    Cumsum:  {cumsum_ms:.3f} ms")
    print(f"    Scatter: {scatter_ms:.3f} ms")
    print(f"    Sort total: {sort_total:.3f} ms")
    print(f"  Total:     {total:.3f} ms (x10: {total*10:.1f} ms)")

    del x, centroids, x_sq, c_sq, out, centroid_sums, centroid_cnts
    torch.cuda.empty_cache()
