#!/usr/bin/env python3
"""Compare atomic vs sorted centroid update."""
import torch
from flash_kmeans.assign_euclid_triton import compute_sq_norms, euclid_assign_triton, _heuristic_euclid_config
from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_euclid,
    torch_centroid_update_euclid,
)

WORKLOADS = [
    {"label": "stress",      "B": 4,  "N": 262144,"D": 128, "K": 8192},
    {"label": "large-dense", "B": 32, "N": 65536, "D": 128, "K": 4096},
    {"label": "medium-std",  "B": 32, "N": 32768, "D": 128, "K": 1000},
    {"label": "small-batch", "B": 8,  "N": 16384, "D": 128, "K": 100},
]

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    print(f"\n=== {wl['label']} (B={B}, N={N}, K={K}) ===")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    config = _heuristic_euclid_config(N, K, D, device=x.device)
    cluster_ids = euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=config, use_heuristic=False)

    # Pre-allocate
    sums = torch.zeros((B, K, D), device="cuda", dtype=torch.float32)
    cnts = torch.zeros((B, K), device="cuda", dtype=torch.int32)
    c_sq_out = torch.empty_like(c_sq)
    sort_vals = torch.empty((B, N), device="cuda", dtype=torch.int32)
    sort_idx = torch.empty((B, N), device="cuda", dtype=torch.int64)

    reps = 10

    # Warmup
    for _ in range(3):
        triton_centroid_update_euclid(x, cluster_ids, centroids, centroid_sums=sums, centroid_counts=cnts, c_sq_out=c_sq_out)
        triton_centroid_update_sorted_euclid(x, cluster_ids, centroids, BLOCK_N=128, centroid_sums=sums, centroid_cnts=cnts, c_sq_out=c_sq_out, sort_vals_buf=sort_vals, sort_idx_buf=sort_idx)
        torch_centroid_update_euclid(x, cluster_ids, centroids, centroid_sums=sums, centroid_cnts=cnts)
    torch.cuda.synchronize()

    # Atomic
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        triton_centroid_update_euclid(x, cluster_ids, centroids, centroid_sums=sums, centroid_counts=cnts, c_sq_out=c_sq_out)
    e.record(); torch.cuda.synchronize()
    atomic_ms = s.elapsed_time(e) / reps

    # Sorted
    s.record()
    for _ in range(reps):
        triton_centroid_update_sorted_euclid(x, cluster_ids, centroids, BLOCK_N=128, centroid_sums=sums, centroid_cnts=cnts, c_sq_out=c_sq_out, sort_vals_buf=sort_vals, sort_idx_buf=sort_idx)
    e.record(); torch.cuda.synchronize()
    sorted_ms = s.elapsed_time(e) / reps

    # PyTorch scatter_add
    s.record()
    for _ in range(reps):
        torch_centroid_update_euclid(x, cluster_ids, centroids, centroid_sums=sums, centroid_cnts=cnts)
    e.record(); torch.cuda.synchronize()
    torch_ms = s.elapsed_time(e) / reps

    print(f"  Atomic:    {atomic_ms:.3f} ms")
    print(f"  Sorted:    {sorted_ms:.3f} ms")
    print(f"  PyTorch:   {torch_ms:.3f} ms")

    del x, centroids
    torch.cuda.empty_cache()
