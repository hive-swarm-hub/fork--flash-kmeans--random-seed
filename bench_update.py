#!/usr/bin/env python3
"""Benchmark different centroid update strategies per workload."""
import torch
from flash_kmeans.assign_euclid_triton import compute_sq_norms
from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_euclid,
    torch_centroid_update_euclid,
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
NUM_TIMED = 20

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    label = wl["label"]
    print(f"\n{'='*70}")
    print(f"Workload: {label}  (B={B}, N={N}, D={D}, K={K})")
    print(f"{'='*70}")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    cluster_ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    centroid_sums = torch.zeros(B, K, D, device="cuda", dtype=torch.float32)
    centroid_cnts = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    c_sq = torch.empty(B, K, device="cuda", dtype=torch.float16)
    centroids_out = torch.empty_like(centroids)
    sort_vals = torch.empty(B, N, device="cuda", dtype=torch.int32)
    sort_idx = torch.empty(B, N, device="cuda", dtype=torch.int64)

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for block_n in [64, 128, 256]:
        # Sorted update
        for _ in range(NUM_WARMUP):
            triton_centroid_update_sorted_euclid(x, cluster_ids, centroids,
                                                  BLOCK_N=block_n,
                                                  centroid_sums=centroid_sums,
                                                  centroid_cnts=centroid_cnts,
                                                  c_sq_out=c_sq,
                                                  sort_vals_buf=sort_vals,
                                                  sort_idx_buf=sort_idx,
                                                  centroids_out=centroids_out)
        torch.cuda.synchronize()
        s.record()
        for _ in range(NUM_TIMED):
            triton_centroid_update_sorted_euclid(x, cluster_ids, centroids,
                                                  BLOCK_N=block_n,
                                                  centroid_sums=centroid_sums,
                                                  centroid_cnts=centroid_cnts,
                                                  c_sq_out=c_sq,
                                                  sort_vals_buf=sort_vals,
                                                  sort_idx_buf=sort_idx,
                                                  centroids_out=centroids_out)
        e.record(); torch.cuda.synchronize()
        sorted_ms = s.elapsed_time(e) / NUM_TIMED
        print(f"  Sorted BN={block_n}: {sorted_ms:.3f} ms")

    # Atomic update
    for _ in range(NUM_WARMUP):
        triton_centroid_update_euclid(x, cluster_ids, centroids,
                                      centroid_sums=centroid_sums,
                                      centroid_counts=centroid_cnts,
                                      c_sq_out=c_sq,
                                      centroids_out=centroids_out)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        triton_centroid_update_euclid(x, cluster_ids, centroids,
                                      centroid_sums=centroid_sums,
                                      centroid_counts=centroid_cnts,
                                      c_sq_out=c_sq,
                                      centroids_out=centroids_out)
    e.record(); torch.cuda.synchronize()
    atomic_ms = s.elapsed_time(e) / NUM_TIMED
    print(f"  Atomic:       {atomic_ms:.3f} ms")

    # PyTorch scatter_add
    for _ in range(NUM_WARMUP):
        torch_centroid_update_euclid(x, cluster_ids, centroids,
                                     centroid_sums=centroid_sums,
                                     centroid_cnts=centroid_cnts)
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        torch_centroid_update_euclid(x, cluster_ids, centroids,
                                     centroid_sums=centroid_sums,
                                     centroid_cnts=centroid_cnts)
    e.record(); torch.cuda.synchronize()
    scatter_ms = s.elapsed_time(e) / NUM_TIMED
    print(f"  Scatter_add:  {scatter_ms:.3f} ms")

    del x, centroids, cluster_ids, centroid_sums, centroid_cnts, c_sq, centroids_out
    torch.cuda.empty_cache()
