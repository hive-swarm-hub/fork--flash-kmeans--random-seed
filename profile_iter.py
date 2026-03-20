#!/usr/bin/env python3
"""Profile each step of a single k-means iteration."""
import torch
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, compute_sq_norms, _heuristic_euclid_config
from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid

WORKLOADS = [
    {"label": "large-dense", "B": 32, "N": 65536, "D": 128, "K": 4096},
    {"label": "stress",      "B": 4,  "N": 262144,"D": 128, "K": 8192},
    {"label": "medium-std",  "B": 32, "N": 32768, "D": 128, "K": 1000},
]

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    print(f"\n=== {wl['label']} (B={B}, N={N}, D={D}, K={K}) ===")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    centroid_sums = torch.zeros((B, K, D), device="cuda", dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device="cuda", dtype=torch.int32)
    sort_vals = torch.empty((B, N), device="cuda", dtype=torch.int32)
    sort_idx = torch.empty((B, N), device="cuda", dtype=torch.int64)

    config = _heuristic_euclid_config(N, K, D, device=x.device)

    # Warmup
    for _ in range(3):
        cluster_ids = euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=config, use_heuristic=False)
        triton_centroid_update_sorted_euclid(x, cluster_ids, centroids, BLOCK_N=128,
            centroid_sums=centroid_sums, centroid_cnts=centroid_cnts, c_sq_out=c_sq,
            sort_vals_buf=sort_vals, sort_idx_buf=sort_idx)
    torch.cuda.synchronize()

    # Time assignment
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    reps = 10
    s.record()
    for _ in range(reps):
        euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=config, use_heuristic=False)
    e.record(); torch.cuda.synchronize()
    assign_ms = s.elapsed_time(e) / reps

    # Time sort only
    s.record()
    for _ in range(reps):
        torch.sort(cluster_ids, dim=-1, stable=False, out=(sort_vals, sort_idx))
    e.record(); torch.cuda.synchronize()
    sort_ms = s.elapsed_time(e) / reps

    # Time centroid update (includes sort)
    s.record()
    for _ in range(reps):
        triton_centroid_update_sorted_euclid(x, cluster_ids, centroids, BLOCK_N=128,
            centroid_sums=centroid_sums, centroid_cnts=centroid_cnts, c_sq_out=c_sq,
            sort_vals_buf=sort_vals, sort_idx_buf=sort_idx)
    e.record(); torch.cuda.synchronize()
    update_ms = s.elapsed_time(e) / reps

    # Time c_sq
    s.record()
    for _ in range(reps):
        compute_sq_norms(centroids, out=c_sq)
    e.record(); torch.cuda.synchronize()
    csq_ms = s.elapsed_time(e) / reps

    total = assign_ms + update_ms
    print(f"  Assignment:  {assign_ms:.3f} ms ({assign_ms/total*100:.1f}%)")
    print(f"  Sort only:   {sort_ms:.3f} ms ({sort_ms/total*100:.1f}%)")
    print(f"  Update+sort: {update_ms:.3f} ms ({update_ms/total*100:.1f}%)")
    print(f"  c_sq:        {csq_ms:.3f} ms")
    print(f"  Total iter:  {total:.3f} ms")
    print(f"  10 iters:    {total*10:.1f} ms")

    del x, centroids, x_sq, c_sq, out
    torch.cuda.empty_cache()
