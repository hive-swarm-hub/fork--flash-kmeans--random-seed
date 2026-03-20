#!/usr/bin/env python3
"""Profile time breakdown: assign vs sort vs update per workload."""
import torch
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, compute_sq_norms, _heuristic_euclid_config
from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid

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
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty(B, N, device="cuda", dtype=torch.int32)
    centroid_sums = torch.zeros(B, K, D, device="cuda", dtype=torch.float32)
    centroid_cnts = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    sort_vals = torch.empty(B, N, device="cuda", dtype=torch.int32)
    sort_idx = torch.empty(B, N, device="cuda", dtype=torch.int64)
    centroids_out = torch.empty_like(centroids)

    cached_config = _heuristic_euclid_config(N, K, D, device=x.device)
    update_block_n = 64 if B >= 16 and K >= 2048 else 128

    # Warmup
    for _ in range(NUM_WARMUP):
        cluster_ids = euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq,
                                            config=cached_config, use_heuristic=False)
        triton_centroid_update_sorted_euclid(x, cluster_ids, centroids,
                                             BLOCK_N=update_block_n,
                                             centroid_sums=centroid_sums,
                                             centroid_cnts=centroid_cnts,
                                             c_sq_out=c_sq,
                                             sort_vals_buf=sort_vals,
                                             sort_idx_buf=sort_idx,
                                             centroids_out=centroids_out)
    torch.cuda.synchronize()

    # Time assign
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(NUM_TIMED):
        euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq,
                             config=cached_config, use_heuristic=False)
    e.record(); torch.cuda.synchronize()
    assign_ms = s.elapsed_time(e) / NUM_TIMED

    # Time sort alone
    cluster_ids = out
    s.record()
    for _ in range(NUM_TIMED):
        torch.sort(cluster_ids, dim=-1, stable=False, out=(sort_vals, sort_idx))
    e.record(); torch.cuda.synchronize()
    sort_ms = s.elapsed_time(e) / NUM_TIMED

    # Time full centroid update (sort + kernel + finalize)
    s.record()
    for _ in range(NUM_TIMED):
        triton_centroid_update_sorted_euclid(x, cluster_ids, centroids,
                                             BLOCK_N=update_block_n,
                                             centroid_sums=centroid_sums,
                                             centroid_cnts=centroid_cnts,
                                             c_sq_out=c_sq,
                                             sort_vals_buf=sort_vals,
                                             sort_idx_buf=sort_idx,
                                             centroids_out=centroids_out)
    e.record(); torch.cuda.synchronize()
    update_ms = s.elapsed_time(e) / NUM_TIMED

    total = assign_ms + update_ms
    print(f"  Assign:  {assign_ms:.3f} ms ({assign_ms/total*100:.1f}%)")
    print(f"  Sort:    {sort_ms:.3f} ms ({sort_ms/total*100:.1f}%) [included in update]")
    print(f"  Update:  {update_ms:.3f} ms ({update_ms/total*100:.1f}%)")
    print(f"  Total:   {total:.3f} ms (x10 iters: {total*10:.1f} ms)")

    # Also time c_sq computation
    s.record()
    for _ in range(NUM_TIMED):
        compute_sq_norms(centroids, out=c_sq)
    e.record(); torch.cuda.synchronize()
    csq_ms = s.elapsed_time(e) / NUM_TIMED
    print(f"  c_sq:    {csq_ms:.3f} ms")

    del x, centroids, x_sq, c_sq, out, centroid_sums, centroid_cnts
    torch.cuda.empty_cache()
