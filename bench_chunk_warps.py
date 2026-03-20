#!/usr/bin/env python3
"""Test different num_warps for the centroid update chunk kernel."""
import torch
import triton
from flash_kmeans.assign_euclid_triton import compute_sq_norms
from flash_kmeans.centroid_update_triton import (
    _centroid_update_chunk_kernel,
    _finalize_centroids_kernel,
)

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "D": 128, "K": 100},
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

NUM_WARMUP = 5
NUM_TIMED = 30

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    label = wl["label"]
    print(f"\n{label}: B={B}, N={N}, D={D}, K={K}")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    cluster_ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)

    # Sort
    sorted_ids_i16, sorted_idx = torch.sort(cluster_ids.to(torch.int16), dim=-1, stable=False)
    sorted_ids_i32 = sorted_ids_i16.to(torch.int32)

    centroid_sums = torch.zeros(B, K, D, device="cuda", dtype=torch.float32)
    centroid_cnts = torch.zeros(B, K, device="cuda", dtype=torch.int32)

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for BLOCK_N in [64, 128, 256]:
        for nw in [2, 4, 8]:
            centroid_sums.zero_()
            centroid_cnts.zero_()
            grid = (triton.cdiv(N, BLOCK_N), B)

            try:
                for _ in range(NUM_WARMUP):
                    centroid_sums.zero_()
                    centroid_cnts.zero_()
                    _centroid_update_chunk_kernel[grid](
                        x, sorted_idx, sorted_ids_i32,
                        centroid_sums, centroid_cnts,
                        x.stride(0), x.stride(1), x.stride(2),
                        sorted_idx.stride(0), sorted_idx.stride(1),
                        sorted_ids_i32.stride(0), sorted_ids_i32.stride(1),
                        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
                        centroid_cnts.stride(0), centroid_cnts.stride(1),
                        B, N, D, K, BLOCK_N=BLOCK_N, num_warps=nw)
                torch.cuda.synchronize()

                s.record()
                for _ in range(NUM_TIMED):
                    centroid_sums.zero_()
                    centroid_cnts.zero_()
                    _centroid_update_chunk_kernel[grid](
                        x, sorted_idx, sorted_ids_i32,
                        centroid_sums, centroid_cnts,
                        x.stride(0), x.stride(1), x.stride(2),
                        sorted_idx.stride(0), sorted_idx.stride(1),
                        sorted_ids_i32.stride(0), sorted_ids_i32.stride(1),
                        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
                        centroid_cnts.stride(0), centroid_cnts.stride(1),
                        B, N, D, K, BLOCK_N=BLOCK_N, num_warps=nw)
                e.record(); torch.cuda.synchronize()
                print(f"  BN={BLOCK_N:3d} w={nw}: {s.elapsed_time(e)/NUM_TIMED:.3f} ms")
            except Exception as ex:
                print(f"  BN={BLOCK_N:3d} w={nw}: FAILED: {ex}")

    del x, cluster_ids, sorted_ids_i16, sorted_ids_i32, sorted_idx
    torch.cuda.empty_cache()
