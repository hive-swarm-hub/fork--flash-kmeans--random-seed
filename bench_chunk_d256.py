#!/usr/bin/env python3
"""Quick test of chunk kernel num_warps for D=256 workloads."""
import torch
import triton
from flash_kmeans.centroid_update_triton import _centroid_update_chunk_kernel

WORKLOADS = [
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "medium-std",   "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
]

NUM_WARMUP = 5
NUM_TIMED = 30

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    label = wl["label"]
    print(f"\n{label}: B={B}, N={N}, D={D}, K={K}")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    cluster_ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    sorted_ids, sorted_idx = torch.sort(cluster_ids, dim=-1, stable=False)
    centroid_sums = torch.zeros(B, K, D, device="cuda", dtype=torch.float32)
    centroid_cnts = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for BN in [64, 128]:
        for nw in [2, 4]:
            grid = (triton.cdiv(N, BN), B)
            try:
                for _ in range(NUM_WARMUP):
                    centroid_sums.zero_(); centroid_cnts.zero_()
                    _centroid_update_chunk_kernel[grid](
                        x, sorted_idx, sorted_ids, centroid_sums, centroid_cnts,
                        x.stride(0), x.stride(1), x.stride(2),
                        sorted_idx.stride(0), sorted_idx.stride(1),
                        sorted_ids.stride(0), sorted_ids.stride(1),
                        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
                        centroid_cnts.stride(0), centroid_cnts.stride(1),
                        B, N, D, K, BLOCK_N=BN, num_warps=nw)
                torch.cuda.synchronize()
                s.record()
                for _ in range(NUM_TIMED):
                    centroid_sums.zero_(); centroid_cnts.zero_()
                    _centroid_update_chunk_kernel[grid](
                        x, sorted_idx, sorted_ids, centroid_sums, centroid_cnts,
                        x.stride(0), x.stride(1), x.stride(2),
                        sorted_idx.stride(0), sorted_idx.stride(1),
                        sorted_ids.stride(0), sorted_ids.stride(1),
                        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
                        centroid_cnts.stride(0), centroid_cnts.stride(1),
                        B, N, D, K, BLOCK_N=BN, num_warps=nw)
                e.record(); torch.cuda.synchronize()
                print(f"  BN={BN:3d} w={nw}: {s.elapsed_time(e)/NUM_TIMED:.3f} ms")
            except Exception as ex:
                print(f"  BN={BN:3d} w={nw}: FAILED: {ex}")

    del x, cluster_ids, sorted_ids, sorted_idx, centroid_sums, centroid_cnts
    torch.cuda.empty_cache()
