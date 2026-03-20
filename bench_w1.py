#!/usr/bin/env python3
"""Test num_warps=1 for chunk kernel."""
import torch
import triton
from flash_kmeans.centroid_update_triton import _centroid_update_chunk_kernel

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
    print(f"\n{wl['label']}: B={B}, N={N}, D={D}, K={K}")

    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    sorted_ids, sorted_idx = torch.sort(ids, dim=-1, stable=False)
    sums = torch.zeros(B, K, D, device="cuda", dtype=torch.float32)
    cnts = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for BN in [32, 64]:
        for nw in [1, 2]:
            grid = (triton.cdiv(N, BN), B)
            try:
                for _ in range(NUM_WARMUP):
                    sums.zero_(); cnts.zero_()
                    _centroid_update_chunk_kernel[grid](
                        x, sorted_idx, sorted_ids, sums, cnts,
                        x.stride(0), x.stride(1), x.stride(2),
                        sorted_idx.stride(0), sorted_idx.stride(1),
                        sorted_ids.stride(0), sorted_ids.stride(1),
                        sums.stride(0), sums.stride(1), sums.stride(2),
                        cnts.stride(0), cnts.stride(1),
                        B, N, D, K, BLOCK_N=BN, num_warps=nw)
                torch.cuda.synchronize()
                s.record()
                for _ in range(NUM_TIMED):
                    sums.zero_(); cnts.zero_()
                    _centroid_update_chunk_kernel[grid](
                        x, sorted_idx, sorted_ids, sums, cnts,
                        x.stride(0), x.stride(1), x.stride(2),
                        sorted_idx.stride(0), sorted_idx.stride(1),
                        sorted_ids.stride(0), sorted_ids.stride(1),
                        sums.stride(0), sums.stride(1), sums.stride(2),
                        cnts.stride(0), cnts.stride(1),
                        B, N, D, K, BLOCK_N=BN, num_warps=nw)
                e.record(); torch.cuda.synchronize()
                print(f"  BN={BN:3d} w={nw}: {s.elapsed_time(e)/NUM_TIMED:.3f} ms")
            except Exception as ex:
                print(f"  BN={BN:3d} w={nw}: FAILED: {ex}")

    del x, ids, sorted_ids, sorted_idx, sums, cnts
    torch.cuda.empty_cache()
