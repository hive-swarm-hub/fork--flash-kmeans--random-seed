#!/usr/bin/env python3
"""Benchmark counting sort kernel parameters."""
import torch
import triton
from flash_kmeans.centroid_update_triton import _histogram_kernel, _counting_scatter_kernel

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "K": 1000},
    {"label": "large-dense",  "B": 32, "N": 65536,  "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "K": 8192},
]

NUM_WARMUP = 10
NUM_TIMED = 50

for wl in WORKLOADS:
    B, N, K = wl["B"], wl["N"], wl["K"]
    print(f"\n{wl['label']}: B={B}, N={N}, K={K}")

    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    hist = torch.zeros(B, K, device="cuda", dtype=torch.int32)
    offsets = torch.empty(B, K, device="cuda", dtype=torch.int32)
    out_v = torch.empty(B, N, device="cuda", dtype=torch.int16)
    out_i = torch.empty(B, N, device="cuda", dtype=torch.int64)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for BN in [64, 128, 256, 512]:
        for nw in [1, 2, 4]:
            grid = (triton.cdiv(N, BN), B)
            try:
                # Warmup full counting sort
                for _ in range(NUM_WARMUP):
                    hist.zero_()
                    _histogram_kernel[grid](ids, hist, N=N, K=K, BLOCK_N=BN, num_warps=nw)
                    torch.cumsum(hist, dim=1, out=offsets)
                    offsets -= hist
                    _counting_scatter_kernel[grid](ids, offsets, out_v, out_i, N=N, K=K, BLOCK_N=BN, num_warps=nw)
                torch.cuda.synchronize()

                s.record()
                for _ in range(NUM_TIMED):
                    hist.zero_()
                    _histogram_kernel[grid](ids, hist, N=N, K=K, BLOCK_N=BN, num_warps=nw)
                    torch.cumsum(hist, dim=1, out=offsets)
                    offsets -= hist
                    _counting_scatter_kernel[grid](ids, offsets, out_v, out_i, N=N, K=K, BLOCK_N=BN, num_warps=nw)
                e.record(); torch.cuda.synchronize()
                total = s.elapsed_time(e) / NUM_TIMED
                print(f"  BN={BN:3d} w={nw}: {total:.4f} ms")
            except Exception as ex:
                print(f"  BN={BN:3d} w={nw}: FAILED: {ex}")

    # Compare with int16 radix sort
    ids16 = ids.to(torch.int16)
    out16_v = torch.empty_like(ids16)
    out16_i = torch.empty(B, N, device="cuda", dtype=torch.int64)
    for _ in range(NUM_WARMUP):
        torch.sort(ids16, dim=-1, stable=False, out=(out16_v, out16_i))
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        torch.sort(ids16, dim=-1, stable=False, out=(out16_v, out16_i))
    e.record(); torch.cuda.synchronize()
    print(f"  int16 sort: {s.elapsed_time(e)/NUM_TIMED:.4f} ms")

    del ids, hist, offsets, out_v, out_i
    torch.cuda.empty_cache()
