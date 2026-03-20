#!/usr/bin/env python3
"""Quick tuning sweep for assignment kernel configs on bottleneck workloads."""
import torch
import time
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, compute_sq_norms

WORKLOADS = [
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
    {"label": "medium-std",   "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "small-batch",  "B": 8,  "N": 16384,  "D": 128, "K": 100},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
]

CONFIGS = [
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 64,  "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 256, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 256, "num_warps": 8, "num_stages": 1},
]

def bench_config(wl, config, num_warmup=3, num_timed=10):
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)

    try:
        for _ in range(num_warmup):
            euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=config, use_heuristic=False)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_timed):
            euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=config, use_heuristic=False)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / num_timed
        return avg_ms
    except Exception as e:
        return None

for wl in WORKLOADS:
    print(f"\n=== {wl['label']} (B={wl['B']}, N={wl['N']}, D={wl['D']}, K={wl['K']}) ===")
    results = []
    for cfg in CONFIGS:
        if cfg["BLOCK_K"] > wl["D"] and wl["D"] < 256:
            # Skip invalid: BLOCK_K must be <= K but also D must be >= BLOCK_K for the dot
            pass
        ms = bench_config(wl, cfg)
        if ms is not None:
            results.append((ms, cfg))
    results.sort(key=lambda x: x[0])
    for ms, cfg in results[:5]:
        print(f"  {ms:8.3f} ms  BN={cfg['BLOCK_N']:3d} BK={cfg['BLOCK_K']:3d} warps={cfg['num_warps']} stages={cfg['num_stages']}")
    torch.cuda.empty_cache()
