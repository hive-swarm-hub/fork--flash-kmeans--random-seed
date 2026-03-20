#!/usr/bin/env python3
"""Micro-benchmark different assignment kernel configs per workload."""
import torch
from flash_kmeans.assign_euclid_triton import euclid_assign_triton, compute_sq_norms

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "D": 128, "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

CONFIGS = [
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 64,  "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 64,  "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
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

    best_time = float('inf')
    best_cfg = None

    for cfg in CONFIGS:
        # Skip configs where D < BLOCK_K (no issue here since D >= 128)
        try:
            # Warmup
            for _ in range(NUM_WARMUP):
                euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=cfg, use_heuristic=False)
            torch.cuda.synchronize()

            # Timed
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(NUM_TIMED):
                euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq, config=cfg, use_heuristic=False)
            end.record()
            torch.cuda.synchronize()

            avg_ms = start.elapsed_time(end) / NUM_TIMED
            marker = ""
            if avg_ms < best_time:
                best_time = avg_ms
                best_cfg = cfg
                marker = " <-- BEST"

            print(f"  BN={cfg['BLOCK_N']:3d} BK={cfg['BLOCK_K']:3d} w={cfg['num_warps']} s={cfg['num_stages']}  -> {avg_ms:.3f} ms{marker}")
        except Exception as e:
            print(f"  BN={cfg['BLOCK_N']:3d} BK={cfg['BLOCK_K']:3d} w={cfg['num_warps']} s={cfg['num_stages']}  -> FAILED: {e}")

    print(f"  BEST: BN={best_cfg['BLOCK_N']} BK={best_cfg['BLOCK_K']} w={best_cfg['num_warps']} s={best_cfg['num_stages']} -> {best_time:.3f} ms")

    del x, centroids, x_sq, c_sq, out
    torch.cuda.empty_cache()
