#!/usr/bin/env python3
"""Tuning sweep for V2 kernel (without x_sq in inner loop)."""
import torch
import triton
from flash_kmeans.assign_euclid_triton import _euclid_assign_kernel, compute_sq_norms

WORKLOADS = [
    {"label": "stress",      "B": 4,  "N": 262144, "D": 128, "K": 8192},
    {"label": "large-dense", "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "medium-std",  "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "medium-wide", "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "large-scale", "B": 8,  "N": 131072, "D": 128, "K": 1000},
    {"label": "small-batch", "B": 8,  "N": 16384,  "D": 128, "K": 100},
]

CONFIGS = [
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 128, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 64,  "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 64,  "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 8, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 256, "num_warps": 4, "num_stages": 1},
    {"BLOCK_N": 64,  "BLOCK_K": 256, "num_warps": 8, "num_stages": 1},
]

def bench_config(wl, config, num_warmup=3, num_timed=20):
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    centroids = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(centroids)
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    try:
        for _ in range(num_warmup):
            _euclid_assign_kernel[grid](x, centroids, x_sq, c_sq, out, B, N, K, D,
                x.stride(0), x.stride(1), x.stride(2),
                centroids.stride(0), centroids.stride(1), centroids.stride(2),
                x_sq.stride(0), x_sq.stride(1), c_sq.stride(0), c_sq.stride(1),
                out.stride(0), out.stride(1), **config)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(num_timed):
            _euclid_assign_kernel[grid](x, centroids, x_sq, c_sq, out, B, N, K, D,
                x.stride(0), x.stride(1), x.stride(2),
                centroids.stride(0), centroids.stride(1), centroids.stride(2),
                x_sq.stride(0), x_sq.stride(1), c_sq.stride(0), c_sq.stride(1),
                out.stride(0), out.stride(1), **config)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / num_timed
    except:
        return None

for wl in WORKLOADS:
    print(f"\n=== {wl['label']} (B={wl['B']}, N={wl['N']}, D={wl['D']}, K={wl['K']}) ===")
    results = []
    for cfg in CONFIGS:
        ms = bench_config(wl, cfg)
        if ms is not None:
            results.append((ms, cfg))
    results.sort(key=lambda x: x[0])
    for ms, cfg in results[:5]:
        print(f"  {ms:8.3f} ms  BN={cfg['BLOCK_N']:3d} BK={cfg['BLOCK_K']:3d} w={cfg['num_warps']} s={cfg['num_stages']}")
    torch.cuda.empty_cache()
