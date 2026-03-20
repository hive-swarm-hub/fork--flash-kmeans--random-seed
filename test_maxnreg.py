#!/usr/bin/env python3
"""Test maxnreg to improve occupancy."""
import torch
import triton
from flash_kmeans.assign_euclid_triton import _euclid_assign_kernel, compute_sq_norms

def bench(wl, config, num_warmup=5, num_timed=20):
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
    except Exception as ex:
        return f"ERR: {str(ex)[:60]}"

WORKLOADS = [
    {"label": "stress",      "B": 4,  "N": 262144,"D": 128, "K": 8192},
    {"label": "large-dense", "B": 32, "N": 65536, "D": 128, "K": 4096},
]

for wl in WORKLOADS:
    print(f"\n=== {wl['label']} ===")
    base_cfg = {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 1}

    ms = bench(wl, base_cfg)
    print(f"  Default (no maxnreg): {ms}")

    for mr in [64, 96, 128, 160, 192, 255]:
        cfg = {**base_cfg, "maxnreg": mr}
        ms = bench(wl, cfg)
        print(f"  maxnreg={mr}: {ms}")

    torch.cuda.empty_cache()
