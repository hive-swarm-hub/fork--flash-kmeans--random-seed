#!/usr/bin/env python3
"""Benchmark end-to-end batch_kmeans_Euclid to measure CUDA graph efficiency."""
import torch
from flash_kmeans import batch_kmeans_Euclid

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "D": 128, "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

MAX_ITERS = 10
NUM_WARMUP = 3
NUM_TIMED = 10

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    label = wl["label"]

    gen = torch.Generator(device="cuda")
    gen.manual_seed(42)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16, generator=gen)
    indices = torch.randint(0, N, (B, K), device="cuda", generator=gen)
    init_centroids = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D)).clone()

    # Warmup
    for _ in range(NUM_WARMUP):
        batch_kmeans_Euclid(x, K, max_iters=MAX_ITERS, tol=-1.0, init_centroids=init_centroids.clone())
    torch.cuda.synchronize()

    # Timed
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
    for i in range(NUM_TIMED):
        start_events[i].record()
        batch_kmeans_Euclid(x, K, max_iters=MAX_ITERS, tol=-1.0, init_centroids=init_centroids.clone())
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times) / len(times)
    per_iter = avg_ms / MAX_ITERS
    throughput = B * N * MAX_ITERS / avg_ms / 1e3  # Mpts/s

    # Compare with kernel-only estimate
    print(f"{label:15s}: {avg_ms:7.2f} ms total, {per_iter:6.3f} ms/iter, {throughput:8.1f} Mpts/s")

    del x, init_centroids
    torch.cuda.empty_cache()
