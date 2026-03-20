#!/usr/bin/env python3
"""Compare with vs without CUDA graph."""
import torch
from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid, _graph_cache

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "D": 128, "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "D": 128, "K": 1000},
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "D": 256, "K": 512},
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "D": 128, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

SEED = 42
MAX_ITERS = 10
NUM_WARMUP = 3
NUM_TIMED = 10

for idx, wl in enumerate(WORKLOADS):
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]

    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED + idx)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16, generator=gen)
    indices = torch.randint(0, N, (B, K), device="cuda", generator=gen)
    init_centroids = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D)).clone()

    # Clear graph cache to test fresh
    _graph_cache.clear()

    # Warmup (with graphs)
    for _ in range(NUM_WARMUP):
        batch_kmeans_Euclid(x, K, max_iters=MAX_ITERS, tol=-1.0, init_centroids=init_centroids.clone())
    torch.cuda.synchronize()

    # Timed (with graphs - should be all replays)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
    for i in range(NUM_TIMED):
        starts[i].record()
        batch_kmeans_Euclid(x, K, max_iters=MAX_ITERS, tol=-1.0, init_centroids=init_centroids.clone())
        ends[i].record()
    torch.cuda.synchronize()
    times_graph = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    avg_graph = sum(times_graph) / len(times_graph)

    tp_graph = (B * N * MAX_ITERS / (avg_graph / 1000)) / 1e6
    print(f"{wl['label']:15s}  graph={avg_graph:8.2f}ms  tp={tp_graph:8.1f}")

    del x, init_centroids
    torch.cuda.empty_cache()
