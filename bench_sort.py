#!/usr/bin/env python3
"""Benchmark sort performance with different dtypes and approaches."""
import torch

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "K": 1000},
    {"label": "medium-wide",  "B": 8,  "N": 65536,  "K": 512},
    {"label": "large-dense",  "B": 32, "N": 65536,  "K": 4096},
    {"label": "large-scale",  "B": 8,  "N": 131072, "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "K": 8192},
]

NUM_WARMUP = 10
NUM_TIMED = 50

for wl in WORKLOADS:
    B, N, K = wl["B"], wl["N"], wl["K"]
    label = wl["label"]
    print(f"\n{label}: B={B}, N={N}, K={K}")

    ids32 = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    ids16 = ids32.to(torch.int16)
    out32_v = torch.empty_like(ids32)
    out32_i = torch.empty(B, N, device="cuda", dtype=torch.int64)
    out16_v = torch.empty_like(ids16)
    out16_i = torch.empty(B, N, device="cuda", dtype=torch.int64)

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Sort int32
    for _ in range(NUM_WARMUP):
        torch.sort(ids32, dim=-1, stable=False, out=(out32_v, out32_i))
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        torch.sort(ids32, dim=-1, stable=False, out=(out32_v, out32_i))
    e.record(); torch.cuda.synchronize()
    print(f"  int32 sort: {s.elapsed_time(e)/NUM_TIMED:.3f} ms")

    # Sort int16
    for _ in range(NUM_WARMUP):
        torch.sort(ids16, dim=-1, stable=False, out=(out16_v, out16_i))
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        torch.sort(ids16, dim=-1, stable=False, out=(out16_v, out16_i))
    e.record(); torch.cuda.synchronize()
    print(f"  int16 sort: {s.elapsed_time(e)/NUM_TIMED:.3f} ms")

    # Sort int32 with int32 output indices (check if possible via argsort)
    # Actually try: cast to int16 + sort
    for _ in range(NUM_WARMUP):
        ids16_t = ids32.to(torch.int16)
        torch.sort(ids16_t, dim=-1, stable=False, out=(out16_v, out16_i))
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        ids16_t = ids32.to(torch.int16)
        torch.sort(ids16_t, dim=-1, stable=False, out=(out16_v, out16_i))
    e.record(); torch.cuda.synchronize()
    print(f"  cast+int16: {s.elapsed_time(e)/NUM_TIMED:.3f} ms")

    del ids32, ids16, out32_v, out32_i, out16_v, out16_i
    torch.cuda.empty_cache()
