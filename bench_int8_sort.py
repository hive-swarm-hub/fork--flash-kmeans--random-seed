#!/usr/bin/env python3
"""Test int8 sort for small K."""
import torch

WORKLOADS = [
    {"label": "small-batch",  "B": 8,  "N": 16384,  "K": 100},
    {"label": "medium-std",   "B": 32, "N": 32768,  "K": 1000},
    {"label": "stress",       "B": 4,  "N": 262144, "K": 8192},
]

NUM_WARMUP = 10
NUM_TIMED = 50

for wl in WORKLOADS:
    B, N, K = wl["B"], wl["N"], wl["K"]
    print(f"\n{wl['label']}: B={B}, N={N}, K={K}")

    ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # int16 sort (current)
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

    if K <= 127:
        # int8 sort
        ids8 = ids.to(torch.int8)
        out8_v = torch.empty_like(ids8)
        out8_i = torch.empty(B, N, device="cuda", dtype=torch.int64)
        for _ in range(NUM_WARMUP):
            torch.sort(ids8, dim=-1, stable=False, out=(out8_v, out8_i))
        torch.cuda.synchronize()
        s.record()
        for _ in range(NUM_TIMED):
            torch.sort(ids8, dim=-1, stable=False, out=(out8_v, out8_i))
        e.record(); torch.cuda.synchronize()
        print(f"  int8 sort:  {s.elapsed_time(e)/NUM_TIMED:.4f} ms")

        # Also test: cast to int8 + sort
        for _ in range(NUM_WARMUP):
            ids8_t = ids.to(torch.int8)
            torch.sort(ids8_t, dim=-1, stable=False, out=(out8_v, out8_i))
        torch.cuda.synchronize()
        s.record()
        for _ in range(NUM_TIMED):
            ids8_t = ids.to(torch.int8)
            torch.sort(ids8_t, dim=-1, stable=False, out=(out8_v, out8_i))
        e.record(); torch.cuda.synchronize()
        print(f"  cast+int8:  {s.elapsed_time(e)/NUM_TIMED:.4f} ms")

    # Also try: counting sort via bincount+cumsum+scatter
    # Step 1: histogram
    for _ in range(NUM_WARMUP):
        hist = torch.zeros(B, K, device="cuda", dtype=torch.int32)
        hist.scatter_add_(1, ids.to(torch.int64), torch.ones_like(ids))
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        hist = torch.zeros(B, K, device="cuda", dtype=torch.int32)
        hist.scatter_add_(1, ids.to(torch.int64), torch.ones_like(ids))
    e.record(); torch.cuda.synchronize()
    print(f"  histogram:  {s.elapsed_time(e)/NUM_TIMED:.4f} ms")

    del ids
    torch.cuda.empty_cache()
