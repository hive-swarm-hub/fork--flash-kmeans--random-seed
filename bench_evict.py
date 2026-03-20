#!/usr/bin/env python3
"""Test eviction policy impact on assignment kernel."""
import torch
import triton
import triton.language as tl
from flash_kmeans.assign_euclid_triton import (
    _min_argmin_combine, _heuristic_euclid_config, compute_sq_norms
)

# Variant with no eviction policies
@triton.jit
def _assign_no_evict(
    x_ptr, c_ptr, x_sq_ptr, c_sq_ptr, out_ptr,
    B: tl.constexpr, N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    stride_x_b: tl.constexpr, stride_x_n: tl.constexpr, stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr, stride_c_k: tl.constexpr, stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr, stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr, stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1).to(tl.int64)
    n_start = pid_n * BLOCK_N
    n_offsets = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = n_offsets < N
    offs_d = tl.arange(0, D).to(tl.int64)
    x_ptrs = x_ptr + pid_b*stride_x_b + n_offsets[:,None]*stride_x_n + offs_d[None,:]*stride_x_d
    x_tile = tl.load(x_ptrs, mask=n_mask[:,None], other=0.0)  # NO eviction policy
    best_neg_score = tl.full((BLOCK_N,), float('inf'), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), tl.int32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
        k_mask = k_offsets < K
        c_ptrs = c_ptr + pid_b*stride_c_b + k_offsets[None,:]*stride_c_k + offs_d[:,None]*stride_c_d
        c_tile = tl.load(c_ptrs, mask=k_mask[None,:], other=0.0)  # NO eviction policy
        csq_ptrs = c_sq_ptr + pid_b*stride_csq_b + k_offsets*stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        cross = tl.dot(x_tile, c_tile, out_dtype=tl.float16, max_num_imprecise_acc=D)
        neg_score = cent_sq[None,:] - 2.0 * cross.to(tl.float32)
        neg_score = tl.where(k_mask[None,:], neg_score, float('inf'))
        tile_indices = (tl.arange(0, BLOCK_K) + k_start).to(tl.int32)
        tile_indices_2d = tl.broadcast_to(tile_indices[None,:], (BLOCK_N, BLOCK_K))
        curr_min, curr_abs_idx = tl.reduce((neg_score, tile_indices_2d), axis=1, combine_fn=_min_argmin_combine)
        update = curr_min < best_neg_score
        best_neg_score = tl.where(update, curr_min, best_neg_score)
        best_idx = tl.where(update, curr_abs_idx, best_idx)
    out_ptrs = out_ptr + pid_b*stride_out_b + n_offsets*stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

WORKLOADS = [
    {"label": "large-dense",  "B": 32, "N": 65536,  "D": 128, "K": 4096},
    {"label": "stress",       "B": 4,  "N": 262144, "D": 128, "K": 8192},
]

NUM_WARMUP = 3
NUM_TIMED = 15

for wl in WORKLOADS:
    B, N, D, K = wl["B"], wl["N"], wl["D"], wl["K"]
    print(f"\n{wl['label']}:")
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    c = torch.randn(B, K, D, device="cuda", dtype=torch.float16)
    x_sq = compute_sq_norms(x)
    c_sq = compute_sq_norms(c)
    out = torch.empty(B, N, device="cuda", dtype=torch.int32)
    cfg = _heuristic_euclid_config(N, K, D, device=x.device)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    # With eviction policies (current)
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    for _ in range(NUM_WARMUP):
        euclid_assign_triton(x, c, x_sq, out=out, c_sq=c_sq, config=cfg, use_heuristic=False)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(NUM_TIMED):
        euclid_assign_triton(x, c, x_sq, out=out, c_sq=c_sq, config=cfg, use_heuristic=False)
    e.record(); torch.cuda.synchronize()
    print(f"  With eviction:    {s.elapsed_time(e)/NUM_TIMED:.3f} ms")

    # Without eviction policies
    for _ in range(NUM_WARMUP):
        _assign_no_evict[grid](x, c, x_sq, c_sq, out, B, N, K, D,
            *x.stride(), *c.stride(), *x_sq.stride(), *c_sq.stride(), *out.stride(),
            BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    torch.cuda.synchronize()
    s.record()
    for _ in range(NUM_TIMED):
        _assign_no_evict[grid](x, c, x_sq, c_sq, out, B, N, K, D,
            *x.stride(), *c.stride(), *x_sq.stride(), *c_sq.stride(), *out.stride(),
            BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    e.record(); torch.cuda.synchronize()
    print(f"  No eviction:      {s.elapsed_time(e)/NUM_TIMED:.3f} ms")

    del x, c, x_sq, c_sq, out
    torch.cuda.empty_cache()
