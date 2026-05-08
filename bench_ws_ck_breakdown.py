"""Per-kernel WS overhead: fwd vs dgrad vs wgrad.

Backward is two kernels:
  - dgrad (computes grad_a = dy @ b)   → uses forward kernel
  - wgrad (computes grad_b = x^T @ dy) → uses variable-K kernel

Each has different (M, K, N) and therefore different total_tiles, so the
WS overhead percentage differs. This bench calls each kernel directly
(bypassing autograd) and times it with shape-correct ws_local_per_xcd
when WS modes are exercised.

For forward shape (G, M=GM, K, N) with trans_b=True:
  fwd:    a[GM, K] · b[G, N, K]ᵀ   = out[GM, N]
          tiles = sum_g ceil(M_g/M_TILE) * ceil(N/N_TILE)
  dgrad:  dy[GM, N] · b[G, N, K]   = grad_a[GM, K]
          (forward kernel called with trans_b=False; "N" plays K-role)
          tiles = sum_g ceil(M_g/M_TILE) * ceil(K/N_TILE)
  wgrad:  x[GM, K]ᵀ · dy[GM, N]    = grad_b[G, K, N]   (variable-K)
          per-group output is [K, N], same shape across groups
          tiles = G * ceil(K/M_TILE) * ceil(N/N_TILE)
"""

from __future__ import annotations

import argparse
import statistics
import torch
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_impl,
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.core.backend import BackendType


NUM_XCDS_WS = 8
M_TILE = N_TILE = 256


def time_kernel(launch, warmup=20, iters=100):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        launch()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def lpx_for(mode, total_tiles):
    if mode == "global":
        return 0
    if mode == "per-xcd":
        return (total_tiles + NUM_XCDS_WS - 1) // NUM_XCDS_WS
    if mode == "hierarchical":
        return max(1, (total_tiles // 2) // NUM_XCDS_WS)
    raise ValueError(mode)


def report(name, total_tiles, rows):
    static_min = min(rows[0][2])
    print(f"\n--- {name}  (total_tiles={total_tiles}, ~{total_tiles // 256} per CTA) ---")
    print(f"{'config':<28s} {'L/X':>5s}  {'mean':>8s}  {'min':>8s}  {'Δ vs static (min)':>18s}")
    for cname, lpx, times in rows:
        mn = statistics.mean(times)
        mi = min(times)
        d = (mi / static_min - 1.0) * 100.0
        print(f"{cname:<28s} {lpx:>5d}  {mn:>7.3f}ms {mi:>7.3f}ms  {d:>+17.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--M", type=int, default=267424)
    ap.add_argument("--K", type=int, default=1280)
    ap.add_argument("--N", type=int, default=2560)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    counter = torch.zeros(NUM_XCDS_WS + 1, dtype=torch.int32, device=device)
    torch.manual_seed(0)

    # Inputs (trans_b=True forward layout)
    GM = args.G * (args.M // args.G)
    M_per_group = args.M // args.G
    base_gs = [M_per_group] * args.G
    base_gs[-1] += args.M % args.G
    group_lens = torch.tensor(base_gs, dtype=torch.int64, device=device)
    group_offs = grouped_gemm_compute_offs(group_lens)

    a  = torch.randn(GM, args.K, dtype=torch.bfloat16, device=device)
    b  = torch.randn(args.G, args.N, args.K, dtype=torch.bfloat16, device=device)  # trans_b
    dy = torch.randn(GM, args.N, dtype=torch.bfloat16, device=device)

    # Tile counts
    fwd_m_tiles = sum((m_g + M_TILE - 1) // M_TILE for m_g in base_gs)
    fwd_total   = fwd_m_tiles * ((args.N + N_TILE - 1) // N_TILE)
    dgr_total   = fwd_m_tiles * ((args.K + N_TILE - 1) // N_TILE)
    wgr_total   = args.G * ((args.K + M_TILE - 1) // M_TILE) * ((args.N + N_TILE - 1) // N_TILE)

    # ── fwd: grouped_gemm_impl(a, b, trans_b=True) ─────────────────────────
    def fwd_launch(ws, lpx):
        return lambda: grouped_gemm_impl(
            a, b, group_lens, group_offs,
            trans_a=False, trans_b=True, num_cu=None,
            default_backend=BackendType.CK.value, maybe_pre_sync=True,
            work_steal=ws, ws_counter=counter if ws else None, ws_local_per_xcd=lpx)

    rows_fwd = [("static", 0, time_kernel(fwd_launch(False, 0), args.warmup, args.iters))]
    for mode in ("global", "per-xcd", "hierarchical"):
        lpx = lpx_for(mode, fwd_total)
        rows_fwd.append((f"--ws-ck mode={mode}", lpx,
                         time_kernel(fwd_launch(True, lpx), args.warmup, args.iters)))
    report("fwd  (forward kernel)", fwd_total, rows_fwd)

    # ── dgrad: grouped_gemm_impl(dy, b, trans_b=False) ─────────────────────
    # Same forward kernel; "N" of fwd plays the K-loop role here.
    def dgrad_launch(ws, lpx):
        return lambda: grouped_gemm_impl(
            dy, b, group_lens, group_offs,
            trans_a=False, trans_b=False, num_cu=None,
            default_backend=BackendType.CK.value, maybe_pre_sync=True,
            work_steal=ws, ws_counter=counter if ws else None, ws_local_per_xcd=lpx)

    rows_dgrad = [("static", 0, time_kernel(dgrad_launch(False, 0), args.warmup, args.iters))]
    for mode in ("global", "per-xcd", "hierarchical"):
        lpx = lpx_for(mode, dgr_total)
        rows_dgrad.append((f"--ws-ck mode={mode}", lpx,
                           time_kernel(dgrad_launch(True, lpx), args.warmup, args.iters)))
    report("dgrad (forward kernel, rotated)", dgr_total, rows_dgrad)

    # ── wgrad: grouped_gemm_variable_k_impl(a, dy, trans_a=True, trans_c=True)
    def wgrad_launch(ws, lpx):
        return lambda: grouped_gemm_variable_k_impl(
            a, dy, group_lens, group_offs,
            trans_a=True, trans_b=False, trans_c=True, num_cu=None,
            default_backend=BackendType.CK.value,
            work_steal=ws, ws_counter=counter if ws else None, ws_local_per_xcd=lpx)

    rows_wgrad = [("static", 0, time_kernel(wgrad_launch(False, 0), args.warmup, args.iters))]
    for mode in ("global", "per-xcd", "hierarchical"):
        lpx = lpx_for(mode, wgr_total)
        rows_wgrad.append((f"--ws-ck mode={mode}", lpx,
                           time_kernel(wgrad_launch(True, lpx), args.warmup, args.iters)))
    report("wgrad (variable-K kernel)", wgr_total, rows_wgrad)


if __name__ == "__main__":
    main()
