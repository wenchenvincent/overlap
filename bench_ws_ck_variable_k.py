"""Direct bench of the variable-K kernel (wgrad), bypassing autograd.

The autograd pipeline passes the same `ws_local_per_xcd` to both forward
and variable-K backward. Forward shape and variable-K shape have very
different `total_tiles`, so a forward-tuned lpx is wrong for variable-K.
This bench calls `grouped_gemm_variable_k_impl` directly and computes the
correct lpx from variable-K's own total_tiles — giving us the apples-to-
apples per-mode overhead for the wgrad path.

Variable-K shape (for forward G/M/K/N with trans_b=True):
  per-group wgrad C is [K, N]  → m_tile dim is K, n_tile dim is N
  total_tiles_varK = G * ceil(K / M_TILE) * ceil(N / N_TILE)
"""

from __future__ import annotations

import argparse
import statistics
import torch
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
    grouped_gemm_compute_offs,
)
from primus_turbo.pytorch.core.backend import BackendType


NUM_XCDS_WS = 8
M_TILE = N_TILE = 256


def bench_variable_k(args, device):
    """Build a synthetic wgrad call: dy[GM, N], x[GM, K] → dw[G, K, N]."""
    torch.manual_seed(0)
    GM = args.G * (args.M // args.G)
    dy = torch.randn(GM, args.N, dtype=torch.bfloat16, device=device)
    x  = torch.randn(GM, args.K, dtype=torch.bfloat16, device=device)
    base = args.M // args.G
    rem = args.M % args.G
    gs = [base] * args.G
    gs[-1] += rem
    group_lens = torch.tensor(gs, dtype=torch.int64, device=device)
    group_offs = grouped_gemm_compute_offs(group_lens)

    counter = torch.zeros(NUM_XCDS_WS + 1, dtype=torch.int32, device=device)

    # variable-K total_tiles: same M, N across groups; K_g varies but contributes
    # only to the K-loop (same number of tiles per group).
    varK_tiles = args.G * ((args.K + M_TILE - 1) // M_TILE) * ((args.N + N_TILE - 1) // N_TILE)
    print(f"variable-K total_tiles = {varK_tiles}")

    def time_one(work_steal, lpx):
        # variable-K is invoked as grad_b: a=x (T), b=dy, trans_a=True, trans_b=False, trans_c=True
        for _ in range(args.warmup):
            grouped_gemm_variable_k_impl(
                x, dy, group_lens, group_offs,
                trans_a=True, trans_b=False, trans_c=True,
                num_cu=None, default_backend=BackendType.CK.value,
                work_steal=work_steal, ws_counter=counter if work_steal else None,
                ws_local_per_xcd=lpx)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        for i in range(args.iters):
            starts[i].record()
            grouped_gemm_variable_k_impl(
                x, dy, group_lens, group_offs,
                trans_a=True, trans_b=False, trans_c=True,
                num_cu=None, default_backend=BackendType.CK.value,
                work_steal=work_steal, ws_counter=counter if work_steal else None,
                ws_local_per_xcd=lpx)
            ends[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]

    rows = []
    rows.append(("static", 0, time_one(False, 0)))
    rows.append(("--ws-ck mode=global", 0, time_one(True, 0)))
    lpx_perxcd = (varK_tiles + NUM_XCDS_WS - 1) // NUM_XCDS_WS
    rows.append(("--ws-ck mode=per-xcd", lpx_perxcd, time_one(True, lpx_perxcd)))
    lpx_hier = max(1, (varK_tiles // 2) // NUM_XCDS_WS)
    rows.append(("--ws-ck mode=hierarchical", lpx_hier, time_one(True, lpx_hier)))

    static_min = min(rows[0][2])
    print(f"\n{'config':<30s} {'L/X':>5s}  {'mean':>8s}  {'min':>8s}  {'Δ vs static (min)':>18s}")
    for name, lpx, times in rows:
        mn = statistics.mean(times)
        mi = min(times)
        d = (mi / static_min - 1.0) * 100.0
        print(f"{name:<30s} {lpx:>5d}  {mn:>7.3f}ms {mi:>7.3f}ms  {d:>+17.1f}%")


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
    bench_variable_k(args, device)


if __name__ == "__main__":
    main()
