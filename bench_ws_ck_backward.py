"""Benchmark CK WS overhead on the backward pass (single GPU, no overlap).

For each (mode, grid_dim) we time:
  - forward only        (gemm_only-equivalent for the backward bench)
  - backward only       (out.backward(go))

Backward fires both kernels: `grad_a` via the forward kernel (dgrad shape)
and `grad_b` via the variable-K kernel (wgrad). Both use WS when --ws-ck
is on, so we can isolate the variable-K overhead from the forward-kernel
overhead.

Mirrors the no-overlap forward bench in style. Reports mean / min over
many iterations.
"""

from __future__ import annotations

import argparse
import statistics
import torch
from primus_turbo.pytorch.ops import grouped_gemm as pt_grouped_gemm


NUM_XCDS_WS = 8


def make_inputs(G, M, K, N, trans_b, device, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    if trans_b:
        b = torch.randn(G, N, K, dtype=dtype, device=device, requires_grad=True)
    else:
        b = torch.randn(G, K, N, dtype=dtype, device=device, requires_grad=True)
    base = M // G
    rem = M % G
    gs = [base] * G
    gs[-1] += rem
    group_lens = torch.tensor(gs, dtype=torch.int64, device=device)
    return a, b, group_lens


def time_forward_backward(a, b, gs, trans_b, work_steal, counter, lpx,
                          warmup=20, iters=100):
    fwd_times = []
    bwd_times = []

    # Warm up
    for _ in range(warmup):
        a.grad = None
        b.grad = None
        out = pt_grouped_gemm(
            a, b, gs, trans_b=trans_b, work_steal=work_steal,
            ws_counter=counter, ws_local_per_xcd=lpx)
        out.sum().backward()
    torch.cuda.synchronize()

    fwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    fwd_end   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    bwd_end   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # Persistent gradient tensor for the backward upstream — so .backward()
    # itself doesn't create a new tensor each iteration.
    for i in range(iters):
        a.grad = None
        b.grad = None
        fwd_start[i].record()
        out = pt_grouped_gemm(
            a, b, gs, trans_b=trans_b, work_steal=work_steal,
            ws_counter=counter, ws_local_per_xcd=lpx)
        fwd_end[i].record()
        bwd_start[i].record()
        out.sum().backward()
        bwd_end[i].record()
    torch.cuda.synchronize()

    fwd_times = [s.elapsed_time(e) for s, e in zip(fwd_start, fwd_end)]
    bwd_times = [s.elapsed_time(e) for s, e in zip(bwd_start, bwd_end)]
    return fwd_times, bwd_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--M", type=int, default=267424)
    ap.add_argument("--K", type=int, default=1280)
    ap.add_argument("--N", type=int, default=2560)
    ap.add_argument("--trans-b", action="store_true", default=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    counter = torch.zeros(NUM_XCDS_WS + 1, dtype=torch.int32, device=device)

    a, b, gs = make_inputs(args.G, args.M, args.K, args.N, args.trans_b, device)

    M_TILE = N_TILE = 256
    num_n_tiles = (args.N + N_TILE - 1) // N_TILE
    total_tiles = sum((m_g + M_TILE - 1) // M_TILE for m_g in gs.tolist()) * num_n_tiles
    print(f"shape: G={args.G} M={args.M} K={args.K} N={args.N} trans_b={args.trans_b}")
    print(f"total_tiles={total_tiles} (forward shape; backward grad_b is variable-K)")
    print()

    rows = []
    # Static reference
    fwd, bwd = time_forward_backward(
        a, b, gs, args.trans_b, False, None, 0, args.warmup, args.iters)
    rows.append(("static", 0,
                 statistics.mean(fwd), min(fwd),
                 statistics.mean(bwd), min(bwd)))

    for mode in ("global", "per-xcd", "hierarchical"):
        if mode == "global":
            lpx = 0
        elif mode == "per-xcd":
            lpx = (total_tiles + NUM_XCDS_WS - 1) // NUM_XCDS_WS
        else:
            lpx = max(1, (total_tiles // 2) // NUM_XCDS_WS)
        fwd, bwd = time_forward_backward(
            a, b, gs, args.trans_b, True, counter, lpx, args.warmup, args.iters)
        rows.append((f"--ws-ck mode={mode}", lpx,
                     statistics.mean(fwd), min(fwd),
                     statistics.mean(bwd), min(bwd)))

    static_fwd_min = rows[0][3]
    static_bwd_min = rows[0][5]
    print(f"{'config':<28s} {'L/X':>5s}  {'fwd mean':>9s}  {'fwd min':>8s}  "
          f"{'bwd mean':>9s}  {'bwd min':>8s}  {'fwd Δ':>7s}  {'bwd Δ':>7s}")
    for name, lpx, f_mean, f_min, b_mean, b_min in rows:
        f_d = (f_min / static_fwd_min - 1.0) * 100.0
        b_d = (b_min / static_bwd_min - 1.0) * 100.0
        print(f"{name:<28s} {lpx:>5d}  {f_mean:>8.3f}ms {f_min:>7.3f}ms  "
              f"{b_mean:>8.3f}ms {b_min:>7.3f}ms  "
              f"{f_d:>+6.1f}% {b_d:>+6.1f}%")


if __name__ == "__main__":
    main()
