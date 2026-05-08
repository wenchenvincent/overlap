"""Correctness check: primus CK kernel with --ws-ck must match the static run.

Single-GPU. Sweeps shapes including some where total_tiles % NUM_SMS != 0
(those are what unmasked the latent phase-2 NaN bug in the Triton kernel —
make sure the CK kernel doesn't have a parallel issue).

Usage:
    python3 test_ws_ck_correctness.py
"""

from __future__ import annotations

import torch

from primus_turbo.pytorch.ops import grouped_gemm as pt_grouped_gemm


def _make_inputs(G, M, K, N, trans_b, device, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=dtype, device=device)
    if trans_b:
        b = torch.randn(G, N, K, dtype=dtype, device=device)
    else:
        b = torch.randn(G, K, N, dtype=dtype, device=device)
    base = M // G
    rem = M % G
    gs = [base] * G
    gs[-1] += rem
    group_lens = torch.tensor(gs, dtype=torch.int64, device=device)
    return a, b, group_lens


def main():
    device = torch.device("cuda:0")
    NUM_XCDS_WS = 8  # matches grouped_gemm_kernel_ws.hpp
    counter = torch.zeros(NUM_XCDS_WS + 1, dtype=torch.int32, device=device)

    cases = [
        # (G, M, K, N, trans_b)
        (1, 1024, 1280, 2560, True),
        (8, 16384, 1280, 2560, True),
        (8, 16384, 4096, 4096, False),
        (32, 267424, 1280, 2560, True),  # bench shape
        (32, 8192, 1280, 2560, True),    # smaller, more ragged groups
        # Shapes where total_tiles is unlikely to divide evenly across NUM_SMS=256.
        (1, 1024, 1280, 2304, True),     # 4 * 9 = 36 tiles
        (4, 4096, 1280, 5120, True),     # ragged but typical
    ]

    # Sweep all three modes for each shape; each mode picks a different
    # `ws_local_per_xcd`.
    M_TILE = N_TILE = 256
    print(f"primus CK WS correctness sweep ({len(cases)} cases × 3 modes)")
    all_ok = True
    for G, M, K, N, trans_b in cases:
        a, b, gs = _make_inputs(G, M, K, N, trans_b, device)
        out_static = pt_grouped_gemm(a, b, gs, trans_b=trans_b)
        num_n_tiles = (N + N_TILE - 1) // N_TILE
        m_tiles = sum((m_g + M_TILE - 1) // M_TILE for m_g in gs.tolist())
        total_tiles = m_tiles * num_n_tiles
        for mode in ("global", "per-xcd", "hierarchical"):
            if mode == "global":
                local_per_xcd = 0
            elif mode == "per-xcd":
                local_per_xcd = (total_tiles + NUM_XCDS_WS - 1) // NUM_XCDS_WS
            else:
                local_per_xcd = max(1, (total_tiles // 2) // NUM_XCDS_WS)
            out_ws = pt_grouped_gemm(
                a, b, gs, trans_b=trans_b, work_steal=True,
                ws_counter=counter, ws_local_per_xcd=local_per_xcd)
            torch.cuda.synchronize()

            diff = (out_static.float() - out_ws.float()).abs()
            denom = out_static.float().abs().clamp_min(1.0)
            max_abs = diff.max().item()
            max_rel = (diff / denom).max().item()
            nan_ws = torch.isnan(out_ws.float()).sum().item()

            ok = max_abs < 5e-2 and max_rel < 5e-2 and nan_ws == 0
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            slots = counter[:NUM_XCDS_WS].tolist()
            global_slot = counter[NUM_XCDS_WS].item()
            print(
                f"  G={G:>3} M={M:>7} N={N:>5} trans_b={trans_b!s:<5} "
                f"mode={mode:<13s} L/X={local_per_xcd:>5d} "
                f"max_abs={max_abs:.1e} "
                f"slots[{min(slots)}..{max(slots)}] global={global_slot} "
                f"NaN={nan_ws} [{status}]"
            )

    print("ALL PASS" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
