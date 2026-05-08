"""End-to-end backward correctness check for primus CK WS.

Runs `out = grouped_gemm(a, b, group_lens)`, then `out.sum().backward()`,
twice — once with WS off and once with WS on — and asserts grad_a / grad_b
are bit-identical (or close, given bf16 accumulation tolerance).

Exercises:
- The forward kernel (WS path used for `out`)
- `grad_a = grouped_gemm_impl(...)` (forward kernel, dgrad shape)
- `grad_b = grouped_gemm_variable_k_impl(...)` (variable-K kernel, wgrad)

All three should match the static-stride reference.
"""

from __future__ import annotations

import torch
from primus_turbo.pytorch.ops import grouped_gemm as pt_grouped_gemm


NUM_XCDS_WS = 8


def _make_inputs(G, M, K, N, trans_b, device, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    if trans_b:
        b = torch.randn(G, N, K, dtype=dtype, device=device, requires_grad=True)
    else:
        b = torch.randn(G, K, N, dtype=dtype, device=device, requires_grad=True)
    base = M // G
    rem = M % G
    gs_list = [base] * G
    gs_list[-1] += rem
    group_lens = torch.tensor(gs_list, dtype=torch.int64, device=device)
    return a, b, group_lens


def _run_one(a, b, gs, trans_b, work_steal, ws_counter, ws_local_per_xcd):
    a_c = a.detach().clone().requires_grad_(True)
    b_c = b.detach().clone().requires_grad_(True)
    out = pt_grouped_gemm(
        a_c, b_c, gs,
        trans_b=trans_b,
        work_steal=work_steal,
        ws_counter=ws_counter,
        ws_local_per_xcd=ws_local_per_xcd,
    )
    out.sum().backward()
    torch.cuda.synchronize()
    return out.detach(), a_c.grad.detach(), b_c.grad.detach()


def main():
    device = torch.device("cuda:0")
    counter = torch.zeros(NUM_XCDS_WS + 1, dtype=torch.int32, device=device)

    cases = [
        # (G, M, K, N, trans_b)
        (8, 16384, 1280, 2560, True),
        (8, 16384, 4096, 4096, False),
        (32, 267424, 1280, 2560, True),  # bench shape
        (32, 8192, 1280, 2560, True),
    ]

    M_TILE = N_TILE = 256
    print(f"primus CK backward WS sweep ({len(cases)} cases × 3 modes)")
    all_ok = True
    for G, M, K, N, trans_b in cases:
        a, b, gs = _make_inputs(G, M, K, N, trans_b, device)
        # Reference: WS off
        out_ref, gA_ref, gB_ref = _run_one(a, b, gs, trans_b, False, None, 0)

        # total tiles for the FORWARD shape — gives a sensible local_per_xcd
        # for grad_a. (grad_b is variable-K; same kernel runner picks compatible
        # tile shape, the ws_counter just gets zeroed at each launch.)
        num_n_tiles = (N + N_TILE - 1) // N_TILE
        total_tiles = sum((m_g + M_TILE - 1) // M_TILE for m_g in gs.tolist()) * num_n_tiles

        for mode in ("global", "per-xcd", "hierarchical"):
            if mode == "global":
                lpx = 0
            elif mode == "per-xcd":
                lpx = (total_tiles + NUM_XCDS_WS - 1) // NUM_XCDS_WS
            else:
                lpx = max(1, (total_tiles // 2) // NUM_XCDS_WS)

            out_ws, gA_ws, gB_ws = _run_one(a, b, gs, trans_b, True, counter, lpx)

            d_out = (out_ref.float() - out_ws.float()).abs().max().item()
            d_gA  = (gA_ref.float() - gA_ws.float()).abs().max().item()
            d_gB  = (gB_ref.float() - gB_ws.float()).abs().max().item()
            ok = d_out < 5e-2 and d_gA < 5e-2 and d_gB < 5e-2
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"  G={G:>3} M={M:>7} N={N:>5} trans_b={trans_b!s:<5} "
                f"mode={mode:<13s} L/X={lpx:>5d} "
                f"|d_out|={d_out:.1e} |d_gA|={d_gA:.1e} |d_gB|={d_gB:.1e} [{status}]"
            )

    print("ALL PASS" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
