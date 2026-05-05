"""Correctness check: WS kernel output must match upstream static-stride kernel.

Runs the same inputs through both kernels and compares outputs across a sweep
of group counts and M sizes. Single-GPU; no distributed setup needed.

Usage:
    python3 test_ws_correctness.py
"""

from __future__ import annotations

import torch

from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    _grouped_bf16_persistent_gemm_kernel,
)
from vendored_primus import (
    NUM_XCDS,
    allocate_ws_counter_buf,
    compute_total_tiles_host,
    grouped_gemm_triton_kernel_ws,
    resolve_local_per_xcd,
)


def _run_static(a, b, group_offs, num_sms, num_xcds, trans_b):
    M, _ = a.shape
    G, d1, d2 = b.shape
    if trans_b:
        N, K_b = d1, d2
        stride_bn = b.stride(1)
        stride_bk = b.stride(2)
    else:
        K_b, N = d1, d2
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    _grouped_bf16_persistent_gemm_kernel[(num_sms,)](
        a, b, out, group_offs,
        G, N, K_b,
        a.stride(0),
        b.stride(0),
        stride_bn,
        out.stride(0),
        out.stride(1),
        stride_ak=a.stride(1),
        stride_bk=stride_bk,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=4,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=32,
        EVEN_K=(K_b % 64 == 0),
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        num_warps=8,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


def _make_inputs(G, M, K, N, trans_b, device, dtype=torch.bfloat16):
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
    offs = torch.zeros(G + 1, dtype=torch.int64, device=device)
    offs[1:] = torch.cumsum(group_lens, dim=0)
    return a, b, offs


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    counter_buf = allocate_ws_counter_buf(device, num_xcds=NUM_XCDS)

    cases = [
        # (G, M, K, N, trans_b)
        (1, 1024, 1280, 2560, True),
        (8, 16384, 1280, 2560, True),
        (8, 16384, 4096, 4096, False),
        (32, 267424, 1280, 2560, True),  # bench shape
        (32, 8192, 1280, 2560, True),    # smaller, more ragged groups
    ]

    print(f"NUM_SMS={num_sms}, NUM_XCDS={NUM_XCDS}")
    modes = ("auto", "per-xcd", "global", "hierarchical", "quota")
    all_ok = True
    for G, M, K, N, trans_b in cases:
        a, b, offs = _make_inputs(G, M, K, N, trans_b, device)
        out_static = _run_static(a, b, offs, num_sms, NUM_XCDS, trans_b)
        offs_cpu = offs.cpu().tolist()
        group_lens = [offs_cpu[g + 1] - offs_cpu[g] for g in range(G)]
        total_tiles = compute_total_tiles_host(group_lens, N)

        for mode in modes:
            local_per_xcd = resolve_local_per_xcd(
                total_tiles, num_sms, NUM_XCDS, mode
            )
            out_ws = grouped_gemm_triton_kernel_ws(
                a, b, offs, counter_buf,
                num_sms=num_sms,
                num_xcds=NUM_XCDS,
                trans_b=trans_b,
                work_steal=True,
                ws_mode=mode,
                total_tiles=total_tiles,
            )
            torch.cuda.synchronize()

            diff = (out_static.float() - out_ws.float()).abs()
            rel = diff / out_static.float().abs().clamp_min(1.0)
            max_abs = diff.max().item()
            max_rel = rel.max().item()
            c = counter_buf.cpu()
            slot_min = min(c[i * 64].item() for i in range(NUM_XCDS))
            slot_max = max(c[i * 64].item() for i in range(NUM_XCDS))
            global_val = c[NUM_XCDS * 64].item()

            ok = max_abs < 5e-2 and max_rel < 5e-2
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(
                f"  G={G:>3} M={M:>7} N={N:>5} trans_b={trans_b!s:<5} "
                f"mode={mode:<13s} L/X={local_per_xcd:>5d} "
                f"max_abs={max_abs:.1e} "
                f"slots[{slot_min}..{slot_max}] global={global_val} [{status}]"
            )

    print("ALL PASS" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
