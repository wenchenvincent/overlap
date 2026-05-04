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
    grouped_gemm_triton_kernel_ws,
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
    all_ok = True
    for G, M, K, N, trans_b in cases:
        a, b, offs = _make_inputs(G, M, K, N, trans_b, device)
        out_static = _run_static(a, b, offs, num_sms, NUM_XCDS, trans_b)
        out_ws = grouped_gemm_triton_kernel_ws(
            a, b, offs, counter_buf,
            num_sms=num_sms,
            num_xcds=NUM_XCDS,
            trans_b=trans_b,
            work_steal=True,
        )
        torch.cuda.synchronize()

        # bf16 → fp32 for comparison; tolerances match a single bf16 ulp at scale.
        diff = (out_static.float() - out_ws.float()).abs()
        rel = diff / out_static.float().abs().clamp_min(1.0)
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        # Counter sanity: phase-1 slots should each hit at least per_xcd; the
        # global slot is non-zero only when contention forced phase 2.
        c = counter_buf.cpu()
        slot_vals = [c[i * 64].item() for i in range(NUM_XCDS)]
        global_val = c[NUM_XCDS * 64].item()

        ok = max_abs < 5e-2 and max_rel < 5e-2
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  G={G:>3} M={M:>7} K={K:>5} N={N:>5} trans_b={trans_b!s:<5} "
            f"max_abs={max_abs:.2e} max_rel={max_rel:.2e} "
            f"slots={slot_vals} global={global_val} [{status}]"
        )

    print("ALL PASS" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
