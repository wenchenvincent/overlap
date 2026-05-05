###############################################################################
# Vendored from primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py
# (Copyright (c) 2025, Advanced Micro Devices, Inc.)
#
# Modifications:
#   - Extracted the per-tile compute body into `_process_grouped_gemm_tile`
#     so the persistent loop can swap between static-stride and work-stealing.
#   - Added `_grouped_bf16_persistent_gemm_kernel_ws` with a WORK_STEAL constexpr
#     selecting either upstream's static stride or a hierarchical
#     per-XCD + global-fallback work-stealing scheme (modeled on tritonBLAS
#     persistent_gemm_work_stealing.py).
###############################################################################

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Match upstream constants.
NUM_XCDS = 8
# Padding for per-XCD atomic counter slots: 64 * 4B = 256 B = one MI355X L2
# line per slot, so the eight XCDs do not false-share a cache line.
COUNTER_STRIDE = 64

_NUM_CUS: Optional[int] = None


def _get_num_cus() -> int:
    global _NUM_CUS
    if _NUM_CUS is None:
        _NUM_CUS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _NUM_CUS


def allocate_ws_counter_buf(device, num_xcds: int = NUM_XCDS) -> torch.Tensor:
    """Allocate the WS counter buffer.

    Layout: [xcd0_slot, ..., xcd{num_xcds-1}_slot, global_slot], each slot
    occupying COUNTER_STRIDE int32 elements.
    """
    return torch.zeros(
        (num_xcds + 1) * COUNTER_STRIDE, dtype=torch.int32, device=device
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Chiplet transform — verbatim from upstream.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _chiplet_transform_chunked(
    pid,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    if pid > (NUM_SMS // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    xcd = pid % NUM_XCDS
    return chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk


# ═══════════════════════════════════════════════════════════════════════════════
# Per-tile compute body — verbatim from upstream lines 144–230, lifted into a
# @triton.jit helper so both the static and work-stealing loops can call it.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _process_grouped_gemm_tile(
    global_tile_id,
    A,
    B,
    C,
    group_offs_ptr,
    G,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    stride_cn,
    num_pid_n,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # ── Find group via linear scan (O(G)) ──
    group_idx: tl.int32 = 0
    tile_start: tl.int32 = 0
    cumsum: tl.int32 = 0
    for _g in range(G):
        m_g_i = (
            tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)
        ).to(tl.int32)
        tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
        new_cumsum = cumsum + tiles_g
        if global_tile_id >= new_cumsum:
            group_idx = _g + 1
            tile_start = new_cumsum
        cumsum = new_cumsum

    # ── Group-local tile → (pid_m, pid_n) with GROUP_SIZE_M swizzle ──
    local_tile = global_tile_id - tile_start
    m_start_g = tl.load(group_offs_ptr + group_idx)
    M_g = (
        tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)
    ).to(tl.int32)
    tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    swizzle_group = local_tile // num_pid_in_group
    first_pid_m = swizzle_group * GROUP_SIZE_M
    group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
    pid_n = (local_tile % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # ── Address computation ──
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    group_offset_b = group_idx.to(tl.int64) * stride_bg

    A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # ── K-loop ──
    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1
    tl.assume(loop_k > 1)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, loop_k):
        if stride_ak == 1:
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
        else:
            a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

        if stride_bk == 1:
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
        else:
            b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_LAST = (
            A
            + m_start_g * stride_am
            + rm[:, None] * stride_am
            + rk_last[None, :] * stride_ak
        )
        B_LAST = (
            B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
        )
        if stride_ak == 1:
            A_LAST = tl.multiple_of(A_LAST, (1, 16))
        else:
            A_LAST = tl.multiple_of(A_LAST, (16, 1))
        if stride_bk == 1:
            B_LAST = tl.multiple_of(B_LAST, (16, 1))
        else:
            B_LAST = tl.multiple_of(B_LAST, (1, 16))
        a = tl.load(
            A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A
        )
        b = tl.load(
            B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B
        )
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # ── Store ──
    c = acc.to(C.type.element_ty)
    rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
    rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
    C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
    tl.store(C_, c, c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Persistent kernel — static-stride OR per-XCD + global-fallback work stealing.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit()
def _grouped_bf16_persistent_gemm_kernel_ws(
    A,
    B,
    C,
    group_offs_ptr,
    tile_counter_ptr,
    global_counter_ptr,
    G,
    N,
    K,
    stride_am,
    stride_bg,
    stride_bn,
    stride_cm,
    stride_cn,
    local_per_xcd,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    WORK_STEAL: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    QUOTA_MODE: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # ── Compute total tiles across all groups (O(G) per CU) ──
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (
            tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)
        ).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    if WORK_STEAL and QUOTA_MODE:
        # ── Fixed per-CU quota with single global counter ──
        # Adapted from scxiao's branch
        # (https://github.com/scxiao/Primus-Turbo/tree/ws_groupgemm).
        #
        # Each CU does exactly ceil(total_tiles / NUM_SMS) atomic_adds — load
        # is static per CU, only the *tile IDs* each CU receives are dynamic
        # (assigned by the global atomic order). Differs from `WS_MODE=global`
        # in this kernel, which uses `while atomic_add < total_tiles` and so
        # lets fast CUs absorb work that slow CUs would otherwise have done.
        # Quota mode pays exactly `total_tiles` atomic_adds total (no
        # over-claim), but loses straggler-tolerance.
        tiles_per_sm = total_tiles // NUM_SMS
        if pid < total_tiles % NUM_SMS:
            tiles_per_sm += 1
        for _ in range(0, tiles_per_sm):
            tile_id = tl.atomic_add(
                global_counter_ptr, 1, sem="relaxed", scope="gpu"
            )
            _process_grouped_gemm_tile(
                tile_id,
                A, B, C, group_offs_ptr,
                G, N, K,
                stride_am, stride_bg, stride_bn, stride_cm, stride_cn,
                num_pid_n,
                stride_ak=stride_ak,
                stride_bk=stride_bk,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                EVEN_K=EVEN_K,
                CACHE_MODIFIER_A=CACHE_MODIFIER_A,
                CACHE_MODIFIER_B=CACHE_MODIFIER_B,
                ALLOW_TF32=ALLOW_TF32,
            )
    elif WORK_STEAL:
        # ── Per-XCD slot + global fallback (hierarchical) ──
        #
        # Phase 1: each XCD owns the contiguous slice
        #   [xcd_id * per_xcd, (xcd_id + 1) * per_xcd)
        # of tile space, with `per_xcd = local_per_xcd` supplied by the host.
        # All 32 CUs of an XCD race on a single padded counter; faster CUs
        # claim more tiles, automatically rebalancing within an XCD when
        # some CUs are slowed by RCCL contention. The contiguous partition
        # preserves L2 locality: consecutive tiles share B[g] data.
        #
        # Phase 2: tiles [per_xcd * NUM_XCDS, total_tiles) are claimed via a
        # single global counter by whichever CU finishes phase 1 first. The
        # host chooses how much work to reserve for phase 2 via the
        # `local_per_xcd` knob: tritonBLAS's tiles-per-CU heuristic picks
        # ~50% of total_tiles for phase 2 on dense workloads (cross-XCD
        # stealing), and 100% phase-1 (no global work) for sparse ones
        # (≤4 tiles/CU) where the global atomic overhead dominates. Special
        # cases:
        #   local_per_xcd = ceil(total_tiles / NUM_XCDS)  → per-XCD only
        #                                                  (phase 2 is empty)
        #   local_per_xcd = 0                             → global only
        #                                                  (phase 1 is empty)
        #   intermediate                                  → hierarchical
        #
        # AMD pid → XCD mapping is round-robin: pid % NUM_XCDS.
        xcd_id = pid % NUM_XCDS
        local_counter = tile_counter_ptr + xcd_id * COUNTER_STRIDE
        per_xcd = local_per_xcd.to(tl.int32)
        phase1_total = (per_xcd * NUM_XCDS).to(tl.int32)

        # Single unified loop with one call site for the per-tile body.
        # We previously used two separate while loops (phase 1 then phase 2),
        # but inlining `_process_grouped_gemm_tile` twice in the same kernel
        # produced NaN in the phase-2 outputs on this kernel + Triton-AMD
        # backend, even with no register spilling reported. Folding both
        # phases into one while loop, with a single helper call, sidesteps
        # the issue: each iteration claims a tile from either the local
        # (phase-1) or global (phase-2) counter and feeds it to the body.
        local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
        in_phase2 = local_idx >= per_xcd
        if in_phase2:
            g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
            tile_id = phase1_total + g_idx
        else:
            tile_id = xcd_id * per_xcd + local_idx

        while tile_id < total_tiles:
            _process_grouped_gemm_tile(
                tile_id,
                A, B, C, group_offs_ptr,
                G, N, K,
                stride_am, stride_bg, stride_bn, stride_cm, stride_cn,
                num_pid_n,
                stride_ak=stride_ak,
                stride_bk=stride_bk,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                EVEN_K=EVEN_K,
                CACHE_MODIFIER_A=CACHE_MODIFIER_A,
                CACHE_MODIFIER_B=CACHE_MODIFIER_B,
                ALLOW_TF32=ALLOW_TF32,
            )
            if in_phase2:
                g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
                tile_id = phase1_total + g_idx
            else:
                local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
                if local_idx >= per_xcd:
                    in_phase2 = True
                    g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
                    tile_id = phase1_total + g_idx
                else:
                    tile_id = xcd_id * per_xcd + local_idx
    else:
        # Static-stride persistent loop (matches upstream behaviour).
        if NUM_XCDS != 1:
            pid_static = _chiplet_transform_chunked(
                pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE
            )
        else:
            pid_static = pid
        for global_tile_id in range(pid_static, total_tiles, NUM_SMS):
            _process_grouped_gemm_tile(
                global_tile_id,
                A, B, C, group_offs_ptr,
                G, N, K,
                stride_am, stride_bg, stride_bn, stride_cm, stride_cn,
                num_pid_n,
                stride_ak=stride_ak,
                stride_bk=stride_bk,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                EVEN_K=EVEN_K,
                CACHE_MODIFIER_A=CACHE_MODIFIER_A,
                CACHE_MODIFIER_B=CACHE_MODIFIER_B,
                ALLOW_TF32=ALLOW_TF32,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Autotune configs and wrapper.
#
# AMD-specific knobs (waves_per_eu, matrix_instr_nonkdim, kpack) live in the
# Config kwargs dict — Triton's AMD backend reads them at compile time even
# though they are not declared as kernel constexprs. This matches the pattern
# used by aiter (see flash_attn_triton_amd/fwd_prefill.py) and by primus's
# autotuned MoE kernels.
#
# Cache key: (G, N, K). Different M_total share a config; different (G,N,K)
# trigger a fresh search. WORK_STEAL is a constexpr → triton specializes per
# value, so static-stride and WS modes are autotuned independently.
# ═══════════════════════════════════════════════════════════════════════════════


def _make_config(bm, bn, bk, num_warps=8, num_stages=2,
                 waves_per_eu=2, matrix_instr_nonkdim=16, kpack=1):
    return triton.Config(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "kpack": kpack,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )


_AUTOTUNE_CONFIGS = [
    # Upstream default.
    _make_config(256, 256, 64),
    # Tile-size variants.
    _make_config(128, 256, 64),
    _make_config(256, 128, 64),
    _make_config(128, 128, 64, num_warps=4),
    _make_config(64, 256, 64, num_warps=4),
    _make_config(256, 64, 64, num_warps=4),
    # BLOCK_K sweeps.
    _make_config(256, 256, 32),
    _make_config(256, 256, 128),
    _make_config(128, 256, 128),
    # Stage / waves / kpack variants on the default tile.
    _make_config(256, 256, 64, num_stages=1),
    _make_config(256, 256, 64, num_stages=3),
    _make_config(256, 256, 64, waves_per_eu=0),
    _make_config(256, 256, 64, waves_per_eu=3),
    _make_config(256, 256, 64, kpack=2),
    _make_config(256, 256, 64, matrix_instr_nonkdim=32),
]


_grouped_bf16_persistent_gemm_kernel_ws_autotune = triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["G", "N", "K"],
    # Phase-1/-2 counters must be cleared between autotune trials, otherwise
    # the second trial sees stale indices >= per_xcd and runs zero tiles
    # (would record a bogus near-zero time and "win"). The wrapper still
    # zeroes the buffer once before the user-visible launch — this keeps the
    # autotune trials honest.
    reset_to_zero=["tile_counter_ptr", "global_counter_ptr"],
)(_grouped_bf16_persistent_gemm_kernel_ws)


# ═══════════════════════════════════════════════════════════════════════════════
# Mode-selection helpers (host-side).
#
# tritonBLAS's `hierarchical_split()` (defined but currently unused in their
# matmul.py) picks a fraction of total_tiles to reserve for the global counter
# based on tiles-per-CU density. We adopt the same heuristic so phase 2 has
# real work to do — v1 with `per_xcd = total_tiles // NUM_XCDS` left phase 2
# with only the 0–7 ragged-tail tiles, defeating cross-XCD stealing.
# ═══════════════════════════════════════════════════════════════════════════════


def compute_total_tiles_host(group_lens, N: int,
                             block_m: int = 256, block_n: int = 256) -> int:
    """Compute total persistent tiles for the kernel, host-side.

    Args:
        group_lens: per-group row counts. Either a Python sequence of ints or
            a 1-D torch tensor (CPU or CUDA — will be moved to CPU once).
        N: output column count.
        block_m, block_n: must match the kernel's BLOCK_SIZE_M / BLOCK_SIZE_N.

    Returns:
        sum_g( ceil(M_g / block_m) ) * ceil(N / block_n)
    """
    if isinstance(group_lens, torch.Tensor):
        group_lens = group_lens.cpu().tolist()
    num_pid_n = (N + block_n - 1) // block_n
    return sum((m_g + block_m - 1) // block_m for m_g in group_lens) * num_pid_n


def resolve_local_per_xcd(total_tiles: int, num_sms: int, num_xcds: int,
                          ws_mode: str = "auto") -> int:
    """Map `ws_mode` + workload size to the per-XCD phase-1 tile budget.

    Modes:
        "per-xcd" — phase 1 covers everything; phase 2 is empty.
                    local_per_xcd = ceil(total_tiles / num_xcds).
        "global"  — phase 1 is empty; all work via the global counter.
                    local_per_xcd = 0.
        "hierarchical" — apply tritonBLAS's adaptive split unconditionally.
        "quota"   — fixed per-CU quota loop (scxiao's variant). The kernel
                    ignores local_per_xcd in this mode; we return 0 for
                    consistency.
        "auto" (default) — apply tritonBLAS's heuristic with the
                    `tiles_per_cu <= 4` short-circuit to "per-xcd"
                    (matches their stated reasoning that the global atomic
                    overhead dominates when work is sparse).

    Heuristic (from tritonBLAS hierarchical_split, lifted verbatim):
        tiles_per_cu = total_tiles / num_sms
        local_frac = max(0.5, 1.0 - max(0, tiles_per_cu - 4) * 0.05)
        local_per_xcd = max(1, int(total_tiles * local_frac) // num_xcds)
    """
    if ws_mode in ("global", "quota"):
        return 0
    if ws_mode == "per-xcd":
        return (total_tiles + num_xcds - 1) // num_xcds

    tiles_per_cu = total_tiles / max(num_sms, 1)
    if ws_mode == "auto" and tiles_per_cu <= 4.0:
        # Below the threshold: per-XCD only. Global atomic overhead would
        # dominate when there's not enough work to amortise it.
        return (total_tiles + num_xcds - 1) // num_xcds

    # auto (dense) or explicit "hierarchical": adaptive split.
    local_frac = max(0.5, 1.0 - max(0.0, tiles_per_cu - 4.0) * 0.05)
    return max(1, int(total_tiles * local_frac) // num_xcds)


# ═══════════════════════════════════════════════════════════════════════════════
# Host wrapper — same shape as upstream `grouped_gemm_triton_kernel` but takes
# a counter buffer (allocated by the caller, zeroed on the active stream here).
# ═══════════════════════════════════════════════════════════════════════════════


def grouped_gemm_triton_kernel_ws(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    counter_buf: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
    num_xcds: int = NUM_XCDS,
    trans_b: bool = False,
    work_steal: bool = True,
    autotune: bool = False,
    ws_mode: str = "auto",
    total_tiles: Optional[int] = None,
) -> torch.Tensor:
    """Persistent grouped GEMM with optional per-XCD + global work stealing.

    Args:
        a: [M_total, K] BF16/FP16.
        b: [G, K, N] (or [G, N, K] when trans_b=True) BF16/FP16.
        group_offs: [G+1] int64 prefix sum of group lengths.
        counter_buf: int32 buffer of length (num_xcds + 1) * COUNTER_STRIDE.
            The wrapper zeros it on the active stream before launch.
            Allocate once per process via `allocate_ws_counter_buf`.
        out: optional preallocated [M_total, N] output. Reused across calls
            to avoid per-call allocator overhead in benchmarks.
        num_sms: persistent grid size. Defaults to all CUs.
        num_xcds: chiplet count (8 on MI355X / MI350).
        trans_b: True when b is laid out as [G, N, K].
        work_steal: when False, falls back to the upstream static-stride loop.
        autotune: when True, the kernel is launched through @triton.autotune
            over `_AUTOTUNE_CONFIGS`. First call for a given (G, N, K) triggers
            a search (~10–30 s wall, depending on config count); subsequent
            calls reuse the cached winner.
        ws_mode: one of "auto" (default, tritonBLAS heuristic), "per-xcd"
            (phase 2 empty), "global" (phase 1 empty), or "hierarchical"
            (always apply the adaptive split). Ignored when `work_steal=False`.
        total_tiles: optional precomputed total persistent tile count; saves
            a tiny CPU-side reduction over `group_offs`. Caller may use
            `compute_total_tiles_host()` once at startup and pass it here.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in (torch.bfloat16, torch.float16)
    assert b.dtype in (torch.bfloat16, torch.float16)

    M_total, K_a = a.shape
    G = b.shape[0]

    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)
        stride_bn = b.stride(1)
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)
        stride_bn = b.stride(2)

    assert K_a == K_b
    K = K_a

    if num_sms is None:
        num_sms = _get_num_cus()

    expected = (num_xcds + 1) * COUNTER_STRIDE
    assert (
        counter_buf.dtype == torch.int32
        and counter_buf.numel() == expected
        and counter_buf.is_cuda
    ), (
        f"counter_buf must be int32, length {expected}, on CUDA; "
        f"got dtype={counter_buf.dtype}, numel={counter_buf.numel()}"
    )

    if out is None:
        out = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

    # Reset the WS counters on the active stream. Cheap (single async memset)
    # and correctness-critical: skipping it leaves stale per-xcd indices that
    # immediately exceed `per_xcd`, so the kernel does zero work.
    if work_steal:
        counter_buf.zero_()

    tile_counter_ptr = counter_buf
    global_counter_ptr = counter_buf[num_xcds * COUNTER_STRIDE :]

    # Resolve phase-1 size on the host. When the kernel runs with
    # WORK_STEAL=False the kernel ignores `local_per_xcd`; we still pass a
    # sensible value (0) to keep the launch site uniform.
    if work_steal:
        if total_tiles is None:
            # Derive group lengths from the prefix sum.
            offs = group_offs.cpu().tolist()
            group_lens = [offs[g + 1] - offs[g] for g in range(len(offs) - 1)]
            total_tiles = compute_total_tiles_host(group_lens, N)
        local_per_xcd = resolve_local_per_xcd(
            total_tiles, num_sms, num_xcds, ws_mode
        )
    else:
        local_per_xcd = 0

    even_k = K % 64 == 0

    common_kwargs = dict(
        stride_ak=a.stride(1),
        stride_bk=stride_bk,
        GROUP_SIZE_M=4,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        CHUNK_SIZE=32,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=".ca",
        CACHE_MODIFIER_B=".ca",
        WORK_STEAL=work_steal,
        COUNTER_STRIDE=COUNTER_STRIDE,
        QUOTA_MODE=(work_steal and ws_mode == "quota"),
    )

    if autotune:
        # BLOCK_SIZE_*, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim,
        # and kpack are supplied by the autotuner from the chosen Config.
        kernel = _grouped_bf16_persistent_gemm_kernel_ws_autotune
        extra = {}
    else:
        kernel = _grouped_bf16_persistent_gemm_kernel_ws
        extra = dict(
            BLOCK_SIZE_M=256,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            num_warps=8,
            num_stages=2,
            waves_per_eu=2,
            matrix_instr_nonkdim=16,
            kpack=1,
        )

    kernel[(num_sms,)](
        a, b, out, group_offs,
        tile_counter_ptr, global_counter_ptr,
        G, N, K,
        a.stride(0), b.stride(0), stride_bn,
        out.stride(0), out.stride(1),
        local_per_xcd,
        **common_kwargs,
        **extra,
    )
    return out
