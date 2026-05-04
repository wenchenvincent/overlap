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

    if WORK_STEAL:
        # ── Per-XCD slot + global fallback (hierarchical) ──
        #
        # Phase 1: each XCD owns the contiguous slice
        #   [xcd_id * per_xcd, (xcd_id + 1) * per_xcd)
        # of tile space, with `per_xcd = total_tiles // NUM_XCDS` (floor).
        # All 32 CUs of an XCD race on a single padded counter; faster CUs
        # claim more tiles, automatically rebalancing within an XCD when
        # some CUs are slowed by RCCL contention. The contiguous partition
        # preserves L2 locality: consecutive tiles share B[g] data.
        #
        # Phase 2: tiles [per_xcd * NUM_XCDS, total_tiles) (the ragged tail)
        # are claimed via a single global counter by whichever CU finishes
        # phase 1 first. This also handles the total_tiles < NUM_XCDS case
        # where phase 1 has zero work per XCD.
        #
        # AMD pid → XCD mapping is round-robin: pid % NUM_XCDS.
        xcd_id = pid % NUM_XCDS
        local_counter = tile_counter_ptr + xcd_id * COUNTER_STRIDE
        per_xcd = total_tiles // NUM_XCDS
        phase1_total = per_xcd * NUM_XCDS

        # Phase 1: drain own XCD slice. Triton does not support `break`, so
        # the atomic_add is hoisted into the while-condition: each iteration
        # claims a tile only if local_idx < per_xcd.
        local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")
        while local_idx < per_xcd:
            global_tile_id = xcd_id * per_xcd + local_idx
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
            local_idx = tl.atomic_add(local_counter, 1, sem="relaxed", scope="gpu")

        # Phase 2: global fallback for the ragged tail
        # [phase1_total, total_tiles).
        g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
        global_tile_id = phase1_total + g_idx
        while global_tile_id < total_tiles:
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
            g_idx = tl.atomic_add(global_counter_ptr, 1, sem="relaxed", scope="gpu")
            global_tile_id = phase1_total + g_idx
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
        **common_kwargs,
        **extra,
    )
    return out
