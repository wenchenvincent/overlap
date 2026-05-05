"""Vendored fork of primus-turbo grouped-GEMM Triton kernel with work stealing.

Source: primus_turbo/triton/grouped_gemm/grouped_gemm_kernel.py
Modifications: added per-XCD + global-fallback work-stealing path selectable
via the `WORK_STEAL: tl.constexpr` branch.
"""

from .grouped_gemm_kernel_ws import (
    NUM_XCDS,
    COUNTER_STRIDE,
    _grouped_bf16_persistent_gemm_kernel_ws,
    grouped_gemm_triton_kernel_ws,
    allocate_ws_counter_buf,
    compute_total_tiles_host,
    resolve_local_per_xcd,
)

__all__ = [
    "NUM_XCDS",
    "COUNTER_STRIDE",
    "_grouped_bf16_persistent_gemm_kernel_ws",
    "grouped_gemm_triton_kernel_ws",
    "allocate_ws_counter_buf",
    "compute_total_tiles_host",
    "resolve_local_per_xcd",
]
