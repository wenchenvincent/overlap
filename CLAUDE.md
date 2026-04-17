# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

Reproduces and characterizes a performance issue on AMD Instinct GPUs (ROCm) where RCCL communication collectives overlapped with grouped GEMM kernels cause the GEMM to slow down. The root causes are memory bandwidth contention and CU resource competition.

## Running the Scripts

All scripts require multi-GPU (8x MI355X or MI350). The system needs ROCm, PyTorch with RCCL backend, Triton, and Primus-Turbo installed.

### Main reproducer
```bash
# Triton backend
torchrun --nproc_per_node=8 bench_overlap.py --G 8 --M 16384 --K 8192 --N 8192 --ag-size-mb 512 --grid-dims 256

# Primus-Turbo CK backend with real workload shapes
torchrun --nproc_per_node=8 bench_overlap.py --backend primus --trans-b --G 32 --M 267424 --K 1280 --N 2560 --ag-size-mb 512 --grid-dims 240

# Control RCCL channel count (must be env var, read at process init)
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py ...
```

### Colleague's three-phase test
```bash
# NUM_CU and NCCL_MAX_NCHANNELS are env vars, not CLI args
NCCL_MAX_NCHANNELS=16 NUM_CU=240 TEST_OVERLAP_TRACE_DIR=./traces python3 test_overlap.py
```
This uses `mp.spawn` internally (not torchrun). Trace output goes to `TEST_OVERLAP_TRACE_DIR` if set.

### RCCL channel sweep
```bash
bash run.sh --nproc 8
```

### Single-GPU CU occupancy experiment (no distributed)
```bash
python3 bench_cu_occupancy.py --occupy-cus 0,32,64,112 --blocks-per-cu 1,3
```

## Architecture

There are two grouped GEMM backends controlled by `--backend`:
- **triton**: Self-contained persistent Triton kernel in `bench_overlap.py` (lines 25–146). Inlines XCD remapping from AITER. `--grid-dims` controls CU count directly as `GRID_DIM` constexpr.
- **primus**: Primus-Turbo's Composable Kernel (CK) backend via `primus_turbo.pytorch.ops.grouped_gemm`. `--grid-dims` maps to CK's `num_cu` parameter. Supports `--trans-b` for `[G, N, K]` weight layout.

`grouped_gemm_ops.py` is a wrapper around Primus-Turbo's lower-level `grouped_gemm_impl` used by `test_overlap.py`. It adds a `num_cu` passthrough to `grouped_gemm_fprop`.

The global `run_grouped_gemm` function pointer in `bench_overlap.py` is set in `main()` based on `--backend` and referenced by all benchmark functions.

## Key Constraints

- `NCCL_MAX_NCHANNELS` is an **env var read at RCCL init time** — cannot be changed mid-process. This is why `run.sh` launches separate `torchrun` invocations per channel config.
- Primus-Turbo requires `group_lens` as **int64**, while the Triton kernel uses **int32**. The dtype is selected by `--backend` in `bench_overlap.py`.
- The Triton kernel requires K divisible by 64 (`BLOCK_K`).
- MI355X has 256 CUs and 8 XCDs. MI350 may differ — adjust `--grid-dims` and `--num-xcds` accordingly. Use `rocminfo | grep "Compute Unit"` to check.

## Profiling

```bash
# rocprofv3 system trace (shows actual GPU overlap, RCCL kernel resources)
rocprofv3 --sys-trace --rccl-trace -f csv -d ./traces -- torchrun --nproc_per_node=8 bench_overlap.py ...

# Key fields in kernel_trace.csv: Grid_Size_X, Workgroup_Size_X (blocks = Grid/WG), VGPR_Count, LDS_Block_Size
# NCCL_DEBUG=INFO shows channel count and algorithm selection
```

## Important Numbers (MI355X baseline)

- RCCL kernel: 140 VGPRs (alloc 144), 19968B LDS, 256 threads/block. Default 112 channels = 112 blocks on 8 GPUs.
- CK grouped GEMM: 256 VGPRs, 65536B LDS (full 64KB), 256 blocks. Max 1 block per CU, no room for co-scheduling.
- CU partitioning sweet spot: GEMM=240 CUs + RCCL=16 channels eliminates slowdown with ~3% baseline cost.
