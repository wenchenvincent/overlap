# RCCL + Grouped GEMM Overlap Slowdown Reproducer

Reproduces and characterizes a performance issue on AMD Instinct GPUs (ROCm) where
RCCL communication collectives overlapped with grouped GEMM kernels cause the GEMM
to slow down significantly.

## Quick Start

```bash
# Basic overlap test (8 GPUs, Primus-Turbo CK backend, real workload shapes)
torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 256

# CU partitioning (240 gemm + 16 RCCL) to eliminate slowdown
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 240

# Colleague's test script
NCCL_MAX_NCHANNELS=16 NUM_CU=240 TEST_OVERLAP_TRACE_DIR=./traces python3 test_overlap.py
```

## Files

| File | Description |
|---|---|
| `bench_overlap.py` | Main reproducer. Triton or Primus-Turbo CK grouped GEMM + RCCL all-gather overlap benchmark with CUDA event timing. |
| `test_overlap.py` | Three-phase test (AG only, GEMM only, overlapped) with PyTorch profiler trace export and multi-rank trace merging. |
| `grouped_gemm_ops.py` | Primus-Turbo grouped GEMM wrapper (fprop/dgrad/wgrad) with `num_cu` parameter. |
| `bench_cu_occupancy.py` | Controlled experiment: synthetic CU occupier kernel + grouped GEMM to measure slowdown as a function of stolen CUs. |
| `bench_collectives.py` | Compares CU resource usage across all-gather, reduce-scatter, and all-reduce. |
| `run.sh` | Orchestrates `NCCL_MAX_NCHANNELS` sweep across multiple `torchrun` invocations. |
| `check_transb_resources.py` | Utility to compare CU resources between `trans_b=False` and `trans_b=True` kernel paths. |

## Key Findings

All measurements below are from 8x MI355X GPUs (256 CUs, ROCm 7.1).

### 1. Overlap causes 1.5x–1.9x GEMM slowdown

When an RCCL all-gather runs concurrently with a Primus-Turbo CK grouped GEMM on
separate HW queues, the GEMM slows down significantly:

| Config | GEMM alone | GEMM overlapped | Slowdown |
|---|---|---|---|
| Real workload (G=32, M=267K, K=1280, N=2560, trans_b=True) | 2.0ms | 3.4ms | **1.71x** |
| Larger shapes (G=8, M=16K, K=8192, N=8192, trans_b=False) | 2.7ms | 5.1ms | **1.87x** |
| Triton kernel (G=8, M=16K, K=8192, N=8192) | 4.2ms | 5.5ms | **1.31x** |

### 2. CU partitioning eliminates the slowdown

Limiting the GEMM to 240 CUs and RCCL to 16 channels (16 CUs) removes contention:

| Config | GEMM alone | GEMM overlapped | Slowdown |
|---|---|---|---|
| num_cu=256, RCCL=16ch | 2.0ms | 3.4ms | **1.71x** |
| **num_cu=240, RCCL=16ch** | 2.0ms | 2.0ms | **0.97x** |

The baseline cost of using 240 instead of 256 CUs is only ~3%.

### 3. RCCL uses a single generic kernel for all collectives

All-gather, reduce-scatter, and all-reduce launch the same `ncclDevKernel_Generic_1`
kernel with identical CU resources:

| Resource | Value |
|---|---|
| Blocks | 112 (default, 8 GPUs) or set via `NCCL_MAX_NCHANNELS` |
| VGPRs per wave | 140 (allocated as 144) |
| LDS per block | 19,968 bytes |
| Threads per block | 256 (4 waves) |

### 4. CK grouped GEMM fully occupies each CU

The Primus-Turbo CK grouped GEMM kernel uses heavy resources, making co-scheduling
with RCCL on the same CU impossible:

| Resource | CK GEMM | RCCL |
|---|---|---|
| VGPRs per wave | **256** | 140 |
| LDS per block | **65,536 bytes** (full 64KB) | 19,968 bytes |
| Max blocks per CU | **1** | 3 |

With 256 VGPRs, only 2 waves per SIMD can fit (512/256). With 64KB LDS, only 1 block
per CU. There is zero room for an RCCL wave to co-schedule.

### 5. RCCL CUs are mostly idle but still cause contention

HW counter profiling (`SQ_BUSY_CU_CYCLES`) shows RCCL's 112 blocks only keep ~13 CUs
busy on average (out of 112 occupied). The waves are mostly in wait states (waiting on
network/DMA). Despite low compute utilization, they:
- Occupy wave slots and VGPRs, preventing GEMM from scheduling additional waves
- Generate memory bandwidth pressure when they periodically wake up to copy data

### 6. Memory bandwidth is the primary contention source

Evidence:
- Slowdown is roughly constant (1.2x–1.3x) across 4–112 RCCL channels with the
  Triton kernel, despite very different CU counts
- `trans_b=True` and `trans_b=False` use identical kernel resources (same VGPR, LDS,
  blocks) but show different slowdowns (1.54x vs 1.87x) due to different memory
  access patterns
- The synthetic CU occupier experiment shows that CU exclusion alone (without memory
  traffic) produces less slowdown than real RCCL overlap

## Environment

Tested on:
- 8x AMD Instinct MI355X (256 CUs, 8 XCDs)
- ROCm 7.1, PyTorch 2.10.0.dev+rocm7.1, Triton 3.4.0
- RCCL 1.0.70100

## bench_overlap.py CLI Reference

```
--G              Number of expert groups (default: 8)
--M              Total tokens/rows (default: 4096)
--K              Input dimension (default: 4096)
--N              Output dimension (default: 4096)
--ag-size-mb     All-gather tensor size in MB (default: 64)
--grid-dims      Comma-separated GRID_DIM/num_cu values to sweep (default: "256")
--num-xcds       XCD count for Triton PID remapping (default: 8)
--warmup         Warmup iterations (default: 5)
--iters          Measurement iterations (default: 20)
--profile        Enable PyTorch profiler trace export
--num-ag         Number of concurrent all-gathers (default: 1)
--backend        "triton" or "primus" (default: "triton")
--trans-b        Use transposed weight layout [G, N, K] (Primus only)
```

## Profiling

### PyTorch profiler (built-in)

```bash
torchrun --nproc_per_node=8 bench_overlap.py --profile ...
# Traces saved to ./traces/rank{N}_grid{D}/
```

### rocprofv3 (system-level, shows RCCL internals)

```bash
rocprofv3 --sys-trace --rccl-trace -f pftrace -d ./traces -- \
  torchrun --nproc_per_node=8 bench_overlap.py ...
# View in chrome://tracing or https://ui.perfetto.dev
```

### RCCL debug logging

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=8 bench_overlap.py ...
# Shows channel count, algorithm, protocol per collective
```
