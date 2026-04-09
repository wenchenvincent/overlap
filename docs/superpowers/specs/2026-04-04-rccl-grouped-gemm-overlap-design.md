# RCCL + Triton Grouped GEMM Overlap Slowdown Reproducer

## Problem

On ROCm (MI355X), when RCCL all-gather collectives overlap with a Triton grouped gemm kernel on the compute stream, the grouped gemm slows down. This reproducer quantifies the effect and sweeps CU allocation parameters to characterize the interaction.

## Architecture

Two files:

- **`bench_overlap.py`** — Self-contained Python script with Triton kernel, PyTorch distributed setup, and benchmarking harness. Handles grid-dim sweeps within a single process.
- **`run.sh`** — Shell script that orchestrates multiple `torchrun` invocations to sweep `NCCL_MAX_NCHANNELS` (requires separate process init per value).

### Why two files

`NCCL_MAX_NCHANNELS` is an environment variable read at RCCL initialization. Changing it requires a new process group, which is fragile to do within a single process. Separate `torchrun` invocations are the clean approach.

## Triton Grouped GEMM Kernel

### Design

Persistent kernel based on AITER's `gmm_kernel`. Self-contained — no AITER imports.

**Inlined JIT helpers** (from AITER `pid_preprocessing.py`):
- `remap_xcd(pid, GRID_MN, NUM_XCDS=8)` — Redistributes program IDs across 8 XCDs. Adjacent PIDs land on different XCDs, improving L2 utilization.
- `pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=1)` — Maps 1D PID to 2D tile coordinates.

**Kernel**: `grouped_gemm_kernel(lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr, M, K, N, G, ...constexprs)`
- Persistent: `GRID_DIM` programs, each loops `tile += GRID_DIM`
- Outer loop over G groups, inner while-loop over tiles of current group
- Standard tiled matmul: BLOCK_K strips, fp32 accumulation, bf16 store
- `input_precision="ieee"`

**Defaults**: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GRID_DIM=256, NUM_XCDS=8, GROUP_SIZE=1

**Python wrapper**: Computes total tiles, launches with `grid=(min(GRID_DIM, total_tiles),)`

### Simplifications vs AITER
- No TRANS_RHS (always row-major rhs)
- No device_assert
- No config file loading (hardcoded block sizes)
- K guaranteed divisible by BLOCK_K

## Benchmarking Harness

### Setup
- `torch.distributed.init_process_group(backend="nccl")` (maps to RCCL)
- Per-rank `torch.cuda.set_device(local_rank)`

### Tensors
| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| lhs | (M, K) | bf16 | Token activations |
| rhs | (G, K, N) | bf16 | Expert weights |
| group_sizes | (G,) | int32 | On GPU, sums to M |
| out | (M, N) | bf16 | Output buffer |
| ag_input | (ag_numel,) | bf16 | All-gather source |
| ag_output | (world_size * ag_numel,) | bf16 | All-gather dest |

**Defaults**: G=8, M=4096, K=4096, N=4096, ag_size=64MB

### Scenarios (per grid-dim value)

1. **Gemm only** — Grouped gemm on default stream, measure time
2. **Sequential** — All-gather (sync) then grouped gemm, measure gemm time
3. **Overlap** — All-gather async on comm_stream, grouped gemm on default stream concurrently, measure gemm time

### Stream management for overlap
- `comm_stream = torch.cuda.Stream()` — dedicated for RCCL
- All-gather launched with `async_op=True` inside `with torch.cuda.stream(comm_stream)`
- Gemm launched on default stream immediately after
- **No `wait_stream`** between streams — they must run concurrently
- CUDA timing events recorded on default (compute) stream only

### Timing
- CUDA events for per-kernel timing
- Wall-clock measurement of full overlap scenario for verification
- Warmup: 5 iterations per scenario (including overlap pattern)
- Measurement: 20 iterations
- `torch.cuda.synchronize()` + `dist.barrier()` between iterations

### Overlap verification (3 levels)

**A) Timing-based (always on)**:
Measures wall-clock of overlap scenario. If `wall < gemm_alone + allgather_alone`, overlap is confirmed.

**B) PyTorch profiler (`--profile` flag)**:
Wraps iterations in `torch.profiler.profile(activities=[CPU, CUDA])`. Exports Chrome trace JSON per rank. View in `chrome://tracing` or Perfetto.

**C) rocprofv3 (documented in run.sh)**:
```bash
rocprofv3 --sys-trace --rccl-trace -f pftrace -d traces -- torchrun ...
```

## Sweep Matrix (run.sh)

`NCCL_MAX_NCHANNELS` controls RCCL channel count (not CU count directly). More channels = more thread blocks = more CU pressure. All runs include `NCCL_DEBUG=INFO` to log actual RCCL resource allocation.

| Scenario | NCCL_MAX_NCHANNELS | --grid-dims | Purpose |
|----------|-------------------|-------------|---------|
| 1. Default | unset | 256 | Baseline behavior |
| 2. Sweep RCCL, full gemm | 4, 8, 16, 32 | 256 | Isolate RCCL channel impact |
| 3. Sweep RCCL, reduced gemm | 4, 8, 16, 32 | 224, 192, 160, 128 (configurable) | Complementary CU partitioning (grid-dims are estimates; use NCCL_DEBUG to correlate) |

## CLI Arguments (bench_overlap.py)

| Flag | Default | Description |
|------|---------|-------------|
| --G | 8 | Number of expert groups |
| --M | 4096 | Total tokens |
| --K | 4096 | Hidden dimension |
| --N | 4096 | Output dimension |
| --ag-size-mb | 64 | All-gather tensor size (MB) |
| --grid-dims | "256" | Comma-separated GRID_DIM values |
| --num-xcds | 8 | XCD count for PID remapping |
| --warmup | 5 | Warmup iterations |
| --iters | 20 | Measurement iterations |
| --profile | false | Enable PyTorch profiler trace |

## Output Format

### Human-readable (rank 0, stderr)
```
=== GRID_DIM=256, NCCL_MAX_NCHANNELS=default ===
Scenario          | Mean (ms) | Min (ms) | Max (ms) | Slowdown
------------------+-----------+----------+----------+---------
Gemm only         |   X.XXX   |  X.XXX   |  X.XXX   |  1.00x
Sequential        |   X.XXX   |  X.XXX   |  X.XXX   |  X.XXx
Overlap           |   X.XXX   |  X.XXX   |  X.XXX   |  X.XXx
Overlap wall      |   X.XXX   |          |          | (< sum => overlapped)
```

### CSV (rank 0, stdout for run.sh aggregation)
```
grid_dim,nccl_max_nchannels,scenario,mean_ms,min_ms,max_ms
```

## Verification

1. `bash run.sh --nproc 2` completes without errors
2. "Overlap" shows gemm slowdown > 1.0x vs "Gemm only"
3. Wall-clock confirms overlap: `wall < gemm + allgather`
4. `--profile` trace shows concurrent kernels on separate streams
5. Scenario 3 behavior differs from scenario 2
6. NCCL_DEBUG output reveals RCCL resource usage per channel config
