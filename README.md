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
| `att_overlap.json` | `rocprofv3` ATT config: kernel-filter to `GroupedGemm`, full 32-SE coverage, 4 consecutive captures. See [ATT tracing](#per-block-dispatch-tracing-with-att). |
| `run_att.sh` | Rank-0-only wrapper for `rocprofv3 --att`. Other ranks run plain `python`. Used with `torchrun --no-python`. |
| `quick_analysis.py` | Reads `occupancy.json` from an ATT trace and prints: distinct CUs used, oversubscribed CUs (the dispatcher-bug indicator), late-dispatching blocks. |

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

## Per-block dispatch tracing with ATT

Traditional `rocprofv3 --sys-trace` and PMC counters give per-kernel timing and
aggregate counter values, but they cannot tell you **which specific CU each
GEMM block ran on** or **when each block started executing**. That information
is what's needed to see the dispatcher-placement bug behind the overlap
slowdown — where on certain ROCm builds 1–2 GEMM blocks land on a CU that
already has a block, while 30 other CUs sit idle, doubling the kernel wallclock.

ROCm's **Architecture Thread Trace (ATT)** is the only standard tool that
records per-wave start/end events at cycle granularity, per (SE, CU, SIMD).
The three files in this repo wrap rocprofv3's ATT for this specific
investigation.

### Prerequisites

ATT requires the closed-source decoder library. On Ubuntu 22.04:

```bash
cd /tmp
URL=$(curl -s https://api.github.com/repos/ROCm/rocprof-trace-decoder/releases/tags/0.1.6 \
  | python3 -c 'import sys,json; data=json.load(sys.stdin); [print(a["browser_download_url"]) for a in data.get("assets",[]) if "ubuntu-22.04" in a["name"] and a["name"].endswith(".deb")]')
curl -sL -o decoder.deb "$URL"
sudo dpkg -i decoder.deb
ls /opt/rocm/lib/librocprof-trace-decoder.so   # verify
```

### How `att_overlap.json` is configured

```json
{
    "jobs": [{
        "kernel_include_regex": "GroupedGemm",
        "advanced_thread_trace": true,
        "att_target_cu": 0,
        "att_shader_engine_mask": "0xFFFFFFFF",
        "att_simd_select": "0xF",
        "att_buffer_size": "0x60000000",
        "att_consecutive_kernels": 4
    }]
}
```

| Knob | Meaning |
|---|---|
| `kernel_include_regex: GroupedGemm` | Only trace the CK grouped-GEMM kernel. RCCL waves running on the same CUs are still captured (they show up as 100M+-cycle "long" waves). |
| `att_shader_engine_mask: 0xFFFFFFFF` | All 32 SEs (4 SEs × 8 XCDs) — full GPU coverage. Each hex digit is one XCD, each bit is one SE within that XCD. |
| `att_target_cu: 0` | Capture starting from CU 0; despite the name this still records all CUs of the enabled SEs (each with up to ~8 CUs). |
| `att_simd_select: 0xF` | All 4 SIMDs per CU. |
| `att_buffer_size: 0x60000000` | 1.5 GB ATT buffer per SE. Sized to capture multiple consecutive GEMM kernels. |
| `att_consecutive_kernels: 4` | Capture 4 successive GroupedGemm dispatches. The slowdown is bimodal per iter (~2 fast + ~3 slow per 5 iters), so capturing several lets you compare. |

### How `run_att.sh` is structured

`torchrun` spawns 8 worker processes; we only want one of them traced (rank 0).
The wrapper checks `LOCAL_RANK` and either runs `rocprofv3 --att` or plain
`python`. By default it reads `att_overlap.json` from the script's own
directory; you can override with the `ATT_CONFIG` env var.

| Env var | Default | Purpose |
|---|---|---|
| `ATT_CONFIG` | `$(dirname run_att.sh)/att_overlap.json` | Path to the rocprofv3 ATT JSON config. |
| `TRACE_DIR` | `/tmp/att_out` | Where to write trace output. |
| `TRACE_TAG` | `trace` | File-name prefix for output. |

### Running an ATT capture

```bash
rm -rf /tmp/att_out

TRACE_DIR=/tmp/att_out TRACE_TAG=run1 NCCL_MAX_NCHANNELS=16 \
torchrun --nproc_per_node=8 --no-python ./run_att.sh \
  bench_overlap_only.py --grid-dims 228 --seconds 12 --warmup 5
```

The `--no-python` flag tells `torchrun` to launch the wrapper script directly
instead of calling `python <script>`. Expect `Wave incomplete: trace was cutoff`
warnings — the buffer fills and capture stops at a kernel boundary; that's
fine.

To use a different config, point `ATT_CONFIG` at it:
```bash
ATT_CONFIG=/path/to/custom.json TRACE_DIR=/tmp/att_out TRACE_TAG=run1 NCCL_MAX_NCHANNELS=16 \
torchrun --nproc_per_node=8 --no-python ./run_att.sh bench_overlap_only.py ...
```

### Output layout

```
/tmp/att_out/
├── ui_output_agent_<PID>_dispatch_<DID>/
│   ├── occupancy.json        ← per-(SE, CU, SIMD) wave start/end events; primary input to quick_analysis.py
│   ├── filenames.json        ← wave-file index by (SE, SIMD, slot, wave)
│   ├── realtime.json         ← SE-level kernel start/end timestamps
│   ├── code.json             ← disassembled GEMM kernel
│   ├── se*_sm*_sl0_wv*.json  ← individual wave traces (instructions executed)
│   └── wstates*.json         ← wave-state samples
├── run1_<PID>_shader_engine_<N>_<DID>.att   ← raw ATT binaries (need rocprof-trace-decoder; viewable in rocprof-compute-viewer)
├── run1_<PID>_code_object_id_*.out          ← kernel binaries (ELF)
├── run1_<PID>_results.db                    ← SQLite: kernel dispatches + RCCL events
└── stats_ui_output_*.csv                    ← per-instruction Hitcount/Latency/Stall summary
```

### Quick analysis

`quick_analysis.py` answers the headline question: *did the dispatcher
oversubscribe any CU during kernel 1?*

```bash
python3 quick_analysis.py /tmp/att_out/ui_output_agent_*_dispatch_*/occupancy.json
```

Output:
```
Total blocks: 228
Distinct CUs used: 226
Oversubscribed CUs (got 2 blocks): 2
  XCD0 SE3 CU4: 2 blocks
  XCD6 SE25 CU1: 2 blocks
Late waves (start > earliest+1M): 8  (= 2 late blocks)
```

Reading this on the broken host (older amdgpu / firmware):

- **228 blocks on 226 distinct CUs** → 2 CUs got 2 GEMM blocks each.
- **Oversubscribed CUs** are non-deterministic across kernels; the colliding
  CU varies per iter. This is the dispatcher race.
- **Late blocks** start ~3.5M cycles after the others — they're waiting for
  the *first* block on the colliding CU to finish before they can run. Wave
  duration once started is normal (~3.5M cycles); the kernel can't end until
  these last blocks finish, doubling its wallclock.

On the fixed host, the same command reports:
```
Total blocks: 228
Distinct CUs used: 228
Oversubscribed CUs (got 2 blocks): 0
Late waves (start > earliest+1M): 0
```

### Visualization in the GUI

The raw `*.att` files plus `code_object_id_*.out` can be opened in
[`rocprof-compute-viewer`](https://github.com/ROCm/rocprof-compute-viewer)
(install on your laptop, `scp` the trace dir to it). The viewer renders a
per-CU wave timeline; oversubscribed CUs show two bars stacked sequentially
on the same CU row.
