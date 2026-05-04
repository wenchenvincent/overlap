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

# Primus-Turbo Triton grouped GEMM (upstream, static-stride persistent loop)
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus-triton --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 240,256

# Same kernel + per-XCD work stealing (vendored fork; recovers most of the
# overlap slowdown at grid=256 without partitioning CUs)
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus-triton --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 240,256 --ws

# Add Triton autotune over a 15-config sweep on top of work stealing
NCCL_MAX_NCHANNELS=16 torchrun --nproc_per_node=8 bench_overlap.py \
  --backend primus-triton --trans-b \
  --G 32 --M 267424 --K 1280 --N 2560 \
  --ag-size-mb 512 --grid-dims 240,256 --ws --autotune

# Colleague's test script
NCCL_MAX_NCHANNELS=16 NUM_CU=240 TEST_OVERLAP_TRACE_DIR=./traces python3 test_overlap.py
```

## Files

| File | Description |
|---|---|
| `bench_overlap.py` | Main reproducer. Triton or Primus-Turbo CK grouped GEMM + RCCL all-gather overlap benchmark with CUDA event timing. |
| `vendored_primus/` | Vendored fork of `primus_turbo`'s persistent grouped-GEMM Triton kernel with an added per-XCD + global-fallback work-stealing path and an opt-in `@triton.autotune` config sweep. Modeled on tritonBLAS `persistent_gemm_work_stealing.py`. Used by `--backend primus-triton --ws [--autotune]`. |
| `test_ws_correctness.py` | Single-GPU correctness harness: runs the vendored WS kernel and the upstream static-stride kernel on identical inputs and asserts bit-for-bit (or near-bit-for-bit) agreement across G ∈ {1, 8, 32} and a range of M, K, N, trans_b. |
| `test_overlap.py` | Three-phase test (AG only, GEMM only, overlapped) with PyTorch profiler trace export and multi-rank trace merging. |
| `grouped_gemm_ops.py` | Primus-Turbo grouped GEMM wrapper (fprop/dgrad/wgrad) with `num_cu` parameter. |
| `bench_cu_occupancy.py` | Controlled experiment: synthetic CU occupier kernel + grouped GEMM to measure slowdown as a function of stolen CUs. |
| `bench_collectives.py` | Compares CU resource usage across all-gather, reduce-scatter, and all-reduce. |
| `run.sh` | Orchestrates `NCCL_MAX_NCHANNELS` sweep across multiple `torchrun` invocations. |
| `check_transb_resources.py` | Utility to compare CU resources between `trans_b=False` and `trans_b=True` kernel paths. |
| `bench_overlap_only.py` | Stripped-down benchmark: only the overlap loop (no gemm-only / sequential phases). Used as the workload for the ATT trace. |
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

### 7. Work stealing recovers most of the overlap slowdown without CU partitioning

The vendored `primus-triton` kernel adds a hierarchical work-stealing path
(`--ws`): each of the 8 XCDs owns a contiguous slice of `total_tiles` and its
32 CUs race on a single padded atomic counter; faster CUs claim more tiles
when stragglers (e.g. CUs preempted by RCCL) lag. A global counter handles the
ragged tail. With `--autotune` on top, Triton sweeps 15 candidate configs
(BLOCK_M/N/K, num_warps, num_stages, waves_per_eu, kpack, matrix_instr_nonkdim)
and caches the winner per `(G, N, K)`.

All numbers below: G=32, M=267424, K=1280, N=2560, trans_b, 8x MI355X,
NCCL_MAX_NCHANNELS=16.

| Variant | grid | gemm_only (mean / min) | overlap_gemm | Slowdown |
|---|---:|---:|---:|---:|
| Primus CK | 240 | 1.88 / 1.69 ms | 2.39 ms | 1.10× |
| Primus CK | 256 | 2.00 / 1.83 ms | 3.48 ms | 1.74× |
| primus-triton (static) | 240 | 2.17 / 2.13 ms | 2.39 ms | 1.10× |
| primus-triton (static) | 256 | 2.12 / 2.05 ms | 4.29 ms | 2.03× |
| primus-triton `--ws` | 240 | 2.25 / 2.15 ms | 2.47 ms | 1.10× |
| primus-triton `--ws` | 256 | 2.10 / 2.05 ms | 2.46 ms | **1.17×** |
| primus-triton `--ws --autotune` | 240 | 1.96 / 1.91 ms | 2.22 ms | 1.13× |
| primus-triton `--ws --autotune` | 256 | 2.05 / **1.85 ms** | **2.22 ms** | **1.09×** |

Key takeaways:

- **Without WS**: at grid=256 the Triton kernel slows 2.03× under overlap.
  Capping the grid to 240 fully recovers it but is workload-specific
  hand-tuning.
- **With `--ws`** at grid=256: slowdown drops to 1.17×. Intra-XCD stealing
  rebalances tiles automatically when some CUs are slowed by RCCL
  contention — no need to pick a "right" CU count.
- **With `--ws --autotune`** at grid=256: slowdown **1.09×** and the absolute
  overlap_gemm time (2.22 ms) is the **best of any variant**, beating even
  Primus CK with hand-tuned partitioning (2.39 ms at grid=240). Autotune's
  contribution here is mostly in re-tuning per-NUM_SMS specialization;
  the chosen config for `(G=32, N=2560, K=1280)` matches the upstream default
  but for other shapes (G=4, K=1024 etc.) it picks `matrix_instr_nonkdim=32`
  or `num_stages=3`.
- **No-overlap overhead of WS**: ~1% by min times. Eight padded per-XCD
  atomics are essentially free. For autotune the first call per `(G,N,K)`
  spends 10–30 s on the search; subsequent calls hit the cache.

Under heavy uniform contention (NCCL_MAX_NCHANNELS=112) WS still helps but
less dramatically (slowdown ~1.94× at grid=256 with WS vs 2.08× without) —
when every XCD is equally squeezed, there's less per-CU imbalance for
stealing to exploit. The tritonBLAS-style hierarchical fallback to a global
counter is wired in (handles the ragged tail) but a fractional reservation
that would force meaningful cross-XCD stealing under uniform contention is
left as future work.

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
--backend        "triton", "primus", or "primus-triton" (default: "triton")
--trans-b        Use transposed weight layout [G, N, K] (Primus / primus-triton only)
--ws             Enable per-XCD + global-fallback work stealing in the
                 vendored primus-triton kernel (forward only)
--autotune       Run the vendored primus-triton kernel through @triton.autotune
                 over a small config sweep (~10–30s search per (G,N,K) on the
                 first call; cached afterwards). Combine with --ws.
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

### What kernels get captured (and the warmup gotcha)

ATT in dispatch mode captures the **first N matching kernels**, where N =
`att_consecutive_kernels` from the JSON config. The kernel filter
(`kernel_include_regex: GroupedGemm`) matches **every** GroupedGemm dispatch
including the ones during `bench_overlap_only.py`'s warmup loop. With the
default `--warmup 5` and `att_consecutive_kernels: 4`, all 4 captured kernels
are warmup iterations.

That has one important consequence: **kernel 1 of the trace is the very first
warmup iter, where NCCL has not yet completed its device-side setup for the
all-gather**. RCCL's worker workgroups don't dispatch until ≈45 ms after the
host-side `dist.all_gather_into_tensor()` call, by which point GEMM kernel 1
has already finished. So kernel 1 of the trace shows **0 RCCL CUs** — not
because the bench skipped overlap on iter 1, but because NCCL's first-call
setup beats GEMM completion. Kernel 2 onward is steady-state overlap (16 RCCL
CUs concurrent).

`quick_analysis.py` defaults to **kernel 2** to skip this cold iteration. Pass
a different index if you want to inspect a specific kernel:
```bash
python3 quick_analysis.py occupancy.json 1   # kernel 1 (cold; 0 RCCL)
python3 quick_analysis.py occupancy.json 3   # kernel 3 (steady-state)
```

If you want ATT to skip warmup entirely and only capture timed-loop iters,
two options:

1. **Increase `--warmup` enough** that NCCL's first-call setup is amortized
   *and* `att_consecutive_kernels` is exhausted before the timed loop starts.
   Crude but no code changes:
   ```bash
   torchrun ... ./run_att.sh bench_overlap_only.py --warmup 4 --seconds 12
   ```
   With `att_consecutive_kernels: 4`, ATT captures warmup iters 1–4. The
   timed loop starts after warmup but isn't traced — you'd lose data.
   *Not recommended.*

2. **Use roctx markers + `--selected-regions`** to gate ATT on a code region.
   Wrap the timed loop with `roctx_profiler_pause(0)` / `roctx_profiler_resume(0)`
   and add `--selected-regions 1` to the rocprofv3 invocation. This is the
   correct fix; requires the `rocprofiler-sdk-roctx` Python bindings and a
   small edit to `bench_overlap_only.py`. Suggested skeleton:
   ```python
   from rocprofiler_sdk_roctx import profiler_pause, profiler_resume
   profiler_pause(0)               # pause ATT during warmup
   for _ in range(args.warmup):
       ... (existing warmup body)
   dist.barrier()
   profiler_resume(0)              # resume — ATT now captures
   while time.time() < deadline:
       ... (existing timed-loop body)
   ```
   Then run with:
   ```bash
   ATT_EXTRA_FLAGS="--selected-regions 1" \
   torchrun --nproc_per_node=8 --no-python ./run_att.sh bench_overlap_only.py ...
   ```
   (`run_att.sh` would need a small extension to forward `ATT_EXTRA_FLAGS`
   into the rocprofv3 command line.)

For the dispatcher-bug investigation specifically, the kernel-2 default in
`quick_analysis.py` is sufficient — kernel 2 in the warmup loop is already
steady-state overlap and exhibits the bug. The warmup-vs-timed distinction
matters more if you're measuring *timing* (which we do via wallclock from
`bench_overlap.py`'s scenarios, not from ATT).

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
oversubscribe any CU during this overlap iteration?*

```bash
python3 quick_analysis.py /tmp/att_out/ui_output_agent_*_dispatch_*/occupancy.json
```

By default it analyzes **kernel 2** of the captured kernels. Kernel 1 is the
"cold" iteration where NCCL's first AG hasn't yet dispatched its RCCL waves,
so the GPU sees GEMM in isolation — not representative of steady-state
overlap. Kernel 2 (and later) are the real overlap case. To analyze a
different kernel, pass its 1-based index:
```bash
python3 quick_analysis.py occupancy.json 1   # kernel 1 (cold)
python3 quick_analysis.py occupancy.json 3   # kernel 3
```

Output on the broken host (cv350, older amdgpu/firmware):
```
Analyzing kernel 2 of 4 captured  (start ≈ 90,784,660 cycles)

Total GEMM blocks:           228
Distinct CUs hosting GEMM:   227
Oversubscribed CUs (>1 GEMM): 1
  XCD0 SE2 CU6: 2 blocks
RCCL CUs concurrent:         16
Truly idle CUs:              13
  (sanity: 256 = 227 GEMM + 16 RCCL + 13 idle)
Late-dispatching waves:      4  (= 1 late blocks)
```

How to read it:

- **228 blocks on 227 distinct CUs** → 1 CU got 2 GEMM blocks (the bug).
- **16 RCCL CUs concurrent** — RCCL has all 16 channels resident on disjoint
  CUs during this kernel.
- **13 idle CUs** — completely unused. *The bug isn't a CU shortage*: the
  dispatcher chose to put 2 blocks on one CU while leaving 13 others empty.
- **Oversubscribed CU is non-deterministic** across iterations — the
  colliding CU varies per iter. That's the dispatcher race.
- **Late-dispatching wave** starts ~3.5M cycles after the others — it's
  waiting for the *first* block on the colliding CU to finish. Once it
  starts, it runs for the normal ~3.5M cycles; the kernel can't end until
  it finishes, ~doubling the wallclock.

On the fixed host (smci355, newer amdgpu/firmware), the same command reports:
```
Analyzing kernel 2 of 4 captured  (start ≈ 92,717,764 cycles)

Total GEMM blocks:           228
Distinct CUs hosting GEMM:   228
Oversubscribed CUs (>1 GEMM): 0
RCCL CUs concurrent:         16
Truly idle CUs:              12
  (sanity: 256 = 228 GEMM + 16 RCCL + 12 idle)
Late-dispatching waves:      0  (= 0 late blocks)
```

### Visualization in the GUI

The raw `*.att` files plus `code_object_id_*.out` can be opened in
[`rocprof-compute-viewer`](https://github.com/ROCm/rocprof-compute-viewer)
(install on your laptop, `scp` the trace dir to it). The viewer renders a
per-CU wave timeline; oversubscribed CUs show two bars stacked sequentially
on the same CU row.
