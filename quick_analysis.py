"""
Reads an occupancy.json from an ATT trace and reports the dispatcher-bug
indicators for one of the captured GEMM kernels:

  - Total GEMM blocks dispatched
  - Distinct CUs used  (= 256 - idle - RCCL)
  - Oversubscribed CUs (got >1 GEMM block — the dispatcher race)
  - Late-dispatching blocks (started >1M cycles after the kernel's first wave)
  - Concurrent RCCL CU count and idle CU count

Usage:  python3 quick_analysis.py /path/to/occupancy.json [kernel_index]

  kernel_index: 1-based index of the GEMM kernel to analyze. Defaults to 2,
  since kernel 1 is typically "cold" (NCCL's first AG hasn't dispatched its
  RCCL waves yet, so there's no real overlap). Kernel 2+ are steady-state.
"""
import sys
import json
from collections import defaultdict

occ = json.load(open(sys.argv[1]))
kernel_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# First pass: find the global trace end time (used to close orphan waves).
# An orphan wave is one whose start event was captured but whose end event
# fell after the ATT buffer filled and capture stopped — those waves are
# still legitimately "running" at trace end and must be counted.
trace_end = 0
for se_str, events in occ.items():
    if not se_str.isdigit(): continue
    for e in events:
        if e[0] > trace_end: trace_end = e[0]

# Reconstruct waves: pair start→end events per (SE, CU, SIMD).
# Any unmatched start at end-of-trace is closed at trace_end.
waves = []
for se_str, events in occ.items():
    if not se_str.isdigit(): continue
    se = int(se_str); xcd = se // 4
    active = {}
    for t, cu, simd, _, kind, _ in sorted(events, key=lambda e: e[0]):
        key = (cu, simd)
        if kind == 1: active[key] = t
        elif kind == 0 and key in active:
            start = active.pop(key)
            waves.append((xcd, se, cu, simd, start, t, t-start))
    # Close orphan starts at trace_end (wave was still alive when trace cut off)
    for (cu, simd), start in active.items():
        waves.append((xcd, se, cu, simd, start, trace_end, trace_end-start))

gemm = [w for w in waves if 200_000 < w[6] < 10_000_000]   # GEMM ≈ 3.5M cycles
rccl = [w for w in waves if w[6] >= 10_000_000]            # RCCL ≈ tens of M cycles

# Cluster start times into kernels (gap > 5M cycles = next kernel)
starts = sorted(w[4] for w in gemm)
clusters = []
cur = [starts[0]]
for s in starts[1:]:
    if s - cur[-1] > 5_000_000:
        clusters.append(cur); cur = [s]
    else:
        cur.append(s)
clusters.append(cur)

if kernel_idx < 1 or kernel_idx > len(clusters):
    print(f"Trace has {len(clusters)} kernels; requested kernel_idx={kernel_idx} is out of range.")
    sys.exit(1)
k_cluster = clusters[kernel_idx - 1]
k1 = [w for w in gemm if k_cluster[0] <= w[4] <= k_cluster[-1]]
print(f"Analyzing kernel {kernel_idx} of {len(clusters)} captured  (start ≈ {k_cluster[0]:,} cycles)")
print()

# Count blocks per CU (4 SIMD waves per block, so n//4 = #blocks on that CU)
blocks_per_cu = defaultdict(int)
for w in k1:
    blocks_per_cu[(w[0], w[1], w[2])] += 1
oversubscribed = [(cu, n//4) for cu, n in blocks_per_cu.items() if n > 4]

# RCCL CUs concurrent with kernel 1
k1_start = min(w[4] for w in k1)
k1_end   = max(w[5] for w in k1)
rccl_cus = {(w[0], w[1], w[2]) for w in rccl if w[5] > k1_start and w[4] < k1_end}

n_total_blocks = sum(blocks_per_cu.values()) // 4
n_distinct_gemm_cus = len(blocks_per_cu)
n_rccl_cus = len(rccl_cus)
n_idle = 256 - n_distinct_gemm_cus - n_rccl_cus

print(f"Total GEMM blocks:           {n_total_blocks}")
print(f"Distinct CUs hosting GEMM:   {n_distinct_gemm_cus}")
print(f"Oversubscribed CUs (>1 GEMM): {len(oversubscribed)}")
for cu, n in oversubscribed:
    print(f"  XCD{cu[0]} SE{cu[1]} CU{cu[2]}: {n} blocks")
print(f"RCCL CUs concurrent:         {n_rccl_cus}")
print(f"Truly idle CUs:              {n_idle}")
print(f"  (sanity: 256 = {n_distinct_gemm_cus} GEMM + {n_rccl_cus} RCCL + {n_idle} idle)")

# Late-dispatching waves (start > 1M cycles after earliest)
earliest = min(w[4] for w in k1)
late = [w for w in k1 if w[4] > earliest + 1_000_000]
print(f"Late-dispatching waves:      {len(late)}  (= {len(late)//4} late blocks)")
