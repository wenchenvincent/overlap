import sys
import json
from collections import defaultdict

#occ = json.load(open('./att_out/.../occupancy.json'))
occ = json.load(open(sys.argv[1]))

# Reconstruct waves: pair start→end events per (SE, CU, SIMD)
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

gemm = [w for w in waves if 200_000 < w[6] < 10_000_000]   # GEMM ≈ 3.5M cycles
# Take just kernel 1 by clustering start times (gap > 5M = next kernel)
starts = sorted(w[4] for w in gemm)
cluster_end = starts[0]
for s in starts[1:]:
  if s - cluster_end > 5_000_000: break
  cluster_end = s
k1 = [w for w in gemm if w[4] <= cluster_end]

# Count blocks per CU
blocks_per_cu = defaultdict(int)
for w in k1: blocks_per_cu[(w[0], w[1], w[2])] += 1   # 4 waves per block
oversubscribed = [(cu, n//4) for cu, n in blocks_per_cu.items() if n > 4]
print(f"Total blocks: {sum(blocks_per_cu.values())//4}")
print(f"Distinct CUs used: {len(blocks_per_cu)}")
print(f"Oversubscribed CUs (got 2 blocks): {len(oversubscribed)}")
for cu, n in oversubscribed:
  print(f"  XCD{cu[0]} SE{cu[1]} CU{cu[2]}: {n} blocks")

# Late-dispatching waves (start > 1M cycles after earliest)
earliest = min(w[4] for w in k1)
late = [w for w in k1 if w[4] > earliest + 1_000_000]
print(f"Late waves (start > earliest+1M): {len(late)}  (= {len(late)//4} late blocks)")
