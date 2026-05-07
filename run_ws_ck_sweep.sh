#!/usr/bin/env bash
# CK work-stealing overlap sweep: NCCL=16/112 × grid=240/256 × {static, --ws-ck}.
# Mirrors the Triton --ws sweep so the two are directly comparable.
set -uo pipefail

WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
SHAPE="--G 32 --M 267424 --K 1280 --N 2560 --ag-size-mb 512"

for ws in "" "--ws-ck"; do
  for nch in 16 112; do
    for grid in 240 256; do
      tag="${ws:-static}"
      echo "=== CK $tag NCCL=$nch grid=$grid ==="
      NCCL_MAX_NCHANNELS=$nch torchrun --nproc_per_node=8 bench_overlap.py \
        --backend primus --trans-b $SHAPE \
        --grid-dims $grid --warmup "$WARMUP" --iters "$ITERS" $ws \
        2>&1 | grep -E "^${grid},${nch},(gemm_only|sequential|overlap_gemm)" | head -3
    done
  done
done
