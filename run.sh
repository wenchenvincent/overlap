#!/bin/bash
# RCCL + Triton Grouped GEMM Overlap Sweep
#
# Sweeps NCCL_MAX_NCHANNELS and GRID_DIM to characterize how RCCL channel
# count affects grouped gemm performance during overlap.
#
# NCCL_MAX_NCHANNELS controls RCCL channel count (not CU count directly).
# More channels = more thread blocks = more CU pressure from RCCL.
# NCCL_DEBUG=INFO is always set to log actual RCCL resource allocation.
#
# For deeper profiling with rocprofv3:
#   rocprofv3 --sys-trace --rccl-trace -f pftrace -d traces -- \
#     torchrun --nproc_per_node=2 bench_overlap.py
#
# Then view traces in chrome://tracing or https://ui.perfetto.dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/bench_overlap.py"

# Defaults
NPROC="${NPROC:-2}"
RESULTS_FILE="${RESULTS_FILE:-results.csv}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc) NPROC="$2"; shift 2 ;;
        --output) RESULTS_FILE="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS="$*"; break ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "=== RCCL + Triton Grouped GEMM Overlap Sweep ==="
echo "GPUs per node: $NPROC"
echo "Results file:  $RESULTS_FILE"
echo ""

# CSV header
echo "grid_dim,nccl_max_nchannels,scenario,mean_ms,min_ms,max_ms" > "$RESULTS_FILE"

run_bench() {
    local label="$1"
    local nccl_channels="$2"
    local grid_dims="$3"

    echo ""
    echo "================================================================"
    echo "  $label"
    echo "  NCCL_MAX_NCHANNELS=$nccl_channels, --grid-dims=$grid_dims"
    echo "================================================================"

    local env_vars="NCCL_DEBUG=INFO"
    if [[ "$nccl_channels" != "default" ]]; then
        env_vars="$env_vars NCCL_MAX_NCHANNELS=$nccl_channels"
    fi

    # Capture CSV from stdout, let stderr (human-readable) pass through
    env $env_vars \
        torchrun --nproc_per_node="$NPROC" "$BENCH" \
            --grid-dims "$grid_dims" \
            $EXTRA_ARGS \
        2>&2 | grep -v "^grid_dim," >> "$RESULTS_FILE" || true

    echo ""
}

# ---------------------------------------------------------------------------
# Scenario 1: Default RCCL, full gemm CUs
# ---------------------------------------------------------------------------
run_bench "Scenario 1: Default RCCL, GRID_DIM=256" \
    "default" "256"

# ---------------------------------------------------------------------------
# Scenario 2: Sweep RCCL channels, full gemm CUs (GRID_DIM=256)
# ---------------------------------------------------------------------------
for NCHANNELS in 4 8 16 32; do
    run_bench "Scenario 2: NCCL_MAX_NCHANNELS=$NCHANNELS, GRID_DIM=256" \
        "$NCHANNELS" "256"
done

# ---------------------------------------------------------------------------
# Scenario 3: Sweep RCCL channels, reduced gemm CUs
# These grid-dim values are estimates of 256 minus RCCL's CU footprint.
# Check NCCL_DEBUG output to correlate channels with actual CU usage.
# ---------------------------------------------------------------------------
for NCHANNELS in 4 8 16 32; do
    run_bench "Scenario 3: NCCL_MAX_NCHANNELS=$NCHANNELS, sweep grid-dims" \
        "$NCHANNELS" "224,192,160,128"
done

echo ""
echo "================================================================"
echo "  Sweep complete. Results saved to: $RESULTS_FILE"
echo "================================================================"
echo ""
echo "Summary:"
column -t -s',' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
