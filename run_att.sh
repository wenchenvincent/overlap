#!/bin/bash
# Rank-0-only wrapper for rocprofv3 --att.
# Used with `torchrun --no-python ./run_att.sh <python_script> [args...]`.
# Other ranks run plain python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATT_CONFIG="${ATT_CONFIG:-$SCRIPT_DIR/att_overlap.json}"
TRACE_DIR=${TRACE_DIR:-/tmp/att_out}
TAG=${TRACE_TAG:-trace}
mkdir -p "$TRACE_DIR"
if [[ "${LOCAL_RANK:-0}" == "0" ]]; then
	export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib
	exec rocprofv3 -i "$ATT_CONFIG" -d "$TRACE_DIR" -o "$TAG" -- python "$@"
else
	exec python "$@"
fi
