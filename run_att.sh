#!/bin/bash
TRACE_DIR=${TRACE_DIR:-/tmp/att_out}
TAG=${TRACE_TAG:-trace}
mkdir -p "$TRACE_DIR"
if [[ "${LOCAL_RANK:-0}" == "0" ]]; then
	export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib
	exec rocprofv3 -i /tmp/att_overlap.json -d "$TRACE_DIR" -o "$TAG" -- python "$@"
else
	exec python "$@"
fi
