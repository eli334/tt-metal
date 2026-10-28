#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# --- Repository environment ---
# Use the checkout's Python environment and runtime paths.
cd "$(git rev-parse --show-toplevel)"
export TT_METAL_HOME="$PWD"
export TT_METAL_RUNTIME_ROOT="$PWD"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"
source python_env/bin/activate

export TTNN_RUN_GEMM_BENCHMARK=1

# --- Profiling options and output directory ---
# GEMM_DEVICE_TIMING remains an alias for the earlier validation command.
export GEMM_TRACY="${GEMM_TRACY:-${GEMM_DEVICE_TIMING:-1}}"
timeout_seconds="${GEMM_TIMEOUT_SECONDS:-28800}"
results_root=research/gemm/results

# Every invocation gets a new directory; completed runs are never overwritten.
GEMM_RUN_DIR="$results_root/$(TZ=America/New_York date +%Y-%m-%d_%H-%M-%S-%6N_%Z)"
export GEMM_RUN_DIR
mkdir -p "$results_root"
mkdir "$GEMM_RUN_DIR"

# --- Run the sweep, then process successful measurements ---
if [[ "$GEMM_TRACY" == "1" ]]; then
    export TT_METAL_PROFILER_DIR="$PWD/$GEMM_RUN_DIR/profiler"
    # Keep captures for the GUI; use Tracy's own setting to opt into its web viewer.
    export TRACY_NO_WEB_SERVER="${TRACY_NO_WEB_SERVER:-1}"
    python -m tracy -r -p --check-exit-code --dump-device-data-mid-run -o "$TT_METAL_PROFILER_DIR" \
        -m pytest research/gemm/test_gemm.py::test_matmul_sweep -xvs --timeout="$timeout_seconds" "$@"
    python research/gemm/plot.py "$GEMM_RUN_DIR/samples.csv" --require-device --update-csv
else
    # Host-only runs keep the same sweep and plots without device profiling.
    unset TT_METAL_DEVICE_PROFILER TTNN_OP_PROFILER
    python -m pytest research/gemm/test_gemm.py::test_matmul_sweep -xvs --timeout="$timeout_seconds" "$@"
    python research/gemm/plot.py "$GEMM_RUN_DIR/samples.csv" --update-csv
fi
