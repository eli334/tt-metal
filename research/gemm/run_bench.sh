#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export TT_METAL_HOME="$PWD"
export TT_METAL_RUNTIME_ROOT="$PWD"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"
source python_env/bin/activate

export TTNN_RUN_GEMM_BENCHMARK=1

GEMM_RUN_DIR="research/gemm/results/$(TZ=America/New_York date +%Y-%m-%d_%H-%M-%S-%6N_%Z)"
export GEMM_RUN_DIR
mkdir -p research/gemm/results
mkdir "$GEMM_RUN_DIR"

python -m pytest research/gemm/test_gemm.py::test_matmul_sweep -xvs --timeout=7200 "$@"
python research/gemm/plot.py "$GEMM_RUN_DIR/samples.csv"
