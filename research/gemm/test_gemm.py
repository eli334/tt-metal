# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep matrix shapes and core grids, checking correctness and timing each batch."""

import csv
import os
from datetime import datetime
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter
from zoneinfo import ZoneInfo

import pytest
import torch
import ttnn


# Experiment configuration: dimensions are elements; grid is (columns, rows).
SHAPES = [  # M, K, N
    (512, 512, 512),
    (1024, 1024, 1024),
    (1024, 2048, 1024),
]
CORE_GRIDS = [
    (4, 4),
    (8, 8),
    None,  # Allow the full available grid; small shapes may use fewer cores.
]
TRACE_MODES = [False, True]
SEED = 0

DTYPE = ttnn.bfloat16
MATH_FIDELITY = ttnn.MathFidelity.HiFi2
FP32_DEST_ACC_EN = True
PACKER_L1_ACC = True
MATH_APPROX_MODE = False
INPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
OUTPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
MAX_RELATIVE_RMSE = 0.02

NUM_WARMUP_ITERATIONS = 5
NUM_MEASUREMENT_ITERATIONS = 20  # Matmuls per timed batch.
NUM_MEASUREMENT_BATCHES = 5
RESULTS_DIR = Path(__file__).parent / "results"

# Blocking settings are in 32x32 tiles, not elements.
TILE_SIZE = 32
IN0_BLOCK_W = 4  # Number of K tiles processed per input block.
OUT_SUBBLOCK = (1, 1)  # Output tiles accumulated together: (height, width).
OUT_BLOCK = None  # None means one full output block per core; otherwise (height, width) in tiles.


def check_output(output, reference):
    actual = ttnn.to_torch(output).float()
    assert actual.shape == reference.shape
    relative_rmse = ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
    print(f"Relative RMSE: {relative_rmse:.6f}")
    assert relative_rmse < MAX_RELATIVE_RMSE, f"Relative RMSE {relative_rmse:.6f} exceeds {MAX_RELATIVE_RMSE}"


def run_case(device, shape, core_grid, dtype, math_fidelity, use_trace, csv_file):
    m, k, n = shape
    writer = csv.writer(csv_file)
    available_grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = (available_grid.x, available_grid.y) if core_grid is None else core_grid
    subblock_h, subblock_w = OUT_SUBBLOCK

    # Round up so all output tiles fit within the selected grid.
    per_core_m = ceil(m / (grid_y * TILE_SIZE))
    per_core_n = ceil(n / (grid_x * TILE_SIZE))
    block_h, block_w = (per_core_m, per_core_n) if OUT_BLOCK is None else OUT_BLOCK

    print(
        f"M={m}, K={k}, N={n}, grid={grid_x}x{grid_y}, trace={use_trace}, "
        f"available_grid={available_grid.x}x{available_grid.y}, "
        f"output_tiles_per_core={per_core_m}x{per_core_n}, "
        f"output_block={block_h}x{block_w}, output_subblock={subblock_h}x{subblock_w}, "
        f"K_block_tiles={IN0_BLOCK_W}"
    )

    generator = torch.Generator().manual_seed(SEED)
    host_a = torch.randn((m, k), generator=generator)
    host_b = torch.randn((k, n), generator=generator)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=IN0_BLOCK_W,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        out_block_h=block_h,
        out_block_w=block_w,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=MATH_APPROX_MODE,
        fp32_dest_acc_en=FP32_DEST_ACC_EN,
        packer_l1_acc=PACKER_L1_ACC,
    )

    a = ttnn.from_torch(
        host_a,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=INPUT_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        host_b,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=INPUT_MEMORY_CONFIG,
    )
    output = ttnn.empty(
        (m, n),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=OUTPUT_MEMORY_CONFIG,
    )
    # Compare against the actual device-format inputs, including any quantization.
    reference = ttnn.to_torch(a).float() @ ttnn.to_torch(b).float()

    def matmul():
        return ttnn.matmul(
            a,
            b,
            program_config=program_config,
            compute_kernel_config=compute_config,
            memory_config=OUTPUT_MEMORY_CONFIG,
            dtype=dtype,
            optional_output_tensor=output,
        )

    # Compile and check correctness before timing.
    matmul()
    check_output(output, reference)
    for _ in range(NUM_WARMUP_ITERATIONS):
        matmul()

    ttnn.synchronize_device(device)
    trace_id = None
    if use_trace:
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(NUM_MEASUREMENT_ITERATIONS):
            matmul()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

    try:
        if use_trace:
            # Warm up one replay outside the timer.
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)

        times = []
        for batch in range(NUM_MEASUREMENT_BATCHES):
            ttnn.synchronize_device(device)
            start = perf_counter()
            if use_trace:
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            else:
                for _ in range(NUM_MEASUREMENT_ITERATIONS):
                    matmul()
            ttnn.synchronize_device(device)
            seconds_per_matmul = (perf_counter() - start) / NUM_MEASUREMENT_ITERATIONS
            times.append(seconds_per_matmul)
            tflops = 2 * m * k * n / seconds_per_matmul / 1e12
            writer.writerow(
                [m, k, n, grid_x, grid_y, dtype, math_fidelity, use_trace, batch + 1, seconds_per_matmul * 1e6, tflops]
            )
            csv_file.flush()
            print(
                f"Batch {batch + 1}: {seconds_per_matmul * 1e6:.2f} us/matmul, {tflops:.3f} TFLOPS "
                f"(host-observed, {NUM_MEASUREMENT_ITERATIONS} matmuls)"
            )

        check_output(output, reference)
        average_time = mean(times)
        print(
            f"Average host-observed time: {average_time * 1e6:.2f} us/matmul, "
            f"{2 * m * k * n / average_time / 1e12:.3f} TFLOPS "
            "(includes host dispatch and synchronization overhead)"
        )
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        for tensor in (output, b, a):
            ttnn.deallocate(tensor)


@pytest.mark.skipif(
    os.getenv("TTNN_RUN_GEMM_BENCHMARK") != "1",
    reason="Run research/gemm/run_bench.sh or set TTNN_RUN_GEMM_BENCHMARK=1",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 3855488}], indirect=True)
def test_matmul_sweep(device):
    # One CSV per sweep, with one row per timed batch.
    run_id = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S-%f_%Z")
    run_dir = Path(os.getenv("GEMM_RUN_DIR") or RESULTS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "samples.csv"
    print(f"Results: {results_path}")

    with results_path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["m", "k", "n", "grid_x", "grid_y", "dtype", "math_fidelity", "use_trace", "batch", "time_us", "tflops"]
        )
        for shape in SHAPES:
            for core_grid in CORE_GRIDS:
                for use_trace in TRACE_MODES:
                    run_case(device, shape, core_grid, DTYPE, MATH_FIDELITY, use_trace, file)
