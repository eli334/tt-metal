# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep matrix shapes and core grids, checking correctness and timing each batch."""

import csv
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from zoneinfo import ZoneInfo

import pytest
import torch
import ttnn

from research.gemm.shapes import WORK_SCALES as DEFAULT_WORK_SCALES
from research.gemm.shapes import METADATA_COLUMNS, fixed_flop_shapes, shape_metadata, write_manifest

# --- Experiment settings ---
# Dimensions are elements; grid is (columns, rows). TTNN chooses blocking.
WORK_SCALES = DEFAULT_WORK_SCALES  # Sweep every fixed-FLOP cohort in one invocation.
CORE_GRIDS = [None]  # Allow the full available grid; small shapes may use fewer cores.
SEED = 0

# Keep numerical precision and memory placement fixed across the sweep.
DTYPE = ttnn.bfloat16
MATH_FIDELITY = ttnn.MathFidelity.HiFi2
FP32_DEST_ACC_EN = True
PACKER_L1_ACC = True
MATH_APPROX_MODE = False
INPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
OUTPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
MAX_RELATIVE_RMSE = 0.02

# --- Repetition and memory limits ---
# Large matrices use fewer repetitions and sampled output checks.
NUM_WARMUP_ITERATIONS = 5
NUM_MEASUREMENT_ITERATIONS = 20  # Maximum matmuls per timed batch; large cohorts use fewer.
NUM_MEASUREMENT_BATCHES = 5
# Keep the small-case batching, but avoid 100 repetitions of multi-second GEMMs.
MAX_FLOPS_PER_BATCH = 20 * 2 * 8192**3
FULL_CHECK_MAX_FLOPS = 2 * 2048**3
DRAM_HEADROOM_BYTES = 64 * 2**20
HOST_RANDOM_CHUNK_ELEMENTS = 16 * 2**20  # At most 64 MiB of FP32 random-generation scratch.
INPUT_GENERATION = "torch_float32_chunked_to_bfloat16"
RESULTS_DIR = Path(__file__).parent / "results"

# --- Result metadata ---
# These fields accompany every timed batch; shape metadata comes from shapes.py.
TILE_SIZE = 32
LAYOUT_COLUMNS = (
    "padded_m",
    "padded_k",
    "padded_n",
    "measurement_iterations",
    "warmup_iterations",
    "validation_mode",
    "validation_elements",
    "dram_tensor_allocation_bytes",
    "dram_free_bytes_before",
    "input_generation",
    "input_generation_chunk_elements",
    "blocking_policy",
    "fp32_dest_acc_en",
    "math_approx_mode",
    "packer_l1_acc",
)


# --- Tensor preparation and correctness ---


@contextmanager
def preparation_stage(label):
    """Show progress and host elapsed time for work outside the measured GEMMs."""
    print(f"Preparation: {label} ...", flush=True)
    start = perf_counter()
    yield
    print(f"Preparation: {label} finished in {perf_counter() - start:.2f} s", flush=True)


def make_host_input(shape, generator):
    """Generate normal FP32 samples in bounded chunks, rounding into a BF16 input."""
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    flat = tensor.view(-1)
    # FP32 normal generation is much faster than direct BF16 on this host. Reuse
    # one scratch buffer rather than creating a full-size FP32 copy of the input.
    scratch = torch.empty(min(flat.numel(), HOST_RANDOM_CHUNK_ELEMENTS), dtype=torch.float32)
    for start in range(0, flat.numel(), HOST_RANDOM_CHUNK_ELEMENTS):
        count = min(HOST_RANDOM_CHUNK_ELEMENTS, flat.numel() - start)
        samples = scratch[:count]
        samples.normal_(generator=generator)
        flat[start : start + count].copy_(samples)
    return tensor


def check_dram_capacity(device, shape):
    """Reserve enough contiguous space per bank for three tiled BF16 tensors."""
    m, k, n = shape
    view = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    banks = view.num_banks
    # DRAM interleaving rounds each allocation to whole 2048-byte BF16 tiles/bank.
    allocation_per_bank = sum(((elements // 1024 + banks - 1) // banks) * 2048 for elements in (m * k, k * n, m * n))
    required = allocation_per_bank + (DRAM_HEADROOM_BYTES + banks - 1) // banks
    available = min(view.total_bytes_free_per_bank, view.largest_contiguous_bytes_free_per_bank)
    if required > available:
        raise RuntimeError(
            f"Shape {shape} needs {allocation_per_bank * banks / 2**30:.3f} GiB of tiled DRAM plus "
            f"{DRAM_HEADROOM_BYTES / 2**20:.0f} MiB headroom; only "
            f"{available * banks / 2**30:.3f} GiB is contiguously available across banks."
        )
    return allocation_per_bank * banks, view.total_bytes_free_per_bank * banks


def prepare_inputs(device, shape, dtype):
    """Stage one BF16 input at a time; retain only the CPU data needed to check C."""
    m, k, n = shape
    full_check = 2 * m * k * n <= FULL_CHECK_MAX_FLOPS
    # Large cases check tile-aligned corners and center across the full K dimension.
    positions = (
        [(0, 0)]
        if full_check
        else list(
            dict.fromkeys(
                [
                    (0, 0),
                    (0, n - TILE_SIZE),
                    (m - TILE_SIZE, 0),
                    (m - TILE_SIZE, n - TILE_SIZE),
                    ((m // 2 // TILE_SIZE) * TILE_SIZE, (n // 2 // TILE_SIZE) * TILE_SIZE),
                ]
            )
        )
    )
    generator = torch.Generator().manual_seed(SEED)
    tensors = []
    try:
        # Upload A, retain reference rows, then release the full host allocation.
        with preparation_stage(f"generate A ({m * k * 2 / 2**30:.3f} GiB BF16)"):
            host_a = make_host_input((m, k), generator)
        with preparation_stage("A TTNN conversion/upload"):
            a = ttnn.from_torch(
                host_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=INPUT_MEMORY_CONFIG
            )
        tensors.append(a)
        rows = {
            row: host_a.float() if full_check else host_a[row : row + TILE_SIZE].float()
            for row in dict.fromkeys(row for row, _ in positions)
        }
        del host_a

        # Stage B separately so both full host inputs need not coexist.
        with preparation_stage(f"generate B ({k * n * 2 / 2**30:.3f} GiB BF16)"):
            host_b = make_host_input((k, n), generator)
        with preparation_stage("B TTNN conversion/upload"):
            b = ttnn.from_torch(
                host_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=INPUT_MEMORY_CONFIG
            )
        tensors.append(b)
        columns = {
            col: host_b.float() if full_check else host_b[:, col : col + TILE_SIZE].float()
            for col in dict.fromkeys(col for _, col in positions)
        }
        del host_b
        # References use the exact BF16 host values uploaded, with FP32 CPU math.
        with preparation_stage("CPU correctness references"):
            checks = [(row, col, rows[row] @ columns[col]) for row, col in positions]
        # Reuse one output allocation through compilation, warmup and timing.
        with preparation_stage(f"allocate C ({m * n * 2 / 2**30:.3f} GiB BF16)"):
            output = ttnn.empty(
                (m, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=OUTPUT_MEMORY_CONFIG
            )
        tensors.append(output)
        return a, b, output, checks, "full" if full_check else "sampled_tiles"
    except BaseException:
        # Release earlier uploads if a later allocation or reference step fails.
        for tensor in reversed(tensors):
            ttnn.deallocate(tensor)
        raise


def check_output(output, checks):
    """Compare the whole output or selected tiles against the CPU references."""
    maximum_rmse = 0.0
    for row, col, reference in checks:
        full = tuple(output.shape) == tuple(reference.shape)
        sample = (
            output
            if full
            else ttnn.slice(
                output, (row, col), (row + TILE_SIZE, col + TILE_SIZE), (1, 1), memory_config=OUTPUT_MEMORY_CONFIG
            )
        )
        try:
            actual = ttnn.to_torch(sample).float()
        finally:
            # Sliced tiles are temporary; the full output belongs to run_case.
            if not full:
                ttnn.deallocate(sample)
        assert actual.shape == reference.shape
        relative_rmse = ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
        assert relative_rmse < MAX_RELATIVE_RMSE, f"Relative RMSE {relative_rmse:.6f} exceeds {MAX_RELATIVE_RMSE}"
        maximum_rmse = max(maximum_rmse, relative_rmse)
    print(f"Maximum relative RMSE across {len(checks)} checked output block(s): {maximum_rmse:.6f}")


# --- One shape/grid case ---


def run_case(device, shape, core_grid, dtype, math_fidelity, csv_file, work_scale):
    """Prepare, validate and time one case, then release its device tensors."""
    profile_with_tracy = os.getenv("GEMM_TRACY") == "1"
    if profile_with_tracy:
        from tracy import signpost

    m, k, n = shape
    # Resolve the work budget, repetition counts and allowed grid before allocating.
    metadata = shape_metadata(shape, work_scale)
    iterations = max(1, min(NUM_MEASUREMENT_ITERATIONS, MAX_FLOPS_PER_BATCH // metadata["useful_flops"]))
    warmups = min(NUM_WARMUP_ITERATIONS, iterations)
    dram_allocation, dram_free = check_dram_capacity(device, shape)
    writer = csv.writer(csv_file)
    available_grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = (available_grid.x, available_grid.y) if core_grid is None else core_grid
    # Omit core_grid for the standard out-of-box path; explicit grids use TTNN's
    # constrained auto-selection path, even when they span the full device.
    grid_options = {} if core_grid is None else {"core_grid": ttnn.CoreGrid(x=grid_x, y=grid_y)}
    blocking_policy = "oob" if core_grid is None else "auto_grid"

    print(
        f"family={metadata['shape_family']}, work_scale={work_scale}, "
        f"FLOPs={metadata['useful_flops']}, logical_BF16_GiB={metadata['logical_bytes'] / 2**30:.6f}, "
        f"M={m}, K={k}, N={n}, grid={grid_x}x{grid_y}, "
        f"available_grid={available_grid.x}x{available_grid.y}, "
        f"blocking_policy={blocking_policy}, "
        f"matmuls_per_batch={iterations}, warmups={warmups}, "
        f"DRAM_free_GiB={dram_free / 2**30:.3f}"
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=MATH_APPROX_MODE,
        fp32_dest_acc_en=FP32_DEST_ACC_EN,
        packer_l1_acc=PACKER_L1_ACC,
    )

    # All input transfers, reference work and metadata preparation precede timing.
    a, b, output, checks, validation_mode = prepare_inputs(device, shape, dtype)
    print(f"Validation: {validation_mode}, {sum(ref.numel() for _, _, ref in checks):,} output elements")
    layout = [
        int(output.padded_shape[-2]),
        int(a.padded_shape[-1]),
        int(output.padded_shape[-1]),
        iterations,
        warmups,
        validation_mode,
        sum(ref.numel() for _, _, ref in checks),
        dram_allocation,
        dram_free,
        INPUT_GENERATION,
        HOST_RANDOM_CHUNK_ELEMENTS,
        blocking_policy,
        FP32_DEST_ACC_EN,
        MATH_APPROX_MODE,
        PACKER_L1_ACC,
    ]

    def matmul():
        # TTNN selects the implementation and blocks; no program_config is supplied.
        return ttnn.matmul(
            a,
            b,
            compute_kernel_config=compute_config,
            memory_config=OUTPUT_MEMORY_CONFIG,
            dtype=dtype,
            optional_output_tensor=output,
            **grid_options,
        )

    try:
        # Compile and check correctness before timing.
        matmul()
        check_output(output, checks)
        # Finish warmup before collecting any measured batches.
        for _ in range(warmups):
            matmul()
        ttnn.synchronize_device(device)
        times = []
        for batch in range(NUM_MEASUREMENT_BATCHES):
            # Signposts identify each batch in Tracy and stay outside host timing.
            ttnn.synchronize_device(device)
            if profile_with_tracy:
                label = f"gemm-{m}x{k}x{n}-grid-{grid_x}x{grid_y}-batch-{batch + 1}"
                signpost(
                    f"{label}-start",
                    f"dtype={dtype} fidelity={math_fidelity} iterations={iterations} work_scale={work_scale}",
                )
            # Measure submission plus completion, divided by the number of GEMMs.
            start = perf_counter()
            for _ in range(iterations):
                matmul()
            ttnn.synchronize_device(device)
            seconds_per_matmul = (perf_counter() - start) / iterations
            if profile_with_tracy:
                signpost(f"{label}-end")
            times.append(seconds_per_matmul)
            tflops = 2 * m * k * n / seconds_per_matmul / 1e12
            # Flush each completed batch so interrupted runs retain partial results.
            writer.writerow(
                [m, k, n, grid_x, grid_y, dtype, math_fidelity, batch + 1, seconds_per_matmul * 1e6, tflops]
                + [metadata[column] for column in METADATA_COLUMNS]
                + layout
            )
            csv_file.flush()
            print(
                f"Batch {batch + 1}: {seconds_per_matmul * 1e6:.2f} us/matmul, {tflops:.3f} TFLOPS "
                f"(host-observed, {iterations} matmuls)"
            )

        # Recheck results and drain device profiling after all timed work.
        check_output(output, checks)
        if profile_with_tracy:
            # Use Tracy's supported mid-run dump outside timing so larger sweeps fit.
            ttnn.ReadDeviceProfiler(device)
        average_time = mean(times)
        print(
            f"Average host-observed time: {average_time * 1e6:.2f} us/matmul, "
            f"{2 * m * k * n / average_time / 1e12:.3f} TFLOPS "
            "(includes host dispatch and synchronization overhead)"
        )
    finally:
        # Also release allocations when execution or validation raises an error.
        for tensor in (output, b, a):
            ttnn.deallocate(tensor)


# --- Pytest entry point and sweep output ---


@pytest.mark.skipif(
    os.getenv("TTNN_RUN_GEMM_BENCHMARK") != "1",
    reason="Run research/gemm/run_bench.sh or set TTNN_RUN_GEMM_BENCHMARK=1",
)
def test_matmul_sweep(device):
    """Run every selected shape/grid case into one uniquely named results directory."""
    run_id = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S-%f_%Z")
    run_dir = Path(os.getenv("GEMM_RUN_DIR") or RESULTS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "samples.csv"
    assert DTYPE == ttnn.bfloat16, "The fixed-FLOP footprint and input staging currently require BF16"
    # Save the complete shape inventory before dispatching the first case.
    cases = [(scale, shape) for scale in dict.fromkeys(WORK_SCALES) for shape in fixed_flop_shapes(scale)]
    write_manifest(run_dir / "shapes.csv", [shape_metadata(shape, scale) for scale, shape in cases])
    print(f"Results: {results_path}")
    print(f"Sweep: {len(dict.fromkeys(WORK_SCALES))} scales, {len(cases) * len(CORE_GRIDS)} shape/grid cases")

    # Exclusive creation protects earlier measurements; each case appends batch rows.
    with results_path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["m", "k", "n", "grid_x", "grid_y", "dtype", "math_fidelity", "batch", "host_time_us", "host_tflops"]
            + list(METADATA_COLUMNS)
            + list(LAYOUT_COLUMNS)
        )
        case_number = 0
        for scale, shape in cases:
            for core_grid in CORE_GRIDS:
                case_number += 1
                print(f"Case {case_number}/{len(cases) * len(CORE_GRIDS)}: S={scale}, M/K/N={shape}")
                run_case(device, shape, core_grid, DTYPE, MATH_FIDELITY, file, scale)
