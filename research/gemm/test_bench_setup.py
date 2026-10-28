# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of sweep coverage, memory checks and bounded output validation."""

import csv
import io
from types import SimpleNamespace

import pytest
import torch

from research.gemm import test_gemm as bench
from research.gemm.shapes import WORK_SCALES, fixed_flop_shapes


def test_one_invocation_visits_every_scale(monkeypatch, tmp_path):
    visited = []
    monkeypatch.setenv("GEMM_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(bench, "run_case", lambda *args: visited.append((args[-1], args[1], args[2])))
    bench.test_matmul_sweep(object())
    assert {scale for scale, _, _ in visited} == set(WORK_SCALES)
    assert len(visited) == 13 * len(WORK_SCALES) * len(bench.CORE_GRIDS)
    assert all(2 * m * k * n == 2 * scale**3 for scale, (m, k, n), _ in visited)
    with (tmp_path / "shapes.csv").open() as file:
        manifest = list(csv.DictReader(file))
    assert len(manifest) == 117
    assert {int(row["work_scale"]) for row in manifest} == set(WORK_SCALES)


def test_dram_capacity_includes_headroom_and_fragmentation(monkeypatch):
    view = SimpleNamespace(
        num_banks=8, total_bytes_free_per_bank=2**30, largest_contiguous_bytes_free_per_bank=2**30
    )
    monkeypatch.setattr(bench.ttnn, "get_memory_view", lambda *args: view)
    allocation, free = bench.check_dram_capacity(object(), (2048, 512, 8192))
    assert allocation == 42 * 2**20
    assert free == 8 * 2**30
    # Plenty of total free memory, but no sufficiently large contiguous block.
    view.largest_contiguous_bytes_free_per_bank = 2**20
    with pytest.raises(RuntimeError, match="contiguously available"):
        bench.check_dram_capacity(object(), (2048, 512, 8192))


def test_host_input_matches_seeded_float32_samples_rounded_to_bf16():
    shape = (64, 128)
    actual = bench.make_host_input(shape, torch.Generator().manual_seed(17))
    expected = torch.randn(shape, generator=torch.Generator().manual_seed(17), dtype=torch.float32).to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_host_input_bounds_fp32_scratch_and_fills_partial_final_chunk(monkeypatch):
    allocations = []
    original_empty = torch.empty

    def allocate(shape, **kwargs):
        tensor = original_empty(shape, **kwargs)
        allocations.append((tensor.numel(), tensor.dtype))
        return tensor

    monkeypatch.setattr(bench, "HOST_RANDOM_CHUNK_ELEMENTS", 4096)
    monkeypatch.setattr(bench.torch, "empty", allocate)
    shape = (33, 257)  # Two full chunks and a partial final chunk.
    actual = bench.make_host_input(shape, torch.Generator().manual_seed(17))
    repeated = bench.make_host_input(shape, torch.Generator().manual_seed(17))
    assert tuple(actual.shape) == shape
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[-1]).item() > 0
    assert torch.equal(actual, repeated)
    assert [size for size, dtype in allocations if dtype == torch.float32] == [4096, 4096]


@pytest.mark.parametrize("full_check", [True, False])
def test_references_match_uploaded_bf16_and_reject_corruption(monkeypatch, full_check):
    deallocated = []
    monkeypatch.setattr(bench, "FULL_CHECK_MAX_FLOPS", 2**60 if full_check else 0)
    monkeypatch.setattr(bench.ttnn, "from_torch", lambda tensor, **kwargs: tensor.clone())
    monkeypatch.setattr(bench.ttnn, "empty", lambda shape, **kwargs: torch.empty(shape, dtype=torch.bfloat16))
    monkeypatch.setattr(bench.ttnn, "to_torch", lambda tensor: tensor)
    monkeypatch.setattr(bench.ttnn, "deallocate", deallocated.append)
    monkeypatch.setattr(
        bench.ttnn,
        "slice",
        lambda tensor, start, end, step, **kwargs: tensor[start[0] : end[0], start[1] : end[1]].clone(),
    )
    a, b, output, checks, mode = bench.prepare_inputs(object(), (128, 256, 128), bench.DTYPE)
    golden = a.float() @ b.float()
    output.copy_(golden)
    assert mode == ("full" if full_check else "sampled_tiles")
    assert len(checks) == (1 if full_check else 5)
    for row, col, reference in checks:
        assert torch.equal(reference, golden[row : row + reference.shape[0], col : col + reference.shape[1]])
    bench.check_output(output, checks)
    assert len(deallocated) == (0 if full_check else 5)
    output[-1, -1] = 10000
    with pytest.raises(AssertionError, match="Relative RMSE"):
        bench.check_output(output, checks)


def test_failed_upload_releases_already_allocated_input(monkeypatch):
    allocations, released = [], []

    def upload(tensor, **kwargs):
        if allocations:
            raise RuntimeError("upload failed")
        allocations.append(tensor.clone())
        return allocations[0]

    monkeypatch.setattr(bench.ttnn, "from_torch", upload)
    monkeypatch.setattr(bench.ttnn, "deallocate", released.append)
    with pytest.raises(RuntimeError, match="upload failed"):
        bench.prepare_inputs(object(), (128, 128, 128), bench.DTYPE)
    assert len(released) == 1 and released[0] is allocations[0]


@pytest.mark.parametrize("scale, expected_iterations", [(512, 20), (16384, 2), (55296, 1)])
@pytest.mark.parametrize("core_grid", [None, (4, 4)])
def test_manual_matmul_preserves_grid_batch_counts_and_csv(monkeypatch, scale, expected_iterations, core_grid):
    output = SimpleNamespace(shape=(scale, scale), padded_shape=(scale, scale))
    input_a = SimpleNamespace(padded_shape=(scale, scale))
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10), arch=lambda: None)
    calls, checks = [], []
    monkeypatch.setenv("GEMM_TRACY", "0")
    monkeypatch.setattr(bench, "check_dram_capacity", lambda *args: (6 * scale**2, 32 * 2**30))
    monkeypatch.setattr(
        bench,
        "prepare_inputs",
        lambda *args: (input_a, object(), output, [(0, 0, torch.ones(32, 32))], "sampled_tiles"),
    )
    monkeypatch.setattr(bench, "check_output", lambda *args: checks.append(True))
    monkeypatch.setattr(bench.ttnn, "init_device_compute_kernel_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(bench.ttnn, "synchronize_device", lambda *args: None)
    monkeypatch.setattr(bench.ttnn, "deallocate", lambda *args: None)
    monkeypatch.setattr(bench.ttnn, "matmul", lambda *args, **kwargs: calls.append(kwargs))
    timer = iter(range(2 * bench.NUM_MEASUREMENT_BATCHES))
    monkeypatch.setattr(bench, "perf_counter", lambda: next(timer))
    file = io.StringIO()
    bench.run_case(device, (scale, scale, scale), core_grid, bench.DTYPE, bench.MATH_FIDELITY, file, scale)
    names = ["m", "k", "n", "grid_x", "grid_y", "dtype", "math_fidelity", "batch", "host_time_us", "host_tflops"]
    names += list(bench.METADATA_COLUMNS) + list(bench.LAYOUT_COLUMNS)
    rows = list(csv.DictReader(io.StringIO(file.getvalue()), fieldnames=names))
    assert len(rows) == bench.NUM_MEASUREMENT_BATCHES
    assert all(None not in row for row in rows)
    assert {int(row["measurement_iterations"]) for row in rows} == {expected_iterations}
    assert {float(row["host_time_us"]) for row in rows} == {1e6 / expected_iterations}
    assert {int(row["work_scale"]) for row in rows} == {scale}
    assert {row["input_generation"] for row in rows} == {bench.INPUT_GENERATION}
    assert {int(row["input_generation_chunk_elements"]) for row in rows} == {bench.HOST_RANDOM_CHUNK_ELEMENTS}
    assert {row["blocking_policy"] for row in rows} == {"manual"}
    assert {(int(row["grid_x"]), int(row["grid_y"])) for row in rows} == {core_grid or (11, 10)}
    assert len(calls) == 1 + min(bench.NUM_WARMUP_ITERATIONS, expected_iterations) + 5 * expected_iterations
    for kwargs in calls:
        config = kwargs["program_config"]
        assert (config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y) == (
            core_grid or (11, 10)
        )
        assert {int(row["per_core_m_tiles"]) for row in rows} == {config.per_core_M}
        assert {int(row["out_block_w_tiles"]) for row in rows} == {config.out_block_w}
        assert {int(row["in0_block_w_tiles"]) for row in rows} == {config.in0_block_w}
        assert kwargs["optional_output_tensor"] is output
        assert "core_grid" not in kwargs
    assert len(checks) == 2


def test_manual_blocks_cover_all_shapes_and_grids_with_bounded_l1():
    for scale in WORK_SCALES:
        for shape in fixed_flop_shapes(scale):
            mt, kt, nt = (dimension // 32 for dimension in shape)
            for gx, gy in ((1, 1), (2, 2), (4, 4), (8, 8), (11, 10)):
                config, metadata = bench.make_program_config(shape, (gx, gy))
                pm, pn = config.per_core_M, config.per_core_N
                bh, bw = config.out_block_h, config.out_block_w
                sh, sw = config.out_subblock_h, config.out_subblock_w
                assert pm % bh == pn % bw == bh % sh == bw % sw == kt % config.in0_block_w == 0
                assert 0 < sh * sw <= 4
                assert pm <= mt and pn <= nt
                rows, columns = (mt + pm - 1) // pm, (nt + pn - 1) // pn
                assert rows <= gy and columns <= gx
                assert metadata["grid_padding_tiles"] == rows * columns * pm * pn - mt * nt
                assert 0 < metadata["estimated_l1_buffer_bytes"] <= bench.L1_BUFFER_BUDGET_BYTES


def test_prime_partition_cliff_keeps_reusable_blocks():
    for scale, ownership in ((32768, (104, 96)), (55296, (176, 160))):
        config, metadata = bench.make_program_config((scale, scale, scale), (11, 10))
        assert (config.per_core_M, config.per_core_N) == ownership
        assert (config.out_block_h, config.out_block_w) == (8, 8)
        assert (config.out_subblock_h, config.out_subblock_w) == (2, 2)
        assert config.in0_block_w == 8
        assert metadata["estimated_l1_buffer_bytes"] == 917504
    # This shape exhausted L1 with the original full-per-core blocks.
    assert bench.make_program_config((2048, 512, 8192), (11, 10))[1]["estimated_l1_buffer_bytes"] < 2**20


def test_manual_blocks_reject_buffers_above_budget(monkeypatch):
    monkeypatch.setattr(bench, "L1_BUFFER_BUDGET_BYTES", 512 * 2**10)
    with pytest.raises(ValueError, match="above.*budget"):
        bench.make_program_config((32768, 32768, 32768), (11, 10))


@pytest.mark.parametrize("comparison", ["shape", "grid"])
def test_plot_places_each_comparison_in_its_own_image(monkeypatch, tmp_path, comparison):
    import pandas as pd
    from research.gemm import plot

    rows = []
    for scale in WORK_SCALES:
        for m, k, n in fixed_flop_shapes(scale):
            rows.append(
                dict(
                    m=m,
                    k=k,
                    n=n,
                    grid_x=11,
                    grid_y=10,
                    shape_family="fixed_flops",
                    work_scale=scale,
                    shape=f"{m} × {k} × {n}",
                    grid="11 × 10",
                    host_tflops=70,
                    device_tflops=100,
                )
            )
    seen = []

    def inspect(fig, path, **kwargs):
        axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(axes) == 9
        for scale in WORK_SCALES:
            assert sum(f"S={scale} ·" in ax.get_title() for ax in axes) == 1
        assert all(len(ax.lines) == 2 for ax in axes)
        assert all(len(ax.lines[0].get_xdata()) == (13 if comparison == "shape" else 1) for ax in axes)
        assert all(("Shape comparison" in ax.get_title()) == (comparison == "shape") for ax in axes)
        assert len({ax.get_ylim() for ax in axes}) == 1
        seen.append(path)

    monkeypatch.setattr(plot.plt.Figure, "savefig", inspect)
    path = tmp_path / f"by_{comparison}.png"
    plot.plot_all_scales(pd.DataFrame(rows), path, "BF16 / HiFi2", comparison)
    assert seen == [path]
