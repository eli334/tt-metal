# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Offline checks of the experiment design; no TTNN or device needed."""

import csv

import pytest

from research.gemm.shapes import WORK_SCALES, fixed_flop_shapes, shape_metadata, write_manifest


@pytest.mark.parametrize("scale", WORK_SCALES)
def test_fixed_work_and_axis_coverage(scale):
    shapes = fixed_flop_shapes(scale)
    assert len(shapes) == len(set(shapes)) == 13
    assert shapes[0] == (scale, scale, scale)
    assert {2 * m * k * n for m, k, n in shapes} == {2 * scale**3}
    assert all(d % 128 == 0 for shape in shapes for d in shape)
    assert {max(shape) // min(shape) for shape in shapes} == {1, 4, 16}
    for dimension in range(3):
        assert {shape[dimension] for shape in shapes} == {scale // 4, scale // 2, scale, 2 * scale, 4 * scale}


def test_near_capacity_footprints():
    rows = [shape_metadata(shape, 55296) for shape in fixed_flop_shapes(55296)]
    assert {row["logical_bytes"] for row in rows} == {18345885696, 21403533312, 32105299968}
    assert all(row["tensor_elements"] == row["input_elements"] + row["output_elements"] for row in rows)
    # Deliberately crosses 90% of nominal 32 GiB, but is not a runtime fit claim.
    assert 0.9 * 32 * 2**30 < max(row["logical_bytes"] for row in rows) < 32 * 2**30


def test_manifest_preserves_budget_and_large_integer_counts(tmp_path):
    rows = [shape_metadata(shape, 55296) for shape in fixed_flop_shapes(55296)]
    path = tmp_path / "shapes.csv"
    write_manifest(path, rows)
    with path.open() as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 13
    assert {int(row["useful_flops"]) for row in saved} == {338151365148672}
    assert {int(row["work_scale"]) for row in saved} == {55296}
    assert {row["shape_family"] for row in saved} == {"fixed_flops"}
    assert {int(row["logical_bytes"]) for row in saved} == {row["logical_bytes"] for row in rows}
    with pytest.raises(FileExistsError):
        write_manifest(path, rows)


@pytest.mark.parametrize("scale", [0, -512, 128, 768])
def test_reject_unaligned_work_budget(scale):
    with pytest.raises(ValueError):
        fixed_flop_shapes(scale)


def test_reject_shape_outside_selected_work_budget():
    with pytest.raises(ValueError):
        shape_metadata((1024, 2048, 1024), 1024)
