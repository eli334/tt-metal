# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fixed-FLOP GEMM cohorts and their logical BF16 footprints; no device imports."""

import argparse
import csv
from itertools import permutations
from pathlib import Path

# --- Work scales and manifest fields ---
# Cubic-equivalent side lengths, from launch-bound to near single-p150 capacity.
# Large cohorts are candidates: DRAM allocation, L1 blocking and correctness
# validation must be checked before running them. These are not fit guarantees.
WORK_SCALES = (512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 55296)
METADATA_COLUMNS = (
    "shape_family",
    "work_scale",
    "input_elements",
    "output_elements",
    "tensor_elements",
    "useful_flops",
    "logical_bytes",
)


# --- Fixed-FLOP shape generation and storage estimates ---


def fixed_flop_shapes(scale):
    """Return 13 shapes with M*K*N == scale**3 and all dimensions 128-aligned."""
    if not isinstance(scale, int) or scale < 512 or scale % 512:
        raise ValueError("Work scale must be a positive multiple of 512")
    shapes = [(scale, scale, scale)]
    # Axis permutations change shape while preserving M*K*N and useful FLOPs.
    for short, long in ((scale // 2, 2 * scale), (scale // 4, 4 * scale)):
        shapes.extend(sorted(set(permutations((short, scale, long)))))
    return shapes


def shape_metadata(shape, scale):
    """Logical BF16 A+B+C storage excludes allocator overhead and temporaries."""
    m, k, n = shape
    if any(d <= 0 or d % 128 for d in shape) or m * k * n != scale**3:
        raise ValueError(f"Shape {shape} is not 128-aligned fixed-FLOP work at scale {scale}")
    input_elements = m * k + k * n
    output_elements = m * n
    return dict(
        m=m,
        k=k,
        n=n,
        shape_family="fixed_flops",
        work_scale=scale,
        input_elements=input_elements,
        output_elements=output_elements,
        tensor_elements=input_elements + output_elements,
        useful_flops=2 * m * k * n,
        logical_bytes=2 * (input_elements + output_elements),
    )


# --- Offline inventory output ---


def write_manifest(path, rows):
    """Save the selected shapes before measurement, without replacing a file."""
    with Path(path).open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("m", "k", "n", *METADATA_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, nargs="+", default=WORK_SCALES, help="One or more cubic-equivalent sides")
    parser.add_argument("--csv", type=Path, help="Also save this inventory to a new CSV")
    args = parser.parse_args()
    rows = []
    for scale in dict.fromkeys(args.scale):
        # Listing and manifest generation use only logical dimensions, without TTNN.
        try:
            cohort = [shape_metadata(shape, scale) for shape in fixed_flop_shapes(scale)]
        except ValueError as error:
            parser.error(str(error))
        rows.extend(cohort)
        footprints = [row["logical_bytes"] / 2**30 for row in cohort]
        print(
            f"S={scale}: {len(cohort)} shapes, {2 * scale**3:,} FLOPs each; "
            f"BF16 A+B+C {min(footprints):.6f}-{max(footprints):.6f} GiB"
        )
        for row in cohort:
            print(f"  M={row['m']:>6} K={row['k']:>6} N={row['n']:>6}  {row['logical_bytes'] / 2**30:.6f} GiB")
    print("Footprints are logical A+B+C only; they do not certify device fit or execution readiness.")
    if args.csv:
        write_manifest(args.csv, rows)
        print(args.csv)


if __name__ == "__main__":
    main()
