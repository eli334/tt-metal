# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot a saved GEMM sweep without accessing a device."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_SHAPE = None  # None chooses the shape with the largest M × K × N.
BASELINE_GRID = None  # None chooses the largest recorded grid, assumed to be the full grid.
BASELINE_TRACE = True  # Hold trace mode fixed for the shape and grid experiments.


def load_results(path):
    data = pd.read_csv(path).sort_values(["m", "k", "n", "grid_x", "grid_y"])
    # The first saved sweeps predate trace mode.
    if "use_trace" not in data:
        data["use_trace"] = False
    data["use_trace"] = data["use_trace"].astype(str).str.lower().eq("true")
    data["shape"] = data[["m", "k", "n"]].astype(str).agg(" × ".join, axis=1)
    data["grid"] = data[["grid_x", "grid_y"]].astype(str).agg(" × ".join, axis=1)
    return data


def draw_curve(ax, data, x_column, order, color="#3975b4", label=None):
    values = data.groupby(x_column)["tflops"].median().reindex(order)
    ax.plot(
        range(len(order)),
        values,
        color=color,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=label,
    )
    labels = [value.replace(" × ", " ×\n", 1) for value in order] if x_column == "shape" else order
    ax.set_xticks(range(len(order)), labels, fontsize=9)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def save_comparison(fig, ax, path, title):
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_by_grid(data, path, config, shape, use_trace):
    shape_label = " × ".join(map(str, shape))
    subset = data[(data["shape"] == shape_label) & (data["use_trace"] == use_trace)]
    grids = subset.sort_values(["grid_x", "grid_y"])["grid"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_curve(ax, subset, "grid", grids)
    ax.set_xlabel("Allowed core grid (columns × rows)")
    mode = "Trace" if use_trace else "Ordinary"
    save_comparison(fig, ax, path, f"TFLOPS versus grid\n{config} · {mode} · M × K × N = {shape_label}")


def plot_by_shape(data, path, config, grid, use_trace):
    grid_label = " × ".join(map(str, grid))
    subset = data[(data["grid"] == grid_label) & (data["use_trace"] == use_trace)]
    shapes = subset["shape"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_curve(ax, subset, "shape", shapes)
    ax.set_xlabel("Matrix shape (M × K × N)")
    mode = "Trace" if use_trace else "Ordinary"
    save_comparison(fig, ax, path, f"TFLOPS versus shape\n{config} · {mode} · Fixed grid: {grid_label}")


def plot_by_trace(data, path, config, grid):
    grid_label = " × ".join(map(str, grid))
    subset = data[data["grid"] == grid_label]
    if subset["use_trace"].nunique() < 2:
        print("Skipping trace comparison: both ordinary and trace measurements are needed.")
        return
    shapes = (
        subset.assign(matrix_size=subset.m * subset.k * subset.n)
        .sort_values("matrix_size")["shape"]
        .drop_duplicates()
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for use_trace, label, color in [(False, "Untraced", "#3975b4"), (True, "Traced", "#db7929")]:
        draw_curve(ax, subset[subset["use_trace"] == use_trace], "shape", shapes, color=color, label=label)
    ax.set_xlabel("Matrix size (M × K × N)")
    ax.legend(frameon=False)
    save_comparison(fig, ax, path, f"TFLOPS versus matrix size\n{config} · Fixed grid: {grid_label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, help="Run CSV; defaults to the newest results/*/samples.csv.")
    parser.add_argument("--output-dir", type=Path, help="Plot directory; defaults to the CSV's directory.")
    args = parser.parse_args()
    csv_path = args.csv or max(
        RESULTS_DIR.glob("*/samples.csv"), key=lambda path: path.stat().st_mtime_ns, default=None
    )
    if csv_path is None:
        parser.error("No saved runs found. Run bash research/gemm/run_bench.sh first.")
    output_dir = args.output_dir or csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {csv_path}")
    data = load_results(csv_path)
    for (dtype, fidelity), subset in data.groupby(["dtype", "math_fidelity"], sort=False):
        name = f"{dtype.split('.')[-1]}_{fidelity.split('.')[-1]}"
        config = f"{dtype.split('.')[-1]} / {fidelity.split('.')[-1]}"
        shape = BASELINE_SHAPE or tuple(subset.loc[(subset.m * subset.k * subset.n).idxmax(), ["m", "k", "n"]])
        grid = BASELINE_GRID or tuple(subset.loc[(subset.grid_x * subset.grid_y).idxmax(), ["grid_x", "grid_y"]])
        use_trace = BASELINE_TRACE and subset["use_trace"].any()
        plot_by_grid(subset, output_dir / f"{name}_by_grid.png", config, shape, use_trace)
        plot_by_shape(subset, output_dir / f"{name}_by_shape.png", config, grid, use_trace)
        plot_by_trace(subset, output_dir / f"{name}_by_trace.png", config, grid)


if __name__ == "__main__":
    main()
