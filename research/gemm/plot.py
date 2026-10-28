# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot a saved GEMM sweep without accessing a device."""

import argparse
import csv
import re
from math import ceil, isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- Plot settings and CSV identities ---
# The same case/batch keys identify host rows and Tracy signpost intervals.
RESULTS_DIR = Path(__file__).parent / "results"
BASELINE_SHAPE = None  # None chooses the cube in each work cohort, or largest recorded work for legacy data.
BASELINE_GRID = None  # None chooses the largest recorded grid, assumed to be the full grid.
CASE_COLUMNS = ["m", "k", "n", "grid_x", "grid_y", "dtype", "math_fidelity", "batch"]
DEVICE_COLUMNS = ["device_kernel_time_us", "device_tflops", "core_count", "device_samples", "device_id"]
BATCH_MARKER = re.compile(
    r"gemm-(?P<m>\d+)x(?P<k>\d+)x(?P<n>\d+)-grid-(?P<grid_x>\d+)x(?P<grid_y>\d+)"
    r"(?:-trace-False)?-batch-(?P<batch>\d+)-(?P<phase>start|end)"
)


# --- Device timing attribution ---


def add_device_results(data, report):
    """Aggregate Tracy's operation durations inside batch markers and join host rows."""
    batches, seen_executions = [], set()
    active = None
    with report.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["OP TYPE"] == "signpost":
                marker = BATCH_MARKER.fullmatch(row["OP CODE"])
                if marker is None:
                    continue
                label = row["OP CODE"].rsplit("-", 1)[0]
                if marker["phase"] == "start":
                    # Only operations between a matching pair of markers are timed.
                    if active is not None:
                        raise ValueError(f"Nested measured batches in {report}")
                    metadata = dict(item.split("=", 1) for item in row["ATTRIBUTES"].split())
                    case = {key: int(marker[key]) for key in ("m", "k", "n", "grid_x", "grid_y", "batch")}
                    case.update(dtype=metadata["dtype"], math_fidelity=metadata["fidelity"])
                    expected = int(metadata["iterations"])
                    active = (label, case, expected, [], set(), set())
                else:
                    # Reject incomplete batches before converting nanoseconds to us.
                    if active is None or active[0] != label:
                        raise ValueError(f"Unmatched batch end: {row['OP CODE']}")
                    _, case, expected, durations, cores, chips = active
                    if expected <= 0 or len(durations) != expected or len(cores) != 1 or len(chips) != 1:
                        raise ValueError(f"Incomplete device data for {label}: {len(durations)}/{expected} samples")
                    core_count = next(iter(cores))
                    if not 0 < core_count <= case["grid_x"] * case["grid_y"]:
                        raise ValueError(f"Invalid participating core count for {label}: {core_count}")
                    device_us = sum(durations) / expected / 1000
                    batches.append(
                        dict(
                            **case,
                            device_kernel_time_us=device_us,
                            device_tflops=2 * case["m"] * case["k"] * case["n"] / device_us / 1e6,
                            core_count=core_count,
                            device_samples=expected,
                            device_id=next(iter(chips)),
                        )
                    )
                    active = None
            elif active is not None and row["OP TYPE"] == "tt_dnn_device":
                # Each measured execution must be a unique GEMM with valid timing.
                if row["OP CODE"] != "MatmulDeviceOperation":
                    raise ValueError(f"Unexpected operation inside measured GEMM batch: {row['OP CODE']}")
                identity = tuple(
                    row[key]
                    for key in ("DEVICE ID", "GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID")
                )
                if identity in seen_executions:
                    raise ValueError(f"Duplicate measured device execution: {identity}")
                duration = float(row["DEVICE KERNEL DURATION [ns]"])
                if not isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid device kernel duration for {identity}: {duration}")
                seen_executions.add(identity)
                active[3].append(duration)
                active[4].add(int(row["CORE COUNT"]))
                active[5].add(int(row["DEVICE ID"]))
    if active is not None or not batches:
        raise ValueError(f"Missing or unfinished measured batches in {report}")
    # Require an exact host/device match; missing or extra batches are errors.
    merged = data.drop(columns=DEVICE_COLUMNS, errors="ignore").merge(
        pd.DataFrame(batches), on=CASE_COLUMNS, how="outer", validate="one_to_one", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Host samples and Tracy device batches do not match exactly")
    if merged.groupby(CASE_COLUMNS[:-1])["core_count"].nunique().gt(1).any():
        raise ValueError("Participating core count changed between batches of one case")
    if merged["device_id"].nunique() != 1:
        raise ValueError("Expected device measurements from one physical chip")
    return merged.drop(columns="_merge")


# --- Load and validate saved measurements ---


def load_results(path, require_device=False):
    """Read host samples, join device timing when available, and validate both."""
    path = Path(path)
    data = pd.read_csv(path).sort_values(["m", "k", "n", "grid_x", "grid_y"])
    # Keep ordinary-execution rows when reading older CSVs that also contain trace replays.
    if "use_trace" in data:
        data = data.loc[~data["use_trace"].astype(str).str.lower().eq("true")].drop(columns="use_trace")
    if data.empty:
        raise ValueError("CSV contains no ordinary-execution samples")
    # Legacy time_us/tflops columns always described synchronized host batches.
    data = data.rename(columns={"time_us": "host_time_us", "tflops": "host_tflops"})
    if not data[["host_time_us", "host_tflops"]].gt(0).all().all():
        raise ValueError("Host timings and throughput must be positive and complete")
    # Already enriched CSVs can be replotted without reopening the Tracy report.
    if "device_kernel_time_us" not in data or data["device_kernel_time_us"].isna().all():
        reports = list((path.parent / "profiler" / "reports").glob("**/ops_perf_results*.csv"))
        if len(reports) > 1:
            raise ValueError("Expected one Tracy report for this run")
        if reports:
            data = add_device_results(data, reports[0])
        elif require_device:
            raise ValueError("Profiling was requested but the run has no Tracy device report")
        else:
            for column in DEVICE_COLUMNS:
                data[column] = float("nan")
    # Profiling-enabled runs require complete device data rather than host fallbacks.
    data["profiling_enabled"] = data["device_kernel_time_us"].notna()
    if data["profiling_enabled"].any() and not data[DEVICE_COLUMNS].ge(0).all().all():
        raise ValueError("Device measurements must be complete and nonnegative")
    if (
        data["profiling_enabled"].any()
        and not data[["device_kernel_time_us", "device_tflops", "device_samples", "core_count"]].gt(0).all().all()
    ):
        raise ValueError("Device timings, throughput, sample counts, and cores must be positive")
    if require_device and not data["profiling_enabled"].all():
        raise ValueError("Profiling was requested but some batches have no device timing")
    # These ordinary GEMMs run inside the synchronized host interval. Allow a
    # generous margin for clock conversion/overlap, but reject gross clock skew.
    implausible = data["device_kernel_time_us"] > 2 * data["host_time_us"]
    if implausible.any():
        first = data.loc[implausible].iloc[0]
        raise ValueError(
            f"Invalid device timing in {int(implausible.sum())} batches: "
            f"GEMM {int(first.m)}x{int(first.k)}x{int(first.n)}, "
            f"grid {int(first.grid_x)}x{int(first.grid_y)}, batch {int(first.batch)} reports "
            f"{first.device_kernel_time_us:.3f} us on device versus {first.host_time_us:.3f} us on host. "
            "Check inter-core clock alignment in the native profiler log; these device results cannot be used. "
            "Host samples and raw profiler artifacts are preserved."
        )
    data["shape"] = data[["m", "k", "n"]].astype(str).agg(" × ".join, axis=1)
    data["grid"] = data[["grid_x", "grid_y"]].astype(str).agg(" × ".join, axis=1)
    return data


# --- Curves and side-by-side work-scale panels ---


def draw_curve(ax, data, x_column, order):
    """Overlay median host and device throughput in the requested x-axis order."""
    for column, label, color in (
        ("host_tflops", "Host-observed", "#3975b4"),
        ("device_tflops", "Device kernel", "#d47d29"),
    ):
        if data[column].notna().any():
            values = data.groupby(x_column)[column].median().reindex(order)
            ax.plot(range(len(order)), values, color=color, marker="o", markersize=4, linewidth=1.8, label=label)
    labels = [value.replace(" × ", " ×\n") for value in order] if x_column == "shape" else order
    ax.set_xticks(range(len(order)), labels, fontsize=8)
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def draw_by_grid(ax, data, title, shape):
    shape_label = " × ".join(map(str, shape))
    subset = data[data["shape"] == shape_label]
    grids = subset.sort_values(["grid_x", "grid_y"])["grid"].drop_duplicates().tolist()
    draw_curve(ax, subset, "grid", grids)
    ax.set_xlabel("Allowed core grid (columns × rows)")
    ax.set_title(f"{title}\nGrid comparison · M × K × N = {shape_label}", fontsize=11)


def draw_by_shape(ax, data, title, grid):
    grid_label = " × ".join(map(str, grid))
    subset = data[data["grid"] == grid_label]
    shapes = subset["shape"].drop_duplicates().tolist()
    draw_curve(ax, subset, "shape", shapes)
    ax.set_xlabel("Matrix shape (M × K × N)")
    ax.set_title(f"{title}\nShape comparison · Allowed grid: {grid_label}", fontsize=11)


def plot_all_scales(data, path, config, comparison):
    """One image per comparison, with a separate panel for each work scale."""
    if comparison not in ("shape", "grid"):
        raise ValueError(f"Unknown comparison: {comparison}")
    cohorts = list(data.groupby(["shape_family", "work_scale"], sort=True)) if "work_scale" in data else [(None, data)]
    columns = min(3, len(cohorts))
    rows = ceil(len(cohorts) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 4.5 * rows), sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for index, (key, subset) in enumerate(cohorts):
        # Each scale gets its own baseline cube and largest allowed grid.
        title = "Recorded shapes"
        shape = BASELINE_SHAPE
        if key is not None:
            family, scale = key
            title = f"{family} · S={scale} · {2 * int(scale)**3 / 1e12:.5g} TFLOP/GEMM"
            if shape is None and ((subset.m == scale) & (subset.k == scale) & (subset.n == scale)).any():
                shape = (scale, scale, scale)
        if shape is None:
            shape = tuple(subset.loc[(subset.m * subset.k * subset.n).idxmax(), ["m", "k", "n"]])
        grid = BASELINE_GRID or tuple(subset.loc[(subset.grid_x * subset.grid_y).idxmax(), ["grid_x", "grid_y"]])
        ax = axes.flat[index]
        ax.set_visible(True)
        if comparison == "shape":
            draw_by_shape(ax, subset, title, grid)
        else:
            draw_by_grid(ax, subset, title, shape)
    # Keep the throughput scale and legend consistent across all panels.
    maximum = data[["host_tflops", "device_tflops"]].max().max()
    axes[0, 0].set_ylim(0, maximum * 1.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    comparison_title = "Shape comparisons" if comparison == "shape" else "Core-grid scaling"
    fig.suptitle(f"{config} · {comparison_title} across work scales", fontsize=16, y=0.998)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.983), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3, w_pad=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


# --- Command-line loading, optional CSV update and plot generation ---


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, help="Run CSV; defaults to the newest results/*/samples.csv.")
    parser.add_argument("--output-dir", type=Path, help="Plot directory; defaults to the CSV's directory.")
    parser.add_argument(
        "--require-device", action="store_true", help="Fail unless every host batch has device measurements."
    )
    parser.add_argument(
        "--update-csv", action="store_true", help="Save explicit host/device metrics into the run's CSV."
    )
    args = parser.parse_args()
    csv_path = args.csv or max(
        RESULTS_DIR.glob("*/samples.csv"), key=lambda path: path.stat().st_mtime_ns, default=None
    )
    if csv_path is None:
        parser.error("No saved runs found. Run bash research/gemm/run_bench.sh first.")
    output_dir = args.output_dir or csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {csv_path}")
    data = load_results(csv_path, require_device=args.require_device)
    if args.update_csv:
        # Replace the CSV only after the combined host/device data passes validation.
        temporary = csv_path.with_suffix(".csv.tmp")
        data.drop(columns=["shape", "grid"]).fillna("").to_csv(temporary, index=False)
        temporary.replace(csv_path)
    metrics = ["host_time_us", "device_kernel_time_us", "host_tflops", "device_tflops"]
    print("Median batch measurements (us/matmul and TFLOPS):")
    print(
        data.groupby(CASE_COLUMNS[:-1])[metrics]
        .median()
        .to_string(float_format=lambda value: f"{value:.3f}", na_rep="disabled")
    )
    # Produce separate shape and core-grid images for each dtype/fidelity pair.
    for (dtype, fidelity), subset in data.groupby(["dtype", "math_fidelity"], sort=False):
        name = f"{dtype.split('.')[-1]}_{fidelity.split('.')[-1]}"
        config = f"{dtype.split('.')[-1]} / {fidelity.split('.')[-1]}"
        if "blocking_policy" in subset and subset.blocking_policy.nunique() == 1:
            config += f" · blocking: {subset.blocking_policy.iloc[0]}"
        elif (
            "out_block_h_tiles" in subset
            and subset[["out_block_h_tiles", "out_block_w_tiles"]].drop_duplicates().shape[0] == 1
        ):
            h, w = subset.iloc[0][["out_block_h_tiles", "out_block_w_tiles"]]
            config += f" · output blocks: {int(h)} × {int(w)} tiles"
        plot_all_scales(subset, output_dir / f"{name}_by_shape.png", config, "shape")
        plot_all_scales(subset, output_dir / f"{name}_by_grid.png", config, "grid")


if __name__ == "__main__":
    main()
