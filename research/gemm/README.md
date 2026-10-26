# GEMM shape and grid experiments

Run the sweep from the repository root:

```bash
bash research/gemm/run_bench.sh --device-id 0
```

The runner activates `python_env`, sets the required paths, enables the benchmark,
and generates plots after the benchmark passes. Additional pytest arguments are
forwarded. Console output stays in the terminal. Each invocation creates a directory
with a readable timestamp in New York time:

```text
results/2026-10-26_19-03-58-582293_EDT/
  samples.csv
  BFLOAT16_HiFi2_by_shape.png
  BFLOAT16_HiFi2_by_grid.png
  BFLOAT16_HiFi2_by_trace.png
```

The benchmark and plotting script use the same run directory. A failed benchmark
keeps its partial CSV and stops before generating plots.

Edit `SHAPES`, `CORE_GRIDS`, and `TRACE_MODES` near the top of `test_gemm.py` to
change the sweep. Dtype, fidelity, memory placement, blocking, and iteration counts
are defined there too. Matrix dimensions stay fixed when the grid changes.
`None` in `CORE_GRIDS` allows the device's full available grid; the matmul may
use fewer cores for small shapes.

The current defaults run three shapes on three grids in ordinary and trace modes.
Each case runs five timed batches of 20 matmuls, giving 90 CSV rows. Each row holds
the swept values, batch number, average microseconds per matmul in that batch, and
TFLOPS. Compilation, warmup, allocation, input transfers, correctness checks, and
CSV writes are outside timing. Timing includes host submission and the wait for
device completion. Trace mode captures 20 matmuls once, warms up one replay, and
times subsequent replays using the same input and output storage.

Regenerate plots from the newest CSV, without accessing a device:

```bash
source python_env/bin/activate
python research/gemm/plot.py
```

To choose a saved run and an optional output directory:

```bash
python research/gemm/plot.py research/gemm/results/2026-10-26_19-03-58-582293_EDT/samples.csv
python research/gemm/plot.py path/to/run.csv --output-dir path/to/plots
```

By default, plots go beside `samples.csv` in the run directory. Replotting
replaces those images and leaves the CSV unchanged. The script
uses pandas and Matplotlib from `python_env` and produces:

- `*_by_shape.png`: TFLOPS versus shape, using the full grid and a fixed trace mode.
- `*_by_grid.png`: TFLOPS versus allowed grid, using a fixed shape and trace mode.
- `*_by_trace.png`: TFLOPS versus matrix size on the full grid, with separate
  traced and untraced lines. Sizes are ordered by `M × K × N`.

Each figure has one axis. The shape and grid experiments hold other variables
fixed; the trace comparison shows both modes across matrix sizes on the full grid.
The defaults use the largest recorded grid (assumed to be the full grid), the
shape with the largest `M × K × N` for the grid experiment, and trace mode enabled
for the shape and grid experiments. Edit
`BASELINE_GRID`, `BASELINE_SHAPE`, and `BASELINE_TRACE` near the top of `plot.py`
to change those baselines. Titles show the fixed settings.

Plots use clean lines with small markers, with one median TFLOPS value per setting.
Individual batch measurements remain in the CSV. Each dtype/fidelity pair gets its own figures.
Older CSVs without `use_trace` use ordinary execution and skip the trace comparison.
Use CSVs from passing runs; a failed run can leave partial measurements behind.

`plot_by_grid()`, `plot_by_shape()`, and `plot_by_trace()` are independent
functions in `plot.py`. Change them or add another function and call it from
`main()` as the analysis develops. The runner calls the plotting script after a
successful sweep; the script can also be run independently on saved results.

Ordinary pytest discovery skips this benchmark. For a direct invocation, set
`TTNN_RUN_GEMM_BENCHMARK=1` before running pytest.
