# GEMM shape and grid experiments

Run the sweep from the repository root:

```bash
bash research/gemm/run_bench.sh --device-id 0
```

Profiling is enabled by default. The runner activates `python_env`, runs the full
shape/grid sweep through Tenstorrent's standard Tracy tools, combines the device
report with the benchmark's host measurements, and generates plots. Additional
pytest arguments are forwarded. Console output stays in the terminal.

To disable profiling and collect host timing only:

```bash
GEMM_TRACY=0 bash research/gemm/run_bench.sh --device-id 0
```

Every run uses ordinary GEMM execution. TTNN command trace capture/replay is
omitted. `GEMM_TRACY` controls profiling, not command replay.

Edit `WORK_SCALES` and `CORE_GRIDS` near the top of `test_gemm.py` to change the
sweep. One invocation runs every selected fixed-FLOP cohort from `shapes.py`. Dtype,
fidelity, memory placement, and iteration counts are defined in the
benchmark too. Shapes stay fixed when the grid changes. `None` allows the
available compute grid; small shapes may use fewer cores. Profiling does not
restrict the sweep to a single case. The default nine scales and
`CORE_GRIDS=[None]` select 117 shape/grid cases at the full allowed grid:
585 batch rows and 6,825 measured matmuls. The runner produces two combined
images: `BFLOAT16_HiFi2_by_shape.png` contains all shape panels, and
`BFLOAT16_HiFi2_by_grid.png` contains all core-grid scaling panels. Each image
arranges the nine work scales in a 3x3 grid with a shared TFLOPS axis.

Each case has five measured batches. Matmuls per batch are 20 for S=512 through
8192, two for S=16384, and one for S=32768, 49152 and 55296. This is a deterministic
FLOP-based limit shared by all shapes within a cohort. Warmups are the smaller of
five and the matmuls per batch. Both counts are recorded in the CSV and the
actual measured count is included in each Tracy signpost. The default whole-test
timeout is eight hours; `GEMM_TIMEOUT_SECONDS` overrides it.

## Fixed-FLOP shapes (Family B)

For cubic-equivalent side S, the cohort contains `(S,S,S)` and all six
permutations of each of `(S/2,S,2S)` and `(S/4,S,4S)`. Every shape requires exactly
`2*S**3` useful FLOPs. The maximum-to-minimum dimension ratios are 1, 4, and 16.
All dimensions are multiples of 128, so 32x32 tensor tiling requires no logical
shape padding. TTNN's internal block padding and core distribution can still vary.

BF16 A+B+C footprints differ: the cube uses `6*S*S` bytes, moderate shapes
`7*S*S`, and extreme shapes `10.5*S*S`. Equal FLOPs therefore does not hold memory
traffic, arithmetic intensity, or participating core count fixed. Those effects
are part of the observed shape dependence of this implementation.

Inspect the complete scale catalog offline, without importing TTNN or opening a
card; `--scale` restricts the listing and `--csv` saves a new inventory:

```bash
python -m research.gemm.shapes
python -m research.gemm.shapes --scale 512 55296 --csv /tmp/gemm-shapes.csv
```

The catalog covers S = 512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 55296.
Examples of logical BF16 A+B+C footprints are:

| S | FLOPs per GEMM | Footprint range (GiB) |
|---|---:|---:|
| 512 | 268,435,456 | 0.001465–0.002563 |
| 8192 | 1,099,511,627,776 | 0.375–0.65625 |
| 32768 | 70,368,744,177,664 | 6–10.5 |
| 49152 | 237,494,511,599,616 | 13.5–23.625 |
| 55296 | 338,151,365,148,672 | 17.085938–29.900391 |

The largest cohort is intended to approach one p150's memory capacity. For
example, `(13824,55296,221184)` uses 29.900391 GiB. The p150 has
[32 GB GDDR6](https://github.com/tenstorrent/tenstorrent.github.io/blob/main/core/aibs/blackhole/index.md).
These estimates exclude runtime reservations, allocation padding, temporary
buffers, and host reference storage. The four cards' memory is not pooled by
this single-device benchmark.

The default uses **one manual reuse policy** for every shape and allowed grid:
2D multicast, output blocks up to 8x8 tiles, K chunks up to eight tiles, and
compatible register subblocks up to four tiles with FP32 accumulation. Small
partitions clamp the target block dimensions. The policy is deterministic; it
does not search configurations using measured performance.

Per-core output ownership rounds up to block multiples. For the S=32768 cube,
the natural 103x94 partition becomes 104x96 tiles, allowing 8x8 output blocks and
2x2 register subblocks. The completed out-of-box run chose 103x1 and 1x1 blocks
instead and measured about 24.5 TFLOPS. Avoiding the prime-divisor restriction
should remove that particular configuration cliff; a speedup still needs hardware
validation. Shape-dependent traffic, padding, core utilization and other effects
can still produce performance differences.

Logical tensors and useful work stay fixed. CSV rows record effective blocking,
extra output tiles in partial edge blocks, and `blocking_policy=manual`.
Principal BF16-input/BF16-output/FP32-partial buffers are bounded to 1 MiB; an
8x8 block with K chunk eight uses 896 KiB. This estimate excludes runtime and
profiler reservations. Actual L1 allocation remains authoritative.

The implementation is `MatmulMultiCoreReuseMultiCastProgramConfig`, also used by
GEMM_FLOPS's tuned DRAM mode. The earlier out-of-box benchmark is preserved in
commit `431fde920d8`. Keep its measurements separate from this manual policy's
results. Earlier local recipes, including 1x1 output blocks and `reuse`, also
remain identifiable in their original CSVs.

Numerical settings remain BF16/HiFi2, FP32 destination accumulation enabled,
approximate math disabled, and packer L1 accumulation enabled. `GEMM_FLOPS` uses
FP32 destination accumulation disabled and approximate math enabled, including
in its `oob` mode. Matching the implementation family alone does not make the
settings identical or guarantee matching throughput. See [the reference benchmark](../../tests/ttnn/unit_tests/benchmarks/test_benchmark.py).

Before allocating a case, the benchmark checks currently free, contiguous DRAM
per bank. The estimate includes rounding each of the three BF16 allocations to
whole tile pages per bank and 64 MiB of additional total headroom. Insufficient
space fails clearly and preserves completed rows; shapes are not silently reduced
or omitted. The allocator remains authoritative for actual allocations.

Inputs use normal FP32 random samples rounded into BF16, generated in chunks of
16,777,216 elements. A reusable FP32 scratch buffer is at most 64 MiB; there is no
full-size FP32 input allocation. This avoids the slower direct BF16 random
generation path. Inputs are uploaded one at a time, releasing each full host
input after retaining the reference data needed for validation. The fixed seed
and chunk size make runs reproducible, but values differ from earlier
`torch_bfloat16` runs. CSV rows record the generation method and chunk size.

Preparation messages show the start and host elapsed time for generating A/B,
TTNN conversion/upload calls, CPU references and output allocation. These stages
precede the `Validation:` print and all timed GEMMs. The full default sweep
generates about 476 GiB of BF16 input data across its cases, so preparation and
transfer costs can remain substantial even when GEMMs are fast.

Cases through S=2048 use a full FP32 CPU reference and
full output readback. Larger cases validate five 32x32 output tiles: the four
corners and center. Their references use the uploaded BF16 input values and
reduce over the complete K dimension. Tile slicing/readback and all validation
remain outside the measured intervals, before warmup and after measurement.
Sampled validation checks 5,120 output elements; it is not full-output verification.

The nine-scale sweep completed with both earlier 1x1 and out-of-box policies.
CPU checks verify manual blocking across all shapes and five grid sizes,
including the prime-partition cases, and preserve batching and validation. This
manual policy still needs on-card validation across all scales and shapes. The
largest logical tensor alone is 22.78 GiB, so host staging and transfer buffers
still require substantial RAM.

## Saved results

Each invocation creates one directory with a readable New York timestamp:

```text
results/<run-id>/
  shapes.csv                       # selected shapes, FLOP budget, BF16 footprint
  samples.csv
  BFLOAT16_HiFi2_by_shape.png        # all scales' shape comparisons
  BFLOAT16_HiFi2_by_grid.png         # all scales' core-grid scaling comparisons
  profiler/                         # when profiling is enabled
    .logs/
      tracy_profile_log_host.tracy
      profile_log_device.csv
      cpp_device_perf_report.csv
    reports/<timestamp>/
      ops_perf_results_<timestamp>.csv
```

`test_gemm.py` writes and flushes the host measurement after every batch. After a
successful run, `plot.py` matches each host row to the measured operations between
its start/end signposts in Tracy's standard report. It verifies the operation and
sample counts, unique execution IDs, participating cores, and matching case/batch
metadata before adding device columns to `samples.csv`. The CSV update is atomic;
a failure preserves the host CSV and profiler diagnostics and stops before plots.
No custom device-profiler collector is used.

The final CSV records both metrics explicitly:

| Column | Meaning |
|---|---|
| `host_time_us` | Synchronized host batch duration divided by matmul count |
| `host_tflops` | Useful FLOPs divided by host time |
| `device_kernel_time_us` | Mean device kernel duration from Tracy for that batch |
| `device_tflops` | Useful FLOPs divided by device kernel time |
| `core_count`, `device_samples`, `device_id` | Validated profiler metadata |
| `profiling_enabled` | Whether device profiling was collected |
| `shape_family`, `work_scale`, `useful_flops` | Fixed-FLOP cohort identity and work |
| `input_elements`, `output_elements`, `tensor_elements`, `logical_bytes` | Logical BF16 footprint |
| `padded_m`, `padded_k`, `padded_n` | Actual padded tensor dimensions |
| `measurement_iterations`, `warmup_iterations` | Effective repetition counts for that cohort |
| `validation_mode`, `validation_elements` | Full-output or sampled-tile checks and number of checked elements |
| `dram_tensor_allocation_bytes`, `dram_free_bytes_before` | Estimated tile-page allocation and available DRAM before the case |
| `input_generation`, `input_generation_chunk_elements` | `torch_float32_chunked_to_bfloat16` and the maximum FP32 scratch element count; older runs used `torch_bfloat16` |
| `blocking_policy` | `manual` for the fixed reuse policy; older runs include `oob`, `auto_grid` and `reuse` |
| `per_core_m_tiles`, `per_core_n_tiles`, `grid_padding_tiles` | Rounded per-core ownership and extra output tiles in partial edge blocks |
| `out_block_h_tiles`, `out_block_w_tiles`, `out_subblock_h_tiles`, `out_subblock_w_tiles`, `in0_block_w_tiles` | Effective output/register blocks and K chunk |
| `estimated_l1_buffer_bytes` | Principal buffer estimate excluding runtime/profiler reservations |
| `fp32_dest_acc_en`, `math_approx_mode`, `packer_l1_acc` | Numerical/accumulation settings |

Tracy's operation report also records the effective program configuration.
Earlier out-of-box CSVs omit the local block sizes and L1 estimate.

With profiling disabled, host metrics are still recorded on every batch and the
device columns are empty. Profiling enabled requires device results for every
host batch; missing device measurements never fall back to host timing.

The host timer includes Python/TTNN submission and the wait for device completion.
Compilation, warmup, allocation, transfers, correctness checks, signposts, profiler
readback, and CSV writes remain outside it. Output storage is reused. The profiler
is drained after each case, outside timing, to support sweeps beyond its buffer
capacity. Default Tracy capture uses `-p` to avoid tracing every Python function;
standard C++ host/device profiling and the saved GUI capture remain available.

Device timing uses `DEVICE KERNEL DURATION [ns]`, covering the earliest kernel
start to the latest finish across participating cores/RISCs. It includes kernel
data movement and compute but excludes gaps between operations. Microseconds are
`mean(duration_ns) / 1000`; both throughput metrics use `2*M*K*N` useful FLOPs.
Our `host_time_us` is a batch measurement, distinct from Tracy's per-operation
`HOST DURATION [ns]`. Profiling can affect host timing; the host/device difference
does not isolate Python overhead.

The analysis rejects gross contradictions between device and synchronized host
time, including when reloading an already enriched CSV. Invalid measurements stop
analysis while preserving the host CSV and raw profiler artifacts.

After the run, the console shows median host/device timings and TFLOPS per case.
Both combined images overlay host-observed and device-kernel throughput in each
panel, or shows only host throughput when profiling is disabled:

- Shape panels compare all 13 shapes at a fixed allowed grid.
- Grid panels compare the cohort's cube across recorded allowed grids.

`BASELINE_SHAPE` and `BASELINE_GRID` in `plot.py` select those baselines. By default
they use the cohort's cube and largest recorded grid. Legacy files or subsets
without the cube use a largest-workload shape (the first on ties).
Each family/work scale occupies separate panels with its budget in the title,
so different FLOP budgets are not averaged together. Two combined images are
saved per dtype/fidelity configuration, separating shape and core-grid scaling
comparisons. Earlier per-scale and mixed-comparison image files are retained but
are no longer generated.

Regenerate plots from the newest run without a device, or choose an explicit CSV:

```bash
source python_env/bin/activate
python research/gemm/plot.py
python research/gemm/plot.py path/to/samples.csv --output-dir path/to/plots
```

Replotting leaves the CSV unchanged unless `--update-csv` is supplied; the runner
uses that option to save the combined results. Raw per-operation records remain
in Tracy's report and are not duplicated in another custom CSV. Legacy host-only
`time_us`/`tflops` columns are recognized as host metrics. Older CSVs containing
`use_trace` are filtered to ordinary execution. A legacy run with a matching Tracy
report can also be analyzed directly. Use CSVs from passing runs.

The `.tracy` capture can be opened later in Tenstorrent's GUI. To also launch its
web viewer after a run:

```bash
TRACY_NO_WEB_SERVER=0 bash research/gemm/run_bench.sh --device-id 0
```

Tracy prints the viewer URL; remote access uses SSH forwarding of ports 8080 and
8081. See the [Tracy guide](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html).
The earlier `GEMM_DEVICE_TIMING` environment variable remains a compatibility alias
when `GEMM_TRACY` is unset. Previous results under `results/tracy/` and
`results/device_timing/` are retained; pass their CSV paths explicitly to replot.

Ordinary pytest discovery skips this benchmark unless
`TTNN_RUN_GEMM_BENCHMARK=1`. Use the runner to configure default profiling correctly.
