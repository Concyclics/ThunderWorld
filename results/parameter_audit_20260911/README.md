# Parameter and overlap audit

[Interpretation, parameter tables, and next implementation priorities](../../docs/PARAMETERS_AND_OVERLAP_5090_20260911.md).

- `overlap_probe.json`: 16 real-shape configurations, five alternating-order measurements per configuration; same independent producer/GEMM work, serial versus two streams. All checked outputs are bitwise equal.
- `torch_trace_cases.*.chrome.json`: CUPTI CPU/CUDA timelines for Edge M316 a50/act2; view in a Chrome trace-compatible viewer.
- `trace_analysis.json`, `overlap_timeline.png`: five graph replays per tier/mode, kernel-interval overlap, and the median GPU-span replay shown.
- `sources/`, `evidence_sources.json`, `SHA256SUMS`: measured probe version, final tools, audit inputs and file hashes.
- `overlap_trace.qdstrm`, `nsys_import_failure.log`: unsuccessful old Nsight Systems import; not used as timing evidence.

The independent producer has its own input and mask. It is not producing the
input consumed by the concurrent GEMM. Traces establish overlapping kernel
intervals, not per-SM co-residency or simultaneous SIMT/Tensor instruction issue.
Normal benchmark measurements determine speed; profiler times are diagnostic.
Inputs repeat captured 64-row probes to real M for timing, not new quality data.

```bash
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh tools/auto_budget/probe_overlap.py \
  runs5090/auto_budget_edge_bench/benchmark.json runs5090/auto_budget_nano_bench/benchmark.json \
  --output results/new_overlap/probe.json
bash tools/kernel_handoff/python5090.sh tools/auto_budget/probe_overlap.py \
  runs5090/auto_budget_edge_bench/benchmark.json --trace --torch-trace \
  --output results/new_overlap/torch_trace_cases.json
bash tools/kernel_handoff/python5090.sh tools/auto_budget/analyze_overlap.py results/new_overlap
```
