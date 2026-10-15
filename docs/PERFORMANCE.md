# Performance qualification

The frozen model launcher prioritizes reproducible quality; its shared-resource timings are diagnostic. The paper's measured performance and the current machine's performance are separate evidence. No new latency claim is made by assembling this open-source release.

## Measurement units

- The paper's FPS numerator is **60 generated frames**, excluding the conditioning frame. Save all 61 frames, but do not replace 60 with 61 when reproducing Figure 6.
- Include fresh request-local AR prefill, online quantization/selection/packing, Transformer work and VAE decode in complete-request latency.
- Report cold loading, offline weight preparation, compilation and Graph capture separately.
- Record the exact device, SM architecture, clocks/power conditions, software versions, debug state, batch/concurrency and execution mode.
- Report median and P90 warm request latency, throughput, resident weights, scratch and allocator peaks. Asset bytes alone are not total memory.

Formal speed comparisons require an idle device and `debug=off`, matching inputs/settings, complete current-device tuning-table coverage, five interleaved rounds with three fixed seeds per configuration, and raw request measurements. Shared calibration prohibits formal benchmark/tuning. A short kernel probe or concurrent throughput stress test does not establish output quality; concurrency-only tests must be labeled accordingly.

## Why FP4 peak arithmetic is not request speed

The compute cap concerns issued MMA, including padding and repair intersections. It does not remove activation quantization, amax/reductions, selectors, memory traffic, normalization, softmax, launch overhead, AR work or video decode. Different shape mixes and memory/SM resources can change which stage dominates on PRO 6000, 5090 and Thor. Do not multiply separate probe, Graph and kernel speedups to claim an end-to-end gain.

## Runtime and tuning entry points

`tw-cosmos warmup --tune` records actual request shapes; `tw-cosmos benchmark --tuning ...` requires strict coverage and idle-device conditions. Supply explicit base-model, config, assets and case paths. Use the public quality runner first to establish correctness. Example for a newly prepared research configuration:

```bash
export CUDA_VISIBLE_DEVICES=0
tw-cosmos warmup --model /path/to/base --config /path/to/config.json \
  --assets /path/to/assets --case /path/to/case.json \
  --resource-mode exclusive --ar-cache --first-frame \
  --tune --output work/timing-warmup

tw-cosmos benchmark --model /path/to/base --config /path/to/config.json \
  --assets /path/to/assets --case /path/to/case.json \
  --resource-mode exclusive --ar-cache --first-frame \
  --tuning work/timing-warmup/tuning.json --debug off \
  --seeds 93001 93002 93003 --output work/timing-round1
```

Preserve the frozen profile's additional runtime switches in comparisons. The generic commands above are a development example, not an assertion that arbitrary flags reproduce the paper. `tools/contribution_study/baseline_throughput_suite.py`, `baseline_sweep.py` and `experiments/cutlass_fused_repair/joint_e2e.py` retain the scoped research protocols. Their historical path defaults require explicit replacement on a new machine.

## Optional CUTLASS backend

```bash
python -m thunderworld_fp4.cutlass.build \
  --cutlass /path/to/pinned/cutlass --nvcc /path/to/nvcc \
  --tile-n 128 --output work/build/fp4-repair-n128.so
python -m thunderworld_fp4.cutlass.build \
  --cutlass /path/to/pinned/cutlass --nvcc /path/to/nvcc \
  --tile-n 64 --output work/build/fp4-repair-n64.so
```

Supply both libraries through `tw-cosmos --linear-backend cutlass-fused --cutlass-library ... --cutlass-small-library ...`. Build receipts pin source, compiler and binary identity. Do not assume an SM120 launch table tuned on 5090 is optimal on PRO 6000. CUTLASS changes accumulation order relative to the frozen Triton path; independent GEMM checks alone cannot certify unchanged generation quality.

## Correctness before speed

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q packages/fp4-repair/tests/test_compute_cuda.py
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --error-exitcode 1 \
  python packages/fp4-repair/tests/sanitizer_joint.py
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool racecheck --error-exitcode 1 \
  python packages/fp4-repair/tests/sanitizer_joint.py
```

The supplied sanitizer driver covers the native dense-W / attention cases it names, including M129/M416, GQA and empty/full repair. Run backend-specific reference and sanitizer tests for any newly modified CUTLASS kernel as well; the native driver alone does not cover every optional binary. Recheck full trajectories whenever quantization or accumulation semantics change.

## Device handoffs

`experiments/cutlass_fused_repair/README_5090.md`, `tools/fp4_handoff/` and `tools/thor_handoff/` retain device experiments and recipes. They are research handoff records, not the pinned Hub launcher. Large original handoff payloads are intentionally not bundled in Git; use the public model downloads and calibration workflow to construct inputs for new studies. Thor is SM110 and uses its own native attention implementation and toolchain. Its results must not be relabeled as SM120 / 5090 results.
