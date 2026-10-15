# Cosmos3-Super full quality extension

This is a new experiment root. Edge/Nano manifests, outputs, scores and runtime
sources remain unchanged. Shared-resource times are not performance evidence.

## Frozen scope

- Official `nvidia/Cosmos3-Super`, revision
  `f543c56225b2e04d0ad141e29655be3a45d9c455`.
- Complete RBench (650 scenes) and PAI-Bench-G (1,044 scenes), two existing
  paired seeds per scene: **30,492 videos across nine methods**.
- Same inputs/prompts, 320×192, 61 frames, 24 FPS, 35 steps, CFG 6, flow shift 3,
  HostScheduleUniPC, no audio or prompt-template change.
- BF16, native FP8 W8A8, ModelOpt NVFP4 AWQ-lite W4A16 decoded quality reference,
  plain W4A4, ARCQuant 50%/100%, SharQ, ours 1.5×/2×.
- External adapters and FP8 retain BF16 attention; plain/repaired W4A4 retain
  FP4 attention. These are complete method comparisons, not matched attention
  ablations. AWQ decoded weights do not establish packed-weight speed/storage.

## Super-specific implementation

The published checkpoint contains about 32B understanding and 32B generation
parameters. BF16 I2V weights need about 121.7 GiB on disk and exceed one 96GB GPU.
`super_runtime.py` places generation on physical GPU0 and understanding on GPU2.
Both CFG branches receive a fresh BF16 AR prefill per request; its K/V/output
buffers are copied before the generation Graph executes. No prior request/step
output is reused. A duplicated text embedding is recorded as extra storage.

The existing quantizers, sampler, repair policies and generation tools execute
unchanged through this isolated process adapter. Teacher intervention packing
is released after each local operator evaluation to avoid retaining another
two full FP4 weight planes alongside the complete BF16 teacher. Deployed models
retain normal resident packed assets. CPU and GPU memory admission precedes work.

ARC's upstream K16 packer would launch 1,600 threads at K=25,600. The isolated
`super_arc_build.py` port assigns independent groups to 256-thread CTAs and
gathers the same BF16 values. It retains all scalar quantization operations and
the official GEMM. `super_arc_codec.py` checks packed codes and E4M3 scale bytes
against an independent reference, including random channel order, zero inputs,
tails and 0/50/100% repair. Independent GEMM checks, memcheck and racecheck are
required. The build receipt retains both the failed instantiation-only build and
the additional shape-port patch; this is a Cosmos adaptation, not an unmodified
upstream binary or a tuned performance result.

## Order and outputs

1. Verify every downloaded weight against the pinned Hub LFS checksum.
2. BF16 development smoke, then the original 16-scene full-M teacher capture.
3. Recalibrate all Super linears, FFNs, head/stage quotas and AWQ/channel orders.
   The established 1.5× protected-N and 2× joint recipes are used. The 2.5×
   intermediate is needed by the existing bounded coordinate-search ABI; it is
   not added to the requested final matrix.
4. All nine methods must pass two development scenes, paired seeds and A→B→A.
   Every full-corpus shape is checked against the request MMA cap before use.
5. A resumable queue runs one two-GPU Super process, with 12-scene chunks.
   Exported MP4 and verified lossless FFV1 accompany every float32 NPZ, complete
   request receipt, source hashes and generation log.
6. Full-frame LPIPS/SSIM/PSNR/L2 compare each method with **Super BF16**, not Edge
   or Nano. The cached metric implementation is independently checked first.
7. The completed nine-method Qwen judge protocol is copied to a new isolated
   Super runtime. Only the model list and multiplicity family change (36 pairs
   × two benchmarks = 72). All questions, frame sampling, model revision,
   parsing, scene clustering and 100,000 bootstrap draws remain unchanged.
   No Edge/Nano scores are reused. Judge services run in a separate 48/64 GiB
   bounded CPU-memory cgroup after generation and pixel metrics finish.

Full-corpus tables include calibration/development scenes, as the existing full
cohort did. Per-scene prior roles are retained; this does not establish unseen
scene generalization. Pixel metrics are consistency measurements; Qwen scores
are task/semantic judgements, not robot execution success.

The root is `/home/yufeng/thunderworld/super_quality_20260928`:

- `campaign.json`, `model.lock.json`, `dependencies_verified.json`: definitions.
- `progress.json`, `generation_queue.json`: progress and exact pause reason.
- `super/bootstrap`, `super/capture`, calibration folders: readiness evidence.
- `super/final/generation`, `super/videos`, `super/judge_index.jsonl`: outputs.
- `metrics/report.md`, `metrics/scenes`: pixel comparison and raw frame metrics.
- `judge/report`, `judge/results`, `judge/responses`: task scores and raw answers.

Run from the repository with the isolated dependencies:

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/home/yufeng/thunderworld/super_quality_20260928/deps
export TW_LPIPS_CACHE=/home/yufeng/thunderworld/super_quality_20260928/lpips_cache
export OPENBLAS_NUM_THREADS=1
bash tools/fp4_handoff/python.sh tools/contribution_study/super_quality_campaign.py status \
  --root /home/yufeng/thunderworld/super_quality_20260928
```

Use `prepare` once, and `run` to start/resume after resolving a pause. Only one
campaign process may own its lock. Downloads and unavailable GPU capacity can
be waited for; OOM, changed sources, failed numerical checks or corrupt outputs
pause without reducing precision, dimensions, steps or corpus coverage. The
first real Super smoke, rather than Edge/Nano timings, determines ETA estimates.
