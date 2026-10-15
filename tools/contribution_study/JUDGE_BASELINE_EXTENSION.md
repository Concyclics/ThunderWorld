# Full I2V baseline supplement

This extends `judge_full_i2v_20260919` without modifying its five-method outputs.
The frozen supplement is `/home/yufeng/thunderworld/judge_full_baselines_20260922`.

## Current resource policy (2026-09-22)

Edge generation and export completed before the device transition. The remaining
Nano queue now runs on **GPU0 and GPU2, two independent Nano processes per GPU**.
GPU1 is no longer used by this queue. The four slots qualify all four methods
against historical output hashes and A→B→A before claiming generation work.

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_baseline_devices.py launch \
  --root /home/yufeng/thunderworld/judge_full_baselines_20260922 --gpus 0 2 --slots-per-gpu 2
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_baseline_devices.py status \
  --root /home/yufeng/thunderworld/judge_full_baselines_20260922
```

Use this entry point to inspect or resume the current queue. The old launch
commands below describe the initial schedule and cannot resume the new policy.
The new controller records the physical GPU UUIDs and requires 84 GiB free on
each GPU at startup: two 36 GiB Nano reservations plus 12 GiB headroom. Prior
resource policies and queue state are archived under `resource_transitions/`.
Generation manifests and NPZ resume identities are unchanged; partial tasks
retain completed seeds and resume through their original paths.

| Model | RBench scenes | PAI-Bench scenes | Seeds per scene | Methods | New videos |
|---|---:|---:|---:|---:|---:|
| Edge | 650 | 1,044 | 2 | 4 | 13,552 |
| Nano | 650 | 1,044 | 2 | 4 | 13,552 |

The original full manifests supply every prompt, image, negative prompt and seed.
Parameters remain 320×192, 61 frames at 24 FPS, 35 UniPC steps, CFG 6.0 and flow
shift 3.0. Resolution/duration prompt templates and audio remain disabled. The
source model revisions and original first-frame/AR-cache settings are retained.

The four methods are the already-qualified full-quality matrix definitions:

- `bf16`: original BF16 projections and attention.
- `fp8`: native W8A8 projections, current per-token activation quantization,
  BF16 attention.
- `awq_reference`: the existing ModelOpt NVFP4 AWQ-lite assets decoded to BF16
  for W4A16 quality comparison. Its decoded resident size and latency must not
  be reported as packed AWQ deployment performance.
- `fp4_base`: W4A4 projections and FP4 attention with zero A/W/block repair;
  all base attention blocks remain present.

All four retain their qualified Graph execution with debug disabled. The
external ARCQuant/SharQ methods in the original run retain their qualified eager
execution; this supplement makes no performance-equivalence claim.

## Initial run and resource policy

From `ThunderWorld-fp4-next-20260918`:

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_baseline_extension.py prepare \
  --root /home/yufeng/thunderworld/judge_full_baselines_20260922 --gpus 0 1 --chunk-scenes 12
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_baseline_extension.py launch \
  --root /home/yufeng/thunderworld/judge_full_baselines_20260922 --gpus 0 1 \
  --slots-per-gpu 4 --secondary-methods bf16 fp8 awq_reference fp4_base --admission requested
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_baseline_extension.py status \
  --root /home/yufeng/thunderworld/judge_full_baselines_20260922
```

Launch runs a persistent supervisor and exporters. Each eligible slot first
reproduces four historical samples per method bitwise and an A→B→A request
transition. The source hashes are mandatory; failure pauses the pool. The queue
then assigns disjoint 12-scene chunks by estimated cost across both GPUs.

Each GPU has four slots: one may run either model; three run Edge. Historical
Nano BF16/AWQ uses about 31 GiB before additional CUDA process overhead, so
launch reserves 36 GiB for Nano, 12 GiB per Edge slot and 12 GiB headroom. Slots
are not filled with extra Nano processes after Edge finishes. A paused pool can
resume with the same command after its own children have exited; existing NPZ
hashes and manifest identities are verified before reuse.

## Outputs

- `pool_status.json`, `queue.json`: current totals and per-task state.
- `parallel_workers/gpu*.slot*/`: slot status, qualification and event logs.
- `<model>/final/generation/<method>/<case>/`: original float32 NPZ and result
  receipts, linked to their task directory with the full generation log.
- `<model>/videos/<benchmark>/<method>/<case>/`: MP4, lossless FFV1 and metadata.
- `<model>/judge_index.jsonl`: incrementally exported video/NPZ/sample mapping.
- `judge_index.jsonl`, `judge_inputs.jsonl`: combined indices at completion.

The expected final totals are 5,200 RBench and 8,352 PAI-Bench videos per model,
27,104 new videos overall. Together with the original 33,880 videos this yields
60,984 videos for nine methods. Shared/concurrent request times are diagnostic;
this run does not update a performance leaderboard.
