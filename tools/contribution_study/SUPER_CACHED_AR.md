# Super: exact AR prefixes and two independent DM workers

This continues `/home/yufeng/thunderworld/super_quality_20260928`. The model,
quantization assets, repair plans, scene/seed pairs, 35 steps, CFG, resolution,
frames, numerical kernels and frozen generation manifest remain unchanged.
Previously completed NPZ files, videos and receipts are retained.

The original deployment placed the BF16 understanding branch on GPU2 and the
generation branch on GPU0. The understanding branch only computes a text prefix
for each CFG branch at request entry. Its outputs are independent of the video
noise, seed, denoising step and generation-side quantizer in this I2V protocol.

`super_prefix_prepare.py` uses the actual pipeline input assembly to capture
both CFG prefixes and the original graphed BF16 AR operation order to produce
per-layer K/V/output tensors. It saves exact BF16 tensors. The dummy conditional
velocity only lets input assembly reach the first unconditional call; collection
aborts there, before a denoising step or decoded output is produced.

Cache keys contain all actual input IDs, text indexes, position IDs, sequence
lengths and modality boundaries. The bank records the model identity and hashes
of every tensor file. A new request validates the current prefix once per CFG
branch. Missing or changed inputs fail; no unrelated prefix is substituted.
All noisy generation inputs and online activation repair remain current.

`super_cached_runtime.py` releases unused AR weights after the unchanged official
CPU checkpoint load and keeps only the DM branch on its single GPU. Changed
equal-shape prefixes update the same GPU cache buffers used by captured graphs.
There is no per-step prefix hash, host copy, prefill or diagnostic event.

Before deployment, `super_cached_campaign.py qualify` requires both GPU0 and GPU2
to reproduce the original full-video signal hashes for all nine methods, two
development scenes, and two seeds per scene, plus A-B-A round trips. Compute
budget ratios must also match. The deployment manifest then freezes the new
runtime hashes, cache bank and qualification receipt separately from the
historical numerical manifest. New request rows identify this deployment and
the exact prefix keys they consumed.

The scheduler preserves the original 12-scene task identities. Each resident
session processes up to four chunks of one method, reducing model reloads.
GPU0 and GPU2 each own one independent DM worker. Per-device leases, initial
88 GiB free-memory admission and a bounded host-memory cgroup prevent accidental
oversubscription. Partial tasks resume through the original hash-checking worker.
Existing CPU export, pixel metrics and Qwen judge run after/alongside generation
as before. Failures retain receipts and pause the new queue.

Offline prefix preparation time and disk bytes are recorded in
`cached_ar_deployment_20261003/prefix_bank/manifest.json`. Request timing now
excludes that offline work and is explicitly labeled accordingly. These are
fixed-corpus workflow throughput diagnostics, not an exclusive-GPU model-speed
comparison or a claim about previously unseen prompts.

From the repository, with the same Python wrapper and persistent dependencies:

```bash
ROOT=/home/yufeng/thunderworld/super_quality_20260928
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH="$ROOT/deps"
export TW_LPIPS_CACHE="$ROOT/lpips_cache"
export OPENBLAS_NUM_THREADS=1

# After a qualified deployment exists, resume only when no other campaign owns
# campaign.lock. This preserves task identities and already completed outputs.
bash tools/fp4_handoff/python.sh tools/contribution_study/super_cached_campaign.py run \
  --root "$ROOT"
```

Operational progress remains in `progress.json` and `generation_queue.json`.
`active_jobs` maps physical GPUs to their processes, methods and chunk IDs.
Full logs and the previous queue snapshot are under
`cached_ar_deployment_20261003/`.

The 2026-10-03 rollout passed all 18 method/device checks: 72 full-video outputs
matched the original signal hashes, with 18 successful A-B-A checks and unchanged
MMA ratios. All 7,602 pre-existing case receipts (15,204 videos) retained their
hashes. `rollout_audit.json` records that audit and the first production batches;
the second W4A4 chunk produced new outputs in the same process without another
checkpoint load. Use the live root `progress.json` for later progress.

The bank contains 855 prefix assets for 854 cases (852 remaining benchmark cases
and two qualification cases), occupying 189,527,764,067 bytes, about 176.5 GiB.
This is additional offline disk storage. The largest observed single-GPU reserved
memory in the qualification suite was about 64.2 GiB. Initial throughput samples
cover only the methods active in those batches, exclude cache preparation and
qualification, and must not be extrapolated into a whole-campaign speedup.
