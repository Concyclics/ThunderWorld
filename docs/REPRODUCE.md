# Reproduce quality comparisons

The public workflow freezes the model, code, assets, cases, seeds and method settings before generation. It reuses the same underlying runtime and adapters as the research experiments, while removing dependencies on the original workstation directory layout.

## 1. Prepare data and dependencies

Follow [INSTALL.md](INSTALL.md). The FP8 baseline additionally requires the ABI-compatible vLLM native extensions. Download the pinned NVIDIA model, then reconstruct inputs:

```bash
python tools/public/data.py --model edge --base-model /path/to/Cosmos3-Edge \
  --output work/edge
```

This downloads 136 unique scenes across calibration/development/validation/final roles. `--full` also downloads the full 650 RBench and 1,044 PAI-Bench-G inputs. The public full-cohort rerun freezes a new two-seed-per-scene sequence starting at 501001; it is not advertised as byte-for-byte replay of every historical benchmark video. Frozen release loading is separately checked against the historical signal hashes.

For baseline adapters, prepare assets/builds using [BASELINES.md](BASELINES.md). For just the five core methods, no calibration run is needed: the default manifest downloads the frozen ThunderWorld assets.

## 2. Freeze

```bash
python tools/public/run.py prepare --model edge --work work/edge --split final
```

Defaults: BF16, native FP8, ordinary FP4 W4A4, ThunderWorld 1.5× and 2×. To evaluate the complete corpus, use `data.py --full`, then `--split full`. `--release-dir` reuses a local model release. `--candidates` plus `--methods` selects new calibration candidates; those are new experiments, not automatically promoted releases.

All methods generate 320×192, 61 stored frames, 35 steps, CFG 6, flow shift 3, with the same prompt, image, negative prompt and seeds. AR caching is request-local and fresh prefill is included. The quality manifest uses `debug=off`, Graph for native paths and the documented eager execution for external adapters. The full corpus overlaps earlier development; neither this setup nor same-scene new seeds proves cross-scene generalization.

## 3. Generate and resume

```bash
CUDA_VISIBLE_DEVICES=0 python tools/public/run.py generate --model edge --work work/edge
python tools/public/run.py videos --model edge --work work/edge
```

The public runner executes **one worker at a time**. Select a free device with sufficient memory; Super requires two large devices (`CUDA_VISIBLE_DEVICES=0,1 --model super`) for the documented DM/AR placement. This workflow intentionally does not launch an uncontrolled multi-process queue.

Each case has `generation/METHOD/CASE/result.json` and `seedSEED.npz`; progress and prepare receipts are retained. Resume validates manifest, code, assets and output hashes before skipping completed seeds. Changing an experiment requires a fresh manifest **and** output directory; use `--evaluation-dir work/edge/another-run` on generate/metrics/report/videos to keep runs separate. The video export writes MP4s plus a `videos.json` index for later judging; metrics use original float NPZ arrays, not compressed MP4 pixels.

## 4. Score

```bash
python tools/public/run.py metrics --model edge --work work/edge
python tools/public/run.py report  --model edge --work work/edge
```

LPIPS (AlexNet v0.1), SSIM, PSNR, RMSE and relative L2 are computed against the paired BF16 output. LPIPS may download its pretrained feature-network weights once. Complete clips are required for a report. Seeds are averaged within a scene; categories are averaged equally. Scene-clustered, category-stratified bootstrap intervals and familywise intervals compare candidates with FP8. JSON also retains individual-scene and individual-seed results and worst-case differences.

PSNR uses the entire RGB clip's MSE with `data_range=1`. Exact equality means positive infinity, serialized as `null` plus an exact-match flag; BF16's self-reference is not plotted as an arbitrary finite PSNR. Pixel consistency is not a robot-task success rate.

The report writes `report/quality.json`, `report/costs.json` and a Markdown table. Request compute bounds and allocator peaks are separated from timing; shared-GPU elapsed times are never promoted to acceleration claims.

## Judge and historical experiment tools

The complete Qwen3-VL-8B-Instruct-FP8 protocol, prompt adaptation, bounded vLLM concurrency, ingestion checks, repeatability checks and resumable JSON logs are in:

- `qwen_task_protocol.py`, `qwen_task_judge.py`, `qwen_task_report.py`;
- `qwen_judge_pool.py`, `qwen_resource_session.py` for resource-controlled serving;
- `super_task_judge.py` and the full-benchmark inventory/metric tools for model extensions.

These drivers consume the historical inventory schema and explicit judge server manifests; they are not automatically started by `run.py`. Their CLI accepts paths and preserves raw responses. See [JUDGE.md](JUDGE.md) for the inventory adaptation boundary and endpoint contract. Judge scores are adapted task proxies, not the official benchmark leaderboard or physical robot execution accuracy.

## Evidence already included

[RESULTS.md](RESULTS.md) contains all three models × both benchmarks, with links to source CSVs and model-release qualification receipts. The [Hugging Face benchmark dataset](https://huggingface.co/datasets/Concyclics/thunderworld-benchmark) hosts generated artifacts separately. Full videos and original checkpoints are intentionally outside Git.
