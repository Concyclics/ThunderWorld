# Adapted task judge

The released research implementation uses **Qwen/Qwen3-VL-8B-Instruct-FP8**, revision `9cdc6310a8cb770ce18efaf4e9935334512aee45`, through vLLM. The prompts and scoring rules are pinned in `tools/contribution_study/qwen_task_protocol.py`.

## Protocol and evidence

- RBench: upstream category/task rubrics evaluated on a chronological 3×2 PNG frame grid. Visual-reasoning question chains are generated once per scene and frozen across methods/seeds.
- PAI-Bench-G: official binary VQA questions with constrained YES/NO output, 16 uniform video frames, timestamps preserved. Empty official annotations are excluded from Domain VQA; supplemental adherence is separate.
- No precision or method labels are exposed in prompts. Raw judge responses, parsed scores, retry records, source hashes and environment/server receipts are saved.
- Frame ingestion qualification and repeatability checks precede the campaign. Concurrency is bounded; CPU video preprocessing is also bounded.

These are adapted VLM task scores, not official leaderboard scores, ground-truth video distances or robot execution success. Category-macro and scene-micro scores, coverage and paired uncertainty are retained in [RESULTS.md](RESULTS.md).

## Files and entry points

| File | Purpose |
|---|---|
| `qwen_task_protocol.py` | Pinned model, benchmark evaluator revisions, frame selection, prompts, strict parsing |
| `qwen_task_judge.py` | `prepare`, `qualify`, `score`, `repeat`, `report`; immutable campaign manifests |
| `qwen_task_report.py` | Paired task-score / decision-consistency analysis |
| `judge_video_export.py` | Verified lossless RGB/FFV1 export from original float NPZ |
| `qwen_judge_pool.py`, `qwen_resource_session.py` | Dedicated vLLM replica and resource management |
| `qwen_task_extend_baselines.py`, `super_task_judge.py` | Historical additional methods and Super campaign |

The judge consumes the **historical task inventory schema**, which includes source NPZ hashes, verified lossless export metadata, source evaluator revisions and per-scene annotations. `tools/public/run.py videos` produces viewing MP4s and an index; it deliberately does not pretend those lossy files already satisfy the judge's lossless-ingestion contract. When adapting a new public generation run, use `judge_video_export.py` and `qwen_task_protocol.build_inputs` as the schema reference and retain the source-generation identities. The historical full-corpus worker enforces its fixed cohort cardinality; change and freeze the protocol explicitly for a smaller/new cohort.

For a prepared inventory and dedicated server pool:

```bash
python tools/contribution_study/qwen_task_judge.py prepare \
  --root /path/to/judge-campaign --metrics-root /path/to/task-inventory \
  --cosmos-root /path/to/pinned/NVIDIA-cosmos
python tools/contribution_study/qwen_task_judge.py qualify --root /path/to/judge-campaign
python tools/contribution_study/qwen_task_judge.py score --root /path/to/judge-campaign --video-workers 12
python tools/contribution_study/qwen_task_judge.py repeat --root /path/to/judge-campaign
python tools/contribution_study/qwen_task_judge.py report --root /path/to/judge-campaign
```

`servers.json` names vLLM endpoints serving `qwen3-vl-8b-fp8-judge`; `model_weights_verified.json` records the pinned judge checkpoint. Source evaluator checkouts and PAI VQA files live under the campaign's `sources/` with their revisions verified. Use the resource-session tools to prepare these records on your own devices; do not copy workstation PIDs, ports or paths from historical campaign files. GPU server launch and full judge reruns are not automatic side effects of the standard quality workflow.
