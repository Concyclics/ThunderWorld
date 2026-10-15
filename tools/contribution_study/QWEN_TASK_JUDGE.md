# Task fidelity and decision agreement with Qwen3-VL

This evaluates the **existing** Edge/Nano videos with
`Qwen/Qwen3-VL-8B-Instruct-FP8` at revision
`9cdc6310a8cb770ce18efaf4e9935334512aee45`, using vLLM continuous batching.
The comparison is BF16, native FP8, `b1.5_protected_n` (our 1.5× policy),
and `b2_joint` (our 2× policy).
It does not regenerate videos or change quantization policies.

## Frozen scope

- Edge and Nano; full RBench 650 scenes and PAI-Bench-G 1044 scenes.
- Two original seeds per scene, four precision methods: **27,104 videos**.
- Original 320×192, 61 frames, 24 fps, 35 steps, CFG/graph settings retained.
- **100,704 judge requests**, plus 50 shared text-only question chains and
  ingestion/repeatability controls. Every request/result can resume independently.
- CPU tests cover answer leakage, missing annotations, invalid outputs,
  task-vs-answer agreement, score arithmetic and scene-cluster statistics.

This is a **Qwen judge adaptation**, not an official leaderboard score or a real
robot execution success rate. BF16 is a comparison video, not task ground truth.
All methods receive identical scoring prompts and sampling. Opaque filenames and
blind prompts hide method/model/precision labels from the judge.

## Benchmark definitions

### RBench task VQA

Use the official [ReVidgen evaluator](https://github.com/DAGroup-PKU/ReVidgen) at
`b03df27f0376faa148dcd8cd620a1989a32ca979`:

- Five task-category rubrics, retaining all category-specific aspects.
- Task-adherence rubric for the four embodiment categories.
- Official two-stage sampling (16 frames → 6 frames), chronological 3×2 grid.
  For these 61-frame videos, indices are `[0,12,24,36,48,60]`.
- Grid pixels come from verified lossless FFV1 exports and lossless PNG encoding.
- Visual-reasoning question chains are generated once per scene from its text,
  before comparing videos, and frozen across models, precisions and seeds.
- The primary 1–5 task VQA score recomputes the rubric's arithmetic and upstream
  final clamp deterministically. The judge's original total and the corresponding
  unmodified upstream final score are retained, including arithmetic discrepancies.

Motion smoothness/amplitude, subject stability and the official full composite
score are outside this task-VQA comparison. The short generation configuration
also differs from the benchmarks' higher-resolution official reproduction setup.

### PAI-Bench Domain VQA

Use the official [PAI evaluator](https://github.com/SHI-Labs/physical-ai-bench) at
`2f3b687410029b98397fbc51fa4de36bfd45627d`, questions from
`shi-labs/physical-ai-bench-generation` at
`c36b7beb69bd7c4adcedc1431407fce98f9dfe44`.

- All **5,636 official binary questions** on 1,036 annotated scenes.
- One question per request; the official system/user templates are preserved.
  Reference answers are never sent to the model. YES/NO output is constrained.
- Native video input: 16 uniform frames from the original lossless video, true
  24 fps timestamps and no second sampling. This replaces upstream 2 fps sampling
  and is explicitly part of this adaptation.
- Eight official question files are empty. Their 128 videos receive separate
  supplemental prompt-adherence scores, **never fabricated Domain accuracy**.
- Domain accuracy: questions → seed → scene. Category macro and scene micro are
  both retained. Label-balanced accuracy and an always-YES baseline expose
  the official question set's substantial YES imbalance.

## Outputs and interpretation

`report/summary.csv` reports each method's absolute task scores.
`report/paired_comparisons.csv` reports 1.5×/2× versus BF16/FP8, 1.5× versus 2×,
and FP8 versus BF16:

- Paired score differences and stratified scene-cluster bootstrap intervals,
  keeping both seeds and question sets together; 95% and family-24 intervals.
- PAI exact YES/NO agreement; retention on reference-correct question instances.
- RBench exact task-aspect rating agreement, within-one-point agreement, and
  agreement of the prespecified rating≥4 indicator. The latter is not called
  task success.
- Mean absolute score differences and worst-scene degradation.

A confidence interval crossing zero is not proof of equivalence. High agreement
can mean that both variants fail a task; use it alongside absolute task scores.
`judge_repeatability.json` records fresh judgements in another batch order on a
fixed, score-independent subset. Repeats never replace primary answers.

## Local operation

Repository: `/home/yufeng/thunderworld/ThunderWorld-fp4-next-20260918`

Run root: `/home/yufeng/thunderworld/qwen_task_judge_four_methods_20260923`

The earlier three-method run is preserved at
`/home/yufeng/thunderworld/qwen_task_judge_20260923`. Its completed answers can be
imported with `reuse --from-root` only after verifying identical video sources,
media hashes, questions, judge protocol and parsed scores. The old scoring code
is snapshotted and hashed. Each reused result retains its original provenance.
The initial extension reuses the running service; `servers.json.control_root`
identifies the service owner for cleanup.

The launcher checks free memory, holds the existing shared/global and
exclusive/per-device resource leases, binds only to localhost, limits model
memory, and cleans up its own service process groups. It never terminates other
users' GPU processes. Start one replica per available GPU, not multiple copies
of the model on a GPU; the replica batches up to 32 simultaneous requests.

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
ROOT=/home/yufeng/thunderworld/qwen_task_judge_four_methods_20260923
MODEL=/home/yufeng/thunderworld/models/Qwen3-VL-8B-Instruct-FP8-9cdc631

# Run the service in a dedicated terminal or detached process. Add GPU2 only
# when its free memory meets the launcher's reserve check.
bash tools/fp4_handoff/python.sh tools/contribution_study/qwen_judge_pool.py \
  --root "$ROOT" --model "$MODEL" --gpus 0

# In a second terminal: prepare → qualify → pilot → repeat → full → report.
bash tools/fp4_handoff/python.sh tools/contribution_study/qwen_task_campaign.py \
  --root "$ROOT"

# Inspect scores without contacting the model. Partial reports are labelled.
bash tools/fp4_handoff/python.sh tools/contribution_study/qwen_task_judge.py \
  report --root "$ROOT"
```

The full run writes `STOP_SERVERS` on completion or failure. Before resuming,
inspect `campaign_failure.json` and the failed stage log, correct the cause,
remove that stop sentinel explicitly, restart the pool and campaign. Successful
responses retain their frozen identities and are reused. A protocol/code change
requires a fresh manifest and run rather than silently mixing results.

The server uses an isolated `ROOT/deps/flashinfer-python==0.6.13` overlay,
matching vLLM 0.25.1's requirement and installed `flashinfer-cubin==0.6.13`.
This fixes the shared environment's mismatched Python package without changing
the generation environment. vLLM research plugins are disabled; Ninja is resolved
from its installed environment. Weights were SHA256-verified against Hub metadata.

Inspect `campaign.json`, `pilot_progress.json`, `progress.json`, per-stage logs,
`results/`, and `responses/`. Raw model replies, retry failures, token usage,
prompt/source hashes, media sampling and original video links remain available.
Scores are complete only when `report/complete.json` exists with all 27,104 videos.
Shared-GPU scoring throughput is not used as a generation performance result.

## Recovery without changing the frozen judge

`qwen_task_resume.py --root "$ROOT"` validates every completed result against
the frozen jobs, questions, parser and aggregation, counts all existing results
immediately, requalifies the currently running service replicas and resumes only
missing videos. It does not change prompts, model sampling or media selection.

A malformed or truncated answer is isolated while other videos continue. Up to
three queue passes use the original three-attempt request policy; replicas rotate
between passes. The first valid answer is retained regardless of its score. Old
attempts are preserved in `retry_history/`, and recovered failures are moved to
`resolved_failures/`. A persistent invalid answer is never assigned zero or
marked complete. The final report remains explicitly partial if any are unresolved.
Service/integrity failures still pause the queue. `progress.json` reports actual
saved-video coverage, including imported results, and the current run's rate.

On 2026-09-23 the original queue paused on one repetitive, length-truncated RBench
response after 5,962 valid videos. The recovery runner preserved those results,
resumed on GPU0/2 with the identical protocol, and successfully rejudged that item.
Fault-injection tests verify isolated retry, bounded persistent failure, and the
absence of fabricated scores or false completion markers.

## Ordinary W4A4 extension (2026-09-24)

The independent run root is
`/home/yufeng/thunderworld/qwen_task_judge_five_methods_20260924`.
It adds **6,776 `fp4_base` videos / 25,176 judge requests**, covering both models,
both complete benchmarks and the same two seeds per scene. Its full comparison
contains **33,880 videos / 125,880 primary judge requests**.

`fp4_base` uses FP4 generation projections, FFN and attention, with zero A/W
repair, no cross repair and zero attention repair coverage. All base attention
blocks remain. Every source export was checked against this definition, alongside
the existing 320×192, 61-frame, 35-step generation settings.

`qwen_task_extend_w4a4.py` prepares a private `runtime_code/` snapshot. Only the
method list and report comparison set change; the old run and its frozen parsing,
prompts, question chains and sampling are retained. All **27,104** previous valid
results are imported after verifying their input/media identity, parsed scores,
and complete reuse provenance, including answers inherited from the initial
three-method campaign. Tests reject edited responses and broken source receipts.

The four new comparison pairs are W4A4−BF16, W4A4−FP8, 1.5×−W4A4 and 2×−W4A4.
Together with the original six pairs, across four model/benchmark groups, the
new report uses **family-40 Bonferroni intervals** and ordinary paired 95%
intervals. These contrasts are frozen before new W4A4 scores are obtained; the
historical family-24 report remains unchanged. No large loss or repair gain is
assumed in advance. The inherited repeatability subset covers the original four
methods and is labelled accordingly.

For a new extension root:

```bash
env TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python \
  bash tools/fp4_handoff/python.sh tools/contribution_study/qwen_task_extend_w4a4.py \
  --root /home/yufeng/thunderworld/qwen_task_judge_five_methods_20260924 \
  --from-root /home/yufeng/thunderworld/qwen_task_judge_four_methods_20260923
```

Launch and resume this prepared run using its frozen
`runtime_code/qwen_resource_session.py`, not the original four-method client.
`resource_launch.json` records the exact systemd command: GPU0/2, one vLLM replica
per card, 12 video workers per replica, 48 GiB memory-high, 64 GiB memory-max,
no swap, four media-loading threads and one FFmpeg thread per decoder. The
supervisor records CPU PSS and cgroup memory every five seconds and releases its
own server processes after completion. `runtime_patch.diff` exposes the exact
extension changes; `extension_prepared.json`, `reuse_audit.json`, and
`w4a4_definition_audit.json` retain preparation evidence.

## Remaining baseline extension (2026-09-24)

Run root: `/home/yufeng/thunderworld/qwen_task_judge_all_methods_20260924`.
This adds **AWQ NVFP4 W4A16, ARCQuant 50%, ARCQuant 100%, and SharQ**:
27,104 new videos / 100,704 new primary requests. After validating and importing
the five-method run's 33,880 answers, the complete scope is **60,984 videos /
226,584 primary requests** over nine methods. No existing video is regenerated.

```bash
env TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python \
  OPENBLAS_NUM_THREADS=1 bash tools/fp4_handoff/python.sh \
  tools/contribution_study/qwen_task_extend_baselines.py \
  --root /home/yufeng/thunderworld/qwen_task_judge_all_methods_20260924 \
  --from-root /home/yufeng/thunderworld/qwen_task_judge_five_methods_20260924 \
  --gpus 2 3
```

Use the new run's frozen `runtime_code/qwen_resource_session.py` with
`--gpus 2 3 --video-workers 12` inside the existing bounded systemd launcher.
Each GPU has one vLLM replica; continuous batching, the checkpoint, parser,
frame selection and prompts are unchanged. Device-specific caches are local
to this new run. `resource_launch.json` records the executable launch command.

AWQ remains the **ModelOpt NVFP4 AWQ-lite decoded quality reference**, not a
qualified packed W4 performance measurement. ARCQuant and SharQ retain their
native Cosmos adapters and **BF16 attention**, while our repaired policies and
ordinary W4A4 retain FP4 attention. `baseline_definition_audit.json` checks all
new exports and stores their exact configuration fingerprints and limitations.
Legacy Nano export metadata stores only the generic dispatch configuration;
the audit also verifies the frozen generation-receipt hashes and actual adapter
records, including ARCQuant channel fractions and SharQ's fusion setting.
This compares complete methods, not identical attention or MMA budgets.

The report includes all **36 unordered method pairs × 4 model/benchmark groups**,
using paired scene-cluster bootstrap with ordinary 95% and **family-144**
Bonferroni intervals. The expanded report uses 100,000 bootstrap draws to better
resolve the more extreme tails. Historical 20,000-draw reports and original
answers remain unchanged. New comparisons are frozen before new scores; this
does not retroactively preregister the earlier comparisons. The inherited
repeatability subset still covers only the original four methods.

Each raw response, retry, source video identity and reuse chain is retained.
Resume with the new run's frozen client; completion requires all 60,984 video
results and `report/complete.json` with zero missing entries. The operational
runner shuts down its own replicas after reporting or a persistent failure.

The first GPU2/3 launch was interrupted by changing shared GPU memory use before
new scores were produced. On September 24, the user reassigned this run to
**GPU0/2**. The resumed launch uses `--gpus 0 2 --video-workers 12`; its exact
command and prior launch receipt are recorded in `resource_launch.json` and
`history/device_switch_gpu0_gpu2_*/`. The initial manifest's device template is
historical; `servers.json` records the active replicas. The frozen scientific
manifest, input cohort and 33,880 imported results remain unchanged.
