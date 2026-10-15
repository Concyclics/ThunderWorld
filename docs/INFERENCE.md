# Inference with the released models

## Frozen profiles

The [release registry](../packages/fp4-repair/src/thunderworld_fp4/presets/releases.json) pins both Hub and original-model revisions. `tw-infer` invokes the runtime bundled with that exact model release so later research edits cannot silently change an existing pretrained profile.

| Model | Hub revision | NVIDIA base revision |
|---|---|---|
| Edge | `25b48e8671186dd42255d32a03db647a0619597c` | `a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba` |
| Nano | `4ba1469c314b679d455798abd0d1c63e476e5c22` | `411f42a8fdfb8c5b2583cb8786e0938f49796eaa` |
| Super | `495c35e6007585e782a029790aade157a050e216` | `f543c56225b2e04d0ad141e29655be3a45d9c455` |

- **2× / `b2_joint`:** joint W0/W1 representation/support calibration, activation-weighted sensitivity and measured repair allocation.
- **1.5× / `b1.5_protected_n`:** identity-prescale RTN with calibrated allocation and compact protected N128 weight supports.
- Both retain every base attention block and use current-input activation residuals. Neither enables outlier splitting or the second-order cross term.
- Published asset reload and one complete 35-step generation per profile/model matched the historical float outputs bitwise. See the six checks in `docs/results/*-release-validation.json`; these loader checks do not replace the full benchmark.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 tw-infer --model nano --profile 1.5x \
  --image /path/to/input.png --prompt "Describe the desired continuation." \
  --seed 12345 --output outputs/nano-example

CUDA_VISIBLE_DEVICES=0,1 tw-infer --model super --profile 2x \
  --image /path/to/input.png --prompt "Describe the desired continuation." \
  --seed 12345 --output outputs/super-example
```

Select a new output directory. Local reuse:

```bash
CUDA_VISIBLE_DEVICES=0 tw-infer --model edge --profile 2x \
  --release-dir /path/to/Cosmos3-Edge-ThunderWorldFP4 \
  --base-model /path/to/Cosmos3-Edge --local-files-only \
  --case-json /path/to/case.json --seed 12345 --output outputs/offline-example
```

A case JSON has `id`, `task: "i2v"`, absolute `image`, `prompt`, `height: 192`, `width: 320`, `frames: 61`, and optionally `negative_prompt`. The canonical benchmark negative prompt is `distorted objects, flickering, blurry, low quality`. The released loader fixes 35 steps, CFG 6 and flow shift 3; output playback is 24 fps. Different shapes or step counts require a new plan, budget verification and quality validation.

`video.mp4` is for viewing. `outputs.npz` retains the original float signals used for pixel metrics. `result.json` records hashes, compute accounting, memory and the actual execution settings. Shared-mode timings in it are diagnostic only.

## Debug and execution

`--debug off|timing|trace` defaults to `off`; `--execution graph|eager` defaults to `graph`.

- **off:** no trajectory hooks or per-step diagnostic events / host reads.
- **timing:** preallocated stage/step events, read after the request.
- **trace:** adds trajectory summaries, repair statistics and NVTX.

Debug settings do not intentionally change quantization or repair decisions. Probe-enabled results are excluded from formal performance comparisons. Preparation and warmup occur before inference; current inputs and timesteps update the prepared Graph. Each instance processes one request at a time. Use separate processes for experiments; this is not a concurrent serving engine.

## Development API

The installed runtime also exposes `tw-cosmos prepare|warmup|infer|benchmark` and a Python API:

```python
from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
import json

config = RunConfig.from_dict(json.load(open("/path/to/config.json")))
case = json.load(open("/path/to/case.json"))
runner = CosmosRunner("/path/to/base-model", config,
                      assets="/path/to/assets", ar_cache=True, first_frame=True).prepare()
runner.warmup(case, steps=35)
signals, receipt = runner.infer(case, seed=12345, steps=35)
```

This API uses the checked-out source. For the exact pretrained release, use `tw-infer`; for comparisons, use the frozen manifest workflow in [REPRODUCE.md](REPRODUCE.md), which records source identities and the model-specific runtime options. Do not reuse a Graph after changing a plan, mask or shape. Super's two-device placement is initialized by the public worker / released loader, not by the bare example above.

Independent FFN, projection and attention policy files are accepted by `tw-cosmos`. The schema and examples are in [ARCHITECTURE.md](ARCHITECTURE.md) and `tools/joint_inference/`. Changing budgets beyond the stored W1 support union requires rebuilding assets; a missing support is an error, not a hidden BF16 fallback.
