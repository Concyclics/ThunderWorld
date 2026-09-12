#!/usr/bin/env bash
# Resume this machine's interrupted September 2026 cohort; new machines use README.md.
# Keeps previous attempts. Requires the original /tmp matrices and their fixed inputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$PWD/packages/fp4-repair/src:/tmp/thunderworld-joint-deps${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
EDGE=/home/yufeng/thunderworld/ThunderWorld/artifacts/huggingface/models/Cosmos3-Edge-a9d944e
NANO=/home/yufeng/thunderworld/ThunderWorld/artifacts/huggingface/models/Cosmos3-Nano-411f42a
complete_json() {
  "$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(not (p.exists() and json.loads(p.read_text()).get("complete")))' "$1"
}
run_once() {
  local marker="$1"; shift
  if complete_json "$marker"; then return; fi
  "$PY" -c 'from thunderworld_fp4.cosmos.cli import idle_guard;idle_guard()'
  "$@"
}
"$PY" -c 'from thunderworld_fp4.cosmos.cli import idle_guard;idle_guard()'
if ! complete_json /tmp/joint-nano-i2v-gpu3-performance/matrix.json; then
  "$PY" tools/joint_inference/resume.py /tmp/joint-nano-i2v-gpu3-performance/matrix.json
fi
run_once /tmp/joint-edge-all-tiers-gpu3/result.json "$PY" tools/joint_inference/tune_ffn_tiers.py --model "$EDGE" --case tools/joint_inference/cases/policy_table.json --output /tmp/joint-edge-all-tiers-gpu3 > /tmp/joint-edge-all-tiers-gpu3.log 2>&1
run_once /tmp/joint-nano-all-tiers-gpu3/result.json "$PY" tools/joint_inference/tune_ffn_tiers.py --model "$NANO" --case tools/joint_inference/cases/policy_table.json --output /tmp/joint-nano-all-tiers-gpu3 > /tmp/joint-nano-all-tiers-gpu3.log 2>&1
run_once /tmp/joint-edge-ablation-gpu3.json "$PY" tools/joint_inference/ablate.py --model "$EDGE" --config tools/joint_inference/configs/dual.json --case tools/joint_inference/cases/policy_table.json --tuning /tmp/joint-edge-gpu3-tuning/profiles/policy_table_dual.json --output /tmp/joint-edge-ablation-gpu3.json > /tmp/joint-edge-ablation-gpu3.log 2>&1
run_once /tmp/joint-edge-gpu3-tuned-equivalence.json "$PY" tools/joint_inference/verify_tuned.py --model "$EDGE" --quality-matrix /tmp/joint-edge-final/matrix.json --tuning-root /tmp/joint-edge-gpu3-tuning/profiles --jobs robot_table:dual policy_table:dual --output /tmp/joint-edge-gpu3-tuned-equivalence.json > /tmp/joint-edge-gpu3-tuned-equivalence.log 2>&1
run_once /tmp/joint-nano-i2v-gpu3-tuned-equivalence.json "$PY" tools/joint_inference/verify_tuned.py --model "$NANO" --quality-matrix /tmp/joint-nano-final/matrix.json --tuning-root /tmp/joint-nano-i2v-gpu3-tuning/profiles --jobs robot_table:four --output /tmp/joint-nano-i2v-gpu3-tuned-equivalence.json > /tmp/joint-nano-i2v-gpu3-tuned-equivalence.log 2>&1
run_once /tmp/joint-edge-gpu3-cold/matrix.json "$PY" tools/joint_inference/matrix.py --model "$EDGE" --model-name edge --suite cold --configs bf16 fp8 base dual dual_legacy ffn_only --tuning-root /tmp/joint-edge-gpu3-tuning/profiles --output /tmp/joint-edge-gpu3-cold > /tmp/joint-edge-gpu3-cold.log 2>&1
run_once /tmp/joint-nano-i2v-gpu3-cold/matrix.json "$PY" tools/joint_inference/matrix.py --model "$NANO" --model-name nano --suite cold --cases robot_table --case-configs tools/joint_inference/configs/nano_performance_matrix.json --tuning-root /tmp/joint-nano-i2v-gpu3-tuning/profiles --output /tmp/joint-nano-i2v-gpu3-cold > /tmp/joint-nano-i2v-gpu3-cold.log 2>&1
run_once /tmp/joint-nano-gpu3-cold/matrix.json "$PY" tools/joint_inference/matrix.py --model "$NANO" --model-name nano --suite cold --cases policy_table av_pottery --case-configs tools/joint_inference/configs/nano_performance_matrix.json --tuning-root /tmp/joint-nano-gpu3-tuning/profiles --output /tmp/joint-nano-gpu3-cold > /tmp/joint-nano-gpu3-cold.log 2>&1
"$PY" -m pytest -q packages/fp4-repair/tests > /tmp/joint-runtime-tests-f721e84.log 2>&1
run_once /home/yufeng/thunderworld/joint_fp4_assets_20260912/edge_dual/result.json "$PY" -m thunderworld_fp4.cosmos prepare --model "$EDGE" --config tools/joint_inference/configs/dual.json --output /home/yufeng/thunderworld/joint_fp4_assets_20260912/edge_dual > /tmp/joint-final-prepare-edge.log 2>&1
run_once /home/yufeng/thunderworld/joint_fp4_assets_20260912/nano_four/result.json "$PY" -m thunderworld_fp4.cosmos prepare --model "$NANO" --config tools/joint_inference/configs/four.json --output /home/yufeng/thunderworld/joint_fp4_assets_20260912/nano_four > /tmp/joint-final-prepare-nano.log 2>&1
run_once /tmp/joint-final-assets-edge-reload/result.json "$PY" -m thunderworld_fp4.cosmos infer --model "$EDGE" --assets /home/yufeng/thunderworld/joint_fp4_assets_20260912/edge_dual/assets --config tools/joint_inference/configs/dual.json --case tools/joint_inference/cases/robot_table.json --tuning /tmp/joint-edge-gpu3-tuning/profiles/robot_table_dual.json --steps 35 --seeds 91001 --output /tmp/joint-final-assets-edge-reload > /tmp/joint-final-assets-edge-reload.log 2>&1
run_once /tmp/joint-final-assets-nano-reload/result.json "$PY" -m thunderworld_fp4.cosmos infer --model "$NANO" --assets /home/yufeng/thunderworld/joint_fp4_assets_20260912/nano_four/assets --config tools/joint_inference/configs/four.json --case tools/joint_inference/cases/robot_table.json --tuning /tmp/joint-nano-i2v-gpu3-tuning/profiles/robot_table_four.json --steps 35 --seeds 91001 --output /tmp/joint-final-assets-nano-reload > /tmp/joint-final-assets-nano-reload.log 2>&1
"$PY" - <<'PY'
import json
from pathlib import Path
rows=[]
for model,config in [('edge','dual'),('nano','four')]:
 matrix=json.load(open(f'/tmp/joint-{model}-final/matrix.json'))
 job=next(j for j in matrix['jobs'] if j['case']=='robot_table' and j['config']==config)
 original=json.load(open(Path(job['output'])/'result.json'))
 reference=next(r for r in original['rows'] if r['seed']==91001)
 reload=json.load(open(f'/tmp/joint-final-assets-{model}-reload/result.json'))
 assert reload['complete'] and not reload['tuning_missing']
 assert reference['sha256']==reload['rows'][0]['sha256'],model
 assetroot=Path(f'/home/yufeng/thunderworld/joint_fp4_assets_20260912/{model}_{config}/assets')
 manifest=json.load(open(assetroot/'manifest.json'))
 rows.append(dict(model=model,config=config,asset_path=str(assetroot),manifest=manifest,
                  disk_bytes=sum(p.stat().st_size for p in assetroot.iterdir()),
                  replay_sha256=reference['sha256'],seed=91001,steps=35,bitwise_equal=True))
Path('/tmp/joint-final-assets-validation.json').write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')
PY
"$PY" tools/joint_inference/collect_local_20260912.py
