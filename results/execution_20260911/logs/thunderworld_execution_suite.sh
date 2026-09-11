#!/usr/bin/env bash
set -euo pipefail
cd /home/chenhan/ThunderWorldDev/ThunderWorld
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
bash tools/kernel_handoff/python5090.sh tools/auto_budget/bench_execution.py --model nano --output runs5090/graph_fusion_nano_bench/result.json > /tmp/thunderworld_execution_bench_nano.log 2>&1
printf 'Nano local fusion and double-buffer benchmarks complete\n'
export PYTHONPATH="$PWD/packages/fp4-repair/src${PYTHONPATH:+:$PYTHONPATH}"
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$THUNDERWORLD_PYTHON" -m pytest tools/auto_budget/test_execution.py -q > /tmp/thunderworld_execution_memcheck.log 2>&1
printf 'Execution memory checks complete\n'
for model in edge nano; do
  cases=(robot_table policy_table)
  if [[ "$model" == nano ]]; then cases+=(av_pottery); fi
  for variant in eager graph fused; do
    flags=()
    if [[ "$variant" != eager ]]; then flags+=(--phase-graph); fi
    if [[ "$variant" == fused ]]; then flags+=(--fuse-ffn); fi
    bash tools/kernel_handoff/python5090.sh tools/finding_resample/run.py --model "$model" --cases "${cases[@]}" --seeds 7403 7404 --steps 35 --modes bf16 ffn_act2 ffn_auto50 ffn_auto75 --budget-policy "runs5090/auto_budget_${model}_calibration/policy_validated.json" "${flags[@]}" --output "runs5090/execution_${model}_${variant}" > "/tmp/thunderworld_execution_${model}_${variant}.log" 2>&1
    printf 'Complete trajectories %s %s\n' "$model" "$variant"
  done
done
