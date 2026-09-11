#!/usr/bin/env bash
set -euo pipefail
cd /home/chenhan/ThunderWorldDev/ThunderWorld
export THUNDERWORLD_PYTHON="$PWD/data_local/e2e_venv/bin/python"
for model in edge nano; do
  bash tools/kernel_handoff/python5090.sh tools/auto_budget/bench_chunk_tuning.py \
    --model "$model" --output "runs5090/graph_fusion_${model}_chunk_tuning/result.json" \
    > "/tmp/thunderworld_execution_chunk_tuning_${model}.log" 2>&1
  printf 'Complete chunk retuning %s\n' "$model"
done
