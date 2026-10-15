#!/usr/bin/env bash
# Source from the independent Thor checkout; never changes the desktop env.
TW_THOR_ROOT="${TW_THOR_ROOT:-/home/chenhan/thunderworld}"
export TW_THOR_ROOT
export PATH="$TW_THOR_ROOT/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$TW_THOR_ROOT/compiler/triton-3.8.0:$TW_THOR_ROOT/src/ThunderWorld/packages/fp4-repair/src:$TW_THOR_ROOT/src/ThunderWorld/tools/contribution_study:$TW_THOR_ROOT/src/ThunderWorld/experiments/cutlass_fused_repair${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CACHE_DIR="$TW_THOR_ROOT/cache/triton38-sm110"
export HF_HOME="$TW_THOR_ROOT/cache/huggingface"
export THUNDERWORLD_EXPERIMENTAL_SM110=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
