#!/usr/bin/env bash
# Use the existing, version-verified environment without modifying it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_CACHE_DIR="$PWD/.cache5090/triton"
export TORCH_EXTENSIONS_DIR="$PWD/.cache5090/torch_extensions"
export PATH="/usr/local/cuda-12.9/bin:/usr/lib/wsl/lib:$PATH"
export PYTHONPATH="$PWD/packages/fp4-repair/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${THUNDERWORLD_PYTHON:-python}" "$@"
