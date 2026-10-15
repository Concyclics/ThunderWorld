#!/usr/bin/env bash
# Reproduce the unprivileged Thor environment from the copied source snapshot.
set -euo pipefail
TW_THOR_ROOT="${TW_THOR_ROOT:-/home/chenhan/thunderworld}"
TW_THOR_SOURCE="$TW_THOR_ROOT/src/ThunderWorld"
test "$(uname -m)" = aarch64
test -x /usr/local/cuda/bin/ptxas
mkdir -p "$TW_THOR_ROOT"/{logs,results,cache,compiler,models,data,assets}
python3 -m venv "$TW_THOR_ROOT/.venv"
TW_THOR_PYTHON="$TW_THOR_ROOT/.venv/bin/python"
"$TW_THOR_PYTHON" -m pip install --upgrade pip setuptools wheel
"$TW_THOR_PYTHON" -m pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 --index-url https://download.pytorch.org/whl/cu130
"$TW_THOR_PYTHON" -m pip install -r "$TW_THOR_SOURCE/tools/fp4_handoff/requirements.txt"
"$TW_THOR_PYTHON" -m pip install psutil nvidia-ml-py pydantic msgspec py-cpuinfo pybase64 pyzmq cbor2 blake3 cloudpickle cachetools
# Use official ARM native FP8 ops without upgrading the frozen Torch version.
"$TW_THOR_PYTHON" -m pip install --no-deps vllm==0.25.1
"$TW_THOR_PYTHON" -m pip install --no-deps --target "$TW_THOR_ROOT/compiler/triton-3.8.0" triton==3.8.0
"$TW_THOR_PYTHON" -m pip install --no-deps -e "$TW_THOR_SOURCE/packages/fp4-repair"
"$TW_THOR_PYTHON" "$TW_THOR_SOURCE/tools/thor_handoff/overlay.py" --root "$TW_THOR_SOURCE"
source "$TW_THOR_SOURCE/tools/thor_handoff/env.sh"
python "$TW_THOR_SOURCE/tools/thor_handoff/doctor.py" --output "$TW_THOR_ROOT/results/doctor.json"
