#!/usr/bin/env bash
set -euo pipefail
TW_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$TW_REPO_ROOT/packages/fp4-repair/src:$TW_REPO_ROOT/tools/contribution_study:$TW_REPO_ROOT/experiments/cutlass_fused_repair${PYTHONPATH:+:$PYTHONPATH}"
exec "${TW_PYTHON:-python}" "$@"
