#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
BASE_DIR="$ROOT/gsm8k_improved/baseline_0485_confirm200"
RUNS_DIR="$BASE_DIR/runs"
SCRIPT="$ROOT/llama3_1_(8b)_grpo.py"

mkdir -p "$RUNS_DIR"

# shellcheck disable=SC1091
source "$BASE_DIR/baseline.env"

STAMP="$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="$RUNS_DIR/run_${STAMP}"

python3 "$SCRIPT"

echo "run_summary: $OUTPUT_DIR/run_summary.json"
