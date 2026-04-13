#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/user/图片"
REPO_DIR="$ROOT_DIR/gsm8k_training_repo"
MAR_DIR="$REPO_DIR/multi-agent-autoresearch"
PYTHONPATH_DIR="$MAR_DIR/src"

OUTPUT_DIR="${OUTPUT_DIR:-$MAR_DIR/outputs/gsm8k_best0565_official_validation_anchor_20260412_215801}"
BASELINE_ENV="${BASELINE_ENV:-$REPO_DIR/gsm8k_improved/best_seed_confirm200_neartop_override_explicit_20260406.env}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-$REPO_DIR/gsm8k_improved/confirm200_neartop_override_explicit_20260406_validation_run_20260410/run_summary.json}"
RUNNER_PATH="${RUNNER_PATH:-$REPO_DIR/gsm8k_improved/run_checklist_on_0565_baseline.sh}"
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_DIR/llama3_1_(8b)_grpo.py}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ROOT_DIR/gsm8k_improved}"
QUERY="${QUERY:-How should the GSM8K 0.565 anchored line improve next when search is driven only by held-out train validation failures and test is reserved for periodic confirmation?}"
TRAIN_VALIDATION_MOD="${TRAIN_VALIDATION_MOD:-5}"
TRAIN_VALIDATION_BUCKET="${TRAIN_VALIDATION_BUCKET:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-15}"

mkdir -p "$OUTPUT_DIR"

while true; do
  (
    cd "$REPO_DIR"
    env \
      TRAIN_VALIDATION_MOD="$TRAIN_VALIDATION_MOD" \
      TRAIN_VALIDATION_BUCKET="$TRAIN_VALIDATION_BUCKET" \
      PYTHONPATH="$PYTHONPATH_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      python3 -m multi_agent_autoresearch.cli gsm8k-loop \
      --output-dir "$OUTPUT_DIR" \
      --baseline-env "$BASELINE_ENV" \
      --baseline-summary "$BASELINE_SUMMARY" \
      --script-path "$SCRIPT_PATH" \
      --runner-path "$RUNNER_PATH" \
      --experiment-root "$EXPERIMENT_ROOT" \
      --metric-section eval_after \
      --iterations 0 \
      --disable-research-wave \
      --local-root "$ROOT_DIR/gsm8k_improved" \
      --local-root "$REPO_DIR" \
      --query "$QUERY"
  ) || true
  sleep "$SLEEP_SECONDS"
done
