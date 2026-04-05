#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
SCRIPT_PATH="$ROOT/llama3_1_(8b)_grpo.py"
ADAPTER_PATH="$ROOT/outputs_llama3_1_grpo_gsm8k_rerank_eval/adapter"
OUT_BASE="$ROOT/gsm8k_improved"
EXPERIMENT_TIMEOUT_SECONDS="${EXPERIMENT_TIMEOUT_SECONDS:-300}"
MINISWEEP_NUM_EVAL_SAMPLES="${MINISWEEP_NUM_EVAL_SAMPLES:-30}"

MODE="${1:-list}"
TARGET="${2:-}"

BASE_ENV=(
  "SMOKE_TEST=0"
  "RUN_MODE=single"
  "EVAL_ONLY=1"
  "SKIP_EVAL_BEFORE=1"
  "SKIP_SAMPLE_GENERATION=1"
  "DATASET_SOURCE=gsm8k"
  "DATASET_SPLIT=train"
  "DATASET_CONFIG=main"
  "MAX_TRAIN_SAMPLES=1000"
  "NUM_EVAL_SAMPLES=$MINISWEEP_NUM_EVAL_SAMPLES"
  "DATASET_START_INDEX=0"
  "MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct"
  "ADAPTER_PATH=$ADAPTER_PATH"
  "EVAL_USE_CONFIDENCE_RERANK=1"
  "EVAL_NUM_CANDIDATES=8"
  "ANSWER_AGG_COUNT_WEIGHT=0.85"
  "ANSWER_AGG_MARGIN=0.24"
)

list_configs() {
  cat <<'EOF'
retained_base
  temp/top_p=0.70/0.95 lcw=0.35 consensus=0.35 count=0.85 margin=0.24
  Purpose: re-anchor the current retained-best control on the active scout slice.

temp_soft
  temp/top_p=0.65/0.92 lcw=0.35 consensus=0.35 count=0.85 margin=0.24
  Purpose: the current best signal; soften sampling only.

temp_soft_consensus_up
  temp/top_p=0.65/0.92 lcw=0.35 consensus=0.40 count=0.85 margin=0.24
  Purpose: keep the winning softer sampling and only lift consensus slightly.

temp_soft_count_up
  temp/top_p=0.65/0.92 lcw=0.35 consensus=0.35 count=0.95 margin=0.24
  Purpose: keep softer sampling and strengthen answer-group count support.

temp_soft_margin_loose
  temp/top_p=0.65/0.92 lcw=0.35 consensus=0.35 count=0.85 margin=0.26
  Purpose: keep softer sampling and allow slightly looser answer-group aggregation.

temp_soft_margin_tight
  temp/top_p=0.65/0.92 lcw=0.35 consensus=0.35 count=0.85 margin=0.22
  Purpose: keep softer sampling and require slightly tighter answer-group aggregation.

balanced_mid
  temp/top_p=0.65/0.92 lcw=0.40 consensus=0.40 count=0.85 margin=0.24
  Purpose: combined conservative move around the temp_soft neighborhood.

Promotion rule for scout mode
  Default scout uses MINISWEEP_NUM_EVAL_SAMPLES=30 under a 300s budget.
  Promote a candidate only if it beats retained_base on the same slice by at least 2 hits and reaches at least 0.50.
  Use 60-sample confirmation before any 200-sample verification.
EOF
}

run_config() {
  local name="$1"
  local temp top_p lcw consensus agg_count agg_margin output_dir hypothesis summary_path report_path status started_at ended_at exit_code

  case "$name" in
    retained_base)
      temp="0.7"; top_p="0.95"; lcw="0.35"; consensus="0.35"; agg_count="0.85"; agg_margin="0.24"
      hypothesis="control on the current scout slice for the retained best defaults"
      ;;
    temp_soft)
      temp="0.65"; top_p="0.92"; lcw="0.35"; consensus="0.35"; agg_count="0.85"; agg_margin="0.24"
      hypothesis="slightly lower sampling noise may improve stability without harming answer-group recovery"
      ;;
    temp_soft_consensus_up)
      temp="0.65"; top_p="0.92"; lcw="0.35"; consensus="0.4"; agg_count="0.85"; agg_margin="0.24"
      hypothesis="temp_soft plus a mild consensus boost may help stable answer groups on mixed problem types"
      ;;
    temp_soft_count_up)
      temp="0.65"; top_p="0.92"; lcw="0.35"; consensus="0.35"; agg_count="0.95"; agg_margin="0.24"
      hypothesis="temp_soft plus stronger answer-group count support may reinforce corroborated groups without changing penalties"
      ;;
    temp_soft_margin_loose)
      temp="0.65"; top_p="0.92"; lcw="0.35"; consensus="0.35"; agg_count="0.85"; agg_margin="0.26"
      hypothesis="temp_soft plus a slightly looser answer-group margin may recover near-miss groups that the current margin drops"
      ;;
    temp_soft_margin_tight)
      temp="0.65"; top_p="0.92"; lcw="0.35"; consensus="0.35"; agg_count="0.85"; agg_margin="0.22"
      hypothesis="temp_soft plus a slightly tighter answer-group margin may suppress borderline group overrides that are hurting exact match"
      ;;
    balanced_mid)
      temp="0.65"; top_p="0.92"; lcw="0.4"; consensus="0.4"; agg_count="0.85"; agg_margin="0.24"
      hypothesis="combine temp_soft with mild rerank tightening as a bounded higher-risk scout"
      ;;
    *)
      echo "Unknown config: $name" >&2
      exit 1
      ;;
  esac

  output_dir="$OUT_BASE/minisweep_${name}_${MINISWEEP_NUM_EVAL_SAMPLES}"
  summary_path="$output_dir/run_summary.json"
  report_path="$output_dir/run_report.txt"
  echo "[run] $name -> $output_dir"
  echo "[run] timeout=${EXPERIMENT_TIMEOUT_SECONDS}s"
  echo "[run] samples=${MINISWEEP_NUM_EVAL_SAMPLES}"
  mkdir -p "$output_dir"
  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  status="completed"

  set +e
  env \
    "${BASE_ENV[@]}" \
    "EVAL_RERANK_TEMPERATURE=$temp" \
    "EVAL_RERANK_TOP_P=$top_p" \
    "LOW_CONFIDENCE_WEIGHT=$lcw" \
    "CONSENSUS_WEIGHT=$consensus" \
    "ANSWER_AGG_COUNT_WEIGHT=$agg_count" \
    "ANSWER_AGG_MARGIN=$agg_margin" \
    "OUTPUT_DIR=$output_dir" \
    "EXPERIMENT_HYPOTHESIS=$hypothesis" \
    timeout --signal=TERM --kill-after=20 "${EXPERIMENT_TIMEOUT_SECONDS}s" python3 "$SCRIPT_PATH"
  exit_code=$?
  set -e

  ended_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  if [[ "$exit_code" -eq 124 ]]; then
    status="timeout"
    cat >"$summary_path" <<EOF
{
  "status": "timeout",
  "timed_out": true,
  "timeout_seconds": $EXPERIMENT_TIMEOUT_SECONDS,
  "num_eval_samples": $MINISWEEP_NUM_EVAL_SAMPLES,
  "config_name": "$name",
  "output_dir": "$output_dir",
  "started_at": "$started_at",
  "ended_at": "$ended_at",
  "experiment_hypothesis": "$hypothesis",
  "eval_after": {
    "num_eval_samples": $MINISWEEP_NUM_EVAL_SAMPLES,
    "exact_match_count": 0,
    "exact_match_rate": 0.0
  }
}
EOF
    cat >"$report_path" <<EOF
GRPO 运行记录
====================
status: timeout
config_name: $name
timeout_seconds: $EXPERIMENT_TIMEOUT_SECONDS
num_eval_samples: $MINISWEEP_NUM_EVAL_SAMPLES
started_at: $started_at
ended_at: $ended_at
hypothesis: $hypothesis
EOF
    echo "[timeout] $name exceeded ${EXPERIMENT_TIMEOUT_SECONDS}s and was terminated"
  elif [[ "$exit_code" -ne 0 ]]; then
    echo "[error] $name failed with exit code $exit_code" >&2
    return "$exit_code"
  fi

  if [[ -f "$summary_path" ]]; then
    tmp_summary="$(mktemp)"
    jq --argjson timeout "$EXPERIMENT_TIMEOUT_SECONDS" --argjson samples "$MINISWEEP_NUM_EVAL_SAMPLES" --arg status "$status" '
      .status = $status
      | .runtime_config = ((.runtime_config // {}) + {experiment_timeout_seconds: $timeout, minisweep_num_eval_samples: $samples})
    ' "$summary_path" >"$tmp_summary"
    mv "$tmp_summary" "$summary_path"
  fi
}

case "$MODE" in
  list)
    list_configs
    ;;
  run)
    if [[ -z "$TARGET" ]]; then
      echo "Usage: $0 run <config_name>" >&2
      exit 1
    fi
    run_config "$TARGET"
    ;;
  run-all)
    for name in retained_base temp_soft temp_soft_consensus_up temp_soft_count_up temp_soft_margin_loose temp_soft_margin_tight balanced_mid; do
      run_config "$name"
    done
    ;;
  *)
    echo "Usage: $0 [list|run <config_name>|run-all]" >&2
    exit 1
    ;;
esac
