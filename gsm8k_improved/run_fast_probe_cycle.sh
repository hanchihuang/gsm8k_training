#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
MINISWEEP_SCRIPT="$ROOT/gsm8k_improved/run_retained_best_minisweep.sh"
QUEUE_STATE_PATH="${QUEUE_STATE_PATH:-$ROOT/gsm8k_improved/fast_probe_queue_state.json}"
RESULTS_PATH="${RESULTS_PATH:-$ROOT/research-results.tsv}"
STATE_PATH="${STATE_PATH:-$ROOT/autoresearch-state.json}"
RECORD_SCRIPT="/home/user/.codex/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py"
SAMPLES="${FAST_PROBE_NUM_EVAL_SAMPLES:-20}"
TIMEOUT_SECONDS="${FAST_PROBE_TIMEOUT_SECONDS:-300}"
FAST_BASELINE_CANDIDATE="${FAST_BASELINE_CANDIDATE:-temp_soft}"
FAST_BASELINE_RATE="${FAST_BASELINE_RATE:-0.45}"

CANDIDATES=(
  temp_soft
  temp_soft_consensus_up
  temp_soft_count_up
  balanced_mid
  retained_base
)

current_index=0
if [[ -f "$QUEUE_STATE_PATH" ]]; then
  current_index="$(jq -r '.next_index // 0' "$QUEUE_STATE_PATH" 2>/dev/null || echo 0)"
fi

candidate_count="${#CANDIDATES[@]}"
if [[ "$candidate_count" -eq 0 ]]; then
  echo "No candidates configured" >&2
  exit 1
fi

selected_index=$(( current_index % candidate_count ))
next_index=$(( (selected_index + 1) % candidate_count ))
candidate="${CANDIDATES[$selected_index]}"

MINISWEEP_NUM_EVAL_SAMPLES="$SAMPLES" \
EXPERIMENT_TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
bash "$MINISWEEP_SCRIPT" run "$candidate"

summary_path="$ROOT/gsm8k_improved/minisweep_${candidate}_${SAMPLES}/run_summary.json"
if [[ ! -f "$summary_path" ]]; then
  echo "Missing summary: $summary_path" >&2
  exit 1
fi

status="$(jq -r '.status // "completed"' "$summary_path")"
metric="$(jq -r '.eval_after.exact_match_rate // 0' "$summary_path")"
hits="$(python3 - "$metric" "$SAMPLES" <<'PY'
import sys
rate = float(sys.argv[1])
samples = int(sys.argv[2])
print(int(round(rate * samples)))
PY
)"

record_status="discard"
description="Fast probe ${candidate} on ${SAMPLES} GSM8K samples finished at ${hits}/${SAMPLES} = ${metric} under the ${TIMEOUT_SECONDS}s budget. Fast-loop control is ${FAST_BASELINE_CANDIDATE}@${SAMPLES} = ${FAST_BASELINE_RATE}, so this result should be judged against that short-run baseline rather than the full-run 0.505 keep."
guard_status="pass"
commit_id="no-git"

if [[ "$status" == "timeout" ]]; then
  metric="0"
  hits="0"
  description="Fast probe ${candidate} timed out at the ${TIMEOUT_SECONDS}s budget on ${SAMPLES} GSM8K samples, so it was discarded immediately to preserve the five-minute cadence. Fast-loop control remains ${FAST_BASELINE_CANDIDATE}@${SAMPLES} = ${FAST_BASELINE_RATE}."
fi

python3 "$RECORD_SCRIPT" \
  --results-path "$RESULTS_PATH" \
  --state-path "$STATE_PATH" \
  --status "$record_status" \
  --metric "$metric" \
  --commit "$commit_id" \
  --guard "$guard_status" \
  --description "$description"

tmp_queue="$(mktemp)"
jq -n --argjson next_index "$next_index" --arg last_candidate "$candidate" --arg last_summary_path "$summary_path" \
  '{next_index: $next_index, last_candidate: $last_candidate, last_summary_path: $last_summary_path}' >"$tmp_queue"
mv "$tmp_queue" "$QUEUE_STATE_PATH"

jq -n \
  --arg candidate "$candidate" \
  --arg status "$status" \
  --arg metric "$metric" \
  --argjson samples "$SAMPLES" \
  --argjson timeout_seconds "$TIMEOUT_SECONDS" \
  --arg summary_path "$summary_path" \
  '{
    candidate: $candidate,
    status: $status,
    metric: ($metric | tonumber),
    samples: $samples,
    timeout_seconds: $timeout_seconds,
    summary_path: $summary_path
  }'
