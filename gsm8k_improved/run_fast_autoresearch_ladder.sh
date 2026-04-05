#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
MINISWEEP_SCRIPT="$ROOT/gsm8k_improved/run_retained_best_minisweep.sh"
SCOUT_SAMPLES="${SCOUT_SAMPLES:-30}"
CONFIRM_SAMPLES="${CONFIRM_SAMPLES:-60}"
SCOUT_TIMEOUT_SECONDS="${SCOUT_TIMEOUT_SECONDS:-300}"
CONFIRM_TIMEOUT_SECONDS="${CONFIRM_TIMEOUT_SECONDS:-600}"
PROMOTION_MIN_RATE="${PROMOTION_MIN_RATE:-0.50}"

CANDIDATES=(
  temp_soft
  temp_soft_consensus_up
  temp_soft_count_up
  temp_soft_margin_loose
  balanced_mid
)

run_one() {
  local config="$1"
  local samples="$2"
  local timeout_seconds="$3"

  MINISWEEP_NUM_EVAL_SAMPLES="$samples" \
  EXPERIMENT_TIMEOUT_SECONDS="$timeout_seconds" \
  bash "$MINISWEEP_SCRIPT" run "$config"
}

summary_path_for() {
  local config="$1"
  local samples="$2"
  echo "$ROOT/gsm8k_improved/minisweep_${config}_${samples}/run_summary.json"
}

metric_for() {
  local config="$1"
  local samples="$2"
  local summary_path

  summary_path="$(summary_path_for "$config" "$samples")"
  jq -r '.eval_after.exact_match_rate // 0' "$summary_path"
}

status_for() {
  local config="$1"
  local samples="$2"
  local summary_path

  summary_path="$(summary_path_for "$config" "$samples")"
  jq -r '.status // "completed"' "$summary_path"
}

echo "[ladder] scout_samples=$SCOUT_SAMPLES confirm_samples=$CONFIRM_SAMPLES"
echo "[ladder] running retained_base scout"
run_one retained_base "$SCOUT_SAMPLES" "$SCOUT_TIMEOUT_SECONDS"
baseline_rate="$(metric_for retained_base "$SCOUT_SAMPLES")"
baseline_hits="$(python3 - <<'PY' "$baseline_rate" "$SCOUT_SAMPLES"
import sys
rate = float(sys.argv[1])
samples = int(sys.argv[2])
print(int(round(rate * samples)))
PY
)"
echo "[ladder] retained_base scout rate=$baseline_rate hits=$baseline_hits/$SCOUT_SAMPLES"

best_candidate=""
best_rate="$baseline_rate"
best_hits="$baseline_hits"

for candidate in "${CANDIDATES[@]}"; do
  echo "[ladder] running scout for $candidate"
  run_one "$candidate" "$SCOUT_SAMPLES" "$SCOUT_TIMEOUT_SECONDS"
  candidate_status="$(status_for "$candidate" "$SCOUT_SAMPLES")"
  candidate_rate="$(metric_for "$candidate" "$SCOUT_SAMPLES")"
  candidate_hits="$(python3 - <<'PY' "$candidate_rate" "$SCOUT_SAMPLES"
import sys
rate = float(sys.argv[1])
samples = int(sys.argv[2])
print(int(round(rate * samples)))
PY
)"
  echo "[ladder] scout $candidate status=$candidate_status rate=$candidate_rate hits=$candidate_hits/$SCOUT_SAMPLES"
  if [[ "$candidate_status" != "completed" ]]; then
    continue
  fi
  if python3 - "$candidate_rate" "$best_rate" <<'PY' | grep -q '^1$'
import sys
print("1" if float(sys.argv[1]) > float(sys.argv[2]) else "0")
PY
  then
    best_candidate="$candidate"
    best_rate="$candidate_rate"
    best_hits="$candidate_hits"
  fi
done

if [[ -z "$best_candidate" ]]; then
  echo "[ladder] no candidate beat retained_base on scout"
  jq -n \
    --arg stage "scout_only" \
    --arg baseline_rate "$baseline_rate" \
    --arg selected "retained_base" \
    '{stage: $stage, baseline_rate: ($baseline_rate | tonumber), selected: $selected}'
  exit 0
fi

if ! python3 - "$best_rate" "$baseline_rate" "$best_hits" "$baseline_hits" "$PROMOTION_MIN_RATE" <<'PY' | grep -q '^1$'
import sys
best_rate = float(sys.argv[1])
baseline_rate = float(sys.argv[2])
best_hits = int(sys.argv[3])
baseline_hits = int(sys.argv[4])
promotion_min_rate = float(sys.argv[5])
eligible = best_rate >= promotion_min_rate and (best_hits - baseline_hits) >= 2
print("1" if eligible else "0")
PY
then
  echo "[ladder] best scout candidate=$best_candidate rate=$best_rate did not clear promotion gate"
  jq -n \
    --arg stage "scout_rejected" \
    --arg baseline_rate "$baseline_rate" \
    --arg best_candidate "$best_candidate" \
    --arg best_rate "$best_rate" \
    '{stage: $stage, baseline_rate: ($baseline_rate | tonumber), best_candidate: $best_candidate, best_rate: ($best_rate | tonumber)}'
  exit 0
fi

echo "[ladder] promoting $best_candidate to confirm"
run_one retained_base "$CONFIRM_SAMPLES" "$CONFIRM_TIMEOUT_SECONDS"
run_one "$best_candidate" "$CONFIRM_SAMPLES" "$CONFIRM_TIMEOUT_SECONDS"
confirm_baseline_rate="$(metric_for retained_base "$CONFIRM_SAMPLES")"
confirm_candidate_rate="$(metric_for "$best_candidate" "$CONFIRM_SAMPLES")"

jq -n \
  --arg stage "confirm_complete" \
  --arg scout_best_candidate "$best_candidate" \
  --arg scout_best_rate "$best_rate" \
  --arg scout_baseline_rate "$baseline_rate" \
  --arg confirm_baseline_rate "$confirm_baseline_rate" \
  --arg confirm_candidate_rate "$confirm_candidate_rate" \
  '{
    stage: $stage,
    scout_best_candidate: $scout_best_candidate,
    scout_best_rate: ($scout_best_rate | tonumber),
    scout_baseline_rate: ($scout_baseline_rate | tonumber),
    confirm_baseline_rate: ($confirm_baseline_rate | tonumber),
    confirm_candidate_rate: ($confirm_candidate_rate | tonumber)
  }'
