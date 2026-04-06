#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
SCRIPT="$ROOT/llama3_1_(8b)_grpo.py"
BASE_DIR="$ROOT/gsm8k_improved/baseline_0495_numc12_confirm200"
RUN_ROOT="$ROOT/gsm8k_improved/confirm200_supportscorekeep_transport_numc12_20260406"

STAMP="${2:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="$RUN_ROOT/run_${STAMP}"
LOG="$RUN_ROOT/run_${STAMP}.stdout.log"
STATUS="$RUN_ROOT/run_${STAMP}.status.log"

mkdir -p "$RUN_ROOT" "$OUTDIR"

if [[ "${1:-}" != "__run" ]]; then
    nohup bash "$0" __run "$STAMP" >/dev/null 2>&1 &
    PID=$!
    echo "launched_pid: $PID"
    echo "run_dir: $OUTDIR"
    echo "stdout_log: $LOG"
    echo "status_log: $STATUS"
    exit 0
fi

# shellcheck disable=SC1091
source "$BASE_DIR/baseline.env"

touch "$STATUS"
exec >>"$LOG" 2>&1

log_status() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$STATUS"
}

log_status "supervisor started pid=$$"
log_status "base_dir=$BASE_DIR"
log_status "output_dir=$OUTDIR"
log_status "stdout_log=$LOG"

PYTHONUNBUFFERED=1 \
OUTPUT_DIR="$OUTDIR" \
ANSWER_AGG_PAIR_COUNT_WEIGHT=0.45 \
FAILURE_SLICE_SUMMARY_PATH="$ROOT/gsm8k_improved/verifier_fair_temp_soft_confirm200/run_summary.json" \
ENABLE_FAILURE_SLICE_EVAL_GATE=1 \
ENABLE_SUPPORTSCORE_SLICE_GATE=1 \
ENABLE_SUPPORTSCORE_SLICE_GROUP_BIAS=1 \
SUPPORTSCORE_SLICE_MARGIN_BONUS=-0.08 \
SUPPORTSCORE_SLICE_PAIR_COUNT_BONUS=0.15 \
python3 "$SCRIPT"

OUTDIR="$OUTDIR" STATUS="$STATUS" python3 - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["OUTDIR"]) / "run_summary.json"
status_path = Path(os.environ["STATUS"])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
eval_before = summary.get("eval_before", {})
eval_after = summary.get("eval_after", {})
before = float(eval_before.get("exact_match_rate", 0.0))
after = float(eval_after.get("exact_match_rate", 0.0))
exact = int(eval_after.get("exact_match_count", 0))
samples = int(eval_after.get("num_eval_samples", 0))
answer_tag_rate = float(eval_after.get("answer_tag_rate", 0.0))
strict_xml_rate = float(eval_after.get("strict_xml_rate", 0.0))
with status_path.open("a", encoding="utf-8") as fh:
    fh.write(
        f"{summary_path.parent.name} completed "
        f"eval_before={before:.3f} "
        f"eval_after={after:.3f} "
        f"exact_match_count={exact}/{samples} "
        f"answer_tag_rate={answer_tag_rate:.3f} "
        f"strict_xml_rate={strict_xml_rate:.3f}\n"
    )
PY

log_status "supervisor completed"
echo "run_summary: $OUTDIR/run_summary.json"
echo "stdout_log: $LOG"
echo "status_log: $STATUS"
