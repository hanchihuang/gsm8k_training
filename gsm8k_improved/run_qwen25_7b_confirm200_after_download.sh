#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
SCRIPT="$ROOT/llama3_1_(8b)_grpo.py"
OUTDIR="$ROOT/gsm8k_improved/qwen25_7b_evalonly_confirm200_20260331"
LOG="$ROOT/gsm8k_improved/qwen25_7b_evalonly_confirm200_20260331.stdout.log"
STATUS="$ROOT/gsm8k_improved/qwen25_7b_evalonly_confirm200_20260331.status.log"

CACHE_ROOT="/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
SNAPSHOT="$CACHE_ROOT/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
BLOBS="$CACHE_ROOT/blobs"

URL_00003="https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00003-of-00004.safetensors"
URL_00004="https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00004-of-00004.safetensors"

HASH_00003="8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5"
HASH_00004="1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd"

EXPECTED_00003="3864726424"
EXPECTED_00004="3556377672"

TMP_00003="$BLOBS/$HASH_00003.incomplete"
TMP_00004="$BLOBS/$HASH_00004.incomplete"
FINAL_00003="$BLOBS/$HASH_00003"
FINAL_00004="$BLOBS/$HASH_00004"

mkdir -p "$OUTDIR" "$BLOBS" "$SNAPSHOT"
touch "$STATUS"
exec >>"$LOG" 2>&1

log_status() {
    printf '%s %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$STATUS"
}

file_size() {
    local path="$1"
    if [[ -f "$path" ]]; then
        stat -c '%s' "$path"
    else
        printf '0\n'
    fi
}

ensure_symlink() {
    local snapshot_name="$1"
    local blob_hash="$2"
    ln -sfn "../../blobs/$blob_hash" "$SNAPSHOT/$snapshot_name"
}

download_one() {
    local label="$1"
    local url="$2"
    local tmp_path="$3"
    local final_path="$4"
    local expected_size="$5"

    mkdir -p "$(dirname "$tmp_path")"
    if [[ -f "$final_path" ]]; then
        local final_size
        final_size="$(file_size "$final_path")"
        if [[ "$final_size" == "$expected_size" ]]; then
            log_status "$label already complete size=$final_size"
            return 0
        fi
        log_status "$label final blob size mismatch size=$final_size expected=$expected_size moving back to incomplete"
        mv -f "$final_path" "$tmp_path"
    fi

    while true; do
        local before_size
        before_size="$(file_size "$tmp_path")"
        log_status "$label download/resume start size=$before_size expected=$expected_size"

        curl -L -C - --fail --retry 999 --retry-delay 5 \
            --output "$tmp_path" \
            "$url"

        local after_size
        after_size="$(file_size "$tmp_path")"
        if [[ "$after_size" == "$expected_size" ]]; then
            mv -f "$tmp_path" "$final_path"
            log_status "$label completed size=$after_size"
            return 0
        fi

        log_status "$label incomplete after curl exit size=$after_size expected=$expected_size retrying"
        sleep 5
    done
}

log_status "supervisor started"
log_status "initial size model-00003=$(file_size "$TMP_00003") final=$(file_size "$FINAL_00003")"
log_status "initial size model-00004=$(file_size "$TMP_00004") final=$(file_size "$FINAL_00004")"

download_one "model-00003-of-00004" "$URL_00003" "$TMP_00003" "$FINAL_00003" "$EXPECTED_00003" &
PID_00003=$!
download_one "model-00004-of-00004" "$URL_00004" "$TMP_00004" "$FINAL_00004" "$EXPECTED_00004" &
PID_00004=$!

wait "$PID_00003"
wait "$PID_00004"

ensure_symlink "model-00003-of-00004.safetensors" "$HASH_00003"
ensure_symlink "model-00004-of-00004.safetensors" "$HASH_00004"
log_status "snapshot symlinks updated"

PYTHONUNBUFFERED=1 \
HF_LOCAL_FILES_ONLY=1 \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
RUN_PROTOCOL=confirm \
RUN_MODE=single \
EVAL_ONLY=1 \
SKIP_EVAL_BEFORE=1 \
SKIP_EVAL_WARMUP=1 \
SKIP_SAMPLE_GENERATION=1 \
SAVE_ADAPTER=0 \
WRITE_EXPLANATION=0 \
DATASET_SOURCE=gsm8k \
DATASET_SPLIT=train \
DATASET_CONFIG=main \
DATASET_START_INDEX=0 \
NUM_EVAL_SAMPLES=200 \
MAX_TRAIN_SAMPLES=1000 \
EVAL_USE_CONFIDENCE_RERANK=1 \
EVAL_NUM_CANDIDATES=8 \
EVAL_RERANK_TEMPERATURE=0.65 \
EVAL_RERANK_TOP_P=0.92 \
CONFIDENCE_WEIGHT=1.0 \
CONSENSUS_WEIGHT=0.35 \
FORMAT_WEIGHT=0.15 \
LOW_CONFIDENCE_WEIGHT=0.35 \
NOVELTY_WEIGHT=0.1 \
ANSWER_AGG_COUNT_WEIGHT=0.85 \
ANSWER_AGG_STRICT_WEIGHT=0.2 \
ANSWER_AGG_EQUATION_WEIGHT=0.25 \
ANSWER_AGG_DIVERSITY_WEIGHT=0.08 \
ANSWER_AGG_LOW_CONF_WEIGHT=0.2 \
ANSWER_AGG_MIN_GROUP_SIZE=2 \
ANSWER_AGG_MARGIN=0.24 \
ANSWER_AGG_PAIR_COUNT_WEIGHT=0.45 \
ANSWER_AGG_PAIR_MAX_SINGLE_GAP=0.12 \
VERIFIER_BUNDLE_PATH="" \
VERIFIER_SCORE_WEIGHT=0.2 \
VERIFIER_TIE_MARGIN=0.15 \
VERIFIER_MIN_CANDIDATES=2 \
VERIFIER_REQUIRE_ANSWER_DISAGREEMENT=1 \
OUTPUT_DIR="$OUTDIR" \
python3 "$SCRIPT"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("/home/user/图片")
outdir = root / "gsm8k_improved" / "qwen25_7b_evalonly_confirm200_20260331"
status_path = root / "gsm8k_improved" / "qwen25_7b_evalonly_confirm200_20260331.status.log"
summary = json.loads((outdir / "run_summary.json").read_text(encoding="utf-8"))
eval_after = summary.get("eval_after", {})
metric = float(eval_after.get("exact_match_rate", 0.0))
exact = int(eval_after.get("exact_match_count", 0))
samples = int(eval_after.get("num_eval_samples", 0))
answer_tag_rate = float(eval_after.get("answer_tag_rate", 0.0))
strict_xml_rate = float(eval_after.get("strict_xml_rate", 0.0))
with status_path.open("a", encoding="utf-8") as fh:
    fh.write(
        f"{outdir.name} completed exact_match_rate={metric:.3f} "
        f"exact_match_count={exact}/{samples} "
        f"answer_tag_rate={answer_tag_rate:.3f} "
        f"strict_xml_rate={strict_xml_rate:.3f}\n"
    )
PY

log_status "supervisor completed"
