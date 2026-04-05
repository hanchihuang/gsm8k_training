#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
SCRIPT="$ROOT/llama3_1_(8b)_grpo.py"
RESULTS="$ROOT/research-results.tsv"
STATE="$ROOT/autoresearch-state.json"
HELPER="$ROOT/.agents/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py"
OUTDIR="$ROOT/gsm8k_improved/qwen25_3b_evalonly_confirm200_20260331"
LOG="$ROOT/gsm8k_improved/qwen25_3b_evalonly_confirm200_20260331.stdout.log"
MODEL_NAME_VALUE="Qwen/Qwen2.5-3B-Instruct"

mkdir -p "$OUTDIR"

PYTHONUNBUFFERED=1 \
MODEL_NAME="$MODEL_NAME_VALUE" \
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
python3 "$SCRIPT" >>"$LOG" 2>&1

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path("/home/user/图片")
outdir = root / "gsm8k_improved" / "qwen25_3b_evalonly_confirm200_20260331"
summary = json.loads((outdir / "run_summary.json").read_text(encoding="utf-8"))
eval_after = summary.get("eval_after", {})
metric = float(eval_after.get("exact_match_rate", 0.0))
exact = int(eval_after.get("exact_match_count", 0))
samples = int(eval_after.get("num_eval_samples", 0))
answer_tag_rate = float(eval_after.get("answer_tag_rate", 0.0))
strict_xml_rate = float(eval_after.get("strict_xml_rate", 0.0))

description = (
    "[CONFIRM-3B] Base-model Qwen2.5-3B-Instruct eval-only confirmation finished on "
    f"GSM8K test[:{samples}] with exact_match_rate={metric:.3f} ({exact}/{samples}) in {outdir}. "
    f"answer_tag_rate={answer_tag_rate:.3f} and strict_xml_rate={strict_xml_rate:.3f}. "
    "This run explicitly pins MODEL_NAME=Qwen/Qwen2.5-3B-Instruct so the confirm200 path "
    "cannot silently fall back to the 0.5B default."
)

cmd = [
    "python3",
    str(root / ".agents/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py"),
    "--results-path",
    str(root / "research-results.tsv"),
    "--state-path",
    str(root / "autoresearch-state.json"),
    "--status",
    "note",
    "--metric",
    repr(metric),
    "--commit",
    "workspace-no-git-baseline",
    "--guard",
    "pass",
    "--description",
    description,
]
for label in [
    "qwen25_3b",
    "base_model",
    "eval_only",
    "confirm200",
    "fair_rerank",
]:
    cmd.extend(["--label", label])

subprocess.run(cmd, check=True)
PY
