#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/图片"
SCRIPT="$ROOT/llama3_1_(8b)_grpo.py"
RESULTS="$ROOT/research-results.tsv"
STATE="$ROOT/autoresearch-state.json"
HELPER="$ROOT/.agents/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py"
OUTDIR="$ROOT/gsm8k_improved/autoresearch_confirm200_iter99_promptreplay_masktrunc_keep75c"
LOG="$ROOT/gsm8k_improved/autoresearch_confirm200_iter99_promptreplay_masktrunc_keep75c.stdout.log"

mkdir -p "$OUTDIR"

PYTHONUNBUFFERED=1 \
RUN_PROTOCOL=confirm \
DATASET_SOURCE=gsm8k \
NUM_EVAL_SAMPLES=200 \
MAX_TRAIN_SAMPLES=1000 \
SFT_WARMUP_STEPS=20 \
MAX_STEPS=40 \
EVAL_USE_CONFIDENCE_RERANK=1 \
EVAL_NUM_CANDIDATES=8 \
SKIP_EVAL_BEFORE=1 \
SKIP_EVAL_WARMUP=1 \
SKIP_SAMPLE_GENERATION=1 \
WRITE_EXPLANATION=0 \
ADAPTER_PATH="$ROOT/gsm8k_improved/autoresearch_confirm200_masktrunc_iter75c/adapter" \
OUTPUT_DIR="$OUTDIR" \
CONTINUATION_SAFE_DYNAMICS=1 \
SYNTHETIC_SFT_ONLY=1 \
SYNTHETIC_AUGMENT_COUNT=0 \
MINE_CHALLENGING_SYNTHETIC=0 \
ENABLE_TEACHER_COMPLETION_REPLAY=1 \
TEACHER_COMPLETION_BANK_PATH="$ROOT/gsm8k_improved/train_teacher_bank_pct_ratio_v1.json" \
TEACHER_REPLAY_COUNT=16 \
TEACHER_REPLAY_SLICES='percentage,rate_or_ratio' \
ENABLE_SFT_TEACHER_BANK_PRIORITY=1 \
SFT_TEACHER_BANK_TARGET=96 \
SFT_TEACHER_BANK_KEEP_AUX_COUNT=12 \
SFT_TEACHER_BANK_SLICES='percentage,rate_or_ratio' \
ENABLE_PROMPT_REPLAY=1 \
PROMPT_REPLAY_COUNT=24 \
PROMPT_REPLAY_SLICES='percentage,rate_or_ratio' \
ENABLE_ANCHOR_REPLAY=1 \
ANCHOR_REPLAY_COUNT=24 \
ANCHOR_REPLAY_SLICES='percentage,rate_or_ratio' \
NUM_GENERATIONS=6 \
PER_DEVICE_TRAIN_BATCH_SIZE=6 \
MASK_TRUNCATED_COMPLETIONS=1 \
GRPO_LOSS_TYPE=dapo \
python3 "$SCRIPT" >>"$LOG" 2>&1

python3 - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path("/home/user/图片")
outdir = root / "gsm8k_improved" / "autoresearch_confirm200_iter99_promptreplay_masktrunc_keep75c"
summary = json.loads((outdir / "run_summary.json").read_text(encoding="utf-8"))
eval_after = summary.get("eval_after", {})
metric = float(eval_after.get("exact_match_rate", 0.0))
answer_tag_rate = float(eval_after.get("answer_tag_rate", 0.0))
strict_xml_rate = float(eval_after.get("strict_xml_rate", 0.0))
status = "keep" if metric > 0.48 else "discard"

if status == "keep":
    description = (
        "[CONFIRM] Zero-synthetic prompt replay plus teacher-bank-priority warmup on the retained "
        "iter75c mask-truncated adapter finished the 200-sample fair verifier-rerank confirmation "
        f"at exact_match_rate={metric:.3f} in {outdir}. "
        f"answer_tag_rate={answer_tag_rate:.3f} and strict_xml_rate={strict_xml_rate:.3f}. "
        "This beats the retained 0.48 neighborhood, so the replay-heavy retained-adapter "
        "continuation recipe becomes the new kept best."
    )
else:
    description = (
        "[CONFIRM] Zero-synthetic prompt replay plus teacher-bank-priority warmup on the retained "
        "iter75c mask-truncated adapter finished the 200-sample fair verifier-rerank confirmation "
        f"at exact_match_rate={metric:.3f} in {outdir}. "
        f"answer_tag_rate={answer_tag_rate:.3f} and strict_xml_rate={strict_xml_rate:.3f}. "
        "The high 30-sample scout signal did not exceed the retained 0.48 neighborhood under full "
        "confirmation, so this branch is a false positive and should not replace the kept recipe."
    )

cmd = [
    "python3",
    str(root / ".agents/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py"),
    "--results-path",
    str(root / "research-results.tsv"),
    "--state-path",
    str(root / "autoresearch-state.json"),
    "--status",
    status,
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
    "continuation_path",
    "prompt_replay",
    "teacher_bank_priority",
    "zero_synthetic",
    "corrected_retained_adapter",
    "mask_truncated_completions",
    "num_generations6",
    "confirm200",
    "full_confirmation",
]:
    cmd.extend(["--label", label])

subprocess.run(cmd, check=True)
subprocess.run(
    [
        "python3",
        str(root / ".agents/skills/codex-autoresearch/scripts/autoresearch_supervisor_status.py"),
        "--repo",
        str(root),
        "--write-state",
    ],
    check=True,
)
PY
