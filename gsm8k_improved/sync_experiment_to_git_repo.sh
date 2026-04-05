#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-/home/user/图片}"
REPO_DIR="${2:-/home/user/图片/gsm8k_training_repo}"
LABEL="${3:-manual-sync}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "sync repo missing: $REPO_DIR" >&2
  exit 1
fi

python3 - "$SRC_ROOT" "$REPO_DIR" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

copy_relpaths = [
    Path("llama3_1_(8b)_grpo.py"),
    Path("research-results.tsv"),
    Path("autoresearch-state.json"),
    Path("reranker_notes_20260404.txt"),
]

project_roots = [
    Path("multi-agent-autoresearch"),
]

allowed_names = {
    "run_summary.json",
    "run_report.txt",
    "metrics.json",
    "dataset_report.json",
}
allowed_suffixes = {
    ".py",
    ".sh",
    ".md",
    ".txt",
}
skip_dir_names = {
    ".git",
    "__pycache__",
    "adapter",
}
skip_prefixes = (
    "checkpoint-",
)
skip_suffixes = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".arrow",
    ".log",
)

def should_copy(rel: Path) -> bool:
    if rel.name in allowed_names:
        return True
    if rel.suffix in allowed_suffixes:
        return True
    return False

for rel in copy_relpaths:
    src_path = src / rel
    if src_path.exists() and src_path.is_file():
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

for project_root in project_roots:
    src_project = src / project_root
    if not src_project.exists():
        continue
    for path in src_project.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir():
            continue
        parts = rel.parts
        if any(part in skip_dir_names for part in parts):
            continue
        if any(part.startswith(skip_prefixes) for part in parts):
            continue
        if path.suffix in skip_suffixes:
            continue
        if should_copy(rel):
            dst_path = dst / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst_path)

root = src / "gsm8k_improved"
if root.exists():
    for path in root.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir():
            continue
        parts = rel.parts
        if any(part in skip_dir_names for part in parts):
            continue
        if any(part.startswith(skip_prefixes) for part in parts):
            continue
        if path.suffix in skip_suffixes:
            continue
        if should_copy(rel):
            dst_path = dst / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst_path)
PY

git -C "$REPO_DIR" add .

if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
  git -C "$REPO_DIR" commit -m "sync: ${LABEL}"
  git -C "$REPO_DIR" push origin HEAD:main
fi
