#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${REPO_ROOT}/gsm8k_improved/autoresearch_restore_backups/${STAMP}"

mkdir -p "${BACKUP_DIR}"

for name in \
  research-results.tsv \
  autoresearch-state.json \
  autoresearch-launch.json \
  autoresearch-runtime.json \
  autoresearch-runtime.log
do
  if [[ -f "${REPO_ROOT}/${name}" ]]; then
    cp "${REPO_ROOT}/${name}" "${BACKUP_DIR}/${name}"
  fi
done

cp "${REPO_ROOT}/research-results.prev.tsv" "${REPO_ROOT}/research-results.tsv"
cp "${REPO_ROOT}/autoresearch-state.prev.json" "${REPO_ROOT}/autoresearch-state.json"
cp "${REPO_ROOT}/autoresearch-launch.prev.json" "${REPO_ROOT}/autoresearch-launch.json"

cat <<EOF
Restored active autoresearch artifacts from the archived best-known run.
Backup directory: ${BACKUP_DIR}
Active results: ${REPO_ROOT}/research-results.tsv
Active state: ${REPO_ROOT}/autoresearch-state.json
Active launch: ${REPO_ROOT}/autoresearch-launch.json

Next useful checks:
  python3 /home/user/.codex/skills/codex-autoresearch/scripts/autoresearch_resume_check.py --repo "${REPO_ROOT}"
  python3 /home/user/.codex/skills/codex-autoresearch/scripts/autoresearch_supervisor_status.py --repo "${REPO_ROOT}"
EOF
