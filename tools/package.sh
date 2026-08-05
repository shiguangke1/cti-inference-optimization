#!/usr/bin/env bash
# Package a submission from online_best/ with the required files at ZIP root.
# Usage: bash tools/package.sh online_best

set -euo pipefail

LABEL="${1:-online_best}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/online_best"
OUTPUT_DIR="${REPO_ROOT}/dist"
DATE="$(date +%Y%m%d)"
OUTPUT_PATH="${OUTPUT_DIR}/submit_${LABEL}_${DATE}.zip"

for file in infer.py build_env.sh requirements.txt; do
    if [[ ! -f "${SOURCE_DIR}/${file}" ]]; then
        echo "Missing required submission file: ${SOURCE_DIR}/${file}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

(
    cd "${SOURCE_DIR}"
    zip -y "${OUTPUT_PATH}" infer.py build_env.sh requirements.txt
)

ls -lh "${OUTPUT_PATH}"
