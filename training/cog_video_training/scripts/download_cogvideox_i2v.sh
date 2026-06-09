#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cluster_env.sh"

MODEL_DIR="${MODEL_PATH}"
REPO_ID="${COGVIDEOX_REPO_ID:-THUDM/CogVideoX-5b-I2V}"

if [[ -f "${MODEL_DIR}/model_index.json" ]]; then
  echo "Model already present at ${MODEL_DIR}"
  exit 0
fi

mkdir -p "${MODEL_DIR}"
export HF_HOME

PYTHON="${PYTHON:-$(command -v python)}"
"${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_dir = os.environ["MODEL_PATH"]
repo_id = os.environ.get("COGVIDEOX_REPO_ID", "THUDM/CogVideoX-5b-I2V")
print(f"Downloading {repo_id} -> {model_dir}")
snapshot_download(repo_id=repo_id, local_dir=model_dir)
print("Download complete.")
PY

if [[ ! -f "${MODEL_DIR}/model_index.json" ]]; then
  echo "Download finished but model_index.json is missing under ${MODEL_DIR}" >&2
  exit 1
fi

echo "CogVideoX-5b-I2V ready at ${MODEL_DIR}"
