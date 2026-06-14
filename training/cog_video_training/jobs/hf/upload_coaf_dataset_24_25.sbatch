#!/bin/bash
#SBATCH --job-name=hfupds25
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/hf/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/hf/%x-%j.err

# Upload coaf_dataset_24_25 to BlackZipper/coaf_dataset_24_25 on Hugging Face Hub.
#
# Usage:
#   HF_TOKEN=hf_xxx sbatch jobs/hf/upload_coaf_dataset_24_25.sbatch
#   HF_TOKEN=hf_xxx HF_UPLOAD_WORKERS=16 sbatch jobs/hf/upload_coaf_dataset_24_25.sbatch

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
CASUAL_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/hf

export HF_HOME="${CASUAL_ROOT}/.cache/huggingface"
export HF_REPO_ID="${HF_REPO_ID:-BlackZipper/coaf_dataset_24_25}"
export DATASET_DIR="${DATASET_DIR:-/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25}"
export HF_UPLOAD_WORKERS="${HF_UPLOAD_WORKERS:-16}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -n "${HF_STORED_TOKENS:-}" && -f "${HF_STORED_TOKENS}" ]]; then
    :
  elif [[ -f /project/llmsvgen/yazhou/huggingface/stored_tokens ]]; then
    export HF_STORED_TOKENS=/project/llmsvgen/yazhou/huggingface/stored_tokens
    export HF_TOKEN_NAME="${HF_TOKEN_NAME:-123}"
  else
    echo "HF_TOKEN is required. Export HF_TOKEN before submitting this job." >&2
    exit 1
  fi
fi

python -m py_compile "${CASUAL_ROOT}/scripts/upload_coaf_dataset_to_hf.py"
python "${CASUAL_ROOT}/scripts/upload_coaf_dataset_to_hf.py" \
  --repo-id "${HF_REPO_ID}" \
  --dataset-dir "${DATASET_DIR}" \
  --num-workers "${HF_UPLOAD_WORKERS}"
