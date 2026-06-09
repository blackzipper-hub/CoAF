#!/usr/bin/env bash
# Shared cluster configuration for mscaisuperpod.
# Source this from sbatch jobs and local launch scripts.

PROJECT_ROOT="${PROJECT_ROOT:-/project/mscaisuperpod/sunkai/Casual_CoAF}"
CASUAL_ROOT="${CASUAL_ROOT:-${PROJECT_ROOT}/training/cog_video_training}"
COAF_ROOT="${COAF_ROOT:-${CASUAL_ROOT}}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/coaf_dataset_24_25}"
DATASET_TEST_ROOT="${DATASET_TEST_ROOT:-${PROJECT_ROOT}/coaf_dataset_test}"

CONDA_SH="${CONDA_SH:-/cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi

TARGET_CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "${TARGET_CONDA_ENV}" ]]; then
  if [[ -d "${HOME}/.conda/envs/coaf_train" ]]; then
    TARGET_CONDA_ENV="${HOME}/.conda/envs/coaf_train"
  else
    TARGET_CONDA_ENV="coaf_train"
  fi
fi

if [[ "${COAF_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  conda activate "${TARGET_CONDA_ENV}"
fi

export HF_HOME="${HF_HOME:-${CASUAL_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

export MODEL_PATH="${MODEL_PATH:-${CASUAL_ROOT}/models/CogVideoX-5b-I2V}"
export STATE_NORM_STATS="${STATE_NORM_STATS:-${DATASET_ROOT}/state_norm_stats.pt}"
export ACTION_NORM_STATS="${ACTION_NORM_STATS:-${DATASET_ROOT}/action_norm_stats.pt}"
