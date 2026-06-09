#!/bin/bash
#SBATCH --job-name=iqnt2cv1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2av_pt/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2av_pt/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/i2av_pt outputs/infer/i2av_pt

export MODEL_NAME="i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt"
export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_test
export TRAIN_DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb
export TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-8}"
CHECKPOINT_BASE_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt"
CHECKPOINTS="${CHECKPOINTS:-${CHECKPOINT_STEP:-}}"
if [[ -n "${CHECKPOINTS}" ]]; then
  if [[ "${CHECKPOINTS}" = /* ]]; then
    export CHECKPOINT_DIR="${CHECKPOINTS}"
  else
    export CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/checkpoint-${CHECKPOINTS#checkpoint-}"
  fi
else
  LATEST_CHECKPOINT_STEP=-1
  LATEST_CHECKPOINT_DIR=""
  for path in "${CHECKPOINT_BASE_DIR}"/checkpoint-*; do
    [[ -d "${path}" ]] || continue
    step="${path##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > LATEST_CHECKPOINT_STEP )); then
      LATEST_CHECKPOINT_STEP="${step}"
      LATEST_CHECKPOINT_DIR="${path}"
    fi
  done
  if [[ -z "${LATEST_CHECKPOINT_DIR}" ]]; then
    echo "No checkpoint-* directories found under ${CHECKPOINT_BASE_DIR}" >&2
    exit 1
  fi
  export CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR}"
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Expected I2AV checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2av_pt/${MODEL_NAME}_${CHECKPOINT_TAG}"
export INFER_OUTPUT_DIR_IS_FINAL=1
export STATE_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt
export ACTION_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/action_quantile_norm_stats.pt

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export NUM_SAMPLES="${NUM_SAMPLES:-14}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"
export SEED="${SEED:-42}"
export INFER_DEVICE="${INFER_DEVICE:-cuda}"
export INFER_STAGE=stage2
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh"
