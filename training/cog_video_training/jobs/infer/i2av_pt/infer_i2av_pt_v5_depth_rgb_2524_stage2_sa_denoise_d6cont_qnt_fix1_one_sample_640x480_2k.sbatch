#!/bin/bash
#SBATCH --job-name=i1os6k2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2av_pt/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2av_pt/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/i2av_pt outputs/infer/i2av_pt

export MODEL_NAME="i2av_pt_v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_one_sample_640x480_2k"
export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb_one_sample_640x480
export TRAIN_DATA_ROOT="${DATA_ROOT}"
export NUM_SAMPLES=1
export TRAIN_NUM_SAMPLES=1

export CHECKPOINT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_one_sample_640x480_2k/checkpoint-2000"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Expected I2AV checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2av_pt/${MODEL_NAME}_${CHECKPOINT_TAG}"
export INFER_OUTPUT_DIR_IS_FINAL=1

export STATE_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt
export ACTION_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/action_quantile_norm_stats.pt

export HEIGHT=480 WIDTH=640 FPS=8 MAX_NUM_FRAMES=49
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"
export SA_GUIDANCE_SCALE="${SA_GUIDANCE_SCALE:-1}"
export SEED="${SEED:-42}"
export INFER_DEVICE="${INFER_DEVICE:-cuda}"
export INFER_STAGE=stage2
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh"
