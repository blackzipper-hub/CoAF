#!/bin/bash
#SBATCH --job-name=t2dn6kv1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av_pt/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av_pt/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av_pt outputs/checkpoints/i2av


export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont"
export STATE_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt
export ACTION_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/action_norm_stats.pt

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=4
export TRAIN_STEPS="${TRAIN_STEPS:-15000}" CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}" CHECKPOINTS_TOTAL_LIMIT=14
export LR="${LR:-5e-5}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
export TRACKER_NAME=casual-coaf-i2av-pt-v5-depth-rgb-2524-stage2-sa-denoise-d6cont
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export TRAIN_STAGE=stage2
export STAGE2_TRAIN_TRANSFORMER_LORA=1
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export LAMBDA_S="${LAMBDA_S:-1.0}"
export LAMBDA_A="${LAMBDA_A:-2.0}"
export LAMBDA_G="${LAMBDA_G:-1.0}"
export LAMBDA_C="${LAMBDA_C:-0.1}"
export LOAD_TENSORS=1
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
