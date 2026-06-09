#!/bin/bash
#SBATCH --job-name=tflrg3cv2
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.err

# I2AV P5: Flow + RGB causal joint video/state/action training at 256x256, 5000 samples, 60k steps.

set -euo pipefail

module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av outputs/checkpoints


export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset/composed/v2_flow_rgb
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v2_flow_rgb"
export STATE_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset/state_norm_stats.pt

export HEIGHT=256
export WIDTH=256
export FPS=8
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1
export NUM_GPUS=8
export GRADIENT_ACCUMULATION_STEPS=8
export TRAIN_STEPS=60000
export CHECKPOINTING_STEPS=5000
export CHECKPOINTS_TOTAL_LIMIT=12
export LR=1e-4
export LR_WARMUP_STEPS=200
export TRACKER_NAME=casual-coaf-i2av-v2-flow-rgb
export TEMPORAL_CAUSAL_ATTENTION=1

nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
