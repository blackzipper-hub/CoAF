#!/bin/bash
#SBATCH --job-name=iasmoke1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.err

# I2AV smoke test: v1_pose_rgb, 10 steps, 1 GPU.

set -euo pipefail

module purge
source /project/llmsvgen/yazhoux/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

CASUAL_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av outputs/checkpoints

export HF_HOME="${COAF_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${COAF_ROOT}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${COAF_ROOT}/.cache/huggingface/datasets"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v1_pose_rgb
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/smoke_v1_pose_rgb"
export STATE_NORM_STATS=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/state_norm_stats.pt

export HEIGHT=256
export WIDTH=256
export FPS=8
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1
export NUM_GPUS=1
export GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_STEPS=10
export CHECKPOINTING_STEPS=10
export CHECKPOINTS_TOTAL_LIMIT=2
export VALIDATION_STEPS=0
export LR=1e-4
export TRACKER_NAME=casual-coaf-i2av-smoke-v1
export TEMPORAL_CAUSAL_ATTENTION=1

nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
