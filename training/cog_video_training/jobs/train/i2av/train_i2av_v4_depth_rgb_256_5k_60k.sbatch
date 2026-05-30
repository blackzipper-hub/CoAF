#!/bin/bash
#SBATCH --job-name=ia4d256c
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.err

# I2AV P2: Depth + RGB causal joint video/state/action training at 256x256, 5000 samples, 60k steps.

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
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v4_depth_rgb
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v4_depth_rgb_256"
export STATE_NORM_STATS=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/state_norm_stats.pt

export HEIGHT=256
export WIDTH=256
export FPS=8
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1
export NUM_GPUS=4
export GRADIENT_ACCUMULATION_STEPS=8
export TRAIN_STEPS=60000
export CHECKPOINTING_STEPS=5000
export CHECKPOINTS_TOTAL_LIMIT=12
export LR=1e-4
export LR_WARMUP_STEPS=200
export TRACKER_NAME=casual-coaf-i2av-v4-depth-rgb-256
export TEMPORAL_CAUSAL_ATTENTION=1

nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
