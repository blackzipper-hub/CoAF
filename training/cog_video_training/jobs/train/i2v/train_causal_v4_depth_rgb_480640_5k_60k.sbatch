#!/bin/bash
#SBATCH --job-name=ca4d480c
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2v/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2v/%x-%j.err

# P1: Depth + RGB causal I2V at 480x640, 5000 samples, 60k steps.

set -euo pipefail

module purge
source /project/llmsvgen/yazhoux/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

CASUAL_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2v outputs/checkpoints

export HF_HOME="${COAF_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${COAF_ROOT}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${COAF_ROOT}/.cache/huggingface/datasets"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v4_depth_rgb_480640
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/v4_depth_rgb_480640_causal"

export HEIGHT=480
export WIDTH=640
export FPS=8
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1
export NUM_GPUS=4
export GRADIENT_ACCUMULATION_STEPS=8
export TRAIN_STEPS=15000
export CHECKPOINTING_STEPS=1000
export CHECKPOINTS_TOTAL_LIMIT=12
export LR=1e-4
export LR_WARMUP_STEPS=200
export TRACKER_NAME=casual-coaf-v4-depth-rgb-480640-causal
export TEMPORAL_CAUSAL_ATTENTION=1
export SKIP_PREPARE_I2V_DATA=1

nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2v_lora_causal.sh"
