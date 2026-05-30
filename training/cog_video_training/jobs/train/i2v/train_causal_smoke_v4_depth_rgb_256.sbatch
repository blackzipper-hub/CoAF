#!/bin/bash
#SBATCH --job-name=casmk256
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2v/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2v/%x-%j.err

# Optional 10-step smoke test on v4_depth_rgb (256) before full 60k runs.

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
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v4_depth_rgb
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/_smoke_v4_depth_rgb_256"

export HEIGHT=256
export WIDTH=256
export MAX_NUM_FRAMES=49
export FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1
export NUM_GPUS=1
export GRADIENT_ACCUMULATION_STEPS=8
export TRAIN_STEPS=10
export CHECKPOINTING_STEPS=10
export CHECKPOINTS_TOTAL_LIMIT=2
export LR=1e-4
export LR_WARMUP_STEPS=2
export TRACKER_NAME=casual-coaf-smoke-v4-depth-rgb-256
export TEMPORAL_CAUSAL_ATTENTION=1
export SKIP_PREPARE_I2V_DATA=1

nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2v_lora_causal.sh"
