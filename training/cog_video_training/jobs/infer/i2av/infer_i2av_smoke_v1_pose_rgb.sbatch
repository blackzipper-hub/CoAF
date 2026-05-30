#!/bin/bash
#SBATCH --job-name=iismoke1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/infer/i2av/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/infer/i2av/%x-%j.err

set -euo pipefail
module purge
source /project/llmsvgen/yazhoux/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

CASUAL_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/i2av outputs/infer/i2av

export HF_HOME="${COAF_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${COAF_ROOT}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${COAF_ROOT}/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed/v1_pose_rgb
export CHECKPOINT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/smoke_v1_pose_rgb"
export CHECKPOINT_STEP=10
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2av/smoke_v1_pose_rgb"
export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh"
