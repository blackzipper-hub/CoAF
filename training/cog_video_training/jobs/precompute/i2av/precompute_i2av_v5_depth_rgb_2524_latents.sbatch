#!/bin/bash
#SBATCH --job-name=pdprgcv5
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --array=0-3
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/precompute/i2av/%x-%A_%a.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/precompute/i2av/%x-%A_%a.err

set -euo pipefail
module purge
source /project/llmsvgen/yazhoux/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

CASUAL_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/precompute/i2av

export HF_HOME="${COAF_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${COAF_ROOT}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${COAF_ROOT}/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb

python "${CASUAL_ROOT}/scripts/precompute_i2av_latents.py" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --height 256 \
  --width 256 \
  --max_num_frames 49 \
  --max_sequence_length 226 \
  --batch_size "${PRECOMPUTE_BATCH_SIZE:-1}" \
  --dtype "${PRECOMPUTE_DTYPE:-bf16}" \
  --shard_index "${SLURM_ARRAY_TASK_ID:-0}" \
  --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --use_slicing \
  --use_tiling
