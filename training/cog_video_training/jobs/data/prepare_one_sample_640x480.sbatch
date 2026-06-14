#!/bin/bash
#SBATCH --job-name=p1os6kv1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/data/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/data/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/data

export DATASET_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25
export COMPOSED_NAME=v4_depth_rgb_one_sample_640x480
export DATA_ROOT="${DATASET_ROOT}/composed/${COMPOSED_NAME}"

python "${DATASET_ROOT}/scripts/prepare_one_sample_640x480.py" \
  --episode-idx 0 \
  --output-name "${COMPOSED_NAME}"

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
python "${CASUAL_ROOT}/scripts/precompute_i2av_latents.py" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --height 480 \
  --width 640 \
  --max_num_frames 49 \
  --max_sequence_length 226 \
  --batch_size 1 \
  --dtype bf16 \
  --overwrite \
  --use_slicing \
  --use_tiling

echo "Prepared and precomputed latents at ${DATA_ROOT}"
