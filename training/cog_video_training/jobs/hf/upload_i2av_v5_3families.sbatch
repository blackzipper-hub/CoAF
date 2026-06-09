#!/bin/bash
#SBATCH --job-name=hfup3cv5
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/hf/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/hf/%x-%j.err

# Upload three I2AV checkpoint families (weights only) to BlackZipper/cogvideo.
# Families:
#   v5_depth_rgb_2524_stage1
#   v5_depth_rgb_2524_stage2_raw_action_lingbot_d6cont
#   v5_depth_rgb_2524_stage2_sa_denoise_d6cont
#
# Usage:
#   sbatch jobs/hf/upload_i2av_v5_3families.sbatch
#   HF_TOKEN_NAME=123 sbatch jobs/hf/upload_i2av_v5_3families.sbatch

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
CASUAL_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/hf

export HF_HOME=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/.cache/huggingface
export HF_REPO_ID="${HF_REPO_ID:-BlackZipper/cogvideo}"
export HF_TOKEN_NAME="${HF_TOKEN_NAME:-123}"
export HF_PATH_PREFIX="${HF_PATH_PREFIX:-checkpoints/i2av}"
export UPLOAD_FAMILIES="${UPLOAD_FAMILIES:-v5_depth_rgb_2524_stage1,v5_depth_rgb_2524_stage2_raw_action_lingbot_d6cont,v5_depth_rgb_2524_stage2_sa_denoise_d6cont}"

python -m py_compile "${CASUAL_ROOT}/scripts/upload_i2av_checkpoints_to_hf.py"
python "${CASUAL_ROOT}/scripts/upload_i2av_checkpoints_to_hf.py"
