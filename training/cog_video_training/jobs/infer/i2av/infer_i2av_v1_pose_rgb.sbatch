#!/bin/bash
#SBATCH --job-name=ii1prgbc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2av/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2av/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/i2av outputs/infer/i2av


export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset/composed/v1_pose_rgb
export CHECKPOINT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v1_pose_rgb"
export CHECKPOINT_STEP=60000
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2av/v1_pose_rgb"
export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh"
