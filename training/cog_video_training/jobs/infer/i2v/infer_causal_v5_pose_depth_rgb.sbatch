#!/bin/bash
#SBATCH --job-name=iv5pdprc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2v/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2v/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/i2v outputs/infer/i2v


export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset/composed/v5_pose_depth_rgb
export CHECKPOINT_DIR="${CASUAL_ROOT}/outputs/checkpoints/v5_pose_depth_rgb_causal_gradient_accumulation_2"
export CHECKPOINT_STEP=60000
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2v/v5_pose_depth_rgb_causal"
export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=73
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2v_lora_causal.sh"
