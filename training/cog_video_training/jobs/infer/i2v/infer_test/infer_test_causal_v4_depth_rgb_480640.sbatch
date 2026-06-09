#!/bin/bash
#SBATCH --job-name=it4d480c
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2v/infer_test/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/infer/i2v/infer_test/%x-%j.err

# Test-set inference for v4_depth_rgb 480x640 causal I2V LoRA.

set -euo pipefail

module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
CASUAL_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training
TEST_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_test
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/i2v/infer_test outputs/infer/i2v/test outputs/infer/i2v/test_manifests


export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT="${CASUAL_ROOT}/outputs/infer/i2v/test_manifests/coaf_dataset_test_1k"
export CHECKPOINT_DIR="${CASUAL_ROOT}/outputs/checkpoints/v4_depth_rgb_480640_causal"
export CHECKPOINT_STEP=15000
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/i2v/test/v4_depth_rgb_480640_causal"

export HEIGHT=480
export WIDTH=640
export FPS=8
export MAX_NUM_FRAMES=49
export NUM_SAMPLES="${NUM_SAMPLES:-1000}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"

python "${CASUAL_ROOT}/scripts/build_test_i2v_validation.py" \
  --test-root "${TEST_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --max-samples "${NUM_SAMPLES}"

nvidia-smi || true

bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2v_lora_causal.sh"
