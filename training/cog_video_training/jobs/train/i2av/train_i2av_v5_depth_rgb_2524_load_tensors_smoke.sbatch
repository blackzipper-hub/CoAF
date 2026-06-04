#!/bin/bash
#SBATCH --job-name=tdpltcv5
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/train/i2av/%x-%j.err

set -euo pipefail
module purge
source /project/llmsvgen/yazhoux/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

CASUAL_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training
export FULL_DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb
export SMOKE_DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb_load_tensors_smoke
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av outputs/checkpoints

export HF_HOME="${COAF_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${COAF_ROOT}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${COAF_ROOT}/.cache/huggingface/datasets"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

python - <<'PY'
import json
import os
from pathlib import Path

full = Path(os.environ["FULL_DATA_ROOT"])
smoke = Path(os.environ["SMOKE_DATA_ROOT"])
stem = "episode_000000"

for dirname in ("videos", "condition_images", "video_latents", "image_latents", "prompt_embeds"):
    (smoke / dirname).mkdir(parents=True, exist_ok=True)

def first_line(name: str) -> str:
    return next(line.strip() for line in (full / name).read_text().splitlines() if line.strip())

video_src = Path(first_line("videos.txt"))
image_src = Path(first_line("images.txt"))

video_dst = smoke / "videos" / video_src.name
image_dst = smoke / "condition_images" / image_src.name
for src, dst in ((video_src, video_dst), (image_src, image_dst)):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

for dirname in ("video_latents", "image_latents", "prompt_embeds"):
    src = full / dirname / f"{stem}.pt"
    dst = smoke / dirname / f"{stem}.pt"
    if not src.is_file():
        raise SystemExit(f"Missing precomputed tensor for smoke test: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

(smoke / "videos.txt").write_text(f"{video_dst}\n")
(smoke / "images.txt").write_text(f"{image_dst}\n")
(smoke / "prompt.txt").write_text(first_line("prompt.txt") + "\n")
(smoke / "state_paths.txt").write_text(first_line("state_paths.txt") + "\n")
(smoke / "action_paths.txt").write_text(first_line("action_paths.txt") + "\n")
(smoke / "validation.json").write_text(
    json.dumps(
        {
            "data": [
                {
                    "sample_index": 0,
                    "caption": first_line("prompt.txt"),
                    "image_path": str(image_dst),
                    "video_path": str(video_dst),
                }
            ]
        },
        indent=2,
    )
    + "\n"
)
print(f"Prepared smoke load_tensors dataset at {smoke}")
PY

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT="${SMOKE_DATA_ROOT}"
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/_smoke_v5_depth_rgb_2524_load_tensors"
export STATE_NORM_STATS=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_STEPS=2 CHECKPOINTING_STEPS=100 CHECKPOINTS_TOTAL_LIMIT=1
export LR=1e-4 LR_WARMUP_STEPS=0
export TRACKER_NAME=casual-coaf-i2av-v5-depth-rgb-2524-load-tensors-smoke
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export LOAD_TENSORS=1
export VALIDATION_STEPS=0

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
