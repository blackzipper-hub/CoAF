#!/bin/bash
#SBATCH --job-name=t1of2kv1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod
#SBATCH --output=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av_pt/%x-%j.out
#SBATCH --error=/project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/logs/train/i2av_pt/%x-%j.err

set -euo pipefail
module purge
source /project/mscaisuperpod/sunkai/Casual_CoAF/training/cog_video_training/scripts/cluster_env.sh
cd "${CASUAL_ROOT}"
mkdir -p logs/train/i2av_pt outputs/checkpoints/i2av

export SRC_DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb
export TINY_DATA_ROOT=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/composed/v4_depth_rgb_one_sample_overfit

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["SRC_DATA_ROOT"])
dst = Path(os.environ["TINY_DATA_ROOT"])
dst.mkdir(parents=True, exist_ok=True)

def first_nonempty(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            return line
    raise RuntimeError(f"No non-empty lines in {path}")

first_video = first_nonempty(src / "videos.txt")
first_image = first_nonempty(src / "images.txt")
first_prompt = first_nonempty(src / "prompt.txt")
first_state = first_nonempty(src / "state_paths.txt")
first_action = first_nonempty(src / "action_paths.txt")

for name, value in {
    "videos.txt": first_video,
    "images.txt": first_image,
    "prompt.txt": first_prompt,
    "state_paths.txt": first_state,
    "action_paths.txt": first_action,
}.items():
    (dst / name).write_text(value + "\n", encoding="utf-8")

payload = json.loads((src / "validation.json").read_text(encoding="utf-8"))
payload["data"] = payload.get("data", [])[:1]
if payload["data"]:
    payload["data"][0]["video_path"] = first_video
    payload["data"][0]["image_path"] = first_image
    payload["data"][0]["caption"] = first_prompt
(dst / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

stem = Path(first_video).stem
for dirname in ("video_latents", "image_latents", "prompt_embeds"):
    (dst / dirname).mkdir(exist_ok=True)
    source = src / dirname / f"{stem}.pt"
    target = dst / dirname / f"{stem}.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.is_symlink() or target.is_file():
        target.unlink()
    target.symlink_to(source)

print(f"Prepared one-sample overfit data at {dst}")
print(f"Sample stem: {stem}")
PY

export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT="${TINY_DATA_ROOT}"
export OUTPUT_DIR="${CASUAL_ROOT}/outputs/checkpoints/i2av/v5_depth_rgb_2524_stage2_sa_denoise_d6cont_qnt_fix1_one_sample_2k"
export STATE_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt
export ACTION_NORM_STATS=/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/action_quantile_norm_stats.pt

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49 FRAME_BUCKETS=49
export TRAIN_BATCH_SIZE=1 NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_STEPS=2000 CHECKPOINTING_STEPS=200 CHECKPOINTS_TOTAL_LIMIT=12
export LR="${LR:-1e-4}" LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"
export TRACKER_NAME=casual-coaf-i2av-pt-v5-depth-rgb-2524-stage2-sa-denoise-d6cont-qnt-fix1-one-sample-2k
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export TRAIN_STAGE=stage2
export STAGE2_TRAIN_TRANSFORMER_LORA=1
export GRIPPER_CONTINUOUS_ACTION=1
export SA_DENOISE_LOSS=1
export LAMBDA_S="${LAMBDA_S:-1.0}"
export LAMBDA_A="${LAMBDA_A:-2.0}"
export LAMBDA_DECODED_STATE="${LAMBDA_DECODED_STATE:-0.1}"
export LAMBDA_DECODED_ACTION="${LAMBDA_DECODED_ACTION:-1.0}"
export LAMBDA_G="${LAMBDA_G:-1.0}"
export LAMBDA_C="${LAMBDA_C:-0.1}"
export LOAD_TENSORS=1
export RESUME_FROM_CHECKPOINT=none
export VALIDATION_STEPS=0

nvidia-smi || true
bash "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
