#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cluster_env.sh"

FINETRAINERS_ROOT="${CASUAL_ROOT}/finetrainers"
LEGACY_COGVIDEOX_ROOT="${FINETRAINERS_ROOT}/examples/_legacy/training/cogvideox"
ACCELERATE_CONFIG_DIR="${ACCELERATE_CONFIG_DIR:-${CASUAL_ROOT}/finetrainers/accelerate_configs}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT must be set}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

CONDA_BIN="${CONDA_PREFIX:-}/bin"
if [[ -n "${CONDA_BIN}" && -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON="${PYTHON:-$(command -v python)}"
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Missing python in active conda env (CONDA_PREFIX=${CONDA_PREFIX:-unset})" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
  echo "Missing I2V model at ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${STATE_NORM_STATS}" ]]; then
  echo "Missing state norm stats at ${STATE_NORM_STATS}" >&2
  exit 1
fi

for required_file in videos.txt images.txt prompt.txt validation.json state_paths.txt action_paths.txt; do
  if [[ ! -f "${DATA_ROOT}/${required_file}" ]]; then
    echo "Missing ${DATA_ROOT}/${required_file}" >&2
    exit 1
  fi
done

if [[ -n "${LOAD_TENSORS:-}" ]]; then
  "${PYTHON}" - <<'PY'
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
video_paths = [Path(line.strip()) for line in (data_root / "videos.txt").read_text().splitlines() if line.strip()]
missing = []
for video_path in video_paths:
    stem = video_path.stem
    for dirname in ("video_latents", "image_latents", "prompt_embeds"):
        tensor_path = data_root / dirname / f"{stem}.pt"
        if not tensor_path.is_file():
            missing.append(str(tensor_path))
            break
if missing:
    preview = "\n".join(missing[:10])
    raise SystemExit(
        f"LOAD_TENSORS=1 but {len(missing)} / {len(video_paths)} samples are missing precomputed tensors. "
        f"First missing paths:\n{preview}"
    )
print(f"LOAD_TENSORS=1: found precomputed tensors for {len(video_paths)} samples under {data_root}")
PY
fi

FIRST_VALIDATION_PROMPT="$("${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["DATA_ROOT"]) / "validation.json"
print(json.loads(path.read_text())["data"][0]["caption"])
PY
)"
FIRST_VALIDATION_IMAGE="$("${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["DATA_ROOT"]) / "validation.json"
print(json.loads(path.read_text())["data"][0]["image_path"])
PY
)"

cd "${LEGACY_COGVIDEOX_ROOT}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ACCELERATE_CONFIG_DIR}/uncompiled_${NUM_GPUS}.yaml}"
  if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
    echo "Missing accelerate config: ${ACCELERATE_CONFIG}" >&2
    exit 1
  fi
  if ! "${PYTHON}" -c "import accelerate" >/dev/null 2>&1; then
    echo "accelerate is not installed in ${PYTHON}" >&2
    exit 1
  fi
  LAUNCHER=("${PYTHON}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}")
else
  LAUNCHER=("${PYTHON}")
fi

EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
echo "I2AV launch: ${NUM_GPUS} GPU(s), per-GPU batch=${TRAIN_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH_SIZE}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "none" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
ACTION_ARGS=()
if [[ -n "${ACTION_NORM_STATS:-}" ]]; then
  if [[ ! -f "${ACTION_NORM_STATS}" ]]; then
    echo "Missing action norm stats at ${ACTION_NORM_STATS}" >&2
    exit 1
  fi
  ACTION_ARGS=(--action_norm_stats "${ACTION_NORM_STATS}")
fi
STAGE_ARGS=()
if [[ "${STAGE2_TRAIN_TRANSFORMER_LORA:-0}" == "1" ]]; then
  STAGE_ARGS+=(--stage2_train_transformer_lora)
fi
if [[ "${GRIPPER_CONTINUOUS_ACTION:-0}" == "1" ]]; then
  STAGE_ARGS+=(--gripper_continuous_action)
fi
if [[ "${SA_DENOISE_LOSS:-0}" == "1" ]]; then
  STAGE_ARGS+=(--sa_denoise_loss)
fi

"${LAUNCHER[@]}" cogvideox_image_to_video_lora_i2av.py \
  --pretrained_model_name_or_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --caption_column prompt.txt \
  --video_column videos.txt \
  --image_column images.txt \
  --id_token COAF \
  --height "${HEIGHT:-256}" \
  --width "${WIDTH:-256}" \
  --height_buckets "${HEIGHT:-256}" \
  --width_buckets "${WIDTH:-256}" \
  --max_num_frames "${MAX_NUM_FRAMES:-49}" \
  --frame_buckets "${FRAME_BUCKETS:-${MAX_NUM_FRAMES:-49}}" \
  --fps "${FPS:-8}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_train_steps "${TRAIN_STEPS:-60000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-5000}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-12}" \
  "${RESUME_ARGS[@]}" \
  --rank 128 \
  --lora_alpha 128 \
  --learning_rate "${LR:-1e-4}" \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-200}" \
  --optimizer adamw \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 1e-4 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --ignore_learned_positional_embeddings \
  --enable_slicing \
  --enable_tiling \
  --allow_tf32 \
  --validation_prompt "${FIRST_VALIDATION_PROMPT}" \
  --validation_images "${FIRST_VALIDATION_IMAGE}" \
  --validation_steps "${VALIDATION_STEPS:-0}" \
  --num_validation_videos 0 \
  --guidance_scale "${GUIDANCE_SCALE:-6}" \
  --enable_model_cpu_offload \
  --output_dir "${OUTPUT_DIR}" \
  --tracker_name "${TRACKER_NAME:-casual-coaf-i2av-causal}" \
  --nccl_timeout "${NCCL_TIMEOUT:-7200}" \
  --temporal_causal_attention \
  --enable_i2av \
  --state_norm_stats "${STATE_NORM_STATS}" \
  "${ACTION_ARGS[@]}" \
  --lambda_sa "${LAMBDA_SA:-0.1}" \
  --lambda_s "${LAMBDA_S:-1.0}" \
  --lambda_a "${LAMBDA_A:-1.0}" \
  --lambda_g "${LAMBDA_G:-1.0}" \
  --lambda_c "${LAMBDA_C:-0.5}" \
  --lambda_decoded_state "${LAMBDA_DECODED_STATE:-0.0}" \
  --lambda_decoded_action "${LAMBDA_DECODED_ACTION:-0.0}" \
  --sa_per_frame "${SA_PER_FRAME:-8}" \
  --s0_cond_tokens "${S0_COND_TOKENS:-4}" \
  --i2av_layout "${I2AV_LAYOUT:-legacy}" \
  --pose_pixel_frames "${POSE_PIXEL_FRAMES:-25}" \
  --rgb_pixel_frames "${RGB_PIXEL_FRAMES:-24}" \
  --train_stage "${TRAIN_STAGE:-joint}" \
  "${STAGE_ARGS[@]}" \
  ${RELAYOUT_V5:+--relayout_v5} \
  ${LOAD_TENSORS:+--load_tensors}
