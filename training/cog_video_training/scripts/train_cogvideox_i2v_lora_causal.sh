#!/usr/bin/env bash
set -euo pipefail

COAF_ROOT="/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training"
CASUAL_ROOT="/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training"
FINETRAINERS_ROOT="${CASUAL_ROOT}/finetrainers"
LEGACY_COGVIDEOX_ROOT="${FINETRAINERS_ROOT}/examples/_legacy/training/cogvideox"
ACCELERATE_CONFIG_DIR="${COAF_ROOT}/finetrainers/accelerate_configs"
MODEL_PATH="${MODEL_PATH:-${COAF_ROOT}/models/CogVideoX-5b-I2V}"
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
export PYTHONPATH="${FINETRAINERS_ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-$(command -v python)}"
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Missing python in active conda env (CONDA_PREFIX=${CONDA_PREFIX:-unset})" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
  echo "Missing I2V model at ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_PREPARE_I2V_DATA:-1}" != "1" ]]; then
  "${PYTHON}" "${COAF_ROOT}/scripts/prepare_coaf_i2v_rgb_data.py" \
    --source-data-root "${SOURCE_DATA_ROOT:-${DATA_ROOT}}" \
    --condition-data-root "${CONDITION_DATA_ROOT:-${DATA_ROOT}}" \
    --output-data-root "${DATA_ROOT}" \
    --validation-indices "${VALIDATION_INDICES:-0:7}"
else
  for required_file in videos.txt images.txt prompt.txt validation.json; do
    if [[ ! -f "${DATA_ROOT}/${required_file}" ]]; then
      echo "Missing ${DATA_ROOT}/${required_file}" >&2
      exit 1
    fi
  done
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

EXTRA_TRAIN_ARGS=()
if [[ "${TEMPORAL_CAUSAL_ATTENTION:-1}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--temporal_causal_attention)
fi

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
echo "Training launch: ${NUM_GPUS} GPU(s), per-GPU batch=${TRAIN_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH_SIZE}"

"${LAUNCHER[@]}" cogvideox_image_to_video_lora.py \
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
  --resume_from_checkpoint latest \
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
  --num_validation_videos 1 \
  --guidance_scale "${GUIDANCE_SCALE:-6}" \
  --enable_model_cpu_offload \
  --output_dir "${OUTPUT_DIR}" \
  --tracker_name "${TRACKER_NAME:-casual-coaf-i2v-causal}" \
  --nccl_timeout "${NCCL_TIMEOUT:-7200}" \
  "${EXTRA_TRAIN_ARGS[@]}"
