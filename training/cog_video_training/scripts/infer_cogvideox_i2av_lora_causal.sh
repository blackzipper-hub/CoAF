#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cluster_env.sh"

DATA_ROOT="${DATA_ROOT:?DATA_ROOT must be set}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:?INFER_OUTPUT_DIR must be set}"

CONDA_BIN="${CONDA_PREFIX:-}/bin"
if [[ -n "${CONDA_BIN}" && -d "${CONDA_BIN}" ]]; then
  export PATH="${CONDA_BIN}:${PATH}"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYTHON="${PYTHON:-$(command -v python)}"

export PYTHONPATH="${CASUAL_ROOT}/finetrainers:${PYTHONPATH:-}"

CHECKPOINT_BASENAME="$(basename "${CHECKPOINT_DIR}")"
CHECKPOINTS="${CHECKPOINTS:-${CHECKPOINT_STEP:-}}"
if [[ -n "${CHECKPOINTS}" ]]; then
  if [[ "${CHECKPOINTS}" = /* ]]; then
    CHECKPOINT_DIR="${CHECKPOINTS}"
  elif [[ "${CHECKPOINT_BASENAME}" =~ ^checkpoint-[0-9]+$ ]]; then
    CHECKPOINT_DIR="$(dirname "${CHECKPOINT_DIR}")/checkpoint-${CHECKPOINTS#checkpoint-}"
  else
    CHECKPOINT_DIR="${CHECKPOINT_DIR}/checkpoint-${CHECKPOINTS#checkpoint-}"
  fi
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Expected I2AV checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
  CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"
elif [[ "${CHECKPOINT_BASENAME}" =~ ^checkpoint-[0-9]+$ ]]; then
  CHECKPOINT_TAG="${CHECKPOINT_BASENAME}"
else
  LATEST_CHECKPOINT_STEP=-1
  LATEST_CHECKPOINT_DIR=""
  for path in "${CHECKPOINT_DIR}"/checkpoint-*; do
    [[ -d "${path}" ]] || continue
    step="${path##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > LATEST_CHECKPOINT_STEP )); then
      LATEST_CHECKPOINT_STEP="${step}"
      LATEST_CHECKPOINT_DIR="${path}"
    fi
  done

  if [[ -n "${LATEST_CHECKPOINT_DIR}" ]]; then
    CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR}"
    CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"
  else
    CHECKPOINT_TAG="checkpoint-final"
  fi
fi

if [[ "${INFER_OUTPUT_DIR_IS_FINAL:-0}" == "1" ]]; then
  :
elif [[ -n "${MODEL_NAME:-}" && "$(basename "${INFER_OUTPUT_DIR}")" != "${MODEL_NAME}_${CHECKPOINT_TAG}" ]]; then
  INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR}/${MODEL_NAME}_${CHECKPOINT_TAG}"
elif [[ "$(basename "${INFER_OUTPUT_DIR}")" != "${CHECKPOINT_TAG}" ]]; then
  INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR}/${CHECKPOINT_TAG}"
fi
mkdir -p "${INFER_OUTPUT_DIR}"
echo "I2AV checkpoint: ${CHECKPOINT_DIR}"
echo "Inference output: ${INFER_OUTPUT_DIR}"

I2AV_EXTRA_ARGS=()
if [[ -n "${I2AV_LAYOUT:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--i2av_layout "${I2AV_LAYOUT}")
fi
if [[ -n "${TRAIN_DATA_ROOT:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--train_data_root "${TRAIN_DATA_ROOT}")
fi
if [[ -n "${TRAIN_NUM_SAMPLES:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--train_num_samples "${TRAIN_NUM_SAMPLES}")
fi
if [[ -n "${ACTION_NORM_STATS:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--action_norm_stats "${ACTION_NORM_STATS}")
fi
if [[ "${GRIPPER_CONTINUOUS_ACTION:-0}" == "1" ]]; then
  I2AV_EXTRA_ARGS+=(--gripper_continuous_action)
fi
if [[ "${SA_DENOISE_LOSS:-0}" == "1" ]]; then
  I2AV_EXTRA_ARGS+=(--sa_denoise_loss)
fi
if [[ -n "${POSE_PIXEL_FRAMES:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--pose_pixel_frames "${POSE_PIXEL_FRAMES}")
fi
if [[ -n "${RGB_PIXEL_FRAMES:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--rgb_pixel_frames "${RGB_PIXEL_FRAMES}")
fi
if [[ -n "${INFER_DEVICE:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--device "${INFER_DEVICE}")
fi
if [[ -n "${INFER_STAGE:-}" ]]; then
  I2AV_EXTRA_ARGS+=(--infer_stage "${INFER_STAGE}")
fi
if [[ "${ENABLE_MODEL_CPU_OFFLOAD:-0}" == "1" ]]; then
  I2AV_EXTRA_ARGS+=(--enable_model_cpu_offload)
fi

"${PYTHON}" "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora.py" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --lora_dir "${CHECKPOINT_DIR}" \
  --output_dir "${INFER_OUTPUT_DIR}" \
  --state_norm_stats "${STATE_NORM_STATS}" \
  --height "${HEIGHT:-256}" \
  --width "${WIDTH:-256}" \
  --num_frames "${MAX_NUM_FRAMES:-49}" \
  --fps "${FPS:-8}" \
  --guidance_scale "${GUIDANCE_SCALE:-6}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --seed "${SEED:-42}" \
  "${I2AV_EXTRA_ARGS[@]}"
