#!/usr/bin/env bash
set -euo pipefail

COAF_ROOT="/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training"
CASUAL_ROOT="/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training"
MODEL_PATH="${MODEL_PATH:-${COAF_ROOT}/models/CogVideoX-5b-I2V}"
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
if [[ "${CHECKPOINT_BASENAME}" =~ ^checkpoint-[0-9]+$ ]]; then
  CHECKPOINT_TAG="${CHECKPOINT_BASENAME}"
elif [[ -n "${CHECKPOINT_STEP:-}" ]]; then
  CHECKPOINT_TAG="checkpoint-${CHECKPOINT_STEP#checkpoint-}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR}/${CHECKPOINT_TAG}"
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Expected LoRA checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    exit 1
  fi
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

if [[ "$(basename "${INFER_OUTPUT_DIR}")" != "${CHECKPOINT_TAG}" ]]; then
  INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR}/${CHECKPOINT_TAG}"
fi
mkdir -p "${INFER_OUTPUT_DIR}"
echo "LoRA checkpoint: ${CHECKPOINT_DIR}"
echo "Inference output: ${INFER_OUTPUT_DIR}"

"${PYTHON}" "${CASUAL_ROOT}/scripts/infer_cogvideox_i2v_lora.py" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --lora_dir "${CHECKPOINT_DIR}" \
  --output_dir "${INFER_OUTPUT_DIR}" \
  --height "${HEIGHT:-256}" \
  --width "${WIDTH:-256}" \
  --num_frames "${MAX_NUM_FRAMES:-49}" \
  --fps "${FPS:-8}" \
  --guidance_scale "${GUIDANCE_SCALE:-6}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --seed "${SEED:-42}" \
  --temporal_causal_attention
