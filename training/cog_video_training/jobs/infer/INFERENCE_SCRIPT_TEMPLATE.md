# Inference Script Template

This document defines the common pattern for adding inference jobs under `training/cog_video_training/jobs/infer`.

## Placement

New inference sbatch files must live under:

```text
training/cog_video_training/jobs/infer/<task_type>/
```

Use the task type already used by nearby scripts, for example `i2v` or `i2av`.

## Sbatch Structure

Start from a nearby working sbatch script and keep these sections:

```bash
#!/bin/bash
#SBATCH --job-name=<short_name>
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=llmsvgen
#SBATCH --output=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/infer/<task_type>/%x-%j.out
#SBATCH --error=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training/logs/infer/<task_type>/%x-%j.err

set -euo pipefail
module purge
source /project/llmsvgen/yazhoux/miniconda3/etc/profile.d/conda.sh
conda activate coaf_train

CASUAL_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training
COAF_ROOT=/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training
cd "${CASUAL_ROOT}"
mkdir -p logs/infer/<task_type> outputs/infer/<task_type>

export HF_HOME="${COAF_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${COAF_ROOT}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${COAF_ROOT}/.cache/huggingface/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
```

## Checkpoint Selection

Every inference script should support `CHECKPOINTS`:

```bash
CHECKPOINT_BASE_DIR="${CASUAL_ROOT}/outputs/checkpoints/<checkpoint_family>"
CHECKPOINTS="${CHECKPOINTS:-${CHECKPOINT_STEP:-}}"
if [[ -n "${CHECKPOINTS}" ]]; then
  if [[ "${CHECKPOINTS}" = /* ]]; then
    export CHECKPOINT_DIR="${CHECKPOINTS}"
  else
    export CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/checkpoint-${CHECKPOINTS#checkpoint-}"
  fi
else
  LATEST_CHECKPOINT_STEP=-1
  LATEST_CHECKPOINT_DIR=""
  for path in "${CHECKPOINT_BASE_DIR}"/checkpoint-*; do
    [[ -d "${path}" ]] || continue
    step="${path##*-}"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    if (( step > LATEST_CHECKPOINT_STEP )); then
      LATEST_CHECKPOINT_STEP="${step}"
      LATEST_CHECKPOINT_DIR="${path}"
    fi
  done
  if [[ -z "${LATEST_CHECKPOINT_DIR}" ]]; then
    echo "No checkpoint-* directories found under ${CHECKPOINT_BASE_DIR}" >&2
    exit 1
  fi
  export CHECKPOINT_DIR="${LATEST_CHECKPOINT_DIR}"
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Expected checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"
```

Usage examples:

```bash
sbatch jobs/infer/i2av/example.sbatch
CHECKPOINTS=checkpoint-10000 sbatch jobs/infer/i2av/example.sbatch
CHECKPOINTS=10000 sbatch jobs/infer/i2av/example.sbatch
CHECKPOINTS=/abs/path/to/checkpoint-10000 sbatch jobs/infer/i2av/example.sbatch
```

## Output Directory

The output folder name must include both the model name and checkpoint:

```bash
export MODEL_NAME="<model_name>"
export INFER_OUTPUT_DIR="${CASUAL_ROOT}/outputs/infer/<task_type>/${MODEL_NAME}_${CHECKPOINT_TAG}"
export INFER_OUTPUT_DIR_IS_FINAL=1
```

This produces paths like:

```text
outputs/infer/i2av/i2av_v5_depth_rgb_2524_checkpoint-10000/
```

## Performance Defaults

For custom inference loops, especially I2AV loops that directly call transformer forward functions, prefer CUDA-resident inference:

```bash
export INFER_DEVICE="${INFER_DEVICE:-cuda}"
export ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"
```

Use CPU offload only when the job cannot fit in GPU memory:

```bash
ENABLE_MODEL_CPU_OFFLOAD=1 sbatch jobs/infer/<task_type>/<script>.sbatch
```

Reason: `enable_model_cpu_offload()` is safe for diffusers native `pipe(...)` calls, but custom denoising loops can repeatedly move tensors between CPU and GPU and become much slower. On H800-class GPUs, keeping the model on CUDA is the preferred default.

## Inference Parameters

Expose these common environment overrides:

```bash
export HEIGHT="${HEIGHT:-256}"
export WIDTH="${WIDTH:-256}"
export FPS="${FPS:-8}"
export MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-49}"
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"
export SEED="${SEED:-42}"
```

For test-set inference, use the test dataset and default to the first 14 samples:

```bash
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_test
export NUM_SAMPLES="${NUM_SAMPLES:-14}"
```

For quick validation, run fewer samples and fewer steps:

```bash
NUM_SAMPLES=1 NUM_INFERENCE_STEPS=10 sbatch jobs/infer/<task_type>/<script>.sbatch
```

## Dataset Loading

Inference code should support both the composed validation layout and the test-set layout.

Composed validation layout:

```text
<data_root>/validation.json
<data_root>/state_paths.txt
<data_root>/action_paths.txt
```

The loader should read `validation.json`, use each item's `sample_index`, and map it to `state_paths.txt` and `action_paths.txt`.

Test-set layout:

```text
<data_root>/splits/test_1k_metadata.json
<data_root>/raw/episode_000000/rgb/frame_0001.png
<data_root>/raw/episode_000000/state/state.npy
<data_root>/raw/episode_000000/action/action.npy
<data_root>/raw/episode_000000/instruction/instruction.txt
```

If `validation.json` is missing, the loader should fall back to `splits/test_1k_metadata.json`. Use the first `NUM_SAMPLES` metadata rows, map each row's `dataset_idx` to `raw/episode_<dataset_idx:06d>`, and use:

```text
image_path: first rgb/frame_*.png
prompt: instruction from metadata, or instruction/instruction.txt
state_path: state/state.npy
action_path: action/action.npy
```

This keeps the same inference code working for both training validation data and `/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_test`.

## Action Trajectory Outputs

If the model predicts action trajectories, the inference script must save them together with the video outputs. Use this naming pattern:

```text
sample_000.mp4
sample_000_pred_state.npy
sample_000_pred_action.npy
sample_000.json
```

The JSON should include:

```json
{
  "prompt": "...",
  "image_path": "...",
  "state_path": "...",
  "action_path": "...",
  "lora_dir": "...",
  "pred_state_path": "...",
  "pred_action_path": "..."
}
```

For I2AV models, do not treat the job as complete unless the action trajectory file is written.

## Validation Checklist

Before submitting a full job:

- Run `bash -n` on the sbatch and called shell script.
- Run `python -m py_compile` on any edited Python inference file.
- Submit a short validation job with `NUM_SAMPLES=1 NUM_INFERENCE_STEPS=10`.
- Check both `.err` and `.out` logs.
- Confirm the output directory includes model name and checkpoint.
- Confirm test-set inference works without `validation.json` by loading `splits/test_1k_metadata.json`.
- Confirm action models write `*_pred_action.npy`.
- Compare step speed against nearby reference logs. Large slowdowns usually indicate CPU offload or repeated device transfers.

## Minimal I2AV Job Example

```bash
export MODEL_NAME="i2av_v5_depth_rgb_2524"
export MODEL_PATH="${COAF_ROOT}/models/CogVideoX-5b-I2V"
export DATA_ROOT=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_test
export STATE_NORM_STATS=/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_24_25/state_norm_stats.pt

export HEIGHT=256 WIDTH=256 FPS=8 MAX_NUM_FRAMES=49
export I2AV_LAYOUT=v5 POSE_PIXEL_FRAMES=25 RGB_PIXEL_FRAMES=24
export NUM_SAMPLES="${NUM_SAMPLES:-14}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-6}"
export INFER_DEVICE="${INFER_DEVICE:-cuda}"
export ENABLE_MODEL_CPU_OFFLOAD="${ENABLE_MODEL_CPU_OFFLOAD:-0}"

bash "${CASUAL_ROOT}/scripts/infer_cogvideox_i2av_lora_causal.sh"
```
