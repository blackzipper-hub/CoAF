#!/usr/bin/env bash
# Verify manifests and training wrapper preflight for all 5 causal jobs.
set -euo pipefail

CASUAL_ROOT="/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/training/cog_video_training"
DATASET_ROOT="/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/composed"
COAF_ROOT="/project/llmsvgen/sunkai/robomaster_3d/CoAF/training/cog_video_training"

check_dataset() {
  local name="$1" height="$2" width="$3" frames="$4"
  local root="${DATASET_ROOT}/${name}"
  for f in videos.txt images.txt prompt.txt validation.json; do
    [[ -f "${root}/${f}" ]] || { echo "MISSING ${root}/${f}"; exit 1; }
  done
  local count first_video first_image
  count="$(wc -l < "${root}/videos.txt")"
  first_video="$(head -1 "${root}/videos.txt")"
  first_image="$(head -1 "${root}/images.txt")"
  [[ -f "${first_video}" ]] || { echo "UNREADABLE video: ${first_video}"; exit 1; }
  [[ -f "${first_image}" ]] || { echo "UNREADABLE image: ${first_image}"; exit 1; }
  python3 - <<PY
import cv2
count = ${count}
cap = cv2.VideoCapture("${first_video}")
ret, frame = cap.read()
cap.release()
if not ret:
    raise SystemExit("Cannot read ${first_video}")
h, w = frame.shape[:2]
if (h, w) != (${height}, ${width}):
    raise SystemExit(f"${name}: expected ${height}x${width}, got {h}x{w}")
print(f"OK ${name}: {count} samples, first frame {h}x{w}, target ${frames} frames")
PY
}

echo "=== Symlink check ==="
test -f /project/llmsvgen/sunkai/robomaster_3d/coaf_dataset/composed/v4_depth_rgb/videos/episode_000000.mp4
echo "OK: coaf_dataset symlink resolves"

echo "=== Model check ==="
test -f "${COAF_ROOT}/models/CogVideoX-5b-I2V/model_index.json"
echo "OK: CogVideoX-5b-I2V present"

echo "=== Manifest + path checks ==="
check_dataset v4_depth_rgb_480640 480 640 49
check_dataset v4_depth_rgb 256 256 49
check_dataset v1_pose_rgb 256 256 49
check_dataset v5_pose_depth_rgb 256 256 73
check_dataset v2_flow_rgb 256 256 49

echo "=== validation.json path check ==="
DATA_ROOT="${DATASET_ROOT}/v4_depth_rgb" python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["DATA_ROOT"]) / "validation.json"
d = json.loads(p.read_text())["data"][0]
assert Path(d["image_path"]).is_file()
assert Path(d["video_path"]).is_file()
print("OK: validation.json paths readable")
PY

echo "=== All I2V checks passed ==="

echo "=== I2AV checks ==="
STATE_NORM_STATS="/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset/state_norm_stats.pt"
test -f "${STATE_NORM_STATS}"
echo "OK: state_norm_stats.pt present"

check_i2av_dataset() {
  local name="$1"
  local root="${DATASET_ROOT}/${name}"
  for f in state_paths.txt action_paths.txt; do
    [[ -f "${root}/${f}" ]] || { echo "MISSING ${root}/${f}"; exit 1; }
  done
  local vcount scount
  vcount="$(wc -l < "${root}/videos.txt")"
  scount="$(wc -l < "${root}/state_paths.txt")"
  if [[ "${vcount}" != "${scount}" ]]; then
    echo "MISMATCH ${name}: videos=${vcount} state_paths=${scount}"
    exit 1
  fi
  echo "OK I2AV ${name}: ${vcount} aligned state/action paths"
}

check_i2av_dataset v1_pose_rgb
check_i2av_dataset v2_flow_rgb
check_i2av_dataset v4_depth_rgb
check_i2av_dataset v4_depth_rgb_480640
check_i2av_dataset v5_pose_depth_rgb

test -f "${CASUAL_ROOT}/finetrainers/finetrainers/patches/models/cogvideox/state_action.py"
test -f "${CASUAL_ROOT}/scripts/train_cogvideox_i2av_lora_causal.sh"
echo "OK: Casual I2AV fork and launch script present"

echo "=== All checks passed (I2V + I2AV) ==="
