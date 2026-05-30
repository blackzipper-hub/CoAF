#!/usr/bin/env bash
# Rewrite manifest paths from old prefix to Casual_CoAF/coaf_dataset (optional fallback).
set -euo pipefail

OLD_PREFIX="/project/llmsvgen/sunkai/robomaster_3d/coaf_dataset"
NEW_PREFIX="/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset"
COMPOSED_ROOT="${NEW_PREFIX}/composed"

for variant in v1_pose_rgb v2_flow_rgb v4_depth_rgb v4_depth_rgb_480640 v5_pose_depth_rgb; do
  dir="${COMPOSED_ROOT}/${variant}"
  [[ -d "${dir}" ]] || continue
  for file in videos.txt images.txt validation.json metadata.csv; do
    path="${dir}/${file}"
    [[ -f "${path}" ]] || continue
    sed -i "s|${OLD_PREFIX}|${NEW_PREFIX}|g" "${path}"
  done
  echo "Updated paths in ${dir}"
done
