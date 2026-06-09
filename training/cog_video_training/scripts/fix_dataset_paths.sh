#!/usr/bin/env bash
# Rewrite dataset manifest paths from llmsvgen to mscaisuperpod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cluster_env.sh"

python3 - <<'PY'
from pathlib import Path

PROJECT_ROOT = Path("/project/mscaisuperpod/sunkai/Casual_CoAF")
REPLACEMENTS = [
    (
        "/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF",
        "/project/mscaisuperpod/sunkai/Casual_CoAF",
    ),
    (
        "/project/llmsvgen/sunkai/robomaster_3d/coaf_dataset",
        "/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25",
    ),
]

manifest_names = [
    "videos.txt",
    "images.txt",
    "prompt.txt",
    "validation.json",
    "metadata.csv",
    "state_paths.txt",
    "action_paths.txt",
]

def iter_manifests(root: Path):
    if not root.exists():
        return
    composed = root / "composed"
    if composed.is_dir():
        for variant_dir in sorted(composed.iterdir()):
            if not variant_dir.is_dir():
                continue
            for name in manifest_names:
                path = variant_dir / name
                if path.is_file():
                    yield path
    splits = root / "splits"
    if splits.is_dir():
        for path in sorted(splits.glob("*.json")):
            yield path

changed_files = []
for root in [PROJECT_ROOT / "coaf_dataset_24_25", PROJECT_ROOT / "coaf_dataset_test"]:
    for path in iter_manifests(root):
        text = path.read_text()
        new_text = text
        for old, new in REPLACEMENTS:
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text)
            changed_files.append(str(path))

print(f"Updated {len(changed_files)} manifest files")
for path in changed_files[:20]:
    print(f"  {path}")
if len(changed_files) > 20:
    print(f"  ... and {len(changed_files) - 20} more")
PY

echo "Verifying v4_depth_rgb images.txt"
first_image="$(head -1 "${DATASET_ROOT}/composed/v4_depth_rgb/images.txt")"
echo "${first_image}"
test -f "${first_image}"
echo "OK: first image path resolves"
