#!/usr/bin/env python3
"""Create v5 25+24 manifest mirrors for runtime-relayout I2AV training.

This helper keeps source videos untouched and writes manifests that can be used
with ``--relayout_v5``. Full video re-composition can replace these manifests
later without changing the trainer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MANIFESTS = ("videos.txt", "images.txt", "prompt.txt", "state_paths.txt", "action_paths.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copy-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in MANIFESTS:
        src = args.source / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing source manifest: {src}")
        shutil.copyfile(src, args.output / name)

    validation = args.source / "validation.json"
    if validation.is_file() and args.copy_validation:
        payload = json.loads(validation.read_text(encoding="utf-8"))
        (args.output / "validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    readme = (
        "This directory mirrors an existing [24 reason | 25 RGB] composed dataset.\n"
        "Use it with cogvideox_image_to_video_lora_i2av.py --i2av_layout v5 --relayout_v5 "
        "to interpret samples as [25 reason | 24 RGB] at runtime.\n"
    )
    (args.output / "README_v5_2524.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
