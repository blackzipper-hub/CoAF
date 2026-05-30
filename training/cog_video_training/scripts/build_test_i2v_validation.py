#!/usr/bin/env python3
"""Build CogVideoX I2V validation manifests from coaf_dataset_test raw episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_instruction(episode_dir: Path) -> str:
    instruction_path = episode_dir / "instruction" / "instruction.txt"
    if instruction_path.is_file():
        return instruction_path.read_text(encoding="utf-8").strip()

    manifest_path = episode_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("instruction", "").strip()

    raise FileNotFoundError(f"Missing instruction for {episode_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("/project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF/coaf_dataset_test"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--id-token", default="COAF")
    args = parser.parse_args()

    raw_root = args.test_root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)

    episodes = sorted(path for path in raw_root.glob("episode_*") if path.is_dir())
    if args.max_samples > 0:
        episodes = episodes[: args.max_samples]

    data = []
    image_lines = []
    prompt_lines = []
    for episode_dir in episodes:
        image_path = episode_dir / "rgb" / "frame_0001.png"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        prompt = load_instruction(episode_dir)
        if args.id_token:
            prompt = f"{args.id_token} {prompt}"

        image_lines.append(str(image_path.resolve()))
        prompt_lines.append(prompt)
        data.append(
            {
                "image_path": str(image_path.resolve()),
                "caption": prompt,
                "episode": episode_dir.name,
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (args.output_root / "prompt.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    (args.output_root / "validation.json").write_text(json.dumps({"data": data}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(data)} test validation samples to {args.output_root}")


if __name__ == "__main__":
    main()
