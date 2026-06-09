#!/usr/bin/env python3
"""Build 7D action quantile normalization stats for I2AV training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def read_action_paths(path_file: Path) -> list[Path]:
    if not path_file.is_file():
        raise FileNotFoundError(f"Missing action path manifest: {path_file}")
    paths: list[Path] = []
    for line in path_file.read_text().splitlines():
        value = line.strip()
        if value:
            paths.append(Path(value))
    if not paths:
        raise ValueError(f"No action paths found in {path_file}")
    return paths


def load_actions(paths: list[Path]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing action file: {path}")
        action = np.load(path).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != 7:
            raise ValueError(f"Expected action shape (T, 7), got {action.shape} from {path}")
        chunks.append(action)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action_paths",
        type=Path,
        default=Path(
            "/project/mscaisuperpod/sunkai/Casual_CoAF/"
            "coaf_dataset_24_25/composed/v4_depth_rgb/action_paths.txt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25/action_quantile_norm_stats.pt"),
    )
    parser.add_argument("--lower_quantile", type=float, default=0.01)
    parser.add_argument("--upper_quantile", type=float, default=0.99)
    parser.add_argument("--gripper_threshold", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=1.5)
    args = parser.parse_args()

    if not 0.0 <= args.lower_quantile < args.upper_quantile <= 1.0:
        raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1")

    actions = load_actions(read_action_paths(args.action_paths))
    q01 = np.quantile(actions, args.lower_quantile, axis=0).astype(np.float32)
    q99 = np.quantile(actions, args.upper_quantile, axis=0).astype(np.float32)
    span = q99 - q01
    if np.any(span <= 1e-6):
        bad = np.nonzero(span <= 1e-6)[0].tolist()
        raise ValueError(f"Quantile span too small for action dims: {bad}")

    payload = {
        "norm_method": "quantile",
        "q01": torch.from_numpy(q01),
        "q99": torch.from_numpy(q99),
        "lower_quantile": float(args.lower_quantile),
        "upper_quantile": float(args.upper_quantile),
        "clip": float(args.clip),
        "mean": torch.from_numpy(actions.mean(axis=0).astype(np.float32)),
        "std": torch.from_numpy(actions.std(axis=0).astype(np.float32)),
        "count": int(actions.shape[0]),
        "num_files": int(len(read_action_paths(args.action_paths))),
        "gripper_positive_rate": float(np.mean(actions[:, 6] >= args.gripper_threshold)),
        "gripper_threshold": float(args.gripper_threshold),
        "valid_action_mask": torch.ones(7, dtype=torch.float32),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote {args.output}")
    print(f"count={payload['count']} files={payload['num_files']} method={payload['norm_method']}")
    print(f"q01={q01.tolist()}")
    print(f"q99={q99.tolist()}")
    print(f"gripper_positive_rate={payload['gripper_positive_rate']:.6f}")


if __name__ == "__main__":
    main()
