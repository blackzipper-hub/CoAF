"""Dataset helpers for I2AV training with state/action manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dataset import VideoDatasetWithResizing


class I2AVVideoDataset(VideoDatasetWithResizing):
    def __init__(
        self,
        *args,
        state_column: str = "state_paths.txt",
        action_column: str = "action_paths.txt",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.state_paths = self._load_paths(state_column, "state_paths.txt")
        self.action_paths = self._load_paths(action_column, "action_paths.txt")
        if len(self.state_paths) != len(self.video_paths):
            raise ValueError(
                f"state/video count mismatch: {len(self.state_paths)} vs {len(self.video_paths)}"
            )
        if len(self.action_paths) != len(self.video_paths):
            raise ValueError(
                f"action/video count mismatch: {len(self.action_paths)} vs {len(self.video_paths)}"
            )

    def _load_paths(self, column: str, fallback_name: str) -> list[Path]:
        if self.dataset_file is not None:
            df = pd.read_csv(self.dataset_file)
            if column not in df.columns:
                raise KeyError(f"Missing column {column!r} in {self.dataset_file}")
            paths = [Path(str(value)) for value in df[column].tolist()]
        else:
            path_file = self.data_root / column
            if not path_file.is_file():
                path_file = self.data_root / fallback_name
            if not path_file.is_file():
                raise FileNotFoundError(f"Missing manifest {path_file}")
            paths = [Path(line.strip()) for line in path_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else self.data_root / path for path in paths]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, list):
            return index
        sample = super().__getitem__(index)
        state = np.load(self.state_paths[index]).astype(np.float32)
        action = np.load(self.action_paths[index]).astype(np.float32)
        if state.ndim != 2 or state.shape[-1] != 7:
            raise ValueError(f"Bad state shape {state.shape} from {self.state_paths[index]}")
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError(f"Bad action shape {action.shape} from {self.action_paths[index]}")
        sample["state"] = torch.from_numpy(state)
        sample["action"] = torch.from_numpy(action)
        return sample


class I2AVCollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]
        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = torch.stack([x["image"] for x in data[0]]).to(dtype=self.weight_dtype, non_blocking=True)
        videos = torch.stack([x["video"] for x in data[0]]).to(dtype=self.weight_dtype, non_blocking=True)
        states = torch.stack([x["state"] for x in data[0]]).to(dtype=torch.float32, non_blocking=True)
        actions = torch.stack([x["action"] for x in data[0]]).to(dtype=torch.float32, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "state": states,
            "action": actions,
        }
