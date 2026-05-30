# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import logging
import math
import os
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import diffusers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel


from args import get_args  # isort:skip
from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)


logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    videos=None,
    base_model: str = None,
    validation_prompt=None,
    repo_folder=None,
    fps=8,
):
    widget_dict = []
    if videos is not None:
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(repo_folder, f"final_video_{i}.mp4", fps=fps))
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": f"video_{i}.mp4"},
                }
            )

    model_description = f"""
# CogVideoX LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of the CogVideoX model `{base_model}`.

The model was trained using [CogVideoX Factory](https://github.com/a-r-r-o-w/cogvideox-factory) - a repository containing memory-optimized training scripts for the CogVideoX family of models using [TorchAO](https://github.com/pytorch/ao) and [DeepSpeed](https://github.com/microsoft/DeepSpeed). The scripts were adopted from [CogVideoX Diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/cogvideo/train_cogvideox_lora.py).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

```py
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")

# The LoRA adapter weights are determined by what was used for training.
# In this case, we assume `--lora_alpha` is 32 and `--rank` is 64.
# It can be made lower or higher from what was used in training to decrease or amplify the effect
# of the LoRA upto a tolerance, beyond which one might notice no effect at all or overflows.
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

image = load_image("/path/to/image.png")
video = pipe(image=image, prompt="{validation_prompt}", guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=validation_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "image-to-video",
        "diffusers-training",
        "diffusers",
        "lora",
        "cogvideox",
        "cogvideox-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    accelerator: Accelerator,
    pipe: CogVideoXImageToVideoPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    return videos


def run_validation(
    args: Dict[str, Any],
    accelerator: Accelerator,
    transformer,
    scheduler,
    model_config: Dict[str, Any],
    weight_dtype: torch.dtype,
) -> None:
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=unwrap_model(accelerator, transformer),
        scheduler=scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
    validation_images = args.validation_images.split(args.validation_prompt_separator)
    for validation_image, validation_prompt in zip(validation_images, validation_prompts):
        pipeline_args = {
            "image": load_image(validation_image),
            "prompt": validation_prompt,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "num_frames": args.max_num_frames,
            "max_sequence_length": model_config.max_text_seq_length,
        }

        log_validation(
            pipe=pipe,
            args=args,
            accelerator=accelerator,
            pipeline_args=pipeline_args,
        )

    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)


class SevenDoFHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_features)


class CogVideoXFeatureHooks:
    def __init__(self, transformer: torch.nn.Module, layer_indices: list[int]) -> None:
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise AttributeError("CogVideoX transformer does not expose transformer_blocks; cannot attach 7DoF hooks")
        if not layer_indices:
            raise ValueError("At least one feature layer is required for 7DoF supervision")
        self.features: list[torch.Tensor] = []
        self.handles = []
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(blocks):
                raise IndexError(f"Feature layer {layer_idx} is outside transformer block range 0..{len(blocks) - 1}")
            self.handles.append(blocks[layer_idx].register_forward_hook(self._hook))

    @staticmethod
    def _first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = CogVideoXFeatureHooks._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = CogVideoXFeatureHooks._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _hook(self, _module, _inputs, output) -> None:
        tensor = self._first_tensor(output)
        if tensor is not None and tensor.ndim >= 3:
            self.features.append(tensor)

    def clear(self) -> None:
        self.features.clear()

    def pooled_features(self) -> torch.Tensor:
        if not self.features:
            raise RuntimeError("No CogVideoX features were captured for 7DoF head")
        pooled = []
        for feature in self.features:
            if feature.ndim == 3:
                pooled.append(feature.mean(dim=1))
            else:
                pooled.append(feature.flatten(1, -2).mean(dim=1))
        return torch.stack(pooled, dim=0).mean(dim=0)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def resolve_feature_layers(transformer: torch.nn.Module, spec: str) -> list[int]:
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise AttributeError("CogVideoX transformer does not expose transformer_blocks")
    num_layers = len(blocks)
    spec = spec.strip().lower()
    if spec == "auto":
        start = max(0, int(num_layers * 0.7))
        stop = min(num_layers, max(start + 1, int(num_layers * 0.8)))
        return list(range(start, stop))
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def infer_transformer_hidden_size(config) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "inner_dim"):
        return int(config.inner_dim)
    if hasattr(config, "num_attention_heads") and hasattr(config, "attention_head_dim"):
        return int(config.num_attention_heads) * int(config.attention_head_dim)
    raise AttributeError("Cannot infer CogVideoX transformer hidden size for 7DoF head")


def match_state_length(states: torch.Tensor, target_length: int) -> torch.Tensor:
    if states.shape[1] == target_length:
        return states
    states = states.transpose(1, 2)
    states = F.interpolate(states, size=target_length, mode="linear", align_corners=False)
    return states.transpose(1, 2)


def rgb_anchor_latent_index(rgb_anchor_frame: int, latent_num_frames: int) -> int:
    return max(0, min(latent_num_frames - 1, rgb_anchor_frame // 4))


def build_pose_rgb_condition_latents(
    pose_latents: torch.Tensor,
    rgb_latents: torch.Tensor,
    video_latents: torch.Tensor,
    rgb_anchor_frame: int,
) -> torch.Tensor:
    condition_latents = torch.zeros_like(video_latents)
    rgb_index = rgb_anchor_latent_index(rgb_anchor_frame, video_latents.shape[1])
    condition_latents[:, :1] = pose_latents[:, :1]
    condition_latents[:, rgb_index : rgb_index + 1] = rgb_latents[:, :1]
    return condition_latents


class PoseRGBVideoDatasetWithResizing(VideoDatasetWithResizing):
    def __init__(self, *args, pose_image_column: str = "pose_image", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pose_image_paths = self._load_pose_image_paths(pose_image_column)
        if len(self.pose_image_paths) != len(self.video_paths):
            raise ValueError(
                f"Expected one pose image per video but found {len(self.pose_image_paths)} poses and {len(self.video_paths)} videos"
            )

    def _load_pose_image_paths(self, pose_image_column: str) -> list[Path]:
        if self.dataset_file is not None:
            df = pd.read_csv(self.dataset_file)
            if pose_image_column not in df.columns:
                raise KeyError(f"Missing pose image column {pose_image_column!r} in {self.dataset_file}")
            paths = [Path(str(value)) for value in df[pose_image_column].tolist()]
        else:
            pose_file = self.data_root / pose_image_column
            if not pose_file.is_file():
                pose_file = self.data_root / "pose_images.txt"
            if not pose_file.is_file():
                raise FileNotFoundError(
                    f"Missing pose image manifest under {self.data_root}; expected {pose_image_column} or pose_images.txt"
                )
            paths = [Path(line.strip()) for line in pose_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else self.data_root / path for path in paths]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index
        sample = super().__getitem__(index)
        _, _, height, width = sample["image"].shape
        sample["pose_image"] = self._preprocess_condition_image(self.pose_image_paths[index], height, width)
        return sample


class StateVideoDatasetWithResizing(PoseRGBVideoDatasetWithResizing):
    def __init__(self, *args, state_column: str = "state_path", dof_target_length: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dof_target_length = dof_target_length
        self.state_paths = self._load_state_paths(state_column)
        if len(self.state_paths) != len(self.video_paths):
            raise ValueError(f"Expected one state path per video but found {len(self.state_paths)} states and {len(self.video_paths)} videos")

    def _load_state_paths(self, state_column: str) -> list[Path]:
        if self.dataset_file is not None:
            df = pd.read_csv(self.dataset_file)
            if state_column not in df.columns:
                raise KeyError(f"Missing state column {state_column!r} in {self.dataset_file}")
            paths = [Path(str(value)) for value in df[state_column].tolist()]
        else:
            state_file = self.data_root / state_column
            if not state_file.is_file():
                state_file = self.data_root / "state_paths.txt"
            if not state_file.is_file():
                raise FileNotFoundError(f"Missing state path manifest under {self.data_root}; expected {state_column} or state_paths.txt")
            paths = [Path(line.strip()) for line in state_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else self.data_root / path for path in paths]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index
        sample = super().__getitem__(index)
        state = np.load(self.state_paths[index]).astype(np.float32)
        if state.ndim != 2 or state.shape[-1] != 7:
            raise ValueError(f"Expected state array with shape [T, 7], got {state.shape} from {self.state_paths[index]}")
        sample["state"] = torch.from_numpy(state)
        return sample


class StateVideoDatasetSingleRGBOnly(VideoDatasetWithResizing):
    """7DoF targets + standard I2V RGB condition only (no separate pose_image column)."""

    def __init__(self, *args, state_column: str = "state_path", dof_target_length: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dof_target_length = dof_target_length
        self.state_paths = self._load_state_paths(state_column)
        if len(self.state_paths) != len(self.video_paths):
            raise ValueError(
                f"Expected one state path per video but found {len(self.state_paths)} states and {len(self.video_paths)} videos"
            )

    def _load_state_paths(self, state_column: str) -> list[Path]:
        if self.dataset_file is not None:
            df = pd.read_csv(self.dataset_file)
            if state_column not in df.columns:
                raise KeyError(f"Missing state column {state_column!r} in {self.dataset_file}")
            paths = [Path(str(value)) for value in df[state_column].tolist()]
        else:
            state_file = self.data_root / state_column
            if not state_file.is_file():
                state_file = self.data_root / "state_paths.txt"
            if not state_file.is_file():
                raise FileNotFoundError(
                    f"Missing state path manifest under {self.data_root}; expected {state_column} or state_paths.txt"
                )
            paths = [Path(line.strip()) for line in state_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else self.data_root / path for path in paths]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index
        sample = super().__getitem__(index)
        state = np.load(self.state_paths[index]).astype(np.float32)
        if state.ndim != 2 or state.shape[-1] != 7:
            raise ValueError(f"Expected state array with shape [T, 7], got {state.shape} from {self.state_paths[index]}")
        sample["state"] = torch.from_numpy(state)
        return sample


class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool, single_rgb_condition: bool = False) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors
        self.single_rgb_condition = single_rgb_condition

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        batch: Dict[str, Any] = {
            "images": images,
            "videos": videos,
            "prompts": prompts,
        }
        if not self.single_rgb_condition:
            pose_images = [x["pose_image"] for x in data[0]]
            batch["pose_images"] = torch.stack(pose_images).to(dtype=self.weight_dtype, non_blocking=True)
        if "state" in data[0][0]:
            states = [x["state"] for x in data[0]]
            batch["states"] = torch.stack(states).to(dtype=torch.float32, non_blocking=True)
        return batch


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.single_rgb_condition and not args.enable_7dof_head:
        raise ValueError(
            "--single_rgb_condition requires --enable_7dof_head (needs state supervision). "
            "For standard I2V without 7DoF, use cogvideox_image_to_video_lora.py."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    # These changes will also be required when trying to run inference with the trained lora
    if args.ignore_learned_positional_embeddings:
        del transformer.patch_embed.pos_embedding
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    dof_target_length = args.dof_target_length
    feature_layers: list[int] | None = None
    feature_dim: int | None = None
    dof_head: SevenDoFHead | None = None
    feature_hooks: CogVideoXFeatureHooks | None = None
    if args.enable_7dof_head:
        feature_layers = resolve_feature_layers(transformer, args.feature_layers)
        feature_dim = infer_transformer_hidden_size(transformer.config)
        dof_head_hidden_dim = args.dof_head_hidden_dim or feature_dim
        dof_head = SevenDoFHead(feature_dim, dof_head_hidden_dim, dof_target_length * 7)
        dof_head.to(accelerator.device, dtype=torch.float32)
        feature_hooks = CogVideoXFeatureHooks(transformer, feature_layers)
        ablation_note = (
            "; ablation: single RGB+text I2V condition (standard image_latents padding, no dual pose/RGB slots)"
            if args.single_rgb_condition
            else ""
        )
        logger.info(
            f"7DoF head enabled: feature_layers={feature_layers}, feature_dim={feature_dim}, target_length={dof_target_length}{ablation_note}"
        )
    else:
        logger.info(
            f"7DoF head disabled; training CogVideoX LoRA with pose+RGB dual visual conditions "
            f"(pose latent 0, RGB frame {args.rgb_anchor_frame})"
        )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            dof_head_state_to_save = None

            for model in models:
                unwrapped_model = unwrap_model(accelerator, model)
                if isinstance(unwrapped_model, type(unwrap_model(accelerator, transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(unwrapped_model)
                elif args.enable_7dof_head and isinstance(unwrapped_model, SevenDoFHead):
                    dof_head_state_to_save = {
                        key: value.detach().cpu() for key, value in unwrapped_model.state_dict().items()
                    }
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            CogVideoXImageToVideoPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )
            if dof_head_state_to_save is not None:
                torch.save(
                    {
                        "state_dict": dof_head_state_to_save,
                        "feature_layers": feature_layers,
                        "feature_dim": feature_dim,
                        "dof_target_length": dof_target_length,
                        "dof_loss_weight": args.dof_loss_weight,
                    },
                    os.path.join(output_dir, "dof_head.pt"),
                )

    def load_model_hook(models, input_dir):
        transformer_ = None
        dof_head_ = None

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                unwrapped_model = unwrap_model(accelerator, model)

                if isinstance(unwrapped_model, type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrapped_model
                elif args.enable_7dof_head and isinstance(unwrapped_model, SevenDoFHead):
                    dof_head_ = unwrapped_model
                else:
                    raise ValueError(f"Unexpected save model: {unwrapped_model.__class__}")
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        dof_head_path = os.path.join(input_dir, "dof_head.pt")
        if dof_head_ is not None and os.path.isfile(dof_head_path):
            payload = torch.load(dof_head_path, map_location="cpu")
            dof_head_.load_state_dict(payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload)
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    dof_head_parameters = list(dof_head.parameters()) if dof_head is not None else []

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    if dof_head_parameters:
        dof_head_parameters_with_lr = {
            "params": dof_head_parameters,
            "lr": args.dof_head_lr or args.learning_rate,
        }
        params_to_optimize.append(dof_head_parameters_with_lr)
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )

    # Dataset and DataLoader
    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "image_column": args.image_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": args.load_tensors,
        "random_flip": args.random_flip,
        "image_to_video": True,
    }
    if args.video_reshape_mode is None:
        if args.single_rgb_condition:
            dataset_cls = StateVideoDatasetSingleRGBOnly
            dataset_kwargs = {
                "state_column": args.state_column,
                "dof_target_length": args.dof_target_length,
            }
        elif args.enable_7dof_head:
            dataset_cls = StateVideoDatasetWithResizing
            dataset_kwargs = {
                "pose_image_column": args.pose_image_column,
                "state_column": args.state_column,
                "dof_target_length": args.dof_target_length,
            }
        else:
            dataset_cls = PoseRGBVideoDatasetWithResizing
            dataset_kwargs = {"pose_image_column": args.pose_image_column}
        train_dataset = dataset_cls(**dataset_init_kwargs, **dataset_kwargs)
    else:
        raise ValueError("Pose+RGB CogVideo trainer currently supports resizing mode only; unset --video_reshape_mode")

    collate_fn = CollateFunction(weight_dtype, args.load_tensors, single_rgb_condition=args.single_rgb_condition)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=BucketSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )

    # Prepare everything with our `accelerator`.
    if dof_head is not None:
        transformer, dof_head, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, dof_head, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    if args.load_tensors:
        del vae, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if dof_head is not None:
            dof_head.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer] if dof_head is None else [transformer, dof_head]
            logs = {}

            with accelerator.accumulate(models_to_accumulate):
                images = batch["images"].to(accelerator.device, non_blocking=True)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]
                state_targets = (
                    batch["states"].to(accelerator.device, dtype=torch.float32, non_blocking=True)
                    if args.enable_7dof_head
                    else None
                )

                # Encode videos and build conditioning (standard I2V vs dual pose+RGB latents)
                if args.single_rgb_condition:
                    if not args.load_tensors:
                        images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        image_noise_sigma = torch.normal(
                            mean=-3.0, std=0.5, size=(images.size(0),), device=accelerator.device, dtype=weight_dtype
                        )
                        image_noise_sigma = torch.exp(image_noise_sigma)
                        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                        image_latent_dist = vae.encode(noisy_images).latent_dist

                        videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        latent_dist = vae.encode(videos).latent_dist
                    else:
                        image_latent_dist = DiagonalGaussianDistribution(images)
                        latent_dist = DiagonalGaussianDistribution(videos)

                    image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                    image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                    video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
                    latent_padding = image_latents.new_zeros(padding_shape)
                    image_latents = torch.cat([image_latents, latent_padding], dim=1)

                    if random.random() < args.noised_image_dropout:
                        image_latents = torch.zeros_like(image_latents)
                else:
                    pose_images = batch["pose_images"].to(accelerator.device, non_blocking=True)
                    if not args.load_tensors:
                        images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        pose_images = pose_images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        image_noise_sigma = torch.normal(
                            mean=-3.0, std=0.5, size=(images.size(0),), device=accelerator.device, dtype=weight_dtype
                        )
                        image_noise_sigma = torch.exp(image_noise_sigma)
                        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                        noisy_pose_images = pose_images + torch.randn_like(pose_images) * image_noise_sigma[:, None, None, None, None]
                        image_latent_dist = vae.encode(noisy_images).latent_dist
                        pose_latent_dist = vae.encode(noisy_pose_images).latent_dist

                        videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                        latent_dist = vae.encode(videos).latent_dist
                    else:
                        image_latent_dist = DiagonalGaussianDistribution(images)
                        pose_latent_dist = DiagonalGaussianDistribution(pose_images)
                        latent_dist = DiagonalGaussianDistribution(videos)

                    rgb_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                    rgb_latents = rgb_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    rgb_latents = rgb_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    pose_latents = pose_latent_dist.sample() * VAE_SCALING_FACTOR
                    pose_latents = pose_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    pose_latents = pose_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                    video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                    video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    condition_latents = build_pose_rgb_condition_latents(
                        pose_latents,
                        rgb_latents,
                        video_latents,
                        args.rgb_anchor_frame,
                    )

                    if random.random() < args.noised_image_dropout:
                        condition_latents = torch.zeros_like(condition_latents)

                # Encode prompts
                if not args.load_tensors:
                    prompt_embeds = compute_prompt_embeddings(
                        tokenizer,
                        text_encoder,
                        prompts,
                        model_config.max_text_seq_length,
                        accelerator.device,
                        weight_dtype,
                        requires_grad=False,
                    )
                else:
                    prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise that will be added to the latents
                noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                if args.single_rgb_condition:
                    noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
                else:
                    noisy_model_input = torch.cat([noisy_video_latents, condition_latents], dim=2)

                ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
                ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)
                # Predict the noise residual and collect intermediate CogVideoX features for the 7DoF probe.
                if feature_hooks is not None:
                    feature_hooks.clear()
                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                diffusion_loss = torch.mean(
                    (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                    dim=1,
                )
                diffusion_loss = diffusion_loss.mean()
                loss = diffusion_loss
                dof_loss = None
                if args.enable_7dof_head:
                    if feature_hooks is None or dof_head is None or state_targets is None:
                        raise RuntimeError("7DoF head is enabled but features/head/state targets are missing")
                    pooled_features = feature_hooks.pooled_features().float()
                    dof_pred = dof_head(pooled_features).view(batch_size, dof_target_length, 7)
                    state_targets = match_state_length(state_targets, dof_target_length)
                    dof_loss = F.mse_loss(dof_pred.float(), state_targets.float())
                    loss = loss + args.dof_loss_weight * dof_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    trainable_parameters = list(transformer.parameters())
                    if dof_head is not None:
                        trainable_parameters += list(dof_head.parameters())
                    gradient_norm_before_clip = get_gradient_norm(trainable_parameters)
                    accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(trainable_parameters)
                    logs.update(
                        {
                            "gradient_norm_before_clip": gradient_norm_before_clip,
                            "gradient_norm_after_clip": gradient_norm_after_clip,
                        }
                    )

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Validation
                should_run_validation = args.validation_prompt is not None and (
                    args.validation_steps is not None and global_step % args.validation_steps == 0
                )
                if should_run_validation:
                    run_validation(args, accelerator, transformer, scheduler, model_config, weight_dtype)

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs.update(
                {
                    "loss": loss.detach().item(),
                    "diffusion_loss": diffusion_loss.detach().item(),
                    "lr": last_lr,
                }
            )
            if dof_loss is not None:
                logs["dof_loss"] = dof_loss.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            should_run_validation = args.validation_prompt is not None and (
                args.validation_epochs is not None and (epoch + 1) % args.validation_epochs == 0
            )
            if should_run_validation:
                run_validation(args, accelerator, transformer, scheduler, model_config, weight_dtype)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(accelerator, transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        CogVideoXImageToVideoPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        if dof_head is not None:
            dof_head_to_save = unwrap_model(accelerator, dof_head)
            torch.save(
                {
                    "state_dict": {key: value.detach().cpu() for key, value in dof_head_to_save.state_dict().items()},
                    "feature_layers": feature_layers,
                    "feature_dim": feature_dim,
                    "dof_target_length": dof_target_length,
                    "dof_loss_weight": args.dof_loss_weight,
                    "dof_supervision": args.dof_supervision,
                    "supervision_column": args.state_column,
                    "rgb_anchor_frame": args.rgb_anchor_frame,
                    "pose_image_column": args.pose_image_column,
                    "rgb_image_column": args.image_column,
                    "single_rgb_condition": args.single_rgb_condition,
                },
                os.path.join(args.output_dir, "dof_head.pt"),
            )

        # Cleanup trained models to save memory
        if args.load_tensors:
            del transformer
        else:
            del transformer, text_encoder, vae

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

        accelerator.print("===== Memory before testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)

        # Final test inference
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        if args.ignore_learned_positional_embeddings:
            del pipe.transformer.patch_embed.pos_embedding
            pipe.transformer.patch_embed.use_learned_positional_embeddings = False
            pipe.transformer.config.use_learned_positional_embeddings = False
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        # Load LoRA weights
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        # Run inference
        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            validation_images = args.validation_images.split(args.validation_prompt_separator)
            for validation_image, validation_prompt in zip(validation_images, validation_prompts):
                pipeline_args = {
                    "image": load_image(validation_image),
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.max_num_frames,
                }

                video = log_validation(
                    accelerator=accelerator,
                    pipe=pipe,
                    args=args,
                    pipeline_args=pipeline_args,
                    is_final_validation=True,
                )
                validation_outputs.extend(video)

        accelerator.print("===== Memory after testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                videos=validation_outputs,
                base_model=args.pretrained_model_name_or_path,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
                fps=args.fps,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    if feature_hooks is not None:
        feature_hooks.close()
    accelerator.end_training()


def parse_args_with_7dof():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable_7dof_head", action="store_true", help="Enable the auxiliary 7DoF MLP head")
    parser.add_argument("--pose_image_column", default="pose_image", help="CSV column containing pose first-frame images")
    parser.add_argument("--rgb_anchor_frame", type=int, default=24, help="Video frame whose latent slot receives the RGB condition")
    parser.add_argument("--state_column", default="state_path", help="CSV column or line manifest containing BridgeV2 7DoF .npy paths")
    parser.add_argument(
        "--dof_supervision",
        choices=("state", "action"),
        default="state",
        help="7DoF head supervision target: BridgeV2 observation/state or action",
    )
    parser.add_argument("--dof_target_length", type=int, default=13, help="Number of 7DoF targets predicted per sample")
    parser.add_argument("--dof_loss_weight", type=float, default=1.0, help="Weight for the 7DoF MSE loss")
    parser.add_argument("--dof_head_hidden_dim", type=int, default=0, help="Hidden dimension for the 7DoF MLP; defaults to transformer hidden size")
    parser.add_argument("--dof_head_lr", type=float, default=0.0, help="Optional learning rate for the 7DoF head; defaults to LoRA LR")
    parser.add_argument("--feature_layers", default="auto", help="Comma-separated CogVideoX transformer block indices, or auto for 70-80% layers")
    parser.add_argument(
        "--single_rgb_condition",
        action="store_true",
        help="Ablation: only RGB image + text as I2V condition (standard padded image_latents). "
        "Target video can still be pose+RGB composite; requires --enable_7dof_head and state_path/action_path in CSV.",
    )
    custom_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = get_args()
    for key, value in vars(custom_args).items():
        setattr(args, key, value)
    if args.dof_supervision == "action" and args.state_column == "state_path":
        args.state_column = "action_path"
    if args.dof_head_hidden_dim == 0:
        args.dof_head_hidden_dim = None
    if args.dof_head_lr == 0.0:
        args.dof_head_lr = None
    return args


if __name__ == "__main__":
    args = parse_args_with_7dof()
    main(args)
