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

_FINETRAINERS_ROOT = Path(__file__).resolve().parents[4]
if str(_FINETRAINERS_ROOT) not in sys.path:
    sys.path.insert(0, str(_FINETRAINERS_ROOT))

import diffusers
import torch
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
from finetrainers.patches.models.cogvideox.causal_attention import (
    install_temporal_causal_attention,
    write_causal_attention_metadata,
)
from finetrainers.patches.models.cogvideox.i2av_forward import forward_i2av_transformer, forward_i2av_v5_transformer
from finetrainers.patches.models.cogvideox.i2av_layout import compute_i2av_v5_layout
from finetrainers.patches.models.cogvideox.state_action import (
    ChunkedStateActionTokenizer,
    S0Encoder,
    StateActionTokenizer,
    compute_sa_denoise_loss,
    compute_sa_raw_action_loss,
    compute_sa_loss,
    load_state_action_modules,
    prepare_gt,
    prepare_gt_chunked,
    prepare_raw_action_gt_chunked,
    save_state_action_modules,
)
from dataset_i2av import I2AVCollateFunction, I2AVVideoDataset  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_i2av_rotary_positional_embeddings,
    prepare_i2av_v5_rotary_positional_embeddings,
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


class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype, load_tensors: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data[0]]

        if self.load_tensors:
            prompts = torch.stack(prompts).to(dtype=self.weight_dtype, non_blocking=True)

        images = [x["image"] for x in data[0]]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data[0]]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        return {
            "images": images,
            "videos": videos,
            "prompts": prompts,
        }


class I2AVAuxModules(torch.nn.Module):
    def __init__(self, sa_tokenizer: StateActionTokenizer, s0_encoder: S0Encoder) -> None:
        super().__init__()
        self.sa_tokenizer = sa_tokenizer
        self.s0_encoder = s0_encoder


def get_transformer_hidden_dim(transformer: CogVideoXTransformer3DModel) -> int:
    config = transformer.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "num_attention_heads") and hasattr(config, "attention_head_dim"):
        return int(config.num_attention_heads * config.attention_head_dim)
    if hasattr(transformer, "norm_final") and hasattr(transformer.norm_final, "normalized_shape"):
        return int(transformer.norm_final.normalized_shape[0])
    raise ValueError("Cannot infer transformer hidden dimension from CogVideoX config/modules.")


def get_text_embed_dim(transformer: CogVideoXTransformer3DModel) -> int:
    config = transformer.config
    if hasattr(config, "text_embed_dim"):
        return int(config.text_embed_dim)
    text_proj = getattr(transformer.patch_embed, "text_proj", None)
    if text_proj is not None and hasattr(text_proj, "in_features"):
        return int(text_proj.in_features)
    raise ValueError("Cannot infer text embedding dimension from CogVideoX config/modules.")


def main(args):
    if not args.enable_i2av:
        raise ValueError("cogvideox_image_to_video_lora_i2av.py requires --enable_i2av")
    if not args.state_norm_stats:
        raise ValueError("I2AV training requires --state_norm_stats")
    if not args.temporal_causal_attention:
        raise ValueError("I2AV training requires --temporal_causal_attention")
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
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

    hidden_dim = get_transformer_hidden_dim(transformer)
    text_embed_dim = get_text_embed_dim(transformer)
    v5_layout = None
    if args.i2av_layout == "v5":
        v5_layout = compute_i2av_v5_layout(
            transformer.config,
            pixel_height=args.height,
            pixel_width=args.width,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
            text_seq_length=transformer.config.max_text_seq_length,
            s0_cond_tokens=args.s0_cond_tokens,
            vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
        )
        sa_tokenizer = ChunkedStateActionTokenizer(
            hidden_dim=hidden_dim,
            steps_per_chunk=v5_layout.steps_per_chunk,
            first_chunk_pad_steps=v5_layout.first_chunk_pad_steps,
            real_trajectory_steps=v5_layout.real_trajectory_steps,
        )
    else:
        sa_tokenizer = StateActionTokenizer(hidden_dim=hidden_dim, num_state_tokens=4, num_action_tokens=4)
    s0_encoder = S0Encoder(hidden_dim=text_embed_dim, num_tokens=args.s0_cond_tokens)
    i2av_aux = I2AVAuxModules(sa_tokenizer, s0_encoder)
    norm_stats = torch.load(args.state_norm_stats, map_location="cpu")
    action_norm_stats = torch.load(args.action_norm_stats, map_location="cpu") if args.action_norm_stats else None

    text_seq_length = transformer.config.max_text_seq_length + args.s0_cond_tokens
    if args.temporal_causal_attention:
        causal_meta = install_temporal_causal_attention(
            transformer,
            num_pixel_frames=args.max_num_frames,
            pixel_height=args.height,
            pixel_width=args.width,
            text_seq_length=text_seq_length,
            vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
            device=accelerator.device,
            dtype=torch.float32,
            enable_state_action=True,
            sa_per_frame=args.sa_per_frame,
            s0_cond_tokens=args.s0_cond_tokens,
            i2av_layout=args.i2av_layout,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
        )
        if accelerator.is_main_process:
            write_causal_attention_metadata(args.output_dir, causal_meta)
            logger.info("Enabled temporal causal attention: %s", causal_meta)

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
    i2av_aux.to(accelerator.device, dtype=weight_dtype)
    sa_tokenizer.requires_grad_(True)
    s0_encoder.requires_grad_(True)

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

    if args.train_stage == "stage1":
        i2av_aux.requires_grad_(False)
    elif args.train_stage == "stage2":
        if not args.stage2_train_transformer_lora:
            transformer.requires_grad_(False)
        i2av_aux.requires_grad_(True)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                unwrapped = unwrap_model(accelerator, model)
                if isinstance(unwrapped, type(unwrap_model(accelerator, transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(unwrapped)
                elif isinstance(unwrapped, I2AVAuxModules):
                    save_state_action_modules(
                        os.path.join(output_dir, "state_action.pt"),
                        unwrapped.sa_tokenizer,
                        unwrapped.s0_encoder,
                        tokenizer_type=args.i2av_layout,
                        steps_per_chunk=getattr(unwrapped.sa_tokenizer, "steps_per_chunk", None),
                    )
                else:
                    raise ValueError(f"Unexpected save model: {unwrapped.__class__}")

                if weights:
                    weights.pop()

            CogVideoXImageToVideoPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        i2av_aux_ = None

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                unwrapped = unwrap_model(accelerator, model)
                if isinstance(unwrapped, type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrapped
                elif isinstance(unwrapped, I2AVAuxModules):
                    i2av_aux_ = unwrapped
                else:
                    raise ValueError(f"Unexpected save model: {unwrapped.__class__}")
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)
            if args.temporal_causal_attention:
                install_temporal_causal_attention(
                    transformer_,
                    num_pixel_frames=args.max_num_frames,
                    pixel_height=args.height,
                    pixel_width=args.width,
                    vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                    text_seq_length=text_seq_length,
                    enable_state_action=True,
                    sa_per_frame=args.sa_per_frame,
                    s0_cond_tokens=args.s0_cond_tokens,
                    i2av_layout=args.i2av_layout,
                    pose_pixel_frames=args.pose_pixel_frames,
                    rgb_pixel_frames=args.rgb_pixel_frames,
                )

        state_action_path = os.path.join(input_dir, "state_action.pt")
        if i2av_aux_ is not None and os.path.isfile(state_action_path):
            load_state_action_modules(
                state_action_path,
                i2av_aux_.sa_tokenizer,
                i2av_aux_.s0_encoder,
                device=accelerator.device,
            )

        lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
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
    i2av_parameters = list(filter(lambda p: p.requires_grad, i2av_aux.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    if i2av_parameters:
        params_to_optimize.append({"params": i2av_parameters, "lr": args.learning_rate})
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
        "relayout_v5": args.relayout_v5,
        "v5_reason_frames": args.pose_pixel_frames,
        "v5_rgb_frames": args.rgb_pixel_frames,
    }
    if args.video_reshape_mode is None:
        train_dataset = I2AVVideoDataset(**dataset_init_kwargs)
    else:
        raise ValueError("I2AV training does not support video_reshape_mode yet")

    collate_fn = I2AVCollateFunction(weight_dtype, args.load_tensors)

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
    transformer, i2av_aux, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, i2av_aux, optimizer, train_dataloader, lr_scheduler
    )
    i2av_aux_unwrapped = unwrap_model(accelerator, i2av_aux)
    sa_tokenizer = i2av_aux_unwrapped.sa_tokenizer
    s0_encoder = i2av_aux_unwrapped.s0_encoder

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
            checkpoint_path = os.path.join(args.output_dir, path)
            try:
                accelerator.load_state(checkpoint_path)
            except ValueError as exc:
                if "different number of parameter groups" not in str(exc):
                    raise
                accelerator.print(
                    "Checkpoint model weights were loaded, but optimizer state is incompatible "
                    "with the current trainable parameter groups. Resetting optimizer/scheduler "
                    "state and continuing from the checkpoint step."
                )
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            if lr_scheduler is not None:
                for _ in range(global_step):
                    lr_scheduler.step()

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
        i2av_aux.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, i2av_aux]
            logs = {}

            with accelerator.accumulate(models_to_accumulate):
                images = batch["images"].to(accelerator.device, non_blocking=True)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                states = batch["state"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]

                # Encode videos
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

                noise = torch.randn_like(video_latents)
                batch_size, num_frames, num_channels, height, width = video_latents.shape
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                state_delta_gt = None
                if args.i2av_layout == "v5" and action_norm_stats is not None:
                    if v5_layout is None:
                        raise RuntimeError("v5 layout was not initialized.")
                    state_gt, action_gt, state_delta_gt, s0_norm, _ = prepare_raw_action_gt_chunked(
                        states,
                        batch["action"].to(device=accelerator.device),
                        norm_stats,
                        action_norm_stats,
                        pose_pixel_frames=v5_layout.pose_pixel_frames,
                        steps_per_chunk=v5_layout.steps_per_chunk,
                        gripper_continuous=args.gripper_continuous_action,
                    )
                elif args.i2av_layout == "v5":
                    if v5_layout is None:
                        raise RuntimeError("v5 layout was not initialized.")
                    state_gt, action_gt, s0_norm, _ = prepare_gt_chunked(
                        states,
                        norm_stats,
                        pose_pixel_frames=v5_layout.pose_pixel_frames,
                        steps_per_chunk=v5_layout.steps_per_chunk,
                    )
                else:
                    state_gt, action_gt, s0_norm = prepare_gt(states, norm_stats, num_latent_frames=num_frames)
                s0_cond = s0_encoder(s0_norm.to(dtype=weight_dtype))
                prompt_embeds = torch.cat([prompt_embeds, s0_cond], dim=1)

                clean_sa = sa_tokenizer.encode(
                    state_gt.to(dtype=weight_dtype),
                    action_gt.to(dtype=weight_dtype),
                )
                noise_sa = torch.randn_like(clean_sa)
                if args.train_stage == "stage1":
                    noisy_sa = clean_sa
                else:
                    noisy_sa = scheduler.add_noise(clean_sa, noise_sa, timesteps)

                patch_size = model_config.patch_size
                grid_h = height * VAE_SCALE_FACTOR_SPATIAL // (VAE_SCALE_FACTOR_SPATIAL * patch_size)
                grid_w = width * VAE_SCALE_FACTOR_SPATIAL // (VAE_SCALE_FACTOR_SPATIAL * patch_size)
                patches_per_frame = grid_h * grid_w

                # Prepare rotary embeds
                if model_config.use_rotary_positional_embeddings and args.i2av_layout == "v5":
                    image_rotary_emb = prepare_i2av_v5_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        layout=v5_layout,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    )
                else:
                    image_rotary_emb = (
                        prepare_i2av_rotary_positional_embeddings(
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
                        sa_per_frame=args.sa_per_frame,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                    )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.train_stage == "stage2":
                    noisy_video_latents = video_latents
                else:
                    noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)

                ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
                ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)
                if args.i2av_layout == "v5":
                    model_output, sa_pred = forward_i2av_v5_transformer(
                        unwrap_model(accelerator, transformer),
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        noisy_chunk_tokens=noisy_sa,
                        timestep=timesteps,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        layout=v5_layout,
                        return_dict=False,
                    )
                else:
                    model_output, sa_pred = forward_i2av_transformer(
                        unwrap_model(accelerator, transformer),
                        hidden_states=noisy_model_input,
                        encoder_hidden_states=prompt_embeds,
                        noisy_sa_tokens=noisy_sa,
                        timestep=timesteps,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        patches_per_frame=patches_per_frame,
                        sa_per_frame=args.sa_per_frame,
                        return_dict=False,
                    )

                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                if args.train_stage == "stage2":
                    loss_video = model_pred.new_zeros(())
                else:
                    loss_video = torch.mean(
                        (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                        dim=1,
                    ).mean()

                if args.sa_denoise_loss:
                    if args.train_stage != "stage2":
                        raise RuntimeError("--sa_denoise_loss is currently intended for train_stage=stage2.")
                    sa_loss_dict = compute_sa_denoise_loss(
                        sa_pred,
                        clean_sa.to(dtype=weight_dtype),
                        noise_sa.to(dtype=weight_dtype),
                        sa_tokenizer,
                        lambda_s=args.lambda_s,
                        lambda_a=args.lambda_a,
                    )
                elif action_norm_stats is not None:
                    if state_delta_gt is None:
                        raise RuntimeError("--action_norm_stats raw-action loss currently requires i2av_layout=v5.")
                    sa_loss_dict = compute_sa_raw_action_loss(
                        sa_pred,
                        sa_tokenizer,
                        state_gt.to(dtype=weight_dtype),
                        action_gt.to(dtype=weight_dtype),
                        state_delta_gt.to(dtype=weight_dtype),
                        norm_stats,
                        action_norm_stats,
                        lambda_s=args.lambda_s,
                        lambda_a=args.lambda_a,
                        lambda_g=args.lambda_g,
                        lambda_c=args.lambda_c,
                        gripper_continuous=args.gripper_continuous_action,
                    )
                else:
                    sa_loss_dict = compute_sa_loss(
                        sa_pred,
                        sa_tokenizer,
                        state_gt.to(dtype=weight_dtype),
                        action_gt.to(dtype=weight_dtype),
                        lambda_s=args.lambda_s,
                        lambda_a=args.lambda_a,
                        lambda_c=args.lambda_c,
                    )
                loss_video_balanced = None
                loss_sa_balanced = None
                if args.train_stage == "stage1":
                    loss = loss_video
                elif args.train_stage == "stage2":
                    loss = sa_loss_dict["L_sa"]
                elif action_norm_stats is not None:
                    loss_video_balanced = loss_video / loss_video.detach().clamp_min(1e-6)
                    loss_sa_balanced = sa_loss_dict["L_sa"] / sa_loss_dict["L_sa"].detach().clamp_min(1e-6)
                    loss = loss_video_balanced + args.lambda_sa * loss_sa_balanced
                else:
                    loss = loss_video + args.lambda_sa * sa_loss_dict["L_sa"]
                accelerator.backward(loss)

                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    gradient_norm_before_clip = get_gradient_norm(
                        list(transformer.parameters()) + list(i2av_aux.parameters())
                    )
                    accelerator.clip_grad_norm_(
                        list(transformer.parameters()) + list(i2av_aux.parameters()),
                        args.max_grad_norm,
                    )
                    gradient_norm_after_clip = get_gradient_norm(
                        list(transformer.parameters()) + list(i2av_aux.parameters())
                    )
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
                should_checkpoint = (
                    args.checkpointing_steps is not None
                    and args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                )
                if should_checkpoint:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
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
                    accelerator.wait_for_everyone()

                # Validation
                should_run_validation = args.validation_prompt is not None and (
                    args.validation_steps is not None
                    and args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                )
                if should_run_validation:
                    if accelerator.is_main_process:
                        run_validation(args, accelerator, transformer, scheduler, model_config, weight_dtype)
                    accelerator.wait_for_everyone()

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs.update(
                {
                    "loss": loss.detach().item(),
                    "loss_video": loss_video.detach().item(),
                    "L_state": sa_loss_dict["L_state"].detach().item(),
                    "L_action": sa_loss_dict["L_action"].detach().item(),
                    "L_consistency": sa_loss_dict["L_consistency"].detach().item(),
                    "L_sa": sa_loss_dict["L_sa"].detach().item(),
                    "lr": last_lr,
                }
            )
            if "L_gripper" in sa_loss_dict:
                logs["L_gripper"] = sa_loss_dict["L_gripper"].detach().item()
            if "L_delta_gt" in sa_loss_dict:
                logs["L_delta_gt"] = sa_loss_dict["L_delta_gt"].detach().item()
            if "L_sa_denoise" in sa_loss_dict:
                logs["L_sa_denoise"] = sa_loss_dict["L_sa_denoise"].detach().item()
            if loss_video_balanced is not None:
                logs["loss_video_balanced"] = loss_video_balanced.detach().item()
            if loss_sa_balanced is not None:
                logs["loss_sa_balanced"] = loss_sa_balanced.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        should_run_epoch_validation = args.validation_prompt is not None and (
            args.validation_epochs is not None
            and args.validation_epochs > 0
            and (epoch + 1) % args.validation_epochs == 0
        )
        if should_run_epoch_validation:
            if accelerator.is_main_process:
                run_validation(args, accelerator, transformer, scheduler, model_config, weight_dtype)
            accelerator.wait_for_everyone()

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
        i2av_aux_final = unwrap_model(accelerator, i2av_aux)
        save_state_action_modules(
            os.path.join(args.output_dir, "state_action.pt"),
            i2av_aux_final.sa_tokenizer,
            i2av_aux_final.s0_encoder,
            tokenizer_type=args.i2av_layout,
            steps_per_chunk=getattr(i2av_aux_final.sa_tokenizer, "steps_per_chunk", None),
        )
        if args.temporal_causal_attention and hasattr(transformer, "_coaf_causal_meta"):
            write_causal_attention_metadata(args.output_dir, transformer._coaf_causal_meta)

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

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
