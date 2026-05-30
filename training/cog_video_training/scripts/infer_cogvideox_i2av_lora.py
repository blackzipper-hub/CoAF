#!/usr/bin/env python3
"""Run CogVideoX I2AV LoRA inference and save predicted state/action."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_timesteps
from diffusers.utils import export_to_video, load_image
from diffusers.utils.torch_utils import randn_tensor

from finetrainers.patches.models.cogvideox.causal_attention import install_temporal_causal_attention
from finetrainers.patches.models.cogvideox.i2av_forward import forward_i2av_transformer
from finetrainers.patches.models.cogvideox.i2av_sequence import expand_rope_for_i2av
from finetrainers.patches.models.cogvideox.state_action import (
    S0Encoder,
    StateActionTokenizer,
    load_state_action_modules,
)


def latest_checkpoint(root: Path) -> Path | None:
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        if path.is_dir():
            try:
                checkpoints.append((int(path.name.split("-")[-1]), path))
            except ValueError:
                continue
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def resolve_lora_dir(path: Path) -> Path:
    if (path / "pytorch_lora_weights.safetensors").is_file() or (path / "pytorch_lora_weights.bin").is_file():
        return path
    ckpt = latest_checkpoint(path)
    if ckpt is not None:
        return ckpt
    raise FileNotFoundError(f"No LoRA weights found in {path} or checkpoint-* children")


def resolve_paths(data_root: Path, values: list[str]) -> list[str]:
    return [str(Path(value) if Path(value).is_absolute() else data_root / value) for value in values]


def load_manifest_paths(data_root: Path, name: str) -> list[str]:
    path = data_root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing I2AV manifest: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return resolve_paths(data_root, values)


def load_validation_items(data_root: Path, max_samples: int) -> list[dict[str, Any]]:
    validation_path = data_root / "validation.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    state_paths = load_manifest_paths(data_root, "state_paths.txt")
    action_paths = load_manifest_paths(data_root, "action_paths.txt")

    items = []
    for row_idx, item in enumerate(data[:max_samples]):
        sample_index = int(item.get("sample_index", row_idx))
        image_path = item.get("image_path") or item.get("image")
        prompt = item.get("caption") or item.get("prompt") or item.get("text")
        if image_path is None or prompt is None:
            raise ValueError(f"validation item missing image/prompt fields: {item}")
        items.append(
            {
                "sample_index": sample_index,
                "image_path": image_path,
                "prompt": prompt,
                "state_path": state_paths[sample_index],
                "action_path": action_paths[sample_index],
            }
        )
    return items


def disable_learned_positional_embeddings(pipe: CogVideoXImageToVideoPipeline) -> None:
    patch_embed = pipe.transformer.patch_embed
    if hasattr(patch_embed, "pos_embedding"):
        del patch_embed.pos_embedding
    patch_embed.use_learned_positional_embeddings = False
    pipe.transformer.config.use_learned_positional_embeddings = False


def get_transformer_hidden_dim(transformer) -> int:
    config = transformer.config
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "num_attention_heads") and hasattr(config, "attention_head_dim"):
        return int(config.num_attention_heads * config.attention_head_dim)
    if hasattr(transformer, "norm_final") and hasattr(transformer.norm_final, "normalized_shape"):
        return int(transformer.norm_final.normalized_shape[0])
    raise ValueError("Cannot infer transformer hidden dimension from CogVideoX config/modules.")


def get_text_embed_dim(transformer) -> int:
    config = transformer.config
    if hasattr(config, "text_embed_dim"):
        return int(config.text_embed_dim)
    text_proj = getattr(transformer.patch_embed, "text_proj", None)
    if text_proj is not None and hasattr(text_proj, "in_features"):
        return int(text_proj.in_features)
    raise ValueError("Cannot infer text embedding dimension from CogVideoX config/modules.")


def prepare_i2av_rotary_emb(
    pipe: CogVideoXImageToVideoPipeline,
    height: int,
    width: int,
    latent_frames: int,
    device: torch.device,
    sa_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not pipe.transformer.config.use_rotary_positional_embeddings:
        return None

    freqs_cos, freqs_sin = pipe._prepare_rotary_positional_embeddings(height, width, latent_frames, device)
    patch = pipe.transformer.config.patch_size
    grid_h = height // (pipe.vae_scale_factor_spatial * patch)
    grid_w = width // (pipe.vae_scale_factor_spatial * patch)
    patches_per_frame = grid_h * grid_w
    return expand_rope_for_i2av(freqs_cos, freqs_sin, latent_frames, patches_per_frame, sa_per_frame)


def prepare_extra_step_kwargs(scheduler, generator: torch.Generator | None, eta: float) -> dict[str, Any]:
    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta
    if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


@torch.no_grad()
def run_i2av_sample(
    pipe: CogVideoXImageToVideoPipeline,
    sa_tokenizer: StateActionTokenizer,
    s0_encoder: S0Encoder,
    norm_stats: dict[str, torch.Tensor],
    item: dict[str, Any],
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    do_cfg = args.guidance_scale > 1.0
    sa_per_frame = sa_tokenizer.num_tokens

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=item["prompt"],
        negative_prompt=None,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=pipe.transformer.config.max_text_seq_length,
        device=device,
        dtype=dtype,
    )

    state_seq = torch.from_numpy(np.load(item["state_path"]).astype(np.float32)).unsqueeze(0).to(device=device)
    mean = norm_stats["mean"].to(device=device, dtype=torch.float32)
    std = norm_stats["std"].to(device=device, dtype=torch.float32)
    s0_norm = ((state_seq[:, 0] - mean) / std).to(dtype=dtype)
    s0_cond = s0_encoder(s0_norm)
    prompt_embeds = torch.cat([prompt_embeds, s0_cond], dim=1)
    if do_cfg:
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, s0_cond], dim=1)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, device, None)
    pipe._guidance_scale = args.guidance_scale
    pipe._current_timestep = None
    pipe._attention_kwargs = None
    pipe._interrupt = False
    pipe._num_timesteps = len(timesteps)

    latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    additional_frames = 0
    num_frames = args.num_frames
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * pipe.vae_scale_factor_temporal

    image = load_image(item["image_path"])
    image = pipe.video_processor.preprocess(image, height=args.height, width=args.width).to(device, dtype=dtype)
    latent_channels = pipe.transformer.config.in_channels // 2
    latents, image_latents = pipe.prepare_latents(
        image,
        1,
        latent_channels,
        num_frames,
        args.height,
        args.width,
        dtype,
        device,
        generator,
        None,
    )

    latent_frames = latents.shape[1]
    grid_h = args.height // (pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size)
    grid_w = args.width // (pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size)
    patches_per_frame = grid_h * grid_w
    sa_tokens = randn_tensor(
        (1, latent_frames, sa_per_frame, sa_tokenizer.hidden_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    sa_tokens = sa_tokens * pipe.scheduler.init_noise_sigma

    image_rotary_emb = prepare_i2av_rotary_emb(pipe, args.height, args.width, latent_frames, device, sa_per_frame)
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)
    extra_step_kwargs = prepare_extra_step_kwargs(pipe.scheduler, generator, eta=0.0)
    old_pred_original_sample = None
    final_sa_pred = None

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        for i, t in enumerate(timesteps):
            pipe._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_image_input = torch.cat([image_latents] * 2) if do_cfg else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
            sa_model_input = torch.cat([sa_tokens] * 2) if do_cfg else sa_tokens
            timestep = t.expand(latent_model_input.shape[0])

            with pipe.transformer.cache_context("cond_uncond"):
                noise_pred, sa_pred = forward_i2av_transformer(
                    pipe.transformer,
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    noisy_sa_tokens=sa_model_input,
                    timestep=timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=None,
                    patches_per_frame=patches_per_frame,
                    sa_per_frame=sa_per_frame,
                    return_dict=False,
                )
            noise_pred = noise_pred.float()
            sa_pred = sa_pred.float()

            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)
                sa_uncond, sa_text = sa_pred.chunk(2)
                sa_pred = sa_uncond + args.guidance_scale * (sa_text - sa_uncond)

            if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = pipe.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(dtype)
            final_sa_pred = sa_pred.to(dtype)
            sa_tokens = final_sa_pred

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe._current_timestep = None
    if additional_frames:
        latents = latents[:, additional_frames:]

    video = pipe.decode_latents(latents)
    frames = pipe.video_processor.postprocess_video(video=video, output_type="np")[0]

    if final_sa_pred is None:
        raise RuntimeError("I2AV denoising produced no state/action prediction.")
    pred_state_norm, pred_action_norm = sa_tokenizer.decode(final_sa_pred)
    pred_state = pred_state_norm.float() * std + mean
    pred_action = pred_action_norm.float() * std
    return pred_state.squeeze(0).cpu().numpy(), pred_action.squeeze(0).cpu().numpy(), frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--lora_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--state_norm_stats", required=True, type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lora_dir = resolve_lora_dir(args.lora_dir)
    state_action_path = lora_dir / "state_action.pt"
    if not state_action_path.is_file():
        raise FileNotFoundError(f"I2AV checkpoint is missing state_action.pt under {lora_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    disable_learned_positional_embeddings(pipe)
    pipe.load_lora_weights(str(lora_dir), adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [1.0])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    hidden_dim = get_transformer_hidden_dim(pipe.transformer)
    text_embed_dim = get_text_embed_dim(pipe.transformer)
    sa_tokenizer = StateActionTokenizer(hidden_dim=hidden_dim, num_state_tokens=4, num_action_tokens=4)
    s0_encoder = S0Encoder(hidden_dim=text_embed_dim, num_tokens=4)
    load_state_action_modules(str(state_action_path), sa_tokenizer, s0_encoder, device=pipe._execution_device)
    sa_tokenizer.to(device=pipe._execution_device, dtype=torch.bfloat16).eval()
    s0_encoder.to(device=pipe._execution_device, dtype=torch.bfloat16).eval()

    install_temporal_causal_attention(
        pipe.transformer,
        num_pixel_frames=args.num_frames,
        pixel_height=args.height,
        pixel_width=args.width,
        text_seq_length=pipe.transformer.config.max_text_seq_length + s0_encoder.num_tokens,
        vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        device=pipe._execution_device,
        dtype=torch.float32,
        enable_state_action=True,
        sa_per_frame=sa_tokenizer.num_tokens,
        s0_cond_tokens=s0_encoder.num_tokens,
    )

    norm_stats = torch.load(args.state_norm_stats, map_location="cpu")
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)
    for idx, item in enumerate(load_validation_items(args.data_root, args.num_samples)):
        pred_state, pred_action, video = run_i2av_sample(pipe, sa_tokenizer, s0_encoder, norm_stats, item, args, generator)

        stem = f"sample_{idx:03d}"
        video_path = args.output_dir / f"{stem}.mp4"
        pred_state_path = args.output_dir / f"{stem}_pred_state.npy"
        pred_action_path = args.output_dir / f"{stem}_pred_action.npy"
        np.save(pred_state_path, pred_state)
        np.save(pred_action_path, pred_action)
        export_to_video(video, str(video_path), fps=args.fps)

        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "prompt": item["prompt"],
                    "image_path": item["image_path"],
                    "state_path": item["state_path"],
                    "action_path": item["action_path"],
                    "lora_dir": str(lora_dir),
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "pred_state_path": str(pred_state_path),
                    "pred_action_path": str(pred_action_path),
                    "pred_state": pred_state.tolist(),
                    "pred_action": pred_action.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {video_path}")
        print(f"Wrote {pred_action_path}")


if __name__ == "__main__":
    main()
