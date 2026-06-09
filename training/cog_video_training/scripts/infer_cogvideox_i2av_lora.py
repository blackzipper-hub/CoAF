#!/usr/bin/env python3
"""Run CogVideoX I2AV LoRA inference and save predicted state/action."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import retrieve_timesteps
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.utils.torch_utils import randn_tensor

from finetrainers.patches.models.cogvideox.causal_attention import install_temporal_causal_attention
from finetrainers.patches.models.cogvideox.i2av_forward import forward_i2av_transformer, forward_i2av_v5_transformer
from finetrainers.patches.models.cogvideox.i2av_layout import compute_i2av_v5_layout
from finetrainers.patches.models.cogvideox.i2av_sequence import expand_rope_for_i2av
from finetrainers.patches.models.cogvideox.state_action import (
    ChunkedStateActionTokenizer,
    S0Encoder,
    StateActionTokenizer,
    get_action_norm_method,
    load_state_action_modules,
    prepare_gt,
    prepare_gt_chunked,
    prepare_raw_action_gt_chunked,
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


def first_rgb_frame(episode_dir: Path) -> Path:
    frames = sorted((episode_dir / "rgb").glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"Missing RGB frames under {episode_dir / 'rgb'}")
    return frames[0]


def load_test_items(data_root: Path, max_samples: int) -> list[dict[str, Any]]:
    metadata_path = data_root / "splits" / "test_1k_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing test metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    items = []
    for row_idx, item in enumerate(metadata[:max_samples]):
        dataset_idx = int(item.get("dataset_idx", row_idx))
        episode_dir = data_root / "raw" / f"episode_{dataset_idx:06d}"
        instruction_path = episode_dir / "instruction" / "instruction.txt"
        prompt = item.get("instruction")
        if prompt is None and instruction_path.is_file():
            prompt = instruction_path.read_text(encoding="utf-8").strip()
        if prompt is None:
            raise ValueError(f"test item missing instruction: {item}")

        state_path = episode_dir / "state" / "state.npy"
        action_path = episode_dir / "action" / "action.npy"
        video_path = episode_dir / "video.mp4"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing test state file: {state_path}")
        if not action_path.is_file():
            raise FileNotFoundError(f"Missing test action file: {action_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing test video file: {video_path}")

        items.append(
            {
                "sample_index": dataset_idx,
                "episode_idx": item.get("episode_idx", item.get("original_episode_idx")),
                "image_path": str(first_rgb_frame(episode_dir)),
                "prompt": prompt,
                "state_path": str(state_path),
                "action_path": str(action_path),
                "video_path": str(video_path),
            }
        )
    return items


def load_validation_items(data_root: Path, max_samples: int) -> list[dict[str, Any]]:
    validation_path = data_root / "validation.json"
    if not validation_path.is_file():
        return load_test_items(data_root, max_samples)

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    video_paths = load_manifest_paths(data_root, "videos.txt")
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
                "video_path": video_paths[sample_index],
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


def prepare_i2av_v5_rotary_emb(
    pipe: CogVideoXImageToVideoPipeline,
    height: int,
    width: int,
    layout,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not pipe.transformer.config.use_rotary_positional_embeddings:
        return None
    rope_frames = 2 * layout.num_pose_latent_frames + layout.num_rgb_latent_frames
    freqs_cos, freqs_sin = pipe._prepare_rotary_positional_embeddings(height, width, rope_frames, device)
    from finetrainers.patches.models.cogvideox.i2av_sequence import expand_rope_for_chunked_i2av_timeids

    return expand_rope_for_chunked_i2av_timeids(
        freqs_cos,
        freqs_sin,
        layout.num_pose_latent_frames,
        layout.num_rgb_latent_frames,
        layout.patches_per_frame,
        layout.chunk_token_count,
    )


def prepare_extra_step_kwargs(scheduler, generator: torch.Generator | None, eta: float) -> dict[str, Any]:
    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta
    if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def load_ground_truth_video_frames(item: dict[str, Any], num_frames: int) -> list[Any]:
    video_path = item.get("video_path")
    if video_path is None:
        raise ValueError("Stage2 inference requires item['video_path'] for clean video injection.")
    frames = list(load_video(str(video_path)))
    if not frames:
        raise ValueError(f"Ground-truth video has no frames: {video_path}")
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames[:num_frames]


def encode_clean_video_latents(
    pipe: CogVideoXImageToVideoPipeline,
    frames: list[Any],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    video = pipe.video_processor.preprocess_video(frames, height=height, width=width)
    video = video.to(device=device, dtype=dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    latents = latent_dist.sample() * pipe.vae.config.scaling_factor
    latents = latents.permute(0, 2, 1, 3, 4)
    return latents.to(memory_format=torch.contiguous_format, dtype=dtype)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_eval_sample(
    *,
    split_root: Path,
    episode_name: str,
    item: dict[str, Any],
    pred_state: np.ndarray,
    pred_action: np.ndarray,
    pred_video: list[Any],
    fps: int,
    lora_dir: Path,
    infer_stage: str,
    action_has_gripper_prob: bool,
) -> None:
    gt_dir = split_root / "gt" / episode_name
    pred_dir = split_root / "pred" / episode_name
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(item["state_path"], gt_dir / "state.npy")
    shutil.copy2(item["action_path"], gt_dir / "action.npy")
    if item.get("video_path"):
        shutil.copy2(item["video_path"], gt_dir / "video.mp4")
    else:
        export_to_video(pred_video, str(gt_dir / "video.mp4"), fps=fps)
    write_text(gt_dir / "prompt.txt", item["prompt"])

    np.save(pred_dir / "state.npy", pred_state)
    np.save(pred_dir / "action.npy", pred_action)
    if action_has_gripper_prob:
        np.save(pred_dir / "action_gripper_binary.npy", (pred_action[..., 6] >= 0.5).astype(np.float32))
    export_to_video(pred_video, str(pred_dir / "video.mp4"), fps=fps)
    if (gt_dir / "video.mp4").is_file():
        shutil.copy2(gt_dir / "video.mp4", pred_dir / "gt_video.mp4")
    write_text(pred_dir / "prompt.txt", item["prompt"])
    metadata = {
        "episode": episode_name,
        "prompt": item["prompt"],
        "image_path": item["image_path"],
        "video_path": item.get("video_path"),
        "state_path": item["state_path"],
        "action_path": item["action_path"],
        "lora_dir": str(lora_dir),
        "infer_stage": infer_stage,
        "action_has_gripper_prob": action_has_gripper_prob,
        "pred_state_path": str(pred_dir / "state.npy"),
        "pred_action_path": str(pred_dir / "action.npy"),
    }
    write_text(pred_dir / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))


def run_eval_split(
    *,
    split_name: str,
    items: list[dict[str, Any]],
    eval_root: Path,
    pipe: CogVideoXImageToVideoPipeline,
    sa_tokenizer: StateActionTokenizer,
    s0_encoder: S0Encoder,
    norm_stats: dict[str, torch.Tensor],
    args: argparse.Namespace,
    generator: torch.Generator,
    lora_dir: Path,
) -> None:
    split_root = eval_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, item in enumerate(items, start=1):
        episode_name = f"episode_{idx:04d}"
        pred_state, pred_action, video = run_i2av_sample(
            pipe, sa_tokenizer, s0_encoder, norm_stats, item, args, generator
        )
        save_eval_sample(
            split_root=split_root,
            episode_name=episode_name,
            item=item,
            pred_state=pred_state,
            pred_action=pred_action,
            pred_video=video,
            fps=args.fps,
            lora_dir=lora_dir,
            infer_stage=args.infer_stage,
        action_has_gripper_prob=args.action_norm_stats_payload is not None and not args.gripper_continuous_action,
        )
        manifest.append(
            {
                "episode": episode_name,
                "prompt": item["prompt"],
                "image_path": item["image_path"],
                "video_path": item.get("video_path"),
                "state_path": item["state_path"],
                "action_path": item["action_path"],
            }
        )
        print(f"Wrote {split_name}/{episode_name}")
    write_text(split_root / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))


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
    layout = None
    if args.i2av_layout == "v5":
        layout = compute_i2av_v5_layout(
            pipe.transformer.config,
            pixel_height=args.height,
            pixel_width=args.width,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
            text_seq_length=pipe.transformer.config.max_text_seq_length,
            s0_cond_tokens=s0_encoder.num_tokens,
            vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        )
        sa_per_frame = layout.chunk_token_count
    else:
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
    gt_video_frames = None
    if args.infer_stage == "stage2":
        gt_video_frames = load_ground_truth_video_frames(item, num_frames)
        latents = encode_clean_video_latents(
            pipe,
            gt_video_frames,
            height=args.height,
            width=args.width,
            device=device,
            dtype=dtype,
        )

    latent_frames = latents.shape[1]
    if layout is not None and latent_frames != layout.num_latent_frames:
        raise ValueError(
            f"Stage {args.infer_stage} produced {latent_frames} latent frames, "
            f"but v5 layout expects {layout.num_latent_frames}. "
            f"Check GT video frame count and MAX_NUM_FRAMES."
        )
    grid_h = args.height // (pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size)
    grid_w = args.width // (pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size)
    patches_per_frame = grid_h * grid_w
    action_norm_stats = getattr(args, "action_norm_stats_payload", None)
    if args.infer_stage == "stage1":
        if args.i2av_layout == "v5" and action_norm_stats is not None:
            action_seq = torch.from_numpy(np.load(item["action_path"]).astype(np.float32)).unsqueeze(0).to(device=device)
            state_gt, action_gt, _, _, _ = prepare_raw_action_gt_chunked(
                state_seq,
                action_seq,
                norm_stats,
                action_norm_stats,
                pose_pixel_frames=args.pose_pixel_frames,
                steps_per_chunk=layout.steps_per_chunk,
                gripper_continuous=args.gripper_continuous_action,
            )
        elif args.i2av_layout == "v5":
            state_gt, action_gt, _, _ = prepare_gt_chunked(
                state_seq,
                norm_stats,
                pose_pixel_frames=args.pose_pixel_frames,
                steps_per_chunk=layout.steps_per_chunk,
            )
        else:
            state_gt, action_gt, _ = prepare_gt(state_seq, norm_stats, num_latent_frames=latent_frames)
        sa_tokens = sa_tokenizer.encode(state_gt.to(dtype=dtype), action_gt.to(dtype=dtype))
    else:
        sa_frames = layout.num_pose_latent_frames if layout is not None else latent_frames
        sa_tokens = randn_tensor(
            (1, sa_frames, sa_per_frame, sa_tokenizer.hidden_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        sa_tokens = sa_tokens * pipe.scheduler.init_noise_sigma

    if args.i2av_layout == "v5":
        image_rotary_emb = prepare_i2av_v5_rotary_emb(pipe, args.height, args.width, layout, device)
    else:
        image_rotary_emb = prepare_i2av_rotary_emb(pipe, args.height, args.width, latent_frames, device, sa_per_frame)
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)
    extra_step_kwargs = prepare_extra_step_kwargs(pipe.scheduler, generator, eta=0.0)
    old_pred_original_sample = None
    old_sa_pred_original_sample = None
    final_sa_pred = None

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)
        for i, t in enumerate(timesteps):
            pipe._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            if args.infer_stage != "stage2":
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_image_input = torch.cat([image_latents] * 2) if do_cfg else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
            sa_model_input = torch.cat([sa_tokens] * 2) if do_cfg else sa_tokens
            timestep = t.expand(latent_model_input.shape[0])

            with pipe.transformer.cache_context("cond_uncond"):
                if args.i2av_layout == "v5":
                    noise_pred, sa_pred = forward_i2av_v5_transformer(
                        pipe.transformer,
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        noisy_chunk_tokens=sa_model_input,
                        timestep=timestep,
                        ofs=ofs_emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=None,
                        layout=layout,
                        return_dict=False,
                    )
                else:
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

            if args.infer_stage != "stage2":
                scheduler_noise_pred = noise_pred.to(device=latents.device)
                if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                    latents = pipe.scheduler.step(scheduler_noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = pipe.scheduler.step(
                        scheduler_noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(dtype)
            if args.infer_stage == "stage1":
                final_sa_pred = sa_tokens
            elif args.sa_denoise_loss:
                sa_noise_pred = sa_pred.to(device=sa_tokens.device)
                if not isinstance(pipe.scheduler, CogVideoXDPMScheduler):
                    sa_tokens = pipe.scheduler.step(
                        sa_noise_pred,
                        t,
                        sa_tokens.float(),
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    sa_tokens, old_sa_pred_original_sample = pipe.scheduler.step(
                        sa_noise_pred,
                        old_sa_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        sa_tokens.float(),
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                sa_tokens = sa_tokens.to(dtype)
                final_sa_pred = sa_tokens
            else:
                # SA/chunk tokens are trained with direct decoded state/action loss,
                # not a scheduler velocity/noise loss. Feed the predicted clean token
                # estimate into the next denoising step.
                sa_tokens = sa_pred.to(dtype)
                final_sa_pred = sa_tokens

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
    action_norm_stats = getattr(args, "action_norm_stats_payload", None)
    if action_norm_stats is not None:
        pred_action = torch.empty_like(pred_action_norm.float())
        action_norm_method = get_action_norm_method(action_norm_stats)
        if action_norm_method == "quantile":
            action_q01 = action_norm_stats["q01"].to(device=device, dtype=torch.float32)
            action_q99 = action_norm_stats["q99"].to(device=device, dtype=torch.float32)
            action_span = (action_q99 - action_q01).clamp_min(1e-6)
            if args.gripper_continuous_action:
                pred_action = (pred_action_norm.float() + 1.0) * 0.5 * action_span + action_q01
            else:
                pred_action[..., :6] = pred_action_norm.float()[..., :6] * 0.5 * action_span[:6] + (
                    action_q01[:6] + action_span[:6] * 0.5
                )
                pred_action[..., 6] = torch.sigmoid(pred_action_norm.float()[..., 6])
        elif action_norm_method == "mean_std":
            action_mean = action_norm_stats["mean"].to(device=device, dtype=torch.float32)
            action_std = action_norm_stats["std"].to(device=device, dtype=torch.float32).clamp_min(1e-6)
            pred_action[..., :6] = pred_action_norm.float()[..., :6] * action_std[:6] + action_mean[:6]
            if args.gripper_continuous_action:
                pred_action[..., 6] = pred_action_norm.float()[..., 6] * action_std[6] + action_mean[6]
            else:
                pred_action[..., 6] = torch.sigmoid(pred_action_norm.float()[..., 6])
        else:
            raise ValueError(f"Unsupported action norm method: {action_norm_method}")
    else:
        pred_action = pred_action_norm.float() * std
    if gt_video_frames is not None:
        return pred_state.squeeze(0).cpu().numpy(), pred_action.squeeze(0).cpu().numpy(), gt_video_frames
    return pred_state.squeeze(0).cpu().numpy(), pred_action.squeeze(0).cpu().numpy(), frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--train_data_root", type=Path)
    parser.add_argument("--lora_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--state_norm_stats", required=True, type=Path)
    parser.add_argument("--action_norm_stats", type=Path)
    parser.add_argument("--gripper_continuous_action", action="store_true")
    parser.add_argument("--sa_denoise_loss", action="store_true")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--train_num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--i2av_layout", choices=["legacy", "v5"], default=None)
    parser.add_argument("--pose_pixel_frames", type=int, default=25)
    parser.add_argument("--rgb_pixel_frames", type=int, default=24)
    parser.add_argument("--infer_stage", choices=["stage1", "stage2", "stage3", "joint"], default="joint")
    parser.add_argument("--device", default="cuda", help="Inference device. Use cuda for fast custom I2AV forward.")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    args = parser.parse_args()

    if args.infer_stage == "stage1" and "clean_sa" not in args.output_dir.name:
        args.output_dir = args.output_dir.with_name(f"{args.output_dir.name}_clean_sa")

    lora_dir = resolve_lora_dir(args.lora_dir)
    state_action_path = lora_dir / "state_action.pt"
    if not state_action_path.is_file():
        raise FileNotFoundError(f"I2AV checkpoint is missing state_action.pt under {lora_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    disable_learned_positional_embeddings(pipe)
    pipe.load_lora_weights(str(lora_dir), adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [1.0])
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        inference_device = pipe._execution_device
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        inference_device = torch.device(args.device)
        pipe.to(inference_device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print(f"Inference device: {inference_device}")
    print(f"Model CPU offload: {args.enable_model_cpu_offload}")

    hidden_dim = get_transformer_hidden_dim(pipe.transformer)
    text_embed_dim = get_text_embed_dim(pipe.transformer)
    state_action_payload = torch.load(state_action_path, map_location="cpu", weights_only=False)
    checkpoint_layout = state_action_payload.get("tokenizer_type", "legacy")
    args.i2av_layout = args.i2av_layout or ("v5" if checkpoint_layout == "v5" else checkpoint_layout)
    if args.i2av_layout == "v5":
        layout = compute_i2av_v5_layout(
            pipe.transformer.config,
            pixel_height=args.height,
            pixel_width=args.width,
            pose_pixel_frames=args.pose_pixel_frames,
            rgb_pixel_frames=args.rgb_pixel_frames,
            text_seq_length=pipe.transformer.config.max_text_seq_length,
            s0_cond_tokens=4,
            vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        )
        sa_tokenizer = ChunkedStateActionTokenizer(
            hidden_dim=hidden_dim,
            steps_per_chunk=int(state_action_payload.get("steps_per_chunk", layout.steps_per_chunk)),
            first_chunk_pad_steps=layout.first_chunk_pad_steps,
            real_trajectory_steps=layout.real_trajectory_steps,
        )
    else:
        sa_tokenizer = StateActionTokenizer(hidden_dim=hidden_dim, num_state_tokens=4, num_action_tokens=4)
    s0_encoder = S0Encoder(hidden_dim=text_embed_dim, num_tokens=4)
    load_state_action_modules(str(state_action_path), sa_tokenizer, s0_encoder, device=inference_device)
    sa_tokenizer.to(device=inference_device, dtype=torch.bfloat16).eval()
    s0_encoder.to(device=inference_device, dtype=torch.bfloat16).eval()

    install_temporal_causal_attention(
        pipe.transformer,
        num_pixel_frames=args.num_frames,
        pixel_height=args.height,
        pixel_width=args.width,
        text_seq_length=pipe.transformer.config.max_text_seq_length + s0_encoder.num_tokens,
        vae_scale_factor_spatial=pipe.vae_scale_factor_spatial,
        device=inference_device,
        dtype=torch.float32,
        enable_state_action=True,
        sa_per_frame=getattr(sa_tokenizer, "num_tokens", getattr(sa_tokenizer, "chunk_token_count", 8)),
        s0_cond_tokens=s0_encoder.num_tokens,
        i2av_layout=args.i2av_layout,
        pose_pixel_frames=args.pose_pixel_frames,
        rgb_pixel_frames=args.rgb_pixel_frames,
    )

    norm_stats = torch.load(args.state_norm_stats, map_location="cpu")
    args.action_norm_stats_payload = (
        torch.load(args.action_norm_stats, map_location="cpu", weights_only=False)
        if args.action_norm_stats is not None
        else None
    )
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)
    if args.train_data_root is not None:
        eval_root = args.output_dir / "eval_dataset"
        validation_items = load_validation_items(args.data_root, args.num_samples)
        run_eval_split(
            split_name="validation",
            items=validation_items,
            eval_root=eval_root,
            pipe=pipe,
            sa_tokenizer=sa_tokenizer,
            s0_encoder=s0_encoder,
            norm_stats=norm_stats,
            args=args,
            generator=generator,
            lora_dir=lora_dir,
        )
        train_num_samples = args.train_num_samples if args.train_num_samples > 0 else args.num_samples
        train_items = load_validation_items(args.train_data_root, train_num_samples)
        run_eval_split(
            split_name="train",
            items=train_items,
            eval_root=eval_root,
            pipe=pipe,
            sa_tokenizer=sa_tokenizer,
            s0_encoder=s0_encoder,
            norm_stats=norm_stats,
            args=args,
            generator=generator,
            lora_dir=lora_dir,
        )
        return

    for idx, item in enumerate(load_validation_items(args.data_root, args.num_samples)):
        pred_state, pred_action, video = run_i2av_sample(pipe, sa_tokenizer, s0_encoder, norm_stats, item, args, generator)

        stem = f"sample_{idx:03d}"
        video_path = args.output_dir / f"{stem}.mp4"
        pred_state_path = args.output_dir / f"{stem}_pred_state.npy"
        pred_action_path = args.output_dir / f"{stem}_pred_action.npy"
        np.save(pred_state_path, pred_state)
        np.save(pred_action_path, pred_action)
        action_gripper_binary_path = None
        if args.action_norm_stats_payload is not None and not args.gripper_continuous_action:
            action_gripper_binary_path = args.output_dir / f"{stem}_pred_action_gripper_binary.npy"
            np.save(action_gripper_binary_path, (pred_action[..., 6] >= 0.5).astype(np.float32))
        export_to_video(video, str(video_path), fps=args.fps)

        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "prompt": item["prompt"],
                    "image_path": item["image_path"],
                    "video_path": item.get("video_path"),
                    "state_path": item["state_path"],
                    "action_path": item["action_path"],
                    "lora_dir": str(lora_dir),
                    "infer_stage": args.infer_stage,
                    "action_has_gripper_prob": args.action_norm_stats_payload is not None and not args.gripper_continuous_action,
                    "gripper_continuous_action": args.gripper_continuous_action,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "pred_state_path": str(pred_state_path),
                    "pred_action_path": str(pred_action_path),
                    "pred_action_gripper_binary_path": (
                        str(action_gripper_binary_path) if action_gripper_binary_path is not None else None
                    ),
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
