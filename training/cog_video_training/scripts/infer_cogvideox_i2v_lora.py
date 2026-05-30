#!/usr/bin/env python3
"""Run CogVideoX I2V LoRA inference from a composed validation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


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


def load_validation_items(data_root: Path, max_samples: int) -> list[dict[str, str]]:
    validation_path = data_root / "validation.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    data = payload.get("data", payload)
    items = []
    for item in data[:max_samples]:
        image_path = item.get("image_path") or item.get("image")
        prompt = item.get("caption") or item.get("prompt") or item.get("text")
        if image_path is None or prompt is None:
            raise ValueError(f"validation item missing image/prompt fields: {item}")
        items.append({"image_path": image_path, "prompt": prompt})
    return items


def maybe_install_causal_attention(pipe: CogVideoXImageToVideoPipeline, args: argparse.Namespace) -> None:
    if not args.temporal_causal_attention:
        return
    from finetrainers.patches.models.cogvideox.causal_attention import install_temporal_causal_attention

    install_temporal_causal_attention(
        pipe.transformer,
        num_pixel_frames=args.num_frames,
        pixel_height=args.height,
        pixel_width=args.width,
        vae_scale_factor_spatial=8,
        device=pipe.device,
        dtype=torch.float32,
    )


def disable_learned_positional_embeddings(pipe: CogVideoXImageToVideoPipeline) -> None:
    """Match training, which used --ignore_learned_positional_embeddings."""
    patch_embed = pipe.transformer.patch_embed
    if hasattr(patch_embed, "pos_embedding"):
        del patch_embed.pos_embedding
    patch_embed.use_learned_positional_embeddings = False
    pipe.transformer.config.use_learned_positional_embeddings = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--lora_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal_causal_attention", action="store_true")
    parser.add_argument("--require_state_action", action="store_true")
    args = parser.parse_args()

    lora_dir = resolve_lora_dir(args.lora_dir)
    if args.require_state_action and not (lora_dir / "state_action.pt").is_file() and not (args.lora_dir / "state_action.pt").is_file():
        raise FileNotFoundError(f"I2AV checkpoint is missing state_action.pt under {lora_dir} or {args.lora_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    disable_learned_positional_embeddings(pipe)
    pipe.load_lora_weights(str(lora_dir), adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [1.0])
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    maybe_install_causal_attention(pipe, args)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)
    for idx, item in enumerate(load_validation_items(args.data_root, args.num_samples)):
        image = load_image(item["image_path"])
        video = pipe(
            image=image,
            prompt=item["prompt"],
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            generator=generator,
            output_type="np",
        ).frames[0]
        out_path = args.output_dir / f"sample_{idx:03d}.mp4"
        export_to_video(video, str(out_path), fps=args.fps)
        (args.output_dir / f"sample_{idx:03d}.json").write_text(
            json.dumps(
                {
                    "prompt": item["prompt"],
                    "image_path": item["image_path"],
                    "lora_dir": str(lora_dir),
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
