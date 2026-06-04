#!/usr/bin/env python3
"""Precompute CogVideoX I2AV video/image latents and prompt embeddings in-place."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLCogVideoX
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


LEGACY_COGVIDEOX_ROOT = Path(__file__).resolve().parents[1] / "finetrainers" / "examples" / "_legacy" / "training" / "cogvideox"
sys.path.insert(0, str(LEGACY_COGVIDEOX_ROOT))

from dataset import VideoDatasetWithResizing  # noqa: E402
from prepare_dataset import compute_prompt_embeddings  # noqa: E402


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max_num_frames", type=int, default=49)
    parser.add_argument("--max_sequence_length", type=int, default=226)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bf16")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_slicing", action="store_true")
    parser.add_argument("--use_tiling", action="store_true")
    return parser.parse_args()


def batched(items: list[int], batch_size: int):
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"Invalid shard {args.shard_index}/{args.num_shards}")

    data_root = args.data_root
    video_latents_dir = data_root / "video_latents"
    image_latents_dir = data_root / "image_latents"
    prompt_embeds_dir = data_root / "prompt_embeds"
    video_latents_dir.mkdir(parents=True, exist_ok=True)
    image_latents_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeds_dir.mkdir(parents=True, exist_ok=True)

    dataset = VideoDatasetWithResizing(
        data_root=str(data_root),
        caption_column="prompt.txt",
        video_column="videos.txt",
        image_column="images.txt",
        max_num_frames=args.max_num_frames,
        id_token="COAF",
        height_buckets=[args.height],
        width_buckets=[args.width],
        frame_buckets=[args.max_num_frames],
        load_tensors=False,
        random_flip=None,
        image_to_video=True,
    )

    indices = [idx for idx in range(len(dataset)) if idx % args.num_shards == args.shard_index]
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder.eval()

    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype).to(device)
    if args.use_slicing:
        vae.enable_slicing()
    if args.use_tiling:
        vae.enable_tiling()
    vae.eval()

    progress = tqdm(list(batched(indices, args.batch_size)), desc=f"precompute shard {args.shard_index}/{args.num_shards}")
    for batch_indices in progress:
        samples = []
        stems = []
        for idx in batch_indices:
            video_path = dataset.video_paths[idx]
            stem = video_path.stem
            outputs = [
                video_latents_dir / f"{stem}.pt",
                image_latents_dir / f"{stem}.pt",
                prompt_embeds_dir / f"{stem}.pt",
            ]
            if not args.overwrite and all(path.is_file() for path in outputs):
                continue
            samples.append(dataset[idx])
            stems.append(stem)

        if not samples:
            continue

        images = torch.stack([sample["image"] for sample in samples]).to(device=device, dtype=dtype, non_blocking=True)
        videos = torch.stack([sample["video"] for sample in samples]).to(device=device, dtype=dtype, non_blocking=True)
        prompts = [sample["prompt"] for sample in samples]

        image_latents = vae._encode(images.permute(0, 2, 1, 3, 4)).to(memory_format=torch.contiguous_format, dtype=dtype)
        video_latents = vae._encode(videos.permute(0, 2, 1, 3, 4)).to(memory_format=torch.contiguous_format, dtype=dtype)
        prompt_embeds = compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            prompts,
            args.max_sequence_length,
            device,
            dtype,
            requires_grad=False,
        )

        for item_idx, stem in enumerate(stems):
            torch.save(video_latents[item_idx].detach().cpu(), video_latents_dir / f"{stem}.pt")
            torch.save(image_latents[item_idx].detach().cpu(), image_latents_dir / f"{stem}.pt")
            torch.save(prompt_embeds[item_idx].detach().cpu(), prompt_embeds_dir / f"{stem}.pt")

    print(f"Completed shard {args.shard_index}/{args.num_shards} under {data_root}")


if __name__ == "__main__":
    main()
