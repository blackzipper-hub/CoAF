#!/usr/bin/env python3
"""Verify that 25 reason + 24 RGB aligns with CogVideoX causal VAE latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLCogVideoX


POSE_LATENT_MAP = [
    (0, [0]),
    (1, [1, 2, 3, 4]),
    (2, [5, 6, 7, 8]),
    (3, [9, 10, 11, 12]),
    (4, [13, 14, 15, 16]),
    (5, [17, 18, 19, 20]),
    (6, [21, 22, 23, 24]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="CogVideoX-I2V model root or HF repo.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pose_frames", type=int, default=25)
    parser.add_argument("--rgb_frames", type=int, default=24)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype).to(device)
    vae.eval()

    pose = torch.ones(1, 3, args.pose_frames, args.height, args.width, device=device, dtype=dtype)
    rgb = torch.zeros(1, 3, args.rgb_frames, args.height, args.width, device=device, dtype=dtype)
    video = torch.cat([pose, rgb], dim=2)

    with torch.no_grad():
        latent_dist = vae.encode(video).latent_dist
        latents = latent_dist.sample()
        decoded = vae.decode(latents).sample.float()

    boundary_left = decoded[:, :, args.pose_frames - 1].mean().item()
    boundary_right = decoded[:, :, args.pose_frames].mean().item()
    payload = {
        "pose_frames": args.pose_frames,
        "rgb_frames": args.rgb_frames,
        "latent_frames": int(latents.shape[2]),
        "expected_latent_frames": (args.pose_frames + args.rgb_frames - 1) // 4 + 1,
        "boundary_left_mean": boundary_left,
        "boundary_right_mean": boundary_right,
        "pose_latent_map": POSE_LATENT_MAP,
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if payload["latent_frames"] != payload["expected_latent_frames"]:
        raise SystemExit("Unexpected VAE latent frame count.")
    if not boundary_left > boundary_right:
        raise SystemExit("Decoded boundary did not preserve pose/RGB ordering.")


if __name__ == "__main__":
    main()
