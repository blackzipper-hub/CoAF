"""Upsample one episode to 640x480, prepare depth, and compose v4_depth_rgb."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

DATASET_ROOT = Path(
    os.environ.get(
        "DATASET_ROOT",
        "/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25",
    )
)
RAW_ROOT = DATASET_ROOT / "raw"
DEPTH_ROOT = DATASET_ROOT / "modalities" / "depth"
RAW_UP_ROOT = DATASET_ROOT / "raw_640x480"
DEPTH_UP_ROOT = DATASET_ROOT / "modalities" / "depth_640x480"
COMPOSED_ROOT = DATASET_ROOT / "composed"

OUT_WIDTH = 640
OUT_HEIGHT = 480
REASON_FRAMES = 25
RGB_FRAMES = 24


def upsample_png_dir(src_dir: Path, dst_dir: Path, num_frames: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, num_frames + 1):
        src = src_dir / f"frame_{i:04d}.png"
        dst = dst_dir / f"frame_{i:04d}.png"
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(src)
        up = cv2.resize(img, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(dst), up)


def upsample_depth_video(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        up = cv2.resize(rgb, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        frames.append(up)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(dst), frames, fps=8, codec="libx264", macro_block_size=1)


def read_video_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames)


def sample_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if len(frames) == num_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return frames[indices]


def read_rgb_pngs(rgb_dir: Path, num_frames: int) -> np.ndarray:
    frames = []
    for i in range(1, num_frames + 1):
        path = rgb_dir / f"frame_{i:04d}.png"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def maybe_run_vda_depth(rgb_align_dir: Path, depth_out: Path) -> bool:
    vda_root = os.environ.get("VDA_ROOT", "").strip()
    if not vda_root:
        return False
    vda_path = Path(vda_root)
    ckpt = vda_path / "checkpoints" / "video_depth_anything_vitl.pth"
    if not ckpt.is_file():
        print(f"[depth] VDA checkpoint missing at {ckpt}, falling back to upsample")
        return False

    import torch

    if not torch.cuda.is_available():
        print("[depth] CUDA unavailable for VDA, falling back to upsample")
        return False

    sys.path.insert(0, str(vda_path))
    os.chdir(str(vda_path))
    from utils.dc_utils import read_video_frames as vda_read_frames
    from utils.dc_utils import save_video
    from video_depth_anything.video_depth import VideoDepthAnything

    model_configs = {
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    model = VideoDepthAnything(**model_configs["vitl"], metric=False)
    state_dict = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda").eval()

    tmp_video = depth_out.parent / "_tmp_rgb_align.mp4"
    frames = [imageio.imread(str(rgb_align_dir / f"frame_{i:04d}.png")) for i in range(1, REASON_FRAMES + 1)]
    tmp_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(tmp_video), frames, fps=8, codec="libx264", macro_block_size=1)
    frames_np, target_fps = vda_read_frames(str(tmp_video), -1, -1, 1280)
    depths, fps = model.infer_video_depth(frames_np, target_fps, input_size=518, device="cuda", fp32=False)
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    save_video(depths, str(depth_out), fps=fps, is_depths=True, grayscale=False)
    tmp_video.unlink(missing_ok=True)
    print(f"[depth] VDA depth written to {depth_out}")
    return True


def compose_episode(episode_idx: int, output_name: str, fps: int = 8) -> Path:
    ep_name = f"episode_{episode_idx:06d}"
    raw_up = RAW_UP_ROOT / ep_name
    depth_path = DEPTH_UP_ROOT / ep_name / "depth.mp4"
    if not depth_path.is_file():
        src_depth = DEPTH_ROOT / ep_name / "depth.mp4"
        if not src_depth.is_file():
            raise FileNotFoundError(f"Missing depth for {ep_name}: {src_depth}")
        print(f"[depth] Upsampling {src_depth} -> {depth_path}")
        upsample_depth_video(src_depth, depth_path)

    output_root = COMPOSED_ROOT / output_name
    videos_dir = output_root / "videos"
    cond_dir = output_root / "condition_images"
    videos_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    depth_frames = sample_frames(read_video_frames(depth_path), REASON_FRAMES)
    rgb_frames = read_rgb_pngs(raw_up / "rgb", RGB_FRAMES)
    combined = np.concatenate([depth_frames, rgb_frames], axis=0)
    assert len(combined) == REASON_FRAMES + RGB_FRAMES

    out_video = videos_dir / f"{ep_name}.mp4"
    imageio.mimsave(str(out_video), combined, fps=fps, codec="libx264", macro_block_size=1)
    cond_image = cond_dir / f"{ep_name}.png"
    imageio.imwrite(str(cond_image), rgb_frames[0])

    state_path = RAW_ROOT / ep_name / "state" / "state.npy"
    action_path = RAW_ROOT / ep_name / "action" / "action.npy"
    instruction_file = RAW_ROOT / ep_name / "instruction" / "instruction.txt"
    prompt = "robot manipulation task"
    if instruction_file.is_file():
        text = instruction_file.read_text(encoding="utf-8").strip()
        if text:
            prompt = text

    for name, lines in {
        "videos.txt": [str(out_video)],
        "images.txt": [str(cond_image)],
        "prompt.txt": [prompt],
        "state_paths.txt": [str(state_path)],
        "action_paths.txt": [str(action_path)],
    }.items():
        (output_root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output_root / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image", "video", "text"])
        writer.writeheader()
        writer.writerow({"index": 0, "image": str(cond_image), "video": str(out_video), "text": prompt})

    val_entry = {
        "sample_index": 0,
        "caption": prompt,
        "image_path": str(cond_image),
        "video_path": str(out_video),
    }
    (output_root / "validation.json").write_text(
        json.dumps({"data": [val_entry]}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[compose] Wrote {output_root}")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--output-name", type=str, default="v4_depth_rgb_one_sample_640x480")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--depth-mode",
        choices=("auto", "vda", "upsample"),
        default="auto",
        help="auto tries VDA if VDA_ROOT is set, vda requires VDA, upsample resizes existing depth.mp4.",
    )
    args = parser.parse_args()

    ep_name = f"episode_{args.episode_idx:06d}"
    src_ep = RAW_ROOT / ep_name
    dst_ep = RAW_UP_ROOT / ep_name
    if not src_ep.is_dir():
        raise FileNotFoundError(src_ep)

    print(f"[raw] Upsampling {src_ep} rgb/rgb_align -> {OUT_WIDTH}x{OUT_HEIGHT}")
    upsample_png_dir(src_ep / "rgb", dst_ep / "rgb", RGB_FRAMES)
    upsample_png_dir(src_ep / "rgb_align", dst_ep / "rgb_align", REASON_FRAMES)

    depth_out = DEPTH_UP_ROOT / ep_name / "depth.mp4"
    if args.depth_mode == "vda":
        if not maybe_run_vda_depth(dst_ep / "rgb_align", depth_out):
            raise RuntimeError("depth-mode=vda requested, but VDA depth preprocessing could not run")
    elif args.depth_mode == "auto" and maybe_run_vda_depth(dst_ep / "rgb_align", depth_out):
        pass
    else:
        src_depth = DEPTH_ROOT / ep_name / "depth.mp4"
        print(f"[depth] Upsampling existing depth {src_depth} -> {depth_out}")
        upsample_depth_video(src_depth, depth_out)

    out_root = compose_episode(args.episode_idx, args.output_name, fps=args.fps)
    sample = cv2.imread(str(out_root / "condition_images" / f"{ep_name}.png"))
    print(f"[done] composed={out_root} cond_shape={sample.shape if sample is not None else None}")


if __name__ == "__main__":
    main()
