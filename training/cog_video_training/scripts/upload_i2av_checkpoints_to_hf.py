#!/usr/bin/env python3
"""Upload selected I2AV checkpoint families to Hugging Face (weights only)."""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "BlackZipper/cogvideo"
DEFAULT_CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / "i2av"
)
DEFAULT_FAMILIES = (
    "v5_depth_rgb_2524_stage1",
    "v5_depth_rgb_2524_stage2_raw_action_lingbot_d6cont",
    "v5_depth_rgb_2524_stage2_sa_denoise_d6cont",
)
DEFAULT_IGNORE = (
    "optimizer.bin",
    "scheduler.bin",
    "random_states_0.pkl",
    "random_states_1.pkl",
    "random_states_2.pkl",
    "random_states_3.pkl",
)


def resolve_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()

    token_name = os.environ.get("HF_TOKEN_NAME", "123")
    store_path = Path(
        os.environ.get(
            "HF_STORED_TOKENS",
            "/project/llmsvgen/yazhou/huggingface/stored_tokens",
        )
    )
    if not store_path.is_file():
        raise FileNotFoundError(
            f"HF token not set. Export HF_TOKEN or create token store: {store_path}"
        )

    cfg = configparser.ConfigParser()
    cfg.read(store_path)
    if token_name not in cfg or "hf_token" not in cfg[token_name]:
        raise KeyError(f"Token name not found in {store_path}: {token_name}")
    return cfg[token_name]["hf_token"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(
            os.environ.get("UPLOAD_FAMILIES", ",".join(DEFAULT_FAMILIES)).split(",")
        ),
    )
    parser.add_argument(
        "--path-prefix",
        default=os.environ.get("HF_PATH_PREFIX", "checkpoints/i2av"),
        help="Repo path prefix, e.g. checkpoints/i2av/<family>/...",
    )
    parser.add_argument("--repo-type", default="model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = resolve_token()
    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name') or who.get('fullname')}")
    print(f"Target repo: {args.repo_id} ({args.repo_type})")

    families = [name.strip() for name in args.families if name.strip()]
    if not families:
        print("No checkpoint families selected.", file=sys.stderr)
        return 1

    for family in families:
        local_dir = args.checkpoint_root / family
        if not local_dir.is_dir():
            print(f"SKIP missing directory: {local_dir}", file=sys.stderr)
            continue

        path_in_repo = f"{args.path_prefix.rstrip('/')}/{family}"
        print(f"Uploading {local_dir} -> {args.repo_id}:{path_in_repo}")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=local_dir,
            path_in_repo=path_in_repo,
            commit_message=f"Upload I2AV weights: {family}",
            ignore_patterns=list(DEFAULT_IGNORE),
        )
        print(f"Done: {family}")

    print("All requested families uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
