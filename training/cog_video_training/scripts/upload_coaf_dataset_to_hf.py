#!/usr/bin/env python3
"""Upload the CoAF dataset folder to a Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_DATASET_DIR = Path(
    "/project/mscaisuperpod/sunkai/Casual_CoAF/coaf_dataset_24_25"
)
DEFAULT_REPO_ID = "BlackZipper/coaf_dataset_24_25"
DEFAULT_TOKEN_STORE = Path("/project/llmsvgen/yazhou/huggingface/stored_tokens")
DEFAULT_IGNORE_PATTERNS = (
    ".cache/**",
    "**/.cache/**",
    "composed/v4_depth_rgb_load_tensors_smoke/**",
)


def resolve_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()

    token_name = os.environ.get("HF_TOKEN_NAME", "123")
    token_store = Path(os.environ.get("HF_STORED_TOKENS", DEFAULT_TOKEN_STORE))
    if not token_store.is_file():
        raise FileNotFoundError(
            f"HF token not set. Export HF_TOKEN or create token store: {token_store}"
        )

    config = configparser.ConfigParser()
    config.read(token_store)
    if token_name not in config or "hf_token" not in config[token_name]:
        raise KeyError(f"Token name not found in {token_store}: {token_name}")
    return config[token_name]["hf_token"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("HF_UPLOAD_WORKERS", "8")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {args.dataset_dir}")

    api = HfApi(token=resolve_token())
    whoami = api.whoami()
    print(f"Authenticated as: {whoami.get('name') or whoami.get('fullname')}", flush=True)
    print(f"Uploading folder: {args.dataset_dir}", flush=True)
    print(f"Target repo: {args.repo_id} ({args.repo_type})", flush=True)
    print("Note: upload_large_folder uploads the folder contents to the repo root.", flush=True)

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=args.dataset_dir,
        ignore_patterns=list(DEFAULT_IGNORE_PATTERNS),
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=60,
    )
    print("Dataset upload finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
