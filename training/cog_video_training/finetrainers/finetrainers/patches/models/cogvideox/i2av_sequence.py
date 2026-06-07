"""Token sequence helpers for I2AV interleaved visual + state/action layout."""

from __future__ import annotations

import torch


def interleave_visual_sa_tokens(
    visual_tokens: torch.Tensor,
    sa_tokens: torch.Tensor,
    patches_per_frame: int,
    sa_per_frame: int,
) -> torch.Tensor:
    """Legacy v3: visual (B, F*P, D) + sa (B, F, K, D) -> (B, F*(P+K), D)."""
    b, _, d = visual_tokens.shape
    num_frames = sa_tokens.shape[1]
    visual = visual_tokens.reshape(b, num_frames, patches_per_frame, d)
    parts = []
    for t in range(num_frames):
        parts.append(visual[:, t])
        parts.append(sa_tokens[:, t])
    return torch.cat(parts, dim=1)


def deinterleave_visual_sa_tokens(
    tokens: torch.Tensor,
    num_frames: int,
    patches_per_frame: int,
    sa_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy v3: (B, F*(P+K), D) -> visual (B, F*P, D), sa (B, F, K, D)."""
    b, _, d = tokens.shape
    step = patches_per_frame + sa_per_frame
    visual_parts = []
    sa_parts = []
    for t in range(num_frames):
        start = t * step
        visual_parts.append(tokens[:, start : start + patches_per_frame])
        sa_parts.append(tokens[:, start + patches_per_frame : start + step])
    visual = torch.cat(visual_parts, dim=1)
    sa = torch.stack(sa_parts, dim=1)
    return visual, sa


def build_chunked_pose_rgb_tokens(
    pose_visual_tokens: torch.Tensor,
    chunk_tokens: torch.Tensor,
    rgb_visual_tokens: torch.Tensor,
    patches_per_frame: int,
) -> torch.Tensor:
    """v5 layout: pose chunks then RGB segment.

    Args:
        pose_visual_tokens: (B, F_pose * P, D)
        chunk_tokens: (B, F_pose, K, D)
        rgb_visual_tokens: (B, F_rgb * P, D)
    """
    b, _, d = pose_visual_tokens.shape
    num_pose = chunk_tokens.shape[1]
    pose = pose_visual_tokens.reshape(b, num_pose, patches_per_frame, d)
    parts: list[torch.Tensor] = []
    for t in range(num_pose):
        parts.append(pose[:, t])
        parts.append(chunk_tokens[:, t])
    parts.append(rgb_visual_tokens)
    return torch.cat(parts, dim=1)


def deinterleave_chunked_pose_rgb_tokens(
    tokens: torch.Tensor,
    num_pose_latent_frames: int,
    num_rgb_latent_frames: int,
    patches_per_frame: int,
    chunk_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split v5 sequence back into pose visual, chunk, and rgb visual tokens."""
    b, _, d = tokens.shape
    pose_step = patches_per_frame + chunk_token_count
    pose_len = num_pose_latent_frames * pose_step
    pose_parts = []
    chunk_parts = []
    for t in range(num_pose_latent_frames):
        start = t * pose_step
        pose_parts.append(tokens[:, start : start + patches_per_frame])
        chunk_parts.append(tokens[:, start + patches_per_frame : start + pose_step])
    pose_visual = torch.cat(pose_parts, dim=1)
    chunks = torch.stack(chunk_parts, dim=1)
    rgb_start = pose_len
    rgb_visual = tokens[:, rgb_start : rgb_start + num_rgb_latent_frames * patches_per_frame]
    return pose_visual, chunks, rgb_visual


def expand_rope_for_i2av(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_frames: int,
    patches_per_frame: int,
    sa_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert SA RoPE rows after each visual frame block (legacy v3)."""
    cos_parts = []
    sin_parts = []
    for t in range(num_frames):
        start = t * patches_per_frame
        end = start + patches_per_frame
        cos_parts.append(freqs_cos[start:end])
        sin_parts.append(freqs_sin[start:end])
        cos_parts.append(freqs_cos[end - 1 : end].repeat(sa_per_frame, 1))
        sin_parts.append(freqs_sin[end - 1 : end].repeat(sa_per_frame, 1))
    return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)


def expand_rope_for_chunked_i2av(
    pose_freqs_cos: torch.Tensor,
    pose_freqs_sin: torch.Tensor,
    rgb_freqs_cos: torch.Tensor,
    rgb_freqs_sin: torch.Tensor,
    num_pose_latent_frames: int,
    patches_per_frame: int,
    chunk_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for v5 pose+chunk+rgb layout."""
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for t in range(num_pose_latent_frames):
        start = t * patches_per_frame
        end = start + patches_per_frame
        cos_parts.append(pose_freqs_cos[start:end])
        sin_parts.append(pose_freqs_sin[start:end])
        cos_parts.append(pose_freqs_cos[end - 1 : end].repeat(chunk_token_count, 1))
        sin_parts.append(pose_freqs_sin[end - 1 : end].repeat(chunk_token_count, 1))
    cos_parts.append(rgb_freqs_cos)
    sin_parts.append(rgb_freqs_sin)
    return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)


def expand_rope_for_chunked_i2av_timeids(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_pose_latent_frames: int,
    num_rgb_latent_frames: int,
    patches_per_frame: int,
    chunk_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE for v5 using explicit time ids: P_k=2k, c_k=2k+1, R_j=2F_pose+j."""
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for t in range(num_pose_latent_frames):
        pose_start = (2 * t) * patches_per_frame
        pose_end = pose_start + patches_per_frame
        chunk_start = (2 * t + 1) * patches_per_frame
        chunk_end = chunk_start + patches_per_frame
        cos_parts.append(freqs_cos[pose_start:pose_end])
        sin_parts.append(freqs_sin[pose_start:pose_end])
        cos_parts.append(freqs_cos[chunk_end - 1 : chunk_end].repeat(chunk_token_count, 1))
        sin_parts.append(freqs_sin[chunk_end - 1 : chunk_end].repeat(chunk_token_count, 1))
    rgb_base = 2 * num_pose_latent_frames * patches_per_frame
    rgb_len = num_rgb_latent_frames * patches_per_frame
    cos_parts.append(freqs_cos[rgb_base : rgb_base + rgb_len])
    sin_parts.append(freqs_sin[rgb_base : rgb_base + rgb_len])
    return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)
