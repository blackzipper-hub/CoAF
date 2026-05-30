"""Token sequence helpers for I2AV interleaved visual + state/action layout."""

from __future__ import annotations

import torch


def interleave_visual_sa_tokens(
    visual_tokens: torch.Tensor,
    sa_tokens: torch.Tensor,
    patches_per_frame: int,
    sa_per_frame: int,
) -> torch.Tensor:
    """visual (B, F*P, D) + sa (B, F, K, D) -> (B, F*(P+K), D)."""
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
    """(B, F*(P+K), D) -> visual (B, F*P, D), sa (B, F, K, D)."""
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


def expand_rope_for_i2av(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    num_frames: int,
    patches_per_frame: int,
    sa_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert SA RoPE rows after each visual frame block."""
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
