"""Custom CogVideoX forward path with interleaved state/action tokens."""

from __future__ import annotations

from typing import Any

import torch

from .i2av_layout import I2AVV5Layout
from .i2av_sequence import (
    build_chunked_pose_rgb_tokens,
    deinterleave_chunked_pose_rgb_tokens,
    deinterleave_visual_sa_tokens,
    interleave_visual_sa_tokens,
)


def _module_device(module) -> torch.device:
    return next(module.parameters()).device


def _to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def forward_i2av_transformer(
    transformer,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    noisy_sa_tokens: torch.Tensor,
    *,
    timestep,
    timestep_cond=None,
    ofs=None,
    image_rotary_emb=None,
    attention_kwargs: dict[str, Any] | None = None,
    patches_per_frame: int,
    sa_per_frame: int,
    return_dict: bool = False,
):
    batch_size, num_frames, channels, height, width = hidden_states.shape
    attention_kwargs = attention_kwargs or {}
    device = _module_device(transformer.time_embedding)
    hidden_states = hidden_states.to(device=device)
    encoder_hidden_states = encoder_hidden_states.to(device=device)
    noisy_sa_tokens = noisy_sa_tokens.to(device=device)
    timestep = _to_device(timestep, device)
    timestep_cond = _to_device(timestep_cond, device)
    ofs = _to_device(ofs, device)
    image_rotary_emb = _to_device(image_rotary_emb, device)

    timesteps = timestep
    t_emb = transformer.time_proj(timesteps)
    t_emb = t_emb.to(device=device, dtype=hidden_states.dtype)
    emb = transformer.time_embedding(t_emb, timestep_cond)

    if transformer.ofs_embedding is not None and ofs is not None:
        ofs_emb = transformer.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = transformer.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    embeds = transformer.patch_embed(encoder_hidden_states, hidden_states)
    embeds = transformer.embedding_dropout(embeds)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = embeds[:, :text_seq_length]
    video_tokens = embeds[:, text_seq_length:]

    hidden_states = interleave_visual_sa_tokens(
        video_tokens, noisy_sa_tokens, patches_per_frame, sa_per_frame
    )

    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            hidden_states, encoder_hidden_states = transformer._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                attention_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

    hidden_states = transformer.norm_final(hidden_states)
    visual_tokens, sa_tokens = deinterleave_visual_sa_tokens(
        hidden_states, num_frames, patches_per_frame, sa_per_frame
    )

    visual_tokens = transformer.norm_out(visual_tokens, temb=emb)
    visual_tokens = transformer.proj_out(visual_tokens)

    p = transformer.config.patch_size
    p_t = transformer.config.patch_size_t
    if p_t is None:
        output = visual_tokens.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = visual_tokens.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if not return_dict:
        return output, sa_tokens
    return output, sa_tokens


def forward_i2av_v5_transformer(
    transformer,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    noisy_chunk_tokens: torch.Tensor,
    *,
    timestep,
    layout: I2AVV5Layout,
    timestep_cond=None,
    ofs=None,
    image_rotary_emb=None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = False,
):
    """CogVideoX forward for v5 [reason chunks | RGB] I2AV layout."""
    batch_size, num_frames, channels, height, width = hidden_states.shape
    attention_kwargs = attention_kwargs or {}
    device = _module_device(transformer.time_embedding)
    hidden_states = hidden_states.to(device=device)
    encoder_hidden_states = encoder_hidden_states.to(device=device)
    noisy_chunk_tokens = noisy_chunk_tokens.to(device=device)
    timestep = _to_device(timestep, device)
    timestep_cond = _to_device(timestep_cond, device)
    ofs = _to_device(ofs, device)
    image_rotary_emb = _to_device(image_rotary_emb, device)
    if num_frames != layout.num_latent_frames:
        raise ValueError(f"v5 layout expects {layout.num_latent_frames} latent frames, got {num_frames}.")

    t_emb = transformer.time_proj(timestep)
    t_emb = t_emb.to(device=device, dtype=hidden_states.dtype)
    emb = transformer.time_embedding(t_emb, timestep_cond)

    if transformer.ofs_embedding is not None and ofs is not None:
        ofs_emb = transformer.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = transformer.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    embeds = transformer.patch_embed(encoder_hidden_states, hidden_states)
    embeds = transformer.embedding_dropout(embeds)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = embeds[:, :text_seq_length]
    video_tokens = embeds[:, text_seq_length:]

    expected_visual = layout.num_latent_frames * layout.patches_per_frame
    if video_tokens.shape[1] != expected_visual:
        raise ValueError(f"Unexpected visual token count: got {video_tokens.shape[1]}, expected {expected_visual}.")

    pose_len = layout.num_pose_latent_frames * layout.patches_per_frame
    pose_tokens = video_tokens[:, :pose_len]
    rgb_tokens = video_tokens[:, pose_len:]
    hidden_states = build_chunked_pose_rgb_tokens(
        pose_tokens,
        noisy_chunk_tokens,
        rgb_tokens,
        layout.patches_per_frame,
    )

    for block in transformer.transformer_blocks:
        if torch.is_grad_enabled() and transformer.gradient_checkpointing:
            hidden_states, encoder_hidden_states = transformer._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                attention_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

    hidden_states = transformer.norm_final(hidden_states)
    pose_visual, chunk_tokens, rgb_visual = deinterleave_chunked_pose_rgb_tokens(
        hidden_states,
        layout.num_pose_latent_frames,
        layout.num_rgb_latent_frames,
        layout.patches_per_frame,
        layout.chunk_token_count,
    )
    visual_tokens = torch.cat([pose_visual, rgb_visual], dim=1)
    visual_tokens = transformer.norm_out(visual_tokens, temb=emb)
    visual_tokens = transformer.proj_out(visual_tokens)

    p = transformer.config.patch_size
    p_t = transformer.config.patch_size_t
    if p_t is not None:
        raise NotImplementedError("I2AV v5 forward does not support patch_size_t layouts yet.")
    output = visual_tokens.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if not return_dict:
        return output, chunk_tokens
    return output, chunk_tokens
