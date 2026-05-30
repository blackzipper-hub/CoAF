"""Custom CogVideoX forward path with interleaved state/action tokens."""

from __future__ import annotations

from typing import Any

import torch

from .i2av_sequence import deinterleave_visual_sa_tokens, interleave_visual_sa_tokens


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

    timesteps = timestep
    t_emb = transformer.time_proj(timesteps)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
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
