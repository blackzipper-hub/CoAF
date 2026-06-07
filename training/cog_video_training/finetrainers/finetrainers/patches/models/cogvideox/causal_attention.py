"""Temporal causal self-attention for CogVideoX joint text+video tokens."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .i2av_layout import I2AVV5Layout, compute_i2av_v5_layout

try:
    from torch.nn.attention import flex_attention
except ImportError:
    flex_attention = None

if TYPE_CHECKING:
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel


MASK_VALUE = -10000.0


@dataclass(frozen=True)
class CausalAttentionMeta:
    text_seq_length: int
    num_latent_frames: int
    patches_per_frame: int
    num_pixel_frames: int
    pixel_height: int
    pixel_width: int
    patch_size: int
    vae_scale_factor_spatial: int
    sequence_length: int
    sa_per_frame: int = 0
    s0_cond_tokens: int = 0
    tokens_per_step: int = 0
    enable_state_action: bool = False
    i2av_layout: str = "legacy"
    num_pose_latent_frames: int = 0
    num_rgb_latent_frames: int = 0
    chunk_token_count: int = 0
    pose_pixel_frames: int = 0
    rgb_pixel_frames: int = 0
    v5_layout: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def num_latent_frames_from_pixel_frames(num_pixel_frames: int, temporal_compression_ratio: int = 4) -> int:
    return (num_pixel_frames - 1) // temporal_compression_ratio + 1


def patches_per_latent_frame(
    *,
    pixel_height: int,
    pixel_width: int,
    patch_size: int,
    vae_scale_factor_spatial: int,
    patch_size_t: int | None = None,
) -> int:
    if patch_size_t is not None:
        raise NotImplementedError("CogVideoX 1.5 patch_size_t layout is not supported for causal masks yet.")
    latent_height = pixel_height // vae_scale_factor_spatial
    latent_width = pixel_width // vae_scale_factor_spatial
    return (latent_height // patch_size) * (latent_width // patch_size)


def build_temporal_causal_bias(
    text_seq_length: int,
    num_latent_frames: int,
    patches_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = MASK_VALUE,
) -> torch.Tensor:
    """Build additive SDPA mask bias with shape ``(1, 1, S, S)``."""
    sequence_length = text_seq_length + num_latent_frames * patches_per_frame
    q_idx = torch.arange(sequence_length, device=device)
    k_idx = torch.arange(sequence_length, device=device)

    q_grid = q_idx[:, None]
    k_grid = k_idx[None, :]

    is_text_key = k_grid < text_seq_length
    is_video_query = q_grid >= text_seq_length
    is_video_key = k_grid >= text_seq_length

    frame_key = torch.where(is_video_key, (k_grid - text_seq_length) // patches_per_frame, torch.zeros_like(k_grid))
    frame_query = torch.where(is_video_query, (q_grid - text_seq_length) // patches_per_frame, torch.zeros_like(q_grid))

    allow = is_text_key | (is_video_query & is_video_key & (frame_key <= frame_query))

    bias = torch.zeros(1, 1, sequence_length, sequence_length, device=device, dtype=dtype)
    return bias.masked_fill(~allow, mask_value)


class CogVideoXCausalTemporalAttnProcessor2_0:
    """CogVideoX attention with temporal-causal visibility over video tokens.

    A dense additive mask makes PyTorch SDPA leave the fastest kernels for this
    sequence length. Instead, compute text attention separately and run one SDPA
    call per latent video frame, where each query frame sees text plus video
    frames up to itself. This preserves the intended block-causal semantics
    without materializing a large ``S x S`` mask in the hot path.
    """

    def __init__(self, num_latent_frames: int, patches_per_frame: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.num_latent_frames = num_latent_frames
        self.patches_per_frame = patches_per_frame
        self.backend = os.environ.get("COAF_CAUSAL_ATTENTION_BACKEND", "chunked").lower()
        self._block_mask_cache: dict[tuple[torch.device, int, int], Any] = {}

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            causal_bias = build_temporal_causal_bias(
                text_seq_length=text_seq_length,
                num_latent_frames=self.num_latent_frames,
                patches_per_frame=self.patches_per_frame,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            attention_mask = attention_mask + causal_bias
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = self._temporal_causal_attention(query, key, value, text_seq_length)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

    def _temporal_causal_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        text_seq_length: int,
    ) -> torch.Tensor:
        video_seq_length = query.size(2) - text_seq_length
        expected_video_seq_length = self.num_latent_frames * self.patches_per_frame
        if video_seq_length != expected_video_seq_length:
            raise ValueError(
                "Unexpected CogVideoX video token count for temporal causal attention: "
                f"got {video_seq_length}, expected {expected_video_seq_length} "
                f"({self.num_latent_frames} frames x {self.patches_per_frame} patches)."
            )

        if self.backend == "flex":
            if flex_attention is None:
                raise RuntimeError("COAF_CAUSAL_ATTENTION_BACKEND=flex requires torch.nn.attention.flex_attention")
            block_mask = self._get_block_mask(query.device, query.size(2), text_seq_length)
            # CogVideoX q/k normalization can promote q and k to fp32 while v
            # remains bf16. Native SDPA accepts that mix, but FlexAttention
            # requires q/k/v to share a dtype.
            flex_dtype = value.dtype
            return flex_attention.flex_attention(
                query.to(dtype=flex_dtype),
                key.to(dtype=flex_dtype),
                value,
                block_mask=block_mask,
            )

        # CPU fallback, also useful for tests where FlexAttention kernels are unavailable.
        text_output = F.scaled_dot_product_attention(
            query[:, :, :text_seq_length],
            key[:, :, :text_seq_length],
            value[:, :, :text_seq_length],
            dropout_p=0.0,
            is_causal=False,
        )

        video_outputs = []
        for frame_idx in range(self.num_latent_frames):
            query_start = text_seq_length + frame_idx * self.patches_per_frame
            query_end = query_start + self.patches_per_frame
            key_value_end = query_end
            video_outputs.append(
                F.scaled_dot_product_attention(
                    query[:, :, query_start:query_end],
                    key[:, :, :key_value_end],
                    value[:, :, :key_value_end],
                    dropout_p=0.0,
                    is_causal=False,
                )
            )

        return torch.cat([text_output, *video_outputs], dim=2)

    def _get_block_mask(
        self,
        device: torch.device,
        sequence_length: int,
        text_seq_length: int,
    ) -> Any:
        if flex_attention is None:
            raise RuntimeError("FlexAttention is not available in this PyTorch build.")
        cache_key = (device, sequence_length, text_seq_length)
        block_mask = self._block_mask_cache.get(cache_key)
        if block_mask is not None:
            return block_mask

        patches_per_frame = self.patches_per_frame

        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            is_text_key = kv_idx < text_seq_length
            is_video_query = q_idx >= text_seq_length
            is_video_key = kv_idx >= text_seq_length
            q_frame = (q_idx - text_seq_length) // patches_per_frame
            kv_frame = (kv_idx - text_seq_length) // patches_per_frame
            return is_text_key | (is_video_query & is_video_key & (kv_frame <= q_frame))

        block_mask = flex_attention.create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=device,
        )
        self._block_mask_cache[cache_key] = block_mask
        return block_mask


class CogVideoXI2AVCausalTemporalAttnProcessor2_0(CogVideoXCausalTemporalAttnProcessor2_0):
    """Temporal-causal attention with interleaved state/action tokens per frame."""

    def __init__(
        self,
        num_latent_frames: int,
        patches_per_frame: int,
        sa_per_frame: int = 8,
    ):
        super().__init__(num_latent_frames=num_latent_frames, patches_per_frame=patches_per_frame)
        self.sa_per_frame = sa_per_frame
        self.tokens_per_step = patches_per_frame + sa_per_frame

    def _temporal_causal_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        text_seq_length: int,
    ) -> torch.Tensor:
        video_seq_length = query.size(2) - text_seq_length
        expected_video_seq_length = self.num_latent_frames * self.tokens_per_step
        if video_seq_length != expected_video_seq_length:
            raise ValueError(
                "Unexpected I2AV video token count: "
                f"got {video_seq_length}, expected {expected_video_seq_length} "
                f"({self.num_latent_frames} frames x {self.tokens_per_step} tokens)."
            )

        text_output = F.scaled_dot_product_attention(
            query[:, :, :text_seq_length],
            key[:, :, :text_seq_length],
            value[:, :, :text_seq_length],
            dropout_p=0.0,
            is_causal=False,
        )

        video_outputs = []
        for frame_idx in range(self.num_latent_frames):
            chunk_start = text_seq_length + frame_idx * self.tokens_per_step
            chunk_end = chunk_start + self.tokens_per_step
            q_chunk = query[:, :, chunk_start:chunk_end]
            k_chunk = key[:, :, :chunk_end]
            v_chunk = value[:, :, :chunk_end]

            kv_len = k_chunk.size(2)
            allow = torch.ones(self.tokens_per_step, kv_len, dtype=torch.bool, device=query.device)
            sa_key_start = chunk_end - self.sa_per_frame
            allow[: self.patches_per_frame, sa_key_start:chunk_end] = False

            attn_mask = torch.zeros(1, 1, self.tokens_per_step, kv_len, device=query.device, dtype=query.dtype)
            attn_mask = attn_mask.masked_fill(~allow.view(1, 1, self.tokens_per_step, kv_len), MASK_VALUE)

            video_outputs.append(
                F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            )

        return torch.cat([text_output, *video_outputs], dim=2)


class CogVideoXI2AVV5CausalAttnProcessor2_0(CogVideoXCausalTemporalAttnProcessor2_0):
    """Joint-attention v5 mask: condition tokens + pose chunks + RGB render segment."""

    def __init__(self, layout: I2AVV5Layout):
        super().__init__(num_latent_frames=layout.num_latent_frames, patches_per_frame=layout.patches_per_frame)
        self.layout = layout

    def _temporal_causal_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        text_seq_length: int,
    ) -> torch.Tensor:
        if text_seq_length != self.layout.condition_tokens:
            raise ValueError(
                f"I2AV v5 condition length mismatch: got {text_seq_length}, "
                f"expected {self.layout.condition_tokens}."
            )
        video_seq_length = query.size(2) - text_seq_length
        if video_seq_length != self.layout.video_tokens:
            raise ValueError(
                f"Unexpected I2AV v5 video token count: got {video_seq_length}, "
                f"expected {self.layout.video_tokens}."
            )
        if self.backend == "flex":
            raise NotImplementedError("I2AV v5 FlexAttention mask is not implemented yet; use chunked backend.")

        condition_output = F.scaled_dot_product_attention(
            query[:, :, :text_seq_length],
            key[:, :, :text_seq_length],
            value[:, :, :text_seq_length],
            dropout_p=0.0,
            is_causal=False,
        )

        outputs = []
        p = self.layout.patches_per_frame
        k_tokens = self.layout.chunk_token_count
        step = self.layout.pose_step_tokens
        pose_base = text_seq_length
        for chunk_idx in range(self.layout.num_pose_latent_frames):
            chunk_start = pose_base + chunk_idx * step
            chunk_end = chunk_start + step
            q_chunk = query[:, :, chunk_start:chunk_end]
            k_chunk = key[:, :, :chunk_end]
            v_chunk = value[:, :, :chunk_end]

            kv_len = k_chunk.size(2)
            allow = torch.ones(step, kv_len, dtype=torch.bool, device=query.device)
            allow[:p, kv_len - k_tokens : kv_len] = False
            attn_mask = torch.zeros(1, 1, step, kv_len, device=query.device, dtype=query.dtype)
            attn_mask = attn_mask.masked_fill(~allow.view(1, 1, step, kv_len), MASK_VALUE)
            outputs.append(
                F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            )

        rgb_start = pose_base + self.layout.pose_video_tokens
        rgb_end = rgb_start + self.layout.rgb_video_tokens
        rgb_output = F.scaled_dot_product_attention(
            query[:, :, rgb_start:rgb_end],
            key[:, :, :rgb_end],
            value[:, :, :rgb_end],
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(rgb_output)
        return torch.cat([condition_output, *outputs], dim=2)


def install_temporal_causal_attention(
    transformer: "CogVideoXTransformer3DModel",
    *,
    num_pixel_frames: int,
    pixel_height: int,
    pixel_width: int,
    text_seq_length: int | None = None,
    vae_scale_factor_spatial: int = 8,
    temporal_compression_ratio: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    enable_state_action: bool = False,
    sa_per_frame: int = 8,
    s0_cond_tokens: int = 4,
    i2av_layout: str = "legacy",
    pose_pixel_frames: int = 25,
    rgb_pixel_frames: int = 24,
) -> CausalAttentionMeta:
    """Replace all ``attn1`` processors with a temporal-causal variant."""
    patch_size = transformer.config.patch_size
    patch_size_t = getattr(transformer.config, "patch_size_t", None)
    text_seq_length = text_seq_length or transformer.config.max_text_seq_length

    num_latent_frames = num_latent_frames_from_pixel_frames(num_pixel_frames, temporal_compression_ratio)
    patches_per_frame = patches_per_latent_frame(
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        patch_size=patch_size,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        patch_size_t=patch_size_t,
    )

    device = device or next(transformer.parameters()).device
    v5_layout = None
    if enable_state_action and i2av_layout == "v5":
        base_text_seq_length = (
            text_seq_length - s0_cond_tokens
            if text_seq_length > transformer.config.max_text_seq_length
            else text_seq_length
        )
        v5_layout = compute_i2av_v5_layout(
            transformer.config,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            pose_pixel_frames=pose_pixel_frames,
            rgb_pixel_frames=rgb_pixel_frames,
            text_seq_length=base_text_seq_length,
            s0_cond_tokens=s0_cond_tokens,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        processor = CogVideoXI2AVV5CausalAttnProcessor2_0(v5_layout)
        tokens_per_step = v5_layout.pose_step_tokens
        video_seq_length = v5_layout.video_tokens
    elif enable_state_action:
        processor = CogVideoXI2AVCausalTemporalAttnProcessor2_0(
            num_latent_frames=num_latent_frames,
            patches_per_frame=patches_per_frame,
            sa_per_frame=sa_per_frame,
        )
        tokens_per_step = patches_per_frame + sa_per_frame
        video_seq_length = num_latent_frames * tokens_per_step
    else:
        processor = CogVideoXCausalTemporalAttnProcessor2_0(
            num_latent_frames=num_latent_frames,
            patches_per_frame=patches_per_frame,
        )
        tokens_per_step = patches_per_frame
        video_seq_length = num_latent_frames * patches_per_frame

    transformer.set_attn_processor(processor)

    meta = CausalAttentionMeta(
        text_seq_length=text_seq_length,
        num_latent_frames=num_latent_frames,
        patches_per_frame=patches_per_frame,
        num_pixel_frames=num_pixel_frames,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        patch_size=patch_size,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        sequence_length=text_seq_length + video_seq_length,
        sa_per_frame=sa_per_frame if enable_state_action else 0,
        s0_cond_tokens=s0_cond_tokens if enable_state_action else 0,
        tokens_per_step=tokens_per_step if enable_state_action else patches_per_frame,
        enable_state_action=enable_state_action,
        i2av_layout=i2av_layout if enable_state_action else "legacy",
        num_pose_latent_frames=v5_layout.num_pose_latent_frames if v5_layout is not None else 0,
        num_rgb_latent_frames=v5_layout.num_rgb_latent_frames if v5_layout is not None else 0,
        chunk_token_count=v5_layout.chunk_token_count if v5_layout is not None else 0,
        pose_pixel_frames=pose_pixel_frames if v5_layout is not None else 0,
        rgb_pixel_frames=rgb_pixel_frames if v5_layout is not None else 0,
        v5_layout=v5_layout.to_dict() if v5_layout is not None else None,
    )
    if v5_layout is not None:
        transformer._coaf_i2av_v5_layout = v5_layout  # noqa: SLF001
    transformer._coaf_causal_meta = meta  # noqa: SLF001
    return meta


def write_causal_attention_metadata(output_dir: str | Path, meta: CausalAttentionMeta) -> Path:
    path = Path(output_dir) / "causal_attention.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"causal_attention": True, **meta.to_dict()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
