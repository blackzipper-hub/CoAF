"""Layout helpers for CoAF I2AV v5 chunked pose/reason + RGB sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
        raise NotImplementedError("CogVideoX 1.5 patch_size_t layout is not supported for I2AV v5 yet.")
    latent_height = pixel_height // vae_scale_factor_spatial
    latent_width = pixel_width // vae_scale_factor_spatial
    return (latent_height // patch_size) * (latent_width // patch_size)


@dataclass(frozen=True)
class I2AVV5Layout:
    """Runtime-derived token layout for v5 chunked I2AV."""

    text_seq_length: int
    s0_cond_tokens: int
    pose_pixel_frames: int
    rgb_pixel_frames: int
    num_pixel_frames: int
    num_latent_frames: int
    num_pose_latent_frames: int
    num_rgb_latent_frames: int
    patches_per_frame: int
    patch_size: int
    vae_scale_factor_spatial: int
    temporal_compression_ratio: int
    pixel_height: int
    pixel_width: int
    steps_per_chunk: int
    first_chunk_steps: int
    first_chunk_pad_steps: int
    chunk_token_count: int

    @property
    def condition_tokens(self) -> int:
        return self.text_seq_length + self.s0_cond_tokens

    @property
    def pose_step_tokens(self) -> int:
        return self.patches_per_frame + self.chunk_token_count

    @property
    def pose_video_tokens(self) -> int:
        return self.num_pose_latent_frames * self.pose_step_tokens

    @property
    def rgb_video_tokens(self) -> int:
        return self.num_rgb_latent_frames * self.patches_per_frame

    @property
    def video_tokens(self) -> int:
        return self.pose_video_tokens + self.rgb_video_tokens

    @property
    def sequence_length(self) -> int:
        return self.condition_tokens + self.video_tokens

    @property
    def padded_trajectory_steps(self) -> int:
        return self.num_pose_latent_frames * self.steps_per_chunk

    @property
    def real_trajectory_steps(self) -> int:
        return self.pose_pixel_frames

    def pose_chunk_offset(self, chunk_idx: int) -> int:
        return chunk_idx * self.pose_step_tokens

    def rgb_offset(self) -> int:
        return self.pose_video_tokens

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "condition_tokens": self.condition_tokens,
                "pose_step_tokens": self.pose_step_tokens,
                "pose_video_tokens": self.pose_video_tokens,
                "rgb_video_tokens": self.rgb_video_tokens,
                "video_tokens": self.video_tokens,
                "sequence_length": self.sequence_length,
                "padded_trajectory_steps": self.padded_trajectory_steps,
            }
        )
        return payload


def compute_i2av_v5_layout(
    transformer_config: Any,
    *,
    pixel_height: int,
    pixel_width: int,
    pose_pixel_frames: int = 25,
    rgb_pixel_frames: int = 24,
    text_seq_length: int | None = None,
    s0_cond_tokens: int = 4,
    vae_scale_factor_spatial: int = 8,
    temporal_compression_ratio: int = 4,
) -> I2AVV5Layout:
    patch_size = int(transformer_config.patch_size)
    patch_size_t = getattr(transformer_config, "patch_size_t", None)
    text_seq_length = int(text_seq_length or transformer_config.max_text_seq_length)
    num_pixel_frames = int(pose_pixel_frames + rgb_pixel_frames)
    num_latent_frames = num_latent_frames_from_pixel_frames(num_pixel_frames, temporal_compression_ratio)
    num_pose_latent_frames = num_latent_frames_from_pixel_frames(pose_pixel_frames, temporal_compression_ratio)
    num_rgb_latent_frames = num_latent_frames - num_pose_latent_frames
    if num_rgb_latent_frames <= 0:
        raise ValueError(
            f"Bad v5 frame split: pose={pose_pixel_frames}, rgb={rgb_pixel_frames} "
            f"yields F_pose={num_pose_latent_frames}, F_rgb={num_rgb_latent_frames}."
        )

    patches_per_frame = patches_per_latent_frame(
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        patch_size=patch_size,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        patch_size_t=patch_size_t,
    )

    first_chunk_steps = 1
    if num_pose_latent_frames <= 1:
        steps_per_chunk = max(pose_pixel_frames, 1)
    else:
        remaining_steps = pose_pixel_frames - first_chunk_steps
        remaining_chunks = num_pose_latent_frames - 1
        if remaining_steps % remaining_chunks != 0:
            raise ValueError(
                "I2AV v5 currently expects pose steps to align as first frame + equal chunks; "
                f"got pose_pixel_frames={pose_pixel_frames}, F_pose={num_pose_latent_frames}."
            )
        steps_per_chunk = remaining_steps // remaining_chunks
    first_chunk_pad_steps = max(steps_per_chunk - first_chunk_steps, 0)

    return I2AVV5Layout(
        text_seq_length=text_seq_length,
        s0_cond_tokens=int(s0_cond_tokens),
        pose_pixel_frames=int(pose_pixel_frames),
        rgb_pixel_frames=int(rgb_pixel_frames),
        num_pixel_frames=num_pixel_frames,
        num_latent_frames=num_latent_frames,
        num_pose_latent_frames=num_pose_latent_frames,
        num_rgb_latent_frames=num_rgb_latent_frames,
        patches_per_frame=patches_per_frame,
        patch_size=patch_size,
        vae_scale_factor_spatial=int(vae_scale_factor_spatial),
        temporal_compression_ratio=int(temporal_compression_ratio),
        pixel_height=int(pixel_height),
        pixel_width=int(pixel_width),
        steps_per_chunk=steps_per_chunk,
        first_chunk_steps=first_chunk_steps,
        first_chunk_pad_steps=first_chunk_pad_steps,
        chunk_token_count=2 * steps_per_chunk,
    )
