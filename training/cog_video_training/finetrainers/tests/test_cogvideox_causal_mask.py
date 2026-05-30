import torch

from finetrainers.patches.models.cogvideox.causal_attention import (
    MASK_VALUE,
    build_temporal_causal_bias,
    num_latent_frames_from_pixel_frames,
    patches_per_latent_frame,
)


def test_mask_rules():
    text_len = 4
    num_latent_frames = 3
    patches_per_frame = 2
    bias = build_temporal_causal_bias(text_len, num_latent_frames, patches_per_frame, device="cpu", dtype=torch.float32)
    assert bias.shape == (1, 1, 10, 10)

    allowed = bias[0, 0] == 0
    blocked = bias[0, 0] == MASK_VALUE

    # video can read all text
    assert allowed[text_len, 0]
    assert allowed[text_len + patches_per_frame, text_len - 1]

    # text cannot read video
    assert blocked[0, text_len]
    assert blocked[text_len - 1, text_len + 1]

    # temporal causal on video
    assert allowed[text_len + patches_per_frame, text_len + patches_per_frame - 1]
    assert blocked[text_len, text_len + patches_per_frame]

    # same-frame spatial tokens see each other
    assert allowed[text_len + 1, text_len]


def test_geometry_for_49f_256():
    assert num_latent_frames_from_pixel_frames(49) == 13
    assert patches_per_latent_frame(pixel_height=256, pixel_width=256, patch_size=2, vae_scale_factor_spatial=8) == 256


if __name__ == "__main__":
    test_mask_rules()
    test_geometry_for_49f_256()
