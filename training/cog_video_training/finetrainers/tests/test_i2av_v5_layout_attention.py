from types import SimpleNamespace

import torch

from finetrainers.patches.models.cogvideox.causal_attention import CogVideoXI2AVV5CausalAttnProcessor2_0
from finetrainers.patches.models.cogvideox.i2av_layout import compute_i2av_v5_layout


def test_i2av_v5_layout_256_and_480640():
    config = SimpleNamespace(patch_size=2, patch_size_t=None, max_text_seq_length=226)
    layout_256 = compute_i2av_v5_layout(config, pixel_height=256, pixel_width=256)
    assert layout_256.patches_per_frame == 256
    assert layout_256.num_pose_latent_frames == 7
    assert layout_256.num_rgb_latent_frames == 6
    assert layout_256.chunk_token_count == 8
    assert layout_256.video_tokens == 7 * (256 + 8) + 6 * 256

    layout_480 = compute_i2av_v5_layout(config, pixel_height=480, pixel_width=640)
    assert layout_480.patches_per_frame == 1200
    assert layout_480.video_tokens == 7 * (1200 + 8) + 6 * 1200


def test_i2av_v5_processor_shapes():
    config = SimpleNamespace(patch_size=2, patch_size_t=None, max_text_seq_length=3)
    layout = compute_i2av_v5_layout(
        config,
        pixel_height=16,
        pixel_width=16,
        pose_pixel_frames=25,
        rgb_pixel_frames=24,
        text_seq_length=3,
        s0_cond_tokens=1,
        vae_scale_factor_spatial=8,
    )
    processor = CogVideoXI2AVV5CausalAttnProcessor2_0(layout)
    b, h, d = 1, 2, 4
    seq = layout.sequence_length
    query = torch.randn(b, h, seq, d)
    key = torch.randn(b, h, seq, d)
    value = torch.randn(b, h, seq, d)
    out = processor._temporal_causal_attention(query, key, value, layout.condition_tokens)
    assert out.shape == (b, h, seq, d)
