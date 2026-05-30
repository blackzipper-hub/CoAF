"""State/action tokenizers and losses for I2AV joint training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateActionTokenizer(nn.Module):
    """Encode/decode 7DoF state + action into 8 tokens per latent frame."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 7,
        num_state_tokens: int = 4,
        num_action_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_state_tokens = num_state_tokens
        self.num_action_tokens = num_action_tokens
        self.num_tokens = num_state_tokens + num_action_tokens
        self.hidden_dim = hidden_dim

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_state_tokens * hidden_dim),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_action_tokens * hidden_dim),
        )
        self.state_output = nn.Sequential(
            nn.Linear(num_state_tokens * hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        self.action_output = nn.Sequential(
            nn.Linear(num_action_tokens * hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, state_dim),
        )
        self.state_modality_emb = nn.Parameter(torch.randn(1, 1, num_state_tokens, hidden_dim) * 0.02)
        self.action_modality_emb = nn.Parameter(torch.randn(1, 1, num_action_tokens, hidden_dim) * 0.02)

    def encode(self, state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        b, t, _ = state_norm.shape
        s_tok = self.state_proj(state_norm).reshape(b, t, self.num_state_tokens, self.hidden_dim)
        s_tok = s_tok + self.state_modality_emb
        a_tok = self.action_proj(action_norm).reshape(b, t, self.num_action_tokens, self.hidden_dim)
        a_tok = a_tok + self.action_modality_emb
        return torch.cat([s_tok, a_tok], dim=2)

    def decode(self, token_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, d = token_output.shape
        s_tok = token_output[:, :, : self.num_state_tokens]
        a_tok = token_output[:, :, self.num_state_tokens :]
        pred_state = self.state_output(s_tok.reshape(b, t, self.num_state_tokens * d))
        pred_action = self.action_output(a_tok.reshape(b, t, self.num_action_tokens * d))
        return pred_state, pred_action


class S0Encoder(nn.Module):
    """Initial joint configuration as global condition tokens."""

    def __init__(self, hidden_dim: int, state_dim: int = 7, num_tokens: int = 4) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_tokens * hidden_dim),
        )

    def forward(self, s0_norm: torch.Tensor) -> torch.Tensor:
        b = s0_norm.shape[0]
        return self.proj(s0_norm).reshape(b, self.num_tokens, -1)


def prepare_gt(
    state_seq: torch.Tensor,
    norm_stats: dict[str, torch.Tensor],
    num_latent_frames: int = 13,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Downsample state sequence and derive normalized state/action GT."""
    mean = norm_stats["mean"].to(state_seq.device, dtype=state_seq.dtype)
    std = norm_stats["std"].to(state_seq.device, dtype=state_seq.dtype)

    t = state_seq.shape[1]
    indices = torch.linspace(0, t - 1, num_latent_frames, device=state_seq.device).long()
    state_13 = state_seq[:, indices]
    state_gt_13 = (state_13 - mean) / std

    delta = state_13[:, 1:] - state_13[:, :-1]
    delta_norm = delta / std
    action_gt_13 = F.pad(delta_norm, (0, 0, 0, 1), value=0.0)
    s0_norm = state_gt_13[:, 0]
    return state_gt_13, action_gt_13, s0_norm


def compute_sa_loss(
    sa_output: torch.Tensor,
    state_tokenizer: StateActionTokenizer,
    state_gt_13: torch.Tensor,
    action_gt_13: torch.Tensor,
    lambda_s: float = 1.0,
    lambda_a: float = 1.0,
    lambda_c: float = 0.5,
) -> dict[str, torch.Tensor]:
    pred_state, pred_action = state_tokenizer.decode(sa_output)
    l_state = F.mse_loss(pred_state, state_gt_13)
    l_action = F.mse_loss(pred_action, action_gt_13)
    implied_delta = pred_state[:, 1:] - pred_state[:, :-1]
    predicted_delta = pred_action[:, :-1]
    l_consistency = F.mse_loss(implied_delta, predicted_delta)
    l_sa = lambda_s * l_state + lambda_a * l_action + lambda_c * l_consistency
    return {
        "L_state": l_state,
        "L_action": l_action,
        "L_consistency": l_consistency,
        "L_sa": l_sa,
    }


def save_state_action_modules(
    path: str,
    sa_tokenizer: StateActionTokenizer,
    s0_encoder: S0Encoder,
) -> None:
    torch.save(
        {
            "sa_tokenizer": sa_tokenizer.state_dict(),
            "s0_encoder": s0_encoder.state_dict(),
        },
        path,
    )


def load_state_action_modules(
    path: str,
    sa_tokenizer: StateActionTokenizer,
    s0_encoder: S0Encoder,
    device: torch.device | None = None,
) -> None:
    payload = torch.load(path, map_location=device or "cpu")
    sa_tokenizer.load_state_dict(payload["sa_tokenizer"])
    s0_encoder.load_state_dict(payload["s0_encoder"])
