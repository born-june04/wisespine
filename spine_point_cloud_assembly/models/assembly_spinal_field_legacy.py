"""
Legacy Spinal Field model (checkpoint-compatible).

Why this exists:
- Some older checkpoints (e.g., 2026-01-12_21-07-15) do NOT contain B-spline centerline modules
  (no `control_points_head`, no `cond`, etc.).
- The current `assembly_spinal_field.py` forward uses spline/frame to produce global pose.
  If we instantiate the new model but load an old checkpoint with strict=False, spline modules remain random,
  and inference outputs become meaningless.

This legacy model matches the *state_dict keys* observed in those checkpoints:
- input_proj.*
- encoder.enc.layers.*
- field_pool.*
- s_head.*
- ordering_head.*, pose_head.*, completion_head.*
- (optional) delta_pose_head.*
- mask_token
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assembly_spinal_field import (  # reuse stable utilities
    SetTransformerEncoder,
    masked_mean,
    rot6d_to_matrix,
)


class SpineAssemblySpinalFieldLegacy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_vertebra_types: int = 26,
        use_mask_token: bool = True,
        enable_delta_pose: bool = True,
        extra_dim: int = 0,
        enable_cond: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_vertebra_types
        self.use_mask_token = use_mask_token
        self.enable_delta_pose = enable_delta_pose
        self.extra_dim = extra_dim
        self.enable_cond = enable_cond

        in_dim = embed_dim + extra_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, in_dim))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None

        self.encoder = SetTransformerEncoder(
            d_model=hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Global spine field token g
        self.field_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Optional FiLM-like conditioning (some checkpoints have this, but no spline)
        if self.enable_cond:
            self.cond = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),  # scale, shift
            )
        else:
            self.cond = None

        # Continuous coordinate s_i, conditioned on g
        self.s_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Heads
        self.ordering_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_vertebra_types + 1),
        )

        # Pose head: directly predicts global translation + rot6d (NO spline/frame)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 9),
        )

        self.completion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

        if self.enable_delta_pose:
            self.delta_pose_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 9),
            )

    def _apply_conditioning(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        FiLM conditioning: x' = x * (1 + scale) + shift
        """
        if not self.enable_cond or self.cond is None:
            return x
        B, N, H = x.shape
        ss = self.cond(g)  # (B,2H)
        scale, shift = ss[:, :H], ss[:, H:]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return x * (1.0 + scale) + shift

    @staticmethod
    def _argsort_with_pad_last(s: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # s: (B,N,1)
        s2 = s.squeeze(-1)  # (B,N)
        big = torch.tensor(10.0, device=s.device, dtype=s.dtype)
        s2 = torch.where(pad_mask, big, s2)
        return torch.argsort(s2, dim=1)

    @staticmethod
    def _next_index_from_sorted(s_sorted_idx: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        For each token i, define next as the token with the next larger s (PAD last).
        If i is last valid, next=self.
        """
        B, N = s_sorted_idx.shape
        device = s_sorted_idx.device
        rank = torch.zeros(B, N, dtype=torch.long, device=device)
        rank.scatter_(1, s_sorted_idx, torch.arange(N, device=device).view(1, N).expand(B, N))
        next_rank = (rank + 1).clamp(max=N - 1)
        next_idx = torch.gather(s_sorted_idx, 1, next_rank)
        self_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
        next_is_pad = torch.gather(pad_mask, 1, next_rank)
        next_idx = torch.where(next_is_pad, self_idx, next_idx)
        return next_idx

    def forward(
        self,
        embeddings: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        mask_mask: Optional[torch.Tensor] = None,
        eq_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, D = embeddings.shape
        device = embeddings.device
        if pad_mask is None:
            pad_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if mask_mask is None:
            mask_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        # Input
        if eq_feats is not None:
            if eq_feats.dim() == 4:
                eq_feats = eq_feats.reshape(B, N, -1)
            x_in = torch.cat([embeddings, eq_feats], dim=-1)
        else:
            x_in = embeddings

        if self.use_mask_token and self.mask_token is not None:
            mask_tok = self.mask_token.expand(B, N, -1)
            apply_mask = mask_mask & (~pad_mask)
            x_in = torch.where(apply_mask.unsqueeze(-1), mask_tok, x_in)

        x = self.input_proj(x_in)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # g
        x_pool = masked_mean(x, pad_mask=pad_mask, dim=1)
        g = self.field_pool(x_pool)

        # condition tokens on g (if enabled)
        x = self._apply_conditioning(x, g)

        # s
        g_expand = g.unsqueeze(1).expand(B, N, self.hidden_dim)
        s_in = torch.cat([x, g_expand], dim=-1)
        s = torch.sigmoid(self.s_head(s_in))
        s_sorted_idx = self._argsort_with_pad_last(s, pad_mask)

        ordering_logits = self.ordering_head(x)

        pose_raw = self.pose_head(x)
        t = pose_raw[..., 0:3]
        rot6d = pose_raw[..., 3:9]
        R = rot6d_to_matrix(rot6d)

        pred_embedding = self.completion_head(x)

        if self.enable_delta_pose:
            next_index = self._next_index_from_sorted(s_sorted_idx, pad_mask)
            x_neighbor = torch.gather(
                x, dim=1, index=next_index.unsqueeze(-1).expand(B, N, self.hidden_dim)
            )
            dp_in = torch.cat([x, x_neighbor], dim=-1)
            dp_raw = self.delta_pose_head(dp_in)
            d_t = dp_raw[..., 0:3]
            d_rot6d = dp_raw[..., 3:9]
            d_R = rot6d_to_matrix(d_rot6d)
            delta_pose = {"d_t": d_t, "d_rot6d": d_rot6d, "d_R": d_R, "next_index": next_index}
        else:
            delta_pose = {
                "d_t": torch.zeros(B, N, 3, device=device, dtype=x.dtype),
                "d_rot6d": torch.zeros(B, N, 6, device=device, dtype=x.dtype),
                "d_R": torch.eye(3, device=device, dtype=x.dtype).view(1, 1, 3, 3).expand(B, N, 3, 3),
                "next_index": torch.arange(N, device=device).view(1, N).expand(B, N),
            }

        return {
            "ordering": ordering_logits,
            "pose": {"t": t, "rot6d": rot6d, "R": R},
            "missing_completion": pred_embedding,
            "spine_field": {"g": g, "s": s, "s_sorted_idx": s_sorted_idx},
            "delta_pose": delta_pose,
            "pad_mask": pad_mask,
            "mask_mask": mask_mask,
        }


