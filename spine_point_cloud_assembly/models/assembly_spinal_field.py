# assembly_spinal_field.py
"""
Spine Assembly Transformer + Spinal Field Module (copy-paste ready)

What this adds on top of your baseline assembly.py
--------------------------------------------------
Baseline behavior:
- Set Transformer encoder (no absolute PE)
- [MASK] token for missing completion
- Heads: ordering logits, global pose (t, R via rot6d), masked embedding completion

NEW: "Spinal Field" modules (architectural inductive bias)
----------------------------------------------------------
(1) Global spine field token g (learned from set):
    - g = pool(encoder_outputs)  # (B, H)
    - used to condition all heads

(2) Continuous spine coordinate head s_i (per vertebra):
    - s: (B, N, 1)  # continuous position along the spine axis
    - helps disambiguate adjacent types (T12 vs T13, etc.)
    - you can add monotonic/spacing priors in loss (not in this file)

(3) Local neighbor delta-pose head (optional but recommended):
    - Predict relative pose between nearby vertebrae using attention-derived adjacency
    - Outputs:
        delta_pose:
          - d_t: (B, N, 3)   predicted delta translation to "next" vertebra in spine order
          - d_rot6d: (B, N, 6)
          - d_R: (B, N, 3, 3)
          - next_index: (B, N) long  # neighbor target index used for delta supervision (optional)
    - "next" is defined by sorting s (softly / discretely here: argsort, OK for training loss with stop-grad)

Inputs / Masks
--------------
- pad_mask: (B, N) bool, True where PAD (ignored by attention + pooling + losses)
- mask_mask: (B, N) bool, True where masked for completion (still attends, replaced by [MASK] token)

Outputs (dict)
--------------
{
  "ordering": (B, N, num_types+1),
  "pose": {"t": (B,N,3), "rot6d": (B,N,6), "R": (B,N,3,3)},
  "missing_completion": (B, N, embed_dim),

  # NEW:
  "spine_field": {
      "g": (B, hidden_dim),      # global spine field token
      "s": (B, N, 1),            # per-vertebra continuous spine coordinate
      "s_sorted_idx": (B, N),    # indices that sort s (PAD last)
  },
  "delta_pose": {                # neighbor relative pose (optional supervision)
      "d_t": (B, N, 3),
      "d_rot6d": (B, N, 6),
      "d_R": (B, N, 3, 3),
      "next_index": (B, N),      # neighbor index (PAD -> self)
  },

  "pad_mask": pad_mask,
  "mask_mask": mask_mask,
}

Notes
-----
- This file is purely model code. Losses are NOT implemented here.
- For s_i training: you can supervise with known vertebra ordering labels or add structural priors:
    * monotonicity: s should increase along GT order
    * smooth spacing: (s_{i+1}-s_i) close to mean spacing
- For delta_pose training: supervise (t_{next}-t_i) and (R_i^T R_next) if you have GT global poses.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Rotation: 6D -> SO(3)
# ---------------------------
def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix (Zhou et al.).
    Args:
        x: (..., 6)
    Returns:
        R: (..., 3, 3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2_ortho = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_ortho, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def masked_mean(x: torch.Tensor, pad_mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Mean over dim, ignoring pad positions where pad_mask=True.
    x: (B,N,...) if dim=1
    pad_mask: (B,N)
    """
    keep = (~pad_mask).float()  # 1 for valid
    while keep.dim() < x.dim():
        keep = keep.unsqueeze(-1)
    x_sum = (x * keep).sum(dim=dim)
    denom = keep.sum(dim=dim).clamp_min(1.0)
    return x_sum / denom


def bspline_basis(s: torch.Tensor, K: int, degree: int = 3) -> torch.Tensor:
    """
    Compute uniform open B-spline basis for s in [0,1].
    s: (B,N) or (N,) -> returns (B,N,K)
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)  # (1,N)
    B, N = s.shape
    device = s.device
    # Uniform open knot vector
    knots = torch.linspace(0.0, 1.0, K - degree + 1, device=device)
    knots = torch.cat([
        torch.zeros(degree, device=device),
        knots,
        torch.ones(degree, device=device),
    ])
    M = knots.numel()

    # Cox–de Boor recursion
    s_exp = s.unsqueeze(-1)  # (B,N,1)
    N_prev = torch.zeros(B, N, M - 1, device=device)
    for i in range(M - 1):
        left = knots[i]
        right = knots[i + 1]
        N_prev[:, :, i] = ((s_exp[..., 0] >= left) & (s_exp[..., 0] < right)).float()
    N_prev[:, :, -1] = (s_exp[..., 0] == knots[-1]).float()

    for d in range(1, degree + 1):
        N_curr = torch.zeros(B, N, M - d - 1, device=device)
        for i in range(M - d - 1):
            left_denom = knots[i + d] - knots[i]
            right_denom = knots[i + d + 1] - knots[i + 1]
            left = 0.0
            right = 0.0
            if left_denom > 0:
                left = ((s_exp[..., 0] - knots[i]) / left_denom) * N_prev[:, :, i]
            if right_denom > 0:
                right = ((knots[i + d + 1] - s_exp[..., 0]) / right_denom) * N_prev[:, :, i + 1]
            N_curr[:, :, i] = left + right
        N_prev = N_curr
    return N_prev  # (B,N,K)


def spline_points(control_points: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    control_points: (B,K,3), s: (B,N)
    Returns: (B,N,3)
    """
    B, K, _ = control_points.shape
    basis = bspline_basis(s, K)  # (B,N,K)
    return torch.einsum('bnk,bkc->bnc', basis, control_points)


def spline_tangent(control_points: torch.Tensor, s: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Finite-diff tangent of spline at s.
    """
    s0 = (s - eps).clamp(0.0, 1.0)
    s1 = (s + eps).clamp(0.0, 1.0)
    p0 = spline_points(control_points, s0)
    p1 = spline_points(control_points, s1)
    t = p1 - p0
    return F.normalize(t, dim=-1, eps=1e-6)


def tangent_frame(tangent: torch.Tensor) -> torch.Tensor:
    """
    Build a stable frame with z=tangent.
    Returns: (B,N,3,3) with columns [x, y, z].
    """
    device = tangent.device
    up1 = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)
    up2 = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 1, 3)
    dot = torch.abs((tangent * up1).sum(dim=-1, keepdim=True))
    up = torch.where(dot > 0.9, up2.expand_as(tangent), up1.expand_as(tangent))
    x = torch.cross(up, tangent, dim=-1)
    x = F.normalize(x, dim=-1, eps=1e-6)
    y = torch.cross(tangent, x, dim=-1)
    y = F.normalize(y, dim=-1, eps=1e-6)
    frame = torch.stack([x, y, tangent], dim=-1)  # (B,N,3,3)
    return frame


class SetTransformerEncoder(nn.Module):
    """
    Vanilla TransformerEncoder used as a set encoder:
    - No absolute positional encoding
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.enc(x, src_key_padding_mask=src_key_padding_mask)
        return self.out_norm(x)


class SpineAssemblySpinalField(nn.Module):
    """
    Baseline Assembly Transformer + Spinal Field modules.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_vertebra_types: int = 26,
        use_mask_token: bool = True,
        extra_dim: int = 0,  # if concatenating eq feats, set this
        enable_delta_pose: bool = True,
        num_control_points: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_vertebra_types
        self.use_mask_token = use_mask_token
        self.extra_dim = extra_dim
        self.enable_delta_pose = enable_delta_pose
        self.num_control_points = num_control_points

        in_dim = embed_dim + extra_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learned [MASK] token (applied on masked vertebrae only; not padding)
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

        # -------- Baseline heads --------
        self.ordering_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_vertebra_types + 1),
        )

        # Global pose head: 3 translation + 6D rotation
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 9),  # t(3) + rot6d(6)
        )

        # Completion head: predict original embedding (z_inv) for masked vertebrae
        self.completion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

        # -------- NEW: Spinal Field modules --------
        # Global spine field token g (pool + MLP)
        self.field_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-vertebra continuous coordinate s_i, conditioned on g
        self.s_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Condition all heads on g (small FiLM-like adapter)
        self.cond = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # scale, shift
        )

        # Control points for B-spline spine curve (global, conditioned on g)
        self.control_points_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_control_points * 3),
        )

        # -------- NEW: neighbor delta pose head --------
        if self.enable_delta_pose:
            self.delta_pose_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 9),  # d_t(3) + d_rot6d(6)
            )

    def _apply_conditioning(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        FiLM conditioning: x' = x * (1 + scale) + shift
        x: (B,N,H), g: (B,H)
        """
        B, N, H = x.shape
        ss = self.cond(g)  # (B, 2H)
        scale, shift = ss[:, :H], ss[:, H:]
        scale = scale.unsqueeze(1)  # (B,1,H)
        shift = shift.unsqueeze(1)
        return x * (1.0 + scale) + shift

    @staticmethod
    def _argsort_with_pad_last(s: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        s: (B,N,1), pad_mask: (B,N)
        Return indices sorting s ascending, with PAD tokens forced to the end.
        """
        B, N, _ = s.shape
        s_flat = s.squeeze(-1)  # (B,N)
        # Large value for PAD so they go last
        big = torch.finfo(s_flat.dtype).max / 4.0
        s_masked = torch.where(pad_mask, torch.full_like(s_flat, big), s_flat)
        return torch.argsort(s_masked, dim=1)  # (B,N)

    @staticmethod
    def _next_index_from_sorted(sorted_idx: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Define a discrete "next" neighbor by s-sorting order.
        sorted_idx: (B,N) indices sorted by s (PAD last)
        Returns next_index: (B,N) in original index space.
          - For the last valid element, next_index = self
          - For PAD positions, next_index = self
        """
        B, N = sorted_idx.shape
        device = sorted_idx.device

        # inverse permutation: inv[sorted_pos] = original_idx, but we want rank per original idx
        # rank[original_idx] = sorted_pos
        rank = torch.empty_like(sorted_idx)
        ar = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        rank.scatter_(1, sorted_idx, ar)  # rank[b, orig_idx] = position in sorted order

        # determine which positions are valid (non-pad)
        valid = ~pad_mask  # (B,N)

        # For each original idx, its next rank is rank+1 (if still valid), else itself
        next_rank = rank + 1
        next_rank = torch.clamp(next_rank, max=N - 1)

        # Map next_rank back to original index using sorted_idx
        next_idx = torch.gather(sorted_idx, 1, next_rank)  # (B,N) original index at next rank

        # If this token is PAD -> next=self
        self_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        next_idx = torch.where(valid, next_idx, self_idx)

        # If this token is last valid (its next is PAD), set next=self
        next_is_pad = torch.gather(pad_mask, 1, next_rank)  # (B,N) whether next rank is PAD
        next_idx = torch.where(next_is_pad, self_idx, next_idx)

        return next_idx

    def forward(
        self,
        embeddings: torch.Tensor,                 # (B,N,D) invariant embedding (z_inv)
        pad_mask: Optional[torch.Tensor] = None,  # (B,N) True=PAD
        mask_mask: Optional[torch.Tensor] = None, # (B,N) True=MASK (predict)
        eq_feats: Optional[torch.Tensor] = None,  # (B,N,E) or (B,N,K,3)
    ) -> Dict[str, torch.Tensor]:
        B, N, D = embeddings.shape
        device = embeddings.device

        if pad_mask is None:
            pad_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if mask_mask is None:
            mask_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        # Build input features
        if eq_feats is not None:
            if eq_feats.dim() == 4:
                eq_feats = eq_feats.reshape(B, N, -1)  # (B,N,K*3)
            x_in = torch.cat([embeddings, eq_feats], dim=-1)  # (B,N,D+E)
        else:
            x_in = embeddings  # (B,N,D)

        # Apply [MASK] token on masked positions (but NOT on padding)
        if self.use_mask_token and self.mask_token is not None:
            mask_tok = self.mask_token.expand(B, N, -1)  # (B,N,in_dim)
            apply_mask = mask_mask & (~pad_mask)
            x_in = torch.where(apply_mask.unsqueeze(-1), mask_tok, x_in)

        # Project to hidden
        x = self.input_proj(x_in)  # (B,N,H)

        # Transformer encoder: src_key_padding_mask expects True=ignore
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,N,H)

        # -------- NEW: global spine field token g --------
        x_pool = masked_mean(x, pad_mask=pad_mask, dim=1)  # (B,H)
        g = self.field_pool(x_pool)  # (B,H)

        # Condition token features on g (helps adjacent disambiguation)
        x = self._apply_conditioning(x, g)  # (B,N,H)

        # -------- NEW: continuous spine coordinate s_i --------
        g_expand = g.unsqueeze(1).expand(B, N, self.hidden_dim)
        s_in = torch.cat([x, g_expand], dim=-1)  # (B,N,2H)
        s_raw = self.s_head(s_in)  # (B,N,1)
        s = torch.sigmoid(s_raw)   # (B,N,1) in [0,1]

        # We often want sorted indices (PAD last) for evaluation / neighbor delta supervision
        s_sorted_idx = self._argsort_with_pad_last(s, pad_mask)  # (B,N)

        # -------- Baseline heads (conditioned tokens) --------
        ordering_logits = self.ordering_head(x)  # (B,N,num_types+1)

        pose_raw = self.pose_head(x)  # (B,N,9)
        offset_local = pose_raw[..., 0:3]  # local offsets in spline frame
        rot6d_local = pose_raw[..., 3:9]
        rot6d = rot6d_local
        R_local = rot6d_to_matrix(rot6d_local)  # (B,N,3,3)

        pred_embedding = self.completion_head(x)  # (B,N,D)

        # -------- NEW: global B-spline control points --------
        control_points = self.control_points_head(g)  # (B, K*3)
        control_points = control_points.view(B, self.num_control_points, 3)

        # -------- NEW: spline frame + global pose --------
        s_coord = s.squeeze(-1)
        centerline = spline_points(control_points, s_coord)  # (B,N,3)
        tangent = spline_tangent(control_points, s_coord)    # (B,N,3)
        frame = tangent_frame(tangent)                       # (B,N,3,3)
        # Transform local offsets to world
        t = centerline + torch.einsum('bnij,bnj->bni', frame, offset_local)
        # Rotate local -> world
        R = torch.einsum('bnij,bnjk->bnik', frame, R_local)

        # -------- NEW: neighbor delta pose --------
        if self.enable_delta_pose:
            next_index = self._next_index_from_sorted(s_sorted_idx, pad_mask)  # (B,N)

            # gather neighbor token features
            # x_neighbor[b, i] = x[b, next_index[b,i]]
            x_neighbor = torch.gather(
                x,
                dim=1,
                index=next_index.unsqueeze(-1).expand(B, N, self.hidden_dim),
            )

            dp_in = torch.cat([x, x_neighbor], dim=-1)  # (B,N,2H)
            dp_raw = self.delta_pose_head(dp_in)  # (B,N,9)
            d_t = dp_raw[..., 0:3]
            d_rot6d = dp_raw[..., 3:9]
            d_R = rot6d_to_matrix(d_rot6d)
            delta_pose = {
                "d_t": d_t,
                "d_rot6d": d_rot6d,
                "d_R": d_R,
                "next_index": next_index,
            }
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

            # NEW:
            "spine_field": {
                "g": g,
                "s": s,
                "s_raw": s_raw,
                "s_sorted_idx": s_sorted_idx,
                "control_points": control_points,
                "centerline": centerline,
                "tangent": tangent,
                "frame": frame,
            },
            "delta_pose": delta_pose,

            "pad_mask": pad_mask,
            "mask_mask": mask_mask,
            "pose_local": {
                "offset_local": offset_local,
                "rot6d_local": rot6d_local,
                "R_local": R_local,
            },
        }

