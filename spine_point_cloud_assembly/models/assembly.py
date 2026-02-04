"""
Spine Assembly Transformer (Revised, copy-paste ready)

Key fixes vs your version
- NO absolute positional encoding (input is a set / permutation-invariant)
- Supports [MASK] tokens for missing completion (masked vertebrae stay in attention)
- Separates PAD vs MASK:
    pad_mask: True where padding (ignore in attention + pooling)
    mask_mask: True where a vertebra is "masked out" for completion task (still attends)
- Assembly head is well-defined: GLOBAL pose per vertebra (t_i, R_i)
- Rotation uses 6D representation (stable) -> rotation matrix (B, N, 3, 3)
- Optional: you can pass equivariant frame features from encoder as extra inputs
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


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
    # make a2 orthogonal to b1
    a2_ortho = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_ortho, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3) with columns [b1 b2 b3]
    return R


def geodesic_distance(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance on SO(3): acos((trace(Rp^T Rg)-1)/2)
    Args:
        R_pred, R_gt: (..., 3, 3)
    Returns:
        angle: (...) in radians
    """
    RtR = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    tr = RtR[..., 0, 0] + RtR[..., 1, 1] + RtR[..., 2, 2]
    cos = (tr - 1.0) / 2.0
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


# ---------------------------
# Transformer blocks
# ---------------------------
class SetTransformerEncoder(nn.Module):
    """
    Vanilla TransformerEncoder used as a set encoder:
    - No absolute positional encoding
    - Optional learned "type embedding" if you pass token_type_ids
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
            norm_first=True,  # more stable
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.enc(x, src_key_padding_mask=src_key_padding_mask)
        return self.out_norm(x)


# ---------------------------
# Main Assembly
# ---------------------------
class SpineAssemblyTransformer(nn.Module):
    """
    Inputs:
      embeddings: (B, N, D)  -> invariant embedding from encoder (z_inv)
      eq_feats:   optional (B, N, K, 3) or (B, N, E) -> flattened & concatenated

      pad_mask:  (B, N) bool, True where padding token (ignore attention/pooling/loss)
      mask_mask: (B, N) bool, True where token is masked for completion (still attends)

    Outputs:
      ordering_logits: (B, N, num_types+1)  # +1 can be "unknown/masked"
      pose:
        t: (B, N, 3)
        R: (B, N, 3, 3)  # global rotation matrix
        rot6d: (B, N, 6) # raw 6D
      completion:
        pred_embedding: (B, N, D)  # predicted embedding, train it on mask_mask positions
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
        extra_dim: int = 0,  # if you concatenate eq features, set this
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_types = num_vertebra_types
        self.use_mask_token = use_mask_token
        self.extra_dim = extra_dim

        in_dim = embed_dim + extra_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learned mask token (applied on masked vertebrae only; not padding)
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

        # Heads
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

    def forward(
        self,
        embeddings: torch.Tensor,                 # (B, N, D)
        pad_mask: Optional[torch.Tensor] = None,  # (B, N) True=PAD (ignore)
        mask_mask: Optional[torch.Tensor] = None, # (B, N) True=MASK (predict)
        eq_feats: Optional[torch.Tensor] = None,  # (B, N, E) OR (B,N,K,3)
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
                # (B,N,K,3) -> (B,N, K*3)
                eq_feats = eq_feats.reshape(B, N, -1)
            x_in = torch.cat([embeddings, eq_feats], dim=-1)  # (B,N, D+E)
        else:
            x_in = embeddings  # (B,N,D)

        # Apply [MASK] token on masked positions (but NOT on padding)
        if self.use_mask_token and self.mask_token is not None:
            # Expand mask token to batch and length
            mask_tok = self.mask_token.expand(B, N, -1)  # (B,N,in_dim)
            apply_mask = mask_mask & (~pad_mask)
            x_in = torch.where(apply_mask.unsqueeze(-1), mask_tok, x_in)

        # Project to hidden
        x = self.input_proj(x_in)  # (B,N,H)

        # Transformer encoder: src_key_padding_mask expects True=ignore
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B,N,H)

        # Heads
        ordering_logits = self.ordering_head(x)  # (B,N,num_types+1)

        pose_raw = self.pose_head(x)  # (B,N,9)
        t = pose_raw[..., 0:3]
        rot6d = pose_raw[..., 3:9]
        R = rot6d_to_matrix(rot6d)  # (B,N,3,3)

        pred_embedding = self.completion_head(x)  # (B,N,D)

        return {
            "ordering": ordering_logits,
            "pose": {
                "t": t,
                "R": R,
                "rot6d": rot6d,
            },
            "missing_completion": pred_embedding,
            "pad_mask": pad_mask,
            "mask_mask": mask_mask,
        }


