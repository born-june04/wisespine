"""
Position-Aware Point Transformer Encoder for Vertebra Point Clouds

Architecture:
1. Input normalization (centering + unit sphere scaling)
2. Separate embedding for scalar features and vector features (normals)
3. kNN graph-based sparse attention (O(kN) instead of O(N²))
4. Attention-based global pooling
5. Projection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
try:
    from torch_geometric.nn import knn_graph
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def normalize_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Normalize points: center by centroid and scale to unit sphere
    
    Args:
        points: (B, N, 3) - point positions
    Returns:
        points_normalized: (B, N, 3) - normalized points
        centroid: (B, 3) - centroids
        scale: (B,) - scale factors
    """
    B, N, _ = points.shape
    
    # Center by centroid
    centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
    points_centered = points - centroid
    
    # Scale to unit sphere (max distance from origin = 1)
    distances = torch.norm(points_centered, dim=-1)  # (B, N)
    max_dist = distances.max(dim=1, keepdim=True)[0]  # (B, 1)
    max_dist = torch.clamp(max_dist, min=1e-6)  # Avoid division by zero
    scale = 1.0 / max_dist  # (B, 1)
    
    points_normalized = points_centered * scale.unsqueeze(-1)  # (B, N, 3)
    
    return points_normalized, centroid.squeeze(1), scale.squeeze(1)


class ScalarFeatureEmbedding(nn.Module):
    """Embedding for scalar features (curvature, etc.)"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) - scalar features
        Returns:
            (B, N, hidden_dim) - embedded features
        """
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        x = self.mlp(x)
        x = x.view(B, N, -1)
        return x


class VectorFeatureEmbedding(nn.Module):
    """Embedding for vector features (normals) - preserves direction"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Process vector magnitude and direction separately
        self.magnitude_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.direction_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) - vector features (e.g., normals)
        Returns:
            (B, N, hidden_dim) - embedded features
        """
        B, N, _ = x.shape
        
        # Magnitude
        magnitude = torch.norm(x, dim=-1, keepdim=True)  # (B, N, 1)
        mag_feat = self.magnitude_mlp(magnitude)  # (B, N, hidden_dim // 2)
        
        # Direction (normalized)
        direction = F.normalize(x, dim=-1, p=2, eps=1e-8)  # (B, N, input_dim)
        dir_feat = self.direction_mlp(direction)  # (B, N, hidden_dim // 2)
        
        # Concatenate and fuse
        combined = torch.cat([mag_feat, dir_feat], dim=-1)  # (B, N, hidden_dim)
        output = self.fusion(combined)  # (B, N, hidden_dim)
        
        return output


class SparseAttentionBlock(nn.Module):
    """kNN graph-based sparse attention block"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, k: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.k = k
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Self-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Position encoding (relative position in kNN graph)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
    
    def build_knn_graph(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build kNN graph
        
        Args:
            pos: (B, N, 3) - point positions
        Returns:
            edge_index: (2, E) - edge indices
            rel_pos: (E, 3) - relative positions
        """
        B, N, _ = pos.shape
        
        if HAS_TORCH_GEOMETRIC:
            # Use torch_geometric for efficient kNN
            pos_flat = pos.view(B * N, 3)  # (B*N, 3)
            batch_idx = torch.arange(B, device=pos.device).repeat_interleave(N)  # (B*N,)
            
            edge_index = knn_graph(pos_flat, k=self.k, batch=batch_idx, loop=False)  # (2, E)
            
            # Get relative positions
            row, col = edge_index
            rel_pos = pos_flat[col] - pos_flat[row]  # (E, 3)
            
            return edge_index, rel_pos
        else:
            # Fallback: manual kNN computation
            pos_flat = pos.view(B * N, 3)  # (B*N, 3)
            
            # Compute pairwise distances
            dist = torch.cdist(pos_flat, pos_flat)  # (B*N, B*N)
            
            # Get k nearest neighbors (excluding self)
            _, topk_indices = torch.topk(dist, k=self.k + 1, dim=1, largest=False)  # (B*N, k+1)
            topk_indices = topk_indices[:, 1:]  # Remove self (first column)  # (B*N, k)
            
            # Build edge index
            row = torch.arange(B * N, device=pos.device).repeat_interleave(self.k)  # (B*N*k,)
            col = topk_indices.flatten()  # (B*N*k,)
            edge_index = torch.stack([row, col], dim=0)  # (2, E)
            
            # Get relative positions
            rel_pos = pos_flat[col] - pos_flat[row]  # (E, 3)
            
            return edge_index, rel_pos
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim) - point features
            pos: (B, N, 3) - point positions (normalized)
        Returns:
            (B, N, hidden_dim) - updated features
        """
        B, N, _ = x.shape
        
        # Build kNN graph
        edge_index, rel_pos = self.build_knn_graph(pos)  # (2, E), (E, 3)
        E = rel_pos.shape[0]
        
        # Reshape for batch processing
        x_flat = x.view(B * N, self.hidden_dim)  # (B*N, hidden_dim)
        
        # Normalize
        x_norm = self.norm1(x_flat)  # (B*N, hidden_dim)
        x_norm = x_norm.view(B, N, self.hidden_dim)
        
        # Multi-head attention on kNN graph
        q = self.q_proj(x_norm)  # (B, N, hidden_dim)
        k = self.k_proj(x_norm)  # (B, N, hidden_dim)
        v = self.v_proj(x_norm)  # (B, N, hidden_dim)
        
        # Reshape for graph attention
        q_flat = q.view(B * N, self.num_heads, self.head_dim)  # (B*N, H, D)
        k_flat = k.view(B * N, self.num_heads, self.head_dim)  # (B*N, H, D)
        v_flat = v.view(B * N, self.num_heads, self.head_dim)  # (B*N, H, D)
        
        # Get source and target nodes
        row, col = edge_index  # (E,), (E,)
        q_edge = q_flat[row]  # (E, H, D)
        k_edge = k_flat[col]  # (E, H, D)
        v_edge = v_flat[col]  # (E, H, D)
        
        # Attention scores with relative position encoding
        scores = torch.sum(q_edge * k_edge, dim=-1) / np.sqrt(self.head_dim)  # (E, H)
        
        # Add relative position bias
        pos_enc = self.pos_mlp(rel_pos)  # (E, hidden_dim)
        pos_enc = pos_enc.view(E, self.num_heads, self.head_dim)  # (E, H, D)
        pos_bias = torch.sum(q_edge * pos_enc, dim=-1)  # (E, H)
        scores = scores + pos_bias
        
        # Softmax over neighbors for each source node
        # Group by source node (row) and apply softmax per node
        # For numerical stability, compute max per node
        # Simple approach: iterate over unique nodes (not efficient but compatible)
        max_scores_per_node = torch.full((B * N, self.num_heads), float('-inf'), device=x.device, dtype=x.dtype)
        
        # Compute max score per node
        for node_idx in range(B * N):
            mask = (row == node_idx)
            if mask.any():
                max_scores_per_node[node_idx] = scores[mask].max(dim=0)[0]
        
        # For each edge, get the max score of its source node
        max_scores = max_scores_per_node[row]  # (E, H)
        
        # Compute exp(scores - max) for numerical stability
        scores_exp = torch.exp(scores - max_scores)  # (E, H)
        
        # Sum exp scores per node
        scores_sum = torch.zeros(B * N, self.num_heads, device=x.device, dtype=x.dtype)
        scores_sum = scores_sum.index_add_(0, row, scores_exp)  # (B*N, H)
        scores_sum = torch.clamp(scores_sum, min=1e-8)  # Avoid division by zero
        
        # Normalize: attention weights per node
        attn_weights = scores_exp / scores_sum[row]  # (E, H)
        
        # Aggregate: weighted sum over neighbors
        attn_out = attn_weights.unsqueeze(-1) * v_edge  # (E, H, D)
        out_flat = torch.zeros(B * N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out_flat = out_flat.index_add_(0, row, attn_out)  # (B*N, H, D)
        
        # Reshape and project
        out_flat = out_flat.view(B * N, self.hidden_dim)  # (B*N, hidden_dim)
        out_flat = self.out_proj(out_flat)  # (B*N, hidden_dim)
        out = out_flat.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)
        
        # Residual connection
        x = x + out
        
        # FFN
        x_flat = x.view(B * N, self.hidden_dim)
        residual_flat = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.ffn(x_flat)
        x_flat = residual_flat + x_flat
        x = x_flat.view(B, N, self.hidden_dim)
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based global pooling (preserves orientation information)"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.scale = np.sqrt(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim) - point features
        Returns:
            (B, hidden_dim) - global embedding
        """
        B, N, _ = x.shape
        
        # Compute attention weights
        query = self.query.expand(B, -1, -1)  # (B, 1, hidden_dim)
        scores = torch.bmm(query, x.transpose(1, 2)) / self.scale  # (B, 1, N)
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, N)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights, x)  # (B, 1, hidden_dim)
        pooled = pooled.squeeze(1)  # (B, hidden_dim)
        
        return pooled


class VertebraPointEncoder(nn.Module):
    """
    Position-Aware Point Transformer Encoder with kNN Graph Attention
    
    Key improvements:
    1. Input normalization (centering + unit sphere scaling)
    2. Separate processing for scalar and vector features
    3. kNN graph-based sparse attention (O(kN) memory)
    4. Attention-based pooling (preserves orientation)
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # xyz(3) + normals(3) + curvature(2)
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        output_dim: int = 512,
        k: int = 16,  # kNN graph parameter
        use_curvature: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_curvature = use_curvature
        self.k = k
        
        # Separate embeddings for different feature types
        # Scalar features: curvature (2 dims)
        scalar_dim = 2 if use_curvature else 0
        if scalar_dim > 0:
            self.scalar_embedding = ScalarFeatureEmbedding(scalar_dim, hidden_dim)
        else:
            self.scalar_embedding = None
        
        # Vector features: normals (3 dims)
        self.vector_embedding = VectorFeatureEmbedding(3, hidden_dim)
        
        # Position embedding (for points themselves)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Feature fusion
        num_feat_types = 1 + (1 if scalar_dim > 0 else 0)  # vector + (scalar if exists)
        self.fusion = nn.Linear(hidden_dim * num_feat_types, hidden_dim)
        
        # Sparse attention blocks
        self.attention_blocks = nn.ModuleList([
            SparseAttentionBlock(hidden_dim, num_heads, k=k)
            for _ in range(num_layers)
        ])
        
        # Attention-based global pooling
        self.pooling = AttentionPooling(hidden_dim)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) - point positions (will be normalized internally)
            features: (B, N, F) - point features [normals(3), curvature(2), ...]
                      If None, only uses positions
        Returns:
            (B, output_dim) - global vertebra embedding
        """
        B, N, _ = points.shape
        
        # Normalize points: center and scale to unit sphere
        points_norm, centroid, scale = normalize_points(points)  # (B, N, 3), (B, 3), (B,)
        
        # Separate features
        if features is not None:
            normals = features[:, :, :3]  # (B, N, 3) - vector features
            if self.use_curvature and features.shape[-1] >= 5:
                curvature = features[:, :, 3:5]  # (B, N, 2) - scalar features
            else:
                curvature = None
        else:
            normals = None
            curvature = None
        
        # Embed features
        feat_list = []
        
        # Position embedding
        pos_feat = self.pos_embedding(points_norm)  # (B, N, hidden_dim)
        feat_list.append(pos_feat)
        
        # Vector feature embedding (normals)
        if normals is not None:
            # Normalize normals to unit length
            normals_norm = F.normalize(normals, dim=-1, p=2, eps=1e-8)
            vec_feat = self.vector_embedding(normals_norm)  # (B, N, hidden_dim)
            feat_list.append(vec_feat)
        
        # Scalar feature embedding (curvature)
        if curvature is not None and self.scalar_embedding is not None:
            scalar_feat = self.scalar_embedding(curvature)  # (B, N, hidden_dim)
            feat_list.append(scalar_feat)
        
        # Fuse features
        if len(feat_list) > 1:
            x = torch.cat(feat_list, dim=-1)  # (B, N, hidden_dim * num_types)
            x = self.fusion(x)  # (B, N, hidden_dim)
        else:
            x = feat_list[0]
        
        # Sparse attention blocks
        for block in self.attention_blocks:
            x = block(x, points_norm)  # (B, N, hidden_dim)
        
        # Attention-based global pooling
        x_pooled = self.pooling(x)  # (B, hidden_dim)
        
        # Projection
        x_proj = self.projection(x_pooled)  # (B, output_dim)
        
        return x_proj
    
    def encode_points(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode points and return both point-level and global embeddings
        
        Args:
            points: (B, N, 3) - point positions
            features: (B, N, F) - point features
        Returns:
            point_embeddings: (B, N, hidden_dim) - point-level features
            global_embedding: (B, output_dim) - global embedding
        """
        B, N, _ = points.shape
        
        # Normalize points
        points_norm, _, _ = normalize_points(points)
        
        # Separate and embed features (same as forward)
        if features is not None:
            normals = features[:, :, :3]
            if self.use_curvature and features.shape[-1] >= 5:
                curvature = features[:, :, 3:5]
            else:
                curvature = None
        else:
            normals = None
            curvature = None
        
        feat_list = []
        pos_feat = self.pos_embedding(points_norm)
        feat_list.append(pos_feat)
        
        if normals is not None:
            normals_norm = F.normalize(normals, dim=-1, p=2, eps=1e-8)
            vec_feat = self.vector_embedding(normals_norm)
            feat_list.append(vec_feat)
        
        if curvature is not None and self.scalar_embedding is not None:
            scalar_feat = self.scalar_embedding(curvature)
            feat_list.append(scalar_feat)
        
        if len(feat_list) > 1:
            x = torch.cat(feat_list, dim=-1)
            x = self.fusion(x)
        else:
            x = feat_list[0]
        
        # Attention blocks
        for block in self.attention_blocks:
            x = block(x, points_norm)
        
        # Point-level embeddings
        point_embeddings = x  # (B, N, hidden_dim)
        
        # Global pooling
        x_pooled = self.pooling(x)
        global_embedding = self.projection(x_pooled)
        
        return point_embeddings, global_embedding
