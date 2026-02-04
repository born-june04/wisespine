"""
SE(3)-Equivariant Point Encoder for Vertebra Point Clouds (SE(3)-Transformer based)

This is a wrapper around SE(3)-Transformer (Fabian Fuchs et al.) that maintains
compatibility with the existing encoder_se3.py interface.

Architecture:
1. Input normalization (centering + unit sphere scaling)
2. Feature conversion to Fiber format
3. kNN graph construction (DGL format)
4. SE(3)-equivariant message passing (SE(3)-Transformer)
5. Invariant + Equivariant output separation
6. Projection head

Based on: https://github.com/FabianFuchsML/se3-transformer-public
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from torch_scatter import scatter_mean

# SE(3)-Transformer imports
try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    print("WARNING: DGL not installed. SE(3)-Transformer requires DGL.")
    print("Install with: pip install dgl")

try:
    from .se3_transformer.modules import (
        GSE3Res, GNormSE3, GConvSE3, get_basis_and_r, 
        GMaxPooling, GAvgPooling, G1x1SE3
    )
    from .se3_transformer.fibers import Fiber
except ImportError:
    # Fallback for absolute import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from se3_transformer.modules import (
        GSE3Res, GNormSE3, GConvSE3, get_basis_and_r, 
        GMaxPooling, GAvgPooling, G1x1SE3
    )
    from se3_transformer.fibers import Fiber

try:
    from torch_cluster import knn_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    try:
        from torch_geometric.nn import knn_graph
        HAS_TORCH_CLUSTER = True
    except ImportError:
        HAS_TORCH_CLUSTER = False


def normalize_points_batch(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Batch-aware normalization: normalize each vertebra separately
    
    CRITICAL: This preserves batch information - each vertebra is normalized
    independently. Without this, different vertebrae share the same centroid/scale,
    causing information loss before assembly.
    
    Args:
        pos: (N, 3) - flattened point positions
        batch: (N,) - batch indices for each point
    Returns:
        pos_norm: (N, 3) - normalized points (each batch normalized separately)
    """
    pos_norm = pos.clone()
    for b in batch.unique():
        idx = (batch == b)
        pts = pos[idx]
        centroid = pts.mean(0, keepdim=True)
        pts = pts - centroid
        scale = torch.norm(pts, dim=1).max().clamp(min=1e-6)
        pos_norm[idx] = pts / scale
    return pos_norm


def features_to_fiber(features: torch.Tensor, use_curvature: bool = True) -> Dict[str, torch.Tensor]:
    """
    Convert features tensor to SE(3)-Transformer Fiber format
    
    Args:
        features: (B, N, 5) or (N, 5) - [normals(3), curvature(2)]
                  or (B, N, 3) or (N, 3) - [normals(3)] if no curvature
        use_curvature: whether to use curvature features
    Returns:
        fiber_dict: dict with keys '0' (scalars) and '1' (vectors)
    """
    # Handle batch dimension
    is_batch = features.dim() == 3
    if is_batch:
        B, N, feat_dim = features.shape
        features_flat = features.view(B * N, feat_dim)
    else:
        N, feat_dim = features.shape
        features_flat = features
        B = 1
    
    # Separate features
    if use_curvature and feat_dim >= 5:
        normals = features_flat[:, :3]  # (B*N, 3) or (N, 3)
        curvature = features_flat[:, 3:5]  # (B*N, 2) or (N, 2)
    else:
        normals = features_flat[:, :3]  # (B*N, 3) or (N, 3)
        curvature = None
    
    # Normalize normals
    normals = F.normalize(normals, dim=-1, p=2, eps=1e-8)
    
    # Prepare fiber dict
    # '0': scalar features (curvature)
    # '1': vector features (normals)
    fiber_dict = {}
    
    if curvature is not None:
        # Scalar features: (N, 2) -> (N, 2, 1) for degree 0
        fiber_dict['0'] = curvature.unsqueeze(-1)  # (N, 2, 1)
    else:
        # Dummy scalar feature
        fiber_dict['0'] = torch.ones(features_flat.shape[0], 1, 1, device=features_flat.device)
    
    # Vector features: (N, 3) -> (N, 1, 3) for degree 1
    fiber_dict['1'] = normals.unsqueeze(1)  # (N, 1, 3)
    
    return fiber_dict


def build_dgl_graph(pos: torch.Tensor, batch: torch.Tensor, k: int = 16) -> dgl.DGLGraph:
    """
    Build DGL graph from point cloud using kNN
    
    Args:
        pos: (N, 3) - point positions
        batch: (N,) - batch indices
        k: number of nearest neighbors
    Returns:
        G: DGL graph with node features 'x' and edge features 'd'
    """
    if not HAS_DGL:
        raise ImportError("DGL required for SE(3)-Transformer")
    if not HAS_TORCH_CLUSTER:
        raise ImportError("torch_cluster required for kNN graph")
    
    # Build kNN graph (batch-aware: only connects points within same batch)
    edge_index = knn_graph(pos, k=k, batch=batch, loop=False)
    
    # Create DGL graph
    src, dst = edge_index
    G = dgl.graph((src, dst), num_nodes=pos.shape[0])
    
    # Add node positions
    G.ndata['x'] = pos
    
    # Add relative positions as edge features (required for SE(3)-Transformer)
    rel_pos = pos[dst] - pos[src]  # (E, 3)
    G.edata['d'] = rel_pos
    
    return G


class SE3PointEncoder(nn.Module):
    """
    SE(3)-Equivariant Point Encoder using SE(3)-Transformer
    
    Maintains compatibility with encoder_se3.py interface
    """
    
    def __init__(
        self,
        irreps_in: str = "2x0e + 1x1o",  # Not used, kept for compatibility
        irreps_hidden: str = "16x0e + 8x1o + 4x2e",  # Not used directly
        irreps_inv_out: str = "32x0e",  # Not used directly
        irreps_eq_out: str = "8x1o",    # Not used directly
        out_dim: int = 512,  # final projection dimension
        num_layers: int = 4,
        k: int = 16,  # kNN parameter
        lmax: int = 2,  # max spherical harmonics degree (num_degrees = lmax + 1)
        num_channels: int = 32,  # channels per degree
        use_curvature: bool = True,
    ):
        super().__init__()
        self.k = k
        self.num_degrees = lmax + 1  # SE(3)-Transformer uses 0-based degrees
        self.num_channels = num_channels
        self.use_curvature = use_curvature
        self.out_dim = out_dim
        
        if not HAS_DGL:
            raise ImportError("DGL required for SE(3)-Transformer. Install with: pip install dgl")
        
        # Define fiber structures
        # Input: scalar (curvature) + vector (normals)
        if use_curvature:
            self.fiber_in = Fiber(structure=[(2, 0), (1, 1)])  # 2 scalars, 1 vector
        else:
            self.fiber_in = Fiber(structure=[(1, 0), (1, 1)])  # 1 scalar, 1 vector
        
        # Hidden: multiple degrees with channels
        self.fiber_mid = Fiber(self.num_degrees, self.num_channels)
        
        # Output: scalar (invariant) + vector (equivariant)
        # Extract dimensions from irreps strings for compatibility
        inv_dim = 32  # Default
        eq_dim = 8    # Default
        if 'x0e' in irreps_inv_out:
            inv_dim = int(irreps_inv_out.split('x')[0])
        if 'x1o' in irreps_eq_out:
            eq_dim = int(irreps_eq_out.split('x')[0])
        
        self.fiber_out = Fiber(structure=[(inv_dim, 0), (eq_dim, 1)])
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        
        # Input projection
        self.layers.append(G1x1SE3(self.fiber_in, self.fiber_mid))
        
        # SE(3)-equivariant layers
        for i in range(num_layers):
            # Attention block with skip connection
            self.layers.append(GSE3Res(
                self.fiber_mid, 
                self.fiber_mid,
                edge_dim=0,  # No edge features beyond relative position
                div=4,
                n_heads=1,
                skip='sum',
                selfint='1x1'
            ))
            # Normalization + nonlinearity
            self.layers.append(GNormSE3(self.fiber_mid, num_layers=1))
        
        # Output projection
        self.layers.append(G1x1SE3(self.fiber_mid, self.fiber_out))
        
        # Projection head for invariant embedding
        self.project = nn.Sequential(
            nn.Linear(inv_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    
    def build_knn_graph(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> dgl.DGLGraph:
        """Build kNN graph as DGL graph"""
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        return build_dgl_graph(pos, batch, k=self.k)
    
    def forward(
        self,
        pos: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pos: (N, 3) or (B, N, 3) - point positions (will be normalized)
            feat: Optional (N, F) or (B, N, F) - point features [normals(3), curvature(2)]
            edge_index: Optional (2, E) - pre-computed edges (not used, we build DGL graph)
            batch: Optional (N,) - batch indices
        Returns:
            dict with:
                'invariant': (B, inv_dim) - invariant embedding
                'equivariant': (B, eq_dim, 3) - equivariant features
                'embedding': (B, out_dim) - final projected embedding
        """
        # Handle batch dimension
        if pos.dim() == 3:
            B, N, _ = pos.shape
            pos_flat = pos.view(B * N, 3)
            if batch is None:
                batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        else:
            pos_flat = pos
            if batch is None:
                batch = torch.zeros(pos_flat.shape[0], dtype=torch.long, device=pos_flat.device)
            B = batch.max().item() + 1
        
        # CRITICAL: Batch-aware normalization
        pos_norm = normalize_points_batch(pos_flat, batch)
        
        # Convert features to fiber format
        if feat is not None:
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)  # Add batch dimension
            feat_fiber = features_to_fiber(feat, use_curvature=self.use_curvature)
        else:
            # Create dummy features
            N_total = pos_flat.shape[0]
            if self.use_curvature:
                dummy_feat = torch.zeros(N_total, 5, device=pos_flat.device)
                dummy_feat[:, :3] = torch.randn(N_total, 3, device=pos_flat.device)  # Random normals
            else:
                dummy_feat = torch.zeros(N_total, 3, device=pos_flat.device)
                dummy_feat[:, :3] = torch.randn(N_total, 3, device=pos_flat.device)
            feat_fiber = features_to_fiber(dummy_feat, use_curvature=self.use_curvature)
        
        # Build DGL graph
        G = self.build_knn_graph(pos_norm, batch)
        
        # Compute basis and distances (required for SE(3)-Transformer)
        basis, r = get_basis_and_r(G, max_degree=self.num_degrees-1)
        
        # Forward pass through SE(3)-Transformer layers
        h = feat_fiber
        for layer in self.layers:
            if isinstance(layer, (GSE3Res, GConvSE3, GNormSE3)):
                h = layer(h, G=G, r=r, basis=basis)
            elif isinstance(layer, G1x1SE3):
                h = layer(h)
            else:
                h = layer(h, G=G)
        
        # Extract invariant and equivariant features
        # h is a dict with keys '0' (scalars) and '1' (vectors)
        # Shape: '0': (N, inv_dim, 1), '1': (N, eq_dim, 3)
        
        # Extract scalar features (invariant)
        h_inv = h['0'].squeeze(-1)  # (N, inv_dim)
        
        # Extract vector features (equivariant)
        h_eq = h['1']  # (N, eq_dim, 3)
        
        # Global pooling (batch-aware)
        x_inv_pooled = scatter_mean(h_inv, batch, dim=0)  # (B, inv_dim)
        x_eq_pooled = scatter_mean(h_eq, batch, dim=0)  # (B, eq_dim, 3)
        
        # Project invariant to final embedding
        embedding = self.project(x_inv_pooled)  # (B, out_dim)
        
        return {
            'invariant': x_inv_pooled,      # (B, inv_dim)
            'equivariant': x_eq_pooled,     # (B, eq_dim, 3)
            'embedding': embedding,          # (B, out_dim) - for backward compatibility
        }
    
    def encode_points(
        self,
        pos: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode points and return both point-level and global embeddings
        
        Args:
            pos: (N, 3) or (B, N, 3) - point positions
            feat: Optional (N, F) or (B, N, F) - point features
            edge_index: Optional (2, E) - edge indices (not used)
            batch: Optional (N,) - batch indices
        Returns:
            point_inv: (N, inv_dim) - point-level invariant features
            point_eq: (N, eq_dim, 3) - point-level equivariant features
            global_output: dict with 'invariant', 'equivariant', 'embedding'
        """
        # Handle batch dimension
        if pos.dim() == 3:
            B, N, _ = pos.shape
            pos_flat = pos.view(B * N, 3)
            if batch is None:
                batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        else:
            pos_flat = pos
            if batch is None:
                batch = torch.zeros(pos_flat.shape[0], dtype=torch.long, device=pos_flat.device)
            B = batch.max().item() + 1
        
        # CRITICAL: Batch-aware normalization
        pos_norm = normalize_points_batch(pos_flat, batch)
        
        # Convert features to fiber format
        if feat is not None:
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            feat_fiber = features_to_fiber(feat, use_curvature=self.use_curvature)
        else:
            N_total = pos_flat.shape[0]
            if self.use_curvature:
                dummy_feat = torch.zeros(N_total, 5, device=pos_flat.device)
                dummy_feat[:, :3] = torch.randn(N_total, 3, device=pos_flat.device)
            else:
                dummy_feat = torch.zeros(N_total, 3, device=pos_flat.device)
                dummy_feat[:, :3] = torch.randn(N_total, 3, device=pos_flat.device)
            feat_fiber = features_to_fiber(dummy_feat, use_curvature=self.use_curvature)
        
        # Build DGL graph
        G = self.build_knn_graph(pos_norm, batch)
        
        # Compute basis and distances
        basis, r = get_basis_and_r(G, max_degree=self.num_degrees-1)
        
        # Forward pass
        h = feat_fiber
        for layer in self.layers:
            if isinstance(layer, (GSE3Res, GConvSE3, GNormSE3)):
                h = layer(h, G=G, r=r, basis=basis)
            elif isinstance(layer, G1x1SE3):
                h = layer(h)
            else:
                h = layer(h, G=G)
        
        # Extract point-level features
        point_inv = h['0'].squeeze(-1)  # (N, inv_dim)
        point_eq = h['1']  # (N, eq_dim, 3)
        
        # Global pooling
        x_inv_pooled = scatter_mean(point_inv, batch, dim=0)  # (B, inv_dim)
        x_eq_pooled = scatter_mean(point_eq, batch, dim=0)  # (B, eq_dim, 3)
        
        # Project invariant to final embedding
        embedding = self.project(x_inv_pooled)  # (B, out_dim)
        
        global_output = {
            'invariant': x_inv_pooled,
            'equivariant': x_eq_pooled,
            'embedding': embedding,
        }
        
        return point_inv, point_eq, global_output

