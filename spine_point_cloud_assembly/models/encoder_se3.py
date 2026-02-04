"""
SE(3)-Equivariant Point Encoder for Vertebra Point Clouds (TorchMD-Net style)

Based on TorchMD-Net architecture principles:
- Equivariant GNN backbone
- SE(3) message passing with spherical harmonics
- Distance embedding (RBF)
- Global pooling for invariant/equivariant outputs

Encoder Role:
- Input: single vertebra point cloud (pos, feat, batch)
- Output: 
  - z_inv ∈ R^D: rotation-invariant global embedding
  - z_eq ∈ R^{K×3}: rotation-equivariant orientation features

Design Principles:
1. Encoder sees only local geometry (per-vertebra normalization)
2. SE(3) equivariance guaranteed in encoder
3. Point → Graph → Equivariant Message Passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from torch_scatter import scatter_mean

try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    try:
        from torch_geometric.nn import radius_graph
        HAS_TORCH_CLUSTER = True
    except ImportError:
        HAS_TORCH_CLUSTER = False
        print("WARNING: torch_cluster or torch_geometric required for radius_graph")


def normalize_points_batch(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Batch-aware normalization: normalize each vertebra separately.
    
    Principle 1: "Encoder sees only local geometry"
    - Center by centroid
    - Scale to unit sphere (max norm = 1)
    
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
        
        # Center by centroid
        centroid = pts.mean(dim=0, keepdim=True)
        pts = pts - centroid
        
        # Scale to unit sphere
        max_norm = torch.norm(pts, dim=1).max().clamp(min=1e-6)
        pts = pts / max_norm
        
        pos_norm[idx] = pts
    return pos_norm


def features_to_irreps(
    features: torch.Tensor,
    use_curvature: bool = True,
    irreps_in: Optional[Irreps] = None,
) -> torch.Tensor:
    """
    Convert features to e3nn Irreps format.
    
    Feature composition:
    - curvature (mean, gauss): scalar (0e)
    - intensity/thickness: scalar (0e)
    - normal vector: vector (1o) - MUST be normalized
    
    Args:
        features: (B, N, F) or (N, F) where F=5 (normals(3) + curvature(2)) or F=3 (normals(3))
        use_curvature: whether to use curvature features
        irreps_in: optional Irreps specification (for compatibility)
    Returns:
        feat_tensor: (N, irreps_dim) - features tensor in irreps order
    """
    if features.dim() == 3:
        B, N, F = features.shape
        features_flat = features.view(B * N, F)
    elif features.dim() == 2:
        N, F = features.shape
        features_flat = features
    else:
        raise ValueError(f"Expected 2D or 3D features, got {features.dim()}D")
    
    # Separate and normalize normals
    normals = features_flat[:, :3]  # (N, 3)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
    
    # Prepare features in irreps order
    # For "2x0e + 1x1o": [scalar1, scalar2, vec_x, vec_y, vec_z]
    feat_list = []
    
    if use_curvature and F >= 5:
        # Scalar features: curvature (2x0e)
        curvature = features_flat[:, 3:5]  # (N, 2)
        feat_list.append(curvature)
    
    # Vector features: normals (1x1o)
    feat_list.append(normals)  # (N, 3)
    
    # Concatenate: [scalars..., vectors...]
    feat_tensor = torch.cat(feat_list, dim=-1)  # (N, 5) or (N, 3)
    
    return feat_tensor


class RadialBasisFunction(nn.Module):
    """
    Radial Basis Function for distance embedding.
    TorchMD-Net style: Gaussian RBF with learnable centers.
    """
    def __init__(self, num_radial: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # Learnable RBF centers
        centers = torch.linspace(0.0, cutoff, num_radial)
        self.register_buffer('centers', centers)
        
        # Gaussian width (sigma)
        sigma = cutoff / num_radial
        self.register_buffer('sigma', torch.tensor(sigma))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (E,) - edge distances
        Returns:
            rbf: (E, num_radial) - RBF embeddings
        """
        # Gaussian RBF
        diff = distances.unsqueeze(-1) - self.centers.unsqueeze(0)  # (E, num_radial)
        rbf = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        return rbf


class EquivariantMessageBlock(nn.Module):
    """
    SE(3)-Equivariant Message Passing Block (TorchMD-Net style).
    
    Components:
    1. Spherical harmonics (rotation equivariance)
    2. Tensor product (feature × geometry)
    3. Distance embedding (RBF)
    4. Residual connection
    """
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_hidden: Irreps,
        irreps_out: Irreps,
        num_radial: int = 16,
        lmax: int = 2,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_hidden = Irreps(irreps_hidden)
        self.irreps_out = Irreps(irreps_out)
        self.lmax = lmax
        self.cutoff = cutoff
        
        # RBF for distance embedding
        self.rbf = RadialBasisFunction(num_radial=num_radial, cutoff=cutoff)
        
        # Spherical harmonics irreps: 0e + 1o + 2e + ... up to lmax
        irreps_sh = o3.Irreps([(1, (l, (-1)**l)) for l in range(lmax + 1)])
        
        # Message computation: combine features with spherical harmonics
        # Input: (scalar features) + (SH features from geometry)
        # We need to compute tensor product: features ⊗ SH
        irreps_tp_in = self.irreps_in + irreps_sh
        
        # Radial network: RBF → hidden features
        self.radial_net = nn.Sequential(
            nn.Linear(num_radial, 64),
            nn.SiLU(),
            nn.Linear(64, self.irreps_hidden.num_irreps),
        )
        
        # Tensor product: combine input features with SH
        # TorchMD-Net style: internal_weights=True for learnable weights
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=self.irreps_hidden,
            internal_weights=True,
        )
        
        # Linear layer for residual connection
        if self.irreps_in == self.irreps_out:
            self.residual_linear = None
        else:
            self.residual_linear = o3.Linear(self.irreps_in, self.irreps_out)
        
        # Output projection
        self.output_linear = o3.Linear(self.irreps_hidden, self.irreps_out)
        
        # TorchMD-Net style: SiLU activation (no Gate)
        self.act = nn.SiLU()
    
    def forward(
        self,
        node_attr: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,  # relative positions (E, 3)
        edge_length: torch.Tensor,  # distances (E,)
    ) -> torch.Tensor:
        """
        Args:
            node_attr: (N, irreps_in.dim) - node features
            edge_src: (E,) - source node indices
            edge_dst: (E,) - destination node indices
            edge_attr: (E, 3) - relative positions
            edge_length: (E,) - edge distances
        Returns:
            out: (N, irreps_out.dim) - output node features
        """
        # 1. Compute spherical harmonics from relative positions
        sh = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            edge_attr,
            normalize=True,
            normalization='component',
        )  # (E, irreps_sh.dim)
        
        # 2. RBF embedding of distances
        rbf = self.rbf(edge_length)  # (E, num_radial)
        radial_weights = self.radial_net(rbf)  # (E, irreps_hidden.num_irreps)
        
        # 3. Tensor product: node features ⊗ spherical harmonics
        # Batch tensor product for all edges at once
        src_attr = node_attr[edge_src]  # (E, irreps_in.dim)
        
        # FullyConnectedTensorProduct can handle batched inputs
        messages = self.tp(src_attr, sh)  # (E, irreps_hidden.dim)
        
        # Apply radial weights (per-irrep scaling)
        # radial_weights: (E, irreps_hidden.num_irreps)
        # Each irrep gets one weight, expand to match irrep dimension
        radial_expanded = []
        weight_idx = 0
        for mul, ir in self.irreps_hidden:
            ir_dim = ir.dim  # dimension of this irrep (e.g., 1 for 0e, 3 for 1o, 5 for 2e)
            # Each multiplicity gets the same weight
            for _ in range(mul):
                radial_expanded.append(
                    radial_weights[:, weight_idx:weight_idx+1].expand(-1, ir_dim)
                )
                weight_idx += 1
        radial_expanded = torch.cat(radial_expanded, dim=-1)  # (E, irreps_hidden.dim)
        messages = messages * radial_expanded
        
        # 4. Aggregate messages (sum over neighbors)
        out = torch.zeros(node_attr.shape[0], self.irreps_hidden.dim, 
                         device=node_attr.device, dtype=node_attr.dtype)
        out.index_add_(0, edge_dst, messages)
        
        # 5. Output projection
        out = self.output_linear(out)
        
        # 6. TorchMD-Net style activation (SiLU, channel-wise)
        out = self.act(out)
        
        # 7. Residual connection
        if self.residual_linear is not None:
            residual = self.residual_linear(node_attr)
        else:
            residual = node_attr
        
        out = out + residual
        
        return out


class SE3PointEncoder(nn.Module):
    """
    SE(3)-Equivariant Point Encoder (TorchMD-Net style).
    
    Architecture:
    1. Input normalization (per-vertebra)
    2. Feature embedding
    3. Graph construction (radius_graph)
    4. Equivariant message passing blocks
    5. Global pooling
    6. Invariant/Equivariant output separation
    """
    def __init__(
        self,
        irreps_in: str = "2x0e + 1x1o",  # compatibility (not used directly)
        irreps_hidden: str = "32x0e + 16x1o + 8x2e",
        irreps_inv_out: str = "64x0e",
        irreps_eq_out: str = "8x1o",
        out_dim: int = 512,
        num_layers: int = 4,
        num_radial: int = 16,
        lmax: int = 2,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        use_curvature: bool = True,
    ):
        super().__init__()
        self.irreps_hidden = Irreps(irreps_hidden)
        self.irreps_inv_out = Irreps(irreps_inv_out)
        self.irreps_eq_out = Irreps(irreps_eq_out)
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_radial = num_radial
        self.lmax = lmax
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.use_curvature = use_curvature
        
        # Input feature embedding
        # Features: normals (1o) + curvature (0e) or just normals
        # Note: features_to_irreps returns [curvature(2), normals(3)] = "2x0e + 1x1o"
        if use_curvature:
            # Order: curvature (2x0e) + normals (1x1o)
            irreps_feat = Irreps("2x0e + 1x1o")
        else:
            # Only normals (1x1o)
            irreps_feat = Irreps("1x1o")
        
        self.feature_embedding = o3.Linear(irreps_feat, self.irreps_hidden)
        
        # Equivariant message passing blocks
        self.message_blocks = nn.ModuleList([
            EquivariantMessageBlock(
                irreps_in=self.irreps_hidden,
                irreps_hidden=self.irreps_hidden,
                irreps_out=self.irreps_hidden,
                num_radial=num_radial,
                lmax=lmax,
                cutoff=cutoff,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: split into invariant and equivariant
        self.inv_proj = o3.Linear(self.irreps_hidden, self.irreps_inv_out)
        self.eq_proj = o3.Linear(self.irreps_hidden, self.irreps_eq_out)
        
        # Final embedding projection (invariant only)
        inv_dim = self.irreps_inv_out.dim
        self.project = nn.Sequential(
            nn.Linear(inv_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def build_graph(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build radius graph (TorchMD-Net style).
        
        Args:
            pos: (N, 3) - point positions
            batch: (N,) - batch indices
        Returns:
            edge_index: (2, E) - edge indices
            edge_attr: (E, 3) - relative positions
            edge_length: (E,) - edge distances
        """
        if not HAS_TORCH_CLUSTER:
            raise RuntimeError("torch_cluster or torch_geometric required for radius_graph")
        
        # radius_graph returns edge_index in (2, E) format
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
            loop=False,
        )
        
        edge_src, edge_dst = edge_index[0], edge_index[1]
        
        # Compute relative positions and distances
        edge_attr = pos[edge_dst] - pos[edge_src]  # (E, 3)
        edge_length = torch.norm(edge_attr, dim=-1)  # (E,)
        
        return edge_index, edge_attr, edge_length, edge_src, edge_dst
    
    def forward(
        self,
        pos: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            pos: (B, N, 3) or (N, 3) - point positions
            feat: (B, N, F) or (N, F) - point features (normals + curvature)
            edge_index: optional pre-computed edges (ignored, we compute radius_graph)
            batch: (N,) - batch indices (required)
        Returns:
            output: dict with keys:
                - 'invariant': (B, inv_dim) - rotation-invariant features
                - 'equivariant': (B, eq_dim, 3) - rotation-equivariant features
                - 'embedding': (B, out_dim) - final embedding
        """
        # Flatten batch dimension
        if pos.dim() == 3:
            B, N, _ = pos.shape
            pos_flat = pos.view(B * N, 3)
            if batch is None:
                batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        else:
            pos_flat = pos
            if batch is None:
                batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        
        # Principle 1: Batch-aware normalization
        pos_norm = normalize_points_batch(pos_flat, batch)
        
        # Prepare features
        if feat is not None:
            feat_tensor = features_to_irreps(feat, use_curvature=self.use_curvature)
        else:
            # Create dummy features (random normals)
            N_total = pos_flat.shape[0]
            if self.use_curvature:
                feat_tensor = torch.randn(N_total, 5, device=pos_flat.device)
                feat_tensor[:, :3] = F.normalize(feat_tensor[:, :3], dim=-1)
            else:
                feat_tensor = F.normalize(torch.randn(N_total, 3, device=pos_flat.device), dim=-1)
            feat_tensor = features_to_irreps(feat_tensor, use_curvature=self.use_curvature)
        
        # Build graph
        edge_index, edge_attr, edge_length, edge_src, edge_dst = self.build_graph(
            pos_norm, batch
        )
        
        # Convert features to irreps format (for e3nn)
        # feat_tensor is (N, F) where F depends on use_curvature
        # We need to convert to IrrepsArray or use Linear layer
        # For now, use feature embedding directly
        node_attr = self.feature_embedding(feat_tensor)
        
        # Message passing
        for block in self.message_blocks:
            node_attr = block(
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_attr,
                edge_length=edge_length,
            )
        
        # Split into invariant and equivariant
        # Project to output irreps
        inv_attr = self.inv_proj(node_attr)  # (N, irreps_inv_out.dim)
        eq_attr = self.eq_proj(node_attr)  # (N, irreps_eq_out.dim)
        
        # Global pooling (batch-aware)
        # For invariant: mean over points (scalar irreps only)
        inv_pooled = scatter_mean(inv_attr, batch, dim=0)  # (B, irreps_inv_out.dim)
        
        # For equivariant: mean pooling (safe for vectors)
        # eq_attr is (N, irreps_eq_out.dim) where dim includes vector components
        # For "8x1o": dim = 8 * 3 = 24, reshape to (N, 8, 3)
        eq_dim = self.irreps_eq_out.dim
        # Extract vector irreps (l=1): each has dim=3
        num_vectors = sum(mul for mul, ir in self.irreps_eq_out if ir.l == 1)
        if num_vectors > 0:
            eq_reshaped = eq_attr.view(-1, num_vectors, 3)  # (N, num_vectors, 3)
            eq_pooled = scatter_mean(eq_reshaped, batch, dim=0)  # (B, num_vectors, 3)
        else:
            # No vector irreps, return empty
            B = batch.max().item() + 1
            eq_pooled = torch.zeros(B, 0, 3, device=node_attr.device, dtype=node_attr.dtype)
        
        # Final embedding (invariant only)
        embedding = self.project(inv_pooled)
        
        return {
            'invariant': inv_pooled,
            'equivariant': eq_pooled,
            'embedding': embedding,
        }
    
    def encode_points(
        self,
        pos: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode points and return point-level + global features.
        
        Returns:
            point_inv: (N, inv_dim) - point-level invariant features
            point_eq: (N, eq_dim, 3) - point-level equivariant features
            global_output: dict with 'invariant', 'equivariant', 'embedding'
        """
        # Same forward pass but return point-level features too
        if pos.dim() == 3:
            B, N, _ = pos.shape
            pos_flat = pos.view(B * N, 3)
            if batch is None:
                batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        else:
            pos_flat = pos
            if batch is None:
                batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        
        pos_norm = normalize_points_batch(pos_flat, batch)
        
        if feat is not None:
            feat_tensor = features_to_irreps(feat, use_curvature=self.use_curvature)
        else:
            N_total = pos_flat.shape[0]
            if self.use_curvature:
                feat_tensor = torch.randn(N_total, 5, device=pos_flat.device)
                feat_tensor[:, :3] = F.normalize(feat_tensor[:, :3], dim=-1)
            else:
                feat_tensor = F.normalize(torch.randn(N_total, 3, device=pos_flat.device), dim=-1)
            feat_tensor = features_to_irreps(feat_tensor, use_curvature=self.use_curvature)
        
        edge_index, edge_attr, edge_length, edge_src, edge_dst = self.build_graph(
            pos_norm, batch
        )
        
        node_attr = self.feature_embedding(feat_tensor)
        
        for block in self.message_blocks:
            node_attr = block(
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_attr,
                edge_length=edge_length,
            )
        
        inv_attr = self.inv_proj(node_attr)  # (N, irreps_inv_out.dim)
        eq_attr = self.eq_proj(node_attr)  # (N, irreps_eq_out.dim)
        
        # Point-level features
        point_inv = inv_attr  # (N, irreps_inv_out.dim)
        
        # Reshape equivariant to (N, num_vectors, 3)
        num_vectors = sum(mul for mul, ir in self.irreps_eq_out if ir.l == 1)
        if num_vectors > 0:
            point_eq = eq_attr.view(-1, num_vectors, 3)  # (N, num_vectors, 3)
        else:
            point_eq = torch.zeros(inv_attr.shape[0], 0, 3, device=inv_attr.device, dtype=inv_attr.dtype)
        
        # Global pooling
        inv_pooled = scatter_mean(inv_attr, batch, dim=0)  # (B, irreps_inv_out.dim)
        if num_vectors > 0:
            eq_pooled = scatter_mean(point_eq, batch, dim=0)  # (B, num_vectors, 3)
        else:
            B = batch.max().item() + 1
            eq_pooled = torch.zeros(B, 0, 3, device=inv_attr.device, dtype=inv_attr.dtype)
        
        embedding = self.project(inv_pooled)
        
        global_output = {
            'invariant': inv_pooled,
            'equivariant': eq_pooled,
            'embedding': embedding,
        }
        
        return point_inv, point_eq, global_output
