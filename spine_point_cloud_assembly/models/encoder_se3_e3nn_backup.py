"""
SE(3)-Equivariant Point Encoder for Vertebra Point Clouds (e3nn-based)

Architecture:
1. Input normalization (centering + unit sphere scaling)
2. Feature conversion to Irreps (scalar + vector)
3. kNN graph construction
4. SE(3)-equivariant message passing (tensor product + spherical harmonics)
5. Invariant + Equivariant output separation
6. Projection head

Based on project guide: Section 4.A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.nn import Gate
from torch_scatter import scatter, scatter_mean

# e3nn 0.5.9: IrrepsArray is not available, use Irreps with tensor directly
# We'll use Irreps to define structure and pass tensors directly

try:
    from torch_cluster import knn_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    try:
        from torch_geometric.nn import knn_graph
        HAS_TORCH_CLUSTER = True
    except ImportError:
        HAS_TORCH_CLUSTER = False


def normalize_points(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize points: center by centroid and scale to unit sphere
    (Legacy function for single point cloud, use normalize_points_batch for batch processing)
    
    Args:
        points: (N, 3) - point positions
    Returns:
        points_normalized: normalized points (N, 3)
        centroid: (3,) - centroid
        scale: scalar - scale factor
    """
    N, _ = points.shape
    centroid = points.mean(dim=0, keepdim=True)  # (1, 3)
    points_centered = points - centroid
    
    distances = torch.norm(points_centered, dim=-1)  # (N,)
    max_dist = distances.max()
    max_dist = torch.clamp(max_dist, min=1e-6)
    scale = 1.0 / max_dist
    
    points_normalized = points_centered * scale
    
    return points_normalized, centroid.squeeze(0), scale


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


def features_to_irreps(
    features: torch.Tensor,
    use_curvature: bool = True,
    irreps_in: Optional[Irreps] = None,
) -> Tuple[torch.Tensor, Irreps]:
    """
    Convert features tensor to e3nn Irreps format (e3nn 0.5.9 compatible)
    
    Args:
        features: (B, N, 5) or (N, 5) - [normals(3), curvature(2)]
                  or (B, N, 3) or (N, 3) - [normals(3)] if no curvature
        use_curvature: whether to use curvature features
        irreps_in: Optional Irreps specification
    Returns:
        feat_tensor: (N, irreps_dim) - tensor with features in irreps order
        irreps_in: Irreps object
    """
    if irreps_in is None:
        if use_curvature and features.shape[-1] >= 5:
            irreps_in = Irreps("2x0e + 1x1o")  # curvature + normals
        else:
            irreps_in = Irreps("1x1o")  # normals only
    
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
    
    # Prepare features tensor in irreps order
    # For "2x0e + 1x1o": [scalar1, scalar2, vec_x, vec_y, vec_z]
    # e3nn 0.5.9: tensor shape is (N, irreps_dim) directly
    feat_list = []
    
    if curvature is not None:
        # Scalar features: (N, 2) for 2x0e
        feat_list.append(curvature)  # (B*N, 2)
    
    # Vector features: (N, 3) for 1x1o
    feat_list.append(normals)  # (B*N, 3)
    
    # Concatenate along feature dimension
    if len(feat_list) > 1:
        feat_tensor = torch.cat(feat_list, dim=-1)  # (B*N, 5) or (N, 5)
    else:
        feat_tensor = feat_list[0]  # (B*N, 3) or (N, 3)
    
    return feat_tensor, irreps_in


class SE3MessageBlock(nn.Module):
    """SE(3)-Equivariant Message Passing Block"""
    
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        lmax: int = 2,
        num_neighbors: int = 16,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.lmax = lmax
        self.num_neighbors = num_neighbors
        
        # Spherical harmonics for relative position
        self.irreps_sh = Irreps.spherical_harmonics(lmax)  # e.g., "1x0e + 1x1o + 1x2e"
        
        # Tensor product: feature ⊗ spherical_harmonics(rel_pos)
        self.tp = FullyConnectedTensorProduct(
            irreps_in,
            self.irreps_sh,
            irreps_out,
            shared_weights=False
        )
        
        # Equivariant linear layer
        self.lin = o3.Linear(irreps_out, irreps_out)
        
        # Radial basis function (RBF) for distance encoding
        # Distance information is critical for assembly and shape reasoning
        # Without this, only direction is used, losing vertebra size/spacing info
        self.rbf_centers = nn.Parameter(torch.linspace(0.0, 2.0, 8))  # Learnable centers
        self.rbf_sigma = nn.Parameter(torch.tensor(0.5))  # Learnable width
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, irreps_in.dim) - node features as tensor
            pos: (N, 3) - node positions
            edge_index: (2, E) - edge indices
        Returns:
            (N, irreps_out.dim) - updated features as tensor
        """
        src, dst = edge_index  # (E,), (E,)
        
        # Relative positions
        rel = pos[dst] - pos[src]  # (E, 3)
        
        # Distance encoding (RBF) - preserves distance information
        r = torch.norm(rel, dim=1, keepdim=True)  # (E, 1)
        rbf = torch.exp(-((r - self.rbf_centers.unsqueeze(0))**2) / (2 * self.rbf_sigma**2))  # (E, num_centers)
        
        # Spherical harmonics encoding (rotation-equivariant, direction only)
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel,
            normalize=True,
            normalization='component'
        )  # (E, irreps_sh.dim)
        
        # Message: tensor product of features and position encoding
        msg = self.tp(x[src], sh)  # (E, irreps_out.dim)
        
        # Apply RBF distance weighting to scalar components
        # Extract scalar part (0e irreps) and multiply by RBF
        scalar_dim = sum(mul for mul, irrep in self.irreps_out if irrep.l == 0 and irrep.p == 1)
        if scalar_dim > 0:
            # Get scalar indices
            scalar_indices = []
            idx = 0
            for mul, irrep in self.irreps_out:
                if irrep.l == 0 and irrep.p == 1:  # scalar (0e)
                    for i in range(mul):
                        scalar_indices.append(idx + i)
                    idx += mul * irrep.dim
                else:
                    idx += mul * irrep.dim
            
            # Apply RBF to scalar components (weighted by distance)
            rbf_weight = rbf.mean(dim=1, keepdim=True)  # (E, 1) - average over RBF centers
            msg_scalar = msg[:, scalar_indices]  # (E, scalar_dim)
            msg_scalar = msg_scalar * rbf_weight  # Distance-weighted scalars
            msg[:, scalar_indices] = msg_scalar
        
        # Equivariant linear transformation
        msg = self.lin(msg)
        
        # Aggregate messages (equivariant aggregation)
        out = torch.zeros(
            x.shape[0],
            msg.shape[1],
            device=x.device,
            dtype=x.dtype
        )
        out.index_add_(0, dst, msg)
        
        # NOTE: Residual connection removed from here - managed at encoder level
        # This avoids double residual application (block + encoder)
        
        return out


class SE3PointEncoder(nn.Module):
    """
    SE(3)-Equivariant Point Encoder
    
    Based on project guide Section 4.A
    Outputs: invariant embedding (identity) + equivariant features (orientation)
    """
    
    def __init__(
        self,
        irreps_in: str = "2x0e + 1x1o",  # curvature + normals
        irreps_hidden: str = "16x0e + 8x1o + 4x2e",
        irreps_inv_out: str = "32x0e",  # invariant output
        irreps_eq_out: str = "8x1o",    # equivariant output
        out_dim: int = 512,  # final projection dimension
        num_layers: int = 4,
        k: int = 16,  # kNN parameter
        lmax: int = 2,  # max spherical harmonics degree
    ):
        super().__init__()
        self.k = k
        self.lmax = lmax
        
        # Parse irreps
        self.irreps_in = Irreps(irreps_in)
        self.irreps_hidden = Irreps(irreps_hidden)
        self.irreps_inv_out = Irreps(irreps_inv_out)
        self.irreps_eq_out = Irreps(irreps_eq_out)
        
        # Input embedding
        self.embed = o3.Linear(self.irreps_in, self.irreps_hidden)
        
        # Message passing layers with Gate for non-linearity
        # Gate improves expressiveness: scalar → SiLU, vector → gated by sigmoid
        # Structure: MessageBlock → Linear (split to scalars/gates/gated) → Gate
        
        # Decompose irreps_hidden into scalars, gates, and gated components
        # e.g., "16x0e + 8x1o + 4x2e" → scalars="16x0e", gates="8x0e", gated="8x1o + 4x2e"
        scalars_list = []
        gated_list = []
        for mul, irrep in self.irreps_hidden:
            if irrep.l == 0 and irrep.p == 1:  # scalar (0e)
                scalars_list.append(f"{mul}x0e")
            else:  # vector/tensor (1o, 2e, etc.)
                gated_list.append(f"{mul}x{irrep.l}{'e' if irrep.p == 1 else 'o'}")
        
        irreps_scalars_str = " + ".join(scalars_list) if scalars_list else "1x0e"
        irreps_gated_str = " + ".join(gated_list) if gated_list else "1x0e"
        
        irreps_scalars = Irreps(irreps_scalars_str)
        irreps_gated = Irreps(irreps_gated_str)
        
        num_gates = sum(mul for mul, _ in irreps_gated)   # e.g. 8 + 4 = 12
        irreps_gates = Irreps(f"{num_gates}x0e")

        irreps_gate_out = irreps_scalars + irreps_gates + irreps_gated

        # Message passing blocks
        self.message_blocks = nn.ModuleList([
            SE3MessageBlock(self.irreps_hidden, self.irreps_hidden, lmax=lmax, num_neighbors=k)
            for _ in range(num_layers)
        ])
        
        # Linear layers to split features into scalars/gates/gated for Gate
        self.gate_linears = nn.ModuleList([
            o3.Linear(self.irreps_hidden, irreps_gate_out)
            for _ in range(num_layers)
        ])
        
        # Gate layers (e3nn 0.5.9 API)
        # CRITICAL: Gate uses irreps count, not mul (channel) count
        # e.g., "16x0e" → 1 irrep → [F.silu] * 1 (not * 16)
        # Gate internally broadcasts activation across mul channels
        act_scalars = [F.silu] * len(irreps_scalars)  # irreps count, not mul count
        act_gates = [torch.sigmoid] * len(irreps_gates)  # irreps count, not mul count

        self.gates = nn.ModuleList([
            Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated,
            )
            for _ in range(num_layers)
        ])
        
        # Linear layer to map Gate output back to irreps_hidden for residual
        self.gate_output_linears = nn.ModuleList([
            o3.Linear(irreps_gate_out, self.irreps_hidden)
            for _ in range(num_layers)
        ])
        
        # Separate invariant and equivariant outputs
        self.to_invariant = o3.Linear(self.irreps_hidden, self.irreps_inv_out)
        self.to_equivariant = o3.Linear(self.irreps_hidden, self.irreps_eq_out)
        
        # Projection head for invariant embedding
        # Invariant output is scalar, so we need to extract scalar part
        # Count scalar (0e) irreps
        # e3nn 0.5.9: iterate irreps as (mul, irrep) tuples
        inv_dim = sum(mul for mul, irrep in self.irreps_inv_out if irrep.l == 0 and irrep.p == 1)
        if inv_dim == 0:
            inv_dim = 32  # fallback
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
    ) -> torch.Tensor:
        """
        Build kNN graph
        
        Args:
            pos: (N, 3) - point positions
            batch: (N,) - batch indices (optional)
        Returns:
            edge_index: (2, E) - edge indices
        """
        if not HAS_TORCH_CLUSTER:
            raise ImportError("torch_cluster or torch_geometric required for kNN graph")
        
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
        
        return edge_index
    
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
            feat: Optional (N, irreps_in.dim) - point features as tensor
            edge_index: Optional (2, E) - pre-computed edges
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
        
        # CRITICAL: Batch-aware normalization - each vertebra normalized separately
        # Without this, different vertebrae share centroid/scale → information loss
        pos_norm = normalize_points_batch(pos_flat, batch)
        
        # Build kNN graph if not provided
        if edge_index is None:
            edge_index = self.build_knn_graph(pos_norm, batch)
        
        # Convert features to Irreps format if provided
        # feat should be (N, F) where F is raw features [normals(3), curvature(2)]
        if feat is not None:
            # Assume feat is raw features, convert to irreps format
            feat_irreps, _ = features_to_irreps(
                feat.unsqueeze(0) if feat.dim() == 2 else feat,
                use_curvature=True
            )
            if feat.dim() == 3:
                feat_irreps = feat_irreps.view(-1, feat_irreps.shape[-1])
        else:
            # Create zero features if not provided
            feat_irreps = torch.zeros(
                pos_flat.shape[0],
                self.irreps_in.dim,
                device=pos_flat.device,
                dtype=pos_flat.dtype
            )
        
        x = self.embed(feat_irreps)  # (N, irreps_hidden.dim)
        
        # Message passing with residual connections (managed at encoder level)
        # Each layer: MessageBlock → Linear (split) → Gate → Linear (merge) → Residual
        for message_block, gate_linear, gate, gate_output_linear in zip(
            self.message_blocks, self.gate_linears, self.gates, self.gate_output_linears
        ):
            # Message block
            x_new = message_block(x, pos_norm, edge_index)  # (N, irreps_hidden.dim)
            # Split to scalars + gates + gated for Gate
            x_new = gate_linear(x_new)  # (N, irreps_gate_out.dim)
            # Gate (non-linearity: scalar → SiLU, vector → gated by sigmoid)
            x_new = gate(x_new)  # (N, irreps_gate_out.dim)
            # Merge back to irreps_hidden
            x_new = gate_output_linear(x_new)  # (N, irreps_hidden.dim)
            # Residual connection
            x = x + x_new  # (N, irreps_hidden.dim)
        
        # Separate invariant and equivariant
        x_inv = self.to_invariant(x)  # (N, irreps_inv_out.dim)
        x_eq = self.to_equivariant(x)  # (N, irreps_eq_out.dim)
        
        # Global pooling (invariant aggregation)
        # Extract scalar part from invariant (0e irreps only)
        # x_inv is tensor with shape (N, irreps_inv_out.dim)
        # For "32x0e", shape is (N, 32) - each 0e has dim=1
        inv_dim = sum(mul for mul, irrep in self.irreps_inv_out if irrep.l == 0 and irrep.p == 1)
        if inv_dim > 0:
            # Extract scalar components
            scalar_indices = []
            idx = 0
            for mul, irrep in self.irreps_inv_out:
                if irrep.l == 0 and irrep.p == 1:  # scalar (0e), dim=1
                    for i in range(mul):
                        scalar_indices.append(idx + i)
                    idx += mul * irrep.dim
                else:
                    idx += mul * irrep.dim
            
            if scalar_indices:
                x_inv_scalar = x_inv[:, scalar_indices]  # (N, inv_dim)
            else:
                x_inv_scalar = x_inv[:, :inv_dim]  # fallback
        else:
            x_inv_scalar = x_inv[:, :32]  # fallback
        
        x_inv_pooled = scatter_mean(x_inv_scalar, batch, dim=0)  # (B, inv_dim)
        
        # Equivariant pooling (mean)
        # Extract vector components from equivariant output
        # x_eq shape: (N, irreps_eq_out.dim) where for 1o, each vector takes 3 consecutive elements
        eq_dim = sum(mul for mul, irrep in self.irreps_eq_out if irrep.l == 1 and irrep.p == -1)
        if eq_dim > 0:
            # Extract vector components (1o irreps)
            vector_indices = []
            idx = 0
            for mul, irrep in self.irreps_eq_out:
                if irrep.l == 1 and irrep.p == -1:  # vector (1o), dim=3
                    for i in range(mul):
                        vector_indices.extend(range(idx + i*3, idx + i*3 + 3))
                    idx += mul * irrep.dim
                else:
                    idx += mul * irrep.dim
            
            if vector_indices:
                # Reshape to (N, eq_dim, 3)
                x_eq_flat = x_eq[:, vector_indices]  # (N, eq_dim*3)
                x_eq_vectors = x_eq_flat.view(x_eq_flat.shape[0], eq_dim, 3)  # (N, eq_dim, 3)
            else:
                x_eq_vectors = x_eq[:, :eq_dim*3].view(x_eq.shape[0], eq_dim, 3)  # fallback
        else:
            x_eq_vectors = x_eq[:, :8*3].view(x_eq.shape[0], 8, 3)  # fallback
        
        x_eq_pooled = scatter_mean(x_eq_vectors, batch, dim=0)  # (B, eq_dim, 3)
        
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
            feat: Optional (N, irreps_in.dim) - point features as tensor
            edge_index: Optional (2, E) - edge indices
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
        
        # CRITICAL: Batch-aware normalization - each vertebra normalized separately
        pos_norm = normalize_points_batch(pos_flat, batch)
        
        # Build kNN graph if not provided
        if edge_index is None:
            edge_index = self.build_knn_graph(pos_norm, batch)
        
        # Convert features to Irreps format if provided
        if feat is not None:
            feat_irreps, _ = features_to_irreps(
                feat.unsqueeze(0) if feat.dim() == 2 else feat,
                use_curvature=True
            )
            if feat.dim() == 3:
                feat_irreps = feat_irreps.view(-1, feat_irreps.shape[-1])
        else:
            feat_irreps = torch.zeros(
                pos_flat.shape[0],
                self.irreps_in.dim,
                device=pos_flat.device,
                dtype=pos_flat.dtype
            )
        
        x = self.embed(feat_irreps)
        
        # Message passing with residual connections (managed at encoder level)
        # Each layer: MessageBlock → Linear (split) → Gate → Linear (merge) → Residual
        for message_block, gate_linear, gate, gate_output_linear in zip(
            self.message_blocks, self.gate_linears, self.gates, self.gate_output_linears
        ):
            # Message block
            x_new = message_block(x, pos_norm, edge_index)
            # Split to scalars + gates + gated for Gate
            x_new = gate_linear(x_new)
            # Gate (non-linearity: scalar → SiLU, vector → gated by sigmoid)
            x_new = gate(x_new)
            # Merge back to irreps_hidden
            x_new = gate_output_linear(x_new)
            # Residual connection
            x = x + x_new
        
        # Separate invariant and equivariant
        x_inv = self.to_invariant(x)
        x_eq = self.to_equivariant(x)
        
        # Point-level features
        # Extract scalar components from invariant
        inv_dim = sum(mul for mul, irrep in self.irreps_inv_out if irrep.l == 0 and irrep.p == 1)
        if inv_dim > 0:
            scalar_indices = []
            idx = 0
            for mul, irrep in self.irreps_inv_out:
                if irrep.l == 0 and irrep.p == 1:  # scalar (0e)
                    for i in range(mul):
                        scalar_indices.append(idx + i)
                    idx += mul * irrep.dim
                else:
                    idx += mul * irrep.dim
            
            if scalar_indices:
                point_inv = x_inv[:, scalar_indices]  # (N, inv_dim)
            else:
                point_inv = x_inv[:, :inv_dim]  # fallback
        else:
            point_inv = x_inv[:, :32]  # fallback
        
        # Equivariant features: (N, eq_dim, 3)
        # Extract vector components (1o irreps)
        eq_dim = sum(mul for mul, irrep in self.irreps_eq_out if irrep.l == 1 and irrep.p == -1)
        if eq_dim > 0:
            vector_indices = []
            idx = 0
            for mul, irrep in self.irreps_eq_out:
                if irrep.l == 1 and irrep.p == -1:  # vector (1o), dim=3
                    for i in range(mul):
                        vector_indices.extend(range(idx + i*3, idx + i*3 + 3))
                    idx += mul * irrep.dim
                else:
                    idx += mul * irrep.dim
            
            if vector_indices:
                x_eq_flat = x_eq[:, vector_indices]  # (N, eq_dim*3)
                point_eq = x_eq_flat.view(N, eq_dim, 3)  # (N, eq_dim, 3)
            else:
                point_eq = x_eq[:, :eq_dim*3].view(N, eq_dim, 3)  # fallback
        else:
            point_eq = x_eq[:, :8*3].view(N, 8, 3)  # fallback
        
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

