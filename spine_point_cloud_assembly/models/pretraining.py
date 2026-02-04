"""
Self-Supervised Pretraining Tasks and Losses

1. Rotation Canonicalization
2. Contrastive Vertebra-Type Learning
3. Masked Point Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


def random_rotation_matrix(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Generate random SO(3) rotation matrices
    
    Args:
        batch_size: Batch size
        device: Device
    Returns:
        (B, 3, 3) - rotation matrices
    """
    # Generate random rotation using quaternion method
    q = torch.randn(batch_size, 4, device=device)
    q = q / q.norm(dim=1, keepdim=True)  # Normalize to unit quaternion
    
    # Convert quaternion to rotation matrix
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=1),
    ], dim=1)
    
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion
    
    Args:
        R: (B, 3, 3) - rotation matrices
    Returns:
        (B, 4) - quaternions [w, x, y, z]
    """
    B = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    q = torch.zeros(B, 4, device=R.device)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2
    
    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4
    
    return q


def geodesic_distance_so3(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance on SO(3) between two rotation matrices
    
    Args:
        R1: (B, 3, 3) - rotation matrices
        R2: (B, 3, 3) - rotation matrices
    Returns:
        (B,) - geodesic distances
    """
    # R_diff = R1 @ R2.T
    R_diff = torch.bmm(R1, R2.transpose(1, 2))
    
    # Trace of rotation matrix: tr(R) = 1 + 2*cos(theta)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    
    # Clamp trace to valid range for rotation matrices: [-1, 3]
    # For valid rotation matrices, trace should be in [-1, 3]
    trace = torch.clamp(trace, -1.0, 3.0)
    
    # Angle: theta = arccos((tr(R) - 1) / 2)
    # For rotation matrices: (tr(R) - 1) / 2 should be in [-1, 1]
    cos_theta = (trace - 1.0) / 2.0
    
    # Clamp to avoid NaN from acos (domain is [-1, 1])
    # Use tighter bounds to avoid numerical issues
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    
    # Check for NaN/Inf before acos
    nan_mask = torch.isnan(cos_theta) | torch.isinf(cos_theta)
    
    # Compute theta
    theta = torch.acos(cos_theta)
    
    # Handle NaN cases with fallback
    if torch.any(nan_mask):
        # Fallback: use Frobenius norm distance
        diff = R_diff - torch.eye(3, device=R_diff.device, dtype=R_diff.dtype).unsqueeze(0)
        theta_fallback = diff.norm(dim=(1, 2)) / 2.0  # Approximate angle
        theta = torch.where(nan_mask, theta_fallback, theta)
    
    # Final check for NaN/Inf after acos
    nan_mask = torch.isnan(theta) | torch.isinf(theta)
    if torch.any(nan_mask):
        # Fallback: use Frobenius norm distance
        diff = R_diff - torch.eye(3, device=R_diff.device, dtype=R_diff.dtype).unsqueeze(0)
        theta_fallback = diff.norm(dim=(1, 2)) / 2.0
        theta = torch.where(nan_mask, theta_fallback, theta)
    
    # Final safety check: clamp to reasonable range [0, pi]
    theta = torch.clamp(theta, 0.0, math.pi)
    
    return theta


class RotationCanonicalizationLoss(nn.Module):
    """
    Rotation Canonicalization Loss
    
    Task: Apply random rotation R, predict canonical orientation R^-1
    Loss: Geodesic distance on SO(3)
    """
    
    def __init__(self, embedding_dim: int = 512, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
        # Orientation prediction head (from embedding to rotation matrix)
        self.orientation_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 9),  # 3x3 rotation matrix (flattened)
        )
    
    def forward(
        self,
        embedding: torch.Tensor,
        rotation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedding: (B, output_dim) - vertebra embedding
            rotation: (B, 3, 3) - applied rotation matrices
        Returns:
            loss: scalar loss
            predicted_rotation: (B, 3, 3) - predicted rotation matrices
        """
        B = embedding.shape[0]
        
        # Check for NaN/Inf in input embedding
        if torch.any(torch.isnan(embedding)) or torch.any(torch.isinf(embedding)):
            # Return safe fallback
            identity = torch.eye(3, device=embedding.device, dtype=embedding.dtype).unsqueeze(0).expand(B, -1, -1)
            return torch.tensor(0.0, device=embedding.device, requires_grad=True), identity
        
        # Predict rotation from embedding
        rot_flat = self.orientation_head(embedding)  # (B, 9)
        rot_pred = rot_flat.view(B, 3, 3)
        
        # Check for NaN/Inf in predicted rotation
        if torch.any(torch.isnan(rot_pred)) or torch.any(torch.isinf(rot_pred)):
            # Use identity as fallback
            rot_pred = torch.eye(3, device=rot_pred.device, dtype=rot_pred.dtype).unsqueeze(0).expand(B, -1, -1)
        
        # Orthonormalize predicted rotation using Gram-Schmidt (more stable version)
        # Use detach to avoid gradient issues with fallback vectors
        u1 = rot_pred[:, :, 0]
        u1_norm = u1.norm(dim=1, keepdim=True)
        u1_norm = torch.clamp(u1_norm, min=1e-6)
        u1 = u1 / u1_norm
        
        u2 = rot_pred[:, :, 1]
        u2_proj = (u2 * u1).sum(dim=1, keepdim=True) * u1
        u2 = u2 - u2_proj
        u2_norm = u2.norm(dim=1, keepdim=True)
        
        # Handle near-parallel case: if u2 is too small, use a fixed orthogonal vector
        # Use a fixed vector to maintain gradient flow
        parallel_threshold = 1e-4
        parallel_mask = u2_norm.squeeze() < parallel_threshold
        
        if torch.any(parallel_mask):
            # Use a fixed orthogonal vector (e.g., [0, 1, 0] projected onto plane)
            # This maintains gradient flow better than random vectors
            fixed_vec = torch.tensor([0.0, 1.0, 0.0], device=u2.device, dtype=u2.dtype).unsqueeze(0).expand_as(u2)
            fixed_vec = fixed_vec - (fixed_vec * u1).sum(dim=1, keepdim=True) * u1
            fixed_vec_norm = fixed_vec.norm(dim=1, keepdim=True)
            fixed_vec_norm = torch.clamp(fixed_vec_norm, min=1e-6)
            fixed_vec = fixed_vec / fixed_vec_norm
            
            # Only use fallback where needed, maintain gradients elsewhere
            u2 = torch.where(parallel_mask.unsqueeze(1), fixed_vec, u2)
            u2_norm = u2.norm(dim=1, keepdim=True)
        
        u2_norm = torch.clamp(u2_norm, min=1e-6)
        u2 = u2 / u2_norm
        
        u3 = torch.cross(u1, u2, dim=1)
        u3_norm = u3.norm(dim=1, keepdim=True)
        
        # u3 should be close to 1.0 if u1 and u2 are orthonormal
        small_cross_mask = u3_norm.squeeze() < 1e-4
        if torch.any(small_cross_mask):
            # Recompute u2 for problematic cases using a different approach
            # Use a fixed vector perpendicular to u1
            fixed_perp = torch.tensor([1.0, 0.0, 0.0], device=u1.device, dtype=u1.dtype).unsqueeze(0).expand_as(u1)
            # Make it perpendicular to u1
            fixed_perp = fixed_perp - (fixed_perp * u1).sum(dim=1, keepdim=True) * u1
            fixed_perp_norm = fixed_perp.norm(dim=1, keepdim=True)
            fixed_perp_norm = torch.clamp(fixed_perp_norm, min=1e-6)
            fixed_perp = fixed_perp / fixed_perp_norm
            
            # Recompute u2 and u3 for problematic cases
            u2_fixed = torch.where(small_cross_mask.unsqueeze(1), fixed_perp, u2)
            u3_fixed = torch.cross(u1, u2_fixed, dim=1)
            u3_fixed_norm = u3_fixed.norm(dim=1, keepdim=True)
            u3_fixed_norm = torch.clamp(u3_fixed_norm, min=1e-6)
            u3_fixed = u3_fixed / u3_fixed_norm
            
            u2 = u2_fixed
            u3 = u3_fixed
            u3_norm = u3_fixed_norm
        
        u3_norm = torch.clamp(u3_norm, min=1e-6)
        u3 = u3 / u3_norm
        
        rot_pred = torch.stack([u1, u2, u3], dim=2)  # (B, 3, 3)
        
        # Verify rotation matrix validity (determinant should be ~1)
        # Use a more efficient check: |det - 1| < threshold
        det = torch.det(rot_pred)
        invalid_mask = (torch.abs(det - 1.0) > 0.5) | torch.isnan(det) | torch.isinf(det)
        
        if torch.any(invalid_mask):
            # For invalid cases, use identity rotation
            # But maintain gradient flow by using a weighted combination
            identity = torch.eye(3, device=rot_pred.device, dtype=rot_pred.dtype).unsqueeze(0).expand(B, -1, -1)
            # Use where only for truly invalid cases
            rot_pred = torch.where(invalid_mask.unsqueeze(1).unsqueeze(2), identity, rot_pred)
        
        # Target: inverse rotation
        rot_target = rotation.transpose(1, 2)  # R^-1 = R^T
        
        # Geodesic distance loss
        geodesic_dist = geodesic_distance_so3(rot_pred, rot_target)
        
        # Check for NaN in geodesic distance
        if torch.any(torch.isnan(geodesic_dist)) or torch.any(torch.isinf(geodesic_dist)):
            # Fallback: use Frobenius norm
            diff = rot_pred - rot_target
            geodesic_dist = diff.norm(dim=(1, 2))
            geodesic_dist = torch.where(
                torch.isnan(geodesic_dist) | torch.isinf(geodesic_dist),
                torch.zeros_like(geodesic_dist),
                geodesic_dist
            )
        
        loss = geodesic_dist.mean()
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=embedding.device, requires_grad=True)
        
        return loss, rot_pred


class ContrastiveVertebraTypeLoss(nn.Module):
    """
    Contrastive Vertebra-Type Learning Loss
    
    Positive pairs: same vertebra label across patients
    Negative pairs: different vertebra IDs
    Loss: NT-Xent contrastive loss
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, output_dim) - vertebra embeddings
            labels: (B,) - vertebra labels (1-24 for C1-L5)
        Returns:
            loss: scalar contrastive loss
        """
        B = embeddings.shape[0]
        
        # Check for NaN/Inf in inputs
        if torch.any(torch.isnan(embeddings)) or torch.any(torch.isinf(embeddings)):
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1, p=2, eps=1e-8)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)
        
        # Clamp similarity to avoid overflow in exp
        similarity = torch.clamp(similarity, min=-50.0, max=50.0)
        
        # Create positive mask (same label)
        labels_expanded = labels.unsqueeze(1)  # (B, 1)
        positive_mask = (labels_expanded == labels_expanded.T).float()  # (B, B)
        positive_mask = positive_mask - torch.eye(B, device=embeddings.device)  # Remove self
        
        # NT-Xent loss
        exp_sim = torch.exp(similarity)  # (B, B)
        
        # Sum of positive similarities
        positive_sim = (exp_sim * positive_mask).sum(dim=1)  # (B,)
        
        # Sum of all similarities (excluding self)
        all_sim = exp_sim.sum(dim=1) - torch.exp(torch.diag(similarity))  # (B,)
        
        # Loss: -log(positive_sim / all_sim)
        # Add epsilon to avoid log(0)
        ratio = (positive_sim + 1e-8) / (all_sim + 1e-8)
        ratio = torch.clamp(ratio, min=1e-8, max=1.0)  # Ensure valid log domain
        loss = -torch.log(ratio)
        
        # Check for NaN
        if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
            loss = torch.where(
                torch.isnan(loss) | torch.isinf(loss),
                torch.zeros_like(loss),
                loss
            )
        
        return loss.mean()


class MaskedPointModelingLoss(nn.Module):
    """
    Masked Point Modeling Loss
    
    Task: Randomly drop 30-50% points, predict latent representation consistency
    Loss: Cosine similarity between embeddings
    """
    
    def __init__(self, mask_ratio: float = 0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
    
    def forward(
        self,
        embedding_full: torch.Tensor,
        embedding_masked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embedding_full: (B, output_dim) - embedding from full point cloud
            embedding_masked: (B, output_dim) - embedding from masked point cloud
        Returns:
            loss: scalar cosine similarity loss
        """
        # Check for NaN/Inf in inputs
        if torch.any(torch.isnan(embedding_full)) or torch.any(torch.isinf(embedding_full)):
            return torch.tensor(1.0, device=embedding_full.device, requires_grad=True)
        
        if torch.any(torch.isnan(embedding_masked)) or torch.any(torch.isinf(embedding_masked)):
            return torch.tensor(1.0, device=embedding_masked.device, requires_grad=True)
        
        # Use F.normalize with eps to handle zero vectors safely
        embedding_full = F.normalize(embedding_full, dim=1, p=2, eps=1e-8)
        embedding_masked = F.normalize(embedding_masked, dim=1, p=2, eps=1e-8)
        
        # Cosine similarity (should be high)
        cosine_sim = (embedding_full * embedding_masked).sum(dim=1)  # (B,)
        
        # Clamp cosine similarity to valid range [-1, 1] to avoid numerical issues
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
        # Loss: 1 - cosine_sim (minimize distance)
        loss = (1.0 - cosine_sim).mean()
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1.0, device=embedding_full.device, requires_grad=True)
        
        return loss
