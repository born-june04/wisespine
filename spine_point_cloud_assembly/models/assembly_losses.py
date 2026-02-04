"""
Loss Functions for Assembly Tasks

1. Ordering Loss: Cross-entropy for vertebra ID prediction
2. Assembly Loss: L2 loss for relative positions and rotations (6D rotation support)
3. Missing Completion Loss: L2 loss for missing vertebra embedding prediction
4. compute_losses: Unified loss computation function (from assembly.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# Import rotation utilities from assembly
from .assembly import geodesic_distance


class OrderingLoss(nn.Module):
    """
    Loss for vertebra ordering/ID prediction.
    
    Uses cross-entropy loss for classification.
    """
    
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        target_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, N, num_classes) - predicted logits
            target_ids: (B, N) - ground truth vertebra IDs (0-25 for V1-V26, -1 for missing)
            mask: (B, N) - boolean mask indicating present vertebrae (True = present)
        Returns:
            loss: scalar tensor
        """
        B, N, num_classes = pred_logits.shape
        
        # Flatten for cross-entropy
        pred_flat = pred_logits.view(B * N, num_classes)  # (B*N, num_classes)
        target_flat = target_ids.view(B * N)  # (B*N,)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(B * N)  # (B*N,)
            # Only compute loss for present vertebrae
            valid_mask = mask_flat & (target_flat != self.ignore_index)
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
            
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
        else:
            # Ignore missing vertebrae (where target_id == -1)
            valid_mask = target_flat != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
            
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
        
        loss = self.criterion(pred_flat, target_flat)
        return loss


class AssemblyLoss(nn.Module):
    """
    Loss for assembly task (relative positions and rotations).
    
    Uses L2 loss for translation and quaternion loss for rotation.
    """
    
    def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
    
    def forward(
        self,
        pred_translation: torch.Tensor,
        pred_rotation: torch.Tensor,
        target_translation: torch.Tensor,
        target_rotation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_translation: (B, N, 3) - predicted relative translation
            pred_rotation: (B, N, 4) - predicted quaternion rotation
            target_translation: (B, N, 3) - ground truth relative translation
            target_rotation: (B, N, 4) - ground truth quaternion rotation
            mask: (B, N) - boolean mask indicating present vertebrae
        Returns:
            loss_dict: dict with 'translation_loss', 'rotation_loss', 'total_loss'
        """
        if mask is not None:
            # Only compute loss for present vertebrae
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
            
            # Translation loss
            translation_diff = pred_translation - target_translation  # (B, N, 3)
            translation_diff = translation_diff * mask_expanded
            translation_loss = (translation_diff ** 2).sum() / (mask.sum() * 3 + 1e-8)
            
            # Rotation loss: quaternion distance
            # Normalize quaternions
            pred_rot_norm = F.normalize(pred_rotation, p=2, dim=-1)
            target_rot_norm = F.normalize(target_rotation, p=2, dim=-1)
            
            # Quaternion distance: 1 - |q1 · q2|
            # Dot product
            dot_product = (pred_rot_norm * target_rot_norm).sum(dim=-1, keepdim=True)  # (B, N, 1)
            # Clamp to avoid numerical issues
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            # Distance: 1 - |dot|
            rotation_diff = 1.0 - torch.abs(dot_product)  # (B, N, 1)
            rotation_diff = rotation_diff * mask_expanded
            rotation_loss = rotation_diff.sum() / (mask.sum() + 1e-8)
        else:
            # Compute loss for all vertebrae
            translation_loss = F.mse_loss(pred_translation, target_translation)
            
            # Rotation loss
            pred_rot_norm = F.normalize(pred_rotation, p=2, dim=-1)
            target_rot_norm = F.normalize(target_rotation, p=2, dim=-1)
            dot_product = (pred_rot_norm * target_rot_norm).sum(dim=-1)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            rotation_loss = (1.0 - torch.abs(dot_product)).mean()
        
        total_loss = (
            self.translation_weight * translation_loss +
            self.rotation_weight * rotation_loss
        )
        
        return {
            'translation_loss': translation_loss,
            'rotation_loss': rotation_loss,
            'total_loss': total_loss,
        }


class MissingCompletionLoss(nn.Module):
    """
    Loss for missing vertebra completion task.
    
    Uses L2 loss (MSE) for embedding prediction.
    """
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(
        self,
        pred_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_embedding: (B, N, embed_dim) - predicted embeddings
            target_embedding: (B, N, embed_dim) - ground truth embeddings (0 for present vertebrae)
            mask: (B, N) - boolean mask indicating missing vertebrae (True = missing, False = present)
        Returns:
            loss: scalar tensor
        """
        if mask is not None:
            # Only compute loss for missing vertebrae
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
            
            # Zero out predictions for present vertebrae
            pred_masked = pred_embedding * mask_expanded
            target_masked = target_embedding * mask_expanded
            
            # Compute loss only for missing vertebrae
            if mask.sum() > 0:
                loss = F.mse_loss(pred_masked, target_masked, reduction='sum')
                loss = loss / (mask.sum() * pred_embedding.shape[-1] + 1e-8)
            else:
                loss = torch.tensor(0.0, device=pred_embedding.device, requires_grad=True)
        else:
            # Compute loss for all positions
            loss = self.criterion(pred_embedding, target_embedding)
        
        return loss


class CombinedAssemblyLoss(nn.Module):
    """
    Combined loss for all assembly tasks.
    """
    
    def __init__(
        self,
        ordering_weight: float = 1.0,
        assembly_weight: float = 1.0,
        missing_weight: float = 1.0,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ):
        super().__init__()
        self.ordering_weight = ordering_weight
        self.assembly_weight = assembly_weight
        self.missing_weight = missing_weight
        
        self.ordering_loss = OrderingLoss()
        self.assembly_loss = AssemblyLoss(
            translation_weight=translation_weight,
            rotation_weight=rotation_weight,
        )
        self.missing_loss = MissingCompletionLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict with keys:
                - 'ordering': (B, N, num_classes) - ordering logits
                - 'assembly': dict with 'translation' and 'rotation'
                - 'missing_completion': (B, N, embed_dim) - missing embeddings
            targets: dict with keys:
                - 'ordering': (B, N) - vertebra IDs
                - 'assembly': dict with 'translation' and 'rotation'
                - 'missing_completion': (B, N, embed_dim) - target embeddings (0 for present)
            mask: (B, N) - boolean mask indicating present vertebrae
        Returns:
            loss_dict: dict with individual losses and total loss
        """
        losses = {}
        
        # Ordering loss
        if 'ordering' in predictions and 'ordering' in targets:
            ordering_loss = self.ordering_loss(
                predictions['ordering'],
                targets['ordering'],
                mask=mask,
            )
            losses['ordering'] = ordering_loss
        
        # Assembly loss
        if 'assembly' in predictions and 'assembly' in targets:
            assembly_loss_dict = self.assembly_loss(
                predictions['assembly']['translation'],
                predictions['assembly']['rotation'],
                targets['assembly']['translation'],
                targets['assembly']['rotation'],
                mask=mask,
            )
            losses['assembly'] = assembly_loss_dict['total_loss']
            losses['assembly_translation'] = assembly_loss_dict['translation_loss']
            losses['assembly_rotation'] = assembly_loss_dict['rotation_loss']
        
        # Missing completion loss
        if 'missing_completion' in predictions and 'missing_completion' in targets:
            # For missing completion, we want to predict embeddings for missing vertebrae
            # mask should be inverted: True = missing, False = present
            missing_mask = ~mask if mask is not None else None
            missing_loss = self.missing_loss(
                predictions['missing_completion'],
                targets['missing_completion'],
                mask=missing_mask,
            )
            losses['missing_completion'] = missing_loss
        
        # Total loss
        total_loss = (
            self.ordering_weight * losses.get('ordering', torch.tensor(0.0, device=list(predictions.values())[0].device)) +
            self.assembly_weight * losses.get('assembly', torch.tensor(0.0, device=list(predictions.values())[0].device)) +
            self.missing_weight * losses.get('missing_completion', torch.tensor(0.0, device=list(predictions.values())[0].device))
        )
        losses['total'] = total_loss
        
        return losses


# ---------------------------
# Unified loss computation (from assembly.py)
# ---------------------------
def compute_losses(
    out: Dict[str, torch.Tensor],
    gt_types: Optional[torch.Tensor] = None,     # (B,N) long, vertebra type index
    gt_t: Optional[torch.Tensor] = None,         # (B,N,3)
    gt_R: Optional[torch.Tensor] = None,         # (B,N,3,3)
    gt_embedding: Optional[torch.Tensor] = None, # (B,N,D) original encoder embedding
    w_order: float = 1.0,
    w_trans: float = 1.0,
    w_rot: float = 1.0,
    w_comp: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Computes common losses with proper masking.
    - ordering: cross-entropy on non-pad tokens (optionally also exclude mask tokens)
    - translation: L2 on non-pad tokens
    - rotation: geodesic on non-pad tokens
    - completion: MSE on mask tokens only (non-pad)
    
    Args:
        out: Model output dict with keys:
            - 'ordering': (B, N, num_types+1) - ordering logits
            - 'pose': dict with 't' (B, N, 3) and 'R' (B, N, 3, 3)
            - 'missing_completion': (B, N, D) - predicted embeddings
            - 'pad_mask': (B, N) - True where padding
            - 'mask_mask': (B, N) - True where masked for completion
        gt_types: (B, N) - ground truth vertebra type indices
        gt_t: (B, N, 3) - ground truth translations
        gt_R: (B, N, 3, 3) - ground truth rotation matrices
        gt_embedding: (B, N, D) - original encoder embeddings (for completion)
        w_order, w_trans, w_rot, w_comp: Loss weights
    Returns:
        losses: dict with individual losses and 'loss_total'
    """
    ordering_logits = out["ordering"]
    t_pred = out["pose"]["t"]
    R_pred = out["pose"]["R"]
    pred_emb = out["missing_completion"]
    pad_mask = out["pad_mask"]        # True=PAD
    mask_mask = out["mask_mask"]      # True=MASKED vertebra

    valid = ~pad_mask  # (B,N)

    losses = {}
    total = torch.tensor(0.0, device=ordering_logits.device)

    if gt_types is not None:
        # Exclude padding (-1) and invalid indices
        # gt_types: -1 for padding, 0-25 for valid vertebra types
        order_valid = valid & (gt_types >= 0) & (gt_types < ordering_logits.shape[-1])
        # Flatten
        logits = ordering_logits[order_valid]
        targets = gt_types[order_valid]
        if logits.numel() > 0 and targets.numel() > 0:
            L = F.cross_entropy(logits, targets)
        else:
            L = torch.tensor(0.0, device=ordering_logits.device)
        losses["loss_ordering"] = L
        total = total + w_order * L

    if gt_t is not None:
        trans_valid = valid
        if trans_valid.any():
            diff = t_pred[trans_valid] - gt_t[trans_valid]
            L = (diff.pow(2).sum(dim=-1)).mean() if diff.numel() > 0 else torch.tensor(0.0, device=t_pred.device)
        else:
            L = torch.tensor(0.0, device=t_pred.device)
        losses["loss_translation"] = L
        total = total + w_trans * L

    if gt_R is not None:
        rot_valid = valid
        if rot_valid.any():
            Rp = R_pred[rot_valid].float()
            Rg = gt_R[rot_valid].float()
            if Rp.numel() > 0:
                ang = geodesic_distance(Rp, Rg)
                L = ang.mean()
            else:
                L = torch.tensor(0.0, device=R_pred.device)
        else:
            L = torch.tensor(0.0, device=R_pred.device)
        losses["loss_rotation"] = L
        total = total + w_rot * L

    if gt_embedding is not None:
        comp_valid = mask_mask & (~pad_mask)  # only masked vertebrae
        diff = pred_emb[comp_valid] - gt_embedding[comp_valid]
        L = (diff.pow(2).mean()) if diff.numel() > 0 else torch.tensor(0.0, device=pred_emb.device)
        losses["loss_completion"] = L
        total = total + w_comp * L

    losses["loss_total"] = total
    return losses

