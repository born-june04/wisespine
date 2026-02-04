#!/usr/bin/env python3
"""
Evaluate Assembly Transformer Model

Evaluates the trained assembly transformer on test set:
1. Ordering accuracy
2. Translation error (mean, median, std)
3. Rotation error (mean, median, std) in degrees
4. Completion error (if applicable)
5. Per-vertebra-type breakdown
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import sys
import math
from tqdm import tqdm
from collections import defaultdict
from contextlib import nullcontext

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SpineAssemblyTransformer, SpineAssemblySpinalField, compute_losses
from models.assembly_losses import geodesic_distance
from utils.assembly_data_loader import AssemblyDataset
from torch.utils.data import DataLoader


def compute_next_index_gt(vertebra_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """GT adjacency: next vertebra index in ascending GT order."""
    B, N = vertebra_ids.shape
    device = vertebra_ids.device
    next_index = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 2:
            continue
        ids = vertebra_ids[b][valid]
        order = torch.argsort(ids)
        idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        ordered_idx = idx_valid[order]
        for i in range(len(ordered_idx) - 1):
            next_index[b, ordered_idx[i]] = ordered_idx[i + 1]
    return next_index


def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[-1, -1])


def order_by_s(pred_s: torch.Tensor, valid_mask: torch.Tensor) -> list[int]:
    """Return indices sorted by predicted s (valid only)."""
    s_vals = pred_s.squeeze(-1)
    s_vals = s_vals[valid_mask]
    order = torch.argsort(s_vals)
    idx_valid = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    return idx_valid[order].tolist()


def order_by_gt_ids(vertebra_ids: torch.Tensor, valid_mask: torch.Tensor) -> list[int]:
    """Return indices sorted by GT vertebra IDs (valid only)."""
    ids = vertebra_ids[valid_mask]
    order = torch.argsort(ids)
    idx_valid = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    return idx_valid[order].tolist()


def cobb_angle_from_points(points_3d: np.ndarray) -> float:
    """Compute Cobb-like angle using PCA projection and upper/lower line fit."""
    if points_3d.shape[0] < 6:
        return float('nan')
    mean = points_3d.mean(axis=0)
    centered = points_3d - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]]
    proj = centered @ basis
    order_y = np.argsort(proj[:, 1])
    n = len(order_y)
    lower = proj[order_y[:n // 3]]
    upper = proj[order_y[-n // 3:]]

    def fit_dir(pts):
        if len(pts) < 2:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        a, _ = np.polyfit(x, y, 1)
        v = np.array([1.0, a])
        v = v / np.linalg.norm(v)
        return v

    v1 = fit_dir(lower)
    v2 = fit_dir(upper)
    if v1 is None or v2 is None:
        return float('nan')
    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(v1, v2)), -1, 1)))
    return float(angle)


def bspline_basis(s: torch.Tensor, K: int, degree: int = 3) -> torch.Tensor:
    """
    Compute B-spline basis for each s in [0,1].
    s: (n,) or (B,N)
    Returns: (1, n, K) or (B, N, K)
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)  # (1, n)
    elif s.dim() == 2:
        s = s  # (B, N)

    B, N = s.shape
    device = s.device
    knots = torch.linspace(0.0, 1.0, K + degree + 1, device=device)

    # Initial basis (degree 0)
    N0 = torch.zeros(B, N, K, device=device)
    for i in range(K):
        left = knots[i]
        right = knots[i + 1]
        cond = (s >= left) & (s < right)
        N0[:, :, i] = cond.float()

    # Cox-de Boor recursion
    N_prev = N0
    for d in range(1, degree + 1):
        N_curr = torch.zeros(B, N, K, device=device)
        for i in range(K):
            left_denom = knots[i + d] - knots[i]
            right_denom = knots[i + d + 1] - knots[i + 1]
            left = 0.0
            right = 0.0
            if left_denom > 0:
                left = ((s - knots[i]) / left_denom) * N_prev[:, :, i]
            if right_denom > 0 and i + 1 < K:
                right = ((knots[i + d + 1] - s) / right_denom) * N_prev[:, :, i + 1]
            N_curr[:, :, i] = left + right
        N_prev = N_curr

    return N_prev


def bspline_fit_and_smooth(
    control_points: torch.Tensor,
    centroids: torch.Tensor,
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fit_loss, smooth_loss)
    fit_loss: mean squared distance between spline and centroids
    smooth_loss: mean squared second differences of control points
    """
    B, N, _ = centroids.shape
    K = control_points.shape[1]
    device = centroids.device
    fit_total = torch.tensor(0.0, device=device)
    count = 0
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 3:
            continue
        ids = vertebra_ids[b][valid]
        cents = centroids[b][valid]
        order = torch.argsort(ids)
        cents = cents[order]
        cents = cents - cents[0:1]
        n = cents.shape[0]
        s = torch.linspace(0.0, 1.0, n, device=device)
        basis = bspline_basis(s, K).to(torch.float32)  # (1, n, K)
        cp = control_points[b].to(torch.float32)
        cents_f = cents.to(torch.float32)
        pred = torch.matmul(basis.squeeze(0), cp)  # (n, 3)
        fit_total = fit_total + torch.mean(torch.sum((pred - cents_f) ** 2, dim=-1))
        count += 1
    if count > 0:
        fit_total = fit_total / count

    smooth = torch.tensor(0.0, device=device)
    if K >= 3:
        diff2 = control_points.to(torch.float32)[:, 2:] - 2 * control_points.to(torch.float32)[:, 1:-1] + control_points.to(torch.float32)[:, :-2]
        smooth = torch.mean(diff2 ** 2)
    return fit_total, smooth


def spine_s_metrics(
    s: torch.Tensor,
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[list[float], list[float]]:
    """
    Returns (monotonicity_list, spearman_list)
    monotonicity: fraction of adjacent pairs with s_{i+1} > s_i in GT order
    spearman: rank correlation between s and GT order
    """
    mono_list = []
    rho_list = []
    B, N, _ = s.shape
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 2:
            continue
        ids = vertebra_ids[b][valid]
        s_vals = s[b][valid].squeeze(-1)
        order = torch.argsort(ids)
        s_sorted = s_vals[order]
        diffs = s_sorted[1:] - s_sorted[:-1]
        mono = (diffs > 0).float().mean().item()
        mono_list.append(mono)

        n = s_sorted.shape[0]
        gt_rank = torch.arange(n, device=s_sorted.device, dtype=torch.float32)
        s_rank = torch.argsort(torch.argsort(s_sorted)).float()
        denom = n * (n**2 - 1) + 1e-8
        rho = 1.0 - 6.0 * torch.sum((s_rank - gt_rank) ** 2) / denom
        rho_list.append(rho.item())
    return mono_list, rho_list


def collate_assembly_batch(batch):
    """Collate function for assembly batch (module-level for multiprocessing)"""
    embeddings = torch.stack([item['embeddings'] for item in batch])  # (B, N, embed_dim)
    vertebra_ids = torch.stack([item['vertebra_ids'] for item in batch])  # (B, N)
    mask = torch.stack([item['mask'] for item in batch])  # (B, N)
    subject_ids = [item['subject_id'] for item in batch]
    points = torch.stack([item['points'] for item in batch])  # (B, N, M, 3)
    features = torch.stack([item['features'] for item in batch])  # (B, N, M, 5)
    
    # Assembly targets
    translations = torch.stack([item['assembly']['translation'] for item in batch])  # (B, N, 3)
    rotations = torch.stack([item['assembly']['rotation'] for item in batch])  # (B, N, 4)
    
    # Missing completion targets (zeros for now)
    missing_embeddings = torch.stack([item['missing_completion'] for item in batch])  # (B, N, embed_dim)
    
    return {
        'embeddings': embeddings,
        'vertebra_ids': vertebra_ids,
        'mask': mask,
        'subject_ids': subject_ids,
        'points': points,
        'features': features,
        'targets': {
            'ordering': vertebra_ids,
            'assembly': {
                'translation': translations,
                'rotation': rotations,
            },
            'missing_completion': missing_embeddings,
        },
    }


def load_model(model_path: Path, device: torch.device):
    """Load trained assembly transformer model"""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = {
            'embed_dim': 512,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'max_vertebrae': 30,
        }
    
    # Detect model type from state dict keys
    # Spinal field model has 'field_pool', 's_head', 'cond' etc.
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_spinal_field = any('field_pool' in k or 's_head' in k or 'delta_pose_head' in k for k in state_dict.keys())
    
    # Create model based on detected type
    if has_spinal_field:
        # Check if delta_pose is enabled
        has_delta_pose = any('delta_pose_head' in k for k in state_dict.keys())
        model = SpineAssemblySpinalField(
            embed_dim=config.get('embed_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            dropout=0.1,
            num_vertebra_types=26,
            use_mask_token=True,
            enable_delta_pose=has_delta_pose,
        )
        print("✓ Detected Spinal Field model")
    else:
        model = SpineAssemblyTransformer(
            embed_dim=config.get('embed_dim', 512),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            dropout=0.1,
            num_vertebra_types=26,
            use_mask_token=True,
        )
        print("✓ Detected Baseline model")
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {config}")
    return model, config


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
):
    """Evaluate model on dataset"""
    model.eval()
    
    # Metrics accumulators
    total_loss = 0.0
    loss_components = defaultdict(float)
    
    ordering_correct = 0
    ordering_total = 0
    ordering_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Confusion matrix for ordering
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # [true_type][pred_type] -> count
    
    translation_errors = []
    rotation_errors = []
    completion_errors = []
    bspline_fit_errors = []
    bspline_smooth_errors = []
    s_monotonicity = []
    s_spearman = []
    delta_pose_translation_errors = []
    delta_pose_rotation_errors = []
    nvc_scores = []
    soed_scores = []
    delta_trans_smoothness = []
    delta_rot_smoothness = []
    cobb_sensitivity = []
    
    # Per-vertebra-type errors
    translation_errors_by_type = defaultdict(list)
    rotation_errors_by_type = defaultdict(list)
    
    num_batches = 0
    
    # Loss weights (for loss computation)
    loss_weights = {
        'ordering': 1.0,
        'translation': 1.0,
        'rotation': 1.0,
        'completion': 1.0,
    }
    
    autocast_context = autocast(device_type='cuda', dtype=torch.float16) if use_amp and device.type == 'cuda' else nullcontext()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            embeddings = batch['embeddings'].to(device)
            pad_mask = ~batch['mask'].to(device)
            vertebra_ids = batch['vertebra_ids'].to(device)
            mask_mask = torch.zeros_like(pad_mask, dtype=torch.bool, device=device)
            
            # Prepare targets
            targets = batch['targets']
            gt_types = targets['ordering'].to(device)
            gt_t = targets['assembly']['translation'].to(device)
            
            # Convert quaternion to rotation matrix
            if 'rotation' in targets['assembly']:
                gt_rot = targets['assembly']['rotation'].to(device)  # (B, N, 4) [x, y, z, w]
                x, y, z, w = gt_rot[..., 0], gt_rot[..., 1], gt_rot[..., 2], gt_rot[..., 3]
                norm = torch.sqrt(x**2 + y**2 + z**2 + w**2 + 1e-8)
                x, y, z, w = x/norm, y/norm, z/norm, w/norm
                gt_R = torch.stack([
                    torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], dim=-1),
                    torch.stack([2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)], dim=-1),
                    torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)], dim=-1),
                ], dim=-2)  # (B, N, 3, 3)
            else:
                gt_R = None
            
            gt_embedding = embeddings.clone()
            
            # Forward pass
            with autocast_context:
                predictions = model(embeddings, pad_mask=pad_mask, mask_mask=mask_mask)
            
            # Compute loss
            losses = compute_losses(
                predictions,
                gt_types=gt_types,
                gt_t=gt_t,
                gt_R=gt_R,
                gt_embedding=gt_embedding if mask_mask.any() else None,
                w_order=loss_weights['ordering'],
                w_trans=loss_weights['translation'],
                w_rot=loss_weights['rotation'],
                w_comp=loss_weights['completion'],
            )
            
            # Accumulate losses
            total_loss += losses['loss_total'].item()
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] += value.item()
                else:
                    loss_components[key] += value
            
            # Compute metrics
            valid = ~pad_mask  # (B, N)
            
            # 1. Ordering Accuracy
            if gt_types is not None:
                ordering_logits = predictions['ordering']  # (B, N, num_types+1)
                pred_types = ordering_logits.argmax(dim=-1)  # (B, N)
                order_valid = valid & (gt_types >= 0) & (gt_types < ordering_logits.shape[-1])
                
                if order_valid.any():
                    correct = (pred_types[order_valid] == gt_types[order_valid])
                    total_valid = order_valid.sum().item()
                    ordering_correct += correct.sum().item()
                    ordering_total += total_valid
                    
                    # Per-type accuracy and confusion matrix
                    for b in range(gt_types.shape[0]):
                        for n in range(gt_types.shape[1]):
                            if order_valid[b, n]:
                                true_type = gt_types[b, n].item()
                                pred_type = pred_types[b, n].item()
                                ordering_by_type[true_type]['total'] += 1
                                confusion_matrix[true_type][pred_type] += 1
                                if pred_type == true_type:
                                    ordering_by_type[true_type]['correct'] += 1
            
            # 2. Translation Error
            if gt_t is not None:
                t_pred = predictions['pose']['t']  # (B, N, 3)
                trans_valid = valid
                if trans_valid.any():
                    diff = t_pred[trans_valid] - gt_t[trans_valid]  # (N_valid, 3)
                    errors = torch.norm(diff, dim=-1)  # (N_valid,)
                    translation_errors.append(errors.cpu().numpy())
                    
                    # Per-type errors
                    for b in range(gt_types.shape[0]):
                        for n in range(gt_types.shape[1]):
                            if trans_valid[b, n] and gt_types[b, n] >= 0:
                                v_type = gt_types[b, n].item()
                                error = torch.norm(t_pred[b, n] - gt_t[b, n]).item()
                                translation_errors_by_type[v_type].append(error)
            
            # 3. Rotation Error
            if gt_R is not None:
                R_pred = predictions['pose']['R']  # (B, N, 3, 3)
                rot_valid = valid
                if rot_valid.any():
                    Rp = R_pred[rot_valid].float()
                    Rg = gt_R[rot_valid].float()
                    angles = geodesic_distance(Rp, Rg)  # (N_valid,) in radians
                    angles_deg = angles * 180.0 / math.pi  # Convert to degrees
                    rotation_errors.append(angles_deg.cpu().numpy())
                    
                    # Per-type errors
                    for b in range(gt_types.shape[0]):
                        for n in range(gt_types.shape[1]):
                            if rot_valid[b, n] and gt_types[b, n] >= 0:
                                v_type = gt_types[b, n].item()
                                angle = geodesic_distance(
                                    R_pred[b:b+1, n:n+1].float(),
                                    gt_R[b:b+1, n:n+1].float()
                                )[0] * 180.0 / math.pi
                                rotation_errors_by_type[v_type].append(angle.item())
            
            # 4. Completion Error
            if mask_mask.any() and gt_embedding is not None:
                pred_emb = predictions['missing_completion']
                comp_valid = mask_mask & (~pad_mask)
                if comp_valid.any():
                    pred_emb_valid = pred_emb[comp_valid]
                    gt_emb_valid = gt_embedding[comp_valid]
                    pred_emb_norm = F.normalize(pred_emb_valid, p=2, dim=-1)
                    gt_emb_norm = F.normalize(gt_emb_valid, p=2, dim=-1)
                    cos_sim = (pred_emb_norm * gt_emb_norm).sum(dim=-1)
                    errors = 1.0 - cos_sim
                    completion_errors.append(errors.cpu().numpy())

            # 5. Spinal Field metrics (if available)
            if 'spine_field' in predictions:
                control_points = predictions['spine_field']['control_points']
                centroids = batch['points'].mean(dim=2).to(device)
                fit_loss, smooth_loss = bspline_fit_and_smooth(
                    control_points,
                    centroids,
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                )
                bspline_fit_errors.append(fit_loss.detach().cpu().numpy())
                bspline_smooth_errors.append(smooth_loss.detach().cpu().numpy())

                s_vals = predictions['spine_field']['s']
                mono_list, rho_list = spine_s_metrics(
                    s_vals,
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                )
                s_monotonicity.extend(mono_list)
                s_spearman.extend(rho_list)

            # 6. Structure-aware metrics
            if gt_t is not None and gt_types is not None:
                t_pred = predictions['pose']['t']
                R_pred = predictions['pose']['R']
                pred_types = predictions['ordering'].argmax(dim=-1)
                for b in range(gt_types.shape[0]):
                    valid_b = valid[b]
                    if valid_b.sum() < 3:
                        continue
                    # Ordering edit distance (SOED)
                    gt_order_idx = order_by_gt_ids(gt_types[b], valid_b)
                    pred_order_idx = order_by_s(predictions['spine_field']['s'][b], valid_b)
                    gt_seq = [int(gt_types[b, i].item()) for i in gt_order_idx]
                    pred_seq = [int(pred_types[b, i].item()) for i in pred_order_idx]
                    soed = edit_distance(gt_seq, pred_seq) / max(1, len(gt_seq))
                    soed_scores.append(soed)

                    # Neighbor Vertebra Consistency (NVC) - cosine similarity of delta translations
                    if len(gt_order_idx) >= 2:
                        gt_idx = gt_order_idx
                        dp = []
                        dg = []
                        for i in range(len(gt_idx) - 1):
                            i0, i1 = gt_idx[i], gt_idx[i + 1]
                            dp.append((t_pred[b, i1] - t_pred[b, i0]).unsqueeze(0))
                            dg.append((gt_t[b, i1] - gt_t[b, i0]).unsqueeze(0))
                        dp = torch.cat(dp, dim=0)
                        dg = torch.cat(dg, dim=0)
                        cos = F.cosine_similarity(dp, dg, dim=-1)
                        nvc_scores.append(float(cos.mean().item()))

                    # Relative pose smoothness (variance of adjacent deltas)
                    if len(pred_order_idx) >= 2:
                        t_sorted = t_pred[b, pred_order_idx]
                        d_t = t_sorted[1:] - t_sorted[:-1]
                        d_norm = torch.norm(d_t, dim=-1)
                        delta_trans_smoothness.append(float(torch.var(d_norm).item()))

                        R_sorted = R_pred[b, pred_order_idx]
                        dR = torch.matmul(R_sorted[:-1].transpose(-1, -2), R_sorted[1:]).float()
                        ang = geodesic_distance(dR, torch.eye(3, device=device).expand_as(dR).float())
                        delta_rot_smoothness.append(float(torch.var(ang).item()))

                    # Cobb-like angle sensitivity to neighbor swaps
                    pts = t_pred[b, pred_order_idx].detach().cpu().numpy()
                    base_angle = cobb_angle_from_points(pts)
                    if len(pred_order_idx) >= 3 and np.isfinite(base_angle):
                        deltas = []
                        for i in range(len(pred_order_idx) - 1):
                            swap_pts = pts.copy()
                            swap_pts[[i, i + 1]] = swap_pts[[i + 1, i]]
                            swap_angle = cobb_angle_from_points(swap_pts)
                            if np.isfinite(swap_angle):
                                deltas.append(abs(swap_angle - base_angle))
                        if deltas:
                            cobb_sensitivity.append(float(np.mean(deltas)))

            # 6. Delta pose metrics (if available)
            if 'delta_pose' in predictions and predictions['delta_pose']['d_t'] is not None and gt_t is not None:
                d_t = predictions['delta_pose']['d_t']  # (B, N, 3)
                d_R = predictions['delta_pose'].get('d_R', None)
                # Use GT adjacency for stable delta metrics
                next_index = compute_next_index_gt(
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                )

                valid_mask = batch['mask'].to(device)
                next_valid = torch.gather(valid_mask, 1, next_index)
                idx_self = torch.arange(valid_mask.shape[1], device=device).unsqueeze(0).expand_as(next_index)
                pair_valid = valid_mask & next_valid & (next_index != idx_self)

                if pair_valid.any():
                    gt_t_next = torch.gather(gt_t, 1, next_index.unsqueeze(-1).expand(-1, -1, 3))
                    gt_d_t = gt_t_next - gt_t
                    dt_err = torch.norm(d_t - gt_d_t, dim=-1)
                    delta_pose_translation_errors.append(dt_err[pair_valid].cpu().numpy())

                    if d_R is not None and gt_R is not None:
                        gt_R_next = torch.gather(
                            gt_R,
                            1,
                            next_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
                        )
                        gt_R_rel = torch.matmul(gt_R.transpose(-1, -2), gt_R_next)
                        rot_err = geodesic_distance(d_R[pair_valid], gt_R_rel[pair_valid]) * 180.0 / math.pi
                        delta_pose_rotation_errors.append(rot_err.cpu().numpy())
            
            num_batches += 1
    
    # Aggregate metrics
    metrics = {}
    
    # Losses
    metrics['total_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    for key, value in loss_components.items():
        metrics[key] = value / num_batches if num_batches > 0 else 0.0
    
    # Ordering accuracy
    if ordering_total > 0:
        metrics['ordering_accuracy'] = ordering_correct / ordering_total
        metrics['ordering_correct'] = ordering_correct
        metrics['ordering_total'] = ordering_total
    else:
        metrics['ordering_accuracy'] = 0.0
    
    # Translation error
    if translation_errors:
        all_trans_errors = np.concatenate(translation_errors)
        metrics['translation_error_mean'] = float(np.mean(all_trans_errors))
        metrics['translation_error_median'] = float(np.median(all_trans_errors))
        metrics['translation_error_std'] = float(np.std(all_trans_errors))
        metrics['translation_error_min'] = float(np.min(all_trans_errors))
        metrics['translation_error_max'] = float(np.max(all_trans_errors))
    else:
        metrics['translation_error_mean'] = 0.0
        metrics['translation_error_median'] = 0.0
        metrics['translation_error_std'] = 0.0
    
    # Rotation error
    if rotation_errors:
        all_rot_errors = np.concatenate(rotation_errors)
        metrics['rotation_error_mean'] = float(np.mean(all_rot_errors))
        metrics['rotation_error_median'] = float(np.median(all_rot_errors))
        metrics['rotation_error_std'] = float(np.std(all_rot_errors))
        metrics['rotation_error_min'] = float(np.min(all_rot_errors))
        metrics['rotation_error_max'] = float(np.max(all_rot_errors))
    else:
        metrics['rotation_error_mean'] = 0.0
        metrics['rotation_error_median'] = 0.0
        metrics['rotation_error_std'] = 0.0
    
    # Completion error
    if completion_errors:
        all_comp_errors = np.concatenate(completion_errors)
        metrics['completion_error_mean'] = float(np.mean(all_comp_errors))
    else:
        metrics['completion_error_mean'] = 0.0

    # B-spline metrics
    if bspline_fit_errors:
        fit_vals = np.array(bspline_fit_errors)
        metrics['bspline_fit_mean'] = float(np.mean(fit_vals))
        metrics['bspline_fit_std'] = float(np.std(fit_vals))
    else:
        metrics['bspline_fit_mean'] = 0.0
        metrics['bspline_fit_std'] = 0.0

    if bspline_smooth_errors:
        smooth_vals = np.array(bspline_smooth_errors)
        metrics['bspline_smooth_mean'] = float(np.mean(smooth_vals))
        metrics['bspline_smooth_std'] = float(np.std(smooth_vals))
    else:
        metrics['bspline_smooth_mean'] = 0.0
        metrics['bspline_smooth_std'] = 0.0

    # Spine coordinate metrics
    if s_monotonicity:
        metrics['s_monotonicity_mean'] = float(np.mean(s_monotonicity))
        metrics['s_monotonicity_median'] = float(np.median(s_monotonicity))
    else:
        metrics['s_monotonicity_mean'] = 0.0
        metrics['s_monotonicity_median'] = 0.0

    if s_spearman:
        metrics['s_spearman_mean'] = float(np.mean(s_spearman))
        metrics['s_spearman_median'] = float(np.median(s_spearman))
    else:
        metrics['s_spearman_mean'] = 0.0
        metrics['s_spearman_median'] = 0.0

    # Delta pose metrics
    if delta_pose_translation_errors:
        all_dt = np.concatenate(delta_pose_translation_errors)
        metrics['delta_pose_translation_mean'] = float(np.mean(all_dt))
        metrics['delta_pose_translation_median'] = float(np.median(all_dt))
    else:
        metrics['delta_pose_translation_mean'] = 0.0
        metrics['delta_pose_translation_median'] = 0.0

    if delta_pose_rotation_errors:
        all_drot = np.concatenate(delta_pose_rotation_errors)
        metrics['delta_pose_rotation_mean'] = float(np.mean(all_drot))
        metrics['delta_pose_rotation_median'] = float(np.median(all_drot))
    else:
        metrics['delta_pose_rotation_mean'] = 0.0
        metrics['delta_pose_rotation_median'] = 0.0

    # Structure-aware metrics
    if nvc_scores:
        metrics['nvc_mean'] = float(np.mean(nvc_scores))
        metrics['nvc_median'] = float(np.median(nvc_scores))
    else:
        metrics['nvc_mean'] = 0.0
        metrics['nvc_median'] = 0.0

    if soed_scores:
        metrics['soed_mean'] = float(np.mean(soed_scores))
        metrics['soed_median'] = float(np.median(soed_scores))
    else:
        metrics['soed_mean'] = 0.0
        metrics['soed_median'] = 0.0

    if delta_trans_smoothness:
        metrics['delta_trans_smooth_mean'] = float(np.mean(delta_trans_smoothness))
    else:
        metrics['delta_trans_smooth_mean'] = 0.0

    if delta_rot_smoothness:
        metrics['delta_rot_smooth_mean'] = float(np.mean(delta_rot_smoothness))
    else:
        metrics['delta_rot_smooth_mean'] = 0.0

    if cobb_sensitivity:
        metrics['cobb_sensitivity_mean'] = float(np.mean(cobb_sensitivity))
    else:
        metrics['cobb_sensitivity_mean'] = 0.0
    
    # Per-type metrics
    metrics['ordering_by_type'] = {}
    for v_type, stats in ordering_by_type.items():
        if stats['total'] > 0:
            metrics['ordering_by_type'][int(v_type)] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total'],
            }
    
    metrics['translation_error_by_type'] = {}
    for v_type, errors in translation_errors_by_type.items():
        if len(errors) > 0:
            metrics['translation_error_by_type'][int(v_type)] = {
                'mean': float(np.mean(errors)),
                'median': float(np.median(errors)),
                'std': float(np.std(errors)),
                'count': len(errors),
            }
    
    metrics['rotation_error_by_type'] = {}
    for v_type, errors in rotation_errors_by_type.items():
        if len(errors) > 0:
            metrics['rotation_error_by_type'][int(v_type)] = {
                'mean': float(np.mean(errors)),
                'median': float(np.median(errors)),
                'std': float(np.std(errors)),
                'count': len(errors),
            }
    
    # Convert confusion matrix to dict
    metrics['confusion_matrix'] = {
        int(true_type): {int(pred_type): count for pred_type, count in pred_dict.items()}
        for true_type, pred_dict in confusion_matrix.items()
    }
    
    # Find most confused pairs
    confused_pairs = []
    for true_type, pred_dict in confusion_matrix.items():
        for pred_type, count in pred_dict.items():
            if pred_type != true_type and count > 0:
                confused_pairs.append({
                    'true_type': int(true_type),
                    'pred_type': int(pred_type),
                    'count': count,
                })
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    metrics['top_confusions'] = confused_pairs[:20]  # Top 20 most confused pairs
    
    return metrics


def print_results(metrics: dict, output_file: Path = None):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("Assembly Model Evaluation Results")
    print("="*60)
    
    print("\n[Overall Metrics]")
    print(f"  Total Loss: {metrics['total_loss']:.4f}")
    if 'loss_ordering' in metrics:
        print(f"  Ordering Loss: {metrics['loss_ordering']:.4f}")
    if 'loss_translation' in metrics:
        print(f"  Translation Loss: {metrics['loss_translation']:.4f}")
    if 'loss_rotation' in metrics:
        print(f"  Rotation Loss: {metrics['loss_rotation']:.4f}")
    if 'loss_completion' in metrics:
        print(f"  Completion Loss: {metrics['loss_completion']:.4f}")
    
    print("\n[Ordering Task]")
    if 'ordering_accuracy' in metrics:
        print(f"  Accuracy: {metrics['ordering_accuracy']:.2%}")
        print(f"  Correct: {metrics.get('ordering_correct', 0)} / {metrics.get('ordering_total', 0)}")
    
    print("\n[Translation Task]")
    if 'translation_error_mean' in metrics:
        print(f"  Mean Error: {metrics['translation_error_mean']:.4f} mm")
        print(f"  Median Error: {metrics['translation_error_median']:.4f} mm")
        print(f"  Std Error: {metrics['translation_error_std']:.4f} mm")
        print(f"  Range: [{metrics.get('translation_error_min', 0):.4f}, {metrics.get('translation_error_max', 0):.4f}] mm")
    
    print("\n[Rotation Task]")
    if 'rotation_error_mean' in metrics:
        print(f"  Mean Error: {metrics['rotation_error_mean']:.4f}°")
        print(f"  Median Error: {metrics['rotation_error_median']:.4f}°")
        print(f"  Std Error: {metrics['rotation_error_std']:.4f}°")
        print(f"  Range: [{metrics.get('rotation_error_min', 0):.4f}, {metrics.get('rotation_error_max', 0):.4f}]°")
    
    if 'completion_error_mean' in metrics and metrics['completion_error_mean'] > 0:
        print("\n[Completion Task]")
        print(f"  Mean Error: {metrics['completion_error_mean']:.4f} (cosine distance)")

    if metrics.get('bspline_fit_mean', 0.0) > 0 or metrics.get('bspline_smooth_mean', 0.0) > 0:
        print("\n[B-spline Metrics]")
        print(f"  Fit Error: {metrics.get('bspline_fit_mean', 0.0):.4f} ± {metrics.get('bspline_fit_std', 0.0):.4f}")
        print(f"  Smoothness: {metrics.get('bspline_smooth_mean', 0.0):.4f} ± {metrics.get('bspline_smooth_std', 0.0):.4f}")

    if metrics.get('s_monotonicity_mean', 0.0) > 0 or metrics.get('s_spearman_mean', 0.0) > 0:
        print("\n[Spine Coordinate Metrics]")
        print(f"  s Monotonicity: {metrics.get('s_monotonicity_mean', 0.0):.4f} (median {metrics.get('s_monotonicity_median', 0.0):.4f})")
        print(f"  s Spearman: {metrics.get('s_spearman_mean', 0.0):.4f} (median {metrics.get('s_spearman_median', 0.0):.4f})")

    if metrics.get('delta_pose_translation_mean', 0.0) > 0 or metrics.get('delta_pose_rotation_mean', 0.0) > 0:
        print("\n[Delta Pose Metrics]")
        print(f"  Δt Error: {metrics.get('delta_pose_translation_mean', 0.0):.4f} mm (median {metrics.get('delta_pose_translation_median', 0.0):.4f})")
        print(f"  ΔR Error: {metrics.get('delta_pose_rotation_mean', 0.0):.4f}° (median {metrics.get('delta_pose_rotation_median', 0.0):.4f})")

    if metrics.get('nvc_mean', 0.0) > 0 or metrics.get('soed_mean', 0.0) > 0:
        print("\n[Structure Metrics]")
        print(f"  NVC (cosine): {metrics.get('nvc_mean', 0.0):.4f} (median {metrics.get('nvc_median', 0.0):.4f})")
        print(f"  SOED: {metrics.get('soed_mean', 0.0):.4f} (median {metrics.get('soed_median', 0.0):.4f})")
        print(f"  ΔT Smoothness: {metrics.get('delta_trans_smooth_mean', 0.0):.4f}")
        print(f"  ΔR Smoothness: {metrics.get('delta_rot_smooth_mean', 0.0):.4f}")
        print(f"  Cobb Sensitivity: {metrics.get('cobb_sensitivity_mean', 0.0):.4f}°")
    
    # Per-type breakdown (top 10)
    if 'ordering_by_type' in metrics and metrics['ordering_by_type']:
        print("\n[Ordering Accuracy by Vertebra Type] (Top 10)")
        sorted_types = sorted(
            metrics['ordering_by_type'].items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:10]
        for v_type, stats in sorted_types:
            print(f"  Type {v_type}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    if 'translation_error_by_type' in metrics and metrics['translation_error_by_type']:
        print("\n[Translation Error by Vertebra Type] (Top 10)")
        sorted_types = sorted(
            metrics['translation_error_by_type'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]
        for v_type, stats in sorted_types:
            print(f"  Type {v_type}: {stats['mean']:.4f} ± {stats['std']:.4f} mm (n={stats['count']})")
    
    if 'rotation_error_by_type' in metrics and metrics['rotation_error_by_type']:
        print("\n[Rotation Error by Vertebra Type] (Top 10)")
        sorted_types = sorted(
            metrics['rotation_error_by_type'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:10]
        for v_type, stats in sorted_types:
            print(f"  Type {v_type}: {stats['mean']:.4f} ± {stats['std']:.4f}° (n={stats['count']})")
    
    # Confusion analysis
    if 'top_confusions' in metrics and metrics['top_confusions']:
        print("\n[Top Confusions] (Most Common Misclassifications)")
        for i, conf in enumerate(metrics['top_confusions'][:10], 1):
            print(f"  {i}. True: {conf['true_type']} → Pred: {conf['pred_type']} ({conf['count']} times)")
    
    print("\n" + "="*60)
    
    # Save to file
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Assembly Transformer Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory containing pre-extracted embeddings')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing original (non-centered) point clouds')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--max_vertebrae', type=int, default=30,
                        help='Maximum number of vertebrae per subject')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Assembly Model Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Mixed precision (AMP): {args.use_amp}")
    print("="*60)
    print()
    
    # Load model
    model, config = load_model(Path(args.model_path), device)
    
    # Create dataset
    dataset = AssemblyDataset(
        embedding_dir=Path(args.embedding_dir),
        point_cloud_dir=Path(args.point_cloud_dir),
        split=args.split,
        max_vertebrae=args.max_vertebrae,
        augment=False,
    )
    
    print(f"Loaded {len(dataset)} samples from {args.split} split")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_assembly_batch,
    )
    
    # Evaluate
    metrics = evaluate_model(
        model,
        dataloader,
        device,
        use_amp=args.use_amp,
    )
    
    # Print and save results
    output_file = output_dir / 'evaluation_results.json'
    print_results(metrics, output_file)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

