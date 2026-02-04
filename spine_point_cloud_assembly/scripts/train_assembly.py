"""
Phase 3: Assembly Training Script

Trains assembly transformer on encoder embeddings to perform:
1. Ordering: Predict vertebra order/ID
2. Assembly: Predict relative positions and rotations
3. Missing Vertebra Completion: Predict missing vertebrae
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys
import os
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import time
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SpineAssemblyTransformer, SpineAssemblySpinalField, compute_losses
from models.assembly_losses import geodesic_distance


def save_run_config(args, output_dir, timestamp, logger):
    """Persist full run args + module flags for ablation tracking."""
    modules = {
        "delta_pose": bool(args.enable_delta_pose),
        "bspline": (args.bspline_weight > 0) or (args.bspline_smooth_weight > 0),
        "s_monotonic": args.s_monotonic_weight > 0,
        "s_smooth": args.s_smooth_weight > 0,
        "spline_lateral": args.spline_lateral_weight > 0,
        "spline_tangent_smooth": args.spline_tangent_smooth_weight > 0,
        "root_anchor": (args.root_anchor_t_weight > 0) or (args.root_anchor_rot_weight > 0),
        "normalize_translation": bool(args.normalize_translation),
        "freeze_ordering_head": bool(args.freeze_ordering_head),
        "train_pose_only": bool(args.train_pose_only),
    }
    payload = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "args": vars(args),
        "modules": modules,
    }
    run_config_path = Path(output_dir) / "run_config.json"
    try:
        with run_config_path.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved run config to: {run_config_path}")
    except Exception as exc:
        logger.warning(f"Failed to save run config: {exc}")


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


def bspline_basis(s: torch.Tensor, num_control_points: int, degree: int = 3) -> torch.Tensor:
    """
    Compute uniform open B-spline basis for s in [0,1].
    s: (N,) or (B,N) tensor
    returns: (B,N,K) basis
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)
    B, N = s.shape
    K = num_control_points
    device = s.device
    s = torch.clamp(s, 0.0, 1.0 - 1e-6)

    # Open uniform knot vector
    num_internal = max(K - degree - 1, 0)
    if num_internal > 0:
        internal = torch.linspace(0.0, 1.0, num_internal + 2, device=device)[1:-1]
        knots = torch.cat([
            torch.zeros(degree + 1, device=device),
            internal,
            torch.ones(degree + 1, device=device)
        ])
    else:
        knots = torch.cat([
            torch.zeros(degree + 1, device=device),
            torch.ones(degree + 1, device=device)
        ])

    # Initialize N_i,0
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


def bspline_loss(
    control_points: torch.Tensor,
    centroids: torch.Tensor,
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
    weight_smooth: float = 0.1,
) -> torch.Tensor:
    """
    control_points: (B, K, 3)
    centroids: (B, N, 3) original centroids
    vertebra_ids: (B, N) 0-based ids
    mask: (B, N) True for valid
    """
    B, N, _ = centroids.shape
    K = control_points.shape[1]
    device = centroids.device
    total = torch.tensor(0.0, device=device)
    count = 0
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 3:
            continue
        ids = vertebra_ids[b][valid]
        cents = centroids[b][valid]
        order = torch.argsort(ids)
        cents = cents[order]
        # relative to first centroid
        cents = cents - cents[0:1]
        n = cents.shape[0]
        s = torch.linspace(0.0, 1.0, n, device=device)
        basis = bspline_basis(s, K)  # (1,n,K)
        pred = torch.matmul(basis.squeeze(0), control_points[b])  # (n,3)
        total = total + torch.mean(torch.sum((pred - cents) ** 2, dim=-1))
        count += 1
    if count > 0:
        total = total / count

    # smoothness on control points
    if K >= 3:
        diff2 = control_points[:, 2:] - 2 * control_points[:, 1:-1] + control_points[:, :-2]
        smooth = torch.mean(diff2 ** 2)
        total = total + weight_smooth * smooth
    return total


def compute_next_index_gt(vertebra_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Build GT next index along spine order (ascending vertebra_ids).
    vertebra_ids: (B, N)
    mask: (B, N) True for valid
    returns next_index: (B, N) long
    """
    B, N = vertebra_ids.shape
    device = vertebra_ids.device
    next_index = torch.arange(N, device=device).view(1, N).expand(B, N).clone()
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
        # last stays self
    return next_index


def get_root_index_gt(
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
    prefer_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Choose a stable root vertebra index per batch.
    Prefer specific vertebra IDs (e.g., L5/S1) if present, else max GT ID.
    vertebra_ids: (B,N), mask: (B,N)
    Returns: (B,) long indices
    """
    B, N = vertebra_ids.shape
    if prefer_ids is None:
        prefer_ids = []
    root_idx = torch.zeros(B, dtype=torch.long, device=vertebra_ids.device)
    for b in range(B):
        valid = mask[b]
        if valid.sum() == 0:
            root_idx[b] = 0
            continue
        ids_b = vertebra_ids[b]
        found = False
        for pid in prefer_ids:
            matches = torch.where(valid & (ids_b == pid))[0]
            if matches.numel() > 0:
                root_idx[b] = matches[0]
                found = True
                break
        if not found:
            masked_ids = ids_b.clone()
            masked_ids[~valid] = -10**6
            root_idx[b] = torch.argmax(masked_ids)
    return root_idx


def get_top_index_gt(
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
    prefer_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Choose a stable top vertebra index per batch.
    Prefer specific vertebra IDs (e.g., T1) if present, else min GT ID.
    """
    B, N = vertebra_ids.shape
    if prefer_ids is None:
        prefer_ids = []
    top_idx = torch.zeros(B, dtype=torch.long, device=vertebra_ids.device)
    for b in range(B):
        valid = mask[b]
        if valid.sum() == 0:
            top_idx[b] = 0
            continue
        ids_b = vertebra_ids[b]
        found = False
        for pid in prefer_ids:
            matches = torch.where(valid & (ids_b == pid))[0]
            if matches.numel() > 0:
                top_idx[b] = matches[0]
                found = True
                break
        if not found:
            masked_ids = ids_b.clone()
            masked_ids[~valid] = 10**6
            top_idx[b] = torch.argmin(masked_ids)
    return top_idx


def compute_spine_length(
    gt_t: torch.Tensor,
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample spine length (sum of adjacent GT distances).
    Returns: (B,) tensor
    """
    B, N, _ = gt_t.shape
    device = gt_t.device
    lengths = torch.zeros(B, device=device)
    next_index = compute_next_index_gt(vertebra_ids, mask)
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 2:
            lengths[b] = 1.0
            continue
        nxt = next_index[b]
        idx = torch.arange(N, device=device)
        pair_valid = valid & (nxt != idx) & torch.gather(valid, 0, nxt)
        if pair_valid.any():
            t_next = gt_t[b, nxt[pair_valid]]
            t_cur = gt_t[b, pair_valid]
            seg = torch.norm(t_next - t_cur, dim=-1)
            lengths[b] = torch.sum(seg)
        else:
            lengths[b] = 1.0
    return lengths.clamp_min(1e-3)


def s_pairwise_ranking_loss(
    s: torch.Tensor,
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Pairwise ranking loss enforcing s_i < s_{i+1} for GT-adjacent vertebrae.
    s: (B,N,1)
    """
    B, N, _ = s.shape
    device = s.device
    total = torch.tensor(0.0, device=device)
    count = 0
    next_index = compute_next_index_gt(vertebra_ids, mask)
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 2:
            continue
        idx = torch.arange(N, device=device)
        nxt = next_index[b]
        pair_valid = valid & (nxt != idx) & torch.gather(valid, 0, nxt)
        if pair_valid.any():
            s_cur = s[b, pair_valid].squeeze(-1)
            s_next = s[b, nxt[pair_valid]].squeeze(-1)
            diff = s_next - s_cur
            total = total + torch.relu(margin - diff).mean()
            count += 1
    if count > 0:
        total = total / count
    return total


def s_smoothness_loss(
    s: torch.Tensor,
    vertebra_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Smoothness regularization on s differences for GT-adjacent vertebrae.
    """
    B, N, _ = s.shape
    device = s.device
    total = torch.tensor(0.0, device=device)
    count = 0
    next_index = compute_next_index_gt(vertebra_ids, mask)
    for b in range(B):
        valid = mask[b]
        if valid.sum() < 3:
            continue
        idx = torch.arange(N, device=device)
        nxt = next_index[b]
        pair_valid = valid & (nxt != idx) & torch.gather(valid, 0, nxt)
        if pair_valid.any():
            s_cur = s[b, pair_valid].squeeze(-1)
            s_next = s[b, nxt[pair_valid]].squeeze(-1)
            diffs = s_next - s_cur
            total = total + torch.var(diffs)
            count += 1
    if count > 0:
        total = total / count
    return total
from utils.assembly_data_loader import AssemblyDataset

# Import scheduler without loading pretrain_encoder (avoid indentation issues)
try:
    from workspace.utils.model_utils import CosineAnnealingWarmupRestarts
except Exception:
    # Fallback: simple cosine scheduler if needed
    CosineAnnealingWarmupRestarts = None


def setup_logging(output_dir: Path = None):
    """Setup logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log') if output_dir else logging.StreamHandler(),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)




def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger,
    loss_weights: Dict[str, float],
    bspline_weight: float = 1.0,
    bspline_smooth_weight: float = 0.1,
    s_monotonic_weight: float = 0.0,
    s_smooth_weight: float = 0.0,
    delta_pose_t_weight: float = 0.0,
    delta_pose_rot_weight: float = 0.0,
    root_anchor_t_weight: float = 0.0,
    root_anchor_rot_weight: float = 0.0,
    normalize_translation: bool = False,
    delta_curriculum_epochs: int = 0,
    abs_pose_after: float = 0.1,
    delta_pose_warmup: float = 0.1,
    spline_lateral_weight: float = 0.0,
    spline_tangent_smooth_weight: float = 0.0,
    epoch: int = 0,
    is_main_process: bool = True,
    rank: int = 0,
) -> Dict[str, float]:
    """Train for one epoch"""
    use_ddp = isinstance(model, DDP)
    model_module = model.module if use_ddp else model
    model.train()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    num_samples = 0
    gradient_norms = []
    start_time = time.time()
    
    # Position: 0 for overall, rank+1 for each rank's progress bar
    pbar = tqdm(dataloader, desc=f'Training [Rank {rank}]', position=rank+1, leave=True, dynamic_ncols=True)
    for batch in pbar:
        embeddings = batch['embeddings'].to(device)  # (B, N, embed_dim)
        pad_mask = ~batch['mask'].to(device)  # (B, N) - True where padding (invert mask)
        vertebra_ids = batch['vertebra_ids'].to(device)  # (B, N)
        
        # Create mask_mask for completion task (randomly mask some vertebrae)
        # For now, don't mask during training (mask_mask = all False)
        mask_mask = torch.zeros_like(pad_mask, dtype=torch.bool, device=device)
        
        # Prepare targets
        targets = batch['targets']
        gt_types = targets['ordering'].to(device)  # (B, N)
        gt_t = targets['assembly']['translation'].to(device)  # (B, N, 3)
        # Convert quaternion to rotation matrix if needed
        if 'rotation' in targets['assembly']:
            gt_rot = targets['assembly']['rotation'].to(device)  # (B, N, 4) quaternion [x, y, z, w]
            # Convert quaternion to rotation matrix (torch only)
            # q = [x, y, z, w]
            x, y, z, w = gt_rot[..., 0], gt_rot[..., 1], gt_rot[..., 2], gt_rot[..., 3]
            # Normalize
            norm = torch.sqrt(x**2 + y**2 + z**2 + w**2 + 1e-8)
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
            
            # Build rotation matrix
            gt_R = torch.stack([
                torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)], dim=-1),
                torch.stack([2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)], dim=-1),
                torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)], dim=-1),
            ], dim=-2)  # (B, N, 3, 3)
        else:
            gt_R = None
        
        # For completion: use original embeddings
        gt_embedding = embeddings.clone()  # (B, N, embed_dim)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(embeddings, pad_mask=pad_mask, mask_mask=mask_mask)
        
        # Curriculum: shift from absolute pose to delta-pose
        if delta_curriculum_epochs > 0 and epoch >= delta_curriculum_epochs:
            abs_scale = abs_pose_after
            delta_scale = 1.0
            use_pred_order = True
        else:
            abs_scale = 1.0
            delta_scale = delta_pose_warmup
            use_pred_order = False

        # Compute loss using compute_losses (absolute pose weakly supervised)
        losses = compute_losses(
            predictions,
            gt_types=gt_types,
            gt_t=gt_t,
            gt_R=gt_R,
            gt_embedding=gt_embedding if mask_mask.any() else None,
            w_order=loss_weights.get('ordering', 1.0),
            w_trans=0.0 if normalize_translation else loss_weights.get('translation', 1.0) * abs_scale,
            w_rot=loss_weights.get('rotation', 1.0) * abs_scale,
            w_comp=loss_weights.get('completion', 1.0),
        )
        loss = losses['loss_total']

        # B-spline field loss (Spinal Field only)
        if isinstance(model_module, SpineAssemblySpinalField):
            control_points = predictions["spine_field"]["control_points"]
            centroids = batch['points'].mean(dim=2).to(device)  # (B,N,3)
            bs_loss = bspline_loss(
                control_points,
                centroids,
                batch['vertebra_ids'].to(device),
                batch['mask'].to(device),
                weight_smooth=bspline_smooth_weight,
            )
            losses['loss_bspline'] = bs_loss
            loss = loss + bspline_weight * bs_loss

            # s ranking + smoothness losses (ordering potential)
            if s_monotonic_weight > 0.0:
                s_vals = predictions["spine_field"]["s"]
                s_loss = s_pairwise_ranking_loss(
                    s_vals,
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                )
                losses['loss_s_monotonic'] = s_loss
                loss = loss + s_monotonic_weight * s_loss
            if s_smooth_weight > 0.0:
                s_vals = predictions["spine_field"]["s"]
                s_smooth = s_smoothness_loss(
                    s_vals,
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                )
                losses['loss_s_smooth'] = s_smooth
                loss = loss + s_smooth_weight * s_smooth

            # Spline lateral offset loss
            if spline_lateral_weight > 0.0:
                offset_local = predictions["pose_local"]["offset_local"]
                valid_mask = batch['mask'].to(device)
                lateral = offset_local[..., :2]
                lateral_norm = (lateral.pow(2).sum(dim=-1))
                lat_loss = lateral_norm[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=device)
                losses['loss_spline_lateral'] = lat_loss
                loss = loss + spline_lateral_weight * lat_loss

            # Spline tangent smoothness (curvature proxy)
            if spline_tangent_smooth_weight > 0.0:
                tangent = predictions["spine_field"]["tangent"]  # (B,N,3)
                valid_mask = batch['mask'].to(device)
                # sort by predicted s
                s_sorted = predictions["spine_field"]["s_sorted_idx"]
                t_sorted = torch.gather(
                    tangent,
                    1,
                    s_sorted.unsqueeze(-1).expand(-1, -1, 3),
                )
                valid_sorted = torch.gather(valid_mask, 1, s_sorted)
                diff = t_sorted[:, 1:] - t_sorted[:, :-1]
                diff_norm = diff.pow(2).sum(dim=-1)
                mask_pairs = valid_sorted[:, 1:] & valid_sorted[:, :-1]
                tan_loss = diff_norm[mask_pairs].mean() if mask_pairs.any() else torch.tensor(0.0, device=device)
                losses['loss_spline_tangent'] = tan_loss
                loss = loss + spline_tangent_smooth_weight * tan_loss

            # Normalized translation loss (spine length)
            if normalize_translation and gt_t is not None:
                lengths = compute_spine_length(gt_t, batch['vertebra_ids'].to(device), batch['mask'].to(device))
                t_pred = predictions['pose']['t']
                diff = t_pred - gt_t
                diff_norm = diff / lengths.view(-1, 1, 1)
                valid_mask = batch['mask'].to(device)
                t_loss = (diff_norm.pow(2).sum(dim=-1))[valid_mask].mean()
                losses['loss_translation_norm'] = t_loss
                loss = loss + abs_scale * loss_weights.get('translation', 1.0) * t_loss

            # Delta pose supervision
            if (delta_pose_t_weight > 0.0 or delta_pose_rot_weight > 0.0) and gt_t is not None:
                if use_pred_order:
                    next_index = predictions["delta_pose"]["next_index"]
                else:
                    next_index = compute_next_index_gt(
                        batch['vertebra_ids'].to(device),
                        batch['mask'].to(device),
                    )
                valid_mask = batch['mask'].to(device)
                next_valid = torch.gather(valid_mask, 1, next_index)
                idx_self = torch.arange(valid_mask.shape[1], device=device).unsqueeze(0).expand_as(next_index)
                pair_valid = valid_mask & next_valid & (next_index != idx_self)

                if pair_valid.any():
                    d_t = predictions["delta_pose"]["d_t"]
                    gt_t_next = torch.gather(gt_t, 1, next_index.unsqueeze(-1).expand(-1, -1, 3))
                    gt_d_t = gt_t_next - gt_t
                    dt_err = torch.norm(d_t - gt_d_t, dim=-1)
                    dt_loss = torch.mean(dt_err[pair_valid])
                    losses['loss_delta_t'] = dt_loss
                    loss = loss + delta_scale * delta_pose_t_weight * dt_loss

                    if delta_pose_rot_weight > 0.0 and gt_R is not None:
                        d_R = predictions["delta_pose"]["d_R"]
                        gt_R_next = torch.gather(
                            gt_R,
                            1,
                            next_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
                        )
                        gt_R_rel = torch.matmul(gt_R.transpose(-1, -2), gt_R_next)
                        rot_err = geodesic_distance(d_R[pair_valid], gt_R_rel[pair_valid]) * 180.0 / math.pi
                        dR_loss = torch.mean(rot_err)
                        losses['loss_delta_rot'] = dR_loss
                        loss = loss + delta_scale * delta_pose_rot_weight * dR_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        gradient_norms.append(grad_norm.item())
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        # Map loss names to match expected format
        loss_mapping = {
            'loss_ordering': 'ordering',
            'loss_translation': 'assembly_translation',
            'loss_translation_norm': 'assembly_translation_norm',
            'loss_rotation': 'assembly_rotation',
            'loss_completion': 'missing_completion',
            'loss_bspline': 'bspline',
            'loss_s_monotonic': 's_monotonic',
            'loss_s_smooth': 's_smooth',
            'loss_spline_lateral': 'spline_lateral',
            'loss_spline_tangent': 'spline_tangent',
            'loss_root_t': 'root_anchor_t',
            'loss_root_rot': 'root_anchor_rot',
            'loss_delta_t': 'delta_pose_t',
            'loss_delta_rot': 'delta_pose_rot',
            'loss_total': 'total',
        }
        for key, value in losses.items():
            mapped_key = loss_mapping.get(key, key)
            if mapped_key not in loss_components:
                loss_components[mapped_key] = 0.0
            if isinstance(value, torch.Tensor):
                loss_components[mapped_key] += value.item()
            else:
                loss_components[mapped_key] += value
        
        num_batches += 1
        num_samples += embeddings.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'grad_norm': f'{grad_norm.item():.4f}',
        })
    
    # Average losses (aggregate across all processes for DDP)
    if use_ddp:
        import torch.distributed as dist
        # Gather metrics from all processes
        metrics_list = [total_loss, num_batches, num_samples] + list(loss_components.values())
        metrics_tensor = torch.tensor(metrics_list, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss_all = metrics_tensor[0].item()
        num_batches_all = int(metrics_tensor[1].item())
        num_samples_all = int(metrics_tensor[2].item())
        loss_components_all = {name: metrics_tensor[i+3].item() for i, name in enumerate(loss_components.keys())}
        
        avg_loss = total_loss_all / num_batches_all if num_batches_all > 0 else 0.0
        avg_loss_components = {name: val / num_batches_all if num_batches_all > 0 else 0.0 
                              for name, val in loss_components_all.items()}
        total_samples = num_samples_all
    else:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
        total_samples = num_samples
    
    # Compute additional metrics
    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0.0
    
    avg_grad_norm = float(np.mean(gradient_norms)) if gradient_norms else 0.0
    max_grad_norm = float(np.max(gradient_norms)) if gradient_norms else 0.0
    
    return {
        'total_loss': avg_loss,
        **avg_loss_components,
        'grad_norm_mean': avg_grad_norm,
        'grad_norm_max': max_grad_norm,
        'samples_per_sec': samples_per_sec,
        'elapsed_time': elapsed_time,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    logger,
    loss_weights: Dict[str, float],
    bspline_weight: float = 1.0,
    bspline_smooth_weight: float = 0.1,
    s_monotonic_weight: float = 0.0,
    s_smooth_weight: float = 0.0,
    delta_pose_t_weight: float = 0.0,
    delta_pose_rot_weight: float = 0.0,
    root_anchor_t_weight: float = 0.0,
    root_anchor_rot_weight: float = 0.0,
    normalize_translation: bool = False,
    delta_curriculum_epochs: int = 0,
    abs_pose_after: float = 0.1,
    delta_pose_warmup: float = 0.1,
    spline_lateral_weight: float = 0.0,
    spline_tangent_smooth_weight: float = 0.0,
    epoch: int = 0,
    is_main_process: bool = True,
    rank: int = 0,
) -> Dict[str, float]:
    """Validate for one epoch"""
    use_ddp = isinstance(model, DDP)
    model_module = model.module if use_ddp else model
    model.eval()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    # Metrics accumulators
    ordering_correct = 0
    ordering_total = 0
    translation_errors = []
    rotation_errors = []
    completion_errors = []
    
    # Spinal field metrics (if using spinal_field model)
    spine_coordinate_errors = []  # For s_i prediction accuracy
    delta_pose_translation_errors = []  # For delta translation
    delta_pose_rotation_errors = []  # For delta rotation
    
    # Position: 0 for overall, rank+1 for each rank's progress bar
    pbar = tqdm(dataloader, desc=f'Validation [Rank {rank}]', disable=not is_main_process, position=rank+1, leave=False)
    with torch.no_grad():
        for batch in pbar:
            embeddings = batch['embeddings'].to(device)
            pad_mask = ~batch['mask'].to(device)
            vertebra_ids = batch['vertebra_ids'].to(device)
            mask_mask = torch.zeros_like(pad_mask, dtype=torch.bool, device=device)
            
            # Prepare targets
            targets = batch['targets']
            gt_types = targets['ordering'].to(device)
            gt_t = targets['assembly']['translation'].to(device)
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
            predictions = model(embeddings, pad_mask=pad_mask, mask_mask=mask_mask)
            
            # Curriculum: shift from absolute pose to delta-pose
            if delta_curriculum_epochs > 0 and epoch >= delta_curriculum_epochs:
                abs_scale = abs_pose_after
                delta_scale = 1.0
                use_pred_order = True
            else:
                abs_scale = 1.0
                delta_scale = delta_pose_warmup
                use_pred_order = False

            # Compute loss
            losses = compute_losses(
                predictions,
                gt_types=gt_types,
                gt_t=gt_t,
                gt_R=gt_R,
                gt_embedding=gt_embedding if mask_mask.any() else None,
                w_order=loss_weights.get('ordering', 1.0),
                w_trans=0.0 if normalize_translation else loss_weights.get('translation', 1.0) * abs_scale,
                w_rot=loss_weights.get('rotation', 1.0) * abs_scale,
                w_comp=loss_weights.get('completion', 1.0),
            )
            loss = losses['loss_total']

            if isinstance(model_module, SpineAssemblySpinalField):
                control_points = predictions["spine_field"]["control_points"]
                centroids = batch['points'].mean(dim=2).to(device)  # (B,N,3)
                bs_loss = bspline_loss(
                    control_points,
                    centroids,
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                    weight_smooth=bspline_smooth_weight,
                )
                losses['loss_bspline'] = bs_loss
                loss = loss + bspline_weight * bs_loss

                # s ranking + smoothness losses (ordering potential)
                if s_monotonic_weight > 0.0:
                    s_vals = predictions["spine_field"]["s"]
                    s_loss = s_pairwise_ranking_loss(
                        s_vals,
                        batch['vertebra_ids'].to(device),
                        batch['mask'].to(device),
                    )
                    losses['loss_s_monotonic'] = s_loss
                    loss = loss + s_monotonic_weight * s_loss
                if s_smooth_weight > 0.0:
                    s_vals = predictions["spine_field"]["s"]
                    s_smooth = s_smoothness_loss(
                        s_vals,
                        batch['vertebra_ids'].to(device),
                        batch['mask'].to(device),
                    )
                    losses['loss_s_smooth'] = s_smooth
                    loss = loss + s_smooth_weight * s_smooth

                # Spline lateral offset loss
                if spline_lateral_weight > 0.0:
                    offset_local = predictions["pose_local"]["offset_local"]
                    valid_mask = batch['mask'].to(device)
                    lateral = offset_local[..., :2]
                    lateral_norm = (lateral.pow(2).sum(dim=-1))
                    lat_loss = lateral_norm[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=device)
                    losses['loss_spline_lateral'] = lat_loss
                    loss = loss + spline_lateral_weight * lat_loss

                # Spline tangent smoothness (curvature proxy)
                if spline_tangent_smooth_weight > 0.0:
                    tangent = predictions["spine_field"]["tangent"]  # (B,N,3)
                    valid_mask = batch['mask'].to(device)
                    s_sorted = predictions["spine_field"]["s_sorted_idx"]
                    t_sorted = torch.gather(
                        tangent,
                        1,
                        s_sorted.unsqueeze(-1).expand(-1, -1, 3),
                    )
                    valid_sorted = torch.gather(valid_mask, 1, s_sorted)
                    diff = t_sorted[:, 1:] - t_sorted[:, :-1]
                    diff_norm = diff.pow(2).sum(dim=-1)
                    mask_pairs = valid_sorted[:, 1:] & valid_sorted[:, :-1]
                    tan_loss = diff_norm[mask_pairs].mean() if mask_pairs.any() else torch.tensor(0.0, device=device)
                    losses['loss_spline_tangent'] = tan_loss
                    loss = loss + spline_tangent_smooth_weight * tan_loss

                # Normalized translation loss (spine length)
                if normalize_translation and gt_t is not None:
                    lengths = compute_spine_length(gt_t, batch['vertebra_ids'].to(device), batch['mask'].to(device))
                    t_pred = predictions['pose']['t']
                    diff = t_pred - gt_t
                    diff_norm = diff / lengths.view(-1, 1, 1)
                    valid_mask = batch['mask'].to(device)
                    t_loss = (diff_norm.pow(2).sum(dim=-1))[valid_mask].mean()
                    losses['loss_translation_norm'] = t_loss
                    loss = loss + abs_scale * loss_weights.get('translation', 1.0) * t_loss

                # Delta pose supervision
                if (delta_pose_t_weight > 0.0 or delta_pose_rot_weight > 0.0) and gt_t is not None:
                    if use_pred_order:
                        next_index = predictions["delta_pose"]["next_index"]
                    else:
                        next_index = compute_next_index_gt(
                            batch['vertebra_ids'].to(device),
                            batch['mask'].to(device),
                        )
                    valid_mask = batch['mask'].to(device)
                    next_valid = torch.gather(valid_mask, 1, next_index)
                    idx_self = torch.arange(valid_mask.shape[1], device=device).unsqueeze(0).expand_as(next_index)
                    pair_valid = valid_mask & next_valid & (next_index != idx_self)

                    if pair_valid.any():
                        d_t = predictions["delta_pose"]["d_t"]
                        gt_t_next = torch.gather(gt_t, 1, next_index.unsqueeze(-1).expand(-1, -1, 3))
                        gt_d_t = gt_t_next - gt_t
                        dt_err = torch.norm(d_t - gt_d_t, dim=-1)
                        if normalize_translation:
                            lengths = compute_spine_length(gt_t, batch['vertebra_ids'].to(device), batch['mask'].to(device))
                            dt_err = dt_err / lengths.view(-1, 1)
                        dt_loss = torch.mean(dt_err[pair_valid])
                        losses['loss_delta_t'] = dt_loss
                        loss = loss + delta_scale * delta_pose_t_weight * dt_loss

                        if delta_pose_rot_weight > 0.0 and gt_R is not None:
                            d_R = predictions["delta_pose"]["d_R"]
                            gt_R_next = torch.gather(
                                gt_R,
                                1,
                                next_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
                            )
                            gt_R_rel = torch.matmul(gt_R.transpose(-1, -2), gt_R_next)
                            rot_err = geodesic_distance(d_R[pair_valid], gt_R_rel[pair_valid]) * 180.0 / math.pi
                            dR_loss = torch.mean(rot_err)
                            losses['loss_delta_rot'] = dR_loss
                            loss = loss + delta_scale * delta_pose_rot_weight * dR_loss

                # Dual anchoring to fix gauge (world frame)
                if (root_anchor_t_weight > 0.0 or root_anchor_rot_weight > 0.0) and gt_t is not None:
                    root_idx = get_root_index_gt(
                        batch['vertebra_ids'].to(device),
                        batch['mask'].to(device),
                        prefer_ids=[23, 24],  # L5/S1
                    )
                    top_idx = get_top_index_gt(
                        batch['vertebra_ids'].to(device),
                        batch['mask'].to(device),
                        prefer_ids=[7],  # T1
                    )
                    t_pred = predictions['pose']['t']
                    t_root = torch.gather(t_pred, 1, root_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                    t_top = torch.gather(t_pred, 1, top_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                    gt_t_root = torch.gather(gt_t, 1, root_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                    gt_t_top = torch.gather(gt_t, 1, top_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                    t_loss = torch.mean(torch.sum((t_root - gt_t_root) ** 2, dim=-1)) + \
                             torch.mean(torch.sum((t_top - gt_t_top) ** 2, dim=-1))
                    losses['loss_root_t'] = t_loss
                    loss = loss + root_anchor_t_weight * t_loss
                    if root_anchor_rot_weight > 0.0 and gt_R is not None:
                        R_pred = predictions['pose']['R']
                        R_root = torch.gather(R_pred, 1, root_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                        R_top = torch.gather(R_pred, 1, top_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                        R_gt_root = torch.gather(gt_R, 1, root_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                        R_gt_top = torch.gather(gt_R, 1, top_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                        r_loss = geodesic_distance(R_root, R_gt_root).mean() + geodesic_distance(R_top, R_gt_top).mean()
                        losses['loss_root_rot'] = r_loss
                        loss = loss + root_anchor_rot_weight * r_loss

                # Root anchoring to fix gauge (world frame)
                if (root_anchor_t_weight > 0.0 or root_anchor_rot_weight > 0.0) and gt_t is not None:
                    root_idx = get_root_index_gt(batch['vertebra_ids'].to(device), batch['mask'].to(device))
                    t_pred = predictions['pose']['t']
                    t_root = torch.gather(t_pred, 1, root_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                    gt_t_root = torch.gather(gt_t, 1, root_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                    t_loss = torch.mean(torch.sum((t_root - gt_t_root) ** 2, dim=-1))
                    losses['loss_root_t'] = t_loss
                    loss = loss + root_anchor_t_weight * t_loss
                    if root_anchor_rot_weight > 0.0 and gt_R is not None:
                        R_pred = predictions['pose']['R']
                        R_root = torch.gather(R_pred, 1, root_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                        R_gt_root = torch.gather(gt_R, 1, root_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                        r_loss = geodesic_distance(R_root, R_gt_root).mean()
                        losses['loss_root_rot'] = r_loss
                        loss = loss + root_anchor_rot_weight * r_loss

            # Dual anchoring to fix gauge (world frame)
            if (root_anchor_t_weight > 0.0 or root_anchor_rot_weight > 0.0) and gt_t is not None:
                root_idx = get_root_index_gt(
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                    prefer_ids=[23, 24],  # L5/S1
                )
                top_idx = get_top_index_gt(
                    batch['vertebra_ids'].to(device),
                    batch['mask'].to(device),
                    prefer_ids=[7],  # T1
                )
                t_pred = predictions['pose']['t']
                t_root = torch.gather(t_pred, 1, root_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                t_top = torch.gather(t_pred, 1, top_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                gt_t_root = torch.gather(gt_t, 1, root_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                gt_t_top = torch.gather(gt_t, 1, top_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
                t_loss = torch.mean(torch.sum((t_root - gt_t_root) ** 2, dim=-1)) + \
                         torch.mean(torch.sum((t_top - gt_t_top) ** 2, dim=-1))
                losses['loss_root_t'] = t_loss
                loss = loss + root_anchor_t_weight * t_loss
                if root_anchor_rot_weight > 0.0 and gt_R is not None:
                    R_pred = predictions['pose']['R']
                    R_root = torch.gather(R_pred, 1, root_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                    R_top = torch.gather(R_pred, 1, top_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                    R_gt_root = torch.gather(gt_R, 1, root_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                    R_gt_top = torch.gather(gt_R, 1, top_idx.view(-1, 1, 1, 1).expand(-1, 1, 3, 3)).squeeze(1)
                    r_loss = geodesic_distance(R_root, R_gt_root).mean() + geodesic_distance(R_top, R_gt_top).mean()
                    losses['loss_root_rot'] = r_loss
                    loss = loss + root_anchor_rot_weight * r_loss
            
            # Accumulate losses
            total_loss += loss.item()
            loss_mapping = {
                'loss_ordering': 'ordering',
                'loss_translation': 'assembly_translation',
                'loss_translation_norm': 'assembly_translation_norm',
                'loss_rotation': 'assembly_rotation',
                'loss_completion': 'missing_completion',
                'loss_bspline': 'bspline',
                'loss_s_monotonic': 's_monotonic',
                'loss_s_smooth': 's_smooth',
                'loss_spline_lateral': 'spline_lateral',
                'loss_spline_tangent': 'spline_tangent',
                'loss_root_t': 'root_anchor_t',
                'loss_root_rot': 'root_anchor_rot',
                'loss_delta_t': 'delta_pose_t',
                'loss_delta_rot': 'delta_pose_rot',
                'loss_total': 'total',
            }
            for key, value in losses.items():
                mapped_key = loss_mapping.get(key, key)
                if mapped_key not in loss_components:
                    loss_components[mapped_key] = 0.0
                if isinstance(value, torch.Tensor):
                    loss_components[mapped_key] += value.item()
                else:
                    loss_components[mapped_key] += value
            
            # Compute additional metrics
            valid = ~pad_mask  # (B, N)
            
            # 1. Ordering Accuracy
            if gt_types is not None:
                ordering_logits = predictions['ordering']  # (B, N, num_types+1)
                pred_types = ordering_logits.argmax(dim=-1)  # (B, N)
                order_valid = valid & (gt_types >= 0) & (gt_types < ordering_logits.shape[-1])
                if order_valid.any():
                    correct = (pred_types[order_valid] == gt_types[order_valid]).sum().item()
                    total_valid = order_valid.sum().item()
                    ordering_correct += correct
                    ordering_total += total_valid
            
            # 2. Translation Error (L2 distance in mm)
            if gt_t is not None:
                t_pred = predictions['pose']['t']  # (B, N, 3)
                trans_valid = valid
                if trans_valid.any():
                    diff = t_pred[trans_valid] - gt_t[trans_valid]  # (N_valid, 3)
                    errors = torch.norm(diff, dim=-1)  # (N_valid,)
                    translation_errors.append(errors.cpu())
            
            # 3. Rotation Error (geodesic distance in degrees)
            if gt_R is not None:
                R_pred = predictions['pose']['R']  # (B, N, 3, 3)
                rot_valid = valid
                if rot_valid.any():
                    Rp = R_pred[rot_valid]  # (N_valid, 3, 3)
                    Rg = gt_R[rot_valid]  # (N_valid, 3, 3)
                    # Compute geodesic distance
                    angles = geodesic_distance(Rp, Rg)  # (N_valid,) in radians
                    angles_deg = angles * 180.0 / math.pi  # Convert to degrees
                    rotation_errors.append(angles_deg.cpu())
            
            # 4. Completion Error (cosine similarity for embeddings)
            if mask_mask.any() and gt_embedding is not None:
                pred_emb = predictions['missing_completion']  # (B, N, D)
                comp_valid = mask_mask & (~pad_mask)
                if comp_valid.any():
                    pred_emb_valid = pred_emb[comp_valid]  # (N_valid, D)
                    gt_emb_valid = gt_embedding[comp_valid]  # (N_valid, D)
                    # Normalize
                    pred_emb_norm = torch.nn.functional.normalize(pred_emb_valid, p=2, dim=-1)
                    gt_emb_norm = torch.nn.functional.normalize(gt_emb_valid, p=2, dim=-1)
                    # Cosine similarity
                    cos_sim = (pred_emb_norm * gt_emb_norm).sum(dim=-1)  # (N_valid,)
                    # Convert to error (1 - similarity)
                    errors = 1.0 - cos_sim
                    completion_errors.append(errors.cpu())
            
            # 5. Spinal Field Metrics (if available)
            if 'spine_field' in predictions:
                # Spine coordinate s_i: can evaluate ordering consistency
                # For now, we just log that it's available
                s = predictions['spine_field']['s']  # (B, N, 1)
                s_sorted_idx = predictions['spine_field']['s_sorted_idx']  # (B, N)
                # TODO: Add supervision for s_i if ground truth ordering is available
                # For now, we can check if s is monotonic (increasing along GT order)
            
            # 6. Delta Pose Metrics (if available)
            if 'delta_pose' in predictions and predictions['delta_pose']['d_t'] is not None:
                # Delta pose predictions are available
                # TODO: Add supervision if relative poses are available in ground truth
                # For now, we just track that delta pose is being predicted
                d_t = predictions['delta_pose']['d_t']  # (B, N, 3)
                d_R = predictions['delta_pose']['d_R']  # (B, N, 3, 3)
                next_index = predictions['delta_pose']['next_index']  # (B, N)
                # Note: Delta pose supervision would require ground truth relative poses
            
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Aggregate metrics across processes for DDP
    if use_ddp:
        import torch.distributed as dist
        # Loss metrics
        metrics_list = [total_loss, num_batches, ordering_correct, ordering_total] + list(loss_components.values())
        metrics_tensor = torch.tensor(metrics_list, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss_all = metrics_tensor[0].item()
        num_batches_all = int(metrics_tensor[1].item())
        ordering_correct_all = int(metrics_tensor[2].item())
        ordering_total_all = int(metrics_tensor[3].item())
        loss_components_all = {name: metrics_tensor[i+4].item() for i, name in enumerate(loss_components.keys())}
        
        avg_loss = total_loss_all / num_batches_all if num_batches_all > 0 else 0.0
        avg_loss_components = {name: val / num_batches_all if num_batches_all > 0 else 0.0 
                              for name, val in loss_components_all.items()}
        
        # Gather error tensors from all processes
        # Note: For simplicity, we'll compute metrics on each process and aggregate scalars
        # This avoids complex tensor gathering which requires equal sizes
        if translation_errors:
            translation_errors = torch.cat(translation_errors, dim=0).cpu().numpy()
        else:
            translation_errors = np.array([])
        
        if rotation_errors:
            rotation_errors = torch.cat(rotation_errors, dim=0).cpu().numpy()
        else:
            rotation_errors = np.array([])
        
        if completion_errors:
            completion_errors = torch.cat(completion_errors, dim=0).cpu().numpy()
        else:
            completion_errors = np.array([])
        
        # Spinal field metrics (aggregate if available)
        if spine_coordinate_errors:
            spine_coordinate_errors = torch.cat(spine_coordinate_errors, dim=0).cpu().numpy()
        else:
            spine_coordinate_errors = np.array([])
        
        if delta_pose_translation_errors:
            delta_pose_translation_errors = torch.cat(delta_pose_translation_errors, dim=0).cpu().numpy()
        else:
            delta_pose_translation_errors = np.array([])
        
        if delta_pose_rotation_errors:
            delta_pose_rotation_errors = torch.cat(delta_pose_rotation_errors, dim=0).cpu().numpy()
        else:
            delta_pose_rotation_errors = np.array([])
    else:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_components = {k: v / num_batches for k, v in loss_components.items()}
        ordering_correct_all = ordering_correct
        ordering_total_all = ordering_total
        translation_errors = np.concatenate(translation_errors).numpy() if translation_errors else np.array([])
        rotation_errors = np.concatenate(rotation_errors).numpy() if rotation_errors else np.array([])
        completion_errors = np.concatenate(completion_errors).numpy() if completion_errors else np.array([])
        spine_coordinate_errors = np.concatenate(spine_coordinate_errors).numpy() if spine_coordinate_errors else np.array([])
        delta_pose_translation_errors = np.concatenate(delta_pose_translation_errors).numpy() if delta_pose_translation_errors else np.array([])
        delta_pose_rotation_errors = np.concatenate(delta_pose_rotation_errors).numpy() if delta_pose_rotation_errors else np.array([])
    
    # Compute final metrics
    metrics = {
        'total_loss': avg_loss,
        **avg_loss_components,
    }
    
    # Ordering accuracy
    if ordering_total_all > 0:
        metrics['ordering_accuracy'] = ordering_correct_all / ordering_total_all
    else:
        metrics['ordering_accuracy'] = 0.0
    
    # Translation error (mean, median in mm)
    if len(translation_errors) > 0:
        metrics['translation_error_mean'] = float(np.mean(translation_errors))
        metrics['translation_error_median'] = float(np.median(translation_errors))
    else:
        metrics['translation_error_mean'] = 0.0
        metrics['translation_error_median'] = 0.0
    
    # Rotation error (mean, median in degrees)
    if len(rotation_errors) > 0:
        metrics['rotation_error_mean'] = float(np.mean(rotation_errors))
        metrics['rotation_error_median'] = float(np.median(rotation_errors))
    else:
        metrics['rotation_error_mean'] = 0.0
        metrics['rotation_error_median'] = 0.0
    
    # Completion error (mean cosine distance)
    if len(completion_errors) > 0:
        metrics['completion_error_mean'] = float(np.mean(completion_errors))
    else:
        metrics['completion_error_mean'] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Assembly Training')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory containing pre-extracted embeddings (from extract_assembly_embeddings.py)')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing original (non-centered) point clouds')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--max_vertebrae', type=int, default=30,
                        help='Maximum number of vertebrae per subject')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to checkpoint to load weights only (fresh optimizer/scheduler)')
    parser.add_argument('--freeze_ordering_head', action='store_true',
                        help='Freeze ordering head parameters')
    parser.add_argument('--train_pose_only', action='store_true',
                        help='Train only pose heads (freeze backbone, ordering, completion)')
    # Loss weights
    parser.add_argument('--ordering_weight', type=float, default=1.0,
                        help='Weight for ordering loss')
    parser.add_argument('--assembly_weight', type=float, default=1.0,
                        help='Weight for assembly loss')
    parser.add_argument('--missing_weight', type=float, default=1.0,
                        help='Weight for missing completion loss')
    parser.add_argument('--bspline_weight', type=float, default=1.0,
                        help='Weight for B-spline curve fitting loss')
    parser.add_argument('--bspline_smooth_weight', type=float, default=0.1,
                        help='Smoothness weight for B-spline control points')
    parser.add_argument('--bspline_k', type=int, default=8,
                        help='Number of B-spline control points')
    parser.add_argument('--s_monotonic_weight', type=float, default=0.0,
                        help='Weight for s monotonicity loss (spinal_field only)')
    parser.add_argument('--s_smooth_weight', type=float, default=0.0,
                        help='Weight for s smoothness loss (spinal_field only)')
    parser.add_argument('--delta_pose_t_weight', type=float, default=0.0,
                        help='Weight for delta pose translation supervision (spinal_field only)')
    parser.add_argument('--delta_pose_rot_weight', type=float, default=0.0,
                        help='Weight for delta pose rotation supervision (spinal_field only)')
    parser.add_argument('--root_anchor_t_weight', type=float, default=0.0,
                        help='Weight for root vertebra translation anchoring')
    parser.add_argument('--root_anchor_rot_weight', type=float, default=0.0,
                        help='Weight for root vertebra rotation anchoring')
    parser.add_argument('--normalize_translation', action='store_true',
                        help='Normalize translation losses by spine length')
    parser.add_argument('--delta_curriculum_epochs', type=int, default=0,
                        help='Epochs before switching to delta-pose-dominant training')
    parser.add_argument('--abs_pose_after', type=float, default=0.1,
                        help='Absolute pose weight scale after curriculum')
    parser.add_argument('--delta_pose_warmup', type=float, default=0.1,
                        help='Delta pose weight scale during warmup')
    parser.add_argument('--spline_lateral_weight', type=float, default=0.0,
                        help='Penalty for lateral offsets from spline centerline')
    parser.add_argument('--spline_tangent_smooth_weight', type=float, default=0.0,
                        help='Penalty for rapid tangent changes along spline')
    # Scheduler parameters
    parser.add_argument('--first_cycle_steps', type=int, default=20,
                        help='First cycle steps for cosine warmup scheduler')
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help='Warmup steps for cosine warmup scheduler')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7,
        help='Minimum learning rate')
    # Model selection
    parser.add_argument('--model_type', type=str, default='baseline',
        choices=['baseline', 'spinal_field'],
        help='Model type: baseline (SpineAssemblyTransformer) or spinal_field (SpineAssemblySpinalField)')
    parser.add_argument('--enable_delta_pose', action='store_true',
        help='Enable delta pose head (only for spinal_field model)')
    
    args = parser.parse_args()
    
    # DDP setup
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12356')
    
    use_ddp = local_rank >= 0
    if use_ddp:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = rank == 0
    else:
        device = torch.device(args.device)
        is_main_process = True
    
    # Create output directory with timestamp (only on main process for DDP)
    base_output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_output_dir / timestamp
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait for main process to create directory (for DDP)
    if use_ddp:
        dist.barrier()
    
    logger = setup_logging(output_dir if is_main_process else None)
    
    # Log training configuration (only on main process)
    if is_main_process:
        save_run_config(args, output_dir, timestamp, logger)
        logger.info("="*60)
        logger.info("Assembly Training Configuration")
        logger.info("="*60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Embedding directory: {args.embedding_dir}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Number of epochs: {args.num_epochs}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Hidden dimension: {args.hidden_dim}")
        logger.info(f"Number of layers: {args.num_layers}")
        logger.info(f"Number of heads: {args.num_heads}")
        logger.info(f"Max vertebrae: {args.max_vertebrae}")
        logger.info(f"Number of workers: {args.num_workers}")
        logger.info(f"Model type: {args.model_type}")
        if args.model_type == 'spinal_field':
            logger.info(f"  Delta pose enabled: {args.enable_delta_pose}")
        if args.resume:
            logger.info(f"Resume from: {args.resume}")
        if args.pretrain:
            logger.info(f"Pretrain weights from: {args.pretrain}")
        if args.freeze_ordering_head:
            logger.info("Freeze ordering head: True")
        if args.train_pose_only:
            logger.info("Train pose only: True")
        logger.info("="*60)
        logger.info(f"Using device: {device}")
        if use_ddp:
            logger.info(f"DDP: rank {rank}/{world_size} (local_rank: {local_rank})")
        logger.info("")
    
    # Get embed_dim from a sample embedding file
    embedding_dir = Path(args.embedding_dir)
    # Find first embedding file to get dimension
    sample_file = None
    for subject_dir in embedding_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        for emb_file in subject_dir.glob('vertebra_*_embedding.npz'):
            sample_file = emb_file
            break
        if sample_file:
            break
    
    if sample_file is None:
        raise FileNotFoundError(f"No embedding files found in {embedding_dir}")
    
    sample_data = np.load(sample_file)
    embed_dim = sample_data['embedding'].shape[0]
    
    # Create assembly transformer
    if is_main_process:
        logger.info(f"Creating assembly transformer (type: {args.model_type})...")
    
    if args.model_type == 'spinal_field':
        model = SpineAssemblySpinalField(
            embed_dim=embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_vertebra_types=26,
            use_mask_token=True,
            extra_dim=0,  # Can add equivariant features later
            enable_delta_pose=args.enable_delta_pose,
            num_control_points=args.bspline_k,
        ).to(device)
    else:  # baseline
        model = SpineAssemblyTransformer(
            embed_dim=embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_vertebra_types=26,
            use_mask_token=True,
            extra_dim=0,  # Can add equivariant features later
        ).to(device)
    
    # Wrap with DDP if using multi-GPU
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_module = model.module
    else:
        model_module = model

    # Optional freezing for ablations
    if args.freeze_ordering_head or args.train_pose_only:
        if args.freeze_ordering_head:
            for p in model_module.ordering_head.parameters():
                p.requires_grad = False

        if args.train_pose_only:
            for name, p in model_module.named_parameters():
                if name.startswith('pose_head') or name.startswith('delta_pose_head'):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        if is_main_process:
            trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model_module.parameters())
            logger.info(f"Trainable params: {trainable:,} / {total:,}")
    
    if is_main_process:
        logger.info(f"Assembly Transformer created:")
        logger.info(f"  Embed dim: {embed_dim}")
        logger.info(f"  Hidden dim: {args.hidden_dim}")
        logger.info(f"  Num layers: {args.num_layers}")
        logger.info(f"  Num heads: {args.num_heads}")
        logger.info("")
    
    # Create data loaders
    if is_main_process:
        logger.info("Creating data loaders...")
    
    # Create datasets
    point_cloud_dir = Path(args.point_cloud_dir)
    
    train_dataset = AssemblyDataset(
        embedding_dir=embedding_dir,
        point_cloud_dir=point_cloud_dir,  # Add original point cloud directory
        split='train',
        max_vertebrae=args.max_vertebrae,
        augment=True,
    )
    
    val_dataset = AssemblyDataset(
        embedding_dir=embedding_dir,
        point_cloud_dir=point_cloud_dir,  # Add original point cloud directory
        split='val',
        max_vertebrae=args.max_vertebrae,
        augment=False,
    )
    
    # Create samplers for DDP
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_len = len(train_dataset)
    val_len = len(val_dataset)
    train_samples_per_rank = len(train_sampler) if train_sampler is not None else train_len
    val_samples_per_rank = len(val_sampler) if val_sampler is not None else val_len
    if is_main_process:
        logger.info(f"Train subjects: {train_len} (per-rank: {train_samples_per_rank})")
        logger.info(f"Val subjects: {val_len} (per-rank: {val_samples_per_rank})")
        if train_samples_per_rank < args.batch_size:
            logger.warning(
                f"Per-rank train samples ({train_samples_per_rank}) < batch_size ({args.batch_size}); "
                "disabling drop_last to avoid empty epochs."
            )
    
    # Create DataLoaders
    drop_last_train = train_samples_per_rank >= args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        collate_fn=collate_assembly_batch,
    )

    if len(train_loader) == 0:
        if is_main_process:
            logger.error(
                "No training batches created. Reduce batch_size or num_gpus, "
                "or ensure enough training samples are available."
            )
        return
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_assembly_batch,
    )
    
    # Loss function is now compute_losses (imported from models)
    # We'll use it directly in train_epoch
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Scheduler
    if args.warmup_steps >= args.first_cycle_steps:
        logger.warning(f"warmup_steps ({args.warmup_steps}) >= first_cycle_steps ({args.first_cycle_steps})")
        args.warmup_steps = args.first_cycle_steps - 1
    
    if CosineAnnealingWarmupRestarts is not None:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=args.first_cycle_steps,
            warmup_steps=args.warmup_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
        )
    else:
        logger.warning("CosineAnnealingWarmupRestarts not available; using CosineAnnealingLR.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(args.first_cycle_steps, 1),
            eta_min=args.min_lr,
        )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and args.pretrain:
        if is_main_process:
            logger.error("Both --resume and --pretrain are set. Use only one.")
        return

    if args.pretrain is not None:
        if is_main_process:
            logger.info(f"Loading pretrain weights from {args.pretrain}")

        checkpoint_path = Path(args.pretrain)
        if not checkpoint_path.exists():
            if is_main_process:
                logger.error(f"Pretrain checkpoint file not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state only
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing, unexpected = model_module.load_state_dict(state_dict, strict=False)
            if is_main_process:
                logger.info("✓ Loaded model state dict (pretrain)")
                if missing:
                    logger.info(f"  Missing keys: {len(missing)} (e.g., {missing[:3]})")
                if unexpected:
                    logger.info(f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")
        else:
            try:
                state_dict = checkpoint
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                missing, unexpected = model_module.load_state_dict(state_dict, strict=False)
                if is_main_process:
                    logger.info("✓ Loaded model state dict (pretrain, direct)")
                    if missing:
                        logger.info(f"  Missing keys: {len(missing)} (e.g., {missing[:3]})")
                    if unexpected:
                        logger.info(f"  Unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")
            except Exception as e:
                if is_main_process:
                    logger.error(f"Failed to load pretrain model state dict: {e}")
                return

        if is_main_process:
            logger.info("Pretrain weights loaded. Starting from epoch 0 with fresh optimizer/scheduler.")
            logger.info("")

    elif args.resume is not None:
        if is_main_process:
            logger.info(f"Loading checkpoint from {args.resume}")
        
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            if is_main_process:
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix if present (for DDP compatibility)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model_module.load_state_dict(state_dict)
            if is_main_process:
                logger.info("✓ Loaded model state dict")
        else:
            # Try loading directly (for compatibility)
            try:
                state_dict = checkpoint
                # Remove 'module.' prefix if present
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model_module.load_state_dict(state_dict)
                if is_main_process:
                    logger.info("✓ Loaded model state dict (direct)")
            except Exception as e:
                if is_main_process:
                    logger.error(f"Failed to load model state dict: {e}")
                return
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main_process:
                logger.info("✓ Loaded optimizer state dict")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_main_process:
                logger.info("✓ Loaded scheduler state dict")
        
        # Load training state
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            if is_main_process:
                logger.info(f"Resuming from epoch {start_epoch}")
        
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            if is_main_process:
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        if is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
            logger.info("")
    
    # Training loop
    train_metrics = {'total_loss': [], 'ordering': [], 'assembly_translation': [], 'assembly_rotation': [], 'missing_completion': [], 'bspline': []}
    val_metrics = {'total_loss': [], 'ordering': [], 'assembly_translation': [], 'assembly_rotation': [], 'missing_completion': [], 'bspline': []}
    prev_train_loss = None
    prev_val_loss = None
    training_start_time = time.time()
    
    # Overall progress bar (position=0, only on main process)
    if is_main_process:
        if args.resume:
            logger.info(f"Resuming training from epoch {start_epoch}/{args.num_epochs}")
        elif args.pretrain:
            logger.info("Starting training from pretrain weights...")
        else:
            logger.info("Starting training...")
        logger.info("")
        epoch_pbar = tqdm(range(start_epoch, args.num_epochs), desc='Overall Progress', position=0, leave=True)
    else:
        epoch_pbar = None
    
    for epoch in range(start_epoch, args.num_epochs):
        # Update sampler for DDP
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            if epoch_pbar is not None:
                epoch_pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Loss weights
        loss_weights = {
            'ordering': args.ordering_weight,
            'translation': args.assembly_weight,
            'rotation': args.assembly_weight,
            'completion': args.missing_weight,
        }
        
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, device, logger, loss_weights,
            bspline_weight=args.bspline_weight,
            bspline_smooth_weight=args.bspline_smooth_weight,
            s_monotonic_weight=args.s_monotonic_weight,
            s_smooth_weight=args.s_smooth_weight,
            delta_pose_t_weight=args.delta_pose_t_weight,
            delta_pose_rot_weight=args.delta_pose_rot_weight,
            root_anchor_t_weight=args.root_anchor_t_weight,
            root_anchor_rot_weight=args.root_anchor_rot_weight,
            normalize_translation=args.normalize_translation,
            delta_curriculum_epochs=args.delta_curriculum_epochs,
            abs_pose_after=args.abs_pose_after,
            delta_pose_warmup=args.delta_pose_warmup,
            spline_lateral_weight=args.spline_lateral_weight,
            spline_tangent_smooth_weight=args.spline_tangent_smooth_weight,
            epoch=epoch,
            is_main_process=is_main_process, rank=rank
        )
        
        # Validate
        val_stats = validate_epoch(
            model, val_loader, device, logger, loss_weights,
            bspline_weight=args.bspline_weight,
            bspline_smooth_weight=args.bspline_smooth_weight,
            s_monotonic_weight=args.s_monotonic_weight,
            s_smooth_weight=args.s_smooth_weight,
            delta_pose_t_weight=args.delta_pose_t_weight,
            delta_pose_rot_weight=args.delta_pose_rot_weight,
            root_anchor_t_weight=args.root_anchor_t_weight,
            root_anchor_rot_weight=args.root_anchor_rot_weight,
            normalize_translation=args.normalize_translation,
            delta_curriculum_epochs=args.delta_curriculum_epochs,
            abs_pose_after=args.abs_pose_after,
            delta_pose_warmup=args.delta_pose_warmup,
            spline_lateral_weight=args.spline_lateral_weight,
            spline_tangent_smooth_weight=args.spline_tangent_smooth_weight,
            epoch=epoch,
            is_main_process=is_main_process, rank=rank
        )
        
        # Update overall progress bar
        if epoch_pbar is not None:
            epoch_pbar.update(1)
        
        # Update scheduler
        scheduler.step(epoch=epoch)
        
        # Log metrics (only on main process)
        if is_main_process:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute loss changes (if previous epoch exists)
            loss_change_info = ""
            if epoch > start_epoch and prev_train_loss is not None and prev_val_loss is not None:
                train_loss_change = train_stats['total_loss'] - prev_train_loss
                val_loss_change = val_stats['total_loss'] - prev_val_loss
                train_change_str = f"{train_loss_change:+.4f}" if not np.isnan(train_loss_change) else "N/A"
                val_change_str = f"{val_loss_change:+.4f}" if not np.isnan(val_loss_change) else "N/A"
                loss_change_info = f" (Δ: {train_change_str} / {val_change_str})"
            
            logger.info("")
            logger.info("="*60)
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} Summary")
            logger.info("="*60)
            logger.info(f"[Training]")
            logger.info(f"  Total Loss: {train_stats['total_loss']:.4f}{loss_change_info}")
            for key in sorted(train_stats.keys()):
                if key not in ['total_loss', 'grad_norm_mean', 'grad_norm_max', 'samples_per_sec', 'elapsed_time']:
                    # Format key name nicely
                    key_name = key.replace('_', ' ').title()
                    logger.info(f"  {key_name} Loss: {train_stats[key]:.4f}")
            
            # Additional training metrics
            if 'grad_norm_mean' in train_stats:
                logger.info(f"  Gradient Norm: mean={train_stats['grad_norm_mean']:.4f}, max={train_stats['grad_norm_max']:.4f}")
            if 'samples_per_sec' in train_stats:
                logger.info(f"  Training Speed: {train_stats['samples_per_sec']:.2f} samples/sec")
            
            logger.info(f"[Validation]")
            logger.info(f"  Total Loss: {val_stats['total_loss']:.4f}{loss_change_info}")
            for key in sorted(val_stats.keys()):
                if key not in ['total_loss', 'ordering_accuracy', 'translation_error_mean', 'translation_error_median',
                              'rotation_error_mean', 'rotation_error_median', 'completion_error_mean']:
                    # Format key name nicely
                    key_name = key.replace('_', ' ').title()
                    logger.info(f"  {key_name} Loss: {val_stats.get(key, 0.0):.4f}")
            
            # Additional validation metrics
            if 'ordering_accuracy' in val_stats:
                logger.info(f"  Ordering Accuracy: {val_stats['ordering_accuracy']:.2%}")
            if 'translation_error_mean' in val_stats:
                logger.info(f"  Translation Error: {val_stats['translation_error_mean']:.2f}mm (mean), {val_stats.get('translation_error_median', 0.0):.2f}mm (median)")
            if 'rotation_error_mean' in val_stats:
                logger.info(f"  Rotation Error: {val_stats['rotation_error_mean']:.2f}° (mean), {val_stats.get('rotation_error_median', 0.0):.2f}° (median)")
            if 'completion_error_mean' in val_stats and val_stats['completion_error_mean'] > 0:
                logger.info(f"  Completion Error: {val_stats['completion_error_mean']:.4f} (cosine distance)")
            
            # Spinal field metrics (if available)
            if 'spine_coordinate_error_mean' in val_stats and val_stats['spine_coordinate_error_mean'] is not None:
                logger.info(f"  Spine Coordinate Error: {val_stats['spine_coordinate_error_mean']:.4f}")
            if 'delta_pose_translation_error_mean' in val_stats and val_stats['delta_pose_translation_error_mean'] is not None:
                logger.info(f"  Delta Pose Translation Error: {val_stats['delta_pose_translation_error_mean']:.4f}mm")
            if 'delta_pose_rotation_error_mean' in val_stats and val_stats['delta_pose_rotation_error_mean'] is not None:
                logger.info(f"  Delta Pose Rotation Error: {val_stats['delta_pose_rotation_error_mean']:.2f}°")
            
            # Update best_val_loss before logging
            is_improved = val_stats['total_loss'] < best_val_loss
            if is_improved:
                improvement = best_val_loss - val_stats['total_loss']
                best_val_loss = val_stats['total_loss']
            else:
                improvement = None
            
            logger.info(f"[Training Info]")
            logger.info(f"  Learning Rate: {current_lr:.2e}")
            if best_val_loss == float('inf'):
                logger.info(f"  Best Val Loss: N/A (first epoch)")
            else:
                logger.info(f"  Best Val Loss: {best_val_loss:.4f}")
            if is_improved:
                logger.info(f"  ✅ Improved by {improvement:.4f}!")
            
            # Estimate remaining time
            if epoch > start_epoch:
                elapsed_epochs = epoch - start_epoch + 1
                avg_time_per_epoch = (time.time() - training_start_time) / elapsed_epochs
                remaining_epochs = args.num_epochs - (epoch + 1)
                remaining_time = remaining_epochs * avg_time_per_epoch
                remaining_hours = int(remaining_time // 3600)
                remaining_minutes = int((remaining_time % 3600) // 60)
                logger.info(f"  ETA: ~{remaining_hours}h {remaining_minutes}m")
            
            logger.info("="*60)
            logger.info("")
            
            # Store previous losses for next epoch
            prev_train_loss = train_stats['total_loss']
            prev_val_loss = val_stats['total_loss']
            
            # Check for NaN
            if np.isnan(train_stats['total_loss']) or np.isnan(val_stats['total_loss']):
                logger.error(f"NaN detected in metrics! Train: {train_stats}, Val: {val_stats}")
                logger.error("Stopping training due to NaN loss")
                break
            
            # Save metrics
            train_metrics['total_loss'].append(train_stats['total_loss'])
            val_metrics['total_loss'].append(val_stats['total_loss'])
            for key in ['ordering', 'assembly_translation', 'assembly_rotation', 'missing_completion']:
                if key in train_stats:
                    train_metrics[key].append(train_stats[key])
                if key in val_stats:
                    val_metrics[key].append(val_stats[key])
            
            # Save checkpoint (best model)
            if is_improved:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': {
                        'embed_dim': embed_dim,
                        'hidden_dim': args.hidden_dim,
                        'num_layers': args.num_layers,
                        'num_heads': args.num_heads,
                        'max_vertebrae': args.max_vertebrae,
                    },
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                if is_main_process:
                    logger.info(f"  💾 Saved best model checkpoint (epoch {epoch+1})")
            
            # Save latest checkpoint (for resuming)
            latest_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'embed_dim': embed_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                    'max_vertebrae': args.max_vertebrae,
                },
            }
            torch.save(latest_checkpoint, output_dir / 'latest_checkpoint.pth')
            
            # Save metrics to JSON for visualization
            metrics_file = output_dir / 'training_metrics.json'
            if epoch == start_epoch:
                # Initialize metrics history
                metrics_history = {
                    'epochs': [],
                    'train': {'total_loss': [], 'ordering': [], 'assembly_translation': [], 'assembly_rotation': [], 'missing_completion': [], 'bspline': []},
                    'val': {'total_loss': [], 'ordering': [], 'assembly_translation': [], 'assembly_rotation': [], 'missing_completion': [], 'bspline': []},
                    'grad_norm_mean': [],
                    'grad_norm_max': [],
                    'learning_rate': [],
                    'samples_per_sec': [],
                }
            else:
                # Load existing metrics
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_history = json.load(f)
                    except:
                        metrics_history = {
                            'epochs': [],
                            'train': {'total_loss': [], 'ordering': [], 'assembly_translation': [], 'assembly_rotation': [], 'missing_completion': [], 'bspline': []},
                            'val': {'total_loss': [], 'ordering': [], 'assembly_translation': [], 'assembly_rotation': [], 'missing_completion': [], 'bspline': []},
                            'grad_norm_mean': [],
                            'grad_norm_max': [],
                            'learning_rate': [],
                            'samples_per_sec': [],
                        }
            
            # Append current epoch metrics
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train']['total_loss'].append(train_stats['total_loss'])
            metrics_history['val']['total_loss'].append(val_stats['total_loss'])
            for key in ['ordering', 'assembly_translation', 'assembly_rotation', 'missing_completion']:
                if key in train_stats:
                    metrics_history['train'][key].append(train_stats[key])
                else:
                    metrics_history['train'][key].append(0.0)
                if key in val_stats:
                    metrics_history['val'][key].append(val_stats[key])
                else:
                    metrics_history['val'][key].append(0.0)
            
            metrics_history['grad_norm_mean'].append(train_stats.get('grad_norm_mean', 0.0))
            metrics_history['grad_norm_max'].append(train_stats.get('grad_norm_max', 0.0))
            metrics_history['learning_rate'].append(current_lr)
            metrics_history['samples_per_sec'].append(train_stats.get('samples_per_sec', 0.0))
            
            # Save metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
    
    if is_main_process:
        logger.info(f"\n{'='*60}")
        logger.info("Training complete!")
        logger.info(f"Total epochs: {args.num_epochs}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        total_time = time.time() - training_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        logger.info(f"Total training time: {hours}h {minutes}m {seconds}s")
        avg_time_per_epoch = total_time / args.num_epochs
        logger.info(f"Average time per epoch: {avg_time_per_epoch/60:.1f} minutes")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Metrics saved to: {output_dir / 'training_metrics.json'}")
        logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()

