"""
Phase 2: Encoder Pretraining Script

Self-supervised pretraining tasks:
1. Rotation Canonicalization
2. Contrastive Vertebra-Type Learning
3. Masked Point Modeling
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
from typing import Dict, Tuple
from datetime import datetime
import time
import math
from torch.optim.lr_scheduler import _LRScheduler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts and Warmup
    Based on: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int = 6,
        cycle_mult: float = 1.5,
        max_lr: float = 1e-03,
        min_lr: float = 1e-06,
        warmup_steps: int = 2,
        gamma: float = 0.9,
        last_epoch: int = -1
    ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr 
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * 
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                  (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# Use SE(3)-Transformer based encoder (more stable and well-tested)
from models.encoder_se3 import SE3PointEncoder
from models.encoder_se3 import features_to_irreps
from models.pretraining import RotationCanonicalizationLoss, ContrastiveVertebraTypeLoss, MaskedPointModelingLoss, random_rotation_matrix
from utils.data_loader import create_dataloader

# DDP support
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_logging(output_dir=None):
    """Setup logging"""
    import logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if output_dir is not None:
        log_file = Path(output_dir) / 'training.log'
        # Use 'a' mode to append, but clear file first for new training runs
        if log_file.exists():
            log_file.unlink()  # Remove old log file
        handlers.append(logging.FileHandler(log_file, mode='w'))  # Write mode
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True  # Force reconfiguration
    )
    logger = logging.getLogger(__name__)
    
    if output_dir is not None:
        logger.info(f"Logging to: {log_file}")
    
    return logger


def mask_points(points: torch.Tensor, features: torch.Tensor, mask_ratio: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask points for masked point modeling
    
    Args:
        points: (B, N, 3)
        features: (B, N, F)
        mask_ratio: Ratio of points to mask
    Returns:
        masked_points: (B, N, 3)
        masked_features: (B, N, F)
    """
    B, N, _ = points.shape
    num_mask = int(N * mask_ratio)
    
    masked_points = points.clone()
    masked_features = features.clone()
    
    for b in range(B):
        # Random indices to mask
        mask_indices = torch.randperm(N, device=points.device)[:num_mask]
        
        # Zero out masked points
        masked_points[b, mask_indices] = 0
        masked_features[b, mask_indices] = 0
    
    return masked_points, masked_features


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    logger,
    is_main_process: bool = True,
    rank: int = 0,
) -> Dict[str, float]:
    """Train for one epoch"""
    use_ddp = isinstance(model, DDP)
    model.train()
    
    total_loss = 0.0
    loss_components = {name: 0.0 for name in losses.keys()}
    num_batches = 0
    
    # Additional metrics
    gradient_norms = []
    positive_pairs_count = 0
    total_contrastive_batches = 0
    start_time = time.time()
    
    # Position: 0 for overall, rank+1 for each rank's progress bar
    pbar = tqdm(dataloader, desc=f'Training [Rank {rank}]', position=rank+1, leave=True, dynamic_ncols=True)
    for batch_num, batch in enumerate(pbar):
        points = batch['points'].to(device)  # (B, N, 3)
        features = batch['features'].to(device)  # (B, N, F) - [normals(3), curvature(2)]
        labels = batch['label'].to(device)  # (B,)
        
        B, N, _ = points.shape
        
        # Convert features to IrrepsArray format
        feat_irreps = features_to_irreps(features, use_curvature=True)
        
        # Flatten for batch processing
        points_flat = points.view(B * N, 3)  # (B*N, 3)
        batch_indices = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)
        
        # Initialize gradients
        optimizer.zero_grad()
        
        # Task 1: Rotation Canonicalization
        loss_rot = torch.tensor(0.0, device=device)
        if 'rotation' in losses:
            R = random_rotation_matrix(B, device)  # (B, 3, 3)
                # Rotate points: (B, N, 3) @ (B, 3, 3) → (B, N, 3)
            points_rotated = torch.bmm(points, R)  # (B, N, 3)
                points_rotated_flat = points_rotated.view(B * N, 3)  # (B*N, 3)
            
            # Rotate normals as well
            if features.shape[-1] >= 3:
                normals = features[:, :, :3]  # (B, N, 3)
                    normals_rotated = torch.bmm(normals, R)  # (B, N, 3)
                features_rotated = features.clone()
                features_rotated[:, :, :3] = normals_rotated
            else:
                features_rotated = features
            
                # Convert rotated features to IrrepsArray
                feat_rotated_irreps = features_to_irreps(features_rotated, use_curvature=True)
                
            # Encode rotated point cloud
            output_rotated = model(points_rotated_flat, feat_rotated_irreps, batch=batch_indices)
                embedding_rotated = output_rotated['embedding']  # (B, output_dim)
            
            # Rotation canonicalization loss
            loss_rot, _ = losses['rotation'](embedding_rotated, R)
            if torch.isnan(loss_rot) or torch.isinf(loss_rot):
                if is_main_process:
                    logger.warning("NaN/Inf in rotation loss, skipping batch")
                continue
            loss_components['rotation'] += loss_rot.item()
        
        # Task 2: Contrastive Vertebra-Type Learning
        loss_contrast = torch.tensor(0.0, device=device)
        if 'contrastive' in losses:
            output = model(points_flat, feat_irreps, batch=batch_indices)
                embedding = output['embedding']  # (B, output_dim)
                
            unique_labels = torch.unique(labels)
            has_positive_pairs = len(unique_labels) < len(labels)
                        total_contrastive_batches += 1
                        if has_positive_pairs:
                            positive_pairs_count += 1
                loss_contrast = losses['contrastive'](embedding, labels)
                            if torch.isnan(loss_contrast) or torch.isinf(loss_contrast):
                                if is_main_process:
                        logger.warning("NaN/Inf in contrastive loss, skipping.")
                                loss_contrast = torch.tensor(0.0, device=device)
                        else:
                            loss_contrast = torch.tensor(0.0, device=device)
            loss_components['contrastive'] += loss_contrast.item()
        
        # Task 3: Masked Point Modeling
        loss_masked = torch.tensor(0.0, device=device)
        if 'masked' in losses:
            output_full = model(points_flat, feat_irreps, batch=batch_indices)
                embedding_full = output_full['embedding']  # (B, output_dim)
            
            points_masked, features_masked = mask_points(points, features, mask_ratio=0.4)
                points_masked_flat = points_masked.view(B * N, 3)  # (B*N, 3)
                feat_masked_irreps = features_to_irreps(features_masked, use_curvature=True)
                
            output_masked = model(points_masked_flat, feat_masked_irreps, batch=batch_indices)
                embedding_masked = output_masked['embedding']  # (B, output_dim)
            
            loss_masked = losses['masked'](embedding_full, embedding_masked)
            loss_components['masked'] += loss_masked.item()
        
        # Combined loss
        total_batch_loss = (
            loss_weights.get('rotation', 1.0) * loss_rot +
            loss_weights.get('contrastive', 1.0) * loss_contrast +
            loss_weights.get('masked', 1.0) * loss_masked
        )
        
        # Backward pass
        total_batch_loss.backward()
        
        # Compute gradient norms (before optimizer step)
        total_grad_norm = 0.0
        max_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                max_grad_norm = max(max_grad_norm, param_norm.item())
        total_grad_norm = total_grad_norm ** (1. / 2)
        gradient_norms.append(total_grad_norm)
        
        # Update optimizer
        optimizer.step()
            optimizer.zero_grad()
        
        # Convert to Python float for logging (frees GPU memory immediately)
        loss_val = float(total_batch_loss.item())
        rot_val = float(loss_rot.item()) if 'rotation' in losses else 0.0
        cont_val = float(loss_contrast.item()) if 'contrastive' in losses else 0.0
        mask_val = float(loss_masked.item()) if 'masked' in losses else 0.0
        
        total_loss += loss_val
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'rot': f'{rot_val:.4f}' if 'rotation' in losses else 'N/A',
            'cont': f'{cont_val:.4f}' if 'contrastive' in losses else 'N/A',
            'mask': f'{mask_val:.4f}' if 'masked' in losses else 'N/A',
        })
        
        # Free memory: clear intermediate variables
        if True:  # Always clear to free memory
            try:
                del total_batch_loss, loss_rot, loss_contrast, loss_masked
                if 'rotation' in losses:
                    del embedding_rotated, points_rotated, features_rotated, R
                if 'masked' in losses:
                    del embedding_full, embedding_masked, points_masked, features_masked
                if 'contrastive' in losses and 'embedding' in locals():
                    del embedding
            except:
                pass  # Variables may already be deleted
    
    # Average losses (aggregate across all processes for DDP)
    if use_ddp:
        # Gather metrics from all processes
        metrics_list = [total_loss, num_batches] + list(loss_components.values())
        metrics_tensor = torch.tensor(metrics_list, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss_all = metrics_tensor[0].item()
        num_batches_all = int(metrics_tensor[1].item())
        loss_components_all = {name: metrics_tensor[i+2].item() for i, name in enumerate(loss_components.keys())}
        
        avg_loss = total_loss_all / num_batches_all if num_batches_all > 0 else 0.0
        avg_components = {name: val / num_batches_all if num_batches_all > 0 else 0.0 
                          for name, val in loss_components_all.items()}
    else:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_components = {name: val / num_batches if num_batches > 0 else 0.0 
                          for name, val in loss_components.items()}
    
    # Compute additional metrics
    elapsed_time = time.time() - start_time if start_time else 0.0
    num_samples = num_batches
    samples_per_sec = num_samples / elapsed_time if elapsed_time > 0 else 0.0
    
    avg_grad_norm = float(np.mean(gradient_norms)) if gradient_norms else 0.0
    max_grad_norm = float(np.max(gradient_norms)) if gradient_norms else 0.0
    
    positive_pairs_ratio = (positive_pairs_count / total_contrastive_batches * 100.0) if total_contrastive_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        **avg_components,
        'grad_norm_mean': avg_grad_norm,
        'grad_norm_max': max_grad_norm,
        'positive_pairs_ratio': positive_pairs_ratio,
        'samples_per_sec': samples_per_sec,
        'elapsed_time': elapsed_time,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    losses: Dict[str, nn.Module],
    loss_weights: Dict[str, float],
    logger,
    is_main_process: bool = True,
    rank: int = 0,
) -> Dict[str, float]:
    """Validate for one epoch"""
    use_ddp = isinstance(model, DDP)
    model.eval()
    
    total_loss = 0.0
    loss_components = {name: 0.0 for name in losses.keys()}
    num_batches = 0
    
    # Additional metrics for detailed logging
    gradient_norms = []
    positive_pairs_count = 0
    total_contrastive_batches = 0
    import time
    start_time = time.time()
    
    with torch.no_grad():
        # Position: 0 for overall, rank+1 for each rank's progress bar
        pbar = tqdm(dataloader, desc=f'Validation [Rank {rank}]', disable=not is_main_process, position=rank+1, leave=False)
        for batch in pbar:
            points = batch['points'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            B, N, _ = points.shape
            
            # Convert features to IrrepsArray format
            feat_irreps = features_to_irreps(features, use_curvature=True)
            points_flat = points.view(B * N, 3)
            batch_indices = torch.arange(B, device=device).repeat_interleave(N)
            
            # Task 1: Rotation Canonicalization
            if 'rotation' in losses:
                R = random_rotation_matrix(B, device)
                points_rotated = torch.bmm(points, R)
                    points_rotated_flat = points_rotated.view(B * N, 3)
                
                if features.shape[-1] >= 3:
                    normals = features[:, :, :3]
                    normals_rotated = torch.bmm(normals, R)
                    features_rotated = features.clone()
                    features_rotated[:, :, :3] = normals_rotated
                else:
                    features_rotated = features
                
                    feat_rotated_irreps = features_to_irreps(features_rotated, use_curvature=True)
                batch_indices = torch.arange(B, device=device).repeat_interleave(N)
                output_rotated = model(points_rotated_flat, feat_rotated_irreps, batch=batch_indices)
                    embedding_rotated = output_rotated['embedding']
                loss_rot, _ = losses['rotation'](embedding_rotated, R)
            else:
                loss_rot = torch.tensor(0.0, device=device)
            
            if 'rotation' in losses:
                loss_components['rotation'] += loss_rot.item()
            
            # Task 2: Contrastive
            if 'contrastive' in losses:
                batch_indices = torch.arange(B, device=device).repeat_interleave(N)
                output = model(points_flat, feat_irreps, batch=batch_indices)
                    embedding = output['embedding']
                    
                    # Check if we have positive pairs (same labels in batch)
                    unique_labels = torch.unique(labels)
                    has_positive_pairs = len(unique_labels) < len(labels)
                    
                    if has_positive_pairs:
                loss_contrast = losses['contrastive'](embedding, labels)
                    else:
                        # No positive pairs in batch (e.g., batch_size=1)
                        loss_contrast = torch.tensor(0.0, device=device)
            else:
                loss_contrast = torch.tensor(0.0, device=device)
            
            if 'contrastive' in losses:
                loss_components['contrastive'] += loss_contrast.item()
            
            # Task 3: Masked
            if 'masked' in losses:
                batch_indices = torch.arange(B, device=device).repeat_interleave(N)
                output_full = model(points_flat, feat_irreps, batch=batch_indices)
                    embedding_full = output_full['embedding']
                    
                points_masked, features_masked = mask_points(points, features, mask_ratio=0.4)
                    points_masked_flat = points_masked.view(B * N, 3)
                    feat_masked_irreps = features_to_irreps(features_masked, use_curvature=True)
                    
                output_masked = model(points_masked_flat, feat_masked_irreps, batch=batch_indices)
                    embedding_masked = output_masked['embedding']
                loss_masked = losses['masked'](embedding_full, embedding_masked)
            else:
                loss_masked = torch.tensor(0.0, device=device)
            
            if 'masked' in losses:
                loss_components['masked'] += loss_masked.item()
            
            total_batch_loss = (
                loss_weights.get('rotation', 1.0) * loss_rot +
                loss_weights.get('contrastive', 1.0) * loss_contrast +
                loss_weights.get('masked', 1.0) * loss_masked
            )
            
            total_loss += total_batch_loss.item()
            num_batches += 1
    
    # Average losses (aggregate across all processes for DDP)
    if use_ddp:
        # Gather metrics from all processes
        metrics_list = [total_loss, num_batches] + list(loss_components.values())
        metrics_tensor = torch.tensor(metrics_list, device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss_all = metrics_tensor[0].item()
        num_batches_all = int(metrics_tensor[1].item())
        loss_components_all = {name: metrics_tensor[i+2].item() for i, name in enumerate(loss_components.keys())}
        
        avg_loss = total_loss_all / num_batches_all if num_batches_all > 0 else 0.0
        avg_components = {name: val / num_batches_all if num_batches_all > 0 else 0.0 
                          for name, val in loss_components_all.items()}
    else:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_components = {name: val / num_batches if num_batches > 0 else 0.0 
                          for name, val in loss_components.items()}
    
    return {
        'total_loss': avg_loss,
        **avg_components,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Encoder Pretraining')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing point cloud data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of attention layers')
    parser.add_argument('--output_dim', type=int, default=512,
                        help='Output embedding dimension')
    parser.add_argument('--max_points', type=int, default=2048,
                        help='Maximum number of points per vertebra')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--use_rotation', action='store_true',
                        help='Use rotation canonicalization task')
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Use contrastive learning task')
    parser.add_argument('--use_masked', action='store_true',
                        help='Use masked point modeling task')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    # Cosine warmup scheduler parameters
    parser.add_argument('--first_cycle_steps', type=int, default=20,
                        help='First cycle steps for cosine warmup scheduler')
    parser.add_argument('--cycle_mult', type=float, default=1.5,
                        help='Cycle multiplier for cosine warmup scheduler')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate for cosine warmup scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate for cosine warmup scheduler')
    parser.add_argument('--warmup_steps', type=int, default=5,
                        help='Warmup steps for cosine warmup scheduler (must be < first_cycle_steps)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9,
                        help='Gamma (decrease rate) for cosine warmup scheduler')
    
    args = parser.parse_args()
    
    # DDP setup - torchrun sets these environment variables (need to check before creating output dir)
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    use_ddp = local_rank >= 0
    if use_ddp:
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
    
    # Log training configuration (only on main process, but setup logging first)
    if is_main_process:
        logger.info("="*60)
        logger.info("Encoder Pretraining Configuration")
        logger.info("="*60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Point cloud directory: {args.point_cloud_dir}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Number of epochs: {args.num_epochs}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Hidden dimension: {args.hidden_dim}")
        logger.info(f"Number of layers: {args.num_layers}")
        logger.info(f"Output dimension: {args.output_dim}")
        logger.info(f"Max points: {args.max_points}")
        logger.info(f"Number of workers: {args.num_workers}")
        logger.info(f"Use rotation: {args.use_rotation}")
        logger.info(f"Use contrastive: {args.use_contrastive}")
        logger.info(f"Use masked: {args.use_masked}")
        if args.resume:
            logger.info(f"Resume from: {args.resume}")
        logger.info("="*60)
        logger.info(f"Using device: {device}")
        if use_ddp:
            logger.info(f"DDP: rank {rank}/{world_size} (local_rank: {local_rank})")
        logger.info("")
    
    # Create data loaders
    if is_main_process:
        logger.info("Creating data loaders...")
    
    # DDP: use DistributedSampler
    train_dataset = create_dataloader(
        point_cloud_dir=Path(args.point_cloud_dir),
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_points=args.max_points,
        augment=True,
        shuffle=False,  # DistributedSampler handles shuffling
        return_dataset=True,  # Return dataset instead of DataLoader
    )
    
    val_dataset = create_dataloader(
        point_cloud_dir=Path(args.point_cloud_dir),
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_points=args.max_points,
        augment=False,
        shuffle=False,
        return_dataset=True,
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
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Create model
    if is_main_process:
        logger.info("Creating SE(3)-equivariant encoder (SE(3)-Transformer based)...")
    model = SE3PointEncoder(
        irreps_in="2x0e + 1x1o",  # curvature + normals (for compatibility, not used directly)
        irreps_hidden="32x0e + 16x1o + 8x2e",  # TorchMD-Net style hidden representation
        irreps_inv_out="64x0e",  # invariant output
        irreps_eq_out="8x1o",    # equivariant output
        out_dim=args.output_dim,
        num_layers=args.num_layers,
        num_radial=16,  # RBF radial basis size
        lmax=2,  # max spherical harmonics degree
        cutoff=5.0,  # radius graph cutoff distance
        max_num_neighbors=32,  # max neighbors per node
        use_curvature=True,
    ).to(device)
    
    # Wrap with DDP if using distributed training
    if use_ddp:
        # find_unused_parameters=True: Some parameters may not be used in every forward pass
        # (e.g., when certain loss tasks are disabled)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model_module = model.module  # Access underlying model
    else:
        model_module = model
    
    # Create losses
    losses = {}
    if args.use_rotation:
        losses['rotation'] = RotationCanonicalizationLoss(embedding_dim=args.output_dim).to(device)
    if args.use_contrastive:
        losses['contrastive'] = ContrastiveVertebraTypeLoss().to(device)
    if args.use_masked:
        losses['masked'] = MaskedPointModelingLoss().to(device)
    
    if not losses:
        logger.error("No pretraining tasks selected! Use --use_rotation, --use_contrastive, or --use_masked")
        return
    
    # Log active losses
    if is_main_process:
        logger.info(f"Active pretraining tasks: {list(losses.keys())}")
        logger.info(f"Batch size: {args.batch_size}")
        if 'contrastive' in losses:
            if args.batch_size < 4:
                logger.warning(f"⚠️  WARNING: Small batch size ({args.batch_size}) may result in few positive pairs")
            else:
                logger.info(f"✅ Batch size ({args.batch_size}) should provide enough positive pairs for contrastive learning")
    
    # Loss weights
    # Adjusted based on evaluation results:
    # - Rotation: increased (rotation invariance is poor, need more focus)
    # - Contrastive: maintained (working well)
    # - Masked: maintained (working well)
    loss_weights = {
        'rotation': 1.2,     # Increased: rotation invariance needs improvement (was 0.3)
        'contrastive': 3.0,   # Maintained: fine-grained discrimination (working well)
        'masked': 2.0,        # Maintained: embedding diversity (working well)
    }
    
    if is_main_process:
        logger.info(f"Loss weights: {loss_weights}")
        logger.info("  - Rotation: 1.2 (increased to improve rotation invariance)")
        logger.info("  - Contrastive: 3.0 (fine-grained vertebra type discrimination)")
        logger.info("  - Masked: 2.0 (embedding diversity)")
    
    # Optimizer
    optimizer = optim.AdamW(
        list(model_module.parameters()) + [p for loss_fn in losses.values() for p in loss_fn.parameters()],
        lr=args.learning_rate,
    )
    
    # Learning rate scheduler (always enabled with default parameters)
    # Validate scheduler parameters: warmup_steps must be < first_cycle_steps
    if args.warmup_steps >= args.first_cycle_steps:
        if is_main_process:
            logger.warning(f"warmup_steps ({args.warmup_steps}) >= first_cycle_steps ({args.first_cycle_steps})")
            logger.warning(f"Adjusting warmup_steps to {args.first_cycle_steps - 1}")
        args.warmup_steps = args.first_cycle_steps - 1
    
        max_lr = args.max_lr if args.max_lr else args.learning_rate
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=args.first_cycle_steps,
            cycle_mult=args.cycle_mult,
            max_lr=max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            gamma=args.scheduler_gamma,
        )
        if is_main_process:
        logger.info(f"✅ Using CosineAnnealingWarmupRestarts scheduler (default)")
            logger.info(f"   First cycle steps: {args.first_cycle_steps}")
            logger.info(f"   Cycle multiplier: {args.cycle_mult}")
            logger.info(f"   Max LR: {max_lr:.2e}")
            logger.info(f"   Min LR: {args.min_lr:.2e}")
            logger.info(f"   Warmup steps: {args.warmup_steps}")
            logger.info(f"   Gamma: {args.scheduler_gamma}")
    
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume is not None:
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
            model_module.load_state_dict(checkpoint['model_state_dict'])
            if is_main_process:
                logger.info("Loaded model state dict")
        else:
            # Try loading directly (for compatibility)
            try:
                model_module.load_state_dict(checkpoint)
                if is_main_process:
                    logger.info("Loaded model state dict (direct)")
            except Exception as e:
                if is_main_process:
                    logger.error(f"Failed to load model state dict: {e}")
                return
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main_process:
                logger.info("Loaded optimizer state dict")
        
        
        # Load scheduler state (if using scheduler)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_main_process:
                logger.info("Loaded scheduler state dict")
        
        # Load training state
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            if is_main_process:
                logger.info(f"Resuming from epoch {start_epoch}")
        
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            if is_main_process:
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        if is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
    
    # Training loop
    if is_main_process:
        if args.resume:
            logger.info(f"Resuming training from epoch {start_epoch}/{args.num_epochs}")
        else:
            logger.info("Starting training...")
        
        # Initialize previous losses for loss change tracking
        prev_train_loss = None
        prev_val_loss = None
        
        # Estimate total training time
        import time
        training_start_time = time.time()
    
    # Overall progress bar (position=0, only on main process)
    if is_main_process:
        epoch_pbar = tqdm(range(start_epoch, args.num_epochs), desc='Overall Progress', position=0, leave=True)
    else:
        epoch_pbar = None
    
    # Get rank (default to 0 if not in DDP)
    current_rank = rank if rank >= 0 else 0
    
    for epoch in range(start_epoch, args.num_epochs):
        # DDP: set epoch for sampler
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
            if epoch_pbar is not None:
                epoch_pbar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, losses, loss_weights, logger, is_main_process, rank=current_rank
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, device, losses, loss_weights, logger, is_main_process, rank=current_rank
        )
        
        # Update overall progress bar
        if epoch_pbar is not None:
            epoch_pbar.update(1)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(epoch=epoch)
        
        # Log metrics (only on main process)
        if is_main_process:
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute loss changes (if previous epoch exists)
            loss_change_info = ""
            if epoch > start_epoch and prev_train_loss is not None and prev_val_loss is not None:
                train_loss_change = train_metrics['total_loss'] - prev_train_loss
                val_loss_change = val_metrics['total_loss'] - prev_val_loss
                train_change_str = f"{train_loss_change:+.4f}" if not np.isnan(train_loss_change) else "N/A"
                val_change_str = f"{val_loss_change:+.4f}" if not np.isnan(val_loss_change) else "N/A"
                loss_change_info = f" (Δ: {train_change_str} / {val_change_str})"
            
            logger.info("")
            logger.info("="*60)
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} Summary")
            logger.info("="*60)
            logger.info(f"[Training]")
            logger.info(f"  Total Loss: {train_metrics['total_loss']:.4f}{loss_change_info}")
            for key in train_metrics:
                if key not in ['total_loss', 'grad_norm_mean', 'grad_norm_max', 'positive_pairs_ratio', 
                            'samples_per_sec', 'elapsed_time']:
                    logger.info(f"  {key.capitalize()} Loss: {train_metrics[key]:.4f}")
            
            # Additional training metrics
            if 'grad_norm_mean' in train_metrics:
                logger.info(f"  Gradient Norm: mean={train_metrics['grad_norm_mean']:.4f}, max={train_metrics['grad_norm_max']:.4f}")
            if 'positive_pairs_ratio' in train_metrics:
                logger.info(f"  Contrastive Positive Pairs: {train_metrics['positive_pairs_ratio']:.1f}%")
            if 'samples_per_sec' in train_metrics:
                logger.info(f"  Training Speed: {train_metrics['samples_per_sec']:.2f} samples/sec")
            
            logger.info(f"[Validation]")
            logger.info(f"  Total Loss: {val_metrics['total_loss']:.4f}{loss_change_info}")
            for key in val_metrics:
                if key not in ['total_loss', 'grad_norm_mean', 'grad_norm_max', 'positive_pairs_ratio', 
                              'samples_per_sec', 'elapsed_time']:
                    logger.info(f"  {key.capitalize()} Loss: {val_metrics.get(key, 0.0):.4f}")
            
            # Update best_val_loss before logging
            is_improved = val_metrics['total_loss'] < best_val_loss
            if is_improved:
                improvement = best_val_loss - val_metrics['total_loss']
                best_val_loss = val_metrics['total_loss']
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
            prev_train_loss = train_metrics['total_loss']
            prev_val_loss = val_metrics['total_loss']
            
            # Check for NaN
            if np.isnan(train_metrics['total_loss']) or np.isnan(val_metrics['total_loss']):
                logger.error(f"NaN detected in metrics! Train: {train_metrics}, Val: {val_metrics}")
                logger.error("Stopping training due to NaN loss")
                break
            
            # Save metrics to JSON for visualization
            metrics_file = output_dir / 'training_metrics.json'
            if epoch == start_epoch:
                # Initialize metrics history
                metrics_history = {
                    'epochs': [],
                    'train': {'total_loss': [], 'rotation': [], 'contrastive': [], 'masked': []},
                    'val': {'total_loss': [], 'rotation': [], 'contrastive': [], 'masked': []},
                    'grad_norm_mean': [],
                    'grad_norm_max': [],
                    'positive_pairs_ratio': [],
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
                            'train': {'total_loss': [], 'rotation': [], 'contrastive': [], 'masked': []},
                            'val': {'total_loss': [], 'rotation': [], 'contrastive': [], 'masked': []},
                            'grad_norm_mean': [],
                            'grad_norm_max': [],
                            'positive_pairs_ratio': [],
                            'learning_rate': [],
                            'samples_per_sec': [],
                        }
                else:
                    metrics_history = {
                        'epochs': [],
                        'train': {'total_loss': [], 'rotation': [], 'contrastive': [], 'masked': []},
                        'val': {'total_loss': [], 'rotation': [], 'contrastive': [], 'masked': []},
                        'grad_norm_mean': [],
                        'grad_norm_max': [],
                        'positive_pairs_ratio': [],
                        'learning_rate': [],
                        'samples_per_sec': [],
                    }
            
            # Append current metrics
            metrics_history['epochs'].append(epoch + 1)
            metrics_history['train']['total_loss'].append(float(train_metrics['total_loss']))
            metrics_history['train']['rotation'].append(float(train_metrics.get('rotation', 0.0)))
            metrics_history['train']['contrastive'].append(float(train_metrics.get('contrastive', 0.0)))
            metrics_history['train']['masked'].append(float(train_metrics.get('masked', 0.0)))
            metrics_history['val']['total_loss'].append(float(val_metrics['total_loss']))
            metrics_history['val']['rotation'].append(float(val_metrics.get('rotation', 0.0)))
            metrics_history['val']['contrastive'].append(float(val_metrics.get('contrastive', 0.0)))
            metrics_history['val']['masked'].append(float(val_metrics.get('masked', 0.0)))
            metrics_history['grad_norm_mean'].append(float(train_metrics.get('grad_norm_mean', 0.0)))
            metrics_history['grad_norm_max'].append(float(train_metrics.get('grad_norm_max', 0.0)))
            metrics_history['positive_pairs_ratio'].append(float(train_metrics.get('positive_pairs_ratio', 0.0)))
            metrics_history['learning_rate'].append(float(current_lr))
            metrics_history['samples_per_sec'].append(float(train_metrics.get('samples_per_sec', 0.0)))
            
            # Save metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_history, f, indent=2)
        
        # Save checkpoint (only on main process)
        if is_main_process:
            # Check if this is an improvement (best_val_loss was already updated above)
            is_improved = val_metrics['total_loss'] == best_val_loss
            if is_improved:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_module.state_dict(),  # Save underlying model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['total_loss'],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint, output_dir / 'best_model.pth')
                logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Save latest checkpoint
            latest_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if scheduler is not None:
                latest_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(latest_checkpoint, output_dir / 'latest_model.pth')
    
    # Close overall progress bar
    if epoch_pbar is not None:
        epoch_pbar.close()
    
    if is_main_process:
        total_training_time = time.time() - training_start_time
        hours = int(total_training_time // 3600)
        minutes = int((total_training_time % 3600) // 60)
        seconds = int(total_training_time % 60)
        
        logger.info("")
        logger.info("="*60)
        logger.info("Training complete!")
        logger.info(f"Total epochs: {args.num_epochs}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Total training time: {hours}h {minutes}m {seconds}s")
        if args.num_epochs > 0:
            logger.info(f"Average time per epoch: {total_training_time / args.num_epochs / 60:.1f} minutes")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Metrics saved to: {output_dir / 'training_metrics.json'}")
        logger.info("="*60)
    
    # Cleanup DDP
    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

