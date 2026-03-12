#!/usr/bin/env python3
"""
Adversary Gym Environment for Phase 2: Mask Corruption as RL actions.

This environment allows an RL agent to corrupt vertebra masks adversarially
while respecting TS-like prior constraints (plausibility, budget, etc.).

The goal: Generate hard-but-realistic corruptions that maximize assembly loss.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from pathlib import Path

try:
    from scipy import ndimage as ndi
except ImportError:
    raise RuntimeError("scipy required: pip install scipy")


@dataclass
class CorruptionBudget:
    """Constraints on how much corruption the adversary can apply."""
    max_voxel_changes: int = 10000  # Max voxels that can be changed
    max_operations: int = 5  # Max number of operations per episode
    max_radius: int = 3  # Max morphology radius
    

class MaskCorruptionEnv(gym.Env):
    """
    Gym environment for adversarial mask corruption.
    
    **State**: Current corrupted mask + statistics
    **Action**: Apply one corruption operation (erosion/dilation/cutout/label-swap)
    **Reward**: Assembly loss increase + TS-like prior penalty
    
    Observation space:
      - Per-label features: volume, centroid, bbox, connected components
      - Global features: total labels, adjacency violations, budget used
    
    Action space:
      - operation: [erode, dilate, cutout, label_swap]
      - target_label: which vertebra to corrupt
      - magnitude: radius or extent parameter
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        clean_mask: np.ndarray,
        gt_mask: np.ndarray,
        budget: CorruptionBudget,
        max_steps: int = 10,
        assembly_module=None,
        ts_prior_weight: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            clean_mask: Clean TS mask (starting point)
            gt_mask: Ground truth mask (for assembly loss computation)
            budget: Corruption budget constraints
            max_steps: Maximum steps per episode
            assembly_module: Assembly module for computing mesh loss
            ts_prior_weight: Weight of TS-like prior penalty in reward
            seed: Random seed
        """
        super().__init__()
        
        self.clean_mask = clean_mask.copy()
        self.gt_mask = gt_mask.copy()
        self.budget = budget
        self.max_steps = max_steps
        self.assembly_module = assembly_module
        self.ts_prior_weight = ts_prior_weight
        
        # Extract labels
        self.labels = sorted(set(np.unique(clean_mask).tolist()) - {0})
        self.num_labels = len(self.labels)
        
        # Action space: [operation_type, target_label_idx, magnitude]
        # operation_type: 0=erode, 1=dilate, 2=cutout, 3=label_swap
        self.action_space = gym.spaces.MultiDiscrete([
            4,  # operation type
            self.num_labels,  # target label
            budget.max_radius + 1,  # magnitude (0 to max_radius)
        ])
        
        # Observation space: per-label features + global features
        # Per-label: [volume, cx, cy, cz, num_cc, bbox_volume] (6 features)
        # Global: [num_labels, budget_used_ratio, total_volume_change, adjacency_violations] (4 features)
        obs_dim = self.num_labels * 6 + 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Internal state
        self.current_mask = None
        self.voxels_changed = 0
        self.operations_used = 0
        self.step_count = 0
        self.initial_assembly_loss = None
        
        # Random state
        self.np_random = np.random.default_rng(seed)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to clean mask."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        self.current_mask = self.clean_mask.copy()
        self.voxels_changed = 0
        self.operations_used = 0
        self.step_count = 0
        
        # Compute initial assembly loss (for reward computation)
        # Use fast Dice proxy instead of slow mesh generation
        try:
            from ablation.run_phase0_baselines import dice_iou
        except:
            def dice_iou(pred, gt):
                pred = pred.astype(bool)
                gt = gt.astype(bool)
                inter = np.logical_and(pred, gt).sum()
                ps = pred.sum()
                gs = gt.sum()
                dice = (2.0 * inter + 1e-6) / (ps + gs + 1e-6)
                return float(dice), 0.0
        
        pred_bin = self.clean_mask > 0
        gt_bin = self.gt_mask > 0
        dice, _ = dice_iou(pred_bin, gt_bin)
        self.initial_assembly_loss = 1.0 - dice
        
        obs = self._get_observation()
        info = {"initial_assembly_loss": self.initial_assembly_loss}
        
        return obs, info
    
    def step(self, action):
        """
        Apply one corruption operation.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        operation_type, target_label_idx, magnitude = action
        
        # Map to actual label
        if target_label_idx >= len(self.labels):
            target_label_idx = len(self.labels) - 1
        target_label = self.labels[target_label_idx]
        
        # Apply corruption
        corrupted_mask, voxels_changed = self._apply_corruption(
            operation_type, target_label, magnitude
        )
        
        # Check budget
        if self.voxels_changed + voxels_changed > self.budget.max_voxel_changes:
            # Budget exceeded → terminate with penalty
            reward = -10.0
            terminated = True
            truncated = False
            obs = self._get_observation()
            info = {"budget_exceeded": True}
            return obs, reward, terminated, truncated, info
        
        # Update state
        self.current_mask = corrupted_mask
        self.voxels_changed += voxels_changed
        self.operations_used += 1
        self.step_count += 1
        
        # Compute reward
        reward, info = self._compute_reward()
        
        # Check termination
        terminated = (self.operations_used >= self.budget.max_operations)
        truncated = (self.step_count >= self.max_steps)
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_corruption(
        self,
        operation_type: int,
        target_label: int,
        magnitude: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Apply one corruption operation to the mask.
        
        Returns:
            corrupted_mask, num_voxels_changed
        """
        mask = self.current_mask.copy()
        
        # Extract binary mask for this label
        label_mask = (mask == target_label)
        if label_mask.sum() == 0:
            return mask, 0  # No voxels to corrupt
        
        # Compute bounding box (to reduce computation)
        coords = np.argwhere(label_mask)
        if len(coords) == 0:
            return mask, 0
        
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        pad = 5
        bbox_min = np.maximum(bbox_min - pad, 0)
        bbox_max = np.minimum(bbox_max + pad + 1, mask.shape)
        
        # Extract bbox
        i0, j0, k0 = bbox_min
        i1, j1, k1 = bbox_max
        bbox_mask = label_mask[i0:i1, j0:j1, k0:k1].copy()
        
        # Generate structuring element
        if magnitude > 0:
            struct = ndi.generate_binary_structure(3, 1)
            struct = ndi.iterate_structure(struct, magnitude)
        else:
            struct = None
        
        # Apply operation
        if operation_type == 0 and struct is not None:  # Erode
            bbox_mask_new = ndi.binary_erosion(bbox_mask, structure=struct)
        elif operation_type == 1 and struct is not None:  # Dilate
            bbox_mask_new = ndi.binary_dilation(bbox_mask, structure=struct)
        elif operation_type == 2:  # Cutout (random rectangular hole)
            bbox_mask_new = bbox_mask.copy()
            # Random cutout size
            cutout_size = magnitude * 5  # Scale magnitude
            if cutout_size > 0:
                # Random position
                sh = bbox_mask_new.shape
                cx = self.np_random.integers(0, max(1, sh[0] - cutout_size))
                cy = self.np_random.integers(0, max(1, sh[1] - cutout_size))
                cz = self.np_random.integers(0, max(1, sh[2] - cutout_size))
                bbox_mask_new[cx:cx+cutout_size, cy:cy+cutout_size, cz:cz+cutout_size] = False
        elif operation_type == 3:  # Label swap (swap with adjacent label)
            # Find adjacent labels (in spatial sense)
            # Simplified: just dilate and see what labels we touch
            if struct is not None:
                # Dilate within bbox
                dilated_bbox = ndi.binary_dilation(bbox_mask, structure=struct)
                full_mask_crop = mask[i0:i1, j0:j1, k0:k1]
                touching_labels = set(np.unique(full_mask_crop[dilated_bbox]).tolist()) - {0, target_label}
                if touching_labels:
                    # Swap with a random adjacent label
                    swap_label = self.np_random.choice(list(touching_labels))
                    # Swap regions
                    swap_mask = (full_mask_crop == swap_label)
                    full_mask_crop[bbox_mask] = swap_label
                    full_mask_crop[swap_mask] = target_label
                    mask[i0:i1, j0:j1, k0:k1] = full_mask_crop
                    voxels_changed = bbox_mask.sum() + swap_mask.sum()
                    return mask, int(voxels_changed)
            bbox_mask_new = bbox_mask  # No-op if can't swap
        else:
            bbox_mask_new = bbox_mask  # No-op
        
        # Count changed voxels
        voxels_changed = int(np.logical_xor(bbox_mask, bbox_mask_new).sum())
        
        # Update mask
        mask[i0:i1, j0:j1, k0:k1][bbox_mask] = 0
        mask[i0:i1, j0:j1, k0:k1][bbox_mask_new] = target_label
        
        return mask, voxels_changed
    
    def _compute_reward(self) -> Tuple[float, dict]:
        """
        Compute reward: assembly loss increase - TS-like prior penalty.
        
        Reward = (current_assembly_loss - initial_assembly_loss) - prior_penalty
        
        NOTE: Assembly mesh generation is VERY slow, so we use Dice as a fast proxy.
        Full mesh evaluation can be done post-training on saved corrupted masks.
        """
        # Fast proxy: use Dice loss instead of mesh-based loss
        # (Mesh generation via marching cubes is too slow for RL training)
        try:
            from ablation.run_phase0_baselines import dice_iou
        except:
            # Fallback inline implementation
            def dice_iou(pred, gt):
                pred = pred.astype(bool)
                gt = gt.astype(bool)
                inter = np.logical_and(pred, gt).sum()
                ps = pred.sum()
                gs = gt.sum()
                dice = (2.0 * inter + 1e-6) / (ps + gs + 1e-6)
                union = np.logical_or(pred, gt).sum()
                iou = (inter + 1e-6) / (union + 1e-6)
                return float(dice), float(iou)
        
        pred_bin = self.current_mask > 0
        gt_bin = self.gt_mask > 0
        dice, iou = dice_iou(pred_bin, gt_bin)
        current_assembly_loss = 1.0 - dice  # Lower dice = higher "loss"
        
        assembly_gain = current_assembly_loss - self.initial_assembly_loss
        
        # TS-like prior penalty
        prior_penalty = self._compute_prior_penalty()
        
        # Total reward
        reward = assembly_gain - self.ts_prior_weight * prior_penalty
        
        info = {
            "assembly_loss": current_assembly_loss,
            "assembly_gain": assembly_gain,
            "prior_penalty": prior_penalty,
            "reward": reward,
            "voxels_changed": self.voxels_changed,
            "operations_used": self.operations_used,
        }
        
        return float(reward), info
    
    def _compute_prior_penalty(self) -> float:
        """
        Compute TS-like prior penalty (plausibility violations).
        
        Penalties:
          - Excessive connected components per label
          - Volume changes outside plausible range
          - Adjacency violations (labels too far apart or overlapping)
        """
        penalty = 0.0
        
        for label in self.labels:
            label_mask = (self.current_mask == label)
            clean_label_mask = (self.clean_mask == label)
            
            if label_mask.sum() == 0:
                # Label completely eroded → high penalty
                penalty += 10.0
                continue
            
            # Connected components penalty
            try:
                labeled_cc, num_cc = ndi.label(label_mask)
                if num_cc > 3:  # Allow up to 3 components
                    penalty += (num_cc - 3) * 0.5
            except:
                # If labeling fails, add small penalty
                penalty += 1.0
            
            # Volume change penalty
            vol_current = label_mask.sum()
            vol_clean = clean_label_mask.sum()
            if vol_clean > 0:
                vol_ratio = vol_current / vol_clean
                if vol_ratio < 0.3 or vol_ratio > 3.0:  # More lenient bounds
                    penalty += min(abs(np.log(vol_ratio + 1e-6)), 5.0)  # Cap penalty
        
        # Adjacency penalty (simplified: check if labels are still in reasonable proximity)
        # TODO: Implement centroid distance checks
        
        return float(np.clip(penalty, 0.0, 50.0))  # Cap total penalty
    
    def _get_observation(self) -> np.ndarray:
        """
        Extract observation features from current mask.
        
        Returns flat vector of per-label + global features.
        """
        per_label_feats = []
        
        for label in self.labels:
            label_mask = (self.current_mask == label)
            
            if label_mask.sum() == 0:
                # Empty label
                feats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # Volume
                volume = float(label_mask.sum())
                
                # Centroid
                coords = np.argwhere(label_mask)
                centroid = coords.mean(axis=0)
                cx, cy, cz = centroid
                
                # Connected components
                _, num_cc = ndi.label(label_mask)
                
                # Bounding box volume
                bbox_min = coords.min(axis=0)
                bbox_max = coords.max(axis=0)
                bbox_vol = float(np.prod(bbox_max - bbox_min + 1))
                
                feats = [volume, cx, cy, cz, float(num_cc), bbox_vol]
            
            per_label_feats.extend(feats)
        
        # Global features
        num_labels_current = len(set(np.unique(self.current_mask).tolist()) - {0})
        budget_used_ratio = self.voxels_changed / (self.budget.max_voxel_changes + 1e-6)
        total_vol_current = (self.current_mask > 0).sum()
        total_vol_clean = (self.clean_mask > 0).sum()
        vol_change = abs(total_vol_current - total_vol_clean) / (total_vol_clean + 1e-6)
        
        global_feats = [
            float(num_labels_current),
            float(budget_used_ratio),
            float(vol_change),
            0.0,  # Placeholder for adjacency violations
        ]
        
        obs = np.array(per_label_feats + global_feats, dtype=np.float32)
        return obs

