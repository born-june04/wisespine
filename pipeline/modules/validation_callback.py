#!/usr/bin/env python3
"""
Validation callback for PyBullet RL training.

Periodically evaluates agent and creates visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import subprocess
import glob
import json

from modules.pybullet_ct_renderer import render_pybullet_to_ct


class ValidationCallback(BaseCallback):
    """
    Callback to periodically evaluate agent and create visualizations.
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        output_dir: str = "validation_outputs",
        n_eval_episodes: int = 3,
        run_ts: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.output_dir = Path(output_dir)
        self.n_eval_episodes = n_eval_episodes
        self.run_ts = run_ts
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track validation history
        self.validation_history = []
        
    def _on_step(self):
        # Check if it's time to validate
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        if self.verbose > 0:
            print("\n" + "="*70)
            print(f"VALIDATION @ Step {self.num_timesteps}")
            print("="*70)
        
        # Evaluate agent
        rewards = []
        displacements_all = []
        
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            rewards.append(ep_reward)
            
            # Get displacements
            disps = self.eval_env.get_fragment_displacements()
            displacements_all.append(disps)
        
        avg_reward = np.mean(rewards)
        
        if self.verbose > 0:
            print(f"\n📊 Evaluation Results:")
            print(f"  Episodes: {self.n_eval_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
        
        # Use best episode for visualization
        best_ep_idx = np.argmax(rewards)
        
        if self.verbose > 0:
            print(f"\n🎨 Creating visualizations (using episode {best_ep_idx})...")
        
        # Reset to best episode state (re-run)
        obs, _ = self.eval_env.reset()
        for _ in range(50):  # Run full episode
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            if terminated or truncated:
                break
        
        # Render CT (but don't save full CT to disk - disk quota issue)
        step_dir = self.output_dir / f"step_{self.num_timesteps:06d}"
        step_dir.mkdir(exist_ok=True)
        
        # Only save visualization, not full CT
        fractured_ct, fractured_mask = render_pybullet_to_ct(
            self.eval_env,
            output_ct_path=None,  # Don't save CT
            output_mask_path=None  # Don't save mask
        )
        
        if self.verbose > 0:
            print(f"  ✓ CT rendered (in-memory, not saved to disk)")
        
        # Create visualization
        self._create_visualization(
            fractured_ct,
            fractured_mask,
            displacements_all[best_ep_idx],
            step_dir
        )
        
        # Skip TotalSegmentator to save disk space
        dice_score = None
        if False:  # Disabled to save disk quota
            if self.verbose > 0:
                print(f"\n🔬 Running TotalSegmentator...")
            
            # Save temporary CT just for TS
            temp_ct_path = step_dir / "temp_ct.nii.gz"
            ct_nii = nib.Nifti1Image(
                fractured_ct,
                self.eval_env.gt_ct_nii.affine,
                self.eval_env.gt_ct_nii.header
            )
            nib.save(ct_nii, str(temp_ct_path))
            
            dice_score = self._run_totalsegmentator(temp_ct_path, step_dir)
            
            # Clean up temp file
            temp_ct_path.unlink()
            
            if dice_score is not None and self.verbose > 0:
                print(f"  ✓ Dice Score: {dice_score:.4f}")
        
        # Save validation entry
        val_entry = {
            'timestep': int(self.num_timesteps),
            'avg_reward': float(avg_reward),
            'rewards': [float(r) for r in rewards],
            'dice_score': float(dice_score) if dice_score is not None else None,
            'max_displacement_mm': float(np.max([np.linalg.norm(d[1])*1000 for d in displacements_all[best_ep_idx]])),
        }
        
        self.validation_history.append(val_entry)
        
        # Save to JSONL
        with open(self.output_dir / "validation_history.jsonl", 'a') as f:
            f.write(json.dumps(val_entry) + '\n')
        
        # Plot training progress
        self._plot_training_progress()
        
        if self.verbose > 0:
            print(f"\n✓ Validation complete! Results saved to: {step_dir}")
            print("="*70 + "\n")
        
        return True
    
    def _create_visualization(self, fractured_ct, fractured_mask, displacements, output_dir):
        """Create comparison visualization."""
        
        # Load original
        original_ct_nii = nib.load("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
        original_ct = original_ct_nii.get_fdata()
        
        gt_mask_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
        gt_mask = gt_mask_nii.get_fdata()
        
        # Find L1 region
        l1_mask = (gt_mask == 20)
        l1_coords = np.where(l1_mask)
        l1_center = [int(np.mean(c)) for c in l1_coords]
        
        margin = 50
        l1_bbox = [
            (max(0, l1_coords[0].min() - margin), min(gt_mask.shape[0], l1_coords[0].max() + margin)),
            (max(0, l1_coords[1].min() - margin), min(gt_mask.shape[1], l1_coords[1].max() + margin)),
            (max(0, l1_coords[2].min() - margin), min(gt_mask.shape[2], l1_coords[2].max() + margin)),
        ]
        
        vmin, vmax = -200, 1500
        
        # 2x3 comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'RL-Generated Fracture @ Step {self.num_timesteps}', 
                     fontsize=16, fontweight='bold')
        
        # Extract slices
        orig_sag = original_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
        orig_ax = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
        orig_cor = original_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]
        
        frac_sag = fractured_ct[l1_center[0], l1_bbox[1][0]:l1_bbox[1][1], l1_bbox[2][0]:l1_bbox[2][1]]
        frac_ax = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_bbox[1][0]:l1_bbox[1][1], l1_center[2]]
        frac_cor = fractured_ct[l1_bbox[0][0]:l1_bbox[0][1], l1_center[1], l1_bbox[2][0]:l1_bbox[2][1]]
        
        # Row 1: Original
        axes[0, 0].imshow(orig_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title('Original - Sagittal', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orig_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title('Original - Axial', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(orig_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 2].set_title('Original - Coronal', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Fractured
        axes[1, 0].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, 0].set_title('RL Fractured - Sagittal', fontsize=12, color='red', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(frac_ax.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, 1].set_title('RL Fractured - Axial', fontsize=12, color='red', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(frac_cor.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, 2].set_title('RL Fractured - Coronal', fontsize=12, color='red', fontweight='bold')
        axes[1, 2].axis('off')
        
        # Add displacement info
        max_disp = np.max([np.linalg.norm(d[1])*1000 for d in displacements])
        fig.text(0.5, 0.02, f'Max Displacement: {max_disp:.2f}mm', 
                ha='center', fontsize=12, color='blue')
        
        plt.tight_layout()
        
        output_path = output_dir / "fracture_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _run_totalsegmentator(self, ct_path, output_dir):
        """Run TotalSegmentator and compute Dice."""
        
        ts_output_dir = output_dir / "ts_predictions"
        
        try:
            # Run TS
            cmd = [
                "TotalSegmentator",
                "-i", str(ct_path),
                "-o", str(ts_output_dir),
                "-ta", "total",
                "--fast"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            
            if result.returncode != 0:
                print(f"  ⚠ TotalSegmentator failed: {result.stderr}")
                return None
            
            # Load predictions
            gt_mask_nii = nib.load("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
            gt_mask = gt_mask_nii.get_fdata()
            
            ts_files = sorted(glob.glob(str(ts_output_dir / "vertebrae_*.nii.gz")))
            ts_mask = np.zeros_like(gt_mask)
            
            for ts_file in ts_files:
                vert_data = nib.load(ts_file).get_fdata()
                ts_mask[vert_data > 0] = 1  # Binary mask
            
            # Compute Dice
            gt_binary = gt_mask > 0
            ts_binary = ts_mask > 0
            intersection = np.logical_and(gt_binary, ts_binary).sum()
            dice = 2.0 * intersection / (gt_binary.sum() + ts_binary.sum() + 1e-6)
            
            return dice
            
        except Exception as e:
            print(f"  ⚠ Error running TotalSegmentator: {e}")
            return None
    
    def _plot_training_progress(self):
        """Plot training curves."""
        
        if len(self.validation_history) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        timesteps = [v['timestep'] for v in self.validation_history]
        rewards = [v['avg_reward'] for v in self.validation_history]
        dice_scores = [v['dice_score'] for v in self.validation_history if v['dice_score'] is not None]
        dice_timesteps = [v['timestep'] for v in self.validation_history if v['dice_score'] is not None]
        displacements = [v['max_displacement_mm'] for v in self.validation_history]
        
        # Reward
        axes[0].plot(timesteps, rewards, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Timesteps', fontsize=12)
        axes[0].set_ylabel('Avg Reward', fontsize=12)
        axes[0].set_title('Reward Progress', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Dice score
        if len(dice_scores) > 0:
            axes[1].plot(dice_timesteps, dice_scores, 'o-', color='red', linewidth=2, markersize=8)
            axes[1].axhline(y=1.0, color='green', linestyle='--', label='Perfect (GT)')
            axes[1].axhline(y=0.82, color='orange', linestyle='--', label='Baseline')
            axes[1].set_xlabel('Timesteps', fontsize=12)
            axes[1].set_ylabel('Dice Score', fontsize=12)
            axes[1].set_title('TS Performance (Lower = More Challenging)', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0.5, 1.05])
        
        # Displacement
        axes[2].plot(timesteps, displacements, 'o-', color='purple', linewidth=2, markersize=8)
        axes[2].set_xlabel('Timesteps', fontsize=12)
        axes[2].set_ylabel('Max Displacement (mm)', fontsize=12)
        axes[2].set_title('Fragment Displacement', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "training_progress.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

