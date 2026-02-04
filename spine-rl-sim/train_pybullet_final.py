#!/usr/bin/env python3
"""
Training with lightweight validation (only PNG visualizations, no CT files).
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
from pathlib import Path
from datetime import datetime

from modules.pybullet_fracture_env import PyBulletFractureEnv
from modules.pybullet_ct_renderer import render_pybullet_to_ct


class LightweightValidationCallback(BaseCallback):
    """
    Lightweight validation: only PNG visualizations, no CT file saves.
    """
    
    def __init__(self, eval_env, eval_freq: int, output_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validation_history = []
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"VALIDATION @ Step {self.num_timesteps}")
            print(f"{'='*70}")
        
        try:
            # Run 3 eval episodes
            rewards = []
            for ep in range(3):
                obs, _ = self.eval_env.reset()
                ep_reward = 0
                for _ in range(50):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    if terminated or truncated:
                        break
                rewards.append(ep_reward)
            
            avg_reward = np.mean(rewards)
            
            if self.verbose > 0:
                print(f"\n📊 Avg Reward: {avg_reward:.3f}")
            
            # Get displacements
            disps = self.eval_env.get_fragment_displacements()
            max_disp_mm = np.max([np.linalg.norm(d[1])*1000 for d in disps])
            
            # Render CT (in memory only)
            if self.verbose > 0:
                print(f"🎨 Creating visualization...")
            
            fractured_ct, fractured_mask = render_pybullet_to_ct(
                self.eval_env,
                output_ct_path=None,
                output_mask_path=None
            )
            
            # Create visualization
            step_dir = self.output_dir / f"step_{self.num_timesteps:06d}"
            step_dir.mkdir(exist_ok=True)
            
            self._create_visualization(fractured_ct, disps, step_dir)
            
            # Save to history
            val_entry = {
                'timestep': int(self.num_timesteps),
                'avg_reward': float(avg_reward),
                'max_displacement_mm': float(max_disp_mm)
            }
            self.validation_history.append(val_entry)
            
            # Plot progress
            self._plot_progress()
            
            if self.verbose > 0:
                print(f"✓ Validation complete! Saved to: {step_dir}")
                print(f"{'='*70}\n")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠ Validation failed: {e}")
        
        return True
    
    def _create_visualization(self, fractured_ct, displacements, output_dir):
        """Create lightweight visualization."""
        
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
        
        vmin, vmax = -200, 1500
        
        # 1x3 comparison (save space)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Step {self.num_timesteps}: Original vs Fractured (Sagittal)', fontsize=14)
        
        # Sagittal slices only
        orig_sag = original_ct[l1_center[0], 
                               max(0, l1_coords[1].min()-margin):min(gt_mask.shape[1], l1_coords[1].max()+margin),
                               max(0, l1_coords[2].min()-margin):min(gt_mask.shape[2], l1_coords[2].max()+margin)]
        
        frac_sag = fractured_ct[l1_center[0],
                                max(0, l1_coords[1].min()-margin):min(gt_mask.shape[1], l1_coords[1].max()+margin),
                                max(0, l1_coords[2].min()-margin):min(gt_mask.shape[2], l1_coords[2].max()+margin)]
        
        diff_sag = frac_sag - orig_sag
        
        axes[0].imshow(orig_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(frac_sag.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1].set_title('Fractured')
        axes[1].axis('off')
        
        axes[2].imshow(diff_sag.T, cmap='RdBu_r', vmin=-100, vmax=100, origin='lower')
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        # Add displacement info
        max_disp = np.max([np.linalg.norm(d[1])*1000 for d in displacements])
        fig.text(0.5, 0.02, f'Max Displacement: {max_disp:.2f}mm', ha='center', fontsize=11)
        
        plt.tight_layout()
        
        output_path = output_dir / "comparison.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_progress(self):
        """Plot training progress."""
        
        if len(self.validation_history) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        timesteps = [v['timestep'] for v in self.validation_history]
        rewards = [v['avg_reward'] for v in self.validation_history]
        displacements = [v['max_displacement_mm'] for v in self.validation_history]
        
        axes[0].plot(timesteps, rewards, 'o-', linewidth=2, markersize=6)
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Avg Reward')
        axes[0].set_title('Reward Progress')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(timesteps, displacements, 'o-', color='purple', linewidth=2, markersize=6)
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Max Displacement (mm)')
        axes[1].set_title('Fragment Displacement')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "training_progress.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


def main():
    print("="*70)
    print("PyBullet Fracture RL Training with Validation")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(f"/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/rl_training/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Total timesteps: 50000")
    print(f"Validation frequency: 10000")
    print(f"Note: Only PNG saved (no CT files)")
    
    # Create environment
    print("\nCreating environment...")
    
    def make_env():
        return PyBulletFractureEnv(
            subject="sub-verse563",
            max_steps=50,
            physics_steps_per_action=10,
            max_force=0.0002,
            max_torque=0.00002,
            gui=False
        )
    
    env = DummyVecEnv([make_env])
    eval_env = make_env()
    
    print("✓ Environment created")
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None,
        device='cpu'
    )
    
    print("✓ PPO agent created")
    
    # Create callback
    callback = LightweightValidationCallback(
        eval_env=eval_env,
        eval_freq=10000,
        output_dir=str(output_dir / "validation"),
        verbose=1
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    model.learn(
        total_timesteps=50000,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    final_model_path = output_dir / "fracture_agent_final.zip"
    model.save(str(final_model_path))
    
    eval_env.close()
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("="*70)
    print(f"\nModel saved: {final_model_path}")
    print(f"Validations: {len(callback.validation_history)}")
    print("="*70)


if __name__ == "__main__":
    main()

