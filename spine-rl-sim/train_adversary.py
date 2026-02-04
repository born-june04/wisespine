#!/usr/bin/env python3
"""
Phase 2: Adversarial Training Loop.

This script implements the alternating adversarial training loop:
1. Train adversary (RL) to generate hard corruptions
2. Train assembly (future) to be robust to those corruptions

For now (Phase 2-A):
  - Assembly is fixed (simple marching cubes)
  - Focus on training adversary with TS-like prior

Usage:
  ssh g3099 "bash -lc 'cd /gscratch/scrubbed/june0604/vindr && conda activate py311 && python spine-rl-sim/train_adversary.py --subject sub-verse563 --num_epochs 10'"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    raise RuntimeError("stable-baselines3 required: pip install stable-baselines3")

# Add spine-rl-sim to path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SPINE_RL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_RL_ROOT))

from modules.adversary_env import MaskCorruptionEnv, CorruptionBudget
from modules.assembly_wrapper import SimpleAssemblyModule


def load_masks(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """Load clean TS mask and GT mask for a subject."""
    # GT mask
    gt_path = REPO_ROOT / "VerSe" / "dataset-03test" / "derivatives" / subject / f"{subject}_dir-iso_seg-vert_msk.nii.gz"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")
    
    import nibabel as nib
    gt_img = nib.load(str(gt_path))
    gt_mask = gt_img.get_fdata().astype(np.uint16)
    
    # TS mask (clean baseline)
    ts_dir = REPO_ROOT / "totalseg_eval" / "predictions_total" / subject
    if not ts_dir.exists():
        raise FileNotFoundError(f"TS predictions not found: {ts_dir}")
    
    # Merge individual vertebrae files
    from ablation.run_phase0_baselines import merge_totalseg_dir
    ts_mask, _ = merge_totalseg_dir(ts_dir)
    
    return ts_mask, gt_mask


class MetricsCallback(BaseCallback):
    """Log metrics during adversary training."""
    
    def __init__(self, log_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episode_rewards = []
        self.episode_info = []
    
    def _on_step(self) -> bool:
        # Log episode metrics
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                if len(self.locals.get("infos", [])) > idx:
                    info = self.locals["infos"][idx]
                    self.episode_info.append(info)
                    
                    # Convert numpy types to Python native types for JSON serialization
                    def convert_to_native(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, (np.integer, np.floating)):
                            return obj.item()
                        elif isinstance(obj, dict):
                            return {k: convert_to_native(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_native(v) for v in obj]
                        return obj
                    
                    # Log to file
                    log_entry = {
                        "timestep": int(self.num_timesteps),
                        "episode": len(self.episode_rewards),
                        **convert_to_native(info),
                    }
                    
                    log_file = self.log_dir / "training_log.jsonl"
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
        
        return True


def train_adversary(
    subject: str,
    num_epochs: int = 10,
    steps_per_epoch: int = 1000,
    budget_voxels: int = 5000,
    budget_ops: int = 5,
    ts_prior_weight: float = 0.3,
    out_dir: Optional[Path] = None,
    seed: int = 42,
):
    """
    Train adversary RL agent to generate hard-but-realistic mask corruptions.
    
    Args:
        subject: Subject ID
        num_epochs: Number of training epochs
        steps_per_epoch: Training steps per epoch
        budget_voxels: Max voxel changes per episode
        budget_ops: Max operations per episode
        ts_prior_weight: Weight of TS-like prior penalty
        out_dir: Output directory
        seed: Random seed
    """
    if out_dir is None:
        out_dir = REPO_ROOT / "spine-rl-sim" / "adversary_outputs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Adversary Training ===")
    print(f"Subject: {subject}")
    print(f"Epochs: {num_epochs}, Steps/epoch: {steps_per_epoch}")
    print(f"Budget: {budget_voxels} voxels, {budget_ops} ops")
    print(f"TS prior weight: {ts_prior_weight}")
    print(f"Output: {out_dir}")
    print("")
    
    # Load masks
    print("Loading masks...")
    ts_mask, gt_mask = load_masks(subject)
    print(f"  TS mask: {ts_mask.shape}, labels: {len(set(np.unique(ts_mask).tolist()) - {0})}")
    print(f"  GT mask: {gt_mask.shape}, labels: {len(set(np.unique(gt_mask).tolist()) - {0})}")
    
    # Create assembly module (for now, simple marching cubes)
    print("Creating assembly module...")
    assembly = SimpleAssemblyModule(spacing=(1.0, 1.0, 1.0))
    
    # Create corruption budget
    budget = CorruptionBudget(
        max_voxel_changes=budget_voxels,
        max_operations=budget_ops,
        max_radius=2,
    )
    
    # Create environment
    print("Creating adversary environment...")
    def make_env():
        return MaskCorruptionEnv(
            clean_mask=ts_mask,
            gt_mask=gt_mask,
            budget=budget,
            max_steps=budget_ops,
            assembly_module=assembly,
            ts_prior_weight=ts_prior_weight,
            seed=seed,
        )
    
    env = DummyVecEnv([make_env])
    
    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        tensorboard_log=None,  # Disable tensorboard (not installed)
    )
    
    # Training callback
    callback = MetricsCallback(log_dir=out_dir / "logs")
    
    # Train
    print("\nStarting training...")
    total_steps = num_epochs * steps_per_epoch
    model.learn(
        total_timesteps=total_steps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save model
    model_path = out_dir / f"adversary_{subject}_final.zip"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save config
    config = {
        "subject": subject,
        "num_epochs": num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "budget_voxels": budget_voxels,
        "budget_ops": budget_ops,
        "ts_prior_weight": ts_prior_weight,
        "seed": seed,
        "ts_mask_shape": list(ts_mask.shape),
        "gt_mask_shape": list(gt_mask.shape),
    }
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Config saved to: {config_path}")
    
    print("\n✓ Adversary training complete!")
    return model, env


def main():
    ap = argparse.ArgumentParser(description="Train adversarial mask corruption agent")
    ap.add_argument("--subject", default="sub-verse563", help="Subject ID")
    ap.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    ap.add_argument("--steps_per_epoch", type=int, default=1000, help="Training steps per epoch")
    ap.add_argument("--budget_voxels", type=int, default=5000, help="Max voxel changes per episode")
    ap.add_argument("--budget_ops", type=int, default=5, help="Max operations per episode")
    ap.add_argument("--ts_prior_weight", type=float, default=0.3, help="TS-like prior penalty weight")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else None
    
    train_adversary(
        subject=args.subject,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        budget_voxels=args.budget_voxels,
        budget_ops=args.budget_ops,
        ts_prior_weight=args.ts_prior_weight,
        out_dir=out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

