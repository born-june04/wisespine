#!/usr/bin/env python3
"""
Simplified training without validation to avoid disk quota issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
from pathlib import Path
from datetime import datetime

from modules.pybullet_fracture_env import PyBulletFractureEnv


class SimpleLoggingCallback(BaseCallback):
    """Simple callback that only logs to console (no file writes)."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        
    def _on_step(self):
        # Only print to console, no disk writes
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'episode' in info:
                    ep_info = info['episode']
                    if isinstance(ep_info, dict):
                        self.episode_count += 1
                        ep_reward = float(ep_info['r'])
                        self.episode_rewards.append(ep_reward)
                        
                        if self.verbose > 0 and self.episode_count % 10 == 0:
                            recent_avg = np.mean(self.episode_rewards[-10:])
                            print(f"Episodes: {self.episode_count}, Recent Avg Reward: {recent_avg:.3f}")
        
        return True


def main():
    print("="*70)
    print("PyBullet Fracture RL Training (No Validation)")
    print("="*70)
    
    output_dir = Path("/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/rl_training/simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Total timesteps: 50000")
    print(f"Note: No validation or file logging to save disk space")
    
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
    callback = SimpleLoggingCallback(verbose=1)
    
    # Train
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    model.learn(
        total_timesteps=50000,
        callback=callback,
        progress_bar=True
    )
    
    # Save final model (small file)
    final_model_path = output_dir / "fracture_agent_final.zip"
    model.save(str(final_model_path))
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("="*70)
    print(f"\nModel saved: {final_model_path}")
    print(f"Total episodes: {callback.episode_count}")
    
    if len(callback.episode_rewards) > 0:
        print(f"Final avg reward (last 100): {np.mean(callback.episode_rewards[-100:]):.3f}")
    
    print("="*70)


if __name__ == "__main__":
    main()

