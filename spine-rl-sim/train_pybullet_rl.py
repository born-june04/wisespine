#!/usr/bin/env python3
"""
Train RL agent to create realistic fractures using PyBullet.

The agent learns to apply forces that:
1. Create visible fractures (fragments separate)
2. Stay plausible (not too extreme)
3. Degrade TotalSegmentator performance
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
from modules.validation_callback import ValidationCallback


class FractureLoggingCallback(BaseCallback):
    """Log training progress."""
    
    def __init__(self, log_file, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        # Log episode stats
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'episode' in info:
                    ep_info = info['episode']
                    if isinstance(ep_info, dict):
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        self.episode_rewards.append(float(ep_reward))
                        self.episode_lengths.append(int(ep_length))
                        
                        # Log to file
                        log_entry = {
                            'timestep': int(self.num_timesteps),
                            'episode_reward': float(ep_reward),
                            'episode_length': int(ep_length),
                        }
                        
                        with open(self.log_file, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                        
                        if self.verbose > 0:
                            print(f"  Episode: reward={ep_reward:.3f}, length={ep_length}")
        
        return True


def train_fracture_agent(
    output_dir: str,
    total_timesteps: int = 50000,
    save_freq: int = 10000,
    eval_freq: int = 10000,  # Validation every 10k steps (was 5k)
    verbose: int = 1
):
    """
    Train PPO agent to create fractures.
    
    Args:
        output_dir: Directory to save models and logs
        total_timesteps: Total training steps
        save_freq: Save model every N steps
        verbose: Verbosity level
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / "training_log.jsonl"
    
    print("="*70)
    print("PyBullet Fracture RL Training")
    print("="*70)
    print(f"\nOutput directory: {output_path}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Save frequency: {save_freq}")
    print(f"Validation frequency: {eval_freq}")
    
    # Create environment
    print("\nCreating environment...")
    
    def make_env():
        return PyBulletFractureEnv(
            subject="sub-verse563",
            max_steps=50,  # 50 steps per episode
            physics_steps_per_action=10,
            max_force=0.0002,  # Slightly higher than tuned (allow agent to explore)
            max_torque=0.00002,
            gui=False
        )
    
    env = DummyVecEnv([make_env])
    
    # Create separate eval environment (not vectorized)
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
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard
        device='cpu'
    )
    
    print("✓ PPO agent created")
    
    # Save config
    config = {
        'subject': 'sub-verse563',
        'total_timesteps': total_timesteps,
        'max_steps_per_episode': 50,
        'max_force': 0.0002,
        'max_torque': 0.00002,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'eval_freq': eval_freq,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Config saved")
    
    # Create callbacks
    logging_callback = FractureLoggingCallback(log_file, verbose=verbose)
    
    validation_callback = ValidationCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        output_dir=str(output_path / "validation"),
        n_eval_episodes=3,
        run_ts=True,
        verbose=verbose
    )
    
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([logging_callback, validation_callback])
    
    # Train
    print("\n" + "="*70)
    print("Starting training with validation...")
    print("="*70)
    print(f"  Validation will run every {eval_freq} steps")
    print(f"  Results will be saved to: {output_path / 'validation'}")
    print("="*70)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Close eval environment
    eval_env.close()
    
    # Save final model
    final_model_path = output_path / "fracture_agent_final.zip"
    model.save(str(final_model_path))
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print("="*70)
    print(f"\nModel saved: {final_model_path}")
    print(f"Log file: {log_file}")
    
    # Summary statistics
    if len(callback.episode_rewards) > 0:
        print(f"\n📊 Training Summary:")
        print(f"  Episodes completed: {len(callback.episode_rewards)}")
        print(f"  Avg reward (last 10): {np.mean(callback.episode_rewards[-10:]):.3f}")
        print(f"  Avg episode length: {np.mean(callback.episode_lengths):.1f}")
    
    print("="*70)
    
    return model, output_path


def main():
    """Main training script."""
    
    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"outputs/phase3_physics_fracture/rl_training/{timestamp}"
    
    # Train
    model, output_path = train_fracture_agent(
        output_dir=output_dir,
        total_timesteps=50000,  # Start with 50k steps
        save_freq=10000,
        verbose=1
    )
    
    print(f"\n🎉 Training complete! Model saved to: {output_path}")


if __name__ == "__main__":
    main()

