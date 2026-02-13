from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from spine_rl import SpineFixConfig, SpineFixEnv
from spine_rl.utils.eval import evaluate_policy


class TrainMetricsCallback(BaseCallback):
    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_success = []
        self._episode_pos_err = []
        self._episode_rot_err = []
        self._current_rewards = None
        self._current_lengths = None

    def _init_callback(self) -> None:
        self._current_rewards = np.zeros(self.training_env.num_envs, dtype=np.float32)
        self._current_lengths = np.zeros(self.training_env.num_envs, dtype=np.int32)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timesteps",
                        "episodes",
                        "success_rate",
                        "avg_return",
                        "avg_steps",
                        "avg_pos_err",
                        "avg_rot_err_rad",
                    ],
                )
                writer.writeheader()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if rewards is None or dones is None or infos is None:
            return True

        self._current_rewards += rewards
        self._current_lengths += 1

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] or {}
            self._episode_rewards.append(float(self._current_rewards[i]))
            self._episode_lengths.append(int(self._current_lengths[i]))
            self._episode_success.append(bool(info.get("success", False)))
            self._episode_pos_err.append(float(info.get("pos_err_mean", np.nan)))
            self._episode_rot_err.append(float(info.get("rot_err_mean_rad", np.nan)))
            self._current_rewards[i] = 0.0
            self._current_lengths[i] = 0

        return True

    def dump(self) -> None:
        if not self._episode_rewards:
            return
        episodes = len(self._episode_rewards)
        row = {
            "timesteps": int(self.num_timesteps),
            "episodes": episodes,
            "success_rate": float(np.mean(self._episode_success)),
            "avg_return": float(np.mean(self._episode_rewards)),
            "avg_steps": float(np.mean(self._episode_lengths)),
            "avg_pos_err": float(np.mean(self._episode_pos_err)),
            "avg_rot_err_rad": float(np.mean(self._episode_rot_err)),
        }
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        self._episode_rewards.clear()
        self._episode_lengths.clear()
        self._episode_success.clear()
        self._episode_pos_err.clear()
        self._episode_rot_err.clear()


def make_env(model_path: Path, num_fractures: int) -> SpineFixEnv:
    config = SpineFixConfig(model_path=model_path, num_fractures=num_fractures)
    return SpineFixEnv(config=config, render_mode=None)


def write_eval_csv(path: Path, timesteps: int, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timesteps",
                "success_rate",
                "avg_return",
                "avg_steps",
                "avg_pos_err",
                "avg_rot_err_rad",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timesteps": timesteps,
                "success_rate": result.success_rate,
                "avg_return": result.avg_return,
                "avg_steps": result.avg_steps,
                "avg_pos_err": result.avg_pos_err,
                "avg_rot_err_rad": result.avg_rot_err_rad,
            }
        )


def main() -> None:
    # Default to repo-local assets/ so this runs on any machine.
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "assets" / "spine_model.xml"
    num_fractures = 3

    env = DummyVecEnv([lambda: make_env(model_path, num_fractures)])
    eval_env = make_env(model_path, num_fractures)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
        learning_rate=3e-4,
    )

    total_timesteps = 200_000
    eval_every = 20_000

    train_csv = repo_root / "metrics" / "train_metrics.csv"
    eval_csv = repo_root / "metrics" / "eval_metrics.csv"
    train_cb = TrainMetricsCallback(train_csv)
    train_cb.init_callback(model)

    total_iters = total_timesteps // eval_every
    pbar = tqdm(total=total_iters, desc="Training", unit="eval")
    for _ in range(total_iters):
        model.learn(total_timesteps=eval_every, reset_num_timesteps=False, callback=train_cb)
        train_cb.dump()
        result = evaluate_policy(eval_env, model, episodes=20)
        write_eval_csv(eval_csv, int(model.num_timesteps), result)
        pbar.set_postfix(
            {
                "success": f"{result.success_rate:.2f}",
                "return": f"{result.avg_return:.2f}",
                "pos_err": f"{result.avg_pos_err:.4f}",
                "rot_err": f"{result.avg_rot_err_rad:.4f}",
            }
        )
        pbar.update(1)
    pbar.close()

    out_path = repo_root / "models" / "ppo_spine_fix"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
