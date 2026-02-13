from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EvalResult:
    success_rate: float
    avg_return: float
    avg_steps: float
    avg_pos_err: float
    avg_rot_err_rad: float


def evaluate_policy(env, model, episodes: int = 20) -> EvalResult:
    successes = 0
    returns = []
    steps = []
    pos_errs = []
    rot_errs = []

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        last_info: Dict = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            last_info = info
            done = terminated or truncated

        returns.append(total_reward)
        steps.append(step_count)
        pos_errs.append(float(last_info.get("pos_err_mean", np.nan)))
        rot_errs.append(float(last_info.get("rot_err_mean_rad", np.nan)))
        if last_info.get("success", False):
            successes += 1

    return EvalResult(
        success_rate=successes / max(1, episodes),
        avg_return=float(np.mean(returns)),
        avg_steps=float(np.mean(steps)),
        avg_pos_err=float(np.mean(pos_errs)),
        avg_rot_err_rad=float(np.mean(rot_errs)),
    )
