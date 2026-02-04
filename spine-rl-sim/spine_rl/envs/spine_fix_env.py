from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np


@dataclass
class SpineFixConfig:
    model_path: Path
    max_steps: int = 200
    num_fractures: int = 1
    fracture_translation: float = 0.02  # meters
    fracture_rotation_deg: float = 10.0
    action_translation: float = 0.002  # meters per step
    action_rotation_deg: float = 2.0
    pos_threshold: float = 0.002
    rot_threshold_deg: float = 2.0
    pos_weight: float = 1.0
    rot_weight: float = 0.5
    action_weight: float = 0.01
    smoothness_weight: float = 0.01
    adjacency_weight: float = 0.05
    max_pos_radius: float = 0.05
    max_rot_deg: float = 25.0
    safety_penalty: float = 1.0


class SpineFixEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config: SpineFixConfig, render_mode: Optional[str] = None):
        self.config = config
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(config.model_path))
        self.data = mujoco.MjData(self.model)

        self.body_names = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name and name.startswith("sub-verse563_"):
                self.body_names.append(name)
        if not self.body_names:
            raise RuntimeError("No vertebra bodies found; did you generate the model XML?")
        if config.num_fractures < 1:
            raise ValueError("num_fractures must be >= 1")
        if config.num_fractures > len(self.body_names):
            raise ValueError("num_fractures exceeds available vertebra bodies")

        self._ordered_bodies = self._order_bodies(self.body_names)
        self._adj_pairs = list(zip(self._ordered_bodies[:-1], self._ordered_bodies[1:]))

        # Action: 6D per fractured body (translation + rotation deltas).
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6 * config.num_fractures,), dtype=np.float32
        )
        # Observation: per-fracture current pose + target pose (pos(3), quat(4) each).
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14 * config.num_fractures,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._fractured_bodies: list[str] = []
        self._target_pos = np.zeros((config.num_fractures, 3), dtype=np.float32)
        self._target_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (config.num_fractures, 1)
        )
        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        self._viewer = None
        self._viewer_available = hasattr(mujoco, "viewer")

    @staticmethod
    def _parse_vertebra(name: str) -> Tuple[int, int]:
        parts = name.split("_")
        if len(parts) < 2:
            return (99, 999)
        region_num = parts[1]
        region = region_num[0] if region_num else "?"
        num = int(region_num[1:]) if region_num[1:].isdigit() else 999
        order = {"C": 0, "T": 1, "L": 2, "S": 3}.get(region, 99)
        return (order, num)

    def _order_bodies(self, names: list[str]) -> list[str]:
        return sorted(names, key=self._parse_vertebra)

    def _body_qpos_adr(self, body_name: str) -> int:
        joint_name = f"{body_name}_free"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {joint_name}")
        return self.model.jnt_qposadr[jid]

    def _get_pose(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        adr = self._body_qpos_adr(body_name)
        pos = self.data.qpos[adr : adr + 3].copy()
        quat = self.data.qpos[adr + 3 : adr + 7].copy()
        return pos, quat

    def _set_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        adr = self._body_qpos_adr(body_name)
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float32)

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def _quat_normalize(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        if n == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q / n).astype(np.float32)

    @staticmethod
    def _angle_from_quat(q: np.ndarray) -> float:
        qn = SpineFixEnv._quat_normalize(q)
        angle = 2.0 * np.arccos(np.clip(qn[0], -1.0, 1.0))
        return float(angle)

    def _random_quat(self, max_deg: float) -> np.ndarray:
        axis = self._rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(self._rng.uniform(-max_deg, max_deg))
        w = np.cos(angle / 2.0)
        xyz = axis * np.sin(angle / 2.0)
        return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)

    def _delta_quat_from_action(self, action_rot: np.ndarray) -> np.ndarray:
        # action_rot is in [-1, 1], scale to degrees per step
        angles = np.deg2rad(action_rot * self.config.action_rotation_deg)
        cx, cy, cz = np.cos(angles / 2.0)
        sx, sy, sz = np.sin(angles / 2.0)
        # ZYX order
        w = cx * cy * cz + sx * sy * sz
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz
        return self._quat_normalize(np.array([w, x, y, z], dtype=np.float32))

    def _get_obs(self) -> np.ndarray:
        obs_parts = []
        for i, body in enumerate(self._fractured_bodies):
            pos, quat = self._get_pose(body)
            obs_parts.append(
                np.concatenate([pos, quat, self._target_pos[i], self._target_quat[i]])
            )
        return np.concatenate(obs_parts).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        self._fractured_bodies = [
            str(x)
            for x in self._rng.choice(
                self.body_names, size=self.config.num_fractures, replace=False
            )
        ]
        for body in self._fractured_bodies:
            pos = self._rng.uniform(
                -self.config.fracture_translation, self.config.fracture_translation, size=3
            )
            quat = self._random_quat(self.config.fracture_rotation_deg)
            self._set_pose(body, pos, quat)

        self._prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {"fractured_bodies": list(self._fractured_bodies)}

    def step(self, action: np.ndarray):
        self._step_count += 1

        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        per_body_pos_err = []
        per_body_rot_err = []
        reward = 0.0
        safety_violated = False

        for i, body in enumerate(self._fractured_bodies):
            sub = action[i * 6 : (i + 1) * 6]
            pos, quat = self._get_pose(body)

            pos_delta = sub[:3] * self.config.action_translation
            rot_delta = self._delta_quat_from_action(sub[3:])

            new_pos = pos + pos_delta
            new_quat = self._quat_mul(rot_delta, quat)
            new_quat = self._quat_normalize(new_quat)

            self._set_pose(body, new_pos, new_quat)

            pos_err = float(np.linalg.norm(new_pos - self._target_pos[i]))
            rot_err = self._angle_from_quat(
                self._quat_mul(self._quat_conj(self._target_quat[i]), new_quat)
            )
            per_body_pos_err.append(pos_err)
            per_body_rot_err.append(rot_err)

            reward += (
                -self.config.pos_weight * pos_err
                -self.config.rot_weight * rot_err
                -self.config.action_weight * float(np.linalg.norm(sub))
            )

            if pos_err > self.config.max_pos_radius:
                safety_violated = True
            if np.rad2deg(rot_err) > self.config.max_rot_deg:
                safety_violated = True

        # Penalize action jerk for smoother control
        smoothness = float(np.linalg.norm(action - self._prev_action))
        reward -= self.config.smoothness_weight * smoothness
        self._prev_action = action.copy()

        # Adjacency penalty: discourage large separation between neighboring vertebrae
        adj_penalty = 0.0
        for a, b in self._adj_pairs:
            pos_a, _ = self._get_pose(a)
            pos_b, _ = self._get_pose(b)
            adj_penalty += float(np.linalg.norm(pos_a - pos_b))
        reward -= self.config.adjacency_weight * adj_penalty

        mujoco.mj_forward(self.model, self.data)

        pos_err_mean = float(np.mean(per_body_pos_err))
        rot_err_mean = float(np.mean(per_body_rot_err))

        terminated = (pos_err_mean < self.config.pos_threshold) and (
            np.rad2deg(rot_err_mean) < self.config.rot_threshold_deg
        )
        if safety_violated:
            reward -= self.config.safety_penalty
            terminated = True

        truncated = self._step_count >= self.config.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "fractured_bodies": list(self._fractured_bodies),
            "pos_err_mean": pos_err_mean,
            "rot_err_mean_rad": rot_err_mean,
            "success": terminated and not safety_violated,
            "safety_violated": safety_violated,
            "adj_penalty": adj_penalty,
            "smoothness": smoothness,
        }

    def render(self):
        if self.render_mode != "human":
            return
        if not self._viewer_available:
            # MuJoCo viewer module not available in this install.
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
