"""
Physical Adversary Environment for MuJoCo

RL agent applies physical forces to vertebrae to create realistic abnormal configurations.

Action Space:
- vertebra_id: which vertebra to affect [0..22]
- action_type: ['displace', 'rotate', 'compress', 'fracture']
- magnitude: force strength [0..1]
- direction: force direction (x, y, z) normalized

Observation Space:
- For each vertebra: position (3), orientation (4 quat), velocity (3), angular_vel (3)
- Total: 23 vertebrae × 13 = 299 dims

Reward:
- Assembly loss after TS segmentation of rendered abnormal CT
- Plausibility penalty (unrealistic deformations)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import nibabel as nib
from pathlib import Path
import tempfile
import subprocess


class PhysicalAdversaryEnv(gym.Env):
    """MuJoCo-based physical adversary environment."""
    
    def __init__(
        self,
        xml_path: str,
        gt_mask_path: str,
        totalseg_cmd: str = "TotalSegmentator",
        max_steps: int = 10,
        force_scale: float = 100.0,  # Max force in Newtons
    ):
        super().__init__()
        
        self.xml_path = xml_path
        self.gt_mask_path = gt_mask_path
        self.totalseg_cmd = totalseg_cmd
        self.max_steps = max_steps
        self.force_scale = force_scale
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get vertebra body IDs
        self.vertebra_names = []
        self.vertebra_ids = []
        for i in range(1, self.model.nbody):  # Skip world (0)
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name != 'world':
                self.vertebra_names.append(name)
                self.vertebra_ids.append(i)
        
        self.num_vertebrae = len(self.vertebra_ids)
        print(f"Loaded {self.num_vertebrae} vertebrae: {self.vertebra_names}")
        
        # Action space: [vertebra_id, force_x, force_y, force_z, torque_x, torque_y, torque_z]
        # Discrete vertebra selection + continuous forces
        self.action_space = spaces.Box(
            low=np.array([0, -1, -1, -1, -1, -1, -1]),
            high=np.array([self.num_vertebrae-1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Observation space: vertebra states
        # For each vertebra: pos(3) + quat(4) + vel(3) + angvel(3) = 13
        obs_dim = self.num_vertebrae * 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.step_count = 0
        self.initial_state = None
    
    def reset(self):
        """Reset to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Save initial state
        self.initial_state = {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
        }
        
        self.step_count = 0
        return self._get_obs()
    
    def step(self, action):
        """Apply physical action."""
        # Parse action
        vertebra_idx = int(np.clip(action[0], 0, self.num_vertebrae-1))
        force = action[1:4] * self.force_scale
        torque = action[4:7] * self.force_scale * 0.1  # Smaller torques
        
        body_id = self.vertebra_ids[vertebra_idx]
        
        # Apply force and torque for one physics step
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = torque
        
        # Simulate for 50 steps (0.1 sec)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Check termination
        done = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        reward = 0.0  # Placeholder, will compute after CT rendering
        info = {
            'vertebra_affected': self.vertebra_names[vertebra_idx],
            'force': force,
            'torque': torque,
        }
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Get observation: all vertebra states."""
        obs = []
        for body_id in self.vertebra_ids:
            # Position
            pos = self.data.xpos[body_id]
            # Orientation (quaternion from rotation matrix)
            rot_mat = self.data.xmat[body_id].reshape(3, 3)
            # Velocity
            # (Note: need to map body to joint DOF - simplified here)
            vel = np.zeros(3)  # Placeholder
            angvel = np.zeros(3)  # Placeholder
            quat = np.array([1, 0, 0, 0])  # Placeholder
            
            obs.extend(pos)
            obs.extend(quat)
            obs.extend(vel)
            obs.extend(angvel)
        
        return np.array(obs, dtype=np.float32)
    
    def render_to_ct(self):
        """Render current vertebrae configuration to CT volume."""
        # TODO: Implement voxelization
        # For now, return None
        raise NotImplementedError("CT rendering not yet implemented")
    
    def run_totalseg(self, ct_path):
        """Run TotalSegmentator on CT."""
        # TODO: Implement TS execution
        raise NotImplementedError("TotalSegmentator integration not yet implemented")
    
    def compute_reward(self):
        """Compute reward after rendering CT and running TS."""
        # TODO: Implement reward computation
        raise NotImplementedError("Reward computation not yet implemented")


if __name__ == "__main__":
    # Test the environment
    xml_path = "outputs/mujoco_per_vertebra/sub-verse563/spine_per_vertebra.xml"
    gt_mask_path = "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz"
    
    env = PhysicalAdversaryEnv(xml_path, gt_mask_path, max_steps=5)
    
    print("\n" + "="*60)
    print("Testing Physical Adversary Environment")
    print("="*60)
    
    obs = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Apply random actions
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {step+1}:")
        print(f"  Action: vertebra={int(action[0])}, force={action[1:4]}")
        print(f"  Affected: {info['vertebra_affected']}")
        print(f"  Force applied: {info['force']}")
        print(f"  Done: {done}")
    
    print("\n✅ P3-1b COMPLETE: Action space defined and tested")
    print("   Next: P3-1c - Refine actions (fracture mechanism)")

