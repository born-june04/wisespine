#!/usr/bin/env python3
"""
PyBullet-based Physical Fracture Environment for RL.

This environment:
1. Loads L1 vertebra fragments in PyBullet
2. Allows RL agent to apply forces
3. Simulates physics
4. Renders fractured state to CT
5. Runs TotalSegmentator
6. Returns reward based on segmentation failure
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import glob
import nibabel as nib
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CT rendering will be integrated later
# from modules.ct_renderer_warping import render_deformed_ct


class PyBulletFractureEnv(gym.Env):
    """
    Gymnasium environment for PyBullet-based vertebra fracture simulation.
    
    State: Fragment positions and orientations (5 fragments × 7D = 35D)
    Action: Forces/torques on each fragment (5 fragments × 6D = 30D)
    Reward: -Dice(TS_pred, GT) with plausibility penalty
    """
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(
        self,
        subject: str = "sub-verse563",
        fragment_dir: str = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured",
        gt_ct_path: str = "VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz",
        gt_mask_path: str = "VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz",
        max_steps: int = 100,
        physics_steps_per_action: int = 10,
        max_force: float = 0.0001,  # Tuned force!
        max_torque: float = 0.00001,  # Tuned torque!
        gui: bool = False,
    ):
        """
        Initialize PyBullet fracture environment.
        
        Args:
            subject: Subject ID
            fragment_dir: Directory containing fragment OBJ files
            gt_ct_path: Path to ground truth CT
            gt_mask_path: Path to ground truth mask
            max_steps: Maximum episode steps
            physics_steps_per_action: Physics steps per RL action
            max_force: Maximum force magnitude
            max_torque: Maximum torque magnitude
            gui: Use GUI visualization
        """
        super().__init__()
        
        self.subject = subject
        self.fragment_dir = Path(fragment_dir)
        self.gt_ct_path = Path(gt_ct_path)
        self.gt_mask_path = Path(gt_mask_path)
        self.max_steps = max_steps
        self.physics_steps_per_action = physics_steps_per_action
        self.max_force = max_force
        self.max_torque = max_torque
        self.gui = gui
        
        # Load GT data
        print(f"Loading GT data for {subject}...")
        self.gt_ct_nii = nib.load(str(self.gt_ct_path))
        self.gt_ct = self.gt_ct_nii.get_fdata()
        self.gt_mask_nii = nib.load(str(self.gt_mask_path))
        self.gt_mask = self.gt_mask_nii.get_fdata()
        
        # Find L1 label
        self.l1_label = 20  # VerSe L1 label
        self.l1_mask = (self.gt_mask == self.l1_label)
        
        if not self.l1_mask.any():
            raise ValueError(f"L1 (label {self.l1_label}) not found in mask!")
        
        # Get L1 centroid in voxel space
        l1_coords = np.where(self.l1_mask)
        self.l1_centroid_voxel = np.array([
            np.mean(l1_coords[0]),
            np.mean(l1_coords[1]),
            np.mean(l1_coords[2])
        ])
        
        print(f"  L1 centroid (voxel): {self.l1_centroid_voxel}")
        
        # Load fragment files
        self.fragment_files = sorted(glob.glob(str(self.fragment_dir / "L1_frag_*.obj")))
        self.num_fragments = len(self.fragment_files)
        
        if self.num_fragments == 0:
            raise ValueError(f"No fragments found in {self.fragment_dir}")
        
        print(f"  Found {self.num_fragments} fragments")
        
        # Define spaces
        # State: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w] × num_fragments
        state_dim = self.num_fragments * 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action: [force_x, force_y, force_z, torque_x, torque_y, torque_z] × num_fragments
        action_dim = self.num_fragments * 6
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Initialize PyBullet
        self.physics_client = None
        self.fragment_bodies = []
        self.step_count = 0
        
        # Episode tracking
        self.episode_count = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1
        
        # Disconnect previous PyBullet instance
        if self.physics_client is not None:
            p.disconnect()
        
        # Connect PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)  # NO GRAVITY - we want controlled fracture!
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Load fragments (no constraints - free bodies)
        self.fragment_bodies = []
        
        for i, frag_file in enumerate(self.fragment_files):
            collision_shape = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=frag_file,
                meshScale=[0.001, 0.001, 0.001]
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_MESH,
                fileName=frag_file,
                meshScale=[0.001, 0.001, 0.001],
                rgbaColor=[0.8, 0.7, 0.6, 1.0]
            )
            
            # Stack fragments vertically with small spacing
            start_pos = [0, 0, 0.3 + i * 0.015]
            
            body_id = p.createMultiBody(
                baseMass=0.001,  # Tuned mass
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=start_pos
            )
            
            # Set damping for realistic movement
            p.changeDynamics(
                body_id,
                -1,
                linearDamping=0.5,
                angularDamping=0.5
            )
            
            self.fragment_bodies.append(body_id)
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def _get_state(self):
        """Get current state (positions + orientations of all fragments)."""
        state = []
        
        for body_id in self.fragment_bodies:
            pos, quat = p.getBasePositionAndOrientation(body_id)
            state.extend(pos)
            state.extend(quat)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Execute one RL step.
        
        Args:
            action: Array of shape (num_fragments * 6,) with forces/torques
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action to force/torque range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply forces and torques to each fragment
        for i in range(self.num_fragments):
            force = action[i*6:i*6+3] * self.max_force
            torque = action[i*6+3:i*6+6] * self.max_torque
            
            p.applyExternalForce(
                self.fragment_bodies[i],
                -1,
                force.tolist(),
                [0, 0, 0],
                p.LINK_FRAME
            )
            
            p.applyExternalTorque(
                self.fragment_bodies[i],
                -1,
                torque.tolist(),
                p.LINK_FRAME
            )
        
        # Step physics
        for _ in range(self.physics_steps_per_action):
            p.stepSimulation()
        
        self.step_count += 1
        
        # Get new state
        state = self._get_state()
        
        # Compute reward (placeholder - will integrate TS later)
        reward = self._compute_reward(state)
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        info = {}
        
        if truncated or terminated:
            # Add episode info for logging
            info['episode'] = {
                'r': sum([self._compute_reward(s) for s in [state]]),  # Total reward approximation
                'l': self.step_count
            }
        
        return state, reward, terminated, truncated, info
    
    def _compute_reward(self, state):
        """
        Compute reward based on current state.
        
        Reward = -Dice(TS_pred, GT) + plausibility_penalty
        
        We want TS to FAIL (low Dice) while keeping fracture plausible.
        
        For now, use a simple proxy: encourage fragments to spread
        (will integrate TS later for full pipeline).
        """
        # Get fragment positions
        positions = state.reshape(self.num_fragments, 7)[:, :3]
        
        # Calculate pairwise distances
        total_dist = 0.0
        count = 0
        
        for i in range(self.num_fragments):
            for j in range(i+1, self.num_fragments):
                dist = np.linalg.norm(positions[i] - positions[j])
                total_dist += dist
                count += 1
        
        avg_dist = total_dist / max(count, 1)
        
        # Target distance: want fragments to be ~0.01-0.02 units apart
        # (corresponds to ~10-20mm in physical space)
        target_dist = 0.015
        distance_error = abs(avg_dist - target_dist)
        
        # Reward for being close to target distance
        distance_reward = -distance_error * 100
        
        # Penalty for fragments going too far
        max_dist_from_origin = np.max(np.linalg.norm(positions, axis=1))
        if max_dist_from_origin > 0.5:
            out_of_bounds_penalty = -(max_dist_from_origin - 0.5) * 1000
        else:
            out_of_bounds_penalty = 0
        
        reward = distance_reward + out_of_bounds_penalty
        
        return reward
    
    def render(self):
        """Render current state (for GUI mode)."""
        if self.gui:
            # GUI mode automatically renders
            pass
        else:
            # Could implement rgb_array capture here
            pass
        
        return None
    
    def close(self):
        """Clean up PyBullet."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
    
    def get_fragment_displacements(self):
        """
        Get current fragment displacements relative to initial positions.
        
        Returns:
            List of (fragment_idx, displacement_3d) tuples
        """
        displacements = []
        
        for i, body_id in enumerate(self.fragment_bodies):
            pos, _ = p.getBasePositionAndOrientation(body_id)
            
            # Initial position was [0, 0, 0.3 + i * 0.015]
            initial_pos = np.array([0, 0, 0.3 + i * 0.015])
            current_pos = np.array(pos)
            
            displacement = current_pos - initial_pos
            displacements.append((i, displacement))
        
        return displacements


if __name__ == "__main__":
    """Test the environment."""
    
    print("="*70)
    print("Testing PyBullet Fracture Environment")
    print("="*70)
    
    # Create environment
    env = PyBulletFractureEnv(
        subject="sub-verse563",
        max_steps=50,
        gui=False  # No GUI on headless server
    )
    
    print("\n✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    # Test reset
    state, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Initial state shape: {state.shape}")
    print(f"  Initial state (first 7): {state[:7]}")
    
    # Test random actions
    print("\n--- Testing random actions ---")
    
    for step in range(10):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Step {step+1}: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            break
    
    # Get final displacements
    displacements = env.get_fragment_displacements()
    print(f"\n--- Final displacements ---")
    for i, disp in displacements:
        print(f"  Fragment {i}: {disp}")
    
    env.close()
    
    print("\n" + "="*70)
    print("✓ Environment test complete!")
    print("="*70)

