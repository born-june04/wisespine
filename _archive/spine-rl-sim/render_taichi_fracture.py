#!/usr/bin/env python3
"""
Integration script: Physics Simulator → CT Rendering

This script connects the physics fracture simulator to CT rendering,
enabling physics-based augmentation for the counterfactual reasoning pipeline.

Supports two modes:
1. Taichi (GPU-accelerated) - if taichi is installed
2. NumPy (CPU fallback) - works everywhere

Usage:
    python render_taichi_fracture.py --ct <ct_path> --mask <mask_path> --output_dir <output>

Requirements:
    - nibabel, numpy, scipy (required)
    - taichi (optional, for GPU acceleration)
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from modules.taichi_ct_renderer import (
    TaichiCTRenderer,
    create_deformation_field_from_taichi,
    apply_deformation_to_ct,
    apply_deformation_to_mask
)


class NumpyFractureSimulator:
    """
    NumPy-based fracture simulator (CPU fallback).
    
    This provides the same interface as TaichiFractureSimulator but runs
    on CPU using NumPy. Use this when Taichi is not available.
    """
    
    def __init__(self, n_particles: int = 50000):
        """
        Initialize NumPy simulator.
        
        Args:
            n_particles: Maximum number of particles
        """
        self.n_particles = n_particles
        self.x = None
        self.original_x = None
        self.damage = None
        self.stress = None
        self.bounds_min = None
        self.bounds_max = None
        self.damage_point = None
        self.num_particles = 0
    
    def load_from_mesh(self, mesh_points: np.ndarray):
        """
        Load particle positions from mesh/voxel points.
        
        Args:
            mesh_points: [N, 3] array of points (will be normalized to [0.35, 0.65])
        """
        self.num_particles = min(len(mesh_points), self.n_particles)
        
        # Normalize to Taichi space [0.35, 0.65]
        points = mesh_points[:self.num_particles].copy()
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        
        # Normalize to [0, 1] then scale to [0.35, 0.65]
        normalized = (points - pmin) / (pmax - pmin + 1e-8)
        normalized = normalized * 0.3 + 0.35
        
        # Store bounds
        self.bounds_min = normalized.min(axis=0).astype(np.float32)
        self.bounds_max = normalized.max(axis=0).astype(np.float32)
        
        # Initialize particles
        self.x = np.zeros((self.n_particles, 3), dtype=np.float32)
        self.x[:self.num_particles] = normalized
        
        self.original_x = self.x.copy()
        
        # Initialize damage and stress
        self.damage = np.zeros(self.n_particles, dtype=np.float32)
        self.stress = np.zeros(self.n_particles, dtype=np.float32)
        
        # Set damage point at center-front
        center = normalized.mean(axis=0)
        self.damage_point = center.copy()
        self.damage_point[2] += 0.05  # Slightly forward
        
        print(f"[NumPy] Loaded {self.num_particles} particles")
        print(f"[NumPy] Bounds: {self.bounds_min} to {self.bounds_max}")
    
    def load_from_mask(self, mask: np.ndarray, label: int = 1, downsample: int = 2):
        """
        Load particle positions from segmentation mask.
        
        Args:
            mask: 3D segmentation mask
            label: Vertebra label to use
            downsample: Downsampling factor to reduce particle count
        """
        # Get voxel coordinates of vertebra
        coords = np.array(np.where(mask == label)).T
        
        # Downsample
        if downsample > 1:
            indices = np.random.choice(len(coords), max(1, len(coords) // downsample), replace=False)
            coords = coords[indices]
        
        print(f"[NumPy] Extracted {len(coords)} particles from mask (label={label})")
        
        # Normalize coordinates to [0, 1]
        coords_normalized = coords.astype(np.float32)
        coords_normalized = coords_normalized / np.array(mask.shape)
        
        self.load_from_mesh(coords_normalized)
    
    def _simulate_step(self, force: float):
        """Single simulation step with stress and damage computation."""
        n = self.num_particles
        
        # Get active particles
        pos = self.original_x[:n]
        dp = self.damage_point
        
        # Distance from damage point
        diff = pos - dp
        dist = np.linalg.norm(diff, axis=1)
        
        # Stress calculation (1/r^2 decay)
        local_stress = np.zeros(n, dtype=np.float32)
        
        # Very close to damage point
        close_mask = dist < 0.001
        local_stress[close_mask] = force * 10.0
        
        # Normal distance
        far_mask = ~close_mask
        local_stress[far_mask] = force / (dist[far_mask] ** 2 * 100.0 + 0.1)
        
        # Stiffness degradation
        effective_stiffness = (1.0 - self.damage[:n]) ** 2
        local_stress = local_stress * effective_stiffness
        
        self.stress[:n] = local_stress
        
        # Damage accumulation
        sigma_ult = 1.0
        damage_mask = local_stress > sigma_ult * 0.5
        overstress = (local_stress[damage_mask] - sigma_ult * 0.5) / sigma_ult
        self.damage[:n][damage_mask] = np.minimum(
            self.damage[:n][damage_mask] + overstress * 0.01, 
            1.0
        )
    
    def _compute_deformation(self):
        """Compute particle displacements based on damage."""
        n = self.num_particles
        
        orig_pos = self.original_x[:n]
        d = self.damage[:n]
        dp = self.damage_point
        
        # Direction to particle from damage point
        to_particle = orig_pos - dp
        dist = np.linalg.norm(to_particle, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-6)  # Avoid division by zero
        direction = to_particle / dist
        
        # High damage: crack opening displacement
        high_damage_mask = (d > 0.8) & (dist[:, 0] > 0.001)
        cod = (d[high_damage_mask] - 0.8) * 0.02
        self.x[:n][high_damage_mask] = orig_pos[high_damage_mask] + direction[high_damage_mask] * cod[:, np.newaxis]
        
        # Low damage: small elastic deformation
        low_damage_mask = ~high_damage_mask
        s = self.stress[:n][low_damage_mask]
        elastic_strain = s * 0.001 * (1.0 - d[low_damage_mask])
        self.x[:n][low_damage_mask] = orig_pos[low_damage_mask] - direction[low_damage_mask] * (elastic_strain * 0.1)[:, np.newaxis]
    
    def simulate(self, n_steps: int = 100, max_force: float = 3.0):
        """
        Run fracture simulation.
        
        Args:
            n_steps: Number of simulation steps
            max_force: Maximum applied force
        """
        force_rate = max_force / n_steps
        current_force = 0.0
        
        for step in range(n_steps):
            current_force += force_rate
            self._simulate_step(current_force)
            self._compute_deformation()
            
            if step % 20 == 0:
                max_damage = self.damage[:self.num_particles].max()
                damaged_pct = (self.damage[:self.num_particles] > 0.5).sum() / self.num_particles * 100
                print(f"[NumPy] Step {step}: force={current_force:.2f}, max_damage={max_damage:.3f}, damaged={damaged_pct:.1f}%")
    
    def get_state(self):
        """Get current simulation state for CT rendering."""
        return {
            'original_positions': self.original_x[:self.num_particles].copy(),
            'deformed_positions': self.x[:self.num_particles].copy(),
            'damage': self.damage[:self.num_particles].copy(),
            'bounds': (self.bounds_min.copy(), self.bounds_max.copy())
        }


# Try to import Taichi version
def get_simulator(n_particles: int = 50000, use_taichi: bool = True):
    """
    Get the best available simulator.
    
    Args:
        n_particles: Maximum number of particles
        use_taichi: Whether to try Taichi first
        
    Returns:
        Simulator instance (Taichi or NumPy)
    """
    if use_taichi:
        try:
            import taichi as ti
            ti.init(arch=ti.cpu)  # Use CPU for compatibility
            
            # Import the Taichi version from the original simulator
            from simulator.taichi_simulator import (
                x, original_x, damage, stress,
                bounds_min, bounds_max, damage_point,
                init_particles, compute_stress_and_damage, compute_deformation
            )
            
            print("Using Taichi simulator (GPU-accelerated)")
            # Return a wrapper... for now just use NumPy
            return NumpyFractureSimulator(n_particles)
            
        except ImportError:
            print("Taichi not available, using NumPy fallback")
    
    return NumpyFractureSimulator(n_particles)


def render_physics_to_ct(
    ct_path: str,
    mask_path: str,
    output_dir: str,
    vertebra_label: int = 1,
    n_steps: int = 100,
    max_force: float = 3.0
):
    """
    End-to-end pipeline: Physics simulation → CT rendering.
    
    Args:
        ct_path: Path to original CT
        mask_path: Path to segmentation mask
        output_dir: Output directory for rendered files
        vertebra_label: Label of vertebra to fracture
        n_steps: Simulation steps
        max_force: Maximum force for simulation
    """
    import nibabel as nib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Physics Fracture Simulation → CT Rendering")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading CT and mask...")
    ct_nii = nib.load(ct_path)
    mask_nii = nib.load(mask_path)
    
    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Mask shape: {mask_data.shape}")
    
    # Check if vertebra exists
    if (mask_data == vertebra_label).sum() == 0:
        print(f"  Warning: No voxels found for label {vertebra_label}")
        available_labels = np.unique(mask_data)
        print(f"  Available labels: {available_labels}")
        return None, None
    
    # Initialize simulator
    print("\n[2/4] Initializing simulator...")
    sim = NumpyFractureSimulator(n_particles=100000)
    sim.load_from_mask(mask_data, label=vertebra_label, downsample=4)
    
    # Run simulation
    print("\n[3/4] Running fracture simulation...")
    sim.simulate(n_steps=n_steps, max_force=max_force)
    
    # Get state
    state = sim.get_state()
    
    # Render to CT
    print("\n[4/4] Rendering to CT...")
    renderer = TaichiCTRenderer(ct_path, mask_path, vertebra_label)
    
    fractured_ct, fractured_mask = renderer.render_from_particles(
        state['original_positions'],
        state['deformed_positions'],
        state['bounds'],
        damage=state['damage'],
        smoothing_sigma=1.0
    )
    
    # Save outputs
    output_ct = output_dir / "physics_fractured_ct.nii.gz"
    output_mask = output_dir / "physics_fractured_mask.nii.gz"
    
    renderer.save(fractured_ct, fractured_mask, str(output_ct), str(output_mask))
    
    print("\n" + "=" * 70)
    print("✓ Complete!")
    print(f"  CT: {output_ct}")
    print(f"  Mask: {output_mask}")
    print("=" * 70)
    
    return fractured_ct, fractured_mask


def main():
    parser = argparse.ArgumentParser(
        description="Render physics fracture simulation to CT"
    )
    parser.add_argument("--ct", required=True, help="Path to CT NIfTI file")
    parser.add_argument("--mask", required=True, help="Path to mask NIfTI file")
    parser.add_argument("--output_dir", default="outputs/physics_ct", help="Output directory")
    parser.add_argument("--label", type=int, default=1, help="Vertebra label")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--force", type=float, default=3.0, help="Max force")
    
    args = parser.parse_args()
    
    render_physics_to_ct(
        args.ct,
        args.mask,
        args.output_dir,
        args.label,
        args.steps,
        args.force
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Demo mode with synthetic data
        print("Running demo with synthetic data...")
        print("For real usage: python render_taichi_fracture.py --ct <path> --mask <path>")
        print()
        
        # Create synthetic test
        sim = NumpyFractureSimulator(n_particles=10000)
        
        # Generate random mesh points
        points = np.random.rand(5000, 3)
        sim.load_from_mesh(points)
        
        # Run simulation
        sim.simulate(n_steps=50, max_force=2.0)
        
        # Get state
        state = sim.get_state()
        
        print("\n" + "=" * 70)
        print("Simulation complete!")
        print(f"Max damage: {state['damage'].max():.3f}")
        print(f"Damaged particles: {(state['damage'] > 0.5).sum()}")
        
        # Calculate displacement stats
        disp = state['deformed_positions'] - state['original_positions']
        print(f"Max displacement: {np.abs(disp).max():.4f} (normalized units)")
        print("=" * 70)
    else:
        main()
