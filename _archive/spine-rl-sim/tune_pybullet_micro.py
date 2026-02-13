#!/usr/bin/env python3
"""
Tune with MUCH smaller forces and apply to all fragments.
"""

import pybullet as p
import pybullet_data
import numpy as np
import glob

def test_micro_forces(
    mass: float,
    force_range: tuple,
    num_steps: int,
    damping: float
):
    """Test with very small forces applied to ALL fragments."""
    
    print(f"\n{'='*70}")
    print(f"Testing: mass={mass}, force={force_range}, steps={num_steps}, damping={damping}")
    print(f"{'='*70}")
    
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    planeId = p.loadURDF("plane.urdf")
    
    fragment_dir = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured"
    fragment_files = sorted(glob.glob(f"{fragment_dir}/L1_frag_*.obj"))
    
    if len(fragment_files) == 0:
        p.disconnect()
        return None
    
    fragment_bodies = []
    
    for i, frag_file in enumerate(fragment_files):
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
        
        start_pos = [0, 0, 0.3 + i * 0.01]
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        
        p.changeDynamics(body_id, -1, linearDamping=damping, angularDamping=damping)
        fragment_bodies.append(body_id)
    
    # Get initial positions
    initial_positions = []
    for body_id in fragment_bodies:
        pos, _ = p.getBasePositionAndOrientation(body_id)
        initial_positions.append(np.array(pos))
    
    # Generate random forces for each fragment (once at start)
    np.random.seed(42)
    fragment_forces = []
    for i in range(len(fragment_bodies)):
        force = np.random.uniform(force_range[0], force_range[1], size=3)
        fragment_forces.append(force)
        print(f"  Fragment {i} force: [{force[0]:.6f}, {force[1]:.6f}, {force[2]:.6f}]")
    
    # Apply forces and simulate
    for step in range(num_steps):
        for i, body_id in enumerate(fragment_bodies):
            p.applyExternalForce(
                body_id,
                -1,
                fragment_forces[i].tolist(),
                [0, 0, 0],
                p.LINK_FRAME
            )
        
        p.stepSimulation()
    
    # Get final displacements
    displacements = []
    for i, body_id in enumerate(fragment_bodies):
        pos, _ = p.getBasePositionAndOrientation(body_id)
        disp = np.array(pos) - initial_positions[i]
        displacements.append(disp)
    
    max_disp = np.max([np.linalg.norm(d) for d in displacements])
    avg_disp = np.mean([np.linalg.norm(d) for d in displacements])
    
    print(f"\n📊 Results:")
    print(f"  Max displacement: {max_disp*1000:.2f} mm")
    print(f"  Avg displacement: {avg_disp*1000:.2f} mm")
    
    for i, disp in enumerate(displacements):
        disp_mm = disp * 1000
        norm_mm = np.linalg.norm(disp_mm)
        print(f"  Fragment {i}: [{disp_mm[0]:7.2f}, {disp_mm[1]:7.2f}, {disp_mm[2]:7.2f}] mm (norm: {norm_mm:.2f})")
    
    ct_spacing = 0.6
    max_disp_voxels = max_disp * 1000 / ct_spacing
    avg_disp_voxels = avg_disp * 1000 / ct_spacing
    
    print(f"\n  → Max displacement: {max_disp_voxels:.1f} voxels")
    print(f"  → Avg displacement: {avg_disp_voxels:.1f} voxels")
    
    if 5 <= max_disp_voxels <= 50:
        rating = "✓ GOOD"
    elif max_disp_voxels < 5:
        rating = "✗ TOO SMALL"
    else:
        rating = "✗ TOO LARGE"
    
    print(f"  {rating}")
    
    p.disconnect()
    
    return {
        'mass': mass,
        'force_range': force_range,
        'num_steps': num_steps,
        'damping': damping,
        'max_disp_voxels': max_disp_voxels,
        'avg_disp_voxels': avg_disp_voxels,
        'rating': rating
    }


def main():
    print("="*70)
    print("PyBullet Micro-Force Tuning")
    print("="*70)
    print("\nGoal: 5-50 voxel displacements")
    print("Target: ~3-30mm")
    
    configs = [
        # (mass, force_range, steps, damping)
        (0.001, (-0.001, 0.001), 100, 0.5),     # Very small forces
        (0.001, (-0.0001, 0.0001), 100, 0.5),   # Even smaller
        (0.001, (-0.00001, 0.00001), 100, 0.5), # Tiny forces
        (0.001, (-0.00005, 0.00005), 100, 0.5), # Between
        (0.001, (-0.0001, 0.0001), 500, 0.5),   # Smaller + more steps
        (0.001, (-0.0001, 0.0001), 100, 0.9),   # High damping
        (0.01, (-0.001, 0.001), 100, 0.5),      # Heavier mass
        (0.0001, (-0.001, 0.001), 100, 0.5),    # Lighter mass
    ]
    
    results = []
    
    for mass, force_range, steps, damping in configs:
        result = test_micro_forces(mass, force_range, steps, damping)
        if result:
            results.append(result)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    good_configs = [r for r in results if "GOOD" in r['rating']]
    
    if good_configs:
        print("\n✓ Found working configurations:")
        for r in good_configs:
            print(f"  mass={r['mass']}, force={r['force_range']}, steps={r['num_steps']}, damping={r['damping']}")
            print(f"    → Max: {r['max_disp_voxels']:.1f} vox, Avg: {r['avg_disp_voxels']:.1f} vox")
    else:
        print("\n✗ No perfect match. Showing closest:")
        results_sorted = sorted(results, key=lambda r: abs(r['max_disp_voxels'] - 20))
        for r in results_sorted[:3]:
            print(f"  mass={r['mass']}, force={r['force_range']}, steps={r['num_steps']}")
            print(f"    → Max: {r['max_disp_voxels']:.1f} vox, Avg: {r['avg_disp_voxels']:.1f} vox")
    
    print("="*70)

if __name__ == "__main__":
    main()

