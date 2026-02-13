#!/usr/bin/env python3
"""
Tune PyBullet physics parameters to get realistic fracture displacements.

We'll test different combinations of:
- Mass (lighter = moves more)
- Force magnitude
- Physics steps
- Damping
"""

import pybullet as p
import pybullet_data
import numpy as np
import glob

def test_physics_config(
    mass: float,
    force: float,
    num_steps: int,
    damping: float,
    gravity: bool = False
):
    """Test a specific physics configuration."""
    
    print(f"\n{'='*70}")
    print(f"Testing: mass={mass}, force={force}, steps={num_steps}, damping={damping}, gravity={gravity}")
    print(f"{'='*70}")
    
    # Initialize PyBullet
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    if gravity:
        p.setGravity(0, 0, -10)
    else:
        p.setGravity(0, 0, 0)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load fragments
    fragment_dir = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured"
    fragment_files = sorted(glob.glob(f"{fragment_dir}/L1_frag_*.obj"))
    
    if len(fragment_files) == 0:
        print("❌ No fragments found!")
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
        
        # Stack fragments with small spacing
        start_pos = [0, 0, 0.3 + i * 0.01]
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        
        # Set damping
        p.changeDynamics(
            body_id,
            -1,
            linearDamping=damping,
            angularDamping=damping
        )
        
        fragment_bodies.append(body_id)
    
    # Get initial positions
    initial_positions = []
    for body_id in fragment_bodies:
        pos, _ = p.getBasePositionAndOrientation(body_id)
        initial_positions.append(np.array(pos))
    
    # Apply force and simulate
    for step in range(num_steps):
        # Apply force to first fragment
        p.applyExternalForce(
            fragment_bodies[0],
            -1,
            [force, 0, 0],
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
    
    # Calculate statistics
    max_disp = np.max([np.linalg.norm(d) for d in displacements])
    avg_disp = np.mean([np.linalg.norm(d) for d in displacements])
    
    print(f"\n📊 Results:")
    print(f"  Max displacement: {max_disp*1000:.2f} mm")
    print(f"  Avg displacement: {avg_disp*1000:.2f} mm")
    
    # Print per-fragment
    for i, disp in enumerate(displacements):
        disp_mm = disp * 1000
        print(f"  Fragment {i}: [{disp_mm[0]:7.2f}, {disp_mm[1]:7.2f}, {disp_mm[2]:7.2f}] mm")
    
    # Convert to voxels (approximate, assuming 0.6mm spacing)
    ct_spacing = 0.6  # mm
    max_disp_voxels = max_disp * 1000 / ct_spacing
    
    print(f"\n  → Max displacement in voxels: {max_disp_voxels:.1f}")
    
    # Rating
    if 5 <= max_disp_voxels <= 50:
        rating = "✓ GOOD (realistic fracture)"
    elif max_disp_voxels < 5:
        rating = "✗ TOO SMALL (barely moved)"
    else:
        rating = "✗ TOO LARGE (exploded)"
    
    print(f"  {rating}")
    
    p.disconnect()
    
    return {
        'mass': mass,
        'force': force,
        'num_steps': num_steps,
        'damping': damping,
        'gravity': gravity,
        'max_disp_mm': max_disp * 1000,
        'max_disp_voxels': max_disp_voxels,
        'rating': rating
    }


def main():
    print("="*70)
    print("PyBullet Physics Parameter Tuning")
    print("="*70)
    print("\nGoal: Get 5-50 voxel displacements (realistic fracture)")
    print("(Roughly 3-30mm given 0.6mm voxel spacing)")
    
    # Test configurations
    configs = [
        # (mass, force, steps, damping, gravity)
        (0.001, 0.1, 100, 0.1, False),      # Light mass, medium force
        (0.001, 0.5, 100, 0.1, False),      # Light mass, higher force
        (0.001, 1.0, 100, 0.1, False),      # Light mass, even higher force
        (0.0001, 0.1, 100, 0.1, False),     # Very light mass
        (0.001, 0.1, 500, 0.1, False),      # More steps (accumulate force)
        (0.001, 0.1, 100, 0.0, False),      # No damping
        (0.001, 0.1, 100, 0.5, False),      # High damping
        (0.001, 0.01, 1000, 0.0, False),    # Small force, many steps, no damping
    ]
    
    results = []
    
    for mass, force, steps, damping, gravity in configs:
        result = test_physics_config(mass, force, steps, damping, gravity)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find best configs
    good_configs = [r for r in results if "GOOD" in r['rating']]
    
    if good_configs:
        print("\n✓ Found working configurations:")
        for r in good_configs:
            print(f"  mass={r['mass']}, force={r['force']}, steps={r['num_steps']}, damping={r['damping']}")
            print(f"    → {r['max_disp_voxels']:.1f} voxels ({r['max_disp_mm']:.2f} mm)")
    else:
        print("\n✗ No good configurations found. Need more tuning.")
        
        # Show closest
        results_sorted = sorted(results, key=lambda r: abs(r['max_disp_voxels'] - 20))
        best = results_sorted[0]
        print(f"\nClosest config:")
        print(f"  mass={best['mass']}, force={best['force']}, steps={best['num_steps']}, damping={best['damping']}")
        print(f"    → {best['max_disp_voxels']:.1f} voxels")
    
    print("="*70)

if __name__ == "__main__":
    main()

