#!/usr/bin/env python3
"""
Test different strategies for breaking PyBullet constraints.

We'll try:
1. Higher force magnitudes
2. Impulse instead of continuous force
3. Weaker constraints (lower breaking force)
4. Point-to-point constraints instead of fixed
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import glob
import os

def test_breaking_strategy(strategy_name, setup_fn, apply_force_fn):
    """Test a constraint breaking strategy."""
    
    print(f"\n{'='*70}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"{'='*70}")
    
    # Initialize PyBullet
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load fragments
    fragment_dir = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured"
    fragment_files = sorted(glob.glob(f"{fragment_dir}/L1_frag_*.obj"))
    
    if len(fragment_files) == 0:
        print("❌ No fragments found!")
        p.disconnect()
        return False
    
    print(f"Loading {len(fragment_files)} fragments...")
    
    # Setup (strategy-specific)
    fragment_bodies = setup_fn(fragment_files)
    
    if not fragment_bodies:
        print("❌ Setup failed!")
        p.disconnect()
        return False
    
    print(f"✓ Loaded {len(fragment_bodies)} fragments")
    
    # Get initial positions
    initial_positions = []
    for body_id in fragment_bodies:
        pos, _ = p.getBasePositionAndOrientation(body_id)
        initial_positions.append(pos)
    
    # Apply force and simulate
    print("\nSimulating...")
    max_displacement = 0.0
    broke = False
    
    for step in range(2000):
        apply_force_fn(fragment_bodies, step)
        p.stepSimulation()
        
        # Check for breaking (significant displacement)
        for i, body_id in enumerate(fragment_bodies):
            pos, _ = p.getBasePositionAndOrientation(body_id)
            displacement = np.linalg.norm(np.array(pos) - np.array(initial_positions[i]))
            max_displacement = max(max_displacement, displacement)
            
            if displacement > 0.01:  # 1cm threshold
                broke = True
        
        if step % 100 == 0:
            print(f"  Step {step}: max_displacement = {max_displacement:.6f}m")
        
        if broke:
            print(f"  🎉 Breaking detected at step {step}!")
            break
    
    print(f"\n📊 Results:")
    print(f"  Max displacement: {max_displacement:.6f}m")
    print(f"  Breaking: {'YES ✓' if broke else 'NO ✗'}")
    
    p.disconnect()
    return broke


# Strategy 1: Much higher continuous force
def strategy1_setup(fragment_files):
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
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        
        fragment_bodies.append(body_id)
    
    # Create constraints with lower breaking force
    for i in range(len(fragment_bodies) - 1):
        constraint_id = p.createConstraint(
            fragment_bodies[i],
            -1,
            fragment_bodies[i + 1],
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0.01]
        )
        
        # LOWER breaking force threshold
        p.changeConstraint(constraint_id, maxForce=50)  # Much lower!
    
    return fragment_bodies

def strategy1_apply_force(fragment_bodies, step):
    # Apply MUCH higher force
    force_magnitude = 1000  # 10x higher!
    p.applyExternalForce(
        fragment_bodies[0],
        -1,
        [force_magnitude, 0, 0],
        [0, 0, 0],
        p.LINK_FRAME
    )


# Strategy 2: Impulse-based (sudden force)
def strategy2_setup(fragment_files):
    return strategy1_setup(fragment_files)  # Same setup

def strategy2_apply_force(fragment_bodies, step):
    # Apply impulse only once
    if step == 100:
        impulse = 10.0  # Strong impulse
        p.applyExternalForce(
            fragment_bodies[0],
            -1,
            [impulse, 0, 0],
            [0, 0, 0],
            p.WORLD_FRAME
        )


# Strategy 3: No constraints at all (free fragments)
def strategy3_setup(fragment_files):
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
        
        # Small initial spacing
        start_pos = [i * 0.002, 0, 0.3]
        
        body_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        
        fragment_bodies.append(body_id)
    
    # NO CONSTRAINTS - fragments are free!
    print("  (No constraints - fragments are free)")
    
    return fragment_bodies

def strategy3_apply_force(fragment_bodies, step):
    # Apply moderate force to first fragment
    if step < 500:
        force = 100
        p.applyExternalForce(
            fragment_bodies[0],
            -1,
            [force, 0, 0],
            [0, 0, 0],
            p.LINK_FRAME
        )


# Strategy 4: Point-to-point constraints (weaker)
def strategy4_setup(fragment_files):
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
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )
        
        fragment_bodies.append(body_id)
    
    # Point-to-point constraints
    for i in range(len(fragment_bodies) - 1):
        constraint_id = p.createConstraint(
            fragment_bodies[i],
            -1,
            fragment_bodies[i + 1],
            -1,
            p.JOINT_POINT2POINT,  # Different joint type!
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0.005]
        )
        
        p.changeConstraint(constraint_id, maxForce=10)  # Very weak
    
    return fragment_bodies

def strategy4_apply_force(fragment_bodies, step):
    # Apply force with torque
    if step < 500:
        p.applyExternalForce(
            fragment_bodies[0],
            -1,
            [500, 0, 0],
            [0, 0, 0],
            p.LINK_FRAME
        )
        
        # Also apply torque to create rotation
        p.applyExternalTorque(
            fragment_bodies[0],
            -1,
            [0, 0, 100],
            p.LINK_FRAME
        )


def main():
    print("="*70)
    print("PyBullet Constraint Breaking Test")
    print("="*70)
    
    strategies = [
        ("High Force + Weak Constraints", strategy1_setup, strategy1_apply_force),
        ("Impulse-based Breaking", strategy2_setup, strategy2_apply_force),
        ("No Constraints (Free Fragments)", strategy3_setup, strategy3_apply_force),
        ("Point-to-Point + Torque", strategy4_setup, strategy4_apply_force),
    ]
    
    results = {}
    
    for name, setup_fn, apply_fn in strategies:
        broke = test_breaking_strategy(name, setup_fn, apply_fn)
        results[name] = broke
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, broke in results.items():
        status = "✓ SUCCESS" if broke else "✗ FAILED"
        print(f"  {name:40s} {status}")
    
    print("\n💡 Recommendation:")
    if any(results.values()):
        working = [name for name, broke in results.items() if broke]
        print(f"   Use strategy: {working[0]}")
    else:
        print("   None worked - try GUI visualization to debug")
    
    print("="*70)

if __name__ == "__main__":
    main()

