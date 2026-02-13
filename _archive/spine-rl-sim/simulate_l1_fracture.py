#!/usr/bin/env python3
"""
PyBullet Fracture Simulation - Visualize fragments breaking apart.

Shows before/after states with fragments separating.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import glob

print("="*70)
print("PYBULLET FRACTURE SIMULATION")
print("="*70)

# Create URDFs for each fragment
def create_fragment_urdfs(fragment_dir):
    """Create individual URDF for each fragment."""
    frag_objs = sorted(glob.glob(os.path.join(fragment_dir, "L1_frag_*.obj")))
    print(f"\nFound {len(frag_objs)} fragments")
    
    urdf_paths = []
    
    for i, obj_path in enumerate(frag_objs):
        obj_name = os.path.basename(obj_path)
        frag_name = obj_name.replace('.obj', '')
        
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{frag_name}">
  <link name="base_link">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" 
               iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{obj_name}" scale="1.0 1.0 1.0"/>
      </geometry>
      <material name="bone"><color rgba="0.9 0.9 0.8 1"/></material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{obj_name}" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        
        urdf_path = os.path.join(fragment_dir, f"{frag_name}.urdf")
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        urdf_paths.append(urdf_path)
        print(f"  Created: {frag_name}.urdf")
    
    return urdf_paths


# Simulate fracture
def simulate_fracture_with_screenshots(urdf_paths, fragment_dir):
    """Simulate and capture before/after states."""
    print("\n--- Starting PyBullet simulation ---")
    
    # Connect in DIRECT mode
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground
    p.loadURDF("plane.urdf", [0, 0, -1.3])
    
    # Load fragments stacked vertically (simulating assembled vertebra)
    fragment_ids = []
    base_pos = [0.005, 0.2, -1.165]  # Center of vertebra
    z_offset = 0.0  # Start at base
    
    print("\nLoading fragments...")
    for i, urdf_path in enumerate(urdf_paths):
        # Stack fragments along Z axis
        pos = [base_pos[0], base_pos[1], base_pos[2] + z_offset]
        frag_id = p.loadURDF(urdf_path, pos)
        fragment_ids.append(frag_id)
        print(f"  Fragment {i} loaded at z={pos[2]:.4f}")
        z_offset += 0.011  # ~11mm per fragment (vertebra is ~54mm tall)
    
    # Create constraints between adjacent fragments
    constraints = []
    print("\nCreating constraints...")
    for i in range(len(fragment_ids) - 1):
        constraint_id = p.createConstraint(
            fragment_ids[i], -1,
            fragment_ids[i+1], -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.005],  # Small offset
            [0, 0, 0.005]
        )
        constraints.append({
            'id': constraint_id,
            'frag_a': i,
            'frag_b': i+1,
            'broken': False
        })
        print(f"  Constraint {i}: fragment {i} <-> {i+1}")
    
    # Set camera
    camera_distance = 0.15
    camera_yaw = 45
    camera_pitch = -20
    camera_target = base_pos
    
    # === STATE 1: Initial (assembled) ===
    print("\n--- Capturing initial state ---")
    for _ in range(10):
        p.stepSimulation()
    
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.01, farVal=100
    )
    
    img_initial = p.getCameraImage(
        width=800, height=800,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Save initial state
    import matplotlib.pyplot as plt
    rgb_initial = np.array(img_initial[2]).reshape(800, 800, 4)
    
    # === Apply force to top fragment ===
    print("\n--- Applying force ---")
    force_magnitude = 5000  # Strong force
    force = [0, 0, force_magnitude]
    
    for step in range(200):
        if step < 50:
            # Apply force for first 50 steps
            p.applyExternalForce(
                fragment_ids[-1], -1,
                force, [0, 0, 0],
                p.LINK_FRAME
            )
        
        p.stepSimulation()
        
        # Check for fractures
        if step >= 50 and step < 150:
            for constraint in constraints:
                if not constraint['broken']:
                    # Check distance between fragments
                    pos_a, _ = p.getBasePositionAndOrientation(fragment_ids[constraint['frag_a']])
                    pos_b, _ = p.getBasePositionAndOrientation(fragment_ids[constraint['frag_b']])
                    dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
                    
                    if dist > 0.03:  # 3cm separation
                        p.removeConstraint(constraint['id'])
                        constraint['broken'] = True
                        print(f"  💥 Fracture! Constraint {constraint['id']} broken at step {step}")
        
        if step % 50 == 0:
            top_pos, _ = p.getBasePositionAndOrientation(fragment_ids[-1])
            broken_count = sum(1 for c in constraints if c['broken'])
            print(f"  Step {step}: Top at z={top_pos[2]:.4f}, Broken: {broken_count}/{len(constraints)}")
    
    # === STATE 2: Fractured ===
    print("\n--- Capturing fractured state ---")
    img_fractured = p.getCameraImage(
        width=800, height=800,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    rgb_fractured = np.array(img_fractured[2]).reshape(800, 800, 4)
    
    # Get final positions
    print("\n--- Final fragment positions ---")
    for i, frag_id in enumerate(fragment_ids):
        pos, _ = p.getBasePositionAndOrientation(frag_id)
        print(f"  Fragment {i}: z={pos[2]:.4f}")
    
    broken_count = sum(1 for c in constraints if c['broken'])
    print(f"\n✓ Simulation complete: {broken_count}/{len(constraints)} constraints broken")
    
    p.disconnect()
    
    return rgb_initial, rgb_fractured, broken_count


# Main execution
fragment_dir = "outputs/phase3_physics_fracture/pybullet_models/L1_fractured"

print("\n=== Step 1: Create URDFs ===")
urdf_paths = create_fragment_urdfs(fragment_dir)

print("\n=== Step 2: Run simulation ===")
img_initial, img_fractured, broken_count = simulate_fracture_with_screenshots(urdf_paths, fragment_dir)

print("\n=== Step 3: Create visualization ===")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('L1 Vertebra Fracture Simulation (PyBullet)', 
             fontsize=16, fontweight='bold')

# Initial state
axes[0].imshow(img_initial[:, :, :3])
axes[0].set_title('Initial State\n(5 fragments connected)', fontsize=14, color='green')
axes[0].axis('off')

# Fractured state
axes[1].imshow(img_fractured[:, :, :3])
axes[1].set_title(f'Fractured State\n({broken_count} constraints broken)', 
                  fontsize=14, color='red', fontweight='bold')
axes[1].axis('off')

plt.tight_layout()

output_path = os.path.join(fragment_dir, "fracture_simulation.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\n✓ Saved: {output_path}")

# Also create a combined view with static fragmentation
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 16))
fig2.suptitle('L1 Vertebra Fragmentation & Fracture Simulation', 
              fontsize=16, fontweight='bold')

# Load the fragmentation visualization
frag_viz = plt.imread(os.path.join(fragment_dir, "fragmentation_visualization.png"))
axes2[0, 0].imshow(frag_viz)
axes2[0, 0].set_title('Fragmentation Process', fontsize=12)
axes2[0, 0].axis('off')

# Initial PyBullet state
axes2[0, 1].imshow(img_initial[:, :, :3])
axes2[0, 1].set_title('PyBullet - Assembled', fontsize=12, color='green')
axes2[0, 1].axis('off')

# Fractured PyBullet state
axes2[1, 0].imshow(img_fractured[:, :, :3])
axes2[1, 0].set_title('PyBullet - Fractured', fontsize=12, color='red')
axes2[1, 0].axis('off')

# Info panel
axes2[1, 1].axis('off')
info_text = f"""
FRACTURE SIMULATION RESULTS

Fragments: 5 pieces

Simulation:
  • Force applied: 5000 N
  • Constraints broken: {broken_count}/4
  • Physics: PyBullet
  • Gravity: Enabled

Next Steps:
  1. ✅ Fragmentation
  2. ✅ PyBullet simulation
  3. ⏳ CT rendering
  4. ⏳ TotalSegmentator
  5. ⏳ RL training

Status: Ready for CT rendering!
"""

axes2[1, 1].text(0.1, 0.5, info_text,
                 transform=axes2[1, 1].transAxes,
                 fontsize=12, verticalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

output_path2 = os.path.join(fragment_dir, "COMPLETE_VISUALIZATION.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')

print(f"✓ Saved: {output_path2}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\n📊 Output files:")
print(f"  • {output_path}")
print(f"  • {output_path2}")
print("\n✨ You can now see:")
print("  1. Original mesh → 5 fragments")
print("  2. Assembled state (PyBullet)")
print("  3. Fractured state (fragments separated)")
print("\n🚀 Ready for CT rendering of fractured vertebra!")
print("="*70)
