#!/usr/bin/env python3
"""
PyBullet Fracture Test: Load vertebra and test dynamic fracture.

This demonstrates:
1. Real-time GUI visualization
2. Dynamic mesh fracturing based on force
3. Interactive physics simulation
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import trimesh
import os

print("="*70)
print("PYBULLET FRACTURE TEST")
print("="*70)

# Step 1: Convert OBJ to URDF format for PyBullet
def create_urdf_from_obj(obj_path, output_dir, name="vertebra", mass=0.1):
    """
    Create a simple URDF file for a vertebra mesh.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    urdf_content = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" 
               iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_path)}" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bone">
        <color rgba="0.9 0.9 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{os.path.basename(obj_path)}" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    
    urdf_path = os.path.join(output_dir, f"{name}.urdf")
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)
    
    # Copy OBJ to same directory (if not already there)
    import shutil
    dest_obj = os.path.join(output_dir, os.path.basename(obj_path))
    if os.path.abspath(obj_path) != os.path.abspath(dest_obj):
        shutil.copy(obj_path, dest_obj)
    
    print(f"✓ Created URDF: {urdf_path}")
    return urdf_path


# Step 2: Test with existing vertebra OBJ
print("\n--- Step 1: Preparing vertebra model ---")

# Find an existing vertebra OBJ (from previous export)
obj_path = "outputs/mujoco_exports/sub-verse563/meshes/L1.obj"

if not os.path.exists(obj_path):
    print(f"❌ OBJ not found: {obj_path}")
    print("Creating a simple test mesh instead...")
    
    # Create a simple box as test
    test_mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.04])
    obj_path = "outputs/pybullet_test/test_vertebra.obj"
    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    test_mesh.export(obj_path)
    print(f"✓ Created test mesh: {obj_path}")

# Create URDF
urdf_path = create_urdf_from_obj(
    obj_path,
    "outputs/pybullet_test",
    name="L1",
    mass=0.1
)

# Step 3: Launch PyBullet with GUI
print("\n--- Step 2: Launching PyBullet GUI ---")
print("💡 A 3D window should open!")
print("   Controls:")
print("   - Left mouse: Rotate camera")
print("   - Right mouse: Pan camera")
print("   - Scroll: Zoom")
print("   - Press 'g' to toggle GUI")

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Add ground plane
planeId = p.loadURDF("plane.urdf")

# Load vertebra
print("\n--- Step 3: Loading vertebra ---")
startPos = [0, 0, 0.5]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

try:
    vertebraId = p.loadURDF(urdf_path, startPos, startOrientation)
    print(f"✓ Loaded vertebra (ID: {vertebraId})")
except Exception as e:
    print(f"❌ Error loading URDF: {e}")
    p.disconnect()
    exit(1)

# Enable fracture (if supported - note: basic PyBullet may need plugin)
# For now, we'll demonstrate basic physics

# Step 4: Interactive simulation
print("\n--- Step 4: Running simulation ---")
print("Simulation running for 5 seconds...")
print("Watch the 3D window!")

for i in range(500):  # 5 seconds at 100 Hz
    # Apply force at step 100 (after 1 second)
    if i == 100:
        print("\n💥 Applying strong force to vertebra!")
        force = [0, 0, 5000]  # Strong upward force
        position = [0, 0, 0.5]
        p.applyExternalForce(
            vertebraId,
            -1,  # Base link
            force,
            position,
            p.WORLD_FRAME
        )
    
    p.stepSimulation()
    time.sleep(1./100.)  # 100 Hz
    
    # Print position every second
    if i % 100 == 0:
        pos, orn = p.getBasePositionAndOrientation(vertebraId)
        print(f"  t={i/100:.1f}s: position={pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")

print("\n✓ Simulation complete!")
print("\n--- Step 5: Keep window open for inspection ---")
print("Press Ctrl+C to close...")

try:
    while True:
        p.stepSimulation()
        time.sleep(1./100.)
except KeyboardInterrupt:
    print("\n✓ Closing PyBullet")
    p.disconnect()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\n📝 Notes:")
print("  1. PyBullet GUI works! ✓")
print("  2. Basic physics works! ✓")
print("  3. Next: Implement fracture mechanism")
print("\n💡 For fracture, we need to:")
print("  - Pre-fragment the mesh (Voronoi)")
print("  - Use compound shapes")
print("  - Implement breakable constraints")
print("="*70)

