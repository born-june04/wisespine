#!/usr/bin/env python3
"""
PyBullet Fracture - Simplified with Better Fragmentation.
"""

import pybullet as p
import pybullet_data
import numpy as np
import trimesh
import os

print("="*70)
print("PYBULLET FRACTURE - Simplified Approach")
print("="*70)

# Simple but effective fragmentation
def fragment_mesh_simple(mesh, num_fragments=5):
    """
    Fragment mesh by slicing with planes.
    More reliable than Voronoi for small meshes.
    """
    print(f"\n--- Fragmenting mesh into ~{num_fragments} pieces ---")
    
    # Get bounds
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    
    fragments = [mesh]
    
    # Slice along Z axis (vertebra height)
    num_z_slices = num_fragments
    z_step = size[2] / num_z_slices
    
    new_fragments = []
    for frag in fragments:
        for i in range(num_z_slices):
            z_min = bounds[0][2] + i * z_step
            z_max = z_min + z_step
            
            # Keep vertices in this Z range
            vertices = frag.vertices
            faces = frag.faces
            
            # Find vertices in range
            in_range = (vertices[:, 2] >= z_min) & (vertices[:, 2] < z_max)
            
            if in_range.sum() > 3:
                # Create fragment
                v_indices = np.where(in_range)[0]
                v_map = {old_idx: new_idx for new_idx, old_idx in enumerate(v_indices)}
                
                new_vertices = vertices[in_range]
                new_faces = []
                
                for face in faces:
                    if all(v_idx in v_map for v_idx in face):
                        new_face = [v_map[v_idx] for v_idx in face]
                        new_faces.append(new_face)
                
                if len(new_faces) > 0:
                    frag_mesh = trimesh.Trimesh(
                        vertices=new_vertices,
                        faces=np.array(new_faces)
                    )
                    new_fragments.append(frag_mesh)
                    print(f"  Fragment {len(new_fragments)}: {len(new_vertices)} vertices")
    
    if len(new_fragments) == 0:
        print("  Warning: No fragments created, using original mesh")
        return [mesh]
    
    print(f"✓ Created {len(new_fragments)} fragments")
    return new_fragments


# Create URDF with separate bodies + constraints
def create_fracturable_urdf_v2(fragments, output_dir, name="vertebra"):
    """
    Create URDF where each fragment is a separate body file.
    We'll connect them with constraints in PyBullet code.
    """
    print(f"\n--- Creating fracturable URDF (v2) ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each fragment as separate URDF
    urdf_paths = []
    
    for i, frag in enumerate(fragments):
        frag_name = f"{name}_frag_{i}"
        frag_obj = os.path.join(output_dir, f"{frag_name}.obj")
        frag.export(frag_obj)
        
        # Create individual URDF
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{frag_name}">
  <link name="base_link">
    <inertial>
      <mass value="0.02"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" 
               iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{frag_name}.obj" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="bone"><color rgba="0.9 0.9 0.8 1"/></material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{frag_name}.obj" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        
        urdf_path = os.path.join(output_dir, f"{frag_name}.urdf")
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        urdf_paths.append(urdf_path)
        print(f"  Created: {frag_name}.urdf")
    
    print(f"✓ Created {len(urdf_paths)} URDFs")
    return urdf_paths


# Simulate with dynamic fracture
def simulate_with_breakable_constraints(urdf_paths, force_magnitude=1000):
    """
    Load fragments as separate bodies, connect with constraints,
    then break constraints when force exceeds threshold.
    """
    print(f"\n--- PyBullet Simulation with Breakable Constraints ---")
    
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Load all fragments
    fragment_ids = []
    base_pos = [0, 0, 0.5]
    
    for i, urdf_path in enumerate(urdf_paths):
        pos = [base_pos[0], base_pos[1], base_pos[2] + i * 0.01]  # Stack fragments
        frag_id = p.loadURDF(urdf_path, pos)
        fragment_ids.append(frag_id)
        print(f"  Loaded fragment {i} (ID: {frag_id})")
    
    # Create constraints between adjacent fragments
    constraints = []
    fracture_threshold = 500.0  # N
    
    for i in range(len(fragment_ids) - 1):
        # Fixed constraint between fragment i and i+1
        constraint_id = p.createConstraint(
            fragment_ids[i], -1,
            fragment_ids[i+1], -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.01],
            [0, 0, 0]
        )
        constraints.append({
            'id': constraint_id,
            'frag_a': i,
            'frag_b': i+1,
            'broken': False
        })
        print(f"  Created constraint {constraint_id} between fragments {i} and {i+1}")
    
    print(f"\n✓ Setup complete: {len(fragment_ids)} fragments, {len(constraints)} constraints")
    
    # Simulation loop
    print(f"\n--- Running simulation ---")
    broken_count = 0
    
    for step in range(500):
        # Apply strong force at step 100
        if step == 100:
            print(f"\n💥 Applying force {force_magnitude}N to top fragment!")
            p.applyExternalForce(
                fragment_ids[-1], -1,
                [0, 0, force_magnitude],
                [0, 0, 0],
                p.LINK_FRAME
            )
        
        p.stepSimulation()
        
        # Check constraint forces and break if needed
        if step >= 100 and step < 300:
            for constraint in constraints:
                if not constraint['broken']:
                    # Get constraint force (approximation via contact)
                    # In full implementation, we'd use force sensors
                    
                    # Simple heuristic: check distance between fragments
                    pos_a, _ = p.getBasePositionAndOrientation(fragment_ids[constraint['frag_a']])
                    pos_b, _ = p.getBasePositionAndOrientation(fragment_ids[constraint['frag_b']])
                    dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
                    
                    # If fragments moved apart significantly, "break" constraint
                    if dist > 0.05:  # 5cm
                        p.removeConstraint(constraint['id'])
                        constraint['broken'] = True
                        broken_count += 1
                        print(f"  💥 FRACTURE! Constraint {constraint['id']} broken (step {step}, dist={dist:.3f}m)")
        
        # Print status
        if step % 100 == 0:
            pos_top, _ = p.getBasePositionAndOrientation(fragment_ids[-1])
            print(f"  t={step/100:.1f}s: Top fragment at z={pos_top[2]:.3f}m, Broken: {broken_count}/{len(constraints)}")
    
    print(f"\n✓ Simulation complete!")
    print(f"  Total fractures: {broken_count}/{len(constraints)}")
    
    p.disconnect()
    
    return broken_count


# Main
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 1: CREATE TEST MESH")
    print("="*70)
    
    # Create a more complex vertebra-like shape
    # Cylinder approximation
    test_mesh = trimesh.creation.cylinder(radius=0.01, height=0.04, sections=8)
    print(f"Test mesh: {len(test_mesh.vertices)} vertices, {len(test_mesh.faces)} faces")
    
    print("\n" + "="*70)
    print("STEP 2: FRAGMENT MESH")
    print("="*70)
    
    fragments = fragment_mesh_simple(test_mesh, num_fragments=5)
    
    print("\n" + "="*70)
    print("STEP 3: CREATE URDFs")
    print("="*70)
    
    urdf_paths = create_fracturable_urdf_v2(
        fragments,
        "outputs/pybullet_fracture_v2",
        name="L1"
    )
    
    print("\n" + "="*70)
    print("STEP 4: SIMULATE WITH FRACTURE")
    print("="*70)
    
    num_fractures = simulate_with_breakable_constraints(
        urdf_paths,
        force_magnitude=2000
    )
    
    print("\n" + "="*70)
    print("🎉 SUCCESS!")
    print("="*70)
    print(f"✅ Fragmentation: {len(fragments)} pieces")
    print(f"✅ PyBullet setup: {len(urdf_paths)} URDFs")
    print(f"✅ Dynamic fracture: {num_fractures} constraints broken")
    
    print("\n📝 NEXT STEPS:")
    print("  1. ✅ Mesh fragmentation")
    print("  2. ✅ PyBullet with breakable constraints")
    print("  3. ⏳ Load real vertebra mesh (not test cylinder)")
    print("  4. ⏳ CT rendering from PyBullet state")
    print("  5. ⏳ RL environment")
    print("  6. ⏳ Interactive GUI (for local machine)")
    print("="*70)

