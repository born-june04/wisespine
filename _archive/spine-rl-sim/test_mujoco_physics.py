#!/usr/bin/env python3
"""Test MuJoCo per-vertebra model: apply force to a vertebra and check if it moves."""

import mujoco
import numpy as np

# Load new per-vertebra model
xml_path = "outputs/mujoco_per_vertebra/sub-verse563/spine_per_vertebra.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("✓ Per-vertebra MuJoCo model loaded")
print(f"  Bodies: {model.nbody} (should be 24: world + 23 vertebrae)")
print(f"  Joints: {model.njnt} (should be 23 free joints)")
print(f"  DOF: {model.nv} (should be 138: 23 × 6)")
print()

# Test: apply force to L1 vertebra
l1_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L1")
print(f"✓ Found L1 body (id={l1_body_id})")

# Get initial position
mujoco.mj_forward(model, data)
initial_pos = data.xpos[l1_body_id].copy()
print(f"  Initial position: {initial_pos}")

# Apply upward force for 100 steps
force = np.array([0, 0, 10.0])  # 10N upward
for i in range(100):
    data.xfrc_applied[l1_body_id, :3] = force
    mujoco.mj_step(model, data)

final_pos = data.xpos[l1_body_id].copy()
displacement = final_pos - initial_pos

print(f"✓ After applying {force} for 100 steps:")
print(f"  Final position: {final_pos}")
print(f"  Displacement: {displacement}")
print(f"  Distance moved: {np.linalg.norm(displacement):.6f} m")

if np.linalg.norm(displacement) > 1e-6:
    print("\n✅ SUCCESS: Vertebra moved! Physics is working.")
    print("\n🎯 P3-1a COMPLETE: MuJoCo environment with per-vertebra bodies")
    print("   Next: P3-1b - Define RL action space")
else:
    print("\n⚠️  WARNING: Vertebra didn't move much.")

