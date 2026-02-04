#!/usr/bin/env python3
"""Quick test of training setup."""

import sys
sys.path.insert(0, '/gscratch/scrubbed/june0604/vindr/spine-rl-sim')

print("="*70)
print("Testing PyBullet Training Setup")
print("="*70)

# Test imports
print("\n1. Testing imports...")
try:
    from modules.pybullet_fracture_env import PyBulletFractureEnv
    print("  ✓ PyBulletFractureEnv")
    from modules.pybullet_ct_renderer import render_pybullet_to_ct
    print("  ✓ render_pybullet_to_ct")
    from stable_baselines3 import PPO
    print("  ✓ PPO")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("  ✓ matplotlib")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test environment creation
print("\n2. Testing environment creation...")
try:
    env = PyBulletFractureEnv(gui=False)
    print("  ✓ Environment created")
    obs, _ = env.reset()
    print(f"  ✓ Reset successful, obs shape: {obs.shape}")
    env.close()
except Exception as e:
    print(f"  ✗ Environment error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test visualization
print("\n3. Testing visualization...")
try:
    import nibabel as nib
    import numpy as np
    
    # Load GT
    gt_ct = nib.load("/gscratch/scrubbed/june0604/vindr/VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz").get_fdata()
    print(f"  ✓ Loaded GT CT: {gt_ct.shape}")
    
    # Create simple plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(gt_ct[300, :, :].T, cmap='gray', origin='lower')
    ax.set_title('Test Plot')
    ax.axis('off')
    
    test_path = "/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/rl_training/current/test_plot.png"
    plt.savefig(test_path, dpi=50, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved test plot: {test_path}")
    
except Exception as e:
    print(f"  ✗ Visualization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)

