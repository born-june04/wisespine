#!/usr/bin/env python3
"""
Evaluate RL-generated fracture vs baselines using TotalSegmentator.

Comparison:
1. Original (clean) - baseline
2. Manual fracture (rule-based)
3. RL fracture (learned)
"""

import sys
import os
sys.path.insert(0, '/gscratch/scrubbed/june0604/vindr/spine-rl-sim')

import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import glob

from modules.pybullet_fracture_env import PyBulletFractureEnv
from modules.pybullet_ct_renderer import render_pybullet_to_ct
from stable_baselines3 import PPO

print("="*70)
print("RL Fracture Evaluation: Dice Score Comparison")
print("="*70)

output_dir = Path("/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

gt_mask_nii = nib.load("/gscratch/scrubbed/june0604/vindr/VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
gt_mask = gt_mask_nii.get_fdata()

def compute_dice(ts_dir, gt_mask):
    """Compute Dice score from TS predictions."""
    ts_files = sorted(glob.glob(str(ts_dir / "vertebrae_*.nii.gz")))
    
    if len(ts_files) == 0:
        print(f"  ⚠ No TS predictions found in {ts_dir}")
        return None
    
    ts_mask = np.zeros_like(gt_mask)
    for ts_file in ts_files:
        vert_data = nib.load(ts_file).get_fdata()
        ts_mask[vert_data > 0] = 1
    
    gt_binary = gt_mask > 0
    ts_binary = ts_mask > 0
    intersection = np.logical_and(gt_binary, ts_binary).sum()
    dice = 2.0 * intersection / (gt_binary.sum() + ts_binary.sum() + 1e-6)
    
    return dice

# =========================================================================
# 1. Original (Baseline)
# =========================================================================
print("\n" + "="*70)
print("1. Evaluating: ORIGINAL (Clean CT)")
print("="*70)

original_ct_path = "/gscratch/scrubbed/june0604/vindr/VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz"
original_ts_dir = output_dir / "ts_original"

if not (original_ts_dir / "vertebrae_L1.nii.gz").exists():
    print("  Running TotalSegmentator on original...")
    cmd = [
        "TotalSegmentator",
        "-i", original_ct_path,
        "-o", str(original_ts_dir),
        "-ta", "total",
        "--fast"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr}")
    else:
        print(f"  ✓ TS completed")
else:
    print("  ✓ Using cached TS results")

dice_original = compute_dice(original_ts_dir, gt_mask)
print(f"\n📊 Dice (Original): {dice_original:.4f}")

# =========================================================================
# 2. Manual Fracture (Rule-based)
# =========================================================================
print("\n" + "="*70)
print("2. Evaluating: MANUAL FRACTURE (Rule-based)")
print("="*70)

manual_ct_path = "/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/ct_renderings/fractured_ct_manual.nii.gz"
manual_ts_dir = output_dir / "ts_manual"

if Path(manual_ct_path).exists():
    if not (manual_ts_dir / "vertebrae_L1.nii.gz").exists():
        print("  Running TotalSegmentator on manual fracture...")
        cmd = [
            "TotalSegmentator",
            "-i", manual_ct_path,
            "-o", str(manual_ts_dir),
            "-ta", "total",
            "--fast"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ Failed: {result.stderr}")
        else:
            print(f"  ✓ TS completed")
    else:
        print("  ✓ Using cached TS results")
    
    dice_manual = compute_dice(manual_ts_dir, gt_mask)
    print(f"\n📊 Dice (Manual): {dice_manual:.4f}")
else:
    print(f"  ⚠ Manual fracture CT not found: {manual_ct_path}")
    dice_manual = None

# =========================================================================
# 3. RL Fracture (Learned)
# =========================================================================
print("\n" + "="*70)
print("3. Evaluating: RL FRACTURE (Learned)")
print("="*70)

# Load trained model
model_path = "/gscratch/scrubbed/june0604/vindr/outputs/phase3_physics_fracture/rl_training/2026-02-02_18-30-42/fracture_agent_final.zip"
print(f"  Loading model: {model_path}")

model = PPO.load(model_path)
print("  ✓ Model loaded")

# Create environment
env = PyBulletFractureEnv(gui=False)
obs, _ = env.reset()
print("  ✓ Environment created")

# Run deterministic policy
print("  Running RL agent...")
for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"  ✓ Completed {step+1} steps")

# Get displacements
disps = env.get_fragment_displacements()
max_disp = np.max([np.linalg.norm(d[1])*1000 for d in disps])
print(f"  Max displacement: {max_disp:.2f}mm")

# Render CT
print("  Rendering fractured CT...")
rl_ct_path = output_dir / "rl_fractured.nii.gz"
fractured_ct, fractured_mask = render_pybullet_to_ct(
    env,
    output_ct_path=str(rl_ct_path),
    output_mask_path=None
)
print(f"  ✓ CT saved: {rl_ct_path}")

env.close()

# Run TS
rl_ts_dir = output_dir / "ts_rl"
print("  Running TotalSegmentator on RL fracture...")
cmd = [
    "TotalSegmentator",
    "-i", str(rl_ct_path),
    "-o", str(rl_ts_dir),
    "-ta", "total",
    "--fast"
]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ✗ Failed: {result.stderr}")
    dice_rl = None
else:
    print(f"  ✓ TS completed")
    dice_rl = compute_dice(rl_ts_dir, gt_mask)
    print(f"\n📊 Dice (RL): {dice_rl:.4f}")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

results = [
    ("Original (Clean)", dice_original),
    ("Manual Fracture", dice_manual),
    ("RL Fracture", dice_rl),
]

print("\nMethod                  | Dice Score | Degradation")
print("-" * 55)
for name, dice in results:
    if dice is not None:
        if dice_original is not None:
            deg = (dice_original - dice) * 100
            print(f"{name:23s} | {dice:.4f}    | {deg:+.2f}%")
        else:
            print(f"{name:23s} | {dice:.4f}    | N/A")
    else:
        print(f"{name:23s} | N/A        | N/A")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if dice_rl is not None and dice_manual is not None and dice_original is not None:
    if dice_rl < dice_manual:
        print("✓ RL outperforms Manual: More challenging for TS!")
        print(f"  RL advantage: {(dice_manual - dice_rl)*100:.2f}% more degradation")
    elif dice_rl < dice_manual * 1.02:  # Within 2%
        print("≈ RL and Manual comparable: Similar challenge level")
    else:
        print("✗ Manual outperforms RL: RL less challenging")
        print(f"  Manual advantage: {(dice_rl - dice_manual)*100:.2f}% more degradation")
    
    if dice_rl < 0.75:
        print("✓ RL creates significant challenge (Dice < 0.75)")
    elif dice_rl < 0.85:
        print("○ RL creates moderate challenge (0.75 ≤ Dice < 0.85)")
    else:
        print("✗ RL creates weak challenge (Dice ≥ 0.85)")

print("\n" + "="*70)
print("Evaluation complete!")
print("="*70)

