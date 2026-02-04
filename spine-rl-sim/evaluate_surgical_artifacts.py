"""
Evaluate TotalSegmentator Performance on Surgical Artifact CT

Compare:
1. Original CT → TS prediction → Dice with GT
2. Artifact CT → TS prediction → Dice with GT

Target: 20-30% Dice degradation (vs Phase 3's 0.33%)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import json


def run_totalsegmentator(ct_path, output_dir, fast=True):
    """
    Run TotalSegmentator on CT volume.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(output_dir)]
    if fast:
        cmd.append("--fast")
    
    print(f"Running TotalSegmentator...")
    print(f"  Input: {ct_path}")
    print(f"  Output: {output_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("TotalSegmentator failed")
    
    print("✓ TotalSegmentator complete!")
    return output_dir


def load_ts_prediction(ts_output_dir, target_affine=None, target_shape=None):
    """
    Load TotalSegmentator prediction for L1 only and resample to GT space.
    """
    # Load L1 prediction
    l1_file = ts_output_dir / "vertebrae_L1.nii.gz"
    if not l1_file.exists():
        raise RuntimeError(f"L1 prediction not found: {l1_file}")
    
    l1_nii = nib.load(str(l1_file))
    l1_mask = l1_nii.get_fdata()
    
    # If no resampling needed, return as is
    if target_affine is None:
        return l1_mask > 0
    
    # Resample to target space
    from scipy.ndimage import affine_transform
    
    # Compute transformation from pred space to GT space
    pred_to_world = l1_nii.affine
    world_to_target = np.linalg.inv(target_affine)
    pred_to_target = world_to_target @ pred_to_world
    
    # Extract transform components
    # For simplicity, use nearest neighbor interpolation
    from scipy.ndimage import map_coordinates
    
    # Create target coordinate grid
    coords_target = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
    coords_target = coords_target.reshape(3, -1)
    
    # Add homogeneous coordinate
    coords_target_homo = np.vstack([coords_target, np.ones((1, coords_target.shape[1]))])
    
    # Transform to pred space
    coords_pred_homo = pred_to_target @ coords_target_homo
    coords_pred = coords_pred_homo[:3, :]
    
    # Interpolate (nearest neighbor for masks)
    resampled = map_coordinates(l1_mask, coords_pred, order=0, mode='constant', cval=0)
    resampled = resampled.reshape(target_shape)
    
    return resampled > 0


def compute_dice(pred_binary, gt_mask, label):
    """
    Compute Dice score for a specific label.
    pred_binary is already a boolean mask.
    """
    gt_binary = (gt_mask == label)
    
    intersection = (pred_binary & gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    dice = 2.0 * intersection / union
    return dice


def evaluate_surgical_artifacts():
    """
    Main evaluation: compare clean vs artifact CT.
    """
    # Paths
    ct_original_path = Path("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
    ct_artifact_path = Path("outputs/phase4_surgical_artifacts/artifact_synthesis/ct_with_pedicle_screws.nii.gz")
    gt_mask_path = Path("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
    
    output_dir = Path("outputs/phase4_surgical_artifacts/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GT
    print("Loading ground truth mask...")
    gt_nii = nib.load(str(gt_mask_path))
    gt_mask = gt_nii.get_fdata()
    
    # Run TS on original CT (if not already done)
    ts_original_dir = output_dir / "ts_original"
    if not ts_original_dir.exists():
        print("\n" + "="*60)
        print("Running TotalSegmentator on ORIGINAL CT...")
        print("="*60)
        run_totalsegmentator(ct_original_path, ts_original_dir)
    else:
        print(f"Using existing TS predictions: {ts_original_dir}")
    
    # Run TS on artifact CT
    ts_artifact_dir = output_dir / "ts_artifact"
    print("\n" + "="*60)
    print("Running TotalSegmentator on ARTIFACT CT (with screws)...")
    print("="*60)
    run_totalsegmentator(ct_artifact_path, ts_artifact_dir)
    
    # Load predictions and resample to GT space
    print("\nLoading and resampling predictions to GT space...")
    pred_original = load_ts_prediction(ts_original_dir, gt_nii.affine, gt_mask.shape)
    pred_artifact = load_ts_prediction(ts_artifact_dir, gt_nii.affine, gt_mask.shape)
    
    # Compute Dice for L1 (label 22)
    print("\nComputing Dice scores...")
    dice_original = compute_dice(pred_original, gt_mask, label=22)
    dice_artifact = compute_dice(pred_artifact, gt_mask, label=22)
    
    # Results
    print("\n" + "="*60)
    print("RESULTS: Phase 4 Surgical Artifacts")
    print("="*60)
    print(f"\nL1 Vertebra Dice Scores:")
    print(f"  Original CT:       {dice_original:.4f}")
    print(f"  Artifact CT:       {dice_artifact:.4f}")
    print(f"  Degradation:       {(dice_original - dice_artifact):.4f}")
    print(f"  Degradation (%):   {100 * (dice_original - dice_artifact) / dice_original:.2f}%")
    
    # Compare with Phase 3
    print(f"\n" + "-"*60)
    print("Comparison with Phase 3 (Fracture):")
    print(f"  Phase 3 degradation:  0.33-1.00%")
    print(f"  Phase 4 degradation:  {100 * (dice_original - dice_artifact) / dice_original:.2f}%")
    print(f"  Improvement factor:   {(100 * (dice_original - dice_artifact) / dice_original) / 1.0:.1f}x")
    
    # Target check
    target_min = 20.0
    target_max = 30.0
    degradation_pct = 100 * (dice_original - dice_artifact) / dice_original
    
    if target_min <= degradation_pct <= target_max:
        print(f"\n✅ SUCCESS: Degradation within target range ({target_min}-{target_max}%)")
    elif degradation_pct < target_min:
        print(f"\n⚠️  Below target: Need stronger artifacts or better placement")
    else:
        print(f"\n✅ EXCELLENT: Exceeds target range!")
    
    # Save results
    results = {
        'dice_original': float(dice_original),
        'dice_artifact': float(dice_artifact),
        'degradation_absolute': float(dice_original - dice_artifact),
        'degradation_percent': float(100 * (dice_original - dice_artifact) / dice_original),
        'phase3_degradation_percent': 1.0,
        'improvement_factor': float((100 * (dice_original - dice_artifact) / dice_original) / 1.0)
    }
    
    results_path = output_dir / "dice_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    print("\n" + "="*60)
    print("✓ Step 3 COMPLETE: Evaluation finished!")
    print("="*60)
    
    if degradation_pct >= target_min:
        print("\n🎉 Phase 4 surgical artifacts are CLINICALLY MEANINGFUL!")
        print("Ready to proceed with RL adversarial training!")
    else:
        print("\n📊 Next: Tune artifact severity or screw placement for stronger effect")


if __name__ == "__main__":
    evaluate_surgical_artifacts()

