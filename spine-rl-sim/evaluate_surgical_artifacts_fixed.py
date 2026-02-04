"""
Fixed evaluation: Find best matching vertebra between TS and GT.
VerSe uses caudal-to-cranial numbering which may not match TS.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import center_of_mass
import json


def find_best_matching_vertebra(ts_dir, gt_mask, gt_label=22):
    """
    Find which TS vertebra best matches GT label.
    """
    gt_binary = (gt_mask == gt_label)
    gt_com = np.array(center_of_mass(gt_binary))
    
    vertebra_files = sorted(ts_dir.glob("vertebrae_*.nii.gz"))
    
    best_match = None
    best_dist = float('inf')
    
    print(f"\nSearching for best match to GT label {gt_label} (L1)...")
    print(f"GT center of mass: {gt_com}")
    print()
    
    for vfile in vertebra_files:
        pred_data = nib.load(str(vfile)).get_fdata() > 0
        if pred_data.sum() == 0:
            continue
            
        pred_com = np.array(center_of_mass(pred_data))
        dist = np.linalg.norm(pred_com - gt_com)
        
        if dist < 50:  # Only consider nearby vertebrae
            print(f"  {vfile.name:25s} COM={pred_com}  dist={dist:.1f}")
        
        if dist < best_dist:
            best_dist = dist
            best_match = vfile
    
    print(f"\n✓ Best match: {best_match.name} (distance: {best_dist:.1f} voxels)")
    return best_match


def compute_dice_direct(pred_path, gt_mask, gt_label):
    """
    Compute Dice using volumetric overlap (no resampling).
    """
    pred_data = nib.load(str(pred_path)).get_fdata() > 0
    gt_binary = (gt_mask == gt_label)
    
    # Get bounding boxes
    pred_coords = np.argwhere(pred_data)
    gt_coords = np.argwhere(gt_binary)
    
    # Find overlap region
    pred_min, pred_max = pred_coords.min(axis=0), pred_coords.max(axis=0)
    gt_min, gt_max = gt_coords.min(axis=0), gt_coords.max(axis=0)
    
    overlap_min = np.maximum(pred_min, gt_min)
    overlap_max = np.minimum(pred_max, gt_max)
    
    # Check if there's overlap
    if (overlap_min >= overlap_max).any():
        return 0.0
    
    # Compute Dice in overlap region
    pred_crop = pred_data[
        overlap_min[0]:overlap_max[0]+1,
        overlap_min[1]:overlap_max[1]+1,
        overlap_min[2]:overlap_max[2]+1
    ]
    gt_crop = gt_binary[
        overlap_min[0]:overlap_max[0]+1,
        overlap_min[1]:overlap_max[1]+1,
        overlap_min[2]:overlap_max[2]+1
    ]
    
    intersection = (pred_crop & gt_crop).sum()
    union = pred_data.sum() + gt_binary.sum()
    
    if union == 0:
        return 1.0
    
    dice = 2.0 * intersection / union
    return dice


def main():
    # Paths
    gt_mask_path = Path("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
    ts_original_dir = Path("outputs/phase4_surgical_artifacts/evaluation/ts_original")
    ts_artifact_dir = Path("outputs/phase4_surgical_artifacts/evaluation/ts_artifact")
    output_dir = Path("outputs/phase4_surgical_artifacts/evaluation")
    
    # Load GT
    print("Loading ground truth...")
    gt_mask = nib.load(str(gt_mask_path)).get_fdata()
    
    # Find best matching vertebra
    best_orig = find_best_matching_vertebra(ts_original_dir, gt_mask, gt_label=22)
    best_artifact = find_best_matching_vertebra(ts_artifact_dir, gt_mask, gt_label=22)
    
    # Compute Dice
    print("\nComputing Dice scores...")
    dice_orig = compute_dice_direct(best_orig, gt_mask, 22)
    dice_artifact = compute_dice_direct(best_artifact, gt_mask, 22)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS: Phase 4 Surgical Artifacts")
    print("="*70)
    print(f"\nL1 Vertebra Dice Scores:")
    print(f"  Original CT:       {dice_orig:.4f}  (matched to {best_orig.name})")
    print(f"  Artifact CT:       {dice_artifact:.4f}  (matched to {best_artifact.name})")
    print(f"  Degradation:       {(dice_orig - dice_artifact):.4f}")
    print(f"  Degradation (%):   {100 * (dice_orig - dice_artifact) / dice_orig:.2f}%")
    
    # Compare with Phase 3
    print(f"\n" + "-"*70)
    print("Comparison with Phase 3 (Fracture):")
    print(f"  Phase 3 degradation:  0.33-1.00%")
    degradation_pct = 100 * (dice_orig - dice_artifact) / dice_orig
    print(f"  Phase 4 degradation:  {degradation_pct:.2f}%")
    if dice_orig > 0:
        print(f"  Improvement factor:   {degradation_pct / 1.0:.1f}x")
    
    # Target check
    target_min, target_max = 20.0, 30.0
    
    if dice_orig > 0 and target_min <= degradation_pct <= target_max:
        print(f"\n✅ SUCCESS: Degradation within target range ({target_min}-{target_max}%)")
    elif dice_orig > 0 and degradation_pct > target_max:
        print(f"\n✅ EXCELLENT: Exceeds target range ({degradation_pct:.1f}%)!")
    elif dice_orig > 0 and degradation_pct > 0:
        print(f"\n⚠️  Below target: Need stronger artifacts ({degradation_pct:.1f}% < {target_min}%)")
    else:
        print(f"\n⚠️  Poor baseline Dice: {dice_orig:.4f}")
    
    # Save results
    results = {
        'dice_original': float(dice_orig),
        'dice_artifact': float(dice_artifact),
        'degradation_absolute': float(dice_orig - dice_artifact),
        'degradation_percent': float(degradation_pct) if dice_orig > 0 else None,
        'matched_vertebra_original': best_orig.name,
        'matched_vertebra_artifact': best_artifact.name,
        'phase3_degradation_percent': 1.0
    }
    
    results_path = output_dir / "dice_comparison_fixed.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    print("="*70)


if __name__ == "__main__":
    main()

