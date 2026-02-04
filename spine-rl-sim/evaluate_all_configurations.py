"""
Evaluate all surgical configurations with artifacts.

Tests:
1. Config 1: L1 screws only
2. Config 2: L1 screws + rod
3. Config 3: L1+L2 multi-level

For each: Generate artifacts → Run TS → Measure Dice
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import subprocess
import json
from scipy.ndimage import center_of_mass
import sys
sys.path.append('spine-rl-sim')
from synthesize_surgical_artifacts import synthesize_surgical_artifacts

# Import project configuration
try:
    from config import (
        get_verse_ct_path, get_verse_seg_path, get_phase4_output_path,
        DEFAULT_SUBJECT
    )
    USE_CONFIG = True
except ImportError:
    print("⚠ Warning: config.py not found, using hardcoded paths")
    USE_CONFIG = False


def run_totalsegmentator(ct_path, output_dir):
    """Run TotalSegmentator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(output_dir), "--fast"]
    
    print(f"Running TotalSegmentator on {ct_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("TotalSegmentator failed")
    
    print("✓ Complete!")
    return output_dir


def find_best_matching_vertebra(ts_dir, gt_mask, gt_label):
    """Find which TS vertebra best matches GT label."""
    gt_binary = (gt_mask == gt_label)
    if gt_binary.sum() == 0:
        return None, float('inf')
    
    gt_com = np.array(center_of_mass(gt_binary))
    
    vertebra_files = sorted(ts_dir.glob("vertebrae_*.nii.gz"))
    
    best_match = None
    best_dist = float('inf')
    
    for vfile in vertebra_files:
        pred_data = nib.load(str(vfile)).get_fdata() > 0
        if pred_data.sum() == 0:
            continue
            
        pred_com = np.array(center_of_mass(pred_data))
        dist = np.linalg.norm(pred_com - gt_com)
        
        if dist < best_dist:
            best_dist = dist
            best_match = vfile
    
    return best_match, best_dist


def compute_dice_direct(pred_path, gt_mask, gt_label):
    """Compute Dice with bounding box overlap."""
    if pred_path is None:
        return 0.0
    
    pred_data = nib.load(str(pred_path)).get_fdata() > 0
    gt_binary = (gt_mask == gt_label)
    
    pred_coords = np.argwhere(pred_data)
    gt_coords = np.argwhere(gt_binary)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0
    
    pred_min, pred_max = pred_coords.min(axis=0), pred_coords.max(axis=0)
    gt_min, gt_max = gt_coords.min(axis=0), gt_coords.max(axis=0)
    
    overlap_min = np.maximum(pred_min, gt_min)
    overlap_max = np.minimum(pred_max, gt_max)
    
    if (overlap_min >= overlap_max).any():
        return 0.0
    
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


def evaluate_configuration(config_name, metal_mask_path, severity='moderate'):
    """Evaluate one configuration."""
    print("\n" + "="*70)
    print(f"Evaluating: {config_name} (severity={severity})")
    print("="*70)
    
    # Paths (use config if available, otherwise fallback to hardcoded)
    if USE_CONFIG:
        ct_original_path = get_verse_ct_path(DEFAULT_SUBJECT)
        gt_mask_path = get_verse_seg_path(DEFAULT_SUBJECT)
        output_dir = get_phase4_output_path(f"evaluation_{severity}")
    else:
        ct_original_path = Path("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
        gt_mask_path = Path("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
        output_dir = Path(f"outputs/phase4_surgical_artifacts/evaluation_{severity}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ct_nii = nib.load(str(ct_original_path))
    ct_original = ct_nii.get_fdata()
    
    metal_mask = nib.load(str(metal_mask_path)).get_fdata().astype(bool)
    gt_mask = nib.load(str(gt_mask_path)).get_fdata()
    
    # Generate artifact CT
    print(f"Generating {severity} artifacts...")
    ct_artifact = synthesize_surgical_artifacts(ct_original, metal_mask, severity=severity)
    
    # Save
    artifact_ct_path = output_dir / f"{config_name}_ct.nii.gz"
    artifact_nii = nib.Nifti1Image(ct_artifact.astype(np.float32), ct_nii.affine)
    nib.save(artifact_nii, str(artifact_ct_path))
    print(f"✓ Artifact CT saved: {artifact_ct_path}")
    
    # Run TotalSegmentator
    ts_output_dir = output_dir / f"{config_name}_ts"
    if not ts_output_dir.exists():
        run_totalsegmentator(artifact_ct_path, ts_output_dir)
    else:
        print(f"Using existing TS predictions: {ts_output_dir}")
    
    # Evaluate L1 and L2
    results = {}
    
    for label_id, label_name in [(22, 'L1'), (23, 'L2')]:
        best_match, dist = find_best_matching_vertebra(ts_output_dir, gt_mask, label_id)
        
        if best_match and dist < 50:
            dice = compute_dice_direct(best_match, gt_mask, label_id)
            results[label_name] = {
                'dice': float(dice),
                'matched_to': best_match.name,
                'distance': float(dist)
            }
            print(f"  {label_name}: Dice = {dice:.4f} (matched to {best_match.name})")
        else:
            results[label_name] = {
                'dice': 0.0,
                'matched_to': None,
                'distance': float('inf')
            }
            print(f"  {label_name}: Not found")
    
    # Save results
    results_path = output_dir / f"{config_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """
    Evaluate all configurations with moderate severity.
    """
    # Get baseline (original CT, no hardware)
    print("="*70)
    print("Getting BASELINE (Original CT, no hardware)")
    print("="*70)
    
    # Paths (use config if available, otherwise fallback to hardcoded)
    if USE_CONFIG:
        ct_original_path = get_verse_ct_path(DEFAULT_SUBJECT)
        gt_mask_path = get_verse_seg_path(DEFAULT_SUBJECT)
        baseline_ts_dir = get_phase4_output_path("evaluation") / "ts_original"
    else:
        ct_original_path = Path("VerSe/dataset-03test/rawdata/sub-verse563/sub-verse563_dir-iso_ct.nii.gz")
        gt_mask_path = Path("VerSe/dataset-03test/derivatives/sub-verse563/sub-verse563_dir-iso_seg-vert_msk.nii.gz")
        baseline_ts_dir = Path("outputs/phase4_surgical_artifacts/evaluation/ts_original")
    
    # Use existing baseline if available
    
    if not baseline_ts_dir.exists():
        print("Running TotalSegmentator on original CT...")
        baseline_ts_dir = run_totalsegmentator(ct_original_path, baseline_ts_dir)
    else:
        print(f"Using existing baseline: {baseline_ts_dir}")
    
    gt_mask = nib.load(str(gt_mask_path)).get_fdata()
    
    baseline_results = {}
    for label_id, label_name in [(22, 'L1'), (23, 'L2')]:
        best_match, dist = find_best_matching_vertebra(baseline_ts_dir, gt_mask, label_id)
        if best_match and dist < 50:
            dice = compute_dice_direct(best_match, gt_mask, label_id)
            baseline_results[label_name] = float(dice)
            print(f"  {label_name}: Dice = {dice:.4f}")
        else:
            baseline_results[label_name] = 0.0
            print(f"  {label_name}: Not found")
    
    print()
    
    # Configurations to test
    configs = [
        ("config1_L1_screws_only", "outputs/phase4_surgical_artifacts/configurations/config1_L1_screws_only.nii.gz"),
        ("config2_L1_screws_rod", "outputs/phase4_surgical_artifacts/configurations/config2_L1_screws_rod.nii.gz"),
        ("config3_multi_level", "outputs/phase4_surgical_artifacts/configurations/config3_multi_level.nii.gz"),
    ]
    
    severity = 'moderate'
    all_results = {'baseline': baseline_results}
    
    for config_name, metal_mask_path in configs:
        results = evaluate_configuration(config_name, metal_mask_path, severity=severity)
        all_results[config_name] = results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Configuration Comparison")
    print("="*70)
    print(f"\nBaseline (No hardware):")
    for vname, dice in baseline_results.items():
        print(f"  {vname}: {dice:.4f}")
    
    print()
    
    for config_name in ['config1_L1_screws_only', 'config2_L1_screws_rod', 'config3_multi_level']:
        config_results = all_results[config_name]
        
        print(f"\n{config_name}:")
        for vname in ['L1', 'L2']:
            if vname in config_results:
                dice = config_results[vname]['dice']
                baseline_dice = baseline_results.get(vname, 0.0)
                
                if baseline_dice > 0:
                    degradation_pct = 100 * (baseline_dice - dice) / baseline_dice
                    print(f"  {vname}: {dice:.4f} (degradation: {degradation_pct:.2f}%)")
                else:
                    print(f"  {vname}: {dice:.4f} (no baseline)")
    
    # Save summary
    summary_path = Path(f"outputs/phase4_surgical_artifacts/evaluation_{severity}/SUMMARY.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    main()


