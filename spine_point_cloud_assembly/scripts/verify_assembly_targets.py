"""
Verify assembly training targets are correctly computed from original (non-centered) points.
This checks if the translation targets are reasonable for proper reconstruction.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.assembly_data_loader import AssemblyDataset


def verify_targets(
    embedding_dir: Path,
    point_cloud_dir: Path,
    num_samples: int = 5,
):
    """Verify assembly targets are correctly computed"""
    
    print("="*80)
    print("Assembly Target Verification")
    print("="*80)
    print(f"Embedding directory: {embedding_dir}")
    print(f"Point cloud directory: {point_cloud_dir}")
    print(f"Checking {num_samples} samples...")
    print("="*80)
    print()
    
    # Create dataset
    dataset = AssemblyDataset(
        embedding_dir=embedding_dir,
        point_cloud_dir=point_cloud_dir,
        split='train',
        max_vertebrae=30,
        augment=False,
    )
    
    if len(dataset) == 0:
        print("ERROR: No samples found in dataset!")
        return
    
    # Check multiple samples
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        subject_id = sample['subject_id']
        points = sample['points']  # (N, M, 3) - original points
        mask = sample['mask']  # (N,)
        translation = sample['assembly']['translation']  # (N, 3)
        
        # Filter valid vertebrae
        valid_indices = torch.where(mask)[0]
        points_valid = points[valid_indices]  # (K, M, 3)
        translation_valid = translation[valid_indices]  # (K, 3)
        
        print(f"\n{'='*80}")
        print(f"Sample {idx+1}: {subject_id}")
        print(f"{'='*80}")
        print(f"Number of valid vertebrae: {len(valid_indices)}")
        
        # Compute centroids from original points
        centroids = points_valid.mean(dim=1)  # (K, 3) - centroids of original points
        first_centroid = centroids[0]  # (3,)
        
        # Expected translation (should match what's in sample)
        expected_translation = centroids - first_centroid.unsqueeze(0)  # (K, 3)
        
        # Point cloud statistics
        point_cloud_sizes = []
        point_cloud_ranges = []
        for i in range(len(points_valid)):
            pc = points_valid[i].numpy()
            size = np.linalg.norm(pc.max(axis=0) - pc.min(axis=0))
            point_cloud_sizes.append(size)
            point_cloud_ranges.append((pc.min(axis=0), pc.max(axis=0)))
        
        print(f"\n[Point Cloud Statistics]")
        print(f"  Point cloud sizes (diameter): {[f'{s:.2f}' for s in point_cloud_sizes]}")
        print(f"  Mean point cloud size: {np.mean(point_cloud_sizes):.2f}")
        
        print(f"\n[Centroids (Original Absolute Coordinates)]")
        for i, (v_idx, centroid) in enumerate(zip(valid_indices, centroids)):
            print(f"  Vertebra {v_idx}: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        
        print(f"\n[Translation Targets (Relative to First Vertebra)]")
        translation_magnitudes = torch.norm(translation_valid, dim=1).numpy()
        for i, (v_idx, trans, mag) in enumerate(zip(valid_indices, translation_valid, translation_magnitudes)):
            print(f"  Vertebra {v_idx}: ({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}) | magnitude: {mag:.2f}")
        
        print(f"\n[Translation Verification]")
        translation_diff = torch.abs(translation_valid - expected_translation)
        max_diff = translation_diff.max().item()
        mean_diff = translation_diff.mean().item()
        print(f"  Max difference between computed and stored: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        if max_diff > 1e-5:
            print(f"  ⚠️ WARNING: Translation mismatch detected!")
        else:
            print(f"  ✓ Translation targets match expected values")
        
        print(f"\n[Translation vs Point Cloud Size]")
        for i, (v_idx, mag, pc_size) in enumerate(zip(valid_indices, translation_magnitudes, point_cloud_sizes)):
            ratio = mag / (pc_size + 1e-6)
            print(f"  Vertebra {v_idx}: translation={mag:.2f}, pc_size={pc_size:.2f}, ratio={ratio:.3f}")
            if i > 0 and ratio < 0.1:
                print(f"    ⚠️ WARNING: Translation is very small compared to point cloud size!")
        
        # Check if translations form a reasonable spine shape
        print(f"\n[Spine Shape Check]")
        # Compute vertical spread (assuming Y or Z is vertical)
        # Check which axis has the largest spread
        axis_spreads = []
        for axis in range(3):
            spread = (centroids[:, axis].max() - centroids[:, axis].min()).item()
            axis_spreads.append(spread)
        
        max_axis = np.argmax(axis_spreads)
        axis_names = ['X', 'Y', 'Z']
        print(f"  Largest spread axis: {axis_names[max_axis]} ({axis_spreads[max_axis]:.2f})")
        print(f"  Axis spreads: X={axis_spreads[0]:.2f}, Y={axis_spreads[1]:.2f}, Z={axis_spreads[2]:.2f}")
        
        # Check if translations are reasonable (should be similar to point cloud size for adjacent vertebrae)
        if len(translation_magnitudes) > 1:
            adjacent_diffs = np.diff(translation_magnitudes)
            print(f"  Adjacent vertebra translation differences: {adjacent_diffs}")
            print(f"  Mean adjacent difference: {np.mean(np.abs(adjacent_diffs)):.2f}")
            
            # Typical vertebra height is ~20-30mm, so translations should be in that range
            if np.mean(translation_magnitudes[1:]) < 5.0:
                print(f"  ⚠️ WARNING: Translation magnitudes are very small (<5mm)")
                print(f"     Expected: ~20-30mm for typical vertebra spacing")
            elif np.mean(translation_magnitudes[1:]) > 200.0:
                print(f"  ⚠️ WARNING: Translation magnitudes are very large (>200mm)")
            else:
                print(f"  ✓ Translation magnitudes are in reasonable range")
        
        # Check first vertebra translation (should be ~0)
        first_trans_mag = translation_magnitudes[0]
        if first_trans_mag > 1e-3:
            print(f"  ⚠️ WARNING: First vertebra translation is not zero: {first_trans_mag:.6f}")
        else:
            print(f"  ✓ First vertebra translation is zero (as expected)")
    
    print(f"\n{'='*80}")
    print("Verification Complete!")
    print(f"{'='*80}")
    print("\nSummary:")
    print("  - If translation magnitudes are ~20-30mm: ✓ Good")
    print("  - If translation magnitudes are <5mm: ⚠️ Too small (may cause overlap)")
    print("  - If translation magnitudes are >200mm: ⚠️ Too large (may be incorrect)")
    print("  - If first vertebra translation is not zero: ⚠️ Check target computation")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Verify assembly training targets')
    parser.add_argument('--embedding_dir', type=str, required=True,
                        help='Directory containing pre-extracted embeddings')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing original (non-centered) point clouds')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to check')
    
    args = parser.parse_args()
    
    verify_targets(
        embedding_dir=Path(args.embedding_dir),
        point_cloud_dir=Path(args.point_cloud_dir),
        num_samples=args.num_samples,
    )


if __name__ == '__main__':
    main()

