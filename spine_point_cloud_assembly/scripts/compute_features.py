#!/usr/bin/env python3
"""
Phase 1.3: Compute directional features for point clouds

Usage:
    python scripts/compute_features.py \
        --point_cloud_dir outputs/point_clouds \
        --output_dir outputs/point_clouds \
        --compute_normals \
        --compute_curvature
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.features import (
    compute_surface_normals,
    compute_curvature as compute_curvature_fn,  # Rename to avoid shadowing
)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def compute_features_for_point_clouds(
    point_cloud_dir: Path,
    output_dir: Path,
    compute_normals: bool = True,
    compute_curvature: bool = False,
    k_nn: int = 20,
):
    """
    Compute directional features for all point clouds.
    
    Args:
        point_cloud_dir: Directory containing point clouds
        output_dir: Output directory (can be same as input)
        compute_normals: Whether to compute surface normals
        compute_curvature: Whether to compute curvature
        k_nn: Number of nearest neighbors for feature computation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subject directories
    subject_dirs = [d for d in point_cloud_dir.iterdir() if d.is_dir()]
    
    logging.info(f"Found {len(subject_dirs)} subjects")
    
    stats = {
        'total_subjects': len(subject_dirs),
        'processed_subjects': 0,
        'total_vertebrae': 0,
        'processed_vertebrae': 0,
    }
    
    # Process each subject
    for subject_dir in tqdm(subject_dirs, desc="Computing features"):
        subject_id = subject_dir.name
        
        # Load metadata
        metadata_file = subject_dir / 'metadata.json'
        if not metadata_file.exists():
            logging.warning(f"No metadata found for {subject_id}, skipping")
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create output directory
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each vertebra
        vertebrae_data = {}
        for vertebra_label, vertebra_info in metadata['vertebrae'].items():
            pc_file = subject_dir / vertebra_info['point_cloud_file']
            
            if not pc_file.exists():
                logging.warning(f"Point cloud file not found: {pc_file}")
                continue
            
            # Load point cloud
            try:
                points = np.load(pc_file)
            except Exception as e:
                logging.warning(f"Failed to load point cloud {pc_file}: {e}")
                continue
            
            # Validate point cloud
            if len(points) < 10:
                logging.warning(f"Point cloud too small: {pc_file} ({len(points)} points)")
                continue
            
            if points.shape[1] != 3:
                logging.warning(f"Invalid point cloud shape: {pc_file} (shape: {points.shape})")
                continue
            
            stats['total_vertebrae'] += 1
            
            # Compute features
            features_dict = {}
            
            if compute_normals:
                try:
                    normals = compute_surface_normals(points, k=k_nn)
                    features_dict['normals'] = normals
                    
                    # Save normals
                    normals_file = subject_output_dir / f'vertebra_{vertebra_label}_normals.npy'
                    np.save(normals_file, normals)
                except Exception as e:
                    logging.error(f"Failed to compute normals for {pc_file}: {e}")
            
            if compute_curvature:
                try:
                    normals = features_dict.get('normals')
                    if normals is None:
                        normals = compute_surface_normals(points, k=k_nn)
                    
                    # Ensure k_nn is an integer and validate
                    k_nn_int = int(k_nn)
                    if k_nn_int < 1:
                        k_nn_int = 1
                    if k_nn_int >= len(points):
                        k_nn_int = max(1, len(points) - 1)
                    
                    # Validate inputs before calling
                    if len(points) < 3:
                        logging.warning(f"Too few points for curvature: {pc_file} ({len(points)} points)")
                    elif normals.shape != points.shape:
                        logging.warning(f"Normals shape mismatch: {pc_file}")
                    else:
                        k1, k2, d1 = compute_curvature_fn(points, normals=normals, k=k_nn_int)
                        features_dict['k1'] = k1
                        features_dict['k2'] = k2
                        features_dict['d1'] = d1
                        
                        # Save curvature
                        curvature_file = subject_output_dir / f'vertebra_{vertebra_label}_curvature.npz'
                        np.savez(curvature_file, k1=k1, k2=k2, d1=d1)
                except Exception as e:
                    logging.error(f"Failed to compute curvature for {pc_file}: {e}")
                    import traceback
                    # Log full traceback to find exact error location
                    error_trace = traceback.format_exc()
                    logging.error(f"Traceback: {error_trace}")
                    # Check if it's the 'bool' object is not callable error
                    if "'bool' object is not callable" in str(e):
                        logging.error(f"Possible variable shadowing issue. Points shape: {points.shape}, k_nn: {k_nn}, k_nn_int: {k_nn_int}")
            
            # Save combined features
            if features_dict:
                features_file = subject_output_dir / f'vertebra_{vertebra_label}_features.npz'
                np.savez(features_file, **features_dict)
            
            vertebrae_data[vertebra_label] = {
                **vertebra_info,
                'has_normals': compute_normals and 'normals' in features_dict,
                'has_curvature': compute_curvature and 'k1' in features_dict,
            }
            
            stats['processed_vertebrae'] += 1
        
        # Update metadata
        metadata['vertebrae'] = vertebrae_data
        metadata['features_computed'] = {
            'normals': compute_normals,
            'curvature': compute_curvature,
            'k_nn': k_nn,
        }
        
        metadata_file = subject_output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if len(vertebrae_data) > 0:
            stats['processed_subjects'] += 1
    
    # Save statistics
    stats_file = output_dir / 'features_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("Feature Computation Summary")
    logging.info("=" * 60)
    logging.info(f"Total subjects: {stats['total_subjects']}")
    logging.info(f"Processed subjects: {stats['processed_subjects']}")
    logging.info(f"Total vertebrae: {stats['total_vertebrae']}")
    logging.info(f"Processed vertebrae: {stats['processed_vertebrae']}")
    logging.info(f"Success rate: {stats['processed_vertebrae'] / max(stats['total_vertebrae'], 1) * 100:.1f}%")
    logging.info(f"Statistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Compute directional features for point clouds')
    parser.add_argument('--point_cloud_dir', type=str, required=True,
                        help='Directory containing point clouds')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (can be same as input)')
    parser.add_argument('--compute_normals', action='store_true',
                        help='Compute surface normals')
    parser.add_argument('--compute_curvature', action='store_true',
                        help='Compute curvature')
    parser.add_argument('--k_nn', type=int, default=20,
                        help='Number of nearest neighbors for feature computation')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Resolve paths (handle relative paths)
    point_cloud_dir = Path(args.point_cloud_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not point_cloud_dir.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent.parent.parent
        point_cloud_dir_alt = (script_dir / args.point_cloud_dir).resolve()
        if point_cloud_dir_alt.exists():
            point_cloud_dir = point_cloud_dir_alt
            logging.info(f"Using alternative path: {point_cloud_dir}")
        else:
            raise FileNotFoundError(
                f"Point cloud directory not found: {args.point_cloud_dir}\n"
                f"  Tried: {Path(args.point_cloud_dir).resolve()}\n"
                f"  Tried: {point_cloud_dir_alt}"
            )
    
    if not args.compute_normals and not args.compute_curvature:
        logging.warning("No features selected to compute. Use --compute_normals or --compute_curvature")
        return
    
    compute_features_for_point_clouds(
        point_cloud_dir=point_cloud_dir,
        output_dir=output_dir,
        compute_normals=args.compute_normals,
        compute_curvature=args.compute_curvature,
        k_nn=args.k_nn,
    )


if __name__ == '__main__':
    main()

