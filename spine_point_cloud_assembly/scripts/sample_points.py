#!/usr/bin/env python3
"""
Phase 1.2: Sample point clouds from meshes

Usage:
    python scripts/sample_points.py \
        --mesh_dir outputs/meshes \
        --output_dir outputs/point_clouds \
        --num_points 2048
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

from utils.geometry import sample_point_cloud


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def sample_point_clouds_from_meshes(
    mesh_dir: Path,
    output_dir: Path,
    num_points: int = 2048,
    method: str = 'uniform',
    seed: int = 42,
):
    """
    Sample point clouds from all meshes.
    
    Args:
        mesh_dir: Directory containing extracted meshes
        output_dir: Output directory for point clouds
        num_points: Target number of points per vertebra
        method: Sampling method ('uniform' or 'poisson')
        seed: Random seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subject directories
    subject_dirs = [d for d in mesh_dir.iterdir() if d.is_dir()]
    
    logging.info(f"Found {len(subject_dirs)} subjects")
    
    stats = {
        'total_subjects': len(subject_dirs),
        'processed_subjects': 0,
        'total_vertebrae': 0,
        'sampled_vertebrae': 0,
    }
    
    # Process each subject
    for subject_dir in tqdm(subject_dirs, desc="Sampling point clouds"):
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
            # mesh_file is relative to mesh_dir, so construct full path correctly
            mesh_file_path = vertebra_info['mesh_file']
            if Path(mesh_file_path).is_absolute():
                mesh_file = Path(mesh_file_path)
            else:
                # Relative path: should be relative to mesh_dir
                mesh_file = mesh_dir / mesh_file_path
            
            if not mesh_file.exists():
                logging.warning(f"Mesh file not found: {mesh_file}")
                continue
            
            # Load mesh
            try:
                mesh_data = np.load(mesh_file)
                vertices = mesh_data['vertices']
                faces = mesh_data['faces']
            except Exception as e:
                logging.warning(f"Failed to load mesh {mesh_file}: {e}")
                continue
            
            # Validate mesh data
            if len(vertices) < 3 or len(faces) < 1:
                logging.warning(f"Mesh too small: {mesh_file} ({len(vertices)} vertices, {len(faces)} faces)")
                continue
            
            # Check for invalid face indices
            if faces.max() >= len(vertices):
                logging.warning(f"Invalid face indices in {mesh_file}: max index {faces.max()}, num vertices {len(vertices)}")
                continue
            
            stats['total_vertebrae'] += 1
            
            # Sample point cloud
            try:
                points = sample_point_cloud(
                    vertices,
                    faces,
                    num_points=num_points,
                    method=method,
                    seed=seed,
                )
                
                # Validate sampled points
                if len(points) < 10:
                    logging.warning(f"Too few points sampled from {mesh_file}: {len(points)}")
                    continue
                    
            except Exception as e:
                logging.warning(f"Failed to sample points from {mesh_file}: {e}")
                continue
            
            # Save point cloud
            pc_file = subject_output_dir / f'vertebra_{vertebra_label}_points.npy'
            np.save(pc_file, points)
            
            vertebrae_data[vertebra_label] = {
                'point_cloud_file': f'vertebra_{vertebra_label}_points.npy',
                'num_points': len(points),
            }
            
            stats['sampled_vertebrae'] += 1
        
        # Save metadata
        pc_metadata = {
            'subject_id': subject_id,
            'num_points': num_points,
            'sampling_method': method,
            'vertebrae': vertebrae_data,
        }
        
        pc_metadata_file = subject_output_dir / 'metadata.json'
        with open(pc_metadata_file, 'w') as f:
            json.dump(pc_metadata, f, indent=2)
        
        if len(vertebrae_data) > 0:
            stats['processed_subjects'] += 1
    
    # Save statistics
    stats_file = output_dir / 'sampling_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("Point Cloud Sampling Summary")
    logging.info("=" * 60)
    logging.info(f"Total subjects: {stats['total_subjects']}")
    logging.info(f"Processed subjects: {stats['processed_subjects']}")
    logging.info(f"Total vertebrae: {stats['total_vertebrae']}")
    logging.info(f"Sampled vertebrae: {stats['sampled_vertebrae']}")
    logging.info(f"Success rate: {stats['sampled_vertebrae'] / max(stats['total_vertebrae'], 1) * 100:.1f}%")
    logging.info(f"Statistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Sample point clouds from meshes')
    parser.add_argument('--mesh_dir', type=str, required=True,
                        help='Directory containing extracted meshes')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for point clouds')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Target number of points per vertebra')
    parser.add_argument('--method', type=str, default='uniform',
                        choices=['uniform', 'poisson'],
                        help='Sampling method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Resolve paths (handle relative paths)
    mesh_dir = Path(args.mesh_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not mesh_dir.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent.parent.parent
        mesh_dir_alt = (script_dir / args.mesh_dir).resolve()
        if mesh_dir_alt.exists():
            mesh_dir = mesh_dir_alt
            logging.info(f"Using alternative path: {mesh_dir}")
        else:
            raise FileNotFoundError(
                f"Mesh directory not found: {args.mesh_dir}\n"
                f"  Tried: {Path(args.mesh_dir).resolve()}\n"
                f"  Tried: {mesh_dir_alt}"
            )
    
    sample_point_clouds_from_meshes(
        mesh_dir=mesh_dir,
        output_dir=output_dir,
        num_points=args.num_points,
        method=args.method,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

