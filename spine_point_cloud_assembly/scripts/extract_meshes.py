#!/usr/bin/env python3
"""
Phase 1.1: Extract meshes from segmentation masks

Usage:
    python scripts/extract_meshes.py \
        --mask_dir VerSe/processed \
        --output_dir outputs/meshes \
        --spacing 1.0 1.0 1.0
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spine_point_cloud_assembly.utils.geometry import extract_mesh_from_mask, validate_mesh

# For loading original mask files
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

# VerSe utilities for preprocessing
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'verse' / 'utils'))
    from data_utilities import resample_nib, reorient_to
    VERSE_UTILS_AVAILABLE = True
except ImportError:
    VERSE_UTILS_AVAILABLE = False

# For loading original mask files
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("nibabel not available. Cannot load original mask files.")

# VerSe utilities for preprocessing
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'verse' / 'utils'))
    from data_utilities import resample_nib, reorient_to
    VERSE_UTILS_AVAILABLE = True
except ImportError:
    VERSE_UTILS_AVAILABLE = False
    logging.warning("VerSe utilities not available. Cannot preprocess original masks.")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def extract_meshes_from_dataset(
    mask_dir: Path,
    output_dir: Path,
    spacing: tuple = (1.0, 1.0, 1.0),
    split: str = 'train',
    sample_fraction: float = 1.0,
):
    """
    Extract meshes from all vertebra masks in the dataset.
    
    Args:
        mask_dir: Directory containing processed data
        output_dir: Output directory for meshes
        spacing: Voxel spacing (dz, dy, dx)
        split: Dataset split ('train', 'val', 'test')
        sample_fraction: Fraction of data to process (for testing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subject directories with mask files
    mask_files = []
    for dataset_dir in mask_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        # Check if this matches the split (rough heuristic)
        if split == 'train' and 'training' not in dataset_dir.name.lower() and 'train' not in dataset_dir.name.lower():
            continue
        if split == 'val' and 'validation' not in dataset_dir.name.lower() and 'val' not in dataset_dir.name.lower():
            continue
        if split == 'test' and 'test' not in dataset_dir.name.lower():
            continue
        
        # Find all subject directories
        for subject_dir in dataset_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            
            mask_file = subject_dir / 'mask_volume_1mm.npy'
            if mask_file.exists():
                mask_files.append({
                    'mask_path': mask_file,
                    'subject_id': subject_dir.name,
                    'dataset': dataset_dir.name,
                })
    
    # Apply sample fraction
    if sample_fraction < 1.0:
        import random
        random.seed(42)
        mask_files = random.sample(mask_files, int(len(mask_files) * sample_fraction))
    
    if len(mask_files) == 0:
        logging.error("No mask files found! Check:")
        logging.error("  1. Data path is correct")
        logging.error("  2. Split filter is correct")
        logging.error("  3. mask_volume_1mm.npy files exist")
        return
    
    logging.info(f"Found {len(mask_files)} mask files to process")
    
    # Statistics
    stats = {
        'total_samples': len(mask_files),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'total_vertebrae': 0,
        'extracted_vertebrae': 0,
        'skipped_no_mask': 0,
    }
    
    # Process each sample
    for sample_info in tqdm(mask_files, desc="Extracting meshes"):
        mask_path = sample_info['mask_path']
        subject_id = sample_info['subject_id']
        dataset_name = sample_info['dataset']
        
        # Load mask
        try:
            mask_volume = np.load(mask_path)
        except Exception as e:
            logging.warning(f"Failed to load mask for {subject_id}: {e}")
            stats['skipped_no_mask'] += 1
            continue
        
        # Create output directory for this subject
        subject_output_dir = output_dir / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract unique vertebra labels
        unique_labels = np.unique(mask_volume)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        stats['total_vertebrae'] += len(unique_labels)
        
        # Extract mesh for each vertebra
        vertebra_meshes = {}
        for label in unique_labels:
            # Create binary mask for this vertebra
            vertebra_mask = (mask_volume == label).astype(np.float32)
            
            # Check if mask has enough voxels (minimum threshold)
            mask_voxels = vertebra_mask.sum()
            if mask_voxels < 10:  # Too small to extract meaningful mesh
                logging.debug(f"Skipping vertebra {label} for {subject_id}: too small ({mask_voxels} voxels)")
                stats['failed_extractions'] += 1
                continue
            
            # Extract mesh
            try:
                mesh_result = extract_mesh_from_mask(
                    vertebra_mask,
                    spacing=spacing,
                    level=0.5,
                )
            except Exception as e:
                logging.warning(f"Exception extracting mesh for {subject_id}, vertebra {label}: {e}")
                stats['failed_extractions'] += 1
                continue
            
            if mesh_result is None:
                logging.debug(f"Failed to extract mesh for {subject_id}, vertebra {label}")
                stats['failed_extractions'] += 1
                continue
            
            # Check mesh has minimum vertices/faces
            if len(mesh_result['vertices']) < 4 or len(mesh_result['faces']) < 2:
                logging.debug(f"Mesh too small for {subject_id}, vertebra {label}: {len(mesh_result['vertices'])} vertices, {len(mesh_result['faces'])} faces")
                stats['failed_extractions'] += 1
                continue
            
            # Validate mesh
            is_valid, issues = validate_mesh(
                mesh_result['vertices'],
                mesh_result['faces'],
            )
            
            if not is_valid:
                logging.debug(f"Invalid mesh for {subject_id}, vertebra {label}: {issues}")
                stats['failed_extractions'] += 1
                continue
            
            # Save mesh
            mesh_file = subject_output_dir / f'vertebra_{int(label)}_mesh.npz'
            np.savez(
                mesh_file,
                vertices=mesh_result['vertices'],
                faces=mesh_result['faces'],
                normals=mesh_result['normals'],
                label=int(label),
            )
            
            vertebra_meshes[int(label)] = {
                'mesh_file': str(mesh_file.relative_to(output_dir)),
                'num_vertices': len(mesh_result['vertices']),
                'num_faces': len(mesh_result['faces']),
            }
            
            stats['extracted_vertebrae'] += 1
        
        # Save metadata
        metadata = {
            'subject_id': subject_id,
            'volume_shape': list(mask_volume.shape),
            'spacing': spacing,
            'vertebrae': vertebra_meshes,
        }
        
        metadata_file = subject_output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if len(vertebra_meshes) > 0:
            stats['successful_extractions'] += 1
        else:
            stats['failed_extractions'] += 1
    
    # Save overall statistics
    stats_file = output_dir / 'extraction_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("Mesh Extraction Summary")
    logging.info("=" * 60)
    logging.info(f"Total samples: {stats['total_samples']}")
    logging.info(f"Skipped (no mask): {stats['skipped_no_mask']}")
    logging.info(f"Successful extractions: {stats['successful_extractions']}")
    logging.info(f"Failed extractions: {stats['failed_extractions']}")
    logging.info(f"Total vertebrae: {stats['total_vertebrae']}")
    logging.info(f"Extracted vertebrae: {stats['extracted_vertebrae']}")
    if stats['total_vertebrae'] > 0:
        logging.info(f"Success rate: {stats['extracted_vertebrae'] / stats['total_vertebrae'] * 100:.1f}%")
    else:
        logging.warning("No vertebrae processed! Check if mask files exist.")
    logging.info(f"Statistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract meshes from segmentation masks')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Directory containing processed data (e.g., VerSe/processed)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for meshes')
    parser.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Voxel spacing (dz, dy, dx)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='Fraction of data to process (for testing)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Resolve paths (handle relative paths)
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    spacing = tuple(args.spacing)
    
    if not mask_dir.exists():
        # Try relative to project root (vindr directory)
        # Script is in spine_point_cloud_assembly/scripts/
        # Project root is spine_point_cloud_assembly/../ (vindr/)
        project_root = Path(__file__).parent.parent.parent
        mask_dir_alt = (project_root / args.mask_dir).resolve()
        if mask_dir_alt.exists():
            mask_dir = mask_dir_alt
            logging.info(f"Using alternative path: {mask_dir}")
        else:
            # Try as-is if it's already an absolute path
            if Path(args.mask_dir).is_absolute():
                mask_dir = Path(args.mask_dir)
            else:
                raise FileNotFoundError(
                    f"Mask directory not found: {args.mask_dir}\n"
                    f"  Tried: {Path(args.mask_dir).resolve()}\n"
                    f"  Tried: {mask_dir_alt}\n"
                    f"  Project root: {project_root}"
                )
    
    extract_meshes_from_dataset(
        mask_dir=mask_dir,
        output_dir=output_dir,
        spacing=spacing,
        split=args.split,
        sample_fraction=args.sample_fraction,
    )


if __name__ == '__main__':
    main()

