#!/usr/bin/env python3
"""
Create simplified CSV for Phase 1 Point Cloud Assembly

This script creates a single CSV file with only the paths needed for Phase 1:
- mask_volume_1mm.npy (required for mesh extraction)
- ct_volume_1mm.npy (optional, for visualization)
- vertebra_centroids.npy (for ROI info)
- vertebra_present.npy (for vertebra existence)

Output CSV columns:
- subject_id
- dataset
- mask_path (mask_volume_1mm.npy)
- ct_path (ct_volume_1mm.npy, optional)
- centroids_path (vertebra_centroids.npy)
- present_path (vertebra_present.npy)
- has_mask
- has_ct
- num_vertebrae
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys


def find_phase1_files(processed_dir, dataset, subject_id):
    """Find Phase 1 required files for a subject"""
    subject_dir = processed_dir / dataset / subject_id
    
    if not subject_dir.exists():
        return None
    
    files = {
        'mask_path': '',
        'ct_path': '',
        'centroids_path': '',
        'present_path': '',
        'has_mask': False,
        'has_ct': False,
    }
    
    # Required files
    mask_file = subject_dir / 'mask_volume_1mm.npy'
    if mask_file.exists():
        files['mask_path'] = str(mask_file.absolute())
        files['has_mask'] = True
    
    # Optional files
    ct_file = subject_dir / 'ct_volume_1mm.npy'
    if ct_file.exists():
        files['ct_path'] = str(ct_file.absolute())
        files['has_ct'] = True
    
    centroids_file = subject_dir / 'vertebra_centroids.npy'
    if centroids_file.exists():
        files['centroids_path'] = str(centroids_file.absolute())
    
    present_file = subject_dir / 'vertebra_present.npy'
    if present_file.exists():
        files['present_path'] = str(present_file.absolute())
    
    # Only return if at least mask exists
    if files['has_mask']:
        return files
    return None


def create_phase1_csv(input_csv_path, processed_dir, output_csv_path):
    """Create simplified Phase 1 CSV"""
    
    print("="*60)
    print("Creating Phase 1 CSV")
    print("="*60)
    print(f"Input CSV: {input_csv_path}")
    print(f"Processed directory: {processed_dir}")
    print(f"Output CSV: {output_csv_path}")
    print()
    
    # Load input CSV
    if not Path(input_csv_path).exists():
        print(f"Error: Input CSV not found: {input_csv_path}")
        return
    
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} rows from input CSV")
    
    processed_dir = Path(processed_dir)
    
    # Process each subject
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning files"):
        subject_id = row.get('subject_id', '')
        dataset = row.get('dataset', '')
        
        if pd.isna(subject_id) or subject_id == '':
            continue
        if pd.isna(dataset) or dataset == '':
            continue
        
        # Find Phase 1 files
        files = find_phase1_files(processed_dir, dataset, subject_id)
        if files is None:
            continue
        
        # Get num_vertebrae from metadata if available
        num_vertebrae = 0
        metadata_file = processed_dir / dataset / subject_id / 'metadata.json'
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                num_vertebrae = metadata.get('num_vertebrae', 0)
            except:
                pass
        
        # Create result row
        result_row = {
            'subject_id': subject_id,
            'dataset': dataset,
            'mask_path': files['mask_path'],
            'ct_path': files['ct_path'],
            'centroids_path': files['centroids_path'],
            'present_path': files['present_path'],
            'has_mask': files['has_mask'],
            'has_ct': files['has_ct'],
            'num_vertebrae': num_vertebrae,
        }
        
        results.append(result_row)
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Save CSV
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    
    print()
    print("="*60)
    print("Phase 1 CSV Creation Complete!")
    print("="*60)
    print(f"Output: {output_csv_path}")
    print(f"Total subjects: {len(result_df)}")
    print(f"Subjects with mask: {result_df['has_mask'].sum()}")
    print(f"Subjects with CT: {result_df['has_ct'].sum()}")
    print(f"Average vertebrae per subject: {result_df['num_vertebrae'].mean():.1f}")
    print()
    print("Columns in output CSV:")
    for col in result_df.columns:
        print(f"  - {col}")


def main():
    parser = argparse.ArgumentParser(description='Create simplified Phase 1 CSV')
    parser.add_argument('--input_csv', type=str, 
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/preprocessed_data_subject_with_paths.csv',
                        help='Input CSV file (subject-level or vertebra-level)')
    parser.add_argument('--processed_dir', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed',
                        help='Processed data directory')
    parser.add_argument('--output_csv', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/preprocessed_data_phase1.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    create_phase1_csv(
        args.input_csv,
        args.processed_dir,
        args.output_csv
    )


if __name__ == '__main__':
    main()

