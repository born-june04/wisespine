#!/usr/bin/env python3
"""
Create comprehensive paths CSV file for all subjects

This script scans all processed data directories and creates a CSV file
with actual file paths for masks, CT volumes, and other processed files.
This eliminates the need to search for files every time.
"""

import argparse
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm


def find_files_for_subject(processed_dir, dataset, subject_id, row=None):
    """Find all relevant files for a subject"""
    subject_dir = processed_dir / dataset / subject_id
    
    if not subject_dir.exists():
        return {}
    
    files = {}
    import pandas as pd
    
    # Mask files
    mask_candidates = [
        'mask_volume_1mm.npy',
        'processed_mask_1mm.npy',
    ]
    for mask_file in mask_candidates:
        mask_path = subject_dir / mask_file
        if mask_path.exists():
            files['mask_path'] = str(mask_path.absolute())
            files['has_mask'] = True
            break
    else:
        files['has_mask'] = False
    
    # CT files - check multiple possible names and patterns
    # First, check original CSV for processed_ct_1mm_path
    if row is not None and 'processed_ct_1mm_path' in row:
        ct_path_str = row.get('processed_ct_1mm_path', '')
        if pd.notna(ct_path_str) and ct_path_str != '':
            ct_path_from_csv = Path(str(ct_path_str).strip())
            if ct_path_from_csv.exists():
                files['ct_path'] = str(ct_path_from_csv.absolute())
                files['has_ct'] = True
                # Continue to check other files, don't return early
    
    # If not found in CSV, try standard locations and names
    ct_candidates = [
        'ct_volume_1mm.npy',
        'processed_ct_1mm.npy',
        'ct_volume.npy',
        'ct.npy',
        'image_volume_1mm.npy',
        'volume_1mm.npy',
    ]
    
    # First try exact matches
    for ct_file in ct_candidates:
        ct_path = subject_dir / ct_file
        if ct_path.exists():
            files['ct_path'] = str(ct_path.absolute())
            files['has_ct'] = True
            break
    else:
        # Try to find any .npy file that's likely CT (large file, not mask/centroid/present/position)
        npy_files = list(subject_dir.glob('*.npy'))
        for npy_file in npy_files:
            name_lower = npy_file.name.lower()
            # Skip known non-CT files
            if any(skip in name_lower for skip in ['mask', 'centroid', 'present', 'position', 'roi', 'localization']):
                continue
            
            # Check file size - CT files are typically large (> 10MB for 3D volumes)
            file_size_mb = npy_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 10.0:  # Likely CT if > 10MB (3D volume)
                # Only set if not already set from CSV
                if 'ct_path' not in files or not files.get('ct_path'):
                    files['ct_path'] = str(npy_file.absolute())
                    files['has_ct'] = True
                break
    
    # If CT file not found in processed directory, try to find original NIfTI file
    if 'ct_path' not in files or not files.get('ct_path'):
        # Check original CSV for ct_path (original NIfTI)
        if row is not None and 'ct_path' in row:
            original_ct_path_str = row.get('ct_path', '')
            if pd.notna(original_ct_path_str) and original_ct_path_str != '':
                original_ct_path = Path(str(original_ct_path_str).strip())
                if original_ct_path.exists():
                    files['original_ct_nifti_path'] = str(original_ct_path.absolute())
                    files['has_ct'] = True  # We have original, can load it
                    # Also set ct_path to original for now (can be converted later)
                    files['ct_path'] = str(original_ct_path.absolute())
        
        # If still not found, try to find in VerSe rawdata directory
        if 'ct_path' not in files or not files.get('ct_path'):
            # Try multiple possible verse directory structures
            possible_verse_dirs = [
                processed_dir.parent,  # VerSe/processed -> VerSe
                processed_dir.parent.parent,  # If nested deeper
                Path('/gscratch/scrubbed/june0604/vindr/VerSe'),  # Absolute path
            ]
            
            for verse_dir in possible_verse_dirs:
                if not verse_dir.exists():
                    continue
                    
                rawdata_dir = verse_dir / dataset / 'rawdata' / subject_id
                if rawdata_dir.exists():
                    # Look for CT NIfTI files
                    ct_nifti_candidates = list(rawdata_dir.glob('*ct*.nii.gz'))
                    if not ct_nifti_candidates:
                        # Try any .nii.gz that's not a mask
                        ct_nifti_candidates = [f for f in rawdata_dir.glob('*.nii.gz') 
                                              if 'seg' not in f.name.lower() and 'msk' not in f.name.lower()]
                    
                    if ct_nifti_candidates:
                        files['original_ct_nifti_path'] = str(ct_nifti_candidates[0].absolute())
                        files['ct_path'] = str(ct_nifti_candidates[0].absolute())
                        files['has_ct'] = True
                        break
                
                # Also check derivatives directory (sometimes CT is there)
                derivatives_dir = verse_dir / dataset / 'derivatives' / subject_id
                if derivatives_dir.exists():
                    ct_nifti_candidates = list(derivatives_dir.glob('*ct*.nii.gz'))
                    if not ct_nifti_candidates:
                        ct_nifti_candidates = [f for f in derivatives_dir.glob('*.nii.gz') 
                                              if 'seg' not in f.name.lower() and 'msk' not in f.name.lower()]
                    
                    if ct_nifti_candidates:
                        files['original_ct_nifti_path'] = str(ct_nifti_candidates[0].absolute())
                        files['ct_path'] = str(ct_nifti_candidates[0].absolute())
                        files['has_ct'] = True
                        break
        
        # Final check - if still no CT, mark as not found
        if 'ct_path' not in files or not files.get('ct_path'):
            files['has_ct'] = False
    
    # Other files
    other_files = {
        'centroids_path': 'vertebra_centroids.npy',
        'present_path': 'vertebra_present.npy',
        'positions_path': 'vertebra_positions.npy',
        'metadata_path': 'metadata.json',
    }
    
    for key, filename in other_files.items():
        file_path = subject_dir / filename
        if file_path.exists():
            files[key] = str(file_path.absolute())
    
    # ROI directory
    roi_dir = subject_dir / 'rois'
    if roi_dir.exists():
        files['roi_dir'] = str(roi_dir.absolute())
    
    # Localization directory
    loc_dir = subject_dir / 'localization'
    if loc_dir.exists():
        files['localization_dir'] = str(loc_dir.absolute())
    
    return files


def create_paths_csv(input_csv_path, processed_dir, output_csv_path):
    """Create comprehensive paths CSV from existing CSV"""
    
    print("="*60)
    print("Creating Comprehensive Paths CSV")
    print("="*60)
    print(f"Input CSV: {input_csv_path}")
    print(f"Processed directory: {processed_dir}")
    print(f"Output CSV: {output_csv_path}")
    print()
    
    # Load existing CSV
    if not Path(input_csv_path).exists():
        print(f"Error: Input CSV not found: {input_csv_path}")
        return
    
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} subjects from CSV")
    
    processed_dir = Path(processed_dir)
    
    # Process each subject
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning files"):
        subject_id = row.get('subject_id', '')
        dataset = row.get('dataset', '')
        
        if pd.isna(subject_id) or subject_id == '':
            continue
        
        # Find files (pass row for CSV path checking)
        files = find_files_for_subject(processed_dir, dataset, subject_id, row)
        
        # Create result row
        result_row = {
            'subject_id': subject_id,
            'dataset': dataset,
        }
        
        # Add existing columns
        for col in df.columns:
            if col not in ['subject_id', 'dataset']:
                result_row[col] = row.get(col, '')
        
        # Add found file paths
        result_row.update(files)
        
        results.append(result_row)
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Ensure all path columns exist (fill with empty string if missing)
    path_columns = [
        'mask_path', 'ct_path', 'centroids_path', 'present_path',
        'positions_path', 'metadata_path', 'roi_dir', 'localization_dir'
    ]
    for col in path_columns:
        if col not in result_df.columns:
            result_df[col] = ''
    
    # Ensure boolean columns
    if 'has_mask' not in result_df.columns:
        result_df['has_mask'] = False
    if 'has_ct' not in result_df.columns:
        result_df['has_ct'] = False
    
    # Save CSV
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    
    print()
    print("="*60)
    print("CSV Creation Complete!")
    print("="*60)
    print(f"Output: {output_csv_path}")
    print(f"Total subjects: {len(result_df)}")
    print(f"Subjects with mask: {result_df['has_mask'].sum()}")
    print(f"Subjects with CT: {result_df['has_ct'].sum()}")
    print()
    print("Columns in output CSV:")
    for col in result_df.columns:
        non_empty = (result_df[col] != '').sum() if col in path_columns else 'N/A'
        print(f"  - {col}: {non_empty} non-empty" if isinstance(non_empty, int) else f"  - {col}")


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive paths CSV file')
    parser.add_argument('--input_csv', type=str, 
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/preprocessed_data_subject.csv',
                        help='Input CSV file (existing preprocessed_data_subject.csv)')
    parser.add_argument('--processed_dir', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed',
                        help='Processed data directory')
    parser.add_argument('--output_csv', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/preprocessed_data_subject_with_paths.csv',
                        help='Output CSV file with all paths')
    
    args = parser.parse_args()
    
    create_paths_csv(
        input_csv_path=args.input_csv,
        processed_dir=args.processed_dir,
        output_csv_path=args.output_csv,
    )


if __name__ == '__main__':
    main()

