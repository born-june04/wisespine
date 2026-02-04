#!/usr/bin/env python3
"""
Comprehensive CT file finder

Searches for CT files in all possible locations and creates a complete mapping.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json


def find_ct_files_comprehensive(subject_id, dataset, processed_dir, verse_base_dir=None):
    """Comprehensively search for CT files in all possible locations"""
    
    results = {
        'processed_ct_npy': None,
        'original_ct_nifti': None,
        'localization_slices': None,
    }
    
    # 1. Check processed directory for .npy files
    subject_dir = processed_dir / dataset / subject_id
    if subject_dir.exists():
        # Check for CT .npy files
        ct_npy_candidates = [
            'ct_volume_1mm.npy',
            'processed_ct_1mm.npy',
            'ct_volume.npy',
        ]
        for candidate in ct_npy_candidates:
            ct_file = subject_dir / candidate
            if ct_file.exists():
                results['processed_ct_npy'] = str(ct_file.absolute())
                break
        
        # Check localization slices (contains CT data)
        loc_dir = subject_dir / 'localization_slices'
        if loc_dir.exists():
            axial_slices = loc_dir / 'axial_slices.npy'
            if axial_slices.exists():
                results['localization_slices'] = str(axial_slices.absolute())
    
    # 2. Check original NIfTI files in various locations
    if verse_base_dir is None:
        verse_base_dir = processed_dir.parent if processed_dir.name == 'processed' else processed_dir
    
    # Try rawdata directory
    rawdata_dir = verse_base_dir / dataset / 'rawdata' / subject_id
    if rawdata_dir.exists():
        ct_nifti_candidates = list(rawdata_dir.glob('*ct*.nii.gz'))
        if not ct_nifti_candidates:
            ct_nifti_candidates = [f for f in rawdata_dir.glob('*.nii.gz') 
                                  if 'seg' not in f.name.lower() and 'msk' not in f.name.lower()]
        if ct_nifti_candidates:
            results['original_ct_nifti'] = str(ct_nifti_candidates[0].absolute())
    
    # Try derivatives directory
    if not results['original_ct_nifti']:
        derivatives_dir = verse_base_dir / dataset / 'derivatives' / subject_id
        if derivatives_dir.exists():
            ct_nifti_candidates = list(derivatives_dir.glob('*ct*.nii.gz'))
            if not ct_nifti_candidates:
                ct_nifti_candidates = [f for f in derivatives_dir.glob('*.nii.gz') 
                                      if 'seg' not in f.name.lower() and 'msk' not in f.name.lower()]
            if ct_nifti_candidates:
                results['original_ct_nifti'] = str(ct_nifti_candidates[0].absolute())
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Find all CT files comprehensively')
    parser.add_argument('--input_csv', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/preprocessed_data_subject.csv',
                        help='Input CSV file')
    parser.add_argument('--processed_dir', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed',
                        help='Processed data directory')
    parser.add_argument('--output_json', type=str,
                        default='/gscratch/scrubbed/june0604/vindr/VerSe/processed/ct_files_mapping.json',
                        help='Output JSON file with CT file mappings')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    processed_dir = Path(args.processed_dir)
    
    print("="*60)
    print("Comprehensive CT File Search")
    print("="*60)
    print(f"Searching for CT files for {len(df)} subjects...")
    print()
    
    ct_mapping = {}
    stats = {
        'processed_ct_npy': 0,
        'original_ct_nifti': 0,
        'localization_slices': 0,
        'no_ct': 0,
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Searching"):
        subject_id = row['subject_id']
        dataset = row['dataset']
        
        if pd.isna(subject_id) or subject_id == '':
            continue
        
        results = find_ct_files_comprehensive(subject_id, dataset, processed_dir)
        ct_mapping[subject_id] = results
        
        # Update stats
        if results['processed_ct_npy']:
            stats['processed_ct_npy'] += 1
        elif results['original_ct_nifti']:
            stats['original_ct_nifti'] += 1
        elif results['localization_slices']:
            stats['localization_slices'] += 1
        else:
            stats['no_ct'] += 1
    
    # Save results
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(ct_mapping, f, indent=2)
    
    print()
    print("="*60)
    print("CT File Search Complete!")
    print("="*60)
    print(f"Results saved to: {output_json}")
    print()
    print("Statistics:")
    print(f"  Processed CT .npy files: {stats['processed_ct_npy']}")
    print(f"  Original CT NIfTI files: {stats['original_ct_nifti']}")
    print(f"  Localization slices (2D CT): {stats['localization_slices']}")
    print(f"  No CT found: {stats['no_ct']}")
    print()
    
    # Show samples
    print("Sample results:")
    for subject_id, results in list(ct_mapping.items())[:5]:
        print(f"  {subject_id}:")
        if results['processed_ct_npy']:
            print(f"    ✓ Processed CT: {Path(results['processed_ct_npy']).name}")
        if results['original_ct_nifti']:
            print(f"    ✓ Original NIfTI: {Path(results['original_ct_nifti']).name}")
        if results['localization_slices']:
            print(f"    ✓ Localization slices: {Path(results['localization_slices']).name}")
        if not any(results.values()):
            print(f"    ✗ No CT found")


if __name__ == '__main__':
    main()

