#!/usr/bin/env python3
"""
Select diverse subjects for visualization (some with many vertebrae, some with few)
"""

import json
from pathlib import Path
import sys

def select_diverse_subjects(metadata_path: Path, num_subjects: int = 5):
    """Select diverse subjects based on number of vertebrae"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Filter subjects with at least 2 vertebrae
    subjects = [
        (sid, info['num_vertebrae'])
        for sid, info in metadata.items()
        if info['num_vertebrae'] >= 2
    ]
    
    # Sort by number of vertebrae
    subjects.sort(key=lambda x: x[1])
    
    total = len(subjects)
    if total < num_subjects:
        print(f"WARNING: Only {total} subjects available, requested {num_subjects}")
        return [s[0] for s in subjects]
    
    # Select diverse subjects
    selected = []
    
    # 1. One with few vertebrae (bottom 20%)
    few_idx = int(total * 0.1)
    selected.append(subjects[few_idx][0])
    
    # 2. One with medium-few vertebrae (25%)
    med_few_idx = int(total * 0.25)
    selected.append(subjects[med_few_idx][0])
    
    # 3. One with medium vertebrae (50%)
    med_idx = int(total * 0.5)
    selected.append(subjects[med_idx][0])
    
    # 4. One with medium-many vertebrae (75%)
    med_many_idx = int(total * 0.75)
    selected.append(subjects[med_many_idx][0])
    
    # 5. One with many vertebrae (top 10%)
    many_idx = int(total * 0.9)
    selected.append(subjects[many_idx][0])
    
    return selected

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python select_diverse_subjects.py <metadata.json> [num_subjects]")
        sys.exit(1)
    
    metadata_path = Path(sys.argv[1])
    num_subjects = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    subjects = select_diverse_subjects(metadata_path, num_subjects)
    
    # Print as space-separated list for bash
    print(' '.join(subjects))
    
    # Also print details
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\nSelected subjects:", file=sys.stderr)
    for sid in subjects:
        num_v = metadata[sid]['num_vertebrae']
        print(f"  {sid}: {num_v} vertebrae", file=sys.stderr)

