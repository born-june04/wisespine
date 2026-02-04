#!/usr/bin/env python3
"""
Phase 1-2: Run TotalSeg (teacher) on proxy abnormal CT to collect failure patterns.

This script:
  - Takes proxy abnormal CT samples
  - Runs TotalSeg on each to get "teacher failure" masks
  - Saves the results for later teacher-consistency evaluation

Usage:
  python spine-rl-sim/ablation/run_teacher_ts_on_proxy.py \
    --proxy_dir spine-rl-sim/ablation_outputs/2026-01-28/phase1_proxy_abnormal/sub-verse563 \
    --out_dir spine-rl-sim/ablation_outputs/2026-01-28/phase1_teacher_ts
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_proxy_samples(proxy_dir: Path) -> List[Path]:
    """Find all proxy abnormal CT samples in the given directory."""
    samples = []
    for sample_dir in sorted(proxy_dir.glob("sample_*")):
        ct_files = list(sample_dir.glob("*_proxy_abnormal_ct.nii.gz"))
        if ct_files:
            samples.append(ct_files[0])
    return samples


def run_totalseg(ct_path: Path, out_dir: Path) -> None:
    """
    Run TotalSeg on a single CT file.
    
    TotalSeg command:
      TotalSegmentator -i input.nii.gz -o output_dir/ -ta vertebrae_body
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "TotalSegmentator",
        "-i", str(ct_path),
        "-o", str(out_dir),
        "-ta", "vertebrae_body",  # Only segment vertebrae
        "--quiet",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR running TotalSeg:")
        print(result.stderr)
        raise RuntimeError(f"TotalSeg failed for {ct_path}")
    
    print(f"✓ TotalSeg completed for {ct_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TotalSeg teacher on proxy abnormal CT samples")
    ap.add_argument("--proxy_dir", required=True, help="Directory containing proxy abnormal samples")
    ap.add_argument("--out_dir", required=True, help="Output directory for teacher TS results")
    args = ap.parse_args()
    
    proxy_dir = Path(args.proxy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all proxy samples
    samples = find_proxy_samples(proxy_dir)
    if not samples:
        raise RuntimeError(f"No proxy abnormal CT samples found in {proxy_dir}")
    
    print(f"Found {len(samples)} proxy abnormal CT samples")
    
    # Run TotalSeg on each
    results = []
    for i, ct_path in enumerate(samples):
        sample_name = ct_path.parent.name
        print(f"\n[{i+1}/{len(samples)}] Processing {sample_name}...")
        
        # Output directory for this sample
        sample_out_dir = out_dir / sample_name
        
        try:
            run_totalseg(ct_path, sample_out_dir)
            
            # Save metadata
            metadata = {
                "sample_name": sample_name,
                "ct_path": str(ct_path),
                "ts_out_dir": str(sample_out_dir),
                "status": "success",
            }
            results.append(metadata)
            
            # Save per-sample metadata
            meta_path = sample_out_dir / "teacher_ts_metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            
        except Exception as e:
            print(f"ERROR: {e}")
            metadata = {
                "sample_name": sample_name,
                "ct_path": str(ct_path),
                "status": "failed",
                "error": str(e),
            }
            results.append(metadata)
    
    # Save overall summary
    summary_path = out_dir / "teacher_ts_summary.json"
    summary = {
        "total_samples": len(samples),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    
    print(f"\n{'='*60}")
    print(f"Teacher TS completed: {summary['successful']}/{summary['total_samples']} successful")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

