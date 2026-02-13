#!/usr/bin/env python3
"""
Phase 1: Generate proxy abnormal CT volumes via vertebra-level rigid transforms.

This script:
  - Loads a normal CT and its GT vertebra segmentation
  - Applies random rigid transforms (translation + rotation) per vertebra
  - Saves the transformed CT (proxy abnormal) for running TS (teacher)

Ablation A1: Piecewise rigid transform
  - Each vertebra label gets an independent rigid transform
  - Transform parameters are randomized within anatomically plausible bounds
  - Background and non-vertebra voxels remain unchanged

Usage:
  python spine-rl-sim/ablation/generate_proxy_abnormal.py \
    --subject sub-verse563 \
    --out_dir spine-rl-sim/ablation_outputs/2026-01-28/phase1_proxy_abnormal \
    --seed 42 \
    --num_samples 5 \
    --tx_range 0 10 \
    --ty_range 0 10 \
    --tz_range 0 20 \
    --rx_range 0 15 \
    --ry_range 0 15 \
    --rz_range 0 15
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import nibabel as nib
except Exception as e:
    raise RuntimeError("nibabel required (pip install nibabel)") from e

try:
    from scipy import ndimage as ndi
    from scipy.spatial.transform import Rotation
except Exception as e:
    raise RuntimeError("scipy required (pip install scipy)") from e


REPO_ROOT = Path(__file__).resolve().parents[2]


def _vertebra_label_map() -> dict[str, int]:
    """Standard VerSe label map: C1-C7(1-7), T1-T12(8-19), L1-L5(20-24), S1(25)"""
    labels = {}
    for i in range(1, 8):
        labels[f"C{i}"] = i
    for i in range(1, 13):
        labels[f"T{i}"] = 7 + i
    for i in range(1, 6):
        labels[f"L{i}"] = 19 + i
    labels["S1"] = 25
    return labels


def gt_path_for(subject: str) -> Path:
    """Return path to GT vertebra segmentation."""
    p = REPO_ROOT / "VerSe" / "dataset-03test" / "derivatives" / subject / f"{subject}_dir-iso_seg-vert_msk.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"GT segmentation not found: {p}")
    return p


def ct_path_for(subject: str) -> Path:
    """Return path to original CT volume."""
    # Assuming rawdata contains the CT (adjust path if different)
    p = REPO_ROOT / "VerSe" / "dataset-03test" / "rawdata" / subject / f"{subject}_dir-iso_ct.nii.gz"
    if not p.exists():
        # Try alternative location
        p = REPO_ROOT / "VerSe" / "dataset-03test" / subject / f"{subject}_dir-iso_ct.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"CT volume not found for {subject}")
    return p


@dataclass
class RigidTransform:
    """Rigid transform: translation (mm) + rotation (degrees)"""
    tx: float
    ty: float
    tz: float
    rx: float  # rotation around x-axis (degrees)
    ry: float  # rotation around y-axis (degrees)
    rz: float  # rotation around z-axis (degrees)


def sample_rigid_transform(
    rng: np.random.Generator,
    tx_range: Tuple[float, float],
    ty_range: Tuple[float, float],
    tz_range: Tuple[float, float],
    rx_range: Tuple[float, float],
    ry_range: Tuple[float, float],
    rz_range: Tuple[float, float],
) -> RigidTransform:
    """Sample a random rigid transform within specified ranges."""
    return RigidTransform(
        tx=rng.uniform(*tx_range),
        ty=rng.uniform(*ty_range),
        tz=rng.uniform(*tz_range),
        rx=rng.uniform(*rx_range),
        ry=rng.uniform(*ry_range),
        rz=rng.uniform(*rz_range),
    )


def apply_rigid_to_label(
    ct: np.ndarray,
    seg: np.ndarray,
    label: int,
    transform: RigidTransform,
    affine: np.ndarray,
) -> np.ndarray:
    """
    Apply rigid transform to voxels of a specific label in CT.
    
    Returns modified CT (all labels composed), not just this one.
    This is meant to be called iteratively for each label.
    """
    # Extract mask for this label
    mask = (seg == label)
    if mask.sum() == 0:
        return ct  # No voxels for this label
    
    # Compute centroid in voxel coordinates
    coords = np.argwhere(mask)  # shape (N, 3)
    centroid_vox = coords.mean(axis=0)  # shape (3,)
    
    # Convert translation from mm to voxel units (assuming isotropic or near-isotropic)
    # Simplification: use diagonal of affine as voxel size
    voxel_size = np.abs(np.diag(affine[:3, :3]))
    tx_vox = transform.tx / voxel_size[0]
    ty_vox = transform.ty / voxel_size[1]
    tz_vox = transform.tz / voxel_size[2]
    
    # Build rotation matrix (ZYX Euler convention)
    rot_mat = Rotation.from_euler('xyz', [transform.rx, transform.ry, transform.rz], degrees=True).as_matrix()
    
    # Affine transform: rotate around centroid, then translate
    # For each voxel p: p' = R(p - c) + c + t
    # We'll use scipy's affine_transform with appropriate matrix/offset
    
    # Create full 4x4 transform matrix
    # Note: scipy's affine_transform uses *inverse* mapping, so we need to invert
    mat = np.eye(4)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = [tx_vox, ty_vox, tz_vox]
    
    # Adjust for rotation around centroid: T(c) @ R @ T(-c)
    mat_inv = np.linalg.inv(mat)
    
    # Apply to CT values (only where mask is True)
    # Extract bounding box to reduce computation
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    pad = 10  # safety padding
    i0, j0, k0 = np.maximum(coords_min - pad, 0)
    i1, j1, k1 = np.minimum(coords_max + pad + 1, ct.shape)
    
    bbox_ct = ct[i0:i1, j0:j1, k0:k1].copy()
    bbox_mask = mask[i0:i1, j0:j1, k0:k1]
    
    # Offset centroid relative to bbox
    centroid_bbox = centroid_vox - np.array([i0, j0, k0])
    
    # Build transformation: offset to centroid, apply inverse rigid, offset back
    offset_fwd = centroid_bbox
    offset_inv = -mat_inv[:3, :3] @ centroid_bbox + mat_inv[:3, 3]
    
    # Apply affine_transform (uses inverse mapping internally)
    bbox_ct_transformed = ndi.affine_transform(
        bbox_ct,
        matrix=mat_inv[:3, :3],
        offset=offset_inv,
        order=1,  # linear interpolation for CT intensity
        cval=ct.min(),  # background value
    )
    
    # Blend: only update voxels that were originally in this label's mask
    # Simple approach: replace masked region
    ct_out = ct.copy()
    ct_out[i0:i1, j0:j1, k0:k1][bbox_mask] = bbox_ct_transformed[bbox_mask]
    
    return ct_out


def generate_proxy_abnormal(
    ct: np.ndarray,
    seg: np.ndarray,
    affine: np.ndarray,
    rng: np.random.Generator,
    tx_range: Tuple[float, float],
    ty_range: Tuple[float, float],
    tz_range: Tuple[float, float],
    rx_range: Tuple[float, float],
    ry_range: Tuple[float, float],
    rz_range: Tuple[float, float],
    num_labels_to_transform: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Generate proxy abnormal CT by applying random rigid transforms to a subset of vertebrae.
    
    Returns:
        ct_abnormal: transformed CT volume
        transform_log: dict mapping label -> RigidTransform
    """
    labels = sorted(set(np.unique(seg).tolist()) - {0})
    if len(labels) == 0:
        return ct, {}
    
    # Randomly select a subset of labels to transform
    num_to_transform = min(num_labels_to_transform, len(labels))
    selected_labels = rng.choice(labels, size=num_to_transform, replace=False)
    
    ct_out = ct.copy()
    transform_log = {}
    
    for label in selected_labels:
        trans = sample_rigid_transform(rng, tx_range, ty_range, tz_range, rx_range, ry_range, rz_range)
        ct_out = apply_rigid_to_label(ct_out, seg, int(label), trans, affine)
        transform_log[int(label)] = asdict(trans)
    
    return ct_out, transform_log


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate proxy abnormal CT via rigid vertebra transforms")
    ap.add_argument("--subject", default="sub-verse563")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: spine-rl-sim/ablation_outputs/<today>/phase1_proxy_abnormal)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--num_samples", type=int, default=5, help="Number of proxy abnormal samples to generate")
    ap.add_argument("--num_labels_to_transform", type=int, default=3, help="Number of vertebrae to transform per sample")
    ap.add_argument("--tx_range", nargs=2, type=float, default=[0.0, 10.0], help="Translation X range (mm)")
    ap.add_argument("--ty_range", nargs=2, type=float, default=[0.0, 10.0], help="Translation Y range (mm)")
    ap.add_argument("--tz_range", nargs=2, type=float, default=[0.0, 20.0], help="Translation Z range (mm)")
    ap.add_argument("--rx_range", nargs=2, type=float, default=[0.0, 15.0], help="Rotation X range (degrees)")
    ap.add_argument("--ry_range", nargs=2, type=float, default=[0.0, 15.0], help="Rotation Y range (degrees)")
    ap.add_argument("--rz_range", nargs=2, type=float, default=[0.0, 15.0], help="Rotation Z range (degrees)")
    args = ap.parse_args()
    
    subject = args.subject
    rng = np.random.default_rng(args.seed)
    
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "spine-rl-sim" / "ablation_outputs" / "2026-01-28" / "phase1_proxy_abnormal")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CT and segmentation
    ct_path = ct_path_for(subject)
    seg_path = gt_path_for(subject)
    
    print(f"Loading CT: {ct_path}")
    ct_img = nib.load(str(ct_path))
    ct = ct_img.get_fdata().astype(np.float32)
    
    print(f"Loading GT seg: {seg_path}")
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata().astype(np.uint16)
    
    affine = ct_img.affine
    
    # Generate proxy abnormal samples
    for i in range(args.num_samples):
        sample_seed = args.seed + i
        sample_rng = np.random.default_rng(sample_seed)
        
        print(f"\nGenerating sample {i+1}/{args.num_samples} (seed={sample_seed})...")
        ct_abnormal, transform_log = generate_proxy_abnormal(
            ct, seg, affine, sample_rng,
            tuple(args.tx_range),
            tuple(args.ty_range),
            tuple(args.tz_range),
            tuple(args.rx_range),
            tuple(args.ry_range),
            tuple(args.rz_range),
            num_labels_to_transform=args.num_labels_to_transform,
        )
        
        # Save transformed CT
        sample_dir = out_dir / subject / f"sample_{i:03d}_seed{sample_seed}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        ct_out_path = sample_dir / f"{subject}_proxy_abnormal_ct.nii.gz"
        ct_out_img = nib.Nifti1Image(ct_abnormal, affine, ct_img.header)
        nib.save(ct_out_img, str(ct_out_path))
        print(f"  Saved CT: {ct_out_path}")
        
        # Save transform log
        log_path = sample_dir / "transform_log.json"
        log_payload = {
            "subject": subject,
            "seed": sample_seed,
            "num_labels_transformed": len(transform_log),
            "transforms": transform_log,
            "params": {
                "tx_range": args.tx_range,
                "ty_range": args.ty_range,
                "tz_range": args.tz_range,
                "rx_range": args.rx_range,
                "ry_range": args.ry_range,
                "rz_range": args.rz_range,
            },
        }
        log_path.write_text(json.dumps(log_payload, indent=2))
        print(f"  Saved log: {log_path}")
    
    print(f"\n✓ Generated {args.num_samples} proxy abnormal samples in {out_dir / subject}")


if __name__ == "__main__":
    main()

