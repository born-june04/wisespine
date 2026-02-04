#!/usr/bin/env python3
"""
Build point clouds + simple features from TS per-vertebra masks.

We mirror the expected on-disk format used by `extract_assembly_embeddings.py`:
  <out_dir>/<subject_id>/vertebra_<ID>_points.npy
  <out_dir>/<subject_id>/vertebra_<ID>_features.npz   (normals + dummy curvature)

Design choices:
- Use marching cubes to extract a surface mesh (consistent with training pipeline).
- Sample surface points + normals from the mesh.
- Curvature scalars (k1,k2) are set to zeros to satisfy irreps_in="2x0e + 1x1o".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spine_point_cloud_assembly.utils.geometry import extract_mesh_from_mask  # noqa: E402


def _label_id_from_filename(p: Path) -> int | None:
    name = p.name
    if not name.startswith("vertebrae_"):
        return None
    name = name[len("vertebrae_") :]
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]

    if name.startswith(("C", "T", "L")) and len(name) >= 2:
        try:
            prefix = name[0]
            num = int(name[1:])
        except Exception:
            return None
        if prefix == "C" and 1 <= num <= 7:
            return num
        if prefix == "T" and 1 <= num <= 12:
            return 7 + num
        if prefix == "L" and 1 <= num <= 5:
            return 19 + num
    if name == "S1":
        return 25
    return None


def _voxel_spacing_from_affine(aff: np.ndarray) -> tuple[float, float, float]:
    # spacing along each axis = norm of column vectors
    sx = float(np.linalg.norm(aff[:3, 0]))
    sy = float(np.linalg.norm(aff[:3, 1]))
    sz = float(np.linalg.norm(aff[:3, 2]))
    return (sx, sy, sz)


def _sample_points_and_normals_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vnormals: np.ndarray | None,
    num_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample points on triangle faces (area-weighted) + interpolate vertex normals.
    """
    rng = np.random.default_rng(seed)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if total <= 0:
        raise ValueError("Mesh has zero area")
    probs = areas / total
    face_idx = rng.choice(len(faces), size=num_points, replace=True, p=probs)

    u = rng.random(num_points).astype(np.float32)
    v = rng.random(num_points).astype(np.float32)
    swap = (u + v) > 1.0
    u[swap] = 1.0 - u[swap]
    v[swap] = 1.0 - v[swap]
    w = 1.0 - u - v

    f = faces[face_idx]
    p0 = vertices[f[:, 0]]
    p1 = vertices[f[:, 1]]
    p2 = vertices[f[:, 2]]
    pts = (u[:, None] * p0) + (v[:, None] * p1) + (w[:, None] * p2)

    if vnormals is None:
        # fallback: face normal
        fn = cross[face_idx]
        n = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
        return pts.astype(np.float32), n.astype(np.float32)

    n0 = vnormals[f[:, 0]]
    n1 = vnormals[f[:, 1]]
    n2 = vnormals[f[:, 2]]
    nn = (u[:, None] * n0) + (v[:, None] * n1) + (w[:, None] * n2)
    nn = nn / (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-8)
    return pts.astype(np.float32), nn.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/totalseg_eval/predictions_total"),
    )
    parser.add_argument("--subject-id", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/point_clouds_ts"),
    )
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import nibabel as nib

    sid = args.subject_id
    subj_pred = args.pred_root / sid
    if not subj_pred.exists():
        raise FileNotFoundError(f"pred dir not found: {subj_pred}")

    out_subj = args.out_dir / sid
    out_subj.mkdir(parents=True, exist_ok=True)

    masks = sorted(subj_pred.glob("vertebrae_*.nii.gz"))
    if not masks:
        raise FileNotFoundError(f"No vertebrae masks in: {subj_pred}")

    for mp in masks:
        lab = _label_id_from_filename(mp)
        if lab is None:
            continue

        img = nib.as_closest_canonical(nib.load(str(mp)))
        # Load mask array (uint8) without float upcast
        m = np.asanyarray(img.dataobj)
        if m.ndim != 3:
            continue
        if np.count_nonzero(m) < 10:
            continue

        spacing = _voxel_spacing_from_affine(img.affine)
        mesh = extract_mesh_from_mask((m > 0).astype(np.float32), spacing=spacing, level=0.5)
        if mesh is None:
            continue

        pts, nrm = _sample_points_and_normals_from_mesh(
            mesh["vertices"], mesh["faces"], mesh.get("normals"), num_points=int(args.num_points), seed=int(args.seed)
        )

        # Save original (non-centered) points
        np.save(out_subj / f"vertebra_{lab}_points.npy", pts.astype(np.float32))

        # Features: normals + dummy curvature scalars (k1,k2)=0
        k1 = np.zeros((pts.shape[0],), dtype=np.float32)
        k2 = np.zeros((pts.shape[0],), dtype=np.float32)
        np.savez(out_subj / f"vertebra_{lab}_features.npz", normals=nrm.astype(np.float32), k1=k1, k2=k2)

    print(f"✓ Built TS point clouds for {sid} -> {out_subj}")


if __name__ == "__main__":
    main()


