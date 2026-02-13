#!/usr/bin/env python3
"""
Export a single vertebra mesh (GT + TotalSeg) as ONE OBJ for Unity.

This script is designed to be memory-safe on huge NIfTI volumes by:
  - Streaming slices to compute bounding boxes
  - Cropping around the target vertebra before running marching cubes

Example:
  python scripts/export_unity_single_vertebra_obj.py \
    --subject sub-verse563 --vertebra L1 \
    --out_dir outputs/unity_exports

Notes:
  - Colors: by default uses per-vertebra materials (GT brighter, TS darker).
    Disable with: --no-per_vertebra_colors
  - If you omit --vertebra (or pass --all), this script will export ALL vertebrae
    by delegating to scripts/export_unity_all_vertebrae_obj.py.
"""

from __future__ import annotations

import argparse
import colorsys
import gzip
from pathlib import Path
import shutil
import sys
import runpy
from typing import Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception as e:  # pragma: no cover
    raise RuntimeError("nibabel is required. Please install it (e.g., pip install nibabel).") from e

try:
    from skimage import measure
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-image is required. Please install it (e.g., pip install scikit-image).") from e


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def vertebra_to_label(name: str) -> int:
    """
    Map vertebra name to VerSe-style integer labels:
      C1..C7 -> 1..7
      T1..T12 -> 8..19
      L1..L5 -> 20..24
      S1 -> 25
    """
    name = name.strip().upper()
    if name.startswith("C"):
        i = int(name[1:])
        if 1 <= i <= 7:
            return i
    if name.startswith("T"):
        i = int(name[1:])
        if 1 <= i <= 12:
            return 7 + i
    if name.startswith("L"):
        i = int(name[1:])
        if 1 <= i <= 5:
            return 19 + i
    if name == "S1":
        return 25
    raise ValueError(f"Unknown vertebra '{name}' (expected C1..C7, T1..T12, L1..L5, S1)")


def find_gt_path(subject: str) -> Path:
    # sub-verse563 GT lives here in this repo layout
    p = PROJECT_ROOT / "VerSe" / "dataset-03test" / "derivatives" / subject / f"{subject}_dir-iso_seg-vert_msk.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"GT file not found: {p}")
    return p


def find_ts_path(subject: str, vertebra: str) -> Path:
    p = PROJECT_ROOT / "totalseg_eval" / "predictions_total" / subject / f"vertebrae_{vertebra.upper()}.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"TotalSeg file not found: {p}")
    return p


def maybe_cache_unzipped_nifti(path: Path, cache_dir: Path, enabled: bool) -> Path:
    """
    Random-access slicing of `.nii.gz` can be extremely slow because gzip isn't suited for fast seeks.
    If enabled and input is `.nii.gz`, create (or reuse) an uncompressed `.nii` in cache_dir
    so nibabel can memmap / slice efficiently.
    """
    if not enabled:
        return path
    if not path.name.endswith(".nii.gz"):
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / path.name[: -len(".gz")]  # keep ".nii"
    if out.exists() and out.stat().st_size > 0:
        return out

    # Stream-decompress to avoid loading into RAM
    tmp = out.with_suffix(out.suffix + ".tmp")
    with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp.replace(out)
    return out


def _update_bbox(
    bbox: Optional[Tuple[int, int, int, int, int, int]],
    xs: np.ndarray,
    ys: np.ndarray,
    z: int,
) -> Tuple[int, int, int, int, int, int]:
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if bbox is None:
        return x0, x1, y0, y1, z, z
    bx0, bx1, by0, by1, bz0, bz1 = bbox
    return min(bx0, x0), max(bx1, x1), min(by0, y0), max(by1, y1), min(bz0, z), max(bz1, z)


def bbox_from_proxy(
    dataobj,
    predicate,
    z_step: int = 1,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Compute (x0,x1,y0,y1,z0,z1) for voxels satisfying predicate(slice2d).
    Streams along z to avoid loading full volume.
    """
    shape = dataobj.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {shape}")

    bbox = None
    for z in range(0, shape[2], z_step):
        sl = np.asanyarray(dataobj[:, :, z])
        m = predicate(sl)
        if not np.any(m):
            continue
        xs, ys = np.where(m)
        bbox = _update_bbox(bbox, xs, ys, z)
    return bbox


def grow_and_clip_bbox(
    bbox: Tuple[int, int, int, int, int, int],
    shape: Tuple[int, int, int],
    margin: int,
) -> Tuple[int, int, int, int, int, int]:
    x0, x1, y0, y1, z0, z1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    z0 = max(0, z0 - margin)
    x1 = min(shape[0] - 1, x1 + margin)
    y1 = min(shape[1] - 1, y1 + margin)
    z1 = min(shape[2] - 1, z1 + margin)
    return x0, x1, y0, y1, z0, z1


def crop_binary_volume_from_proxy(
    dataobj,
    predicate,
    bbox: Tuple[int, int, int, int, int, int],
) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bbox
    out = np.zeros((x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1), dtype=np.uint8)
    for z in range(z0, z1 + 1):
        sl = np.asanyarray(dataobj[x0 : x1 + 1, y0 : y1 + 1, z])
        m = predicate(sl)
        out[:, :, z - z0] = m.astype(np.uint8, copy=False)
    return out


def apply_affine_to_vertices(affine: np.ndarray, verts_ijk: np.ndarray) -> np.ndarray:
    """
    verts_ijk: (N,3) in voxel index coordinates (i,j,k) matching array axes (x,y,z).
    Returns world coordinates (N,3).
    """
    r = affine[:3, :3].astype(np.float64)
    t = affine[:3, 3].astype(np.float64)
    return (verts_ijk.astype(np.float64) @ r.T) + t


def marching_cubes_binary(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Run in voxel-index space (spacing=1) then map to world with affine later.
    verts, faces, _normals, _values = measure.marching_cubes(vol, level=0.5)
    return verts.astype(np.float32), faces.astype(np.int32)


def _material_color_from_name(mat_name: str) -> tuple[float, float, float]:
    """
    Material naming convention:
      - mat_GT_<VERTEBRA> (e.g., mat_GT_L1)
      - mat_TS_<VERTEBRA>
    """
    if mat_name.startswith("mat_GT_"):
        kind = "GT"
        vname = mat_name[len("mat_GT_") :]
    elif mat_name.startswith("mat_TS_"):
        kind = "TS"
        vname = mat_name[len("mat_TS_") :]
    else:
        kind = "OTHER"
        vname = mat_name

    # Map vname -> label for hue
    label = None
    try:
        if vname.startswith("C"):
            label = int(vname[1:])
        elif vname.startswith("T"):
            label = 7 + int(vname[1:])
        elif vname.startswith("L"):
            label = 19 + int(vname[1:])
        elif vname == "S1":
            label = 25
    except Exception:
        label = None

    if label is None:
        h = (abs(hash(vname)) % 360) / 360.0
    else:
        h = (label - 1) / 25.0

    s = 0.75
    v = 0.95 if kind == "GT" else 0.75 if kind == "TS" else 0.85
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return float(r), float(g), float(b)


def write_obj_with_mtl(
    obj_path: Path,
    mtl_path: Path,
    parts: list[dict],
) -> None:
    """
    parts: list of {name, material, vertices(N,3), faces(M,3)}
    """
    mtl_lines = ["# Unity-friendly materials"]
    materials_seen = sorted({p["material"] for p in parts})
    for mat in materials_seen:
        kd = _material_color_from_name(mat)
        mtl_lines += [
            f"newmtl {mat}",
            f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}",
            "Ka 0.0000 0.0000 0.0000",
            "Ks 0.0000 0.0000 0.0000",
            "d 1.0",
            "illum 1",
            "",
        ]
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.write_text("\n".join(mtl_lines) + "\n")

    # OBJ
    lines = [
        "# Combined GT + TotalSeg OBJ",
        f"mtllib {mtl_path.name}",
        "",
    ]
    v_offset = 0
    for p in parts:
        name = p["name"]
        mat = p["material"]
        v = p["vertices"]
        f = p["faces"]
        lines.append(f"o {name}")
        lines.append(f"usemtl {mat}")
        for vv in v:
            lines.append(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}")
        # faces are 0-based; OBJ uses 1-based with global vertex indexing
        for tri in f:
            a, b, c = (tri + 1 + v_offset).tolist()
            lines.append(f"f {a} {b} {c}")
        lines.append("")
        v_offset += v.shape[0]

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="e.g., sub-verse563")
    ap.add_argument("--vertebra", required=False, help="e.g., L1 (omit to export all)")
    ap.add_argument("--all", action="store_true", help="Export all vertebrae (equivalent to omitting --vertebra)")
    ap.add_argument("--out_dir", default="outputs/unity_exports", help="Output directory")
    ap.add_argument("--margin", type=int, default=6, help="Crop margin in voxels")
    ap.add_argument("--prefer_verse_ts_predict", action="store_true", help="Use VerSe/derivatives/<subject>/ts_predict.nii if present (optional)")
    ap.add_argument(
        "--per_vertebra_colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign material color per vertebra (GT lighter, TS darker).",
    )
    ap.add_argument(
        "--cache_unzipped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache uncompressed `.nii` next to outputs to speed up slicing `.nii.gz` (recommended).",
    )
    args = ap.parse_args()

    subject = args.subject
    if args.all or not args.vertebra:
        # Delegate to the all-vertebra exporter for convenience.
        all_script = PROJECT_ROOT / "scripts" / "export_unity_all_vertebrae_obj.py"
        if not all_script.exists():
            raise FileNotFoundError(f"All-vertebra exporter not found: {all_script}")

        argv = [
            str(all_script),
            "--subject",
            subject,
            "--out_dir",
            args.out_dir,
            "--margin",
            str(args.margin),
        ]
        # propagate booleans
        argv.append("--per_vertebra_colors" if args.per_vertebra_colors else "--no-per_vertebra_colors")
        argv.append("--cache_unzipped_gt" if args.cache_unzipped else "--no-cache_unzipped_gt")

        old_argv = sys.argv
        try:
            sys.argv = argv
            runpy.run_path(str(all_script), run_name="__main__")
        finally:
            sys.argv = old_argv
        return

    vertebra = args.vertebra.upper()
    label = vertebra_to_label(vertebra)

    gt_path = find_gt_path(subject)
    ts_path = find_ts_path(subject, vertebra)

    # Speed trick: `.nii.gz` random slice access is very slow; cache as `.nii` to enable memmap slicing.
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    cache_dir = out_dir / ".nifti_cache"
    gt_path_fast = maybe_cache_unzipped_nifti(gt_path, cache_dir=cache_dir, enabled=args.cache_unzipped)
    ts_path_fast = maybe_cache_unzipped_nifti(ts_path, cache_dir=cache_dir, enabled=args.cache_unzipped)

    gt_img = nib.load(str(gt_path_fast))
    ts_img = nib.load(str(ts_path_fast))

    if gt_img.shape != ts_img.shape:
        raise ValueError(f"Shape mismatch: GT {gt_img.shape} vs TS {ts_img.shape}")

    gt_proxy = gt_img.dataobj
    ts_proxy = ts_img.dataobj

    # BBoxes (streaming)
    gt_bbox = bbox_from_proxy(gt_proxy, predicate=lambda sl: sl == label, z_step=1)
    ts_bbox = bbox_from_proxy(ts_proxy, predicate=lambda sl: sl > 0, z_step=1)

    if gt_bbox is None:
        raise RuntimeError(f"GT label {label} ({vertebra}) not found in {gt_path}")
    if ts_bbox is None:
        raise RuntimeError(f"TS mask not found (empty) in {ts_path}")

    # Union bbox + margin
    ux0 = min(gt_bbox[0], ts_bbox[0])
    ux1 = max(gt_bbox[1], ts_bbox[1])
    uy0 = min(gt_bbox[2], ts_bbox[2])
    uy1 = max(gt_bbox[3], ts_bbox[3])
    uz0 = min(gt_bbox[4], ts_bbox[4])
    uz1 = max(gt_bbox[5], ts_bbox[5])
    union_bbox = (ux0, ux1, uy0, uy1, uz0, uz1)
    union_bbox = grow_and_clip_bbox(union_bbox, gt_img.shape, margin=args.margin)

    # Cropped binary volumes
    gt_crop = crop_binary_volume_from_proxy(gt_proxy, predicate=lambda sl: sl == label, bbox=union_bbox)
    ts_crop = crop_binary_volume_from_proxy(ts_proxy, predicate=lambda sl: sl > 0, bbox=union_bbox)

    # Mesh extraction
    gt_verts, gt_faces = marching_cubes_binary(gt_crop)
    ts_verts, ts_faces = marching_cubes_binary(ts_crop)

    # Map verts back to full-volume voxel indices, then to world coordinates via affine
    x0, _x1, y0, _y1, z0, _z1 = union_bbox
    offset = np.array([x0, y0, z0], dtype=np.float32)
    gt_world = apply_affine_to_vertices(gt_img.affine, gt_verts + offset).astype(np.float32)
    ts_world = apply_affine_to_vertices(gt_img.affine, ts_verts + offset).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{vertebra}_gt_ts"
    obj_path = out_dir / f"{stem}.obj"
    mtl_path = out_dir / f"{stem}.mtl"

    write_obj_with_mtl(
        obj_path=obj_path,
        mtl_path=mtl_path,
        parts=[
            {
                "name": f"{subject}_{vertebra}_GT",
                "material": (f"mat_GT_{vertebra}" if args.per_vertebra_colors else "mat_gt"),
                "vertices": gt_world,
                "faces": gt_faces,
            },
            {
                "name": f"{subject}_{vertebra}_TS",
                "material": (f"mat_TS_{vertebra}" if args.per_vertebra_colors else "mat_ts"),
                "vertices": ts_world,
                "faces": ts_faces,
            },
        ],
    )

    print("Wrote:")
    print(" ", obj_path)
    print(" ", mtl_path)
    print("Details:")
    print("  bbox (x0,x1,y0,y1,z0,z1):", union_bbox)
    print("  GT verts/faces:", int(gt_world.shape[0]), int(gt_faces.shape[0]))
    print("  TS verts/faces:", int(ts_world.shape[0]), int(ts_faces.shape[0]))
    if args.cache_unzipped:
        print("  Cached unzipped NIfTI dir:", cache_dir)


if __name__ == "__main__":
    main()



