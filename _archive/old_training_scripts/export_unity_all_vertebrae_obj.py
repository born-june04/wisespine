#!/usr/bin/env python3
"""
Export ALL vertebrae (GT + TS) as ONE OBJ (+ MTL) for Unity.

Why this is fast / memory-safe:
  - Never allocates a full-size merged volume (which would be multi-GB).
  - Streams through z-slices to compute per-label bounding boxes.
  - For each vertebra label, crops a small subvolume and runs marching cubes there.

Default inputs for this repo layout:
  - GT: VerSe/dataset-03test/derivatives/<subject>/<subject>_dir-iso_seg-vert_msk.nii.gz
  - TS (preferred): VerSe/dataset-03test/derivatives/<subject>/ts_predict.nii  (single multi-label map)

Example:
  python scripts/export_unity_all_vertebrae_obj.py \
    --subject sub-verse563 \
    --out_dir outputs/unity_exports \
    --margin 6

Notes:
  - Colors: by default each vertebra gets its own material/color (GT brighter, TS darker).
    Disable with: --no-per_vertebra_colors
"""

from __future__ import annotations

import argparse
import colorsys
import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

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

LABELS = list(range(1, 26))  # VerSe vertebra labels 1..25


def label_to_vertebra(label: int) -> str:
    if 1 <= label <= 7:
        return f"C{label}"
    if 8 <= label <= 19:
        return f"T{label - 7}"
    if 20 <= label <= 24:
        return f"L{label - 19}"
    if label == 25:
        return "S1"
    return f"V{label}"


def find_gt_path(subject: str) -> Path:
    p = PROJECT_ROOT / "VerSe" / "dataset-03test" / "derivatives" / subject / f"{subject}_dir-iso_seg-vert_msk.nii.gz"
    if not p.exists():
        raise FileNotFoundError(f"GT file not found: {p}")
    return p


def find_ts_predict_path(subject: str) -> Path:
    p = PROJECT_ROOT / "VerSe" / "dataset-03test" / "derivatives" / subject / "ts_predict.nii"
    if not p.exists():
        raise FileNotFoundError(
            f"TS multi-label prediction not found: {p}\n"
            "Tip: this exporter currently prefers VerSe/derivatives/<subject>/ts_predict.nii to avoid opening 25 huge vertebrae_*.nii.gz files."
        )
    return p


def find_totalseg_dir(subject: str) -> Path:
    p = PROJECT_ROOT / "totalseg_eval" / "predictions_total" / subject
    if not p.exists():
        raise FileNotFoundError(f"TotalSeg directory not found: {p}")
    return p


def maybe_cache_unzipped_nifti(path: Path, cache_dir: Path, enabled: bool) -> Path:
    """
    `.nii.gz` random slice access can be extremely slow. Cache as `.nii` for memmap slicing.
    Only used for GT (single file) by default to avoid exploding disk by caching many TS files.
    """
    if not enabled:
        return path
    if not path.name.endswith(".nii.gz"):
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / path.name[: -len(".gz")]
    if out.exists() and out.stat().st_size > 0:
        return out

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


def compute_bboxes_for_labels_streaming(
    dataobj,
    labels: list[int],
    z_step: int = 1,
) -> Dict[int, Optional[Tuple[int, int, int, int, int, int]]]:
    """
    One-pass bbox extraction for multiple labels.
    Returns dict[label] = (x0,x1,y0,y1,z0,z1) or None if absent.
    """
    shape = dataobj.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {shape}")

    wanted = set(int(l) for l in labels)
    bboxes: Dict[int, Optional[Tuple[int, int, int, int, int, int]]] = {int(l): None for l in labels}

    for z in range(0, shape[2], z_step):
        sl = np.asanyarray(dataobj[:, :, z])
        if sl.dtype.kind == "f":
            sl = sl.astype(np.int32, copy=False)

        present = np.unique(sl)
        # iterate only labels present in this slice
        for lab in present.tolist():
            lab_int = int(lab)
            if lab_int == 0 or lab_int not in wanted:
                continue
            xs, ys = np.where(sl == lab_int)
            if xs.size == 0:
                continue
            bboxes[lab_int] = _update_bbox(bboxes[lab_int], xs, ys, z)

    return bboxes


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


def union_bbox(a: Tuple[int, int, int, int, int, int], b: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int, int, int, int]:
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
        min(a[4], b[4]),
        max(a[5], b[5]),
    )


def crop_binary_volume_from_proxy(
    dataobj,
    label: int,
    bbox: Tuple[int, int, int, int, int, int],
) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bbox
    out = np.zeros((x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1), dtype=np.uint8)
    for z in range(z0, z1 + 1):
        sl = np.asanyarray(dataobj[x0 : x1 + 1, y0 : y1 + 1, z])
        out[:, :, z - z0] = (sl == label).astype(np.uint8, copy=False)
    return out


def crop_binary_volume_from_proxy_predicate(
    dataobj,
    predicate,
    bbox: Tuple[int, int, int, int, int, int],
) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bbox
    out = np.zeros((x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1), dtype=np.uint8)
    for z in range(z0, z1 + 1):
        sl = np.asanyarray(dataobj[x0 : x1 + 1, y0 : y1 + 1, z])
        out[:, :, z - z0] = predicate(sl).astype(np.uint8, copy=False)
    return out


def apply_affine_to_vertices(affine: np.ndarray, verts_ijk: np.ndarray) -> np.ndarray:
    r = affine[:3, :3].astype(np.float64)
    t = affine[:3, 3].astype(np.float64)
    return (verts_ijk.astype(np.float64) @ r.T) + t


def marching_cubes_binary(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _normals, _values = measure.marching_cubes(vol, level=0.5)
    return verts.astype(np.float32), faces.astype(np.int32)


def _material_color_from_name(mat_name: str, n_labels: int = 25) -> tuple[float, float, float]:
    """
    Generate deterministic per-vertebra colors from material name.

    Expected names:
      - mat_GT_C1, mat_GT_T12, mat_GT_L1, mat_GT_S1
      - mat_TS_C1, ...

    We color by vertebra identity (hue), and slightly darken TS vs GT to distinguish overlays.
    """
    # Extract vertebra part (e.g., C1)
    if mat_name.startswith("mat_GT_"):
        kind = "GT"
        vname = mat_name[len("mat_GT_") :]
    elif mat_name.startswith("mat_TS_"):
        kind = "TS"
        vname = mat_name[len("mat_TS_") :]
    else:
        kind = "OTHER"
        vname = mat_name

    # Map vname -> label index for hue
    # Fall back to hashing if unexpected
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

    if label is None or not (1 <= label <= n_labels):
        # cheap deterministic fallback
        h = (abs(hash(vname)) % 360) / 360.0
    else:
        h = (label - 1) / float(n_labels)

    s = 0.75
    v = 0.95 if kind == "GT" else 0.75 if kind == "TS" else 0.85
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return float(r), float(g), float(b)


def write_obj_with_mtl(obj_path: Path, mtl_path: Path, parts: list[dict]) -> None:
    mtl_lines = ["# Unity-friendly materials (one per vertebra)"]
    materials = sorted({p["material"] for p in parts})
    for mat in materials:
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

    lines = ["# Combined GT + TS vertebrae OBJ", f"mtllib {mtl_path.name}", ""]
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
    ap.add_argument("--out_dir", default="outputs/unity_exports", help="Output directory")
    ap.add_argument("--margin", type=int, default=6, help="Crop margin in voxels")
    ap.add_argument("--z_step_bbox", type=int, default=1, help="Z step for bbox scan (1 = exact, >1 = faster/rougher)")
    ap.add_argument(
        "--per_vertebra_colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign different material colors per vertebra (recommended for Unity).",
    )
    ap.add_argument(
        "--cache_unzipped_gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache uncompressed `.nii` for GT `.nii.gz` to speed slicing (recommended).",
    )
    ap.add_argument(
        "--ts_source",
        choices=["auto", "verse_ts_predict", "totalseg_dir"],
        default="auto",
        help="TS source: VerSe ts_predict.nii (multi-label) or TotalSeg per-vertebra directory. 'auto' falls back if ts_predict doesn't contain labels.",
    )
    ap.add_argument("--labels", default="all", help="Comma list like 'C1,C2,T12,L1' or 'all'")
    args = ap.parse_args()

    subject = args.subject
    out_dir = (PROJECT_ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".nifti_cache"

    gt_path = find_gt_path(subject)
    gt_path_fast = maybe_cache_unzipped_nifti(gt_path, cache_dir=cache_dir, enabled=args.cache_unzipped_gt)
    gt_img = nib.load(str(gt_path_fast))

    # label selection
    if args.labels.strip().lower() == "all":
        labels = LABELS
    else:
        items = [x.strip().upper() for x in args.labels.split(",") if x.strip()]
        name_to_label = {}
        for l in LABELS:
            name_to_label[label_to_vertebra(l)] = l
        labels = []
        for it in items:
            if it not in name_to_label:
                raise ValueError(f"Unknown label '{it}'. Expected one of: {sorted(name_to_label.keys())} or 'all'")
            labels.append(name_to_label[it])

    # Compute bboxes for each label for both volumes in one pass each
    gt_bboxes = compute_bboxes_for_labels_streaming(gt_img.dataobj, labels=labels, z_step=args.z_step_bbox)

    # TS source selection
    ts_mode = args.ts_source
    ts_img = None
    ts_bboxes = {int(l): None for l in labels}
    totalseg_dir = None

    if ts_mode in ("auto", "verse_ts_predict"):
        try:
            ts_path = find_ts_predict_path(subject)
            ts_img = nib.load(str(ts_path))  # .nii in this repo
            if gt_img.shape != ts_img.shape:
                raise ValueError(f"Shape mismatch: GT {gt_img.shape} vs ts_predict {ts_img.shape}")
            ts_bboxes = compute_bboxes_for_labels_streaming(ts_img.dataobj, labels=labels, z_step=args.z_step_bbox)
            found = sum(1 for v in ts_bboxes.values() if v is not None)
            if ts_mode == "verse_ts_predict" and found == 0:
                raise RuntimeError("ts_predict.nii did not contain any expected labels 1..25")
            if ts_mode == "auto" and found == 0:
                ts_img = None  # trigger fallback
        except Exception:
            if ts_mode == "verse_ts_predict":
                raise
            ts_img = None

    if ts_img is None:
        # Fallback to TotalSeg per-vertebra directory
        totalseg_dir = find_totalseg_dir(subject)

    parts = []
    for lab in labels:
        vname = label_to_vertebra(lab)
        gt_bb = gt_bboxes.get(lab)
        if gt_bb is None:
            continue

        # Default: use GT bbox + margin
        bb = grow_and_clip_bbox(gt_bb, gt_img.shape, margin=args.margin)

        # Crop binary volumes and mesh them
        gt_crop = crop_binary_volume_from_proxy(gt_img.dataobj, label=lab, bbox=bb)
        ts_crop = None
        if ts_img is not None:
            # Multi-label TS prediction in the same 1..25 labeling scheme
            ts_crop = crop_binary_volume_from_proxy(ts_img.dataobj, label=lab, bbox=bb)
        else:
            # TotalSeg per-vertebra binary mask; only read within GT bbox window (fast enough).
            assert totalseg_dir is not None
            ts_file = totalseg_dir / f"vertebrae_{vname}.nii.gz"
            if ts_file.exists():
                ts_bin = nib.load(str(ts_file))
                if ts_bin.shape == gt_img.shape:
                    ts_crop = crop_binary_volume_from_proxy_predicate(ts_bin.dataobj, predicate=lambda sl: sl > 0, bbox=bb)

        # Skip empty crops (just in case)
        if gt_crop.sum() > 0:
            gt_verts, gt_faces = marching_cubes_binary(gt_crop)
            x0, _x1, y0, _y1, z0, _z1 = bb
            offset = np.array([x0, y0, z0], dtype=np.float32)
            gt_world = apply_affine_to_vertices(gt_img.affine, gt_verts + offset).astype(np.float32)
            parts.append(
                {
                    "name": f"{subject}_{vname}_GT",
                    "material": (f"mat_GT_{vname}" if args.per_vertebra_colors else "mat_gt"),
                    "vertices": gt_world,
                    "faces": gt_faces,
                }
            )

        if ts_crop is not None and ts_crop.sum() > 0:
            ts_verts, ts_faces = marching_cubes_binary(ts_crop)
            x0, _x1, y0, _y1, z0, _z1 = bb
            offset = np.array([x0, y0, z0], dtype=np.float32)
            # If we used per-vertebra TotalSeg file, use its affine; otherwise use ts_predict affine.
            if ts_img is not None:
                ts_aff = ts_img.affine
            else:
                ts_aff = gt_img.affine
            ts_world = apply_affine_to_vertices(ts_aff, ts_verts + offset).astype(np.float32)
            parts.append(
                {
                    "name": f"{subject}_{vname}_TS",
                    "material": (f"mat_TS_{vname}" if args.per_vertebra_colors else "mat_ts"),
                    "vertices": ts_world,
                    "faces": ts_faces,
                }
            )

    stem = f"{subject}_ALL_gt_ts"
    obj_path = out_dir / f"{stem}.obj"
    mtl_path = out_dir / f"{stem}.mtl"
    write_obj_with_mtl(obj_path=obj_path, mtl_path=mtl_path, parts=parts)

    print("Wrote:")
    print(" ", obj_path)
    print(" ", mtl_path)
    print("Parts:", len(parts))
    if args.cache_unzipped_gt:
        print("GT cache dir:", cache_dir)


if __name__ == "__main__":
    main()


