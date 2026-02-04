#!/usr/bin/env python3
"""
TotalSegmentator 예측 마스크를 원본 CT 위에 overlay하여 sagittal/axial 뷰로 저장.

- 입력:
  - predictions_total/<subject_id>/vertebrae_*.nii.gz (바이너리 마스크들)
  - VerSe/dataset-03test/rawdata/<subject_id>/*_ct.nii.gz (원본 CT)
  - (optional) VerSe/dataset-03test/derivatives/<subject_id>/*_seg-vert_msk.nii.gz (GT)

- 출력:
  - 각 view마다 3장 저장:
    - <out_dir>/<subject_id>_<view>_raw.png
    - <out_dir>/<subject_id>_<view>_gt.png
    - <out_dir>/<subject_id>_<view>_ts.png
    - <out_dir>/<subject_id>_<view>_combined.png  (RAW | RAW+GT | RAW+TS)

방향(orientation):
  - CT와 mask를 각각 nibabel의 as_closest_canonical()로 RAS에 가깝게 정렬
  - mask가 CT와 shape/affine이 다르면 CT로 nearest neighbor resample

슬라이스 선택:
  - 메모리 안정성을 위해 기본은 가운데 슬라이스 사용
    - sagittal: x-index = shape[0] // 2
    - axial: z-index = shape[2] // 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _lazy_imports():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    import nibabel as nib  # noqa: F401
    import nibabel.processing  # noqa: F401

    return matplotlib


def _load_canonical_nifti(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    img_c = nib.as_closest_canonical(img)
    # NOTE: 대용량 CT는 get_fdata()로 3D 전체를 로드하면 OOM이 날 수 있어
    # 여기서는 이미지 객체(proxy)를 반환하고, 실제 데이터는 필요한 slice만 읽는다.
    return img_c


def _resample_mask_to_image(mask_img, target_img):
    import nibabel as nib
    import nibabel.processing

    # order=0: nearest neighbor
    res = nibabel.processing.resample_from_to(mask_img, target_img, order=0)
    return res


def _vertebra_label_map() -> dict[str, int]:
    labels: dict[str, int] = {}
    for i in range(1, 8):
        labels[f"C{i}"] = i
    for i in range(1, 13):
        labels[f"T{i}"] = 7 + i
    for i in range(1, 6):
        labels[f"L{i}"] = 19 + i
    labels["S1"] = 25
    return labels


def _inv_label_map() -> dict[int, str]:
    m = _vertebra_label_map()
    return {v: k for k, v in m.items()}


def _merge_pred_dir_multiclass(
    pred_dir: Path, pattern: str = "vertebrae_*.nii.gz"
) -> tuple[np.ndarray, "nib.Nifti1Image", dict[int, str]]:
    import nibabel as nib

    candidates = sorted(pred_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No masks matching {pattern} in {pred_dir}")

    base = nib.load(str(candidates[0]))
    merged = np.zeros(base.shape, dtype=np.uint16)
    name_to_id = _vertebra_label_map()
    id_to_name = _inv_label_map()

    for p in candidates:
        name = p.name.replace("vertebrae_", "")
        if name.endswith(".nii.gz"):
            name = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            name = name[: -len(".nii")]
        label_id = name_to_id.get(name)
        if label_id is None:
            continue
        m = nib.load(str(p)).get_fdata() > 0
        merged[m] = int(label_id)

    merged_img = nib.Nifti1Image(merged.astype(np.uint16), base.affine, base.header)
    return merged, merged_img, id_to_name


def _mask_center(mask: np.ndarray) -> tuple[int, int, int] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    c = coords.mean(axis=0)
    return int(round(float(c[0]))), int(round(float(c[1]))), int(round(float(c[2])))


def _clip01(img2d: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    a = np.asarray(img2d, dtype=np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo:
        a = a - a.min()
        denom = a.max() - a.min()
        return a / (denom + 1e-8)
    a = np.clip(a, lo, hi)
    return (a - lo) / (hi - lo + 1e-8)


def _label_centroids_3d(mask_mc: np.ndarray) -> dict[int, np.ndarray]:
    """
    multiclass mask에서 label별 3D centroid (x, y, z)를 반환.
    """
    centroids: dict[int, np.ndarray] = {}
    labels = np.unique(mask_mc)
    labels = labels[labels != 0]
    for lab in labels:
        coords = np.argwhere(mask_mc == lab)
        if coords.size == 0:
            continue
        centroids[int(lab)] = coords.mean(axis=0)
    return centroids


def _label_centroids_2d_from_slice(mk2: np.ndarray) -> dict[int, tuple[float, float]]:
    """
    2D 슬라이스 상에서 label별 centroid를 계산.

    mk2: plot에 그대로 올리는 2D 배열(이미 transpose된 상태)
         - sagittal: (S, A)
         - axial: (A, R)

    Returns:
      dict[label] = (x, y) in plot coordinates
    """
    out: dict[int, tuple[float, float]] = {}
    labs = np.unique(mk2)
    labs = labs[labs != 0]
    for lab in labs:
        coords = np.argwhere(mk2 == lab)  # (row, col)
        if coords.size == 0:
            continue
        cy, cx = coords.mean(axis=0)  # row->y, col->x
        out[int(lab)] = (float(cx), float(cy))
    return out


def _annotate_labels_on_slice(
    ax,
    *,
    view: str,
    slice_idx: int,
    centroids: dict[int, np.ndarray] | None,
    centroids_2d: dict[int, tuple[float, float]] | None,
    id_to_name: dict[int, str] | None,
    text_color: str = "cyan",
):
    """
    현재 view/slice에 걸리는 label 텍스트 표시.
    - centroids_2d가 있으면(권장) 2D 슬라이스에서 보이는 label만 표시(빠짐 방지)
    - 없으면 centroids(3D)로 fallback (기존 방식)

    canonical 좌표: (x=R, y=A, z=S)
    """
    if centroids_2d:
        for lab, (x_plot, y_plot) in centroids_2d.items():
            name = id_to_name.get(lab, str(lab)) if id_to_name else str(lab)
            ax.text(
                x_plot,
                y_plot,
                name,
                color=text_color,
                fontsize=7,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5),
                ha="center",
                va="center",
            )
        return

    if not centroids:
        return

    # fallback: 3D centroid 기반 (이 경우 일부 라벨이 빠질 수 있음)
    for lab, c in centroids.items():
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        name = id_to_name.get(lab, str(lab)) if id_to_name else str(lab)

        if view == "sagittal":
            x_plot = cy  # A
            y_plot = cz  # S
        elif view == "axial":
            x_plot = cx  # R
            y_plot = cy  # A
        else:
            continue

        ax.text(
            x_plot,
            y_plot,
            name,
            color=text_color,
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5),
            ha="center",
            va="center",
        )


def _save_panel(
    *,
    ct2: np.ndarray,
    mk2: np.ndarray | None,
    out_path: Path,
    alpha: float = 0.35,
    contour: bool = True,
    title: str | None = None,
    id_to_name: dict[int, str] | None = None,
):
    """
    2D 슬라이스(이미 transpose된 상태)를 받아 저장.
    - ct2: (H, W)
    - mk2: (H, W) multiclass 또는 None
    """
    import matplotlib.pyplot as plt

    assert ct2.ndim == 2
    if mk2 is not None:
        assert mk2.ndim == 2

    ct2n = _clip01(ct2)
    h, w = ct2n.shape
    aspect = h / max(w, 1)

    fig_w = 7.5
    fig_h = max(6.0, min(12.0, fig_w * aspect))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(ct2n, cmap="gray", origin="lower", aspect="auto")
    if mk2 is not None:
        ax.imshow((mk2 > 0).astype(np.float32), cmap="autumn", origin="lower", alpha=alpha, aspect="auto")
        if contour:
            try:
                ax.contour((mk2 > 0).astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=1.0, origin="lower")
            except Exception:
                pass

        centroids2d = _label_centroids_2d_from_slice(mk2.astype(np.int32, copy=False))
        _annotate_labels_on_slice(
            ax,
            view="",
            slice_idx=0,
            centroids=None,
            centroids_2d=centroids2d,
            id_to_name=id_to_name,
            text_color="cyan",
        )

    ax.set_title(title or f"{out_path.stem}", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_combined_triptych(
    *,
    subject_id: str,
    view: str,
    ct2: np.ndarray,
    gt2: np.ndarray | None,
    ts2: np.ndarray,
    out_path: Path,
    alpha: float,
    id_to_name: dict[int, str] | None,
):
    """
    한 장에 RAW | RAW+GT | RAW+TS 를 순서대로 저장.
    - 동일 slice_idx를 사용
    - GT가 없으면 가운데 칸은 RAW로 대체
    """
    import matplotlib.pyplot as plt

    ct2n = _clip01(ct2)
    h, w = ct2n.shape
    aspect = h / max(w, 1)

    fig_w = 18.0
    fig_h = max(6.0, min(12.0, (fig_w / 3.0) * aspect))
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    def _draw(ax, title: str, mk2: np.ndarray | None, draw_contour: bool):
        ax.imshow(ct2n, cmap="gray", origin="lower", aspect="auto")
        if mk2 is not None:
            ax.imshow((mk2 > 0).astype(np.float32), cmap="autumn", origin="lower", alpha=alpha, aspect="auto")
            if draw_contour:
                try:
                    ax.contour((mk2 > 0).astype(np.uint8), levels=[0.5], colors=["yellow"], linewidths=1.0, origin="lower")
                except Exception:
                    pass

            centroids2d = _label_centroids_2d_from_slice(mk2.astype(np.int32, copy=False))
            _annotate_labels_on_slice(
                ax,
                view="",
                slice_idx=0,
                centroids=None,  # 3D centroid는 메모리/시간이 커서 기본 사용하지 않음
                centroids_2d=centroids2d,
                id_to_name=id_to_name,
                text_color="cyan",
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)

    _draw(axes[0], "RAW", None, False)
    if gt2 is not None:
        _draw(axes[1], "RAW + GT", gt2, True)
    else:
        _draw(axes[1], "RAW (GT 없음)", None, False)
    _draw(axes[2], "RAW + TotalSeg(Pred)", ts2, True)

    fig.suptitle(f"{subject_id} ({view})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _default_ct_path(project_root: Path, subject_id: str) -> Path:
    raw_dir = project_root / "VerSe" / "dataset-03test" / "rawdata" / subject_id
    cts = sorted(raw_dir.glob(f"{subject_id}_*_ct.nii.gz"))
    if not cts:
        raise FileNotFoundError(f"Cannot find CT in {raw_dir} (expected {subject_id}_*_ct.nii.gz)")
    return cts[0]


def _default_gt_path(project_root: Path, subject_id: str) -> Path | None:
    # 1) metrics_results_total json에 gt_path가 있으면 사용
    metrics_json = project_root / "totalseg_eval" / "metrics_results_total" / f"{subject_id}.json"
    if metrics_json.exists():
        try:
            import json

            j = json.loads(metrics_json.read_text())
            gt_path = j.get("gt_path")
            if gt_path:
                p = Path(gt_path)
                if p.exists():
                    return p
        except Exception:
            pass

    # 2) derivatives에서 패턴으로 검색
    der_dir = project_root / "VerSe" / "dataset-03test" / "derivatives" / subject_id
    cands = sorted(der_dir.glob(f"{subject_id}_*_seg-vert_msk.nii.gz"))
    if not cands:
        return None
    return cands[0]


def _ct_slice_2d(ct_img_c, view: str, idx: int) -> np.ndarray:
    """
    canonical CT 이미지(proxy)에서 필요한 2D 슬라이스만 읽어온다.
    반환은 plot-ready(이미 transpose된) 2D array.
    """
    if view == "sagittal":
        return np.asanyarray(ct_img_c.dataobj[idx, :, :], dtype=np.float32).T  # (S, A)
    if view == "axial":
        return np.asanyarray(ct_img_c.dataobj[:, :, idx], dtype=np.float32).T  # (A, R)
    raise ValueError(view)


def _mask_slice_2d_from_multifiles(
    pred_dir: Path,
    *,
    view: str,
    idx: int,
    shape2d: tuple[int, int],
    pattern: str = "vertebrae_*.nii.gz",
) -> tuple[np.ndarray, dict[int, str]]:
    """
    vertebrae_*.nii.gz 파일들을 순회하면서 현재 슬라이스에서의 2D multiclass mask를 만든다.
    (3D 전체 merge를 만들지 않아 메모리 사용량이 매우 작다)
    """
    import nibabel as nib

    name_to_id = _vertebra_label_map()
    id_to_name = _inv_label_map()

    mk2 = np.zeros(shape2d, dtype=np.int32)
    for p in sorted(pred_dir.glob(pattern)):
        name = p.name.replace("vertebrae_", "")
        if name.endswith(".nii.gz"):
            name = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            name = name[: -len(".nii")]
        label_id = name_to_id.get(name)
        if label_id is None:
            continue

        img = nib.as_closest_canonical(nib.load(str(p)))
        if view == "sagittal":
            sl = np.asanyarray(img.dataobj[idx, :, :], dtype=np.float32).T
        elif view == "axial":
            sl = np.asanyarray(img.dataobj[:, :, idx], dtype=np.float32).T
        else:
            raise ValueError(view)

        mk2[sl > 0] = int(label_id)

    return mk2, id_to_name


def _mask_slice_2d_from_single_mask(mask_img_c, view: str, idx: int) -> np.ndarray:
    if view == "sagittal":
        return np.asanyarray(mask_img_c.dataobj[idx, :, :], dtype=np.float32).T.astype(np.int32)
    if view == "axial":
        return np.asanyarray(mask_img_c.dataobj[:, :, idx], dtype=np.float32).T.astype(np.int32)
    raise ValueError(view)


def visualize_subject(
    *,
    project_root: Path,
    subject_id: str,
    pred_root: Path,
    out_dir: Path,
    mask_pattern: str = "vertebrae_*.nii.gz",
    views: Iterable[str] = ("sagittal", "axial"),
    alpha: float = 0.35,
):
    _lazy_imports()
    import nibabel as nib

    pred_dir = pred_root / subject_id
    ct_path = _default_ct_path(project_root, subject_id)
    gt_path = _default_gt_path(project_root, subject_id)

    ct_img = _load_canonical_nifti(ct_path)

    gt_img_c = None
    if gt_path is not None and gt_path.exists():
        gt_img_c = _load_canonical_nifti(gt_path)
        # 메모리 안전을 위해 shape이 다르면 GT overlay는 생략(필요하면 추후 옵션으로 resample)
        if gt_img_c.shape != ct_img.shape or not np.allclose(gt_img_c.affine, ct_img.affine, atol=1e-3):
            gt_img_c = None

    # 기본은 중앙 슬라이스
    cx, cy, cz = ct_img.shape[0] // 2, ct_img.shape[1] // 2, ct_img.shape[2] // 2

    for v in views:
        if v == "sagittal":
            idx = cx
        elif v == "axial":
            idx = cz
        else:
            raise ValueError(f"Unsupported view: {v} (supported: sagittal, axial)")

        ct2 = _ct_slice_2d(ct_img, v, idx)
        ts2, id_to_name = _mask_slice_2d_from_multifiles(
            pred_dir, view=v, idx=idx, shape2d=ct2.shape, pattern=mask_pattern
        )
        gt2 = None if gt_img_c is None else _mask_slice_2d_from_single_mask(gt_img_c, v, idx)

        # 1) raw
        _save_panel(
            ct2=ct2,
            mk2=None,
            out_path=out_dir / f"{subject_id}_{v}_raw.png",
            alpha=alpha,
            contour=False,
            title=f"{subject_id} ({v}) RAW",
        )

        # 2) raw + GT (있을 때만)
        if gt2 is not None:
            _save_panel(
                ct2=ct2,
                mk2=gt2,
                out_path=out_dir / f"{subject_id}_{v}_gt.png",
                alpha=alpha,
                contour=True,
                title=f"{subject_id} ({v}) RAW + GT",
                id_to_name=id_to_name,
            )

        # 3) raw + TS
        _save_panel(
            ct2=ct2,
            mk2=ts2,
            out_path=out_dir / f"{subject_id}_{v}_ts.png",
            alpha=alpha,
            contour=True,
            title=f"{subject_id} ({v}) RAW + TotalSeg(Pred)",
            id_to_name=id_to_name,
        )

        # 4) combined (RAW | RAW+GT | RAW+TS)
        _save_combined_triptych(
            subject_id=subject_id,
            view=v,
            ct2=ct2,
            gt2=gt2,
            ts2=ts2,
            out_path=out_dir / f"{subject_id}_{v}_combined.png",
            alpha=alpha,
            id_to_name=id_to_name,
        )

    return {
        "subject_id": subject_id,
        "ct_path": str(ct_path),
        "pred_dir": str(pred_dir),
        "out_dir": str(out_dir),
        "mask_pattern": mask_pattern,
        "center_xyz": [int(cx), int(cy), int(cz)],
        "ct_shape": list(map(int, ct_img.shape)),
        "gt_path": None if gt_path is None else str(gt_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/gscratch/scrubbed/june0604/vindr"))
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/totalseg_eval/predictions_total"),
    )
    parser.add_argument("--subject-id", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/totalseg_eval/overlay_visualizations"),
    )
    parser.add_argument("--mask-pattern", type=str, default="vertebrae_*.nii.gz")
    parser.add_argument("--views", type=str, default="sagittal,axial")
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    views = [v.strip().lower() for v in args.views.split(",") if v.strip()]
    info = visualize_subject(
        project_root=args.project_root,
        subject_id=args.subject_id,
        pred_root=args.pred_root,
        out_dir=args.out_dir,
        mask_pattern=args.mask_pattern,
        views=views,
        alpha=args.alpha,
    )
    print(info)


if __name__ == "__main__":
    main()


