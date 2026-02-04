#!/usr/bin/env python3
"""
Run pretrained Encoder + Spinal Field(v2) assembly model on TS-derived point clouds.

Pipeline:
  TS masks -> (points+normals) -> encoder embedding -> assembly model -> predicted vertebra IDs (+ PAD)

Outputs:
  outputs/evaluation/ts_spinal_field_predictions/<subject_id>.json
  outputs/evaluation/ts_spinal_field_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spine_point_cloud_assembly.models import SpineAssemblySpinalField  # noqa: E402
from spine_point_cloud_assembly.models.assembly_spinal_field_legacy import SpineAssemblySpinalFieldLegacy  # noqa: E402
from spine_point_cloud_assembly.scripts.extract_assembly_embeddings import load_encoder  # noqa: E402
from spine_point_cloud_assembly.models.encoder_se3 import features_to_irreps  # noqa: E402
from spine_point_cloud_assembly.utils.anatomy_transitions import nvc_transition, strict_nvc_transition  # noqa: E402
from spine_point_cloud_assembly.utils.soed import soed  # noqa: E402


def _load_ts_metrics(subject_id: str) -> dict:
    p = Path("/gscratch/scrubbed/june0604/vindr/totalseg_eval/metrics_results_total") / f"{subject_id}.json"
    return json.loads(p.read_text())


def _load_points_features(subject_dir: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out = []
    for pc in sorted(subject_dir.glob("vertebra_*_points.npy")):
        v_id = int(pc.stem.split("_")[1])
        pts = np.load(pc).astype(np.float32)
        feat_file = pc.parent / f"vertebra_{v_id}_features.npz"
        if feat_file.exists():
            d = np.load(feat_file)
            normals = d["normals"].astype(np.float32)
            k1 = d["k1"].astype(np.float32)
            k2 = d["k2"].astype(np.float32)
            curv = np.stack([k1, k2], axis=-1)
        else:
            normals = np.zeros((pts.shape[0], 3), dtype=np.float32)
            curv = np.zeros((pts.shape[0], 2), dtype=np.float32)
        feat = np.concatenate([normals, curv], axis=-1).astype(np.float32)  # (M,5)
        out.append((v_id, pts, feat))
    return out


def _load_spinal_field_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    # infer config from checkpoint if present (best-effort)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    embed_dim = int(cfg.get("embed_dim", 512))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_layers = int(cfg.get("num_layers", 6))
    num_heads = int(cfg.get("num_heads", 8))
    num_types = int(cfg.get("num_vertebra_types", 26))
    dropout = float(cfg.get("dropout", 0.1))
    sd = None
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if sd is None and isinstance(ckpt, dict):
        sd = ckpt
    if sd is None:
        raise ValueError("Invalid checkpoint format")

    # Detect spline-based vs legacy-by-keys
    has_control_points = any("control_points_head" in k for k in sd.keys())
    has_cond = any(k.startswith("cond.") for k in sd.keys())

    if not has_control_points:
        model = SpineAssemblySpinalFieldLegacy(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_vertebra_types=num_types,
            dropout=dropout,
            use_mask_token=True,
            enable_delta_pose=True,
            extra_dim=0,
            enable_cond=has_cond,
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    # Current model (spline-based)
    extra_dim = int(cfg.get("extra_dim", 0))
    enable_delta_pose = bool(cfg.get("enable_delta_pose", True))
    num_control_points = int(cfg.get("num_control_points", cfg.get("spline_num_ctrl", 8)))
    model = SpineAssemblySpinalField(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_vertebra_types=num_types,
        dropout=dropout,
        use_mask_token=True,
        extra_dim=extra_dim,
        enable_delta_pose=enable_delta_pose,
        num_control_points=num_control_points,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument(
        "--ts-pointcloud-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/point_clouds_ts"),
    )
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/embeddings/2026-01-11_17-50-10/best_model.pth"),
    )
    parser.add_argument(
        "--assembly-model-path",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/assembly/2026-01-12_21-07-15/best_model.pth"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gscratch/scrubbed/june0604/vindr/spine_point_cloud_assembly/outputs/evaluation/ts_spinal_field_predictions"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    sid = args.subject_id
    subj_dir = args.ts_pointcloud_dir / sid
    if not subj_dir.exists():
        raise FileNotFoundError(f"TS pointcloud dir not found: {subj_dir}")

    encoder, enc_cfg = load_encoder(args.encoder_path, device)
    assembly = _load_spinal_field_model(args.assembly_model_path, device)

    # Build embeddings for each vertebra instance in this subject
    items = _load_points_features(subj_dir)
    if len(items) == 0:
        raise RuntimeError(f"No point clouds found for {sid} in {subj_dir}")

    # We use a fixed set size for the assembly model (it is permutation-invariant; padding handled by pad_mask).
    maxN = 30
    embed_dim = 512
    embeddings = np.zeros((maxN, embed_dim), dtype=np.float32)
    present = np.zeros((maxN,), dtype=np.bool_)
    orig_ids = [-1] * maxN

    for i, (v_id, pts, feat) in enumerate(items[:maxN]):
        # Center points as in extract_assembly_embeddings
        pts_c = pts - pts.mean(axis=0, keepdims=True)
        # Subsample/pad to 2048 to match training
        M = 2048
        if pts_c.shape[0] > M:
            idx = np.random.choice(pts_c.shape[0], M, replace=False)
            pts_c = pts_c[idx]
            feat = feat[idx]
        elif pts_c.shape[0] < M:
            pad = M - pts_c.shape[0]
            pts_c = np.pad(pts_c, ((0, pad), (0, 0)), mode="constant")
            feat = np.pad(feat, ((0, pad), (0, 0)), mode="constant")

        pts_t = torch.from_numpy(pts_c).float().to(device).view(-1, 3)
        feat_t = torch.from_numpy(feat).float().to(device).unsqueeze(0)  # (1,M,5)
        feat_ir = features_to_irreps(feat_t, use_curvature=True)
        batch = torch.zeros((pts_t.shape[0],), dtype=torch.long, device=device)
        with torch.no_grad():
            out = encoder(pts_t, feat_ir, batch=batch)
            emb = out["embedding"].detach().cpu().numpy().reshape(-1)
        embeddings[i] = emb.astype(np.float32)
        present[i] = True
        orig_ids[i] = int(v_id)

    # Run assembly model
    emb_t = torch.from_numpy(embeddings).unsqueeze(0).to(device)  # (1,N,512)
    pad_mask = torch.from_numpy(~present).unsqueeze(0).to(device)  # True where PAD
    mask_mask = torch.zeros_like(pad_mask).bool()
    with torch.no_grad():
        pred = assembly(emb_t, pad_mask=pad_mask, mask_mask=mask_mask)

    logits = pred["ordering"].detach().cpu().numpy()[0]  # (N, num_types+1)
    s = pred["spine_field"]["s"].detach().cpu().numpy()[0, :, 0]  # (N,)
    pred_type = logits.argmax(axis=1).astype(int)  # 0..num_types (PAD is last index)
    pad_class = logits.shape[1] - 1

    # Build predicted sequence: keep non-PAD predictions, sort by s
    idx_keep = [i for i in range(maxN) if present[i] and pred_type[i] != pad_class]
    idx_keep.sort(key=lambda i: float(s[i]))
    # Convert type index -> label id (1-based): model uses 0..24 for C1..S1, pad=25
    pred_seq = [int(pred_type[i] + 1) for i in idx_keep]

    # Compare to GT order from totalseg_eval structural (already computed)
    mj = _load_ts_metrics(sid)
    gt_order = [int(x) for x in mj.get("structural", {}).get("gt_order", [])]
    # Use direction-agnostic metrics
    nvc_val = nvc_transition(pred_seq, consider_direction=False) if pred_seq else 0.0
    strict_val = strict_nvc_transition(pred_seq, consider_direction=False) if pred_seq else 0.0
    soed_val = soed(gt_order, pred_seq, normalize_by="gt") if gt_order and pred_seq else float("nan")

    out = {
        "subject_id": sid,
        "orig_instance_ids": orig_ids,
        "present_count": int(present.sum()),
        "pred_seq": pred_seq,
        "pred_len": len(pred_seq),
        "metrics": {"nvc": float(nvc_val), "strict_nvc": float(strict_val), "soed": float(soed_val)},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{sid}.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()


