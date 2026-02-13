#!/usr/bin/env python3
"""
Phase 0 sweep runner: run multiple random-corruption settings and summarize.

IMPORTANT (memory):
  Volumes here are huge (~583x512x1626). Doing a sweep in-process can OOM if we
  repeatedly allocate full-volume copies. To stay stable on shared HPC nodes,
  this runner executes each sweep config in a fresh subprocess by calling
  run_phase0_baselines.py and then aggregating its JSON outputs.

Sweeps over:
  - op: erode/dilate
  - radius: small integers
  - p_apply: probability of corrupting a given label

Outputs:
  - phase0_sweep_<subject>.csv (compact table)
  - phase0_sweep_<subject>.json (config + key numbers)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub-verse563")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ops", default="erode,dilate")
    ap.add_argument("--radii", default="1,2")
    ap.add_argument("--p_apply", default="0.25,0.5,0.75")
    ap.add_argument("--max_runs", type=int, default=0, help="If >0, limit number of sweep configs (debug/quick).")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, skip configs whose metrics JSON already exists (recommended).",
    )
    args = ap.parse_args()

    subject = args.subject
    repo_root = Path(__file__).resolve().parents[2]

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (repo_root / "spine-rl-sim" / "ablation_outputs" / str(date.today()) / f"phase0_sweep_{subject}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    radii = [int(x.strip()) for x in args.radii.split(",") if x.strip()]
    ps = [float(x.strip()) for x in args.p_apply.split(",") if x.strip()]

    # Run each config in a fresh subprocess to avoid OOM from repeated full-volume copies.
    baselines_script = Path(__file__).resolve().parent / "run_phase0_baselines.py"
    if not baselines_script.exists():
        raise FileNotFoundError(f"Expected baseline script next to this file: {baselines_script}")

    rows = []
    run_idx = 0
    jsonl_path = out_dir / f"phase0_sweep_{subject}.jsonl"
    for op in ops:
        for r in radii:
            for p_apply in ps:
                run_idx += 1
                if args.max_runs and run_idx > args.max_runs:
                    break
                cfg_out = out_dir / f"cfg_{op}_r{r}_p{p_apply:.2f}_seed{args.seed}"
                cfg_out.mkdir(parents=True, exist_ok=True)
                metrics_path = cfg_out / f"phase0_{subject}_mask_metrics.json"
                if args.resume and metrics_path.exists():
                    payload = json.loads(metrics_path.read_text())
                    clean_m = payload["clean"]
                    corrupt_m = payload["corrupt"]["metrics"]
                    row = {
                        "subject": subject,
                        "seed": args.seed,
                        "op": op,
                        "radius": r,
                        "p_apply": p_apply,
                        "clean_mean_dice": clean_m["mean_dice"],
                        "clean_mean_iou": clean_m["mean_iou"],
                        "corrupt_mean_dice": corrupt_m["mean_dice"],
                        "corrupt_mean_iou": corrupt_m["mean_iou"],
                        "delta_mean_dice": corrupt_m["mean_dice"] - clean_m["mean_dice"],
                        "delta_mean_iou": corrupt_m["mean_iou"] - clean_m["mean_iou"],
                    }
                    rows.append(row)
                    continue

                print(f"[run] {subject} op={op} radius={r} p_apply={p_apply:.2f} seed={args.seed}", flush=True)
                cmd = [
                    sys.executable,
                    str(baselines_script),
                    "--subject",
                    subject,
                    "--out_dir",
                    str(cfg_out),
                    "--seed",
                    str(args.seed),
                    "--corrupt_op",
                    op,
                    "--corrupt_radius",
                    str(r),
                    "--corrupt_p_apply",
                    str(p_apply),
                ]
                subprocess.run(cmd, check=True)

                payload = json.loads(metrics_path.read_text())
                clean_m = payload["clean"]
                corrupt_m = payload["corrupt"]["metrics"]
                row = {
                    "subject": subject,
                    "seed": args.seed,
                    "op": op,
                    "radius": r,
                    "p_apply": p_apply,
                    "clean_mean_dice": clean_m["mean_dice"],
                    "clean_mean_iou": clean_m["mean_iou"],
                    "corrupt_mean_dice": corrupt_m["mean_dice"],
                    "corrupt_mean_iou": corrupt_m["mean_iou"],
                    "delta_mean_dice": corrupt_m["mean_dice"] - clean_m["mean_dice"],
                    "delta_mean_iou": corrupt_m["mean_iou"] - clean_m["mean_iou"],
                }
                rows.append(row)
                # Incremental save (survives partial runs)
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if args.max_runs and run_idx >= args.max_runs:
                break
        if args.max_runs and run_idx >= args.max_runs:
            break

    csv_path = out_dir / f"phase0_sweep_{subject}.csv"
    # Write incrementally-friendly output: always rewrite CSV at the end from collected rows.
    # (Per-config JSON already exists under cfg_* for resume/debugging.)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / f"phase0_sweep_{subject}.json"
    json_path.write_text(
        json.dumps(
            {
                "subject": subject,
                "seed": args.seed,
                "sweep": {"ops": ops, "radii": radii, "p_apply": ps},
                "max_runs": args.max_runs,
                "rows": rows,
            },
            indent=2,
        )
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()


