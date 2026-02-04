#!/usr/bin/env python3
"""
Create summary figure of improved vs worsened CT overlays.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_scores(subject_dir: Path):
    scores = []
    for npz_path in sorted(subject_dir.glob("ct_overlay_compare_vertebra_*.npz")):
        data = np.load(npz_path)
        asm = float(data.get("asm_overlap", np.nan))
        raw = float(data.get("raw_overlap", np.nan))
        delta = asm - raw if np.isfinite(asm) and np.isfinite(raw) else np.nan
        v_id = int(npz_path.stem.split("_")[-1])
        scores.append((v_id, raw, asm, delta))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Summarize improved/worsened overlays")
    parser.add_argument("--subject_dir", type=str, required=True,
                        help="Directory with ct_overlay_compare_vertebra_*.png/.npz")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of best/worst cases to show")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    args = parser.parse_args()

    subject_dir = Path(args.subject_dir)
    scores = load_scores(subject_dir)
    if not scores:
        raise FileNotFoundError(f"No overlay npz found in {subject_dir}")

    scores = [s for s in scores if np.isfinite(s[3])]
    scores.sort(key=lambda x: x[3], reverse=True)
    best = scores[:args.top_k]
    worst = list(reversed(scores[-args.top_k:]))

    ncols = args.top_k
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=200)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    def plot_row(row_axes, rows, title_prefix):
        for ax, (v_id, raw, asm, delta) in zip(row_axes, rows):
            img_path = subject_dir / f"ct_overlay_compare_vertebra_{v_id}.png"
            if not img_path.exists():
                ax.axis("off")
                continue
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{title_prefix} V{v_id}  Δ={delta:+.3f}", fontsize=12)
            ax.axis("off")

    plot_row(axes[0], best, "Improved")
    plot_row(axes[1], worst, "Worsened")

    fig.tight_layout()
    output = Path(args.output) if args.output else subject_dir / "overlay_summary.png"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"✓ Saved summary: {output}")


if __name__ == "__main__":
    main()

