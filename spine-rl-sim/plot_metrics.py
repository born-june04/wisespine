from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    train_csv = Path("/Users/june/spine-rl-sim/metrics/train_metrics.csv")
    eval_csv = Path("/Users/june/spine-rl-sim/metrics/eval_metrics.csv")
    out_path = Path("/Users/june/spine-rl-sim/metrics/metrics_plot.png")

    if not train_csv.exists() or not eval_csv.exists():
        raise SystemExit("Missing metrics CSVs. Run train_ppo.py first.")

    train = pd.read_csv(train_csv)
    eval_ = pd.read_csv(eval_csv)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_title("Training vs Validation Metrics")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Success Rate")
    ax1.plot(train["timesteps"], train["success_rate"], label="train success", color="#1f77b4")
    ax1.plot(eval_["timesteps"], eval_["success_rate"], label="val success", color="#ff7f0e")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg Return")
    ax2.plot(train["timesteps"], train["avg_return"], label="train return", color="#2ca02c", linestyle="--")
    ax2.plot(eval_["timesteps"], eval_["avg_return"], label="val return", color="#d62728", linestyle="--")
    ax2.tick_params(axis="y")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
