#!/usr/bin/env python3
"""
plot_baseline_stability.py

Reads telemetry/telemetry_fp16_baseline.csv (produced by fp16_baseline.py, the
standalone matched-baseline-only script) and plots trial-to-trial variance for
throughput and energy/token, grouped by prompt category (Poem/Physics/Code).

NOTE: This is a separate data source from academic_validation_results.json
(the benchmark_ablation.py output). This script's baseline power readings run
notably lower (~407-433W) than the ablation script's baseline (~470-479W) despite
both being labeled "matched harness" -- that discrepancy is NOT resolved here and
should not be papered over; this script only visualizes this run's own internal
consistency (how much do repeated trials of the SAME script vary).

Usage:
    python3 plot_baseline_stability.py
    python3 plot_baseline_stability.py --csv telemetry/telemetry_fp16_baseline.csv --out assets
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CATEGORY_COLORS = {
    "Poem": "#60a5fa",
    "Physics": "#f59e0b",
    "Code": "#10b981",
}
DEFAULT_COLOR = "#94a3b8"


def load_trials(csv_path: Path) -> dict:
    """Group rows by prompt category, inferred from the prefix of prompt_label
    (e.g. 'Poem_trial1' -> 'Poem'). Returns {category: [row_dict, ...]} in file order."""
    trials = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("prompt_label", "")
            category = label.split("_trial")[0] if "_trial" in label else label
            row["throughput_tok_sec"] = float(row["throughput_tok_sec"])
            row["joules_per_token"] = float(row["joules_per_token"])
            row["avg_power_watts"] = float(row["avg_power_watts"])
            trials[category].append(row)
    return trials


def plot_metric_by_trial(trials: dict, metric: str, ylabel: str, title: str,
                          filename: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for category, rows in trials.items():
        x = list(range(1, len(rows) + 1))
        y = [r[metric] for r in rows]
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        ax.plot(x, y, marker="o", label=category, color=color, linewidth=1.5)

    ax.set_xlabel("Trial number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def plot_boxplot(trials: dict, metric: str, ylabel: str, title: str,
                  filename: str, out_dir: Path):
    categories = list(trials.keys())
    data = [[r[metric] for r in trials[cat]] for cat in categories]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, tick_labels=categories, patch_artist=True)
    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(CATEGORY_COLORS.get(cat, DEFAULT_COLOR))
        patch.set_alpha(0.7)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "telemetry" / "telemetry_fp16_baseline.csv",
        help="Path to telemetry_fp16_baseline.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "assets",
        help="Directory to write PNG plots into",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    trials = load_trials(args.csv)
    if not trials:
        raise SystemExit(f"No trial rows found in {args.csv}")

    plot_metric_by_trial(
        trials, "throughput_tok_sec", "Tokens / second",
        "FP16 Baseline Throughput Across Trials (per prompt category)",
        "baseline_throughput_by_trial.png", args.out,
    )
    plot_metric_by_trial(
        trials, "joules_per_token", "Joules / token",
        "FP16 Baseline Energy/Token Across Trials (per prompt category)",
        "baseline_energy_by_trial.png", args.out,
    )
    plot_boxplot(
        trials, "joules_per_token", "Joules / token",
        "FP16 Baseline Energy/Token Spread by Prompt Category",
        "baseline_energy_boxplot.png", args.out,
    )
    plot_boxplot(
        trials, "avg_power_watts", "Mean GPU power (W)",
        "FP16 Baseline Power Spread by Prompt Category",
        "baseline_power_boxplot.png", args.out,
    )

    n_trials = {cat: len(rows) for cat, rows in trials.items()}
    print(f"Trial counts per category: {n_trials}")
    print(f"Wrote 4 plots to {args.out}/")


if __name__ == "__main__":
    main()


