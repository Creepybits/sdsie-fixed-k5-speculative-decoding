#!/usr/bin/env python3
"""
plot_ablation_results.py

Reads telemetry/academic_validation_results.json (produced by benchmark_ablation.py)
and renders the core paper figures: throughput, energy/token, power, and accept rate,
FP16 baseline vs. fixed-K=5 speculative decoding, per prompt.

Usage:
    python3 plot_ablation_results.py
    python3 plot_ablation_results.py --telemetry telemetry/academic_validation_results.json --out assets
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE_KEY = "FP16 Baseline"
SPEC_KEY = "Speculative (1B->8B)"

# Display order + friendly labels for prompt keys found in the JSON.
PROMPT_LABELS = {
    "prompt_1": "Poem",
    "prompt_2": "Physics",
    "prompt_3": "Code",
}

BASELINE_COLOR = "#94a3b8"   # slate gray
SPEC_COLOR = "#10b981"       # emerald (matches README badge)
ACCENT_COLOR = "#a855f7"     # purple (matches README badge)


def load_data(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def prompt_order(ablation: dict) -> list:
    """Preserve a stable, sensible order even if prompt keys/labels differ from expectations."""
    keys = list(ablation.keys())
    known = [k for k in PROMPT_LABELS if k in keys]
    unknown = [k for k in keys if k not in PROMPT_LABELS]
    return known + unknown


def label_for(key: str) -> str:
    return PROMPT_LABELS.get(key, key)


def grouped_bar(ax, labels, baseline_vals, spec_vals, ylabel, title,
                 baseline_err=None, spec_err=None, value_fmt="{:.2f}"):
    x = np.arange(len(labels))
    width = 0.35

    b_bars = ax.bar(x - width / 2, baseline_vals, width, label="FP16 Baseline",
                     color=BASELINE_COLOR, yerr=baseline_err, capsize=4)
    s_bars = ax.bar(x + width / 2, spec_vals, width, label="Speculative (K=5)",
                     color=SPEC_COLOR, yerr=spec_err, capsize=4)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bars in (b_bars, s_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(value_fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)


def plot_throughput(ablation, order, out_dir):
    labels = [label_for(k) for k in order]
    baseline = [ablation[k][BASELINE_KEY]["tps_mean"] for k in order]
    baseline_err = [ablation[k][BASELINE_KEY]["tps_std"] for k in order]
    spec = [ablation[k][SPEC_KEY]["tps_mean"] for k in order]
    spec_err = [ablation[k][SPEC_KEY]["tps_std"] for k in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    grouped_bar(ax, labels, baseline, spec, "Tokens / second",
                "Throughput: FP16 Baseline vs. Speculative (K=5)",
                baseline_err=baseline_err, spec_err=spec_err, value_fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_baseline_vs_speculative.png", dpi=200)
    plt.close(fig)


def plot_energy(ablation, order, out_dir):
    labels = [label_for(k) for k in order]
    baseline = [ablation[k][BASELINE_KEY]["j_tok_mean"] for k in order]
    baseline_err = [ablation[k][BASELINE_KEY]["j_tok_std"] for k in order]
    spec = [ablation[k][SPEC_KEY]["j_tok_mean"] for k in order]
    spec_err = [ablation[k][SPEC_KEY]["j_tok_std"] for k in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    grouped_bar(ax, labels, baseline, spec, "Joules / token",
                "Energy per Token: FP16 Baseline vs. Speculative (K=5)",
                baseline_err=baseline_err, spec_err=spec_err, value_fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "energy_per_token_baseline_vs_speculative.png", dpi=200)
    plt.close(fig)


def plot_power(ablation, order, out_dir):
    labels = [label_for(k) for k in order]
    baseline = [ablation[k][BASELINE_KEY]["power_mean"] for k in order]
    spec = [ablation[k][SPEC_KEY]["power_mean"] for k in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    grouped_bar(ax, labels, baseline, spec, "Mean GPU power (W)",
                "Mean GPU Power: FP16 Baseline vs. Speculative (K=5)",
                value_fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(out_dir / "power_baseline_vs_speculative.png", dpi=200)
    plt.close(fig)


def plot_accept_rate(ablation, order, out_dir):
    labels = [label_for(k) for k in order]
    accept = [ablation[k][SPEC_KEY]["accept_rate_pct_mean"] for k in order]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, accept, color=ACCENT_COLOR)
    ax.set_ylabel("Draft accept rate (%)")
    ax.set_title("Speculative Draft Accept Rate by Prompt")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "accept_rate_by_prompt.png", dpi=200)
    plt.close(fig)


def plot_speedup_vs_accept(ablation, order, out_dir):
    """Scatter: does higher accept rate track higher speedup? (It should, per README claims.)"""
    accept = [ablation[k][SPEC_KEY]["accept_rate_pct_mean"] for k in order]
    speedup = [
        ablation[k][SPEC_KEY]["tps_mean"] / ablation[k][BASELINE_KEY]["tps_mean"]
        for k in order
    ]
    labels = [label_for(k) for k in order]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(accept, speedup, s=90, color=SPEC_COLOR, zorder=3)
    for lbl, x, y in zip(labels, accept, speedup):
        ax.annotate(lbl, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axhline(1.0, color=BASELINE_COLOR, linestyle="--", linewidth=1, label="No speedup (1.0x)")
    ax.set_xlabel("Draft accept rate (%)")
    ax.set_ylabel("Speedup vs. FP16 baseline (x)")
    ax.set_title("Speedup Scales with Accept Rate")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_vs_accept_rate.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path(__file__).parent / "telemetry" / "academic_validation_results.json",
        help="Path to academic_validation_results.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "assets",
        help="Directory to write PNG plots into",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    data = load_data(args.telemetry)
    ablation = data["ablation"]
    order = prompt_order(ablation)

    plot_throughput(ablation, order, args.out)
    plot_energy(ablation, order, args.out)
    plot_power(ablation, order, args.out)
    plot_accept_rate(ablation, order, args.out)
    plot_speedup_vs_accept(ablation, order, args.out)

    print(f"Wrote 5 plots to {args.out}/")


if __name__ == "__main__":
    main()
