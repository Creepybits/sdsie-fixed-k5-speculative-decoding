"""
plot_likeforlike.py

Draws the like-for-like comparison figures from a suite's summary.json
(written by likeforlike_analyze.py).

Usage:
  python plot_likeforlike.py telemetry/likeforlike/<suite folder>/summary.json [--out assets] [--formats svg,png]

Writes, into --out (default: ../assets relative to this script):
  likeforlike_speed_energy.<fmt>   speed and energy per token vs. no speculation,
                                   each method at its best configuration + K5,
                                   evaluation prompts only, per engine
  likeforlike_by_category.<fmt>    K5 vs. best EAGLE3 (as shipped), speedup by
                                   prompt category, all prompts, per engine

Written 2026-09-25. Needs matplotlib (present in the vLLM venv).
Engines are drawn in separate panels on purpose: numbers from different
engines are not comparable with each other.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
K5 = "draft_k5"
ENGINE_TITLE = {"sglang": "SGLang", "vllm": "vLLM"}
METHOD_LABEL = {"draft": "Draft model", "eagle3": "EAGLE-3", "eagle3_matched": "EAGLE-3 (async off)",
                "eagle": "EAGLE-1", "ngram": "N-gram"}
ORDER = ["eagle3", "eagle3_matched", "eagle", "draft", "ngram"]
K5_COLOR, OTHER_COLOR, EAGLE_COLOR = "#2563eb", "#9ca3af", "#f59e0b"


def rows_for(R):
    """K5 first, then each method's best configuration (skipping K5 duplicates)."""
    cfg = R["configs"]
    rows = []
    if K5 in cfg:
        rows.append(("K5 (draft, K=5)", K5))
    for m in ORDER:
        tag = R["best_by_method"].get(m)
        if tag and tag != K5:
            rows.append((f"{METHOD_LABEL.get(m, m)}\nbest: {tag}", tag))
    return rows


def color_for(tag, cfg):
    if tag == K5:
        return K5_COLOR
    return EAGLE_COLOR if cfg[tag]["method"].startswith("eagle3") else OTHER_COLOR


def plot_speed_energy(results, out_base, formats):
    engines = [e for e in ("sglang", "vllm") if e in results] + sorted(e for e in results if e not in ("sglang", "vllm") and "configs" in results[e])
    fig, axes = plt.subplots(2, len(engines), figsize=(6.2 * len(engines), 7.6), squeeze=False)
    for j, eng in enumerate(engines):
        R, cfg = results[eng], results[eng]["configs"]
        rows = rows_for(R)[::-1]  # top-to-bottom in reading order
        labels = [r[0] for r in rows]
        tags = [r[1] for r in rows]
        colors = [color_for(t, cfg) for t in tags]
        # speed
        ax = axes[0][j]
        vals = [(cfg[t]["speedup_geo_eval"] - 1) * 100 for t in tags]
        lo = [(cfg[t]["speedup_geo_eval"] - cfg[t]["speedup_ci_eval"][0]) * 100 for t in tags]
        hi = [(cfg[t]["speedup_ci_eval"][1] - cfg[t]["speedup_geo_eval"]) * 100 for t in tags]
        ax.barh(labels, vals, color=colors, xerr=[lo, hi], capsize=3, error_kw={"elinewidth": 1})
        for y, (v, h, l) in enumerate(zip(vals, hi, lo)):
            if v >= 0:
                ax.text(v + h + 3, y, f"{v:+.0f}%", va="center", ha="left", fontsize=9)
            else:
                ax.text(v - l - 3, y, f"{v:+.0f}%", va="center", ha="right", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ENGINE_TITLE.get(eng, eng)}: speed vs. no speculation", fontsize=11)
        ax.set_xlabel("Throughput change (%), 95% range over prompts")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlim(min(-20, min(v - l for v, l in zip(vals, lo)) - 18), max(v + h for v, h in zip(vals, hi)) + 22)
        # energy
        ax = axes[1][j]
        ev = [(cfg[t]["energy_ratio_geo_eval"] - 1) * 100 for t in tags]
        ax.barh(labels, ev, color=colors)
        for y, v in enumerate(ev):
            ax.text(v + 1.5, y, f"{v:+.0f}%", va="center", ha="left", fontsize=9, color="white",
                    fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{ENGINE_TITLE.get(eng, eng)}: energy per token vs. no speculation", fontsize=11)
        ax.set_xlabel("Energy per token change (%) -- lower is better")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlim(min(ev) - 8, 5)
    fig.suptitle("Like-for-like comparison, each method at its best configuration\n"
                 "(evaluation prompts only; batch size 1, 250 tokens, RTX 5090)", fontsize=11.5)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{out_base}.{fmt}", dpi=150)
    plt.close(fig)


def plot_by_category(results, out_base, formats):
    engines = [e for e in ("sglang", "vllm") if e in results] + sorted(e for e in results if e not in ("sglang", "vllm") and "configs" in results[e])
    fig, axes = plt.subplots(1, len(engines), figsize=(6.2 * len(engines), 4.2), squeeze=False)
    for j, eng in enumerate(engines):
        R = results[eng]
        e3 = R["best_by_method"].get("eagle3")
        cats = R["categories"]
        ax = axes[0][j]
        if not e3 or K5 not in R["by_category"] or e3 not in R["by_category"]:
            ax.set_axis_off()
            ax.set_title(f"{ENGINE_TITLE.get(eng, eng)}: no K5 / EAGLE-3 pair to compare", fontsize=10)
            continue
        x = range(len(cats))
        w = 0.4
        k5v = [(R["by_category"][K5][c] - 1) * 100 for c in cats]
        e3v = [(R["by_category"][e3][c] - 1) * 100 for c in cats]
        ax.bar([i - w / 2 for i in x], k5v, w, label="K5 (draft, K=5)", color=K5_COLOR)
        ax.bar([i + w / 2 for i in x], e3v, w, label=f"EAGLE-3 best ({e3})", color=EAGLE_COLOR)
        ax.set_xticks(list(x))
        ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Throughput change vs. no speculation (%)")
        ax.set_title(f"{ENGINE_TITLE.get(eng, eng)}: speedup by prompt category", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(0, max(max(k5v), max(e3v)) * 1.22)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{out_base}.{fmt}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "assets"))
    ap.add_argument("--formats", default="svg,png")
    a = ap.parse_args()
    plt.rcParams["svg.fonttype"] = "none"   # keep text as text: smaller, editable SVGs
    with open(a.summary) as f:
        results = json.load(f)["results"]
    os.makedirs(a.out, exist_ok=True)
    formats = [x.strip() for x in a.formats.split(",") if x.strip()]
    plot_speed_energy(results, os.path.join(a.out, "likeforlike_speed_energy"), formats)
    plot_by_category(results, os.path.join(a.out, "likeforlike_by_category"), formats)
    print(f"[*] Wrote figures to {a.out} ({', '.join(formats)})")


if __name__ == "__main__":
    main()
