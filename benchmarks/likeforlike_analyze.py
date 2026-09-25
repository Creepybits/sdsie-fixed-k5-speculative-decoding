"""
likeforlike_analyze.py

Reads one suite folder written by likeforlike_suite.py and produces:
  report.md     -- human-readable tables
  summary.json  -- every number in the report, machine-readable

Usage:  python likeforlike_analyze.py telemetry/likeforlike/<suite folder>

Written 2026-09-24. Needs only numpy (runs in any venv).

HOW THE NUMBERS ARE COMPUTED (plain version)
  * Speedup is always measured against the baseline runs of the SAME engine
    in the SAME repetition block (the opening and closing baseline, averaged),
    prompt by prompt. Then the per-prompt speedups are combined with a
    geometric mean -- the right average for ratios (a 2x and a 0.5x average
    to 1x, not 1.25x).
  * The 95% range after each speedup is a bootstrap interval over prompts:
    it answers "how much would this number move if we had picked a different
    set of prompts like these?" If two methods' ranges overlap a lot, the
    difference between them is not established.
  * "Best configuration" for each method is chosen on one half of the prompts
    (the tuning half) and reported on the other half (the evaluation half).
    Choosing and grading on the same prompts would flatter every method that
    has more configurations to choose from.
  * Fidelity is how much of each output matches the engine's own first
    baseline run before the first differing token. The baseline-vs-baseline
    row is the control: it shows how much two runs of the SAME
    non-speculative engine already disagree through ordinary floating-point
    rounding. A method is only suspicious if it diverges clearly MORE than
    that control.

SANITY FLAGS (a flagged configuration is shown but never picked as "best")
  not_speculating      accept length ~1.0 -- drafts are almost never accepted
  unexplained_speedup  speedup larger than the accept length, which is
                       physically impossible: each verification step costs at
                       least one ordinary decoding step
  accept_len_unknown   no accept length available, so the check above
                       couldn't run (shown, but treat with caution)
  length_mismatch      output lengths differ from baseline on >20% of prompts
"""

import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

BOOT = 2000
HARD_FLAGS = {"not_speculating", "unexplained_speedup", "length_mismatch"}
K5_TAG = "draft_k5"


def load_runs(suite_dir):
    ok, bad = [], []
    for p in sorted(glob.glob(os.path.join(suite_dir, "runs", "*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception as exc:
            bad.append({"file": os.path.basename(p), "error": f"unreadable: {exc}"})
            continue
        r["_file"] = os.path.basename(p)
        if r.get("status") == "ok" and r.get("batches"):
            ok.append(r)
        else:
            bad.append({"file": r["_file"], "engine": r.get("engine"), "tag": r.get("tag"),
                        "rep": r.get("rep"), "stage": r.get("stage"), "error": r.get("error")})
    return ok, bad


def unit_key(batch):
    return "|".join(batch["labels"])


def seq(batch, i):
    ids = batch["token_ids"][i]
    if ids is not None:
        return list(ids)
    texts = batch.get("texts")
    return texts[i].split() if texts and texts[i] is not None else None


def match_pct(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    d = next((i for i in range(n) if a[i] != b[i]), n)
    return 100.0 * d / n


def geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def boot_geo_ci(xs, seed=0):
    xs = np.array([x for x in xs if x and x > 0], dtype=float)
    if len(xs) < 3:
        return None
    rng = np.random.default_rng(seed)
    logs = np.log(xs)
    means = logs[rng.integers(0, len(xs), size=(BOOT, len(xs)))].mean(axis=1)
    return float(math.exp(np.percentile(means, 2.5))), float(math.exp(np.percentile(means, 97.5)))


def fmt_x(v, ci=None):
    if v is None:
        return "n/a"
    s = f"{(v - 1) * 100:+.1f}%"
    if ci:
        s += f" [{(ci[0] - 1) * 100:+.1f}, {(ci[1] - 1) * 100:+.1f}]"
    return s


def split_units(units, category_of):
    """Alternate within each category so both halves cover every category."""
    if len(units) < 8:
        return list(units), list(units), False
    by_cat = defaultdict(list)
    for u in sorted(units):
        by_cat[category_of.get(u, "?")].append(u)
    tune, ev, n = [], [], 0
    for cat in sorted(by_cat):
        for u in by_cat[cat]:
            (tune if n % 2 == 0 else ev).append(u)
            n += 1
    return tune, ev, True


def analyze_engine(eng, runs):
    base_runs = [r for r in runs if r["method"] == "baseline"]
    if not base_runs:
        return {"error": "no successful baseline run for this engine"}
    base_runs.sort(key=lambda r: (r["rep"], r.get("started_at", "")))
    reference = base_runs[0]

    # reference outputs + categories
    ref_seq, ref_len, category_of = {}, {}, {}
    for b in reference["batches"]:
        for i, lab in enumerate(b["labels"]):
            ref_seq[lab] = seq(b, i)
            ref_len[lab] = b["n_tokens"][i]
    for r in runs:
        for b in r["batches"]:
            u = unit_key(b)
            cats = set(b["categories"])
            category_of[u] = cats.pop() if len(cats) == 1 else "mixed"

    # baseline per rep per unit (mean of open/close), and drift between them
    base_tps = defaultdict(lambda: defaultdict(list))
    base_jpt = defaultdict(lambda: defaultdict(list))
    rep_runs = defaultdict(list)
    for r in base_runs:
        rep_runs[r["rep"]].append(r)
        for b in r["batches"]:
            base_tps[r["rep"]][unit_key(b)].append(b["tokens_per_sec"])
            base_jpt[r["rep"]][unit_key(b)].append(b["joules_per_token"])
    drift = {}
    for rep, rr in rep_runs.items():
        if len(rr) >= 2:
            a = {unit_key(b): b["tokens_per_sec"] for b in rr[0]["batches"]}
            z = {unit_key(b): b["tokens_per_sec"] for b in rr[-1]["batches"]}
            common = [u for u in a if u in z]
            g = geo([z[u] / a[u] for u in common])
            drift[rep] = g
    max_drift = max((abs(g - 1) for g in drift.values() if g), default=None)

    # baseline-vs-baseline fidelity control
    bb = []
    for r in base_runs[1:]:
        for b in r["batches"]:
            for i, lab in enumerate(b["labels"]):
                m = match_pct(seq(b, i), ref_seq.get(lab))
                if m is not None:
                    bb.append(m)

    # per configuration
    configs = defaultdict(list)
    for r in runs:
        if r["method"] != "baseline":
            configs[r["tag"]].append(r)

    per_cfg = {}
    for tag, rr in configs.items():
        sp_units = defaultdict(list)
        en_units = defaultdict(list)
        taus, tau_sources, fid, len_mismatch, len_total = [], set(), [], 0, 0
        for r in rr:
            for b in r["batches"]:
                u = unit_key(b)
                bt = base_tps[r["rep"]].get(u)
                bj = base_jpt[r["rep"]].get(u)
                if bt:
                    sp_units[u].append(b["tokens_per_sec"] / float(np.mean(bt)))
                if bj and all(x is not None for x in bj) and b["joules_per_token"]:
                    en_units[u].append(b["joules_per_token"] / float(np.mean(bj)))
                if b.get("accept_length"):
                    taus.append(b["accept_length"])
                    tau_sources.add(b.get("accept_length_source"))
                for i, lab in enumerate(b["labels"]):
                    m = match_pct(seq(b, i), ref_seq.get(lab))
                    if m is not None:
                        fid.append(m)
                    if lab in ref_len:
                        len_total += 1
                        if b["n_tokens"][i] != ref_len[lab]:
                            len_mismatch += 1
        # combine reps per unit with a geometric mean
        sp = {u: geo(v) for u, v in sp_units.items()}
        en = {u: geo(v) for u, v in en_units.items()}
        tau = float(np.mean(taus)) if taus else None
        g_all = geo(list(sp.values()))
        flags = []
        if tau is None:
            flags.append("accept_len_unknown")
        else:
            if tau < 1.05:
                flags.append("not_speculating")
            if g_all and g_all > tau * 1.15:
                flags.append("unexplained_speedup")
        if len_total and len_mismatch / len_total > 0.2:
            flags.append("length_mismatch")
        r0 = rr[0]
        # Engineering-matched runs compete as their own "method", so they never
        # replace the as-shipped version in "best" and get their own head-to-head row.
        method_group = r0["method"] + ("_matched" if r0["args"].get("no_async_scheduling") else "")
        per_cfg[tag] = {
            "method": method_group,
            "args": {k: r0["args"].get(k) for k in ("k", "topk", "num_draft_tokens", "no_async_scheduling")},
            "n_runs": len(rr), "reps": sorted({r["rep"] for r in rr}),
            "speedup_by_unit": sp, "energy_ratio_by_unit": en,
            "speedup_geo_all": g_all, "speedup_ci_all": boot_geo_ci(list(sp.values())),
            "energy_ratio_geo_all": geo(list(en.values())),
            "accept_length_mean": tau, "accept_length_sources": sorted(s for s in tau_sources if s),
            "fidelity_match_pct_mean": float(np.mean(fid)) if fid else None,
            "fidelity_exact_pct": 100.0 * sum(1 for x in fid if x == 100.0) / len(fid) if fid else None,
            "flags": flags,
        }

    units = sorted({u for c in per_cfg.values() for u in c["speedup_by_unit"]})
    tune, ev, did_split = split_units(units, category_of)

    def geo_on(cfg, which, us):
        return geo([cfg[which][u] for u in us if cfg[which].get(u)])

    best = {}
    for tag, c in per_cfg.items():
        c["speedup_geo_tune"] = geo_on(c, "speedup_by_unit", tune)
        c["speedup_geo_eval"] = geo_on(c, "speedup_by_unit", ev)
        c["speedup_ci_eval"] = boot_geo_ci([c["speedup_by_unit"][u] for u in ev if c["speedup_by_unit"].get(u)])
        c["energy_ratio_geo_eval"] = geo_on(c, "energy_ratio_by_unit", ev)
        if HARD_FLAGS & set(c["flags"]) or c["speedup_geo_tune"] is None:
            continue
        m = c["method"]
        if m not in best or c["speedup_geo_tune"] > per_cfg[best[m]]["speedup_geo_tune"]:
            best[m] = tag

    # head to head: K5 vs every other method's best, on the evaluation half
    h2h = {}
    if K5_TAG in per_cfg:
        k5 = per_cfg[K5_TAG]
        for m, tag in best.items():
            if tag == K5_TAG:
                continue
            o = per_cfg[tag]
            ratios = [k5["speedup_by_unit"][u] / o["speedup_by_unit"][u] for u in ev
                      if k5["speedup_by_unit"].get(u) and o["speedup_by_unit"].get(u)]
            eratios = [k5["energy_ratio_by_unit"][u] / o["energy_ratio_by_unit"][u] for u in ev
                       if k5["energy_ratio_by_unit"].get(u) and o["energy_ratio_by_unit"].get(u)]
            h2h[m] = {"opponent_tag": tag, "n_prompts": len(ratios),
                      "k5_speed_vs_opponent_geo": geo(ratios), "ci": boot_geo_ci(ratios),
                      "k5_wins": sum(1 for x in ratios if x > 1.0),
                      "k5_energy_vs_opponent_geo": geo(eratios)}

    # per-category view (all prompts), best config of each method
    cats = sorted({category_of.get(u, "?") for u in units})
    by_cat = {}
    for m, tag in best.items():
        c = per_cfg[tag]
        by_cat[tag] = {cat: geo([c["speedup_by_unit"][u] for u in units
                                 if category_of.get(u) == cat and c["speedup_by_unit"].get(u)])
                       for cat in cats}
    if K5_TAG in per_cfg and K5_TAG not in by_cat:
        c = per_cfg[K5_TAG]
        by_cat[K5_TAG] = {cat: geo([c["speedup_by_unit"][u] for u in units
                                    if category_of.get(u) == cat and c["speedup_by_unit"].get(u)])
                          for cat in cats}

    versions = sorted({r.get("engine_version", "?") for r in runs})
    return {
        "engine_versions": versions,
        "n_units": len(units), "tune_units": tune, "eval_units": ev, "split_used": did_split,
        "baseline_runs": len(base_runs), "baseline_block_drift_geo": drift,
        "baseline_max_block_drift": max_drift,
        "baseline_vs_baseline_fidelity_mean": float(np.mean(bb)) if bb else None,
        "baseline_vs_baseline_exact_pct": 100.0 * sum(1 for x in bb if x == 100.0) / len(bb) if bb else None,
        "configs": per_cfg, "best_by_method": best, "head_to_head": h2h,
        "by_category": by_cat, "categories": cats,
    }


def render(settings, results, bad):
    L = []
    L.append("# Like-for-like speculative decoding comparison\n")
    L.append(f"Preset `{settings.get('preset')}`, prompt set `{settings.get('prompts_file') or settings.get('prompt_set')}`, "
             f"{settings.get('reps')} repetition(s), {settings.get('tokens')} tokens, concurrency "
             f"{settings.get('concurrency')}, prefix cache {settings.get('prefix_cache')}, order seed {settings.get('seed')}.\n")
    L.append("Numbers from different engines are **not** comparable with each other -- compare within each engine only.\n")
    for eng, R in results.items():
        L.append(f"\n## {eng}  (version {', '.join(R.get('engine_versions', ['?']))})\n")
        if "error" in R:
            L.append(f"**{R['error']}**\n")
            continue
        d = R["baseline_max_block_drift"]
        L.append(f"- Prompts: {R['n_units']}. Best configuration chosen on {len(R['tune_units'])}, "
                 f"reported on {len(R['eval_units'])}"
                 + ("" if R["split_used"] else " -- **same prompts (too few to split): 'best' is optimistic**") + ".")
        L.append(f"- Baseline drift within a block (closing vs opening baseline): "
                 f"{'n/a' if d is None else f'{d * 100:.1f}%'}"
                 + ("  **-- above 3%, treat small differences with suspicion**" if d and d > 0.03 else "") + ".")
        bbm = R["baseline_vs_baseline_fidelity_mean"]
        bbe = R["baseline_vs_baseline_exact_pct"]
        if bbm is None:
            ctrl = "n/a (needs 2+ baseline runs)"
        else:
            ctrl = f"{bbm:.1f}% of tokens match on average, {bbe:.0f}% of outputs identical"
        L.append(f"- Fidelity control (baseline run vs baseline run): {ctrl}.\n")

        L.append("### Every configuration (all prompts)\n")
        L.append("| Config | Speedup [95% range] | Energy/token | Accept len | Fidelity (match / identical) | Runs | Flags |")
        L.append("|---|---|---|---|---|---|---|")
        for tag, c in sorted(R["configs"].items(), key=lambda kv: -(kv[1]["speedup_geo_all"] or 0)):
            en = c["energy_ratio_geo_all"]
            fm, fe = c["fidelity_match_pct_mean"], c["fidelity_exact_pct"]
            tau = c["accept_length_mean"]
            src = "" if not c["accept_length_sources"] else (" (coarse)" if "server_cumulative_coarse" in c["accept_length_sources"] else "")
            mark = " **K5**" if tag == K5_TAG else ""
            L.append(f"| `{tag}`{mark} | {fmt_x(c['speedup_geo_all'], c['speedup_ci_all'])} | "
                     f"{'n/a' if en is None else f'{(en - 1) * 100:+.1f}%'} | "
                     f"{'n/a' if tau is None else f'{tau:.2f}'}{src} | "
                     f"{'n/a' if fm is None else f'{fm:.0f}% / {fe:.0f}%'} | {c['n_runs']} | "
                     f"{', '.join(c['flags']) or '-'} |")

        L.append("\n### Best configuration of each method (evaluation prompts only)\n")
        L.append("| Method | Best config | Speedup [95% range] | Energy/token |")
        L.append("|---|---|---|---|")
        rows = dict(R["best_by_method"])
        for m, tag in sorted(rows.items(), key=lambda kv: -(R["configs"][kv[1]]["speedup_geo_eval"] or 0)):
            c = R["configs"][tag]
            en = c["energy_ratio_geo_eval"]
            L.append(f"| {m} | `{tag}` | {fmt_x(c['speedup_geo_eval'], c['speedup_ci_eval'])} | "
                     f"{'n/a' if en is None else f'{(en - 1) * 100:+.1f}%'} |")
        if K5_TAG in R["configs"] and rows.get("draft") != K5_TAG:
            c = R["configs"][K5_TAG]
            en = c["energy_ratio_geo_eval"]
            en_s = "n/a" if en is None else f"{(en - 1) * 100:+.1f}%"
            L.append(f"| draft (fixed K=5, i.e. K5) | `{K5_TAG}` | "
                     f"{fmt_x(c['speedup_geo_eval'], c['speedup_ci_eval'])} | {en_s} |")

        if R["head_to_head"]:
            L.append("\n### Head to head: K5 vs each method's best (evaluation prompts)\n")
            L.append("| Opponent | K5 speed relative to it [95% range] | K5 faster on | K5 energy/token relative to it |")
            L.append("|---|---|---|---|")
            for m, h in R["head_to_head"].items():
                e = h["k5_energy_vs_opponent_geo"]
                L.append(f"| {m} (`{h['opponent_tag']}`) | {fmt_x(h['k5_speed_vs_opponent_geo'], h['ci'])} | "
                         f"{h['k5_wins']}/{h['n_prompts']} prompts | "
                         f"{'n/a' if e is None else f'{(e - 1) * 100:+.1f}%'} |")
            L.append("\nPositive speed = K5 faster; negative energy = K5 uses less energy per token. "
                     "If the 95% range crosses 0, the difference is not established by this data.")
            if any(m.endswith("_matched") for m in R["head_to_head"]):
                L.append("\n`_matched` rows: the opponent ran with vLLM's async scheduling turned off, as "
                         "vLLM does on its own for K5's draft_model method. This matches async scheduling "
                         "only -- vLLM also runs draft_model on its older V1 model runner, which is NOT "
                         "matched. The unmarked row is the method as vLLM ships it.")

        if R["by_category"]:
            L.append("\n### Speedup by prompt category (all prompts, best config per method)\n")
            tags = list(R["by_category"])
            L.append("| Category | " + " | ".join(f"`{t}`" for t in tags) + " |")
            L.append("|---|" + "---|" * len(tags))
            for cat in R["categories"]:
                L.append(f"| {cat} | " + " | ".join(fmt_x(R["by_category"][t].get(cat)) for t in tags) + " |")

    if bad:
        L.append("\n## Runs that failed or were excluded\n")
        for b in bad:
            L.append(f"- `{b.get('file')}`: {b.get('stage') or ''} {b.get('error')}")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    suite_dir = os.path.abspath(sys.argv[1])
    try:
        with open(os.path.join(suite_dir, "plan.json")) as f:
            settings = json.load(f)["settings"]
    except Exception:
        settings = {}
    ok, bad = load_runs(suite_dir)
    by_engine = defaultdict(list)
    for r in ok:
        by_engine[r["engine"]].append(r)
    results = {eng: analyze_engine(eng, rr) for eng, rr in sorted(by_engine.items())}
    report = render(settings, results, bad)
    with open(os.path.join(suite_dir, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(suite_dir, "summary.json"), "w") as f:
        json.dump({"settings": settings, "results": results, "failed_runs": bad}, f, indent=1, default=str)
    print(report)
    print(f"[*] Wrote {os.path.join(suite_dir, 'report.md')} and summary.json")


if __name__ == "__main__":
    main()
