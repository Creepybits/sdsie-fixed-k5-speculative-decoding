"""
rebuild_summary.py

Recovers the summary JSON for a run whose per-trial rows were written to the
telemetry CSV but whose json.dump crashed (the np.bool_ serialization bug).

run_trial() appends each row to the CSV as it completes, so no GPU work needs
to be repeated. This script walks the CSV backwards from the end, collecting
rows until it sees global_index == 1, which marks the start of the most recent
run, then recomputes the aggregates and drift diagnostics.

What cannot be recovered: the _meta.warmup trace and the device info block,
since those were only ever held in memory. Everything that mattered for the
result -- per-trial power, energy, throughput, temperature, clocks -- is in the
CSV and is reconstructed here.

Usage:
    python rebuild_summary.py
    python rebuild_summary.py --csv path/to/telemetry_fp16_baseline.csv
    python rebuild_summary.py --out path/to/summary.json
"""

import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np

DRIFT_WARN_FRACTION = 0.005

# Columns that should come back as numbers rather than strings.
INT_FIELDS = {"global_index", "tokens", "prompt_tokens", "temp_start_c",
              "temp_end_c", "temp_max_c", "sm_clock_min_mhz",
              "throttle_reasons", "n_power_samples"}
FLOAT_FIELDS = {"latency_sec", "throughput_tok_sec", "avg_power_watts",
                "median_power_watts", "min_power_watts", "max_power_watts",
                "total_energy_joules", "energy_j_counter", "energy_j_sampled",
                "joules_per_token", "sm_clock_mean_mhz", "mem_clock_mean_mhz",
                "effective_sample_hz"}
BOOL_FIELDS = {"hit_eos"}


def _json_safe(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def coerce(row):
    out = {}
    for k, v in row.items():
        if v is None or v == "" or v == "None":
            out[k] = None
        elif k in INT_FIELDS:
            try:
                out[k] = int(float(v))
            except ValueError:
                out[k] = None
        elif k in FLOAT_FIELDS:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
        elif k in BOOL_FIELDS:
            out[k] = v.strip().lower() in ("true", "1", "yes")
        else:
            out[k] = v
    return out


def load_last_run(csv_path):
    with open(csv_path, newline="") as f:
        rows = [coerce(r) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    # Walk backwards until global_index == 1, which starts the newest run.
    # Older rows from the pre-rewrite script have no global_index; stop there.
    collected = []
    for row in reversed(rows):
        gi = row.get("global_index")
        if gi is None:
            break
        collected.append(row)
        if gi == 1:
            break
    collected.reverse()

    if not collected or collected[0].get("global_index") != 1:
        raise SystemExit(
            "Could not find a complete run at the end of the CSV "
            "(no row with global_index == 1). Inspect the file manually."
        )
    return collected


def fit_drift(indices, values):
    pairs = [(x, y) for x, y in zip(indices, values) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    span = float(slope) * float(x.max() - x.min())
    mean = float(y.mean())
    return {
        "slope_per_trial": float(slope),
        "r2": float(r2),
        "total_change_over_run": float(span),
        "fraction_of_mean": float(span / mean) if mean else 0.0,
        "n": len(pairs),
    }


def main():
    default_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "telemetry", "telemetry_fp16_baseline.csv",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=default_csv)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_last_run(args.csv)
    print(f"[*] Recovered {len(rows)} trial rows from {args.csv}")
    print(f"    {rows[0]['timestamp']}  ->  {rows[-1]['timestamp']}")

    by_label = OrderedDict()
    for r in rows:
        label = (r.get("prompt_label") or "").split("_trial")[0] or "Unknown"
        by_label.setdefault(label, []).append(r)

    num_trials = max(len(v) for v in by_label.values())

    aggregated = {}
    for label, trials in by_label.items():
        tps = [t["throughput_tok_sec"] for t in trials if t["throughput_tok_sec"] is not None]
        jtok = [t["joules_per_token"] for t in trials if t["joules_per_token"] is not None]
        pwr = [t["avg_power_watts"] for t in trials if t["avg_power_watts"] is not None]
        aggregated[label] = {
            "n_trials": len(trials),
            "tps_mean": float(np.mean(tps)) if tps else None,
            "tps_std": float(np.std(tps, ddof=1)) if len(tps) > 1 else 0.0,
            "j_tok_mean": float(np.mean(jtok)) if jtok else None,
            "j_tok_std": float(np.std(jtok, ddof=1)) if len(jtok) > 1 else 0.0,
            "power_mean": float(np.mean(pwr)) if pwr else None,
            "power_std": float(np.std(pwr, ddof=1)) if len(pwr) > 1 else 0.0,
            "prompt_tokens": trials[0].get("prompt_tokens"),
            "raw_trials": trials,
        }

    idx = [r["global_index"] for r in rows]
    drift = {
        "power_w": fit_drift(idx, [r["avg_power_watts"] for r in rows]),
        "joules_per_token": fit_drift(idx, [r["joules_per_token"] for r in rows]),
        "throughput_tok_sec": fit_drift(idx, [r["throughput_tok_sec"] for r in rows]),
        "temp_c": fit_drift(idx, [r["temp_end_c"] for r in rows]),
    }
    pdrift = drift["power_w"]
    residual_ok = bool(pdrift is not None
                       and abs(pdrift["fraction_of_mean"]) <= DRIFT_WARN_FRACTION)
    drift["residual_drift_acceptable"] = residual_ok
    drift["threshold_fraction"] = DRIFT_WARN_FRACTION

    energy_sources = sorted({r.get("energy_source") for r in rows if r.get("energy_source")})

    out = {
        "_meta": {
            "reconstructed_from_csv": True,
            "note": "Warmup trace and device info were lost with the crashed run "
                    "and are not recoverable. All per-trial telemetry is intact.",
            "source_csv": os.path.abspath(args.csv),
            "run_start": rows[0]["timestamp"],
            "run_end": rows[-1]["timestamp"],
            "num_trials": num_trials,
            "energy_sources_seen": energy_sources,
            "drift_diagnostics": drift,
        },
        **aggregated,
    }

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)),
        f"fp16_baseline_n{num_trials}_summary.json",
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_json_safe)

    print("\n" + "=" * 95)
    print(f"{'Prompt':<10} | {'prompt_tok':<10} | {'tok/s':<20} | {'J/tok':<20} | {'W':<8}")
    print("-" * 95)
    for label, a in aggregated.items():
        tps_s = f"{a['tps_mean']:.2f} +/- {a['tps_std']:.2f}" if a["tps_mean"] is not None else "n/a"
        j_s = f"{a['j_tok_mean']:.3f} +/- {a['j_tok_std']:.3f}" if a["j_tok_mean"] is not None else "n/a"
        w_s = f"{a['power_mean']:.1f}" if a["power_mean"] is not None else "n/a"
        print(f"{label:<10} | {str(a['prompt_tokens']):<10} | {tps_s:<20} | {j_s:<20} | {w_s:<8}")
    print("=" * 95)

    print("\nDRIFT CHECK (chronological, all categories pooled)")
    if pdrift:
        print(f"  power slope       : {pdrift['slope_per_trial']:+.4f} W/trial "
              f"(R^2={pdrift['r2']:.3f}, total {pdrift['total_change_over_run']:+.2f} W = "
              f"{pdrift['fraction_of_mean'] * 100:+.2f}%)")
    if drift["joules_per_token"]:
        jd = drift["joules_per_token"]
        print(f"  J/tok slope       : {jd['slope_per_trial']:+.5f} /trial "
              f"({jd['fraction_of_mean'] * 100:+.2f}% over run)")
    if drift["temp_c"]:
        print(f"  temperature slope : {drift['temp_c']['slope_per_trial']:+.3f} C/trial")

    if residual_ok:
        print(f"\n[ok] Residual drift within {DRIFT_WARN_FRACTION * 100:.1f}% of mean power. "
              "Means are usable.")
    else:
        print(f"\n[!] Residual drift exceeds {DRIFT_WARN_FRACTION * 100:.1f}% of mean power. "
              "Raise --min-warmup and re-run.")

    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()