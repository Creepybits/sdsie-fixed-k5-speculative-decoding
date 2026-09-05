"""
speculative_scout.py

Real scout(1B)->target(8B) speculative decoding with a lossless verify/rollback
loop, fixed draft window K=5. This is the validated, working mechanism --
see README.md for what "validated" means here and what it doesn't cover.

"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import bench_common

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
MASTER_CSV = os.path.join(TELEMETRY_DIR, "telemetry_scout_accept.csv")

# Length of each discarded warmup unit. Must match the real run length
# (max_target_tokens), not just be "short but nonzero" -- see the
# 2026-09-0X parity-fix note in this module's docstring. A GPU takes real
# time to ramp from an idle/low-load state to its sustained power draw once
# a burst of work starts; if the warmup burst is much shorter than the real
# timed burst, the warmup can report "power and temperature stopped moving"
# while still measuring a lower power level than the real run reaches, and
# the real run then does its own settling *inside the timed window*. This
# was caught empirically: a 15-token warmup unit converged at ~220W, but the
# first real 250-token run still rose from 49C to 57C mid-measurement and
# averaged ~311W. The warmup unit length is now set to match
# max_target_tokens at call time (see run_speculative_scout_benchmark),
# removing that gap at the cost of fewer, longer warmup iterations.


def _run_one_prompt(label, input_ids, target_model, scout_model, monitor, K, max_tokens, eos):
    """Timed speculative run for a single prompt. Returns the log entry dict."""
    e_start = monitor.read_energy_j()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    result = bench_common.speculative_generate(target_model, scout_model, input_ids, K, max_tokens, eos=eos)

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    e_end = monitor.read_energy_j()

    tokens_generated = len(result["generated_ids"])
    total_drafted = result["total_drafted"]
    total_accepted = result["total_accepted"]
    cycle_count = result["cycles"]

    stats = monitor.window_stats(t_start, t_end)
    latency = t_end - t_start

    energy_counter = (e_end - e_start) if (e_start is not None and e_end is not None) else None
    energy_sampled = stats["energy_j_sampled"]
    energy = energy_counter if energy_counter is not None else energy_sampled
    energy_source = "nvml_counter" if energy_counter is not None else "sampled_trapezoid"

    tok_per_sec = tokens_generated / latency if latency > 0 else 0.0
    joules_per_tok = (energy / tokens_generated) if (energy is not None and tokens_generated) else None
    accept_rate = (total_accepted / total_drafted) * 100.0 if total_drafted > 0 else 0.0

    j_str = f"{joules_per_tok:.4f} J/tok" if joules_per_tok is not None else "J/tok n/a"
    print(f"  [{label}] {tokens_generated} tok ({cycle_count} cycles)  "
          f"{tok_per_sec:.2f} tok/s  {j_str}  "
          f"accept={accept_rate:.1f}%  P={stats['mean_w']} W  T={stats['temp_end_c']} C")

    return {
        "timestamp": datetime.now().isoformat(),
        "category": "Speculative Scout (1B->8B)",
        "prompt_label": label,
        "tokens": tokens_generated,
        "cycles": cycle_count,
        "throughput_tok_sec": round(tok_per_sec, 2),
        "avg_power_watts": round(stats["mean_w"], 2) if stats["mean_w"] is not None else None,
        "min_power_watts": round(stats["min_w"], 2) if stats["min_w"] is not None else None,
        "max_power_watts": round(stats["max_w"], 2) if stats["max_w"] is not None else None,
        "total_energy_joules": round(energy, 4) if energy is not None else None,
        "energy_source": energy_source,
        "energy_j_counter": round(energy_counter, 4) if energy_counter is not None else None,
        "energy_j_sampled": round(energy_sampled, 4) if energy_sampled is not None else None,
        "joules_per_token": round(joules_per_tok, 6) if joules_per_tok is not None else None,
        # NOTE: field names corrected from an earlier internal version, which
        # mislabeled these as "high_gear_pct"/"low_gear_pct" (leftover from a
        # different, unrelated experiment's naming) -- this script has no
        # concept of "gears," only draft acceptance/rejection.
        "accept_rate_pct": round(accept_rate, 1),
        "reject_rate_pct": round(100.0 - accept_rate, 1),
        "temp_start_c": stats["temp_start_c"],
        "temp_end_c": stats["temp_end_c"],
        "temp_max_c": stats["temp_max_c"],
        "sm_clock_mean_mhz": round(stats["sm_clock_mean_mhz"], 1) if stats["sm_clock_mean_mhz"] else None,
        "mem_clock_mean_mhz": round(stats["mem_clock_mean_mhz"], 1) if stats["mem_clock_mean_mhz"] else None,
        "throttle_reasons": monitor.throttle_reasons(),
        "n_power_samples": stats["n_power_samples"],
        "effective_sample_hz": round(stats["effective_sample_hz"], 1),
        "prompt_tokens": int(input_ids.shape[-1]),
        "K": K,
        "max_tokens_config": max_tokens,
    }


def run_speculative_scout_benchmark(
    target_model_id="meta-llama/Llama-3.1-8B-Instruct",
    scout_model_id="meta-llama/Llama-3.2-1B-Instruct",
    prompts=None,
    K=bench_common.REFERENCE_K,
    max_target_tokens=bench_common.REFERENCE_MAX_TOKENS,
    lock_clocks=True,
    min_warmup_sec=bench_common.MIN_WARMUP_SEC,
    max_warmup_sec=bench_common.MAX_WARMUP_SEC,
):
    """Run the scout->target speculative loop, timed, once per prompt.

    prompts: list of (label, text) pairs. Defaults to the three reference
    prompts shared with benchmark_ablation.py / new_fp16_baseline.py
    (bench_common.REFERENCE_PROMPTS), so a bare call to this function is
    directly comparable to those scripts' per-category numbers.
    """
    if prompts is None:
        prompts = list(zip(bench_common.REFERENCE_PROMPT_LABELS, bench_common.REFERENCE_PROMPTS))

    print("=" * 95)
    print("SPECULATIVE SCOUT ENGINE (fixed K, lossless verify/rollback)")
    print(f"Target Model (Verifier) : {target_model_id}")
    print(f"Scout Model (Draft)     : {scout_model_id}")
    print(f"Speculative Lookahead   : K = {K} candidate tokens per cycle")
    print(f"Max tokens / prompt     : {max_target_tokens}")
    print(f"Prompts ({len(prompts)})           : {', '.join(label for label, _ in prompts)}")
    print("=" * 95)

    monitor = bench_common.NVMLPowerMonitor(device_index=0)
    dev = monitor.device_info()
    print(f"[*] Device: {dev.get('name')}  driver {dev.get('driver_version')}")
    print(f"[*] Power limit: {dev.get('power_limit_w')} W   "
          f"energy counter: {'yes' if dev['energy_counter_supported'] else 'NO (falling back to sampling)'}")
    if lock_clocks:
        monitor.lock_clocks()
    monitor.start()

    print("\n[1/3] Loading Shared Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = bench_common.eos_ids(tokenizer)

    print("[2/3] Loading Target Model (8B) into VRAM...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    target_model.eval()

    print("[3/3] Loading Scout Model (1B) into VRAM...")
    scout_model = AutoModelForCausalLM.from_pretrained(
        scout_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    scout_model.eval()

    encoded = [(label, bench_common.encode_prompt(tokenizer, text)) for label, text in prompts]
    for label, ids in encoded:
        print(f"[*] {label}: {ids.shape[-1]} prompt tokens (chat-templated)")

    try:
        # -- Closed-loop thermal warmup, cycling through all prompts --------
        def warmup_step(i):
            label, ids = encoded[i % len(encoded)]
            bench_common.speculative_generate(
                target_model, scout_model, ids.clone(), K,
                max_tokens=max_target_tokens, eos=eos,
            )
            return f"{label}/spec"

        warmup = bench_common.warm_to_steady_state(
            monitor, warmup_step, min_sec=min_warmup_sec, max_sec=max_warmup_sec,
        )

        # -- One timed run per prompt -----------------------------------------
        print(f"\nRunning {len(encoded)} timed prompt(s)...")
        results = {}
        for label, ids in encoded:
            entry = _run_one_prompt(label, ids, target_model, scout_model, monitor, K, max_target_tokens, eos)
            results[label] = entry
            bench_common.safe_append_csv(MASTER_CSV, entry)
    finally:
        monitor.stop()
        monitor.unlock_clocks()

    print("=" * 95)
    print("SPECULATIVE SCOUT TELEMETRY SUMMARY:")
    for label, entry in results.items():
        print(f" - {label:<8}: {entry['throughput_tok_sec']:.2f} tok/s  "
              f"accept={entry['accept_rate_pct']:.1f}%  "
              f"P={entry['avg_power_watts']} W")
    print(f" - Warmup: converged={warmup['converged']}  "
          f"elapsed={warmup['elapsed_sec']:.1f}s  iterations={warmup['iterations']}")
    print("=" * 95)

    spec_json = os.path.join(TELEMETRY_DIR, "speculative_scout_report.json")
    with open(spec_json, "w") as f:
        json.dump({
            "_meta": {
                "device": dev,
                "config": {
                    "target_model": target_model_id,
                    "scout_model": scout_model_id,
                    "K": K,
                    "max_target_tokens": max_target_tokens,
                    "clocks_locked": monitor._clocks_locked,
                    "prompts": [label for label, _ in prompts],
                },
                "warmup": warmup,
            },
            "results": results,
        }, f, indent=2, default=bench_common.json_safe)

    print(f"Per-run CSV rows saved to: {MASTER_CSV}")
    print(f"Full report saved to:      {spec_json}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Scout->target speculative decoding. By default runs all three "
                     "reference prompts (Poem/Physics/Code) for direct comparison "
                     "with benchmark_ablation.py"
    )
    ap.add_argument("--prompt-label", choices=bench_common.REFERENCE_PROMPT_LABELS,
                     help="Run only this one reference prompt instead of all three.")
    ap.add_argument("--prompt", default=None,
                     help="Run a single custom prompt instead of the reference set "
                          "(overrides --prompt-label).")
    ap.add_argument("--k", type=int, default=bench_common.REFERENCE_K)
    ap.add_argument("--tokens", type=int, default=bench_common.REFERENCE_MAX_TOKENS)
    ap.add_argument("--min-warmup", type=float, default=bench_common.MIN_WARMUP_SEC,
                     help="Minimum sustained-load seconds before timed runs start.")
    ap.add_argument("--max-warmup", type=float, default=bench_common.MAX_WARMUP_SEC,
                     help="Hard cap on warmup; proceeds with a warning if not converged.")
    ap.add_argument("--no-lock-clocks", action="store_true",
                     help="Skip NVML clock locking (it needs elevated privileges).")
    args = ap.parse_args()

    if args.prompt is not None:
        prompt_list = [("Custom", args.prompt)]
    elif args.prompt_label is not None:
        prompt_list = [(args.prompt_label, bench_common.REFERENCE_PROMPTS_BY_LABEL[args.prompt_label])]
    else:
        prompt_list = None  # defaults to all three reference prompts

    run_speculative_scout_benchmark(
        prompts=prompt_list,
        K=args.k,
        max_target_tokens=args.tokens,
        lock_clocks=not args.no_lock_clocks,
        min_warmup_sec=args.min_warmup,
        max_warmup_sec=args.max_warmup,
    )
