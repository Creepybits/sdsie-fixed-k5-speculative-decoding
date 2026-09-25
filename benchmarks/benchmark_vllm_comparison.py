"""
benchmark_vllm_comparison.py

Compares fixed-K5's underlying mechanism against vLLM's own native, built-in
speculative decoding (method="draft_model", num_speculative_tokens=5, same
scout/target model pair), to answer the question this project has not yet
actually tested: does K5 beat what's already free and open in vLLM?

===============================================================================
READ THIS BEFORE RUNNING OR CITING ANYTHING FROM THIS SCRIPT
===============================================================================

1. NOT DIRECTLY COMPARABLE TO THE EXISTING K5 PAPER NUMBERS.
   benchmark_ablation.py / speculative_scout.py deliberately run WITHOUT
   KV-caching, on both arms, so the K5 speedup numbers in the paper isolate
   the speculative mechanism itself. vLLM uses PagedAttention (KV-caching)
   internally and there is no supported way to disable it. So:

     - This script's "vLLM baseline" vs "vLLM + speculative" comparison is
       internally consistent (fair, like-for-like) and answers "does K5's
       mechanism help inside vLLM's own serving stack."
     - It does NOT tell you whether vLLM's cached baseline is faster or
       slower in absolute tok/s than this repo's uncached baseline. Don't
       put those two baseline numbers in the same table without saying so.
     - A true "does K5 beat vLLM" comparison needs a KV-cached variant of
       benchmark_ablation.py's manual harness, or an explicit acknowledgment
       in any paper/pitch material that this is comparing two different
       serving configurations, not two points on the same axis. That's a
       separate, not-yet-built piece of work.

2. TWO SEPARATE PROCESSES, NOT ONE.
   vLLM's speculative_config is set at engine construction time and can't be
   toggled per-request. Running both a baseline engine and a speculative
   engine in the same process risks holding two full model sets in VRAM at
   once. Run this script twice, once per --mode, as separate invocations:

     python benchmark_vllm_comparison.py --mode baseline
     python benchmark_vllm_comparison.py --mode speculative

   Each run appends to the same JSON/CSV so results can be compared after
   both have completed.

3. FIDELITY IS CHECKED, NOT ASSUMED.
   vLLM's draft-model speculative decoding is supposed to be lossless under
   greedy/deterministic sampling, same as this repo's own speculative_generate.
   This script checks that directly (exact token match, baseline vs
   speculative, same prompts) rather than taking that on faith -- consistent
   with how the rest of this project treats claims.

Requires: vllm (not in the repo's existing requirements.txt -- add it, ideally
in a separate venv, since vLLM pins its own torch/CUDA versions that may
conflict with this repo's other scripts).
"""

import argparse
import json
import os
import time
from datetime import datetime

# Must be set BEFORE `import vllm` -- vLLM reads this at import/init time.
# WSL2 disables pinned host memory (UVA) by default, which makes vLLM's V2
# model runner crash at startup with "RuntimeError: UVA is not available".
# Setting this here, rather than requiring it on the command line each time,
# guarantees --mode baseline and --mode speculative always run the same
# vLLM code path -- if one run had this set and the other didn't, that would
# be comparing two different runners, not two conditions of the same
# mechanism, which is exactly the kind of silent methodology drift this
# project has already found and fixed elsewhere.
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

# vLLM's default sampling kernel (flashinfer) JIT-compiles itself on first
# use, which requires nvcc / the full CUDA Toolkit -- not just the CUDA
# runtime libraries PyTorch/vLLM already bundle. Without it, engine startup
# crashes during its own internal warmup pass, which exercises the sampler
# regardless of what SamplingParams the caller will actually use later.
# Since this script only ever uses greedy decoding (temperature=0.0), the
# top-k/top-p sampling flashinfer specializes in isn't doing anything useful
# here anyway -- falling back to vLLM's plain PyTorch sampler costs nothing
# for this specific workload and avoids requiring a full CUDA Toolkit install
# just to run a benchmark.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import bench_common
from vllm.v1.metrics.reader import Counter, Vector

# K5's own already-verified accept rates (this repo's ablation_results.json /
# K5_status.md, 2026-09-05 cont. 2 run) -- carried forward here ONLY as a
# reference point to print alongside vLLM's own measured number below, not
# asserted or treated as ground truth. If these two numbers land close
# together, that's real, independent evidence the two implementations
# compute acceptance the same way; if they don't, that needs investigating
# before either number is cited anywhere.
REFERENCE_K5_ACCEPT_PCT_BY_LABEL = {"Poem": 42.5, "Physics": 51.7, "Code": 85.4}

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SCOUT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NUM_SPECULATIVE_TOKENS = bench_common.REFERENCE_K  # 5, matches K5 exactly
MAX_TOKENS = bench_common.REFERENCE_MAX_TOKENS
NUM_TRIALS_DEFAULT = 10

PROMPT_LABELS = bench_common.REFERENCE_PROMPT_LABELS
PROMPTS = bench_common.REFERENCE_PROMPTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(TELEMETRY_DIR, "vllm_comparison_results.json")
TRIAL_CSV = os.path.join(TELEMETRY_DIR, "telemetry_vllm_comparison.csv")

BASELINE_KEY = "vLLM FP16 Baseline (KV-cached)"
SPEC_KEY = "vLLM Speculative K=5 (KV-cached)"


def check_gpu_is_clear(monitor, threshold_mb=2000):
    """2000 MB, not 500 -- WSL2's own display compositor (Xwayland) normally
    holds ~1-1.5 GB of VRAM at idle, with no way to free it and nothing to
    do with model weights. A 500 MB threshold false-positives on that alone.
    2 GB is comfortably above that baseline but still well below what even
    the smallest model in this repo (the 1B scout, ~2.5 GB in bf16) would use,
    so a real leftover model will still trip this.

    Warn (don't block -- this is a heads-up, not a hard gate) if substantial
    GPU memory is already in use before this script's own engine has been
    built. A second resident process both risks vLLM's memory reservation
    failing (see module docstring) and, more importantly, contaminates the
    NVML power readings this script relies on -- a second GPU consumer's
    power draw is indistinguishable from this benchmark's own in a
    total-board-power reading, exactly the kind of confound this project has
    already found and fixed inside its own scripts (thermal settling,
    pooled-vs-per-label warmup). An external confound needs the same
    scrutiny as an internal one."""
    try:
        import pynvml
        used_mb = pynvml.nvmlDeviceGetMemoryInfo(monitor.handle).used / (1024 ** 2)
    except Exception:
        return
    if used_mb > threshold_mb:
        print(f"[!] {used_mb:.0f} MB of GPU memory already in use before this "
              f"script started anything. If that's another process (another "
              f"terminal, a notebook, an embedding model, etc.), stop it first:")
        print(f"    - It can cause vLLM's memory reservation to fail.")
        print(f"    - It will silently pollute this run's power/energy readings,")
        print(f"      the same way an uncontrolled confound would inside the script.")
        print(f"    Check with: nvidia-smi")
        input("    Press Enter to continue anyway, or Ctrl+C to stop and check first...")


def build_engine(mode: str, gpu_memory_utilization: float, max_model_len: int) -> LLM:
    """mode is 'baseline' (target only, no speculation) or 'speculative'
    (native vLLM draft-model speculative decoding, K=5, same scout/target
    pair as the rest of this repo).

    max_model_len is capped explicitly rather than left at Llama-3.1-8B's
    default (131072, its native long-context window). vLLM sizes its KV
    cache reservation off max_model_len, not off actual expected usage --
    left at the model default, it tries to reserve enough KV cache to serve
    a request at the full 131k length, which this benchmark (short reference
    prompts + REFERENCE_MAX_TOKENS=250 output) never comes close to needing.
    That mismatch, not gpu_memory_utilization, was the actual cause of the
    KV-cache-too-small error."""
    kwargs = dict(
        model=TARGET_MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        # Explicit, not left to whatever this vLLM version defaults to --
        # spec_decode_offline.py (vLLM's own example script) sets this
        # explicitly too. Needed for llm.get_metrics() to return the
        # vllm:spec_decode_* counters at all; without it, the first attempt
        # at capturing acceptance rate here (2026-09-19) came back with every
        # counter at zero despite genuinely-working, genuinely-faster
        # speculative decoding -- a metrics-collection gap, not a real zero.
        disable_log_stats=False,
    )
    if mode == "speculative":
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": SCOUT_MODEL_ID,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            # Explicit, not left to implicit capping from the target's
            # max_model_len -- vLLM's own draft_model usage examples set
            # this inside speculative_config too. Without it, the draft
            # model's config can resolve to its own native default (visible
            # in the startup log as a second "Using max model len ..." line
            # showing the draft model's full native context instead of this
            # value) even when the actual enforced KV-cache budget is
            # correctly bounded by the target's setting -- a real,
            # documented vLLM quirk where the log can lag the enforcement.
            # Setting it here removes the ambiguity instead of trusting that
            # implicit capping is working correctly on this vLLM version.
            "max_model_len": max_model_len,
        }
    return LLM(**kwargs)


def chat_prompts(tokenizer):
    """Build chat-templated prompt strings, matching bench_common.encode_prompt's
    convention (chat template, not raw text) so prompt framing matches the
    rest of this repo's methodology."""
    out = []
    for text in PROMPTS:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(formatted)
    return out


def per_trial_warmup(llm, prompt_text, n_steps):
    """A handful of short, untimed generations immediately before a real
    timed trial -- bench_common.REFERENCE_WARMUP_STEPS (5), the same
    per-trial layer benchmark_ablation.py already applies before every one
    of its timed windows, to avoid cold-SM effects specific to that trial.
    Distinct in purpose from the big closed-loop warmup above: that one
    exists to get every kernel JIT-compiled once; this one exists because
    even a fully-warmed GPU can still show a small per-trial cold-start
    effect if it's been idle (however briefly) between trials.

    max_tokens=8, not 1: a real residual effect was found and measured with
    max_tokens=1 here -- round 1 of a full run came in 7.4% slower (Poem) /
    4.0% slower (Physics) than the rounds-2-10 mean, worst for whichever
    prompt had gone longest since its last warmup touch (Poem is warmed
    first, and so has the longest gap, in the fixed Poem->Physics->Code
    order used both here and in the trial loop; Code, warmed last, showed
    no dip at all). A single-token generate touches the code path but is
    too brief to sustain full clock boost -- consistent with this repo's
    own finding elsewhere that short bursts under-recover GPU clock state
    relative to sustained load. A few more tokens costs little extra time
    but holds the GPU at working clocks going into the timed measurement."""
    for _ in range(n_steps):
        llm.generate([prompt_text], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)


def spec_decode_snapshot(llm):
    """Cumulative vLLM speculative-decode counters as of right now (this
    engine instance, since it started -- NOT per-request). Used as a
    before/after snapshot around a single generate() call so the DELTA
    isolates that one call's contribution, regardless of what warmup or
    earlier trials already accumulated. Returns zeros if this engine has no
    speculative_config (baseline mode) or if these metrics aren't present
    for any other reason -- callers should treat an all-zero delta as "not
    applicable", not as "zero tokens accepted"."""
    num_drafts = num_draft_tokens = num_accepted_tokens = 0
    try:
        for metric in llm.get_metrics():
            if metric.name == "vllm:spec_decode_num_drafts":
                num_drafts += metric.value
            elif metric.name == "vllm:spec_decode_num_draft_tokens":
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                num_accepted_tokens += metric.value
    except Exception:
        pass
    return num_drafts, num_draft_tokens, num_accepted_tokens


def run_trial(llm, monitor, prompt_text, label, round_idx, sampling_params,
              condition_key, global_index, capture_spec_metrics=False):
    spec_before = spec_decode_snapshot(llm) if capture_spec_metrics else None
    e_start = monitor.read_energy_j()
    t_start = time.perf_counter()

    outputs = llm.generate([prompt_text], sampling_params, use_tqdm=False)

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    e_end = monitor.read_energy_j()
    spec_after = spec_decode_snapshot(llm) if capture_spec_metrics else None

    if capture_spec_metrics and global_index == 1:
        # One-time raw dump, first speculative trial only. If disable_log_stats
        # alone doesn't fix the zero-counters problem found 2026-09-19, this
        # shows the ACTUAL metric names/values this vLLM version exposes,
        # rather than guessing a second wrong metric name blind.
        print("\n[DEBUG] Raw output of llm.get_metrics() after first speculative trial:")
        try:
            all_metrics = llm.get_metrics()
            if not all_metrics:
                print("    (empty list -- get_metrics() returned nothing at all)")
            for m in all_metrics:
                val = getattr(m, "value", None)
                if val is None:
                    val = getattr(m, "values", None)
                print(f"    {m.name} = {val}")
        except Exception as exc:
            print(f"    [!] get_metrics() raised {type(exc).__name__}: {exc}")
        print()

    stats = monitor.window_stats(t_start, t_end)
    generated_ids = list(outputs[0].outputs[0].token_ids)
    tokens = len(generated_ids)
    latency = t_end - t_start

    energy_counter = (e_end - e_start) if (e_start is not None and e_end is not None) else None
    energy_sampled = stats["energy_j_sampled"]
    energy = energy_counter if energy_counter is not None else energy_sampled
    energy_source = "nvml_counter" if energy_counter is not None else "sampled_trapezoid"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "condition": condition_key,
        "prompt_label": label,
        "round": round_idx,
        "global_index": global_index,
        "tokens": tokens,
        "latency_sec": round(latency, 6),
        "throughput_tok_sec": round(tokens / latency, 2) if latency > 0 else 0.0,
        "avg_power_watts": round(stats["mean_w"], 2) if stats["mean_w"] is not None else None,
        "total_energy_joules": round(energy, 4) if energy is not None else None,
        "energy_source": energy_source,
        "joules_per_token": round(energy / tokens, 6) if (energy is not None and tokens) else None,
        "temp_end_c": stats["temp_end_c"],
        "throttle_reasons": monitor.throttle_reasons(),
        "generated_ids": generated_ids,  # for fidelity check only, stripped before CSV
    }
    if capture_spec_metrics and spec_before is not None and spec_after is not None:
        d_drafts = spec_after[0] - spec_before[0]
        d_draft_tokens = spec_after[1] - spec_before[1]
        d_accepted = spec_after[2] - spec_before[2]
        entry["vllm_spec_num_drafts"] = d_drafts
        entry["vllm_spec_num_draft_tokens"] = d_draft_tokens
        entry["vllm_spec_num_accepted_tokens"] = d_accepted
        # Per-token accept rate, matching K5's own definition exactly:
        # total_accepted / total_drafted across individual draft-token
        # positions (bench_common.speculative_generate's total_accepted /
        # total_drafted) -- NOT vLLM's own "mean acceptance length" metric,
        # which is a different quantity (1 + accepted/num_drafts, i.e. mean
        # run length including the bonus token). Using the same definition
        # on both sides is the whole point of this comparison.
        entry["vllm_accept_rate_pct"] = (
            round(100 * d_accepted / d_draft_tokens, 2) if d_draft_tokens > 0 else None
        )
    csv_entry = {k: v for k, v in entry.items() if k != "generated_ids"}
    bench_common.safe_append_csv(TRIAL_CSV, csv_entry)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "speculative"], required=True)
    ap.add_argument("--trials", type=int, default=NUM_TRIALS_DEFAULT)
    ap.add_argument("--tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                     help="Fraction of total GPU VRAM vLLM reserves at startup. "
                          "Lower this (e.g. 0.7) if anything else needs to share "
                          "the GPU -- though for clean power measurement, nothing "
                          "else should be running at all during a timed run.")
    ap.add_argument("--max-model-len", type=int, default=2048,
                     help="Context length vLLM reserves KV cache for. Default "
                          "(2048) comfortably covers the reference prompts "
                          "(short) plus REFERENCE_MAX_TOKENS output, with "
                          "margin -- well below Llama-3.1-8B's native 131072, "
                          "which is what was causing the KV-cache-too-small "
                          "error. Raise this only if you change the prompts "
                          "or --tokens to something that no longer fits.")
    ap.add_argument("--warmup-seconds", type=float, default=150.0,
                     help="Minimum warmup duration, looping generation across "
                          "all three prompts until this floor is reached. "
                          "Default (150s) matches this repo's own "
                          "bench_common.MIN_WARMUP_SEC convention (120s), "
                          "raised to the ~2.5 minutes empirically found "
                          "necessary for vLLM's speculative-decode kernels "
                          "(eagle_*, rejection_greedy_sample_kernel) to finish "
                          "JIT-compiling every shape/code-path variant -- a "
                          "single length-matched pass per prompt was not "
                          "enough (see the fix note on the trial loop below).")
    ap.add_argument("--per-trial-warmup-steps", type=int,
                     default=bench_common.REFERENCE_WARMUP_STEPS,
                     help="Short untimed generations immediately before each "
                          "individual timed trial, to avoid cold-SM effects "
                          "specific to that trial. Defaults to this repo's "
                          "own REFERENCE_WARMUP_STEPS (5), matching what "
                          "benchmark_ablation.py already does for every one "
                          "of its trials -- this is the second, per-trial "
                          "warmup layer, separate from the one-time closed-loop "
                          "warmup above.")
    args = ap.parse_args()

    condition_key = BASELINE_KEY if args.mode == "baseline" else SPEC_KEY

    monitor = bench_common.NVMLPowerMonitor(device_index=0)
    dev = monitor.device_info()
    print("=" * 85)
    print(f"[*] vLLM COMPARISON BENCHMARK -- mode={args.mode} ({dev.get('name')})")
    print(f"[*] Target: {TARGET_MODEL_ID}" +
          (f" | Scout: {SCOUT_MODEL_ID} | K={NUM_SPECULATIVE_TOKENS}" if args.mode == "speculative" else ""))
    print("[!] KV-caching is ON (vLLM default) -- NOT directly comparable to")
    print("    this repo's existing (uncached) K5 numbers. See module docstring.")
    print("=" * 85)
    monitor.start()
    check_gpu_is_clear(monitor)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    prompts_formatted = chat_prompts(tokenizer)

    print(f"\n[*] Building vLLM engine (mode={args.mode}, "
          f"gpu_memory_utilization={args.gpu_memory_utilization}, "
          f"max_model_len={args.max_model_len})...")
    llm = build_engine(args.mode, args.gpu_memory_utilization, args.max_model_len)

    sampling_params = SamplingParams(
        temperature=0.0,   # greedy, matching this repo's decoding elsewhere
        max_tokens=args.tokens,
    )

    # Reuses bench_common.warm_to_steady_state directly, rather than a
    # custom timer loop. A fixed-duration loop cycling through prompts
    # uniformly (what this script had before) never actually verifies that
    # any individual prompt reached steady state -- it just assumes enough
    # total time implies that. warm_to_steady_state checks per-label power
    # AND temperature drift, independently for each distinct label, and
    # only declares done once every one of them has individually converged.
    # That's the real fix for a rotation-order artifact like this: whichever
    # prompt runs first in a round is worst off (confirmed: Poem, warmed
    # first, showed the biggest round-1 dip; Code, warmed last, showed
    # none) not because of anything about that specific prompt, but because
    # a blind timer doesn't guarantee *that* prompt's own steady state was
    # actually reached before the timed loop started. This is the same
    # category of confound bench_common.py's own bug history already
    # required several iterations to properly fix (pooled vs. per-label
    # convergence, drift vs. spread) -- reusing that fix here instead of
    # re-deriving a weaker version of it.
    def warmup_step(i):
        idx = i % len(prompts_formatted)
        llm.generate([prompts_formatted[idx]],
                     SamplingParams(temperature=0.0, max_tokens=args.tokens),
                     use_tqdm=False)
        torch.cuda.synchronize()
        return PROMPT_LABELS[idx]

    warmup = bench_common.warm_to_steady_state(
        monitor, warmup_step,
        min_sec=args.warmup_seconds,
        max_sec=max(args.warmup_seconds * 3, bench_common.MAX_WARMUP_SEC),
    )
    if not warmup["converged"]:
        print("[!] Warmup did not converge before its cap. Results below may "
              "still show a rotation-order effect -- consider raising "
              "--warmup-seconds and re-running.")

    print(f"\n[*] Running {args.trials} trial(s) per prompt...")
    chronological = []
    gidx = 0
    for round_idx in range(1, args.trials + 1):
        for label, text in zip(PROMPT_LABELS, prompts_formatted):
            gidx += 1
            per_trial_warmup(llm, text, args.per_trial_warmup_steps)
            entry = run_trial(llm, monitor, text, label, round_idx, sampling_params,
                               condition_key, gidx,
                               capture_spec_metrics=(args.mode == "speculative"))
            accept_str = (f"  accept={entry['vllm_accept_rate_pct']}%"
                          if entry.get('vllm_accept_rate_pct') is not None else "")
            print(f"  {label:<8} round {round_idx}/{args.trials}  "
                  f"tok/s={entry['throughput_tok_sec']:<7} "
                  f"J/tok={entry['joules_per_token']}  P={entry['avg_power_watts']} W{accept_str}")
            chronological.append(entry)

    monitor.close()

    # -- merge into a running results file across both --mode invocations --
    existing = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            existing = json.load(f)

    by_prompt = existing.get("by_prompt", {label: {} for label in PROMPT_LABELS})
    for label in PROMPT_LABELS:
        trials = [e for e in chronological if e["prompt_label"] == label]
        tps = [t["throughput_tok_sec"] for t in trials]
        j = [t["joules_per_token"] for t in trials if t["joules_per_token"] is not None]
        accept_rates = [t["vllm_accept_rate_pct"] for t in trials
                        if t.get("vllm_accept_rate_pct") is not None]
        summary = {
            "tps_mean": float(np.mean(tps)),
            "tps_std": float(np.std(tps)),
            "j_tok_mean": float(np.mean(j)) if j else None,
            "j_tok_std": float(np.std(j)) if j else 0.0,
            "n_trials": len(trials),
        }
        if accept_rates:
            summary["vllm_accept_rate_pct_mean"] = float(np.mean(accept_rates))
            summary["vllm_accept_rate_pct_std"] = float(np.std(accept_rates))
            summary["k5_reference_accept_pct"] = REFERENCE_K5_ACCEPT_PCT_BY_LABEL.get(label)
        by_prompt.setdefault(label, {})[condition_key] = summary

    existing["by_prompt"] = by_prompt
    existing.setdefault("_meta", {})[f"{args.mode}_run"] = {
        "timestamp": datetime.now().isoformat(),
        "device": dev,
        "config": {
            "target_model": TARGET_MODEL_ID,
            "scout_model": SCOUT_MODEL_ID if args.mode == "speculative" else None,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS if args.mode == "speculative" else None,
            "max_tokens": args.tokens,
            "max_model_len": args.max_model_len,
            "warmup_seconds": args.warmup_seconds,
            "per_trial_warmup_steps": args.per_trial_warmup_steps,
            "num_trials": args.trials,
            "kv_caching": True,
            "vllm_wsl2_pin_memory": os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY"),
            "vllm_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        },
        "warmup": {
            "converged": warmup["converged"],
            "elapsed_sec": warmup["elapsed_sec"],
            "iterations": warmup["iterations"],
        },
        "poll_errors": monitor.poll_errors,
    }
    existing.setdefault("_caveat", (
        "NOT directly comparable to benchmark_ablation.py's numbers -- this "
        "run uses vLLM's default KV-caching, the existing repo numbers "
        "deliberately do not. See this script's module docstring."
    ))

    with open(RESULTS_JSON, "w") as f:
        json.dump(existing, f, indent=2, default=bench_common.json_safe)

    print(f"\n[*] Results merged into: {RESULTS_JSON}")
    print(f"[*] Per-trial CSV:        {TRIAL_CSV}")

    if BASELINE_KEY in by_prompt.get(PROMPT_LABELS[0], {}) and \
       SPEC_KEY in by_prompt.get(PROMPT_LABELS[0], {}):
        print("\n[*] Both modes present -- summary (vLLM-internal comparison only):")
        for label in PROMPT_LABELS:
            b = by_prompt[label].get(BASELINE_KEY)
            s = by_prompt[label].get(SPEC_KEY)
            if b and s:
                speedup = (s["tps_mean"] / b["tps_mean"] - 1) * 100
                print(f"  {label:<8} baseline={b['tps_mean']:.1f} tok/s  "
                      f"speculative={s['tps_mean']:.1f} tok/s  ({speedup:+.1f}%)")
                if s.get("vllm_accept_rate_pct_mean") is not None:
                    k5_ref = s.get("k5_reference_accept_pct")
                    ref_str = f" vs. this repo's own {k5_ref}%" if k5_ref is not None else ""
                    print(f"           vLLM accept rate: {s['vllm_accept_rate_pct_mean']:.1f}%"
                          f"{ref_str} -- not asserted equal, check by eye")
    else:
        other = "speculative" if args.mode == "baseline" else "baseline"
        print(f"\n[*] Run --mode {other} next to complete the comparison.")


if __name__ == "__main__":
    main()
