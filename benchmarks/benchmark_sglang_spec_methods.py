"""
benchmark_sglang_spec_methods.py

SGLang counterpart to benchmark_vllm_spec_methods.py: compares K5's own configuration
(scout + target + K=5) against vLLM's OTHER major open-source competitor -- SGLang --
using SGLang's own built-in speculative-decoding methods, on the same target model,
same prompts, same hardware.

===============================================================================
READ THIS BEFORE RUNNING OR CITING ANYTHING FROM THIS SCRIPT
===============================================================================

WHY THIS SCRIPT EXISTS (session context, 2026-09-20):
  SGLang was previously deprioritized in this project on the reasoning "SGLang's
  speculative decoding is adaptive, closer to SDSIE's entropy-gated mechanism than to
  fixed-K5." That reasoning was checked against SGLang's actual current documentation
  and found WRONG -- SGLang ships the same kind of fixed-method menu vLLM does:
    - STANDALONE  -- a real, separate smaller draft model. This is SGLang's direct
                     equivalent of vLLM's `draft_model` / this repo's own mechanism.
    - NGRAM       -- no separate model, matches repeated text. Equivalent to vLLM's ngram.
    - EAGLE/EAGLE3 -- SGLang's flagship, EAGLE-based method (same EAGLE checkpoint
                     family already used against vLLM).
  A genuinely adaptive layer exists ON TOP of EAGLE (--speculative-eagle-topk 1 + an
  opt-in flag) but is not what any of the three methods above do by default.

USES THE OFFLINE ENGINE API, NOT A SERVER:
  SGLang also supports launching a full HTTP server (the pattern used in SGLang's own
  test fixtures and most of its docs). This script deliberately uses `sgl.Engine(...)`,
  SGLang's in-process API, matching the exact call pattern (`llm.generate(prompts,
  sampling_params)`) already used for vLLM elsewhere in this repo -- no subprocess
  management, no HTTP polling, no port juggling.

KEY DIFFERENCE FROM vLLM'S CONFIG SHAPE:
  vLLM nests speculative settings under one `speculative_config` dict. SGLang's
  `ServerArgs` (and therefore `Engine(**kwargs)`) is FLAT -- `speculative_algorithm`,
  `speculative_draft_model_path`, `speculative_num_steps`, `speculative_eagle_topk`,
  `speculative_num_draft_tokens` are all separate top-level keyword arguments, not a
  nested dict. Also note the renamed equivalents used elsewhere in this repo's vLLM
  scripts: gpu_memory_utilization -> mem_fraction_static, max_model_len -> context_length.

NOT A TUNING STUDY, SAME AS THE vLLM VERSION:
  num_speculative_tokens (`speculative_num_steps` here) is held at bench_common.REFERENCE_K
  (5) for STANDALONE and EAGLE, matching K5's own value, to isolate the drafting
  mechanism rather than each method's own best-tuned depth. `speculative_eagle_topk` is
  fixed at 1 (single-path drafting, no tree) for EAGLE specifically, matching the same
  simplification already used in the vLLM EAGLE test -- SGLang's EAGLE can do much more
  (branching tree search) but exercising that is a different, larger question than "does
  a matched-K comparison favor K5's simple approach or not."

TWO THINGS THIS SCRIPT DOES NOT YET KNOW FOR CERTAIN -- CHECK THE DEBUG DUMP BEFORE
TRUSTING ANY ACCEPT-RATE NUMBER OR ANY NGRAM RESULT:
  1. Accept-rate access. SGLang exposes `avg_spec_accept_length` via
     `internal_states[0]` on whatever `get_server_info()`-equivalent method the
     in-process Engine exposes. The exact method name on `sgl.Engine` was NOT directly
     confirmed before writing this script -- `get_server_info()` is the best-supported
     guess from SGLang's own internals. This script tries a short list of candidate
     method/attribute names and prints a full raw dump of whatever it finds on the
     first trial of each speculative method, exactly like the analogous vLLM script did
     for `get_metrics()`. Do not trust any accept-rate number below until that dump has
     been read.
  2. Accept-rate is a SERVER-CUMULATIVE RUNNING AVERAGE (avg_spec_accept_length), not a
     raw counter pair like vLLM exposed. That means the clean before/after DELTA
     technique used for vLLM (subtract two cumulative counts to isolate one trial) does
     NOT cleanly apply to an average the same way -- averaging an average across an
     unknown internal sample count is not simply subtractable. This script reports the
     RAW server-cumulative value at trial's end as a coarser signal, not a precise
     per-trial figure, and flags this explicitly in the output. If the raw internal
     state turns out to also expose non-averaged counters (unconfirmed until the debug
     dump is read), a cleaner delta-based capture should replace this.
  3. NGRAM's specific tunable parameters (equivalent to vLLM's prompt_lookup_max/min)
     were NOT found in what was checked before writing this script. This script passes
     only `speculative_algorithm="NGRAM"` plus the shared `speculative_num_steps` /
     `speculative_num_draft_tokens` and lets SGLang use whatever its own NGRAM defaults
     are for anything else -- if SGLang's NGRAM needs additional required parameters,
     engine construction will fail with an error naming them, which is the fastest way
     to find out what's actually needed rather than guessing further.

SAME FIDELITY-CHECK DESIGN AS benchmark_vllm_spec_methods.py, SAME CAVEAT:
  --method baseline saves each prompt's exact output tokens; every speculative method
  compares against that, reporting percent-of-tokens-matched and the exact divergence
  index (NOT a binary pass/fail -- seebench_common's cognitive_fidelity_check.py for why
  a binary verdict throws away exactly the information that matters). This has the same
  structural limitation already found for vLLM: SGLang also fixes its speculative
  configuration at Engine construction, so baseline and every speculative method are
  necessarily separate process instances, and separate engine instances can disagree on
  near-tied token predictions from ordinary batch-dependent floating-point arithmetic,
  independent of whether the speculative logic is correct. Treat divergence the same way:
  informative, but not proof of an actual bug, unless corroborated by a
  baseline-vs-baseline cross-check (not yet built for SGLang).

RUN ORDER (baseline first, same as the vLLM version):
    python benchmark_sglang_spec_methods.py --method baseline
    python benchmark_sglang_spec_methods.py --method standalone
    python benchmark_sglang_spec_methods.py --method ngram
    python benchmark_sglang_spec_methods.py --method eagle

Requires sglang installed (per the user's setup: separate `.sglangvenv`). Uses the
already-downloaded EAGLE checkpoint (yuhuili/EAGLE-LLaMA3.1-Instruct-8B) for the eagle
method, and this repo's own scout (Llama-3.2-1B-Instruct) for standalone.
"""

import argparse
import json
import os
import time
from datetime import datetime

os.environ.setdefault("SGLANG_NGRAM_FORCE_GREEDY_VERIFY", "True")
# Found in SGLang's docs 2026-09-20, in direct response to a suspicious ngram result:
# throughput ~370 tok/s and ~76% accept rate on ALL three reference prompts uniformly
# (including open-ended prose that should have little repeated text to exploit), which
# survived even a genuinely fresh, never-before-seen prompt (ruling out a
# repetition/caching explanation). "Force greedy verify" being something that has to be
# explicitly turned ON implies the DEFAULT may not strictly check that a drafted token
# matches the target model's own greedy choice -- i.e. looser-than-lossless acceptance
# by default. Forcing it on here so this repo's own ngram numbers are measuring genuine,
# verified speculative decoding, not an artifact of relaxed verification. If throughput
# drops sharply and/or fidelity improves once this is set, that confirms the theory.

import numpy as np
import torch
from transformers import AutoTokenizer

import bench_common

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SCOUT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_EAGLE_DIR = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
DEFAULT_EAGLE3_DIR = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"  # separately-trained checkpoint,
# NOT interchangeable with DEFAULT_EAGLE_DIR -- EAGLE and EAGLE3 use different draft-head
# architectures. This exact name is sourced from vLLM's own official example script
# (spec_decode_offline.py, uploaded to this project 2026-09-20), not guessed.
NUM_TRIALS_DEFAULT = 10

PROMPT_LABELS = bench_common.REFERENCE_PROMPT_LABELS
PROMPTS = bench_common.REFERENCE_PROMPTS

# K5's own already-verified accept rates -- reference point only, same convention as
# the vLLM scripts. Not asserted equal to anything measured here.
REFERENCE_K5_ACCEPT_PCT_BY_LABEL = {"Poem": 42.5, "Physics": 51.7, "Code": 85.4}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(TELEMETRY_DIR, "sglang_spec_methods_results.json")
TRIAL_CSV = os.path.join(TELEMETRY_DIR, "telemetry_sglang_spec_methods.csv")
BASELINE_TOKENS_JSON = os.path.join(TELEMETRY_DIR, "sglang_spec_methods_baseline_tokens.json")

METHOD_LABELS = {
    "baseline": "SGLang FP16 Baseline",
    "standalone": "SGLang STANDALONE (K5 config)",
    "ngram": "SGLang NGRAM",
    "eagle": "SGLang EAGLE",
    "eagle3": "SGLang EAGLE3",
}


# A prompt that never appears anywhere else in this script -- not in PROMPTS, not
# reused across trials or warmup. Used ONLY for the novel-prompt sanity check below,
# specifically to test whether ngram's suspiciously high, suspiciously uniform accept
# rate (found 2026-09-20: ~76% on ALL three prompts, including open-ended prose that
# should have little repeated text to exploit) comes from genuinely predicting text, or
# from the engine having effectively already seen this exact request's answer once
# before -- this benchmark repeats each of the 3 reference prompts 10 times, plus many
# more times during closed-loop warmup, which is exactly the condition that would let a
# text-repetition-based method "cheat" by recognizing its own prior output rather than
# genuinely predicting fresh continuations.
NOVEL_CHECK_PROMPT = (
    "Describe the process of glassblowing, from gathering molten glass on a punty rod "
    "through shaping, annealing, and final finishing."
)


def build_engine(method, num_spec_tokens, eagle_dir, eagle3_dir, mem_fraction_static, context_length):
    """SGLang's Engine takes speculative settings as FLAT top-level kwargs, unlike
    vLLM's single nested speculative_config dict -- see module docstring."""
    import sglang as sgl

    kwargs = dict(
        model_path=TARGET_MODEL_ID,
        mem_fraction_static=mem_fraction_static,
        context_length=context_length,
        # Explicit, not left to SGLang's per-model "auto" inference -- added
        # 2026-09-21 after a real ValueError on --method eagle ("Mismatched
        # Tensor... expected dtype=bfloat16" in a fused Triton RMSNorm kernel).
        # Without this, SGLang infers dtype separately from each checkpoint's own
        # config.json -- target and draft independently. Every real, working
        # SGLang EAGLE/EAGLE3 launch example found (SGLang's own issue tracker,
        # plus a published example for this exact EAGLE3-LLaMA3.1-8B checkpoint
        # family) explicitly passes --dtype bfloat16, which is the strong signal
        # this fixes it, but this has NOT been re-run to confirm -- flag it as
        # such if it doesn't.
        dtype="bfloat16",
    )
    if method == "standalone":
        kwargs.update(
            speculative_algorithm="STANDALONE",
            speculative_draft_model_path=SCOUT_MODEL_ID,
            speculative_num_steps=num_spec_tokens,
            speculative_eagle_topk=1,  # no tree -- single-path draft, matches K5's own loop
            speculative_num_draft_tokens=num_spec_tokens,
        )
    elif method == "ngram":
        kwargs.update(
            speculative_algorithm="NGRAM",
            speculative_num_draft_tokens=16,
            # NOT speculative_num_steps=num_spec_tokens -- deliberately deviating from
            # this project's own "hold K fixed at 5 across every method" rule, found
            # 2026-09-20 to be a bad fit for ngram specifically. SGLang's own official
            # test fixture for ngram (ngram_fixture.py, NgramServerBase) never sets
            # speculative_num_steps at all, and uses speculative_num_draft_tokens=16,
            # not 5 -- and that same fixture's own accuracy test expects an accept
            # length around 1.8 (gsm8k_accept_length_thres), nowhere near the ~4.8 this
            # project measured at num_draft_tokens=5. Forcing a small, non-standard
            # draft-token budget onto a code path SGLang's own team never tests that way
            # is the likely cause of the earlier anomalous (~370 tok/s, ~76% accept rate
            # on every prompt including open-ended prose) result -- not yet CONFIRMED,
            # but this is the first lead all session backed by SGLang's own source, not
            # a guess. If results at num_draft_tokens=16 land closer to the ~1.8 accept
            # length SGLang's own team expects, that confirms it.
        )
    elif method == "eagle":
        kwargs.update(
            speculative_algorithm="EAGLE",
            speculative_draft_model_path=eagle_dir,
            speculative_num_steps=num_spec_tokens,
            speculative_eagle_topk=1,  # single-path, matches the vLLM EAGLE test's simplicity
            speculative_num_draft_tokens=num_spec_tokens,
        )
    elif method == "eagle3":
        kwargs.update(
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=eagle3_dir,
            speculative_num_steps=num_spec_tokens,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=num_spec_tokens,
            # NOTE: the checkpoint's own published example (SpecForge team, a DIFFERENT
            # EAGLE3 checkpoint for this same target model) recommends num_steps=3,
            # topk=1, num_draft_tokens=4 as its tuned defaults -- deliberately NOT used
            # here. This project holds K fixed at REFERENCE_K across every method on
            # every platform so the comparison isolates the drafting mechanism, not each
            # checkpoint's own best-tuned depth. Worth knowing this checkpoint may be
            # underperforming its own potential at K=5 as a result -- a fair-K
            # comparison and a best-tuned comparison are different, both-valid
            # questions, and this script deliberately answers the first one.
        )
    elif method != "baseline":
        raise ValueError(f"unknown method: {method}")

    return sgl.Engine(**kwargs)


def sglang_spec_snapshot(llm):
    """Best-effort read of SGLang's internal speculative-decoding stats. UNCONFIRMED
    method name -- tries a short candidate list and returns (raw_dict_or_None,
    method_name_used_or_None). Callers should print/inspect raw_dict on first use
    rather than trust any parsed number blind. See module docstring point 1."""
    candidates = ["get_server_info", "get_internal_state"]
    for name in candidates:
        fn = getattr(llm, name, None)
        if fn is None:
            continue
        try:
            info = fn()
            return info, name
        except Exception:
            continue
    return None, None


def extract_avg_accept_length(info):
    """Pull internal_states[0].avg_spec_accept_length out of a server_info-shaped
    dict, unwrapping a 'decode' stage key if present (PD-disaggregated servers use
    this per SGLang's own bench_serving convention). Returns None if not found --
    callers must treat that as 'not confirmed available', not zero."""
    if not isinstance(info, dict):
        return None
    states = info.get("internal_states")
    if not states or not isinstance(states, list):
        return None
    state = states[0]
    if isinstance(state, dict) and "decode" in state and isinstance(state["decode"], dict):
        state = state["decode"]
    if isinstance(state, dict):
        return state.get("avg_spec_accept_length")
    return None


def load_baseline_tokens():
    if not os.path.exists(BASELINE_TOKENS_JSON):
        return None
    with open(BASELINE_TOKENS_JSON) as f:
        return json.load(f)


def per_trial_warmup(llm, prompt_text, n_steps, sampling_params_warmup):
    for _ in range(n_steps):
        llm.generate([prompt_text], sampling_params_warmup)


def run_trial(llm, monitor, prompt_text, label, round_idx, sampling_params,
              method, global_index, baseline_tokens):
    capture_spec = method != "baseline"
    e_start = monitor.read_energy_j()
    t_start = time.perf_counter()

    outputs = llm.generate([prompt_text], sampling_params)

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    e_end = monitor.read_energy_j()

    spec_info = spec_method_name = None
    if capture_spec:
        spec_info, spec_method_name = sglang_spec_snapshot(llm)
        if global_index == 1:
            print(f"\n[DEBUG] SGLang spec-decode state after first '{method}' trial "
                  f"(via {spec_method_name!r}):")
            if isinstance(spec_info, dict):
                print(f"    top-level keys: {list(spec_info.keys())}")
                internal = spec_info.get("internal_states")
                print(f"    internal_states (the part that actually matters -- NOT truncated):")
                print(f"    {json.dumps(internal, indent=2, default=str)}")
            else:
                print(f"    (not a dict -- raw value: {spec_info!r})")
            print()

    stats = monitor.window_stats(t_start, t_end)
    out = outputs[0]
    generated_text = out.get("text", "") if isinstance(out, dict) else str(out)
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    tokens = meta.get("completion_tokens", len(generated_text.split()))
    generated_ids = meta.get("output_ids")  # may be None depending on SGLang version/config
    latency = t_end - t_start

    energy_counter = (e_end - e_start) if (e_start is not None and e_end is not None) else None
    energy_sampled = stats["energy_j_sampled"]
    energy = energy_counter if energy_counter is not None else energy_sampled
    energy_source = "nvml_counter" if energy_counter is not None else "sampled_trapezoid"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
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
        # Fixed-width schema regardless of method/outcome -- see
        # benchmark_vllm_spec_methods.py's 2026-09-19 bug history for why (a
        # sometimes-present column fragments the CSV into many _legacy files).
        "avg_spec_accept_length_cumulative": None,  # coarse, server-cumulative -- see docstring
        "fidelity_exact_match": None,
        "fidelity_match_pct": None,
        "fidelity_first_divergence_index": None,
    }

    if capture_spec:
        accept_len = extract_avg_accept_length(spec_info)
        if accept_len is not None:
            entry["avg_spec_accept_length_cumulative"] = round(float(accept_len), 4)

    # -- fidelity: percent tokens matched + divergence index, NOT binary pass/fail --
    if baseline_tokens is not None and label in baseline_tokens:
        ref = baseline_tokens[label]
        if generated_ids is not None and not isinstance(ref, str):
            common_len = min(len(generated_ids), len(ref))
            exact_match = (generated_ids == ref)
            entry["fidelity_exact_match"] = exact_match
            if exact_match:
                entry["fidelity_match_pct"] = 100.0
            else:
                first_diverge = next((i for i in range(common_len) if generated_ids[i] != ref[i]),
                                      common_len)
                entry["fidelity_first_divergence_index"] = first_diverge
                entry["fidelity_match_pct"] = round(100 * first_diverge / common_len, 2) if common_len else 0.0
                print(f"  [!] FIDELITY: {label} round {round_idx} matches for {first_diverge}/{common_len} "
                      f"tokens ({entry['fidelity_match_pct']:.1f}%) before diverging")
        elif isinstance(ref, str):
            # TOKEN-LEVEL comparison isn't available in this SGLang config (no
            # output_ids), so fall back to a WORD-LEVEL version of the exact same
            # percent-matched + divergence-index check, on the actual generated text.
            # Coarser than token-level (a word boundary isn't the same granularity as
            # a token boundary) but the same real signal, not a silent gap -- fixed
            # 2026-09-20 after the first standalone run recorded zero fidelity data
            # at all with no clear reason why.
            ref_words = ref.split()
            gen_words = generated_text.split()
            common_len = min(len(gen_words), len(ref_words))
            exact_match = (gen_words == ref_words)
            entry["fidelity_exact_match"] = exact_match
            if exact_match:
                entry["fidelity_match_pct"] = 100.0
            else:
                first_diverge = next((i for i in range(common_len) if gen_words[i] != ref_words[i]),
                                      common_len)
                entry["fidelity_first_divergence_index"] = first_diverge
                entry["fidelity_match_pct"] = round(100 * first_diverge / common_len, 2) if common_len else 0.0
                print(f"  [!] FIDELITY (word-level): {label} round {round_idx} matches for "
                      f"{first_diverge}/{common_len} words ({entry['fidelity_match_pct']:.1f}%) "
                      f"before diverging")
    if baseline_tokens is not None and label in baseline_tokens and generated_ids is None \
            and not isinstance(baseline_tokens[label], str):
        if global_index == 1:
            print(f"  [!] output_ids not present in meta_info -- fidelity check unavailable "
                  f"for this SGLang version/config. Check the debug dump above for the actual "
                  f"output shape.")

    csv_entry = dict(entry)
    bench_common.safe_append_csv(TRIAL_CSV, csv_entry)
    return entry, generated_ids, generated_text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["baseline", "standalone", "ngram", "eagle", "eagle3"], required=True)
    ap.add_argument("--trials", type=int, default=NUM_TRIALS_DEFAULT)
    ap.add_argument("--tokens", type=int, default=bench_common.REFERENCE_MAX_TOKENS)
    ap.add_argument("--num-spec-tokens", type=int, default=bench_common.REFERENCE_K,
                     help="speculative_num_steps, held fixed across methods -- see module docstring.")
    ap.add_argument("--eagle-dir", type=str, default=DEFAULT_EAGLE_DIR)
    ap.add_argument("--eagle3-dir", type=str, default=DEFAULT_EAGLE3_DIR)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--warmup-seconds", type=float, default=150.0)
    ap.add_argument("--per-trial-warmup-steps", type=int, default=bench_common.REFERENCE_WARMUP_STEPS)
    ap.add_argument("--skip-novel-check", action="store_true",
                     help="Skip the never-before-seen-prompt sanity check that otherwise runs "
                          "once, immediately after engine construction, before any warmup.")
    args = ap.parse_args()

    if args.method != "baseline":
        baseline_tokens = load_baseline_tokens()
        if baseline_tokens is None:
            print(f"[!] No baseline token reference found at {BASELINE_TOKENS_JSON}. "
                  f"Run --method baseline first. Continuing without fidelity checking.")
    else:
        baseline_tokens = None

    method_key = METHOD_LABELS[args.method]

    monitor = bench_common.NVMLPowerMonitor(device_index=0)
    dev = monitor.device_info()
    print("=" * 85)
    print(f"[*] SGLANG SPEC-METHODS COMPARISON -- method={args.method} ({dev.get('name')})")
    print(f"[*] Target: {TARGET_MODEL_ID}")
    if args.method == "standalone":
        print(f"[*] Scout: {SCOUT_MODEL_ID} | steps={args.num_spec_tokens}")
    elif args.method == "eagle":
        print(f"[*] Draft head: {args.eagle_dir} | steps={args.num_spec_tokens}")
    elif args.method == "eagle3":
        print(f"[*] Draft head: {args.eagle3_dir} | steps={args.num_spec_tokens}")
    print("=" * 85)
    monitor.start()
    bench_common.vllm_check_gpu_is_clear(monitor)  # generic NVML check, name predates SGLang use

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    prompts_formatted = bench_common.vllm_chat_prompts(tokenizer, PROMPTS)  # also generic, name predates SGLang use

    print(f"\n[*] Building SGLang engine (method={args.method})...")
    llm = build_engine(args.method, args.num_spec_tokens, args.eagle_dir, args.eagle3_dir,
                        args.mem_fraction_static, args.context_length)

    if args.method not in ("baseline",) and not args.skip_novel_check:
        # Fired ONCE, immediately after engine construction -- before warmup, before any
        # of the 3 reference prompts have been sent even once. This prompt has never
        # been seen by this engine instance under any circumstances. If accept rate here
        # is dramatically lower than what the same method shows on the (repeated,
        # warmed) reference prompts later in this run, that's real evidence the later
        # number is inflated by repetition exposure, not genuine prediction quality.
        print("\n[*] Novel-prompt sanity check (never-before-seen text, zero warmup exposure):")
        novel_before = sglang_spec_snapshot(llm)[0]
        t0 = time.perf_counter()
        novel_out = llm.generate([NOVEL_CHECK_PROMPT], {"temperature": 0.0, "max_new_tokens": args.tokens})
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        novel_after, novel_method_name = sglang_spec_snapshot(llm)
        novel_accept = extract_avg_accept_length(novel_after)
        novel_meta = novel_out[0].get("meta_info", {}) if isinstance(novel_out[0], dict) else {}
        novel_tokens = novel_meta.get("completion_tokens")
        novel_tps = round(novel_tokens / (t1 - t0), 2) if novel_tokens and (t1 - t0) > 0 else None
        print(f"    tok/s (single cold call): {novel_tps}")
        print(f"    avg_spec_accept_length (server-cumulative, includes this call): {novel_accept}")
        print(f"    Compare this against the SAME method's number on the repeated reference "
              f"prompts later in this run -- a large gap would point at repetition-exposure "
              f"inflation rather than genuine prediction quality.\n")

    sampling_params = {"temperature": 0.0, "max_new_tokens": args.tokens}
    sampling_params_warmup = {"temperature": 0.0, "max_new_tokens": args.tokens}

    def warmup_step(i):
        idx = i % len(prompts_formatted)
        out = llm.generate([prompts_formatted[idx]], sampling_params_warmup)
        torch.cuda.synchronize()
        if i < 3 or i % 50 == 0:
            # Real completion length, not assumed. Added 2026-09-20 specifically for
            # ngram, after warmup rounds were completing in ~0.3s each (hundreds of
            # rounds inside the fixed 150s floor) -- checking whether that's genuine
            # fast, full-length generation or early termination producing short output.
            meta = out[0].get("meta_info", {}) if isinstance(out[0], dict) else {}
            print(f"    [warmup diag] round {i}: completion_tokens={meta.get('completion_tokens')} "
                  f"(requested max_new_tokens={args.tokens})")
        return PROMPT_LABELS[idx]

    warmup = bench_common.warm_to_steady_state(
        monitor, warmup_step,
        min_sec=args.warmup_seconds,
        max_sec=max(args.warmup_seconds * 3, bench_common.MAX_WARMUP_SEC),
    )
    if not warmup["converged"]:
        print("[!] Warmup did not converge before its cap. Consider raising --warmup-seconds.")

    if args.method not in ("baseline",) and not args.skip_novel_check:
        # SAME check as before engine construction, but now AFTER warmup -- if this
        # method's shared internal state (e.g. ngram's trie, which accumulates across
        # every generation the whole process makes, not just per-prompt) has started
        # recognizing content from the hundreds of warmup repetitions of the 3
        # reference prompts, a genuinely novel topic should still show this method's
        # TRUE baseline skill, uncontaminated by that. A large gap between this
        # post-warmup reading and the pre-warmup one (printed earlier) is real evidence
        # of warmup-accumulated state inflating the numbers on the reference prompts
        # specifically, not genuine method quality.
        print("\n[*] Novel-prompt sanity check, POST-warmup (same unseen topic, now after "
              "hundreds of warmup rounds on the 3 reference prompts):")
        t0 = time.perf_counter()
        novel_out2 = llm.generate([NOVEL_CHECK_PROMPT], {"temperature": 0.0, "max_new_tokens": args.tokens})
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        novel_after2, _ = sglang_spec_snapshot(llm)
        novel_accept2 = extract_avg_accept_length(novel_after2)
        novel_meta2 = novel_out2[0].get("meta_info", {}) if isinstance(novel_out2[0], dict) else {}
        novel_tokens2 = novel_meta2.get("completion_tokens")
        novel_tps2 = round(novel_tokens2 / (t1 - t0), 2) if novel_tokens2 and (t1 - t0) > 0 else None
        print(f"    completion_tokens: {novel_tokens2}  tok/s: {novel_tps2}")
        print(f"    avg_spec_accept_length (server-cumulative, now includes ~all warmup rounds "
              f"plus this call): {novel_accept2}")
        print(f"    Compare against the PRE-warmup reading above. If this is much higher, that's "
              f"real evidence of warmup-accumulated state, not genuine skill on novel content.\n")

    print(f"\n[*] Running {args.trials} trial(s) per prompt...")
    chronological = []
    per_prompt_tokens = {label: [] for label in PROMPT_LABELS}
    gidx = 0
    for round_idx in range(1, args.trials + 1):
        for label, text in zip(PROMPT_LABELS, prompts_formatted):
            gidx += 1
            per_trial_warmup(llm, text, args.per_trial_warmup_steps, sampling_params_warmup)
            entry, gen_ids, gen_text = run_trial(llm, monitor, text, label, round_idx,
                                                  sampling_params, args.method, gidx, baseline_tokens)
            per_prompt_tokens[label].append(gen_ids if gen_ids is not None else gen_text)
            fid_str = f"  fidelity={entry['fidelity_match_pct']:.1f}%" if entry.get("fidelity_match_pct") is not None else ""
            accept_str = (f"  avg_accept_len={entry['avg_spec_accept_length_cumulative']}"
                          if entry.get("avg_spec_accept_length_cumulative") is not None else "")
            print(f"  {label:<8} round {round_idx}/{args.trials}  "
                  f"tok/s={entry['throughput_tok_sec']:<7} "
                  f"J/tok={entry['joules_per_token']}  P={entry['avg_power_watts']} W"
                  f"{accept_str}{fid_str}")
            chronological.append(entry)

    monitor.close()

    if args.method == "baseline":
        ref_tokens = {}
        any_ids = all(per_prompt_tokens[label][0] is not None and not isinstance(per_prompt_tokens[label][0], str)
                      for label in PROMPT_LABELS)
        if not any_ids:
            print("\n[!] output_ids not available from this SGLang config/version -- baseline "
                  "reference will store raw TEXT instead of token ids. Fidelity checks against "
                  "this reference will then be text-exact-match, not token-exact-match -- a "
                  "weaker but still real check. See the debug dump printed during this run for "
                  "what meta_info actually contained.")
        for label in PROMPT_LABELS:
            rounds = per_prompt_tokens[label]
            ref_tokens[label] = rounds[0]
            mismatched_rounds = [i for i, r in enumerate(rounds) if r != rounds[0]]
            if mismatched_rounds:
                print(f"[!] Baseline itself is NOT deterministic across rounds for {label}: "
                      f"rounds {mismatched_rounds} differ from round 0.")
        with open(BASELINE_TOKENS_JSON, "w") as f:
            json.dump(ref_tokens, f)
        print(f"\n[*] Baseline reference saved: {BASELINE_TOKENS_JSON}")

    # Save this method's own round-0 sample output for every method, not just baseline --
    # so the actual text can be READ, not just scored as a percentage-match number. A
    # divergence could be a harmless synonym swap or a genuinely broken response, and
    # the match percentage alone can't tell those apart -- only reading it can.
    sample_outputs_path = os.path.join(TELEMETRY_DIR, "sglang_spec_methods_sample_outputs.json")
    existing_samples = {}
    if os.path.exists(sample_outputs_path):
        with open(sample_outputs_path) as f:
            existing_samples = json.load(f)
    for label in PROMPT_LABELS:
        sample = per_prompt_tokens[label][0]
        sample_text = sample if isinstance(sample, str) else None  # token-id lists aren't human-readable directly
        existing_samples.setdefault(label, {})[method_key] = {
            "text": sample_text,
            "is_token_ids": sample_text is None,
        }
    with open(sample_outputs_path, "w") as f:
        json.dump(existing_samples, f, indent=2)
    print(f"[*] Sample outputs (for reading, not just scoring) saved: {sample_outputs_path}")

    existing = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            existing = json.load(f)

    by_prompt = existing.get("by_prompt", {label: {} for label in PROMPT_LABELS})
    for label in PROMPT_LABELS:
        trials = [e for e in chronological if e["prompt_label"] == label]
        tps = [t["throughput_tok_sec"] for t in trials]
        j = [t["joules_per_token"] for t in trials if t["joules_per_token"] is not None]
        accept_lens = [t["avg_spec_accept_length_cumulative"] for t in trials
                        if t.get("avg_spec_accept_length_cumulative") is not None]
        fidelity_checks = [t["fidelity_exact_match"] for t in trials if t.get("fidelity_exact_match") is not None]
        fidelity_match_pcts = [t["fidelity_match_pct"] for t in trials if t.get("fidelity_match_pct") is not None]
        summary = {
            "tps_mean": float(np.mean(tps)),
            "tps_std": float(np.std(tps)),
            "j_tok_mean": float(np.mean(j)) if j else None,
            "j_tok_std": float(np.std(j)) if j else 0.0,
            "n_trials": len(trials),
        }
        if accept_lens:
            # NOTE: coarse, server-cumulative running average -- see module docstring
            # point 2. Reported as the value at the END of this run's trials, not a
            # clean per-trial isolate the way vLLM's delta-based capture was.
            summary["avg_spec_accept_length_final_cumulative"] = accept_lens[-1]
            summary["k5_reference_accept_pct"] = REFERENCE_K5_ACCEPT_PCT_BY_LABEL.get(label)
        if fidelity_checks:
            summary["fidelity_exact_match_rate_pct"] = round(100 * sum(fidelity_checks) / len(fidelity_checks), 1)
        if fidelity_match_pcts:
            summary["fidelity_match_pct_mean"] = round(float(np.mean(fidelity_match_pcts)), 1)
            summary["fidelity_match_pct_min"] = round(float(np.min(fidelity_match_pcts)), 1)
            summary["fidelity_match_pct_max"] = round(float(np.max(fidelity_match_pcts)), 1)
        by_prompt.setdefault(label, {})[method_key] = summary

    existing["by_prompt"] = by_prompt
    existing.setdefault("_meta", {})[f"{args.method}_run"] = {
        "timestamp": datetime.now().isoformat(),
        "device": dev,
        "config": {
            "target_model": TARGET_MODEL_ID,
            "scout_model": SCOUT_MODEL_ID if args.method == "standalone" else None,
            "eagle_dir": args.eagle_dir if args.method == "eagle" else (args.eagle3_dir if args.method == "eagle3" else None),
            "num_speculative_tokens": args.num_spec_tokens if args.method != "baseline" else None,
            "max_tokens": args.tokens,
            "context_length": args.context_length,
            "mem_fraction_static": args.mem_fraction_static,
            "warmup_seconds": args.warmup_seconds,
            "per_trial_warmup_steps": args.per_trial_warmup_steps,
            "num_trials": args.trials,
        },
        "warmup": {
            "converged": warmup["converged"],
            "elapsed_sec": warmup["elapsed_sec"],
            "iterations": warmup["iterations"],
        },
        "poll_errors": monitor.poll_errors,
    }
    existing.setdefault("_caveat", (
        "Accept-rate here (avg_spec_accept_length_final_cumulative) is a SERVER-CUMULATIVE "
        "RUNNING AVERAGE at the end of this run's trials, NOT a clean per-trial isolate the "
        "way vLLM's delta-based accept_rate_pct was. Treat as a coarser signal. Fidelity "
        "checking has the same structural limitation found for vLLM: baseline and every "
        "speculative method are necessarily separate SGLang engine processes, so divergence "
        "can reflect ordinary batch-dependent floating-point drift between engine instances, "
        "not necessarily a correctness issue in the speculative logic itself."
    ))

    with open(RESULTS_JSON, "w") as f:
        json.dump(existing, f, indent=2, default=bench_common.json_safe)

    print(f"\n[*] Results merged into: {RESULTS_JSON}")
    print(f"[*] Per-trial CSV:        {TRIAL_CSV}")

    present_methods = [m for m in METHOD_LABELS if METHOD_LABELS[m] in by_prompt.get(PROMPT_LABELS[0], {})]
    if len(present_methods) > 1:
        print(f"\n[*] Methods present so far: {', '.join(present_methods)}")
        base_key = METHOD_LABELS["baseline"]
        for label in PROMPT_LABELS:
            row = by_prompt[label]
            b = row.get(base_key)
            print(f"  {label}:")
            for m in present_methods:
                if m == "baseline":
                    continue
                s = row.get(METHOD_LABELS[m])
                if not s:
                    continue
                speedup_str = ""
                if b:
                    speedup = (s["tps_mean"] / b["tps_mean"] - 1) * 100
                    speedup_str = f"  ({speedup:+.1f}% vs. baseline)"
                fid_str = (f"  fidelity={s['fidelity_match_pct_mean']}% avg"
                           if s.get("fidelity_match_pct_mean") is not None else "")
                print(f"    {m:<12} {s['tps_mean']:.1f} tok/s{speedup_str}{fid_str}")
    else:
        remaining = [m for m in METHOD_LABELS if m not in present_methods]
        print(f"\n[*] Run remaining methods next: {', '.join(remaining)}")


if __name__ == "__main__":
    main()
