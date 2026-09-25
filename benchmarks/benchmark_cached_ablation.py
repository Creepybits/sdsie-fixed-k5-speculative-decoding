"""
benchmark_cached_ablation.py

N-trial baseline-vs-speculative ablation, fixed K=5, three reference prompts
-- same experiment as benchmark_ablation.py, but with a real KV cache on both
conditions instead of full-context recompute every step.

===============================================================================
WHY THIS FILE EXISTS -- READ THIS BEFORE USING ANY NUMBER FROM IT
===============================================================================

This repo now has three benchmark scripts that each answer a different
question, and none of them should be silently substituted for another:

  benchmark_ablation.py          -- this repo's own mechanism, UNCACHED.
                                     Isolates the speculative mechanism itself
                                     from caching effects. Source of the
                                     paper's headline numbers.

  benchmark_cached_ablation.py   -- this repo's own mechanism, CACHED.
                                     (this file). Same prompts/K/models as
                                     benchmark_ablation.py, but with a real
                                     HF DynamicCache on both baseline and
                                     speculative. Exists to answer two
                                     questions neither other script can:
                                       (a) does the accept/reject mechanism
                                           still pay off once the baseline
                                           it's compared against is a
                                           realistic cached decode, not a
                                           full-recompute one?
                                       (b) is the "speculation seems to help
                                           MORE when cached" observation from
                                           the vLLM comparison (09-19 review)
                                           a property of caching, or an
                                           artifact of comparing two
                                           different implementations? This
                                           script isolates that by changing
                                           only the caching, not the code.

  benchmark_vllm_comparison.py   -- vLLM's OWN native speculative decoding,
                                     CACHED (vLLM's PagedAttention, not
                                     ours). Answers "does the speculative
                                     mechanism, as vLLM implements it, help
                                     inside a real serving engine."

STILL NOT a full "does this repo's implementation beat vLLM's" comparison,
even with this file added. This file's speculative condition uses a plain
HF DynamicCache, not PagedAttention/continuous batching/vLLM's fused
attention kernels -- there are real implementation differences beyond just
"has a cache or not" that could still make raw throughput differ between
this file and vLLM for reasons unrelated to the speculative mechanism. What
this file DOES let you say for the first time: whether the accept-rate-driven
speedup this repo measures uncached still shows up, and by how much, once a
cache is added to *this repo's own code* -- a clean, one-variable-changed
comparison against benchmark_ablation.py that neither of the other two
scripts can offer alone.

===============================================================================
CACHE CORRECTNESS -- WHY THIS GETS EXTRA SCRUTINY
===============================================================================

SDSIE's own root-folder scripts (sdsie_cuda_graph_engine.py,
sdsie_speculative_fast.py, sdsie_static_speculative.py) each independently
got scout/target KV-cache bookkeeping wrong after a verify cycle -- a cache
length pointer ending up one position ahead of what the model actually has
real data for. That is exactly the class of bug this file's approach is
built to avoid, not just hope to avoid:

  1. _selftest_cache_roundtrip() tests ONLY the crop mechanism, in isolation,
     before any generation logic touches it: build two caches with the
     identical prefill, grow one and crop it back, and require the two to
     give bit-identical logits for the same probe token. (Redesigned
     2026-09-21 -- see that function's docstring for why the first design
     was measuring bf16 rounding rather than crop.)
  2. Before any timed trial, this script runs its own cached
     implementation against bench_common.py's existing, already-verified
     UNCACHED implementation on the same prompt and compares the generated
     token IDs. Any difference must be a bf16 near-tie (both picked tokens
     score within rounding distance of the model's best choice) or the
     script refuses to proceed. While the self-test and gate run, every
     forward pass also checks that the position about to be used equals the
     cache's actual length (_check_pos) -- the direct test for the "cache
     pointer one ahead" bug class.
     See _fidelity_gate's docstring for why exact match is not a fair
     requirement between two differently-shaped bf16 computations.
  3. The existing within-run fidelity check (cached baseline vs. cached
     speculative, same as benchmark_ablation.py's Test 1) still runs too.

As of 2026-09-21 the redesigned checks have been run on a small random
Llama model on CPU (transformers 5.17.0), not yet on the real 8B/1B pair on
the RTX 5090 -- these are the checks that need to pass there before
trusting anything below them.
If either self-test fails, stop and get the actual error message /
`transformers.__version__` looked at rather than patching this file blind;
the fallback cache-tensor-access paths below are written for two different
transformers Cache internals because which one is installed here isn't
known with certainty from outside the machine.

===============================================================================
TELEMETRY FILES ARE DELIBERATELY NEW, NOT SHARED
===============================================================================

Writes to ablation_cached_results.json / telemetry_ablation_cached.csv --
never telemetry_ablation.csv or ablation_results.json. Those two already
hold two different runs appended together (see the 2026-09-19 review's
"telemetry-file trap" finding); giving this script its own files avoids
adding a third, differently-configured set of rows to that same pile.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import bench_common

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SCOUT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda:0"
NUM_TRIALS = 10
MAX_TOKENS = bench_common.REFERENCE_MAX_TOKENS
K_DRAFT = bench_common.REFERENCE_K
WARMUP_STEPS = bench_common.REFERENCE_WARMUP_STEPS  # per-trial in-trial warmup

# Matches MAX_TOKENS, not a shorter throwaway value -- benchmark_ablation.py's
# own history (see its WARMUP_UNIT_TOKENS comment) found that shorter warmup
# bursts can report "converged" at a power/temp level the real, longer trial
# then exceeds. Applying that already-learned lesson here rather than
# re-discovering it.
WARMUP_UNIT_TOKENS = MAX_TOKENS

PROMPT_LABELS = bench_common.REFERENCE_PROMPT_LABELS
PROMPTS = bench_common.REFERENCE_PROMPTS

BASELINE_KEY = "FP16 Baseline (KV-cached)"
SPEC_KEY = "Speculative (1B->8B, KV-cached)"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(TELEMETRY_DIR, "ablation_cached_results.json")
TRIAL_CSV = os.path.join(TELEMETRY_DIR, "telemetry_ablation_cached.csv")


# ============================================================================
# Cache crop helpers -- see module docstring for why these get a self-test
# ============================================================================

def _crop_cache(cache, new_len):
    """Truncate a transformers Cache object to its first new_len positions,
    in place, discarding KV for rejected speculative-draft tokens.

    Prefers the library's own Cache.crop() when available (added for exactly
    this use case). Falls back to slicing key_cache/value_cache directly for
    transformers versions that predate .crop(). Neither path is trusted on
    the strength of existing -- see _selftest_cache_roundtrip(), which must
    pass before this function is used for real.
    """
    if hasattr(cache, "crop"):
        # Verified against transformers 5.17.0's cache_utils.py source
        # (2026-09-21): DynamicLayer.crop() with a positive value is
        # deprecated ("will be removed in version 5.18"); a negative value
        # means "remove this many tokens from the end" and physically slices
        # the key/value tensors. Using the negative form here.
        current_len = _cache_seq_length(cache, fallback_len=new_len)
        remove = current_len - new_len
        if remove > 0:
            cache.crop(-remove)
        elif remove < 0:
            raise RuntimeError(
                f"_crop_cache asked to grow a cache from {current_len} to "
                f"{new_len} tokens -- that's not what crop is for, and "
                f"points at a bug in the caller, not this function."
            )
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :new_len, :]
            cache.value_cache[i] = cache.value_cache[i][..., :new_len, :]
        return cache
    raise RuntimeError(
        f"Don't know how to crop this transformers Cache implementation "
        f"({type(cache).__name__}) -- no .crop() method and no key_cache/"
        f"value_cache attributes found. This means the installed "
        f"transformers version uses a Cache internal layout this script "
        f"doesn't recognize. Check `python -c \"import transformers; "
        f"print(transformers.__version__)\"` and fix this function for "
        f"that version before running anything else here -- do not guess."
    )


def _cache_seq_length(cache, fallback_len=None):
    """Best-effort current sequence length held by a Cache object. Used only
    by the self-test's own bookkeeping and diagnostics -- the main
    generation loop tracks prefix_len itself and never relies on this."""
    if hasattr(cache, "get_seq_length"):
        try:
            return cache.get_seq_length()
        except Exception:
            pass
    if hasattr(cache, "key_cache") and cache.key_cache:
        return cache.key_cache[0].shape[-2]
    return fallback_len


# Bookkeeping check: before every forward pass, the position this file is
# about to assign must equal the number of entries the cache actually holds.
# If they ever differ, that is exactly the "cache pointer one position ahead
# of real data" bug SDSIE's root-folder scripts hit three times. Switched ON
# only while the self-test and fidelity gate run (so the real generation
# code is exercised under the check), OFF during timed trials so it adds
# nothing to measured time.
_POSITION_CHECKS = False


def _check_pos(cache, expected, where):
    if _POSITION_CHECKS:
        got = _cache_seq_length(cache)
        if got != expected:
            raise RuntimeError(
                f"[position check] {where}: about to use position {expected}, but the "
                f"cache holds {got} entries. The script's position bookkeeping and the "
                f"cache have fallen out of step -- this is a real cache bug. Do not "
                f"trust any number from this script.")


# Diagnostic phase timing -- added 2026-09-21 (evening) after real GPU runs (all three attn
# backends: sdpa, eager, flash_attention_2) showed the cached SPECULATIVE condition
# consistently SLOWER than the cached BASELINE despite normal accept rates (36-85%,
# matching the uncached ablation closely). That result is the opposite of what the
# mechanism should do, and needs an actual measured breakdown, not another guess.
# Zero overhead when off (no sync, no dict write) -- same on/off pattern as
# _POSITION_CHECKS above. See --profile-phases in main().
_PHASE_TIMING = False
_phase_times = {}


class _time_phase:
    """Context manager: if _PHASE_TIMING, synchronizes CUDA before/after and
    accumulates elapsed wall time into _phase_times[name]. A CUDA sync on
    every entry/exit means this is NOT free even when timing a fast phase --
    only ever enabled by --profile-phases, never during a real timed trial."""
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if _PHASE_TIMING:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _PHASE_TIMING:
            torch.cuda.synchronize()
            _phase_times[self.name] = _phase_times.get(self.name, 0.0) + (time.perf_counter() - self.t0)


def _probe(model, cache, token, pos):
    """One forward pass of `token` at position `pos` through `cache`; returns
    float32 logits for that position. Mutates `cache` (appends one entry)."""
    return model(input_ids=token, past_key_values=cache, use_cache=True,
                 position_ids=torch.tensor([pos], device=token.device).unsqueeze(0)).logits[:, -1, :].float()


def _prefill(model, ids):
    """Fresh DynamicCache holding ids[0 : len]."""
    cache = DynamicCache()
    model(input_ids=ids, past_key_values=cache, use_cache=True,
          position_ids=torch.arange(0, ids.shape[1], device=ids.device).unsqueeze(0))
    return cache


def _selftest_cache_roundtrip(model, sample_ids, grow_by=K_DRAFT):
    """Isolated correctness check for _crop_cache, independent of the
    generation loop. Returns a dict of measured numbers (stored in _meta).

    REDESIGNED 2026-09-21 -- the previous version of this test was wrong,
    not the cache. It compared:
        (a) a cache built by one N-token forward pass, against
        (b) a cache built by one 2N-token forward pass, then cropped to N.
    Positions 0..N-1 are mathematically identical in both, but in bf16 they
    are NOT bit-identical: a 24-token and a 48-token forward pass run
    different-shaped matrix multiplications, which add up the same numbers
    in a different order, which rounds differently. That rounding compounds
    over 32 layers into logit differences of ~0.5 on the 8B model. Proven
    with a control that involves no cache at all (control B below): reading
    position N-1 from an N-token forward vs a 2N-token forward shows the
    same size of difference. So the old test was measuring bf16 rounding,
    not crop. transformers 5.17.0's DynamicLayer.crop() was also read
    directly: it physically slices the key/value tensors, and
    get_seq_length() returns that physical size, so there is no separate
    "logical length" that could fall out of sync.

    The fair test: build BOTH caches with the identical N-token prefill (so
    positions 0..N-1 are bit-identical by construction), then grow one of
    them and crop it back. Any difference is then caused by grow+crop and
    nothing else. Done twice, once per growth pattern the real loop uses:
      - multi-token growth (the target's verify pass: K tokens in one call)
      - single-token growth (the scout's draft loop: K separate calls)
    Pass condition: difference <= a determinism control (two identical
    caches, same probe), which is expected to be exactly 0.
    """
    N = sample_ids.shape[1] // 2
    G = min(grow_by, sample_ids.shape[1] - N - 1)
    if N < 1 or G < 1:
        raise RuntimeError("Sample prompt too short for the cache self-test.")
    probe = sample_ids[:, N:N + 1]

    with torch.inference_mode():
        # D) determinism control: two identically-built caches, same probe
        ref = _probe(model, _prefill(model, sample_ids[:, :N]), probe, N)
        again = _probe(model, _prefill(model, sample_ids[:, :N]), probe, N)
        det_diff = (ref - again).abs().max().item()

        # P) guard: does this transformers version actually READ position_ids?
        #    Same cache, same token, deliberately wrong position -> the logits
        #    must change. 5.17.0 silently ignores cache_position=; if a future
        #    version did the same to position_ids, every position this file
        #    passes would be decoration, and this catches that.
        wrong_pos = _probe(model, _prefill(model, sample_ids[:, :N]), probe, N + 7)
        pos_effect = (ref - wrong_pos).abs().max().item()
        if pos_effect <= det_diff:
            raise RuntimeError(
                "[selftest] position_ids appear to be IGNORED by this transformers "
                f"version ({transformers.__version__}): feeding a token at the wrong "
                "position gave identical logits. This file's position bookkeeping "
                "would have no effect. Fix before running anything else.")

        # C1) grow by G tokens in ONE forward (target verify pattern), crop back
        c = _prefill(model, sample_ids[:, :N])
        model(input_ids=sample_ids[:, N:N + G], past_key_values=c, use_cache=True,
              position_ids=torch.arange(N, N + G, device=sample_ids.device).unsqueeze(0))
        _crop_cache(c, N)
        len_multi = _cache_seq_length(c)
        multi_diff = (ref - _probe(model, c, probe, N)).abs().max().item()

        # C2) grow by G tokens one at a time (scout draft pattern), crop back
        c = _prefill(model, sample_ids[:, :N])
        for j in range(G):
            _probe(model, c, sample_ids[:, N + j:N + j + 1], N + j)
        _crop_cache(c, N)
        len_single = _cache_seq_length(c)
        single_diff = (ref - _probe(model, c, probe, N)).abs().max().item()

        # B) bf16 rounding noise floor -- no cache involved. Logits for
        #    position N-1 computed by an N-token pass vs a 2N-token pass.
        #    Informational only (not pass/fail); also used by the fidelity
        #    gate to judge whether a divergence is a near-tie.
        l_short = model(input_ids=sample_ids[:, :N]).logits[:, N - 1, :].float()
        l_long = model(input_ids=sample_ids[:, :2 * N]).logits[:, N - 1, :].float()
        noise_floor = (l_short - l_long).abs().max().item()

    result = {"N": N, "grow_by": G, "determinism_diff": det_diff,
              "wrong_position_effect": pos_effect,
              "crop_after_multi_token_growth_diff": multi_diff,
              "crop_after_single_token_growth_diff": single_diff,
              "bf16_prefill_length_noise_floor": noise_floor}

    if len_multi != N or len_single != N:
        raise RuntimeError(
            f"[selftest] Cache-crop FAILED on length: expected {N}, got "
            f"{len_multi} (multi-token growth) / {len_single} (single-token growth).")
    if multi_diff > det_diff or single_diff > det_diff:
        raise RuntimeError(
            f"[selftest] Cache-crop FAILED: growing a cache and cropping it back "
            f"changed its output. max abs logit diff {multi_diff:.4g} (multi-token "
            f"growth), {single_diff:.4g} (single-token growth); determinism control "
            f"{det_diff:.4g}. Both caches were built by the identical prefill, so "
            f"this cannot be bf16 prefill-shape rounding -- it is a real crop "
            f"problem. Do not trust any timing number from this script.")

    print(f"[selftest] Cache-crop OK: grow-by-{G}-then-crop is identical to never "
          f"growing (diff {multi_diff:.4g} multi-token, {single_diff:.4g} single-token; "
          f"determinism control {det_diff:.4g}).")
    print(f"[selftest] Info: bf16 rounding noise floor (N-token vs 2N-token prefill, "
          f"no cache) = {noise_floor:.4g} logits. This is the size of difference "
          f"the old self-test was flagging.")
    return result


# ============================================================================
# KV-cached generation (mirrors bench_common.py's uncached functions)
# ============================================================================

def target_only_generate_cached(model, input_ids, max_tokens, eos=None):
    """KV-cached greedy target-only decode. Same greedy policy as
    bench_common.target_only_generate (identical output, given exact
    floating-point equivalence between the cached and uncached forward code
    paths -- checked directly by the fidelity gate in main(), not assumed),
    but O(1) forward-pass cost per token instead of O(n) full-context
    recompute -- the realistic-deployment baseline benchmark_ablation.py
    deliberately does not use."""
    # POSITIONS: passed as position_ids, NOT cache_position. Verified against
    # transformers 5.17.0's modeling_llama.py (2026-09-21): LlamaModel.forward
    # has no cache_position parameter at all -- a cache_position= keyword is
    # silently swallowed by **kwargs and ignored, and positions are derived
    # from cache.get_seq_length() instead. This file previously passed
    # cache_position everywhere, believing it was pinning positions; it never
    # was. (That is also why "pin cache_position" had zero effect during the
    # 2026-09-19 debugging.) position_ids IS read, so our own counters now
    # genuinely control positions, and a counter bug can actually show up.
    generated = []
    device = input_ids.device
    pos = input_ids.shape[1]  # next position to assign, tracked explicitly
    with torch.inference_mode():
        cache = DynamicCache()
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                     position_ids=torch.arange(0, pos, device=device).unsqueeze(0))
        cache = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(max_tokens):
            token_id = next_token.item()
            generated.append(token_id)
            if eos and token_id in eos:
                break
            _check_pos(cache, pos, "baseline decode step")
            with _time_phase("baseline_step"):
                out = model(input_ids=next_token, past_key_values=cache, use_cache=True,
                            position_ids=torch.tensor([pos], device=device).unsqueeze(0))
                cache = out.past_key_values
                pos += 1
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    return {"generated_ids": generated}


def speculative_generate_cached(target_model, scout_model, input_ids, K, max_tokens, eos=None):
    """KV-cached lossless scout(draft)->target(verify) speculative decoding,
    fixed K. Same accept/reject policy as bench_common.speculative_generate
    (greedy on both sides, exact/lossless), reimplemented with incremental
    KV-caching on both models instead of a full-context recompute every
    cycle.

    Cache handling follows the crop-after-mismatch approach HF's own
    built-in assisted-generation code path uses: draft K tokens into the
    scout's cache eagerly, verify all K in one target forward pass, then
    crop BOTH caches back to however many of the K drafts were actually
    accepted, before priming the next cycle with exactly one extra token
    (the target's own replacement on a reject, or the free bonus token on a
    full accept).

    Do not trust this function's output on a new machine until
    _selftest_cache_roundtrip() has passed AND the cross-implementation
    fidelity gate in main() (this function vs. bench_common's uncached
    version, same prompt) has passed.
    """
    generated = []
    total_drafted = 0
    total_accepted = 0
    cycle_count = 0
    device = input_ids.device

    with torch.inference_mode():
        scout_cache = DynamicCache()
        target_cache = DynamicCache()

        prompt_len = input_ids.shape[1]
        prime_pos = torch.arange(0, prompt_len, device=device)

        scout_prime = scout_model(input_ids=input_ids, past_key_values=scout_cache, use_cache=True,
                                   position_ids=prime_pos.unsqueeze(0))
        scout_cache = scout_prime.past_key_values
        draft_logits = scout_prime.logits[:, -1, :]

        target_prime = target_model(input_ids=input_ids, past_key_values=target_cache, use_cache=True,
                                     position_ids=prime_pos.unsqueeze(0))
        target_cache = target_prime.past_key_values
        check_logits0 = target_prime.logits[:, -1, :]

        prefix_len = prompt_len

        while len(generated) < max_tokens:
            cycle_count += 1

            # -- Draft phase: K single-token scout steps, cache grows by K --
            # position for step k is prefix_len + k -- our own tracked
            # position, passed as position_ids (see note in
            # target_only_generate_cached for why not cache_position).
            draft_tokens = []
            step_logits = draft_logits
            with _time_phase("draft"):
                for k in range(K):
                    next_draft_tok = torch.argmax(step_logits, dim=-1, keepdim=True)
                    draft_tokens.append(next_draft_tok)
                    step_pos = torch.tensor([prefix_len + k], device=device)
                    _check_pos(scout_cache, prefix_len + k, f"scout draft step {k}")
                    scout_step = scout_model(input_ids=next_draft_tok, past_key_values=scout_cache, use_cache=True,
                                              position_ids=step_pos.unsqueeze(0))
                    scout_cache = scout_step.past_key_values
                    step_logits = scout_step.logits[:, -1, :]
            total_drafted += K
            draft_tensor = torch.cat(draft_tokens, dim=-1)  # (1, K)

            # -- Verify phase: one target forward over all K draft tokens --
            # positions prefix_len .. prefix_len+K-1, same positions the
            # scout drafted them at.
            verify_pos = torch.arange(prefix_len, prefix_len + K, device=device)
            _check_pos(target_cache, prefix_len, "target verify pass")
            with _time_phase("verify"):
                target_step = target_model(input_ids=draft_tensor, past_key_values=target_cache, use_cache=True,
                                            position_ids=verify_pos.unsqueeze(0))
                target_cache = target_step.past_key_values  # now holds prefix_len + K
                verify_logits = target_step.logits  # (1, K, vocab)

            # verify_logits[:, i, :] predicts the token AFTER draft_tokens[i]
            # (position i was fed as the (i+1)-th new token this call, with
            # causal attention over cache(prefix) + draft_tokens[0:i+1]).
            # check_logits0 (from priming, or from the previous cycle's
            # end-of-cycle extra token) plays the same role for i=0.
            accepted_in_cycle = 0
            mismatch = False
            replacement_token = None
            with _time_phase("accept_check"):
                for i in range(K):
                    check_i = check_logits0 if i == 0 else verify_logits[:, i - 1, :]
                    expected_token = torch.argmax(check_i, dim=-1, keepdim=True)
                    if expected_token.item() == draft_tokens[i].item():
                        accepted_in_cycle += 1
                    else:
                        mismatch = True
                        replacement_token = expected_token
                        break
            total_accepted += accepted_in_cycle

            with _time_phase("bookkeeping"):
                if mismatch:
                    new_last_token = replacement_token
                else:
                    new_last_token = torch.argmax(verify_logits[:, K - 1, :], dim=-1, keepdim=True)

                new_ids_list = [t.item() for t in draft_tokens[:accepted_in_cycle]] + [new_last_token.item()]
                generated.extend(new_ids_list)

            # -- Crop both caches to the accepted prefix, then prime the one
            #    new token so both models are ready for the next cycle. --
            crop_len = prefix_len + accepted_in_cycle
            with _time_phase("crop"):
                _crop_cache(target_cache, crop_len)
                _crop_cache(scout_cache, crop_len)

            # new_last_token occupies position crop_len -- our own tracked
            # value, not whatever the freshly-cropped cache might infer.
            with _time_phase("position_setup"):
                new_pos = torch.tensor([crop_len], device=device)
                _check_pos(target_cache, int(new_pos), "target, token after crop")
                _check_pos(scout_cache, int(new_pos), "scout, token after crop")
            with _time_phase("prime"):
                target_next = target_model(input_ids=new_last_token, past_key_values=target_cache, use_cache=True,
                                            position_ids=new_pos.unsqueeze(0))
                target_cache = target_next.past_key_values
                check_logits0 = target_next.logits[:, -1, :]

                scout_next = scout_model(input_ids=new_last_token, past_key_values=scout_cache, use_cache=True,
                                          position_ids=new_pos.unsqueeze(0))
                scout_cache = scout_next.past_key_values
                draft_logits = scout_next.logits[:, -1, :]

            prefix_len = crop_len + 1

            if eos and any(t in eos for t in new_ids_list):
                break

    return {
        "generated_ids": generated[:max_tokens],
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "cycles": cycle_count,
    }


# ============================================================================
# Cross-implementation fidelity gate (cached vs. bench_common's uncached)
# ============================================================================

# A divergence counts as a bf16 near-tie only if BOTH tokens the two runs
# picked score within this many noise floors of the target model's own best
# choice at that point (recomputed independently, uncached).
NEAR_TIE_NOISE_MULTIPLIER = 2.0
MAX_NEAR_TIES_PER_CHECK = 10


def _classify_divergence(target_model, context_ids, tok_a, tok_b, tie_tol):
    """At a point where two runs picked different tokens (tok_a, tok_b) after
    the same context, recompute the target model's scores there (plain
    uncached forward pass) and check how far each picked token is below the
    best-scoring token.

    Why this distinguishes rounding from bugs: rounding can only change which
    token wins when several candidates score almost the same, so BOTH picked
    tokens will be within rounding distance of the best. (Sometimes three or
    more candidates are that close -- seen in testing -- which is why this
    checks distance-from-best rather than "are they exactly the top two".)
    A cache bookkeeping bug (stale or misaligned key/values) changes what the
    model is looking at, so at least one run picks a token that the correct
    computation scores clearly lower."""
    with torch.inference_mode():
        logits = target_model(context_ids).logits[0, -1, :].float()
    best_id = int(torch.argmax(logits))
    best = float(logits[best_id])
    below_a = best - float(logits[tok_a])
    below_b = best - float(logits[tok_b])
    return {"best_id": best_id, "cached_below_best": below_a,
            "uncached_below_best": below_b, "tie_tol": tie_tol,
            "near_tie": bool(below_a <= tie_tol and below_b <= tie_tol)}


def _compare_with_resync(name, run_a, run_b, target_model, ids, n_tokens, tie_tol):
    """Compare two generators token-for-token over n_tokens. On a divergence:
    if it is a bf16 near-tie (see _classify_divergence), record it, resync
    both runs onto the shared prefix plus the reference's top choice, and
    keep comparing -- so the whole length is still checked, not just the
    part before the first tie. If it is NOT a near-tie, return failure.
    run_a / run_b: fn(ids, n) -> dict with generated_ids (and optionally
    total_drafted / total_accepted, summed for coverage reporting)."""
    ctx = ids.clone()
    checked, ties, drafted, accepted = 0, [], 0, 0
    drafted_b, accepted_b = 0, 0
    while checked < n_tokens:
        remaining = n_tokens - checked
        ra, rb = run_a(ctx.clone(), remaining), run_b(ctx.clone(), remaining)
        a, b = ra["generated_ids"], rb["generated_ids"]
        drafted += ra.get("total_drafted", 0)
        accepted += ra.get("total_accepted", 0)
        drafted_b += rb.get("total_drafted", 0)
        accepted_b += rb.get("total_accepted", 0)
        if a == b:
            return {"ok": True, "near_ties": ties, "drafted": drafted, "accepted": accepted,
                    "drafted_b": drafted_b, "accepted_b": accepted_b}
        n = min(len(a), len(b))
        d = next((i for i in range(n) if a[i] != b[i]), n)
        if d == n:  # one stopped early (EOS) where the other didn't
            return {"ok": False, "reason": f"length mismatch {len(a)} vs {len(b)}",
                    "a": a, "b": b, "index": checked + d, "near_ties": ties}
        prefix = torch.tensor([a[:d]], device=ids.device, dtype=ids.dtype)
        context = torch.cat([ctx, prefix], dim=-1)
        cls = _classify_divergence(target_model, context, a[d], b[d], tie_tol)
        cls.update({"index": checked + d, "cached_tok": a[d], "uncached_tok": b[d]})
        if not cls["near_tie"] or len(ties) >= MAX_NEAR_TIES_PER_CHECK:
            return {"ok": False, "reason": "not a near-tie" if not cls["near_tie"]
                    else f"more than {MAX_NEAR_TIES_PER_CHECK} near-ties",
                    "a": a, "b": b, "index": checked + d, "divergence": cls,
                    "near_ties": ties}
        ties.append(cls)
        chosen = torch.tensor([[cls["best_id"]]], device=ids.device, dtype=ids.dtype)
        ctx = torch.cat([context, chosen], dim=-1)
        checked += d + 1
    return {"ok": True, "near_ties": ties, "drafted": drafted, "accepted": accepted,
            "drafted_b": drafted_b, "accepted_b": accepted_b}


def _fidelity_gate(target_model, scout_model, encoded, eos, gate_tokens, noise_floor):
    """Runs BEFORE any timed trial. Compares this file's cached
    implementation against bench_common.py's existing, already-verified
    UNCACHED implementation, on every prompt.

    REVISED 2026-09-21. The previous version exited on ANY token difference.
    That is too strict in bf16: cached and uncached decoding compute the same
    maths with different-shaped matrix operations, so they round differently,
    and at positions where the model's top two choices are almost tied the
    rounding can flip which one wins (after which the two texts legitimately
    go their separate ways). This was confirmed on a small test model: in
    float32 the cached and uncached code matched exactly across every accept/
    reject pattern; in bf16 the SAME code diverged on 5 of 6 prompts.

    So each divergence is now classified (see _classify_divergence): a
    near-tie is recorded and the check continues past it; anything else is
    treated as a real bug and the script exits. Every near-tie is printed
    and saved to the results JSON -- nothing is hidden.
    """
    tie_tol = NEAR_TIE_NOISE_MULTIPLIER * noise_floor
    print("\n[gate] Cross-implementation fidelity check "
          f"(cached vs. uncached, {gate_tokens} tokens; near-tie tolerance "
          f"{tie_tol:.4g} logits = {NEAR_TIE_NOISE_MULTIPLIER:g} x measured noise floor)...")
    report = {"tie_tol": tie_tol, "gate_tokens": gate_tokens, "prompts": {}}
    for label, ids in encoded:
        checks = {
            "baseline": _compare_with_resync(
                "baseline",
                lambda x, n: target_only_generate_cached(target_model, x, n, eos=eos),
                lambda x, n: bench_common.target_only_generate(target_model, x, n, eos=eos),
                target_model, ids, gate_tokens, tie_tol),
            "speculative": _compare_with_resync(
                "speculative",
                lambda x, n: speculative_generate_cached(target_model, scout_model, x, K_DRAFT, n, eos=eos),
                lambda x, n: bench_common.speculative_generate(target_model, scout_model, x, K_DRAFT, n, eos=eos),
                target_model, ids, gate_tokens, tie_tol),
        }
        for cond, r in checks.items():
            if not r["ok"]:
                _report_mismatch(label, cond, r)
                sys.exit(1)
        sp = checks["speculative"]
        drafted, accepted = sp["drafted"], sp["accepted"]
        rate = (accepted / drafted * 100.0) if drafted else 0.0
        coverage_note = "" if 0 < accepted < drafted else \
            "  [!] only one of accept/reject seen at this length -- not full branch coverage"
        tie_str = ", ".join(f"{c}: {len(r['near_ties'])} near-tie(s)" for c, r in checks.items())
        rate_u = (sp["accepted_b"] / sp["drafted_b"] * 100.0) if sp["drafted_b"] else 0.0
        print(f"  {label}: baseline OK, speculative OK ({tie_str}){coverage_note}")
        # Output fidelity cannot see a SCOUT-side cache bug: the target
        # re-checks every token, so a broken scout only lowers the accept
        # rate (and therefore speed), never changes the output. Cached vs.
        # uncached accept rates are printed side by side for that reason --
        # they should be close (not necessarily identical after a near-tie
        # resync). A clearly lower cached rate means a scout cache problem.
        print(f"      accept rate: cached {accepted}/{drafted} = {rate:.1f}%   "
              f"uncached {sp['accepted_b']}/{sp['drafted_b']} = {rate_u:.1f}%")
        for c, r in checks.items():
            for t in r["near_ties"]:
                print(f"      {c} near-tie at token {t['index']}: tokens {t['cached_tok']} vs "
                      f"{t['uncached_tok']}, {t['cached_below_best']:.3g} / {t['uncached_below_best']:.3g} "
                      f"logits below the model's best choice")
        report["prompts"][label] = {c: {"near_ties": r["near_ties"]} for c, r in checks.items()}
        report["prompts"][label]["speculative"].update({
            "cached_drafted": drafted, "cached_accepted": accepted,
            "uncached_drafted": sp["drafted_b"], "uncached_accepted": sp["accepted_b"]})

    print("[gate] All prompts passed (exact match, or bf16 near-ties only). "
          "Proceeding to the real run.\n")
    return report


def _report_mismatch(label, condition, r):
    print(f"\n[gate] MISMATCH on {label} ({condition}) -- {r['reason']}. Cached and "
          f"uncached implementations disagree in a way bf16 rounding does not "
          f"explain. Do not trust any timing number from this script.")
    a, b = r["a"], r["b"]
    n = min(len(a), len(b))
    local = next((i for i in range(n) if a[i] != b[i]), n)
    lo, hi = max(0, local - 3), local + 4
    print(f"  first divergence at token index {r['index']}")
    print(f"  cached  : {a[lo:hi]}")
    print(f"  uncached: {b[lo:hi]}")
    if "divergence" in r:
        dv = r["divergence"]
        print(f"  at that point, the cached token scores {dv['cached_below_best']:.4g} and the "
              f"uncached token {dv['uncached_below_best']:.4g} logits below the model's best "
              f"choice (near-tie tolerance {dv['tie_tol']:.4g})")
    if r["near_ties"]:
        print(f"  ({len(r['near_ties'])} earlier near-tie(s) had been passed before this)")


# ============================================================================
# Timed trial (either condition)
# ============================================================================

def run_condition_trial(condition, target_model, scout_model, tokenizer, monitor,
                         input_ids, prompt_label, round_idx, max_tokens, eos,
                         warmup_steps=WARMUP_STEPS, K=K_DRAFT, global_index=None,
                         record=True):
    """Run one timed trial of either 'baseline' or 'speculative', cached.

    Per-trial warmup runs a short, separate, throwaway cached generation
    (its own fresh cache) immediately before timing -- same purpose as
    benchmark_ablation.py's per-trial warmup (avoid a cold-SM effect on this
    specific trial), just using the cached code path so the warmed kernels
    match what's actually being timed. This throwaway cache never touches
    the real generation below, which starts fresh from input_ids.
    """
    with torch.inference_mode():
        if condition == "baseline":
            target_only_generate_cached(target_model, input_ids.clone(), warmup_steps, eos=None)
        else:
            speculative_generate_cached(target_model, scout_model, input_ids.clone(), K, warmup_steps, eos=None)
    torch.cuda.synchronize()

    total_drafted = total_accepted = 0
    e_start = monitor.read_energy_j()
    t_start = time.perf_counter()

    if condition == "baseline":
        result = target_only_generate_cached(target_model, input_ids, max_tokens, eos=eos)
    else:
        result = speculative_generate_cached(target_model, scout_model, input_ids, K, max_tokens, eos=eos)
        total_drafted = result["total_drafted"]
        total_accepted = result["total_accepted"]

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    e_end = monitor.read_energy_j()

    stats = monitor.window_stats(t_start, t_end)
    latency = t_end - t_start
    tokens = len(result["generated_ids"])

    energy_counter = (e_end - e_start) if (e_start is not None and e_end is not None) else None
    energy_sampled = stats["energy_j_sampled"]
    energy = energy_counter if energy_counter is not None else energy_sampled
    energy_source = "nvml_counter" if energy_counter is not None else "sampled_trapezoid"

    accept_rate = (total_accepted / total_drafted * 100.0) if total_drafted > 0 else None

    entry = {
        "timestamp": datetime.now().isoformat(),
        "condition": BASELINE_KEY if condition == "baseline" else SPEC_KEY,
        "prompt_label": prompt_label,
        "round": round_idx,
        "global_index": global_index,
        "tokens": tokens,
        "prompt_tokens": int(input_ids.shape[-1]),
        "latency_sec": round(latency, 6),
        "throughput_tok_sec": round(tokens / latency, 2) if latency > 0 else 0.0,
        "avg_power_watts": round(stats["mean_w"], 2) if stats["mean_w"] is not None else None,
        "min_power_watts": round(stats["min_w"], 2) if stats["min_w"] is not None else None,
        "max_power_watts": round(stats["max_w"], 2) if stats["max_w"] is not None else None,
        "total_energy_joules": round(energy, 4) if energy is not None else None,
        "energy_source": energy_source,
        "energy_j_counter": round(energy_counter, 4) if energy_counter is not None else None,
        "energy_j_sampled": round(energy_sampled, 4) if energy_sampled is not None else None,
        "joules_per_token": round(energy / tokens, 6) if (energy is not None and tokens) else None,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "accept_rate_pct": round(accept_rate, 2) if accept_rate is not None else None,
        "temp_start_c": stats["temp_start_c"],
        "temp_end_c": stats["temp_end_c"],
        "temp_max_c": stats["temp_max_c"],
        "sm_clock_mean_mhz": round(stats["sm_clock_mean_mhz"], 1) if stats["sm_clock_mean_mhz"] else None,
        "mem_clock_mean_mhz": round(stats["mem_clock_mean_mhz"], 1) if stats["mem_clock_mean_mhz"] else None,
        "throttle_reasons": monitor.throttle_reasons(),
        "n_power_samples": stats["n_power_samples"],
        "effective_sample_hz": round(stats["effective_sample_hz"], 1),
        "generated_ids": result["generated_ids"],  # only used for fidelity check; not written to CSV
    }

    if record:
        csv_entry = {k: v for k, v in entry.items() if k != "generated_ids"}
        bench_common.safe_append_csv(TRIAL_CSV, csv_entry)

    return entry


# ============================================================================
# Suite
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=NUM_TRIALS)
    ap.add_argument("--tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--min-warmup", type=float, default=bench_common.MIN_WARMUP_SEC)
    ap.add_argument("--max-warmup", type=float, default=bench_common.MAX_WARMUP_SEC)
    ap.add_argument("--fidelity-gate-tokens", type=int, default=60,
                     help="Token count for the cached-vs-uncached fidelity gate that "
                          "runs before any timed trial. Short by default to keep the "
                          "gate fast; long enough at this repo's known accept rates "
                          "(42-85%%) to plausibly exercise both an accept and a reject "
                          "cycle, but this is NOT a formal coverage guarantee -- read "
                          "the printed accept/drafted counts, don't just check for a "
                          "'gate passed' message.")
    ap.add_argument("--skip-selftest", action="store_true",
                     help="Skip both cache self-tests. Do not use this for a real "
                          "measurement run -- only for fast iteration while actively "
                          "debugging this script itself.")
    ap.add_argument("--lock-clocks", action="store_true",
                     help="Attempt NVML clock locking (needs root/elevated privileges -- "
                          "never succeeds under WSL2 without it, which is why this "
                          "defaults to off, matching benchmark_ablation.py).")
    ap.add_argument("--attn-implementation", type=str, default=None,
                     choices=["eager", "sdpa", "flash_attention_2"],
                     help="Force a specific attention backend on BOTH models (target and "
                          "scout). Default: don't pass anything, let transformers pick its "
                          "own default. Recorded in the results JSON either way.")
    ap.add_argument("--profile-phases", type=int, default=0,
                     help="DIAGNOSTIC ONLY, not a real measurement. Run roughly this many "
                          "speculative cycles (about x(K+1) tokens) on the first prompt "
                          "with per-phase wall-clock timing (draft / verify / accept_check "
                          "/ crop / prime), print a breakdown against an equal-length cached "
                          "baseline run, then exit -- no self-test, no gate, no real trials, "
                          "nothing written to telemetry. Added 2026-09-21 (evening): real GPU runs on "
                          "all three attn backends showed the cached SPECULATIVE condition "
                          "consistently SLOWER than the cached BASELINE despite normal "
                          "accept rates (36-85%%) -- this flag exists to find out where the "
                          "time actually goes instead of guessing again. See K5_status.md.")
    args = ap.parse_args()

    num_trials = args.trials
    max_tokens = args.tokens

    monitor = bench_common.NVMLPowerMonitor(device_index=0)
    dev = monitor.device_info()
    print("=" * 85)
    print(f"[*] KV-CACHED ABLATION BENCHMARK ({dev.get('name')})")
    print(f"[*] Target: {TARGET_MODEL_ID} | Scout: {SCOUT_MODEL_ID} | Trials: N={num_trials}")
    print(f"[*] torch {torch.__version__} | transformers {transformers.__version__}")
    print(f"[*] Power limit: {dev.get('power_limit_w')} W   "
          f"energy counter: {'yes' if dev['energy_counter_supported'] else 'NO (falling back to sampling)'}")
    print("=" * 85)

    if args.lock_clocks:
        monitor.lock_clocks()
    monitor.start()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = bench_common.eos_ids(tokenizer)

    # dtype= is the current spelling (transformers 5.x, the installed version
    # as of 2026-09-21 is 5.17.0); torch_dtype= is its deprecated older alias.
    # If this machine is ever moved back to transformers <4.56, change this
    # to torch_dtype=. The installed version is printed below and saved in
    # the results JSON's _meta so this is never a guess.
    model_kwargs = dict(dtype=torch.bfloat16, device_map=DEVICE)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print("\n[*] Loading Target Model (8B)...")
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, **model_kwargs)
    target_model.eval()
    print(f"[*] Target attn_implementation: {target_model.config._attn_implementation}")

    print("[*] Loading Scout Model (1B)...")
    scout_model = AutoModelForCausalLM.from_pretrained(SCOUT_MODEL_ID, **model_kwargs)
    scout_model.eval()
    print(f"[*] Scout attn_implementation: {scout_model.config._attn_implementation}")

    encoded = [(label, bench_common.encode_prompt(tokenizer, text, device=DEVICE))
               for label, text in zip(PROMPT_LABELS, PROMPTS)]
    for label, ids in encoded:
        print(f"[*] {label}: {ids.shape[-1]} prompt tokens")

    if args.profile_phases > 0:
        print(f"\n[profile] Phase-timing probe: ~{args.profile_phases} speculative cycles "
              f"on '{encoded[0][0]}' only. DIAGNOSTIC ONLY -- no self-test, no gate, no "
              f"telemetry written. Every phase below does its own torch.cuda.synchronize() "
              f"to get an honest reading, so this run itself is slower than a real timed "
              f"trial -- only the RELATIVE size of each bucket, and the spec-vs-baseline "
              f"tok/s ratio, are meaningful here.")
        global _PHASE_TIMING
        label0, ids0 = encoded[0]
        n_probe_tokens = args.profile_phases * (K_DRAFT + 1)

        _PHASE_TIMING = True
        _phase_times.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            spec_result = speculative_generate_cached(target_model, scout_model, ids0.clone(),
                                                        K_DRAFT, n_probe_tokens, eos=None)
        torch.cuda.synchronize()
        spec_wall = time.perf_counter() - t0
        spec_times = dict(_phase_times)

        _phase_times.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            base_result = target_only_generate_cached(target_model, ids0.clone(), n_probe_tokens, eos=None)
        torch.cuda.synchronize()
        base_wall = time.perf_counter() - t0
        base_times = dict(_phase_times)
        _PHASE_TIMING = False

        n_spec_tok = len(spec_result["generated_ids"])
        n_base_tok = len(base_result["generated_ids"])
        print(f"\n[profile] Speculative: {spec_result['cycles']} cycles, {n_spec_tok} tokens, "
              f"{spec_wall:.3f}s wall ({n_spec_tok / spec_wall:.1f} tok/s, probe conditions "
              f"only -- not comparable to a real trial's tok/s)")
        for name, t in sorted(spec_times.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<14} {t:.3f}s  ({t / spec_wall * 100:5.1f}% of wall)")
        bucketed = sum(spec_times.values())
        print(f"    {'[unbucketed]':<14} {spec_wall - bucketed:.3f}s  "
              f"({(spec_wall - bucketed) / spec_wall * 100:5.1f}% of wall -- Python control "
              f"flow between phases: the accept/reject branch, tensor bookkeeping, list "
              f"appends -- not captured by any named bucket)")

        print(f"\n[profile] Baseline: {n_base_tok} tokens, {base_wall:.3f}s wall "
              f"({n_base_tok / base_wall:.1f} tok/s, probe conditions only)")
        for name, t in sorted(base_times.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<14} {t:.3f}s  ({t / base_wall * 100:5.1f}% of wall)")

        print(f"\n[profile] Per-output-token wall time: speculative "
              f"{spec_wall / n_spec_tok * 1000:.2f} ms/tok vs baseline "
              f"{base_wall / n_base_tok * 1000:.2f} ms/tok "
              f"({'SLOWER' if spec_wall / n_spec_tok > base_wall / n_base_tok else 'faster'} "
              f"per token if this ratio matches the earlier real-run finding).")
        print("\n[profile] Exiting -- this flag never runs the self-test, gate, or real "
              "trials. Re-run without --profile-phases for an actual measurement.")
        monitor.stop()
        monitor.close()
        return

    selftest_result = gate_report = None
    if args.skip_selftest:
        print("\n[!] --skip-selftest set: SKIPPING both cache self-tests. "
              "Do not trust any number from this run.")
    else:
        print()
        global _POSITION_CHECKS
        _POSITION_CHECKS = True
        selftest_result = _selftest_cache_roundtrip(target_model, encoded[0][1])
        gate_report = _fidelity_gate(target_model, scout_model, encoded, eos,
                                     args.fidelity_gate_tokens,
                                     selftest_result["bf16_prefill_length_noise_floor"])
        _POSITION_CHECKS = False
        print("[gate] Position bookkeeping checks passed on every forward pass of the "
              "gate; now switched off for the timed run.")

    try:
        # -- Closed-loop thermal warmup, alternating baseline/speculative work --
        def warmup_step(i):
            label, ids = encoded[i % len(encoded)]
            if i % 2 == 0:
                target_only_generate_cached(target_model, ids.clone(), WARMUP_UNIT_TOKENS, eos=eos)
                return f"{label}/base"
            else:
                speculative_generate_cached(target_model, scout_model, ids.clone(),
                                             K_DRAFT, WARMUP_UNIT_TOKENS, eos=eos)
                return f"{label}/spec"

        warmup = bench_common.warm_to_steady_state(
            monitor, warmup_step, min_sec=args.min_warmup, max_sec=args.max_warmup,
        )

        # -- TEST 1: Exact Lossless Fidelity (cached baseline vs. cached speculative) --
        print("\n[1/2] Verifying Exact Lossless Fidelity (Baseline vs. Speculative, both cached)...")
        fidelity_by_prompt = {}
        for p_idx, (label, ids) in enumerate(encoded, 1):
            base = run_condition_trial("baseline", target_model, scout_model, tokenizer, monitor,
                                        ids, f"{label}_fidelity", 0, max_tokens, eos,
                                        global_index=None, record=False)
            spec = run_condition_trial("speculative", target_model, scout_model, tokenizer, monitor,
                                        ids, f"{label}_fidelity", 0, max_tokens, eos,
                                        global_index=None, record=False)
            base_tok, spec_tok = base["generated_ids"], spec["generated_ids"]
            match_count = sum(1 for a, b in zip(base_tok, spec_tok) if a == b)
            total_tok = max(len(base_tok), len(spec_tok))
            match_pct = (match_count / total_tok) * 100.0 if total_tok > 0 else 0.0
            fidelity_by_prompt[f"prompt_{p_idx}"] = match_pct
            print(f"  Prompt {p_idx} ({label}): {match_pct:.1f}% exact token match ({len(base_tok)} tokens)")

        agg_fidelity = float(np.mean(list(fidelity_by_prompt.values())))
        print(f"[*] Aggregate Fidelity Match: {agg_fidelity:.2f}%")

        # -- TEST 2: Multi-trial, fully interleaved --------------------------
        print(f"\n[2/2] Running Multi-Trial Performance Benchmark (interleaved order)...")
        by_prompt_condition = {label: {"baseline": [], "speculative": []} for label, _ in encoded}
        chronological = []
        gidx = 0

        for round_idx in range(1, num_trials + 1):
            print(f"\n{'#' * 85}\n# TRIAL ROUND {round_idx}/{num_trials}\n{'#' * 85}")
            order = ["baseline", "speculative"] if round_idx % 2 == 1 else ["speculative", "baseline"]
            for label, ids in encoded:
                for condition in order:
                    gidx += 1
                    entry = run_condition_trial(
                        condition, target_model, scout_model, tokenizer, monitor,
                        ids, label, round_idx, max_tokens, eos, global_index=gidx,
                    )
                    acc_str = f"  accept={entry['accept_rate_pct']:.1f}%" if entry["accept_rate_pct"] is not None else ""
                    print(f"  {label:<8} {entry['condition']:<28} round {round_idx}/{num_trials}  "
                          f"tok/s={entry['throughput_tok_sec']:<7} "
                          f"J/tok={entry['joules_per_token']}  "
                          f"P={entry['avg_power_watts']} W  T={entry['temp_end_c']} C{acc_str}")
                    by_prompt_condition[label]["baseline" if condition == "baseline" else "speculative"].append(entry)
                    chronological.append(entry)
    finally:
        monitor.stop()
        monitor.unlock_clocks()

    # -- aggregate, matching benchmark_ablation.py's ablation_results.json shape --
    ablation = {}
    for p_idx, (label, _) in enumerate(encoded, 1):
        p_key = f"prompt_{p_idx}"
        ablation[p_key] = {}

        base_trials = by_prompt_condition[label]["baseline"]
        tps_b = [t["throughput_tok_sec"] for t in base_trials]
        j_b = [t["joules_per_token"] for t in base_trials if t["joules_per_token"] is not None]
        pwr_b = [t["avg_power_watts"] for t in base_trials if t["avg_power_watts"] is not None]
        ablation[p_key][BASELINE_KEY] = {
            "tps_mean": float(np.mean(tps_b)),
            "tps_std": float(np.std(tps_b)),
            "j_tok_mean": float(np.mean(j_b)) if j_b else None,
            "j_tok_std": float(np.std(j_b)) if j_b else 0.0,
            "power_mean": float(np.mean(pwr_b)) if pwr_b else None,
        }

        spec_trials = by_prompt_condition[label]["speculative"]
        tps_s = [t["throughput_tok_sec"] for t in spec_trials]
        j_s = [t["joules_per_token"] for t in spec_trials if t["joules_per_token"] is not None]
        pwr_s = [t["avg_power_watts"] for t in spec_trials if t["avg_power_watts"] is not None]
        acc_s = [t["accept_rate_pct"] for t in spec_trials if t["accept_rate_pct"] is not None]
        ablation[p_key][SPEC_KEY] = {
            "tps_mean": float(np.mean(tps_s)),
            "tps_std": float(np.std(tps_s)),
            "j_tok_mean": float(np.mean(j_s)) if j_s else None,
            "j_tok_std": float(np.std(j_s)) if j_s else 0.0,
            "power_mean": float(np.mean(pwr_s)) if pwr_s else None,
            "accept_rate_pct_mean": float(np.mean(acc_s)) if acc_s else None,
        }

    # -- drift diagnostics: pooled and per condition -------------------------
    def drift_block(entries):
        idx = [e["global_index"] for e in entries]
        d = {
            "power_w": bench_common.fit_drift(idx, [e["avg_power_watts"] for e in entries]),
            "joules_per_token": bench_common.fit_drift(idx, [e["joules_per_token"] for e in entries]),
            "throughput_tok_sec": bench_common.fit_drift(idx, [e["throughput_tok_sec"] for e in entries]),
            "temp_c": bench_common.fit_drift(idx, [e["temp_end_c"] for e in entries]),
        }
        d["residual_drift_acceptable"] = bench_common.drift_verdict(d["power_w"])
        d["threshold_fraction"] = bench_common.DRIFT_WARN_FRACTION
        return d

    baseline_entries = [e for e in chronological if e["condition"] == BASELINE_KEY]
    spec_entries = [e for e in chronological if e["condition"] == SPEC_KEY]
    drift = {
        "pooled": drift_block(chronological),
        "baseline_only": drift_block(baseline_entries),
        "speculative_only": drift_block(spec_entries),
    }

    print("\n" + "=" * 85)
    for p_key, m_dict in ablation.items():
        print(f"\n-- {p_key} --")
        print(f"{'Mode':<32} | {'Throughput (tok/s)':<20} | {'Energy (J/token)':<18} | {'Accept %':<10}")
        print("-" * 88)
        for m, st in m_dict.items():
            tps_str = f"{st['tps_mean']:.2f} ± {st['tps_std']:.2f}"
            j_str = f"{st['j_tok_mean']:.3f} ± {st['j_tok_std']:.3f}" if st["j_tok_mean"] is not None else "n/a"
            acc_str = f"{st['accept_rate_pct_mean']:.1f}%" if st.get("accept_rate_pct_mean") is not None else "n/a"
            print(f"{m:<32} | {tps_str:<20} | {j_str:<18} | {acc_str:<10}")
    print("=" * 85)

    pdrift = drift["pooled"]["power_w"]
    print("\nDRIFT CHECK (chronological, pooled across both conditions)")
    if pdrift:
        print(f"  power slope       : {pdrift['slope_per_trial']:+.4f} W/trial "
              f"(R^2={pdrift['r2']:.3f}, total {pdrift['total_change_over_run']:+.2f} W = "
              f"{pdrift['fraction_of_mean'] * 100:+.2f}%)")
    if drift["pooled"]["residual_drift_acceptable"]:
        print("[ok] Residual drift within threshold. Means are usable.")
    else:
        print(f"[!] Residual drift exceeds {bench_common.DRIFT_WARN_FRACTION * 100:.1f}% of mean power.")
        print("    Raise --min-warmup and/or lock clocks, then re-run.")
    if not warmup["converged"]:
        print("[!] Warmup did not converge before the cap. Raise --max-warmup.")

    out = {
        "fidelity_by_prompt": fidelity_by_prompt,
        "ablation": ablation,
        "_meta": {
            "timestamp": datetime.now().isoformat(),
            "device": dev,
            "compare_against": {
                "benchmark_ablation.py": "same prompts/K/models, UNCACHED -- the only "
                                          "valid one-variable-changed comparison for "
                                          "'does caching change the speedup'.",
                "benchmark_vllm_comparison.py": "different implementation (vLLM native) "
                                                 "AND different serving stack (PagedAttention, "
                                                 "no continuous batching here) -- do not treat "
                                                 "a difference between this file and that one "
                                                 "as isolating the speculative mechanism alone.",
            },
            "config": {
                "target_model": TARGET_MODEL_ID,
                "scout_model": SCOUT_MODEL_ID,
                "K": K_DRAFT,
                "max_tokens": max_tokens,
                "num_trials": num_trials,
                "warmup_steps_per_trial": WARMUP_STEPS,
                "clocks_locked": monitor._clocks_locked,
                "trial_order": "interleaved: category round-robin, condition alternated each round",
                # Recorded explicitly (benchmark_ablation.py's _meta does not
                # currently do this) so a future reader can tell which prompt
                # set produced this file without cross-referencing bench_common.py's
                # current state, which can change.
                "prompt_labels": PROMPT_LABELS,
                "prompts": PROMPTS,
                # Recorded explicitly so a results file on its own says which
                # attention backend produced it (backends round differently,
                # so throughput and near-tie counts can differ between them).
                "attn_implementation_requested": args.attn_implementation,
                "attn_implementation_actual": target_model.config._attn_implementation,
            },
            "selftest_skipped": args.skip_selftest,
            "fidelity_gate_tokens": args.fidelity_gate_tokens,
            "selftest": selftest_result,
            "fidelity_gate": gate_report,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "warmup": warmup,
            "drift_diagnostics": drift,
            "poll_errors": monitor.poll_errors,
        },
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=bench_common.json_safe)

    monitor.close()
    print(f"\n[*] Results saved to: {RESULTS_JSON}")
    print(f"[*] Per-trial CSV:    {TRIAL_CSV}")


if __name__ == "__main__":
    main()
