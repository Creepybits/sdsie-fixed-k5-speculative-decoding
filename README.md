# Fixed-K Speculative Decoding: Real, Reproducible Energy & Speed Gains  

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22210487.svg)](https://doi.org/10.5281/zenodo.22210487)
[![Hardware](https://img.shields.io/badge/Verified%20On-NVIDIA%20RTX%205090%20Blackwell-10b981.svg)](https://sdsie.github.io/)
[![Live Portal](https://img.shields.io/badge/Interactive%20Portal-sdsie.github.io-a855f7.svg)](https://sdsie.github.io/)  

Real scout(1B)→target(8B) speculative decoding with a lossless verify/rollback loop,
fixed draft window K=5. **Every number below is real, independently reproduced, and
traceable to a script in this repo.** This is the fully validated, production-ready
result from the SDSIE research line — recommended for evaluation or deployment today.
___
Historical note: this repo originated as an extraction from the broader
[SDSIE](https://github.com/Creepybits/software-defined-stochastic-inference-engine)
research project. SDSIE also explores more ambitious entropy-gated *dynamic* speculation
and INT4 quantization-switching mechanisms; both have since been tested for real and, so
far, found not to improve on this simpler, fixed approach — see
[Relationship to SDSIE](#relationship-to-sdsie) below for the full, honest account.
___

## Results (N=10 trials/prompt, RTX 5090 Blackwell, 100% output fidelity on all rows)

*Updated 2026-09-05. Supersedes the numbers previously reported here — a further warmup
convergence fix (see [Methodology](#methodology) below) let the closed-loop warmup reach
genuine steady state for the first time (previous runs hit their time cap without
converging); every figure below is a fresh, re-measured number from a run with the
cleanest drift diagnostics of the project to date.*

| Prompt | Baseline tok/s | Speculative tok/s | Speedup | Baseline J/tok | Speculative J/tok | Energy Δ | Accept % |
|---|---|---|---|---|---|---|---|
| Poem | 40.16 ± 0.28 | 43.55 ± 0.31 | +8.4% | 11.44 ± 0.09 | 7.70 ± 0.06 | −32.7% | 42.5% |
| Physics | 40.05 ± 0.18 | 49.64 ± 0.43 | +24.0% | 11.49 ± 0.06 | 6.99 ± 0.06 | −39.2% | 51.7% |
| Code | 40.04 ± 0.23 | **73.46 ± 0.74** | **+83.5%** | 11.54 ± 0.05 | **4.53 ± 0.04** | **−60.8%** | **85.4%** |

![Throughput: FP16 baseline vs. speculative (K=5), per prompt](assets/throughput_baseline_vs_speculative.png)
![Energy per token: FP16 baseline vs. speculative (K=5), per prompt](assets/energy_per_token_baseline_vs_speculative.png)

Speedup and energy reduction scale with draft-acceptance rate — higher on predictable
content (code), lower but still real on less predictable content (free-form prose).

![Speedup vs. FP16 baseline as a function of draft accept rate](assets/speedup_vs_accept_rate.png)

Accept rates have been independently cross-validated across every run of this ablation,
including across a full rewrite of the measurement harness, and consistently match to
within a few hundredths of a percentage point (deterministic greedy decoding) — strong
evidence the accept/reject logic itself is correct and stable, independent of the
measurement-methodology fixes described below. This run's accept rates (42.5% / 51.7% /
85.4%) also match `speculative_scout.py`'s independent single-run figures exactly, and
its throughput/energy numbers land within a few percent of that script's own
independently-measured values — two different scripts, same underlying `bench_common.py`,
agreeing with each other rather than just being internally consistent with themselves.
Mean GPU power during speculative runs (346–364 W) is lower than during the FP16
baseline (463–466 W) despite two resident models, consistent with fewer full
8B-parameter forward passes required per unit of output as accepted draft batches grow.

### Why these prompts

The three prompts span a range of *token-level predictability* for the model, not
difficulty for a person — and that distinction matters, because the results above can
look backwards at first glance. Code is often considered a more cognitively demanding
task than free-form poetry, yet it gets the largest speedup (85.4% accept rate) while
poetry gets the smallest (42.5%).

The resolution: speculative decoding's accept rate depends on how sharply peaked the
model's next-token probability distribution is, which is a different axis from how hard
a task is for a person. Code has strict, learned syntactic structure — matching
brackets, indentation rules, a constrained vocabulary of keywords and common idioms — so
at most token positions there is essentially one syntactically valid continuation, or a
very small set of them. The scout's greedy guess is usually right, and draft windows
survive largely intact regardless of how logically demanding the underlying code is to
write. Open-ended creative writing has close to the opposite property at the token
level: at nearly every position there are many equally plausible word choices (synonyms,
alternate phrasings, meter- and rhyme-driven word selection for the Chant Royal form
used here), so the model's distribution is flatter and the scout is wrong more often —
again, independent of how hard the poem actually is to compose. Physics explanation
falls in between: more lexical variety than code, but far less than open verse, and its
51.7% accept rate lands squarely between the other two.

This is precisely the axis speculative decoding's speedup is sensitive to (see the
accept-rate figure above), which is why the prompts were chosen to span it deliberately
— not to span perceived task difficulty.

## Comparison to vLLM's own built-in speculative decoding

*Added 2026-09-18. A separate, additional benchmark, not part of the main table above —
see the caveats immediately below before citing this anywhere.*

vLLM ships its own native speculative decoding, including a `draft_model` method with a
fixed `num_speculative_tokens` setting — mechanistically the same approach as this repo
(small draft model, fixed draft window, target model verifies). Until now this repo had
never actually been benchmarked against it. `benchmarks/benchmark_vllm_comparison.py`
does that directly: same scout (`Llama-3.2-1B-Instruct`) and target
(`Llama-3.1-8B-Instruct`) models, same three reference prompts, vLLM's own engine and
KV-cache for both arms, `N=10` trials/prompt, greedy decoding.

| Prompt | vLLM baseline (tok/s) | vLLM + K5 mechanism (tok/s) | Speedup |
|---|---|---|---|
| Poem | 100.6 | 123.1 | +22.4% |
| Physics | 100.8 | 151.8 | +50.6% |
| Code | 100.8 | 201.4 | **+99.9%** |

The K5 mechanism (fixed-K=5, scout→target draft-and-verify) measurably outperforms
vLLM's own FP16 baseline when run natively inside vLLM's serving engine, across all
three reference prompts.

**Important caveats:**
- **Not directly comparable to the main results table above.** vLLM uses PagedAttention
  (KV-caching) for both arms here; the main table deliberately uses no KV-cache on
  either arm, for a different reason (isolating the speculative mechanism itself from
  caching effects). These are two different, both-valid comparisons answering different
  questions — don't put both tables' baseline numbers side by side as if they were one
  data set.
- **vLLM only, not SGLang.** SGLang's own built-in speculative decoding is adaptive
  (EAGLE-based, tiered draft length) — a different mechanism, closer in spirit to
  SDSIE's still-unvalidated entropy-gated approach than to fixed-K5. It has not been
  benchmarked here.
- **Single full run, not yet independently cross-validated by a second script**, unlike
  the main results above (which are corroborated by `speculative_scout.py`
  independently). Measurement methodology (closed-loop per-prompt warmup via
  `bench_common.warm_to_steady_state`, plus a short per-trial warmup) mirrors the rest
  of this repo, and a first-round rotation-order artifact (whichever prompt ran first in
  a round measured slower, regardless of which prompt it was) was found and fixed during
  development — see the script's own docstring and commit history for the full
  diagnostic trail.

## What this does NOT claim

- No quantization/kernel work is included here (see the SDSIE research repo for that,
  including a report of where it currently helps and where it doesn't).
- No dynamic/entropy-gated draft-length adjustment — K is fixed at 5. An entropy-gated
  version was tested and, as of the latest findings from that research repo, does not yet
  outperform this fixed-K approach in single-request testing (nor does a related
  entropy-gated *precision*-switching mechanism, tested separately). This repo
  intentionally ships the simpler, proven approach rather than either more ambitious,
  not-yet-validated alternative.
- No KV-cache in the main results table above (deliberate, for a fair baseline-vs-speculative
  comparison — see "Methodology" below). Absolute throughput numbers there are not
  production-representative; the relative comparison (baseline vs. speculative under
  identical conditions) is what's been validated. The separate vLLM comparison above
  does use KV-caching (vLLM's own default) and is reported separately for that reason.
- No SGLang comparison yet (see above).

## Methodology

Both baseline and speculative paths run with **no KV-cache** (full recompute each step).
This was a deliberate choice to keep the comparison fair — an earlier version of this
work applied caching unevenly, which structurally penalized the speculative path on any
fallback step. Recomputing from scratch for both arms removes that confound, at the cost
of both being slower in absolute terms than a production server with caching would be.

Warmup happens in two layers. Before any timed trial, a **closed-loop thermal warmup**
runs real (discarded) baseline and speculative decoding cycles, alternating across all
three prompts and both conditions, until *each* prompt/condition combination's own power
AND temperature readings individually stop drifting — not until consecutive readings
across different combinations happen to agree with each other, since baseline and
speculative draw genuinely different power (and settle at different temperatures) by
design. This is now a two-stage fix on top of the original design (see
`bench_common.py`'s `warm_to_steady_state` docstring for the full history): first, a
warmup burst shorter than the real trial length was found to converge at a lower
power/temp level than the real, longer trial then reached, so burst length was matched to
trial length; second, and found later, a fixed absolute power tolerance (1.5 W) turned
out to be tighter than the hardware's own sample-to-sample noise floor on this machine
(measured sd: 2.9 W baseline, 6.8 W speculative), so it could never be satisfied
regardless of run length, and temperature was pooled across labels on the assumption that
die temperature is workload-independent — true only when the workload is homogeneous,
false here, since baseline and speculative settle roughly 3-4°C apart. Both fixed: power
tolerance is now relative to each label's own mean draw (1.5%), temperature is tracked
per label like power, and both use a half-split drift statistic (newer-half mean vs.
older-half mean of a rolling window) rather than raw min-max spread, which does not grow
spuriously with window length under pure noise the way spread does. On top of the
closed-loop warmup, each individual timed trial is still preceded by 5 short untimed
warmup forward passes immediately before measurement starts, avoiding cold-SM effects at
the start of each specific trial. This two-layer warmup procedure is shared between
`benchmark_ablation.py` and `speculative_scout.py` via `bench_common.py`, and reused (via
the same `warm_to_steady_state` function) by `benchmark_vllm_comparison.py` above.

Energy per token is read from the GPU's onboard hardware energy counter
(`nvmlDeviceGetTotalEnergyConsumption`) when available — as it was for every trial in
the results above — rather than integrated from sampled power readings, which is a more
direct and less bias-prone measurement; sampled trapezoidal integration is retained as a
fallback and cross-check when the counter isn't supported. Power itself is still sampled
via 100 Hz NVML polling. Fidelity is measured as exact greedy-decoding token match
between baseline and speculative output.

### Measurement validity (drift check)

Each reported run's per-trial telemetry is checked for residual warmup drift: a
least-squares fit of power, energy/token, and throughput against chronological trial
index. The run behind the table above is the first on this project to have both a fully
**converged** closed-loop warmup (241.3 s, well under the 420 s cap — every prior run
hit the cap without converging) and the cleanest drift diagnostics to date: R² for every
metric, pooled and per-condition, is effectively zero (0.00004 to 0.037), and the
script's own pass/fail threshold (±0.5% of mean power) passed cleanly this time (pooled
0.30%, baseline-only 0.38%, speculative-only 0.19%) — not just "passed despite the flag
firing," as in earlier runs, but genuinely within tolerance on every check. This is the
strongest evidence yet that the reported means reflect real steady-state behavior rather
than a residual thermal trend.

## Repository structure

```
benchmarks/
  speculative_scout.py         - Standalone reference implementation, single-run
  benchmark_ablation.py        - N=10 trial ablation across 3 prompts (source of main table)
  benchmark_vllm_comparison.py - K5 mechanism vs. vLLM's own built-in speculative decoding
                                  (source of the vLLM comparison table above)
  bench_common.py              - Shared NVML monitor, closed-loop warmup, accept/reject
                                  decode loop, and drift diagnostics used by all three
                                  benchmark scripts above
  plot_ablation_results.py, rebuild_summary.py
docs/
  fixed_k5_paper.tex / .pdf  - The paper (see below), figures pulled from assets/
telemetry/                 - Raw JSON/CSV output from the runs behind the tables above
assets/                     - Plots generated from telemetry (see plot_*.py scripts)
```

## Running it

```bash
pip install -r requirements.txt
cd benchmarks

# Single run against one prompt
python3 speculative_scout.py

# Full N=10 ablation (takes several minutes, loads two models)
python3 benchmark_ablation.py

# K5 mechanism vs. vLLM's own speculative decoding (requires vllm; see script docstring)
python3 benchmark_vllm_comparison.py --mode baseline
python3 benchmark_vllm_comparison.py --mode speculative

# Regenerate plots from the latest telemetry
python3 plot_ablation_results.py
```

Requires an NVIDIA GPU with enough VRAM for both a ~1B and ~8B parameter model in
bfloat16 (roughly 18-20 GB total), and access to the `meta-llama/Llama-3.2-1B-Instruct`
and `meta-llama/Llama-3.1-8B-Instruct` checkpoints (gated on Hugging Face — request
access first if you haven't already).

## Relationship to SDSIE

This repo originated as an extraction from the broader
[SDSIE](https://github.com/Creepybits/software-defined-stochastic-inference-engine)
research project: an early correction process there (see its README) found that a
claimed unified system (quantization + entropy-gated speculation) wasn't actually wired
together end-to-end, while this specific fixed-K speculative decoding piece was real and
independently reproducible. Since then, SDSIE has gone on to test both of its more
ambitious entropy-gated mechanisms for real — adaptive speculative draft length, and
adaptive INT4/FP16 precision switching — and found neither yet improves on this
simpler, fixed approach. This repo remains the validated, production-ready result from
that research line; SDSIE remains the broader research project, reporting its ongoing
work (including negative results) with the same evidentiary standard.

## License

Apache-2.0.
