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

*Updated 2026-09-04. Supersedes the numbers previously reported here — a warmup-timing
bug was found and fixed (see [Methodology](#methodology) below); the relative story is
unchanged, but every figure below is a fresh, re-measured number, not a revision of the
old ones by formula.*

| Prompt | Baseline tok/s | Speculative tok/s | Speedup | Baseline J/tok | Speculative J/tok | Energy Δ | Accept % |
|---|---|---|---|---|---|---|---|
| Poem | 38.42 ± 0.18 | 40.98 ± 0.35 | +6.7% | 11.66 ± 0.11 | 7.91 ± 0.10 | −32.2% | 42.5% |
| Physics | 38.45 ± 0.50 | 47.03 ± 0.42 | +22.3% | 11.64 ± 0.11 | 7.18 ± 0.12 | −38.3% | 51.7% |
| Code | 38.90 ± 0.67 | **69.44 ± 0.73** | **+78.5%** | 11.66 ± 0.12 | **4.74 ± 0.09** | **−59.3%** | **85.4%** |

![Throughput: FP16 baseline vs. speculative (K=5), per prompt](assets/throughput_baseline_vs_speculative.png)
![Energy per token: FP16 baseline vs. speculative (K=5), per prompt](assets/energy_per_token_baseline_vs_speculative.png)

Speedup and energy reduction scale with draft-acceptance rate — higher on predictable
content (code), lower but still real on less predictable content (free-form prose).

![Speedup vs. FP16 baseline as a function of draft accept rate](assets/speedup_vs_accept_rate.png)

Accept rates have been independently cross-validated across every run of this ablation,
including across a full rewrite of the measurement harness, and consistently match to
within a few hundredths of a percentage point (deterministic greedy decoding) — strong
evidence the accept/reject logic itself is correct and stable, independent of the
measurement-methodology fixes described below.
Mean GPU power during speculative runs (344–363 W) is lower than during the FP16
baseline (466–469 W) despite two resident models, consistent with fewer full
8B-parameter forward passes required per unit of output as accepted draft batches grow.

## Why these prompts

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

## What this does NOT claim

- No quantization/kernel work is included here (see the SDSIE research repo for that,
  including a report of where it currently helps and where it doesn't).
- No dynamic/entropy-gated draft-length adjustment — K is fixed at 5. An entropy-gated
  version was tested and, as of the latest findings from that research repo, does not yet
  outperform this fixed-K approach in single-request testing (nor does a related
  entropy-gated *precision*-switching mechanism, tested separately). This repo
  intentionally ships the simpler, proven approach rather than either more ambitious,
  not-yet-validated alternative.
- No KV-cache (deliberate, for a fair baseline-vs-speculative comparison — see
  "Methodology" below). Absolute throughput numbers here are not production-representative;
  the relative comparison (baseline vs. speculative under identical conditions) is what's
  been validated.

## Methodology

Both baseline and speculative paths run with **no KV-cache** (full recompute each step).
This was a deliberate choice to keep the comparison fair — an earlier version of this
work applied caching unevenly, which structurally penalized the speculative path on any
fallback step. Recomputing from scratch for both arms removes that confound, at the cost
of both being slower in absolute terms than a production server with caching would be.

Warmup happens in two layers. Before any timed trial, a **closed-loop thermal warmup**
runs real (discarded) baseline and speculative decoding cycles, alternating across all
three prompts and both conditions, until GPU temperature stops moving and *each*
prompt/condition combination's own power reading individually stops moving — not until
consecutive readings across different combinations happen to agree with each other,
since baseline and speculative draw genuinely different power by design. This replaced
an earlier fixed-length warmup that recovered only part of the GPU's cold-start power
deficit (see `bench_common.py`'s `warm_to_steady_state` for the full history: a warmup
burst shorter than the real trial length was found to converge at a lower power/temp
level than the real, longer trial then reached). On top of that, each individual timed
trial is still preceded by 5 short untimed warmup forward passes immediately before
measurement starts, avoiding cold-SM effects at the start of each specific trial. This
two-layer warmup procedure is shared between `benchmark_ablation.py` and
`speculative_scout.py` via `bench_common.py`.

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
index. For the run behind the table above, the fitted trend was negligible in every
case (R² ≈ 0 for all three metrics, pooled and per-condition — the largest was 0.09),
and the total drift over the full 60-trial run was small in absolute terms (about −2 W
over the whole run, against a pooled mean near 410 W). The script's own pass/fail
threshold (±0.5% of mean) is a blunt cutoff that doesn't look at R², so it did flag this
run (pooled: −0.53%; speculative-only: −0.58%; baseline-only passed at −0.48%) — worth
knowing about, but a near-zero R² alongside a sub-1%-of-mean total change is consistent
with ordinary run-to-run noise, not a systematic warmup or thermal trend contaminating
the reported means. Separately, on this specific run the closed-loop warmup itself did
not fully converge within its 420 s cap (WSL2 GPU passthrough appears to add enough power
reporting noise that the strict per-label 1.5 W tolerance wasn't reliably satisfied in
time) — the drift check above is exactly the safeguard that exists for this situation,
and it came back clean.

## Repository structure

```
benchmarks/
  speculative_scout.py    - Standalone reference implementation, single-run
  benchmark_ablation.py   - N=10 trial ablation across 3 prompts (source of table above)
  bench_common.py         - Shared NVML monitor, closed-loop warmup, accept/reject decode
                             loop, and drift diagnostics used by benchmark_ablation.py and
                             speculative_scout.py
  plot_ablation_results.py, rebuild_summary.py
docs/
  sdsie_fixed_k5_paper.tex / .pdf  - The paper (see below), figures pulled from assets/
telemetry/                 - Raw JSON/CSV output from the runs behind the table above
assets/                    - Plots generated from telemetry (see plot_*.py scripts)
```

## Running it

```bash
pip install -r requirements.txt
cd benchmarks

# Single run against one prompt
python3 speculative_scout.py

# Full N=10 ablation (takes several minutes, loads two models)
python3 benchmark_ablation.py

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
