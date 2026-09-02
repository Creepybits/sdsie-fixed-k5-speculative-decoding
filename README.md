# Fixed-K Speculative Decoding: Real, Reproducible Energy & Speed Gains  

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22210487.svg)](https://doi.org/10.5281/zenodo.22210487)
[![Hardware](https://img.shields.io/badge/Verified%20On-NVIDIA%20RTX%205090%20Blackwell-10b981.svg)](https://sdsie.github.io/)
[![Live Portal](https://img.shields.io/badge/Interactive%20Portal-sdsie.github.io-a855f7.svg)](https://sdsie.github.io/)  
___
Part of the broader [SDSIE](https://github.com/Creepybits/software-defined-stochastic-inference-engine)
research project, which also explores entropy-gated dynamic speculation and INT4
quantization — pieces still under active investigation, not included here. This repo
contains only the specific mechanism (fixed-K speculative decoding) that's fully
validated and reproducible as claimed. See [Relationship to SDSIE](https://github.com/Creepybits/sdsie-fixed-k5-speculative-decoding#relationship-to-sdsie) below for more.  
___

Real scout(1B)→target(8B) speculative decoding with a lossless verify/rollback loop,
fixed draft window K=5. **Every number below is real, independently reproduced, and
traceable to a script in this repo.**

This is a spin-off of a specific, working piece from a larger, more experimental
research project ([SDSIE](https://github.com/Creepybits/software-defined-stochastic-inference-engine)),
which also explores entropy-gated *dynamic* speculation and INT4 quantization. Those
pieces are still under active investigation and are **not** included here — this repo
contains only the part that's fully validated and works as claimed.

## Results (N=10 trials/prompt, RTX 5090 Blackwell, 100% output fidelity on all rows)

| Prompt | Baseline tok/s | Speculative tok/s | Speedup | Baseline J/tok | Speculative J/tok | Energy Δ | Accept % |
|---|---|---|---|---|---|---|---|
| Poem | 40.0 | 43.4 | +8.6% | 11.58 ± 0.17 | 7.80 ± 0.08 | −32.6% | 42.5% |
| Physics | 40.2 | 50.1 | +24.8% | 11.64 ± 0.10 | 7.09 ± 0.08 | −39.1% | 51.7% |
| Code | 40.7 | **74.1** | **+82.3%** | 11.64 ± 0.10 | **4.62 ± 0.09** | **−60.3%** | **85.4%** |

Speedup and energy reduction scale with draft-acceptance rate — higher on predictable
content (code), lower but still real on less predictable content (free-form prose).

Accept rates were independently cross-validated across three separately-executed runs
(deterministic greedy decoding), consistent with correctly-implemented accept/reject
logic. Mean GPU power during speculative runs (338.6–355.2 W) is lower
than during the FP16 baseline (462.9–473.2 W) despite two resident models,
consistent with fewer full 8B-parameter forward passes required per unit of output as
accepted draft batches grow.

## What this does NOT claim

- No quantization/kernel work is included here (see the parent SDSIE repo for that,
  including a report of where it currently helps and where it doesn't).
- No dynamic/entropy-gated draft-length adjustment — K is fixed at 5. An entropy-gated
  version was tested and, as of the parent project's latest findings, does not yet
  outperform this fixed-K approach in single-request testing. This repo intentionally
  ships the simpler, proven approach rather than the more ambitious, not-yet-validated one.
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

Both baseline and speculative measurements are preceded by 5 untimed warmup forward
passes (not recorded), so reported numbers reflect steady-state GPU clock/thermal
behavior rather than cold-start overhead. This warmup procedure is applied identically
across `benchmark_ablation.py`, `fp16_baseline.py`, and `speculative_scout.py`.

Power measured via 100 Hz NVML polling. Fidelity measured as exact greedy-decoding token
match between baseline and speculative output.

## Repository structure

```
speculative_scout.py      - Standalone reference implementation, single-run
benchmark_ablation.py     - N=10 trial ablation across 3 prompts (source of table above)
fp16_baseline.py          - Standalone matched baseline, same warmup/methodology
telemetry/                - Raw JSON/CSV output from the runs behind the table above
assets/                   - Plots generated from telemetry (see plot_*.py scripts)
```

## Running it

```bash
pip install -r requirements.txt

# Single run against one prompt
python3 speculative_scout.py

# Full N=10 ablation (takes several minutes, loads two models)
python3 benchmark_ablation.py

# Matched FP16 baseline only
python3 fp16_baseline.py

# Regenerate plots from the latest telemetry
python3 plot_ablation_results.py
python3 plot_baseline_stability.py
```

Requires an NVIDIA GPU with enough VRAM for both a ~1B and ~8B parameter model in
bfloat16 (roughly 18-20 GB total), and access to the `meta-llama/Llama-3.2-1B-Instruct`
and `meta-llama/Llama-3.1-8B-Instruct` checkpoints (gated on Hugging Face — request
access first if you haven't already).

## Relationship to SDSIE

This repo exists because the parent SDSIE project's own correction process (see its
README) found that a claimed unified system (quantization + entropy-gated speculation)
wasn't actually wired together end-to-end, while this specific fixed-K speculative
decoding piece was real and independently reproducible. Rather than keep the validated
and experimental results bundled together, this repo isolates the part that's proven,
so it can be evaluated on its own terms without the open research questions attached.

## License

Apache-2.0.
