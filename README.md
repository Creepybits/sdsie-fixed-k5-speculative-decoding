# Fixed-K Speculative Decoding: Real, Reproducible Energy & Speed Gains  

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
| Poem | 38.2 | 40.9 | +7.1% | 12.42 ± 0.17 | 8.38 ± 0.13 | −32.5% | 42.5% |
| Physics | 37.9 | 46.9 | +23.7% | 12.54 ± 0.16 | 7.66 ± 0.10 | −38.9% | 51.7% |
| Code | 38.2 | **69.2** | **+81.2%** | 12.54 ± 0.17 | **5.01 ± 0.10** | **−60.0%** | **85.4%** |

Speedup and energy reduction scale with draft-acceptance rate — higher on predictable
content (code), lower but still real on less predictable content (free-form prose).

Accept rates were independently cross-validated across three separately-executed runs
(deterministic greedy decoding), consistent with correctly-implemented accept/reject
logic. Mean GPU power during speculative runs (343–359 W) is lower than during the FP16
baseline (474–479 W) despite two resident models, consistent with fewer full 8B-parameter
forward passes required per unit of output as accepted draft batches grow.

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

Power measured via 100 Hz NVML polling. Fidelity measured as exact greedy-decoding token
match between baseline and speculative output.

## Repository structure

```
speculative_scout.py      - Standalone reference implementation, single-run
benchmark_ablation.py     - N=10 trial ablation across 3 prompts (source of table above)
fp16_baseline.py          - Standalone matched baseline, same harness/methodology
telemetry/                - Raw JSON/CSV output from the runs behind the table above
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
