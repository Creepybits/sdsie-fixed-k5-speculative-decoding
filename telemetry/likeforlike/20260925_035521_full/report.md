# Like-for-like speculative decoding comparison

Preset `full`, prompt set `extended`, 2 repetition(s), 250 tokens, concurrency 1, prefix cache off, order seed 469507.

Numbers from different engines are **not** comparable with each other -- compare within each engine only.


## sglang  (version 0.5.20)

- Prompts: 27. Best configuration chosen on 14, reported on 13.
- Baseline drift within a block (closing vs opening baseline): 1.5%.
- Fidelity control (baseline run vs baseline run): 100.0% of tokens match on average, 100% of outputs identical.

### Every configuration (all prompts)

| Config | Speedup [95% range] | Energy/token | Accept len | Fidelity (match / identical) | Runs | Flags |
|---|---|---|---|---|---|---|
| `eagle3_k5_topk8_ndt32` | +106.8% [+94.7, +119.7] | -59.2% | 3.53 | 62% / 41% | 2 | - |
| `draft_k5` **K5** | +76.7% [+66.1, +88.3] | -54.3% | 4.28 | 52% / 30% | 2 | - |
| `draft_k7` | +74.0% [+60.4, +89.7] | -52.9% | 5.07 | 55% / 30% | 2 | - |
| `draft_k3` | +65.8% [+58.6, +73.2] | -51.5% | 3.20 | 58% / 37% | 2 | - |
| `eagle3_k5` | +55.2% [+45.1, +66.4] | -47.7% | 2.46 | 52% / 30% | 2 | - |
| `eagle3_k3` | +53.8% [+46.5, +62.5] | -48.8% | 2.20 | 62% / 41% | 2 | - |
| `ngram_ndt16` | +8.0% [+2.7, +13.8] | -23.2% | 1.41 | 52% / 31% | 2 | - |
| `eagle_k5` | -44.1% [-44.4, -43.9] | +49.2% | 1.01 | 54% / 30% | 2 | not_speculating |

### Best configuration of each method (evaluation prompts only)

| Method | Best config | Speedup [95% range] | Energy/token |
|---|---|---|---|
| eagle3 | `eagle3_k5_topk8_ndt32` | +107.5% [+88.5, +129.5] | -59.3% |
| draft | `draft_k5` | +79.4% [+62.8, +96.8] | -55.2% |
| ngram | `ngram_ndt16` | +8.1% [-0.0, +17.8] | -23.4% |

### Head to head: K5 vs each method's best (evaluation prompts)

| Opponent | K5 speed relative to it [95% range] | K5 faster on | K5 energy/token relative to it |
|---|---|---|---|
| eagle3 (`eagle3_k5_topk8_ndt32`) | -13.6% [-19.0, -8.5] | 0/13 prompts | +10.2% |
| ngram (`ngram_ndt16`) | +65.9% [+57.1, +75.2] | 13/13 prompts | -41.5% |

Positive speed = K5 faster; negative energy = K5 uses less energy per token. If the 95% range crosses 0, the difference is not established by this data.

### Speedup by prompt category (all prompts, best config per method)

| Category | `draft_k5` | `eagle3_k5_topk8_ndt32` | `ngram_ndt16` |
|---|---|---|---|
| coding | +120.8% | +137.1% | +12.0% |
| extraction | +104.9% | +138.9% | +9.6% |
| humanities | +64.0% | +101.8% | +0.9% |
| math | +95.5% | +106.9% | +31.1% |
| reasoning | +78.8% | +117.1% | +18.5% |
| reference | +66.3% | +113.6% | +10.4% |
| roleplay | +58.4% | +72.1% | -0.8% |
| stem | +71.2% | +97.3% | +4.2% |
| writing | +43.6% | +85.7% | -8.8% |

## vllm  (version 0.27.1)

- Prompts: 27. Best configuration chosen on 14, reported on 13.
- Baseline drift within a block (closing vs opening baseline): 4.1%  **-- above 3%, treat small differences with suspicion**.
- Fidelity control (baseline run vs baseline run): 100.0% of tokens match on average, 100% of outputs identical.

### Every configuration (all prompts)

| Config | Speedup [95% range] | Energy/token | Accept len | Fidelity (match / identical) | Runs | Flags |
|---|---|---|---|---|---|---|
| `eagle3_k3_matched` | +79.6% [+70.4, +90.2] | -50.1% | 2.21 | 65% / 44% | 2 | - |
| `eagle3_k5_matched` | +76.1% [+64.0, +89.1] | -50.0% | 2.47 | 72% / 48% | 2 | - |
| `eagle3_k3` | +74.0% [+64.9, +84.2] | -49.4% | 2.21 | 65% / 44% | 2 | - |
| `eagle3_k5` | +73.8% [+62.2, +86.1] | -48.9% | 2.47 | 72% / 48% | 2 | - |
| `eagle3_k2` | +65.8% [+59.4, +72.5] | -47.3% | 1.99 | 65% / 44% | 2 | - |
| `eagle_k3` | +59.7% [+51.4, +68.9] | -42.2% | 2.27 | 63% / 37% | 2 | - |
| `draft_k3` | +56.6% [+50.6, +63.2] | -51.3% | 3.18 | 58% / 33% | 2 | - |
| `draft_k5` **K5** | +54.6% [+45.3, +64.7] | -51.7% | 4.19 | 59% / 33% | 2 | - |
| `draft_k7` | +48.7% [+37.1, +61.7] | -49.3% | 4.93 | 62% / 33% | 2 | - |
| `eagle_k5` | +45.3% [+35.8, +56.3] | -36.4% | 2.43 | 58% / 33% | 2 | - |
| `ngram_k5` | -7.6% [-10.9, -4.1] | -14.1% | 1.93 | 67% / 48% | 2 | - |

### Best configuration of each method (evaluation prompts only)

| Method | Best config | Speedup [95% range] | Energy/token |
|---|---|---|---|
| eagle3_matched | `eagle3_k3_matched` | +83.2% [+68.4, +100.3] | -50.7% |
| eagle3 | `eagle3_k5` | +74.2% [+57.1, +95.7] | -48.8% |
| eagle | `eagle_k3` | +61.5% [+47.3, +77.5] | -43.1% |
| draft | `draft_k3` | +58.3% [+48.9, +67.9] | -51.7% |
| ngram | `ngram_k5` | -6.4% [-11.5, -0.8] | -15.4% |
| draft (fixed K=5, i.e. K5) | `draft_k5` | +59.9% [+46.8, +74.1] | -53.4% |

### Head to head: K5 vs each method's best (evaluation prompts)

| Opponent | K5 speed relative to it [95% range] | K5 faster on | K5 energy/token relative to it |
|---|---|---|---|
| eagle3 (`eagle3_k5`) | -8.2% [-14.5, -1.5] | 4/13 prompts | -9.1% |
| draft (`draft_k3`) | +1.0% [-2.8, +4.8] | 8/13 prompts | -3.5% |
| eagle (`eagle_k3`) | -1.0% [-5.3, +3.5] | 7/13 prompts | -18.2% |
| ngram (`ngram_k5`) | +70.9% [+62.5, +79.7] | 13/13 prompts | -45.0% |
| eagle3_matched (`eagle3_k3_matched`) | -12.7% [-17.3, -7.2] | 2/13 prompts | -5.5% |

Positive speed = K5 faster; negative energy = K5 uses less energy per token. If the 95% range crosses 0, the difference is not established by this data.

`_matched` rows: the opponent ran with vLLM's async scheduling turned off, as vLLM does on its own for K5's draft_model method. This matches async scheduling only -- vLLM also runs draft_model on its older V1 model runner, which is NOT matched. The unmarked row is the method as vLLM ships it.

### Speedup by prompt category (all prompts, best config per method)

| Category | `eagle3_k5` | `draft_k3` | `eagle_k3` | `ngram_k5` | `eagle3_k3_matched` | `draft_k5` |
|---|---|---|---|---|---|---|
| coding | +108.0% | +76.4% | +87.0% | -6.1% | +111.8% | +89.2% |
| extraction | +97.6% | +69.3% | +80.2% | +3.0% | +101.4% | +76.5% |
| humanities | +84.7% | +48.7% | +51.4% | -12.1% | +82.3% | +47.7% |
| math | +67.1% | +69.7% | +73.5% | +2.1% | +75.3% | +75.5% |
| reasoning | +74.6% | +54.7% | +70.7% | -1.8% | +75.1% | +52.8% |
| reference | +80.8% | +56.5% | +59.1% | -7.4% | +88.4% | +47.9% |
| roleplay | +42.9% | +40.6% | +38.4% | -12.5% | +54.2% | +37.8% |
| stem | +69.9% | +58.1% | +55.3% | -11.4% | +78.2% | +45.0% |
| writing | +49.0% | +39.6% | +31.0% | -20.0% | +57.2% | +28.9% |
