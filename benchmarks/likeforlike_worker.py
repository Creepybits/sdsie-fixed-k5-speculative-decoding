"""
likeforlike_worker.py

Runs ONE engine (vLLM or SGLang) with ONE speculative-decoding configuration,
measures it, and writes ONE JSON file. Normally launched by
likeforlike_suite.py, not by hand -- the suite decides the order, the
repetitions, and which Python (venv) to use for each engine.

Written 2026-09-24. Replaces nothing: the older benchmark_vllm_* /
benchmark_sglang_* scripts are left untouched.

===============================================================================
WHAT "LIKE-FOR-LIKE" MEANS HERE (the design rules, and why each exists)
===============================================================================

1. Same engine, only the drafting method changes. vLLM results are only ever
   compared with vLLM results, SGLang with SGLang. Numbers from different
   engines are never placed against each other -- they differ by engineering,
   not by algorithm.

2. Every method gets a small set of configurations, not one fixed K. Holding
   everything at K=5 (the older scripts' rule) lets a reviewer say EAGLE was
   handicapped. The analysis picks each method's best configuration on HALF
   the prompts and reports it on the OTHER half, so "best" isn't chosen and
   graded on the same data.

3. Each timed prompt is generated exactly ONCE per engine process. Some
   methods keep memory across requests (SGLang's NGRAM builds a shared table
   of text it has already produced). Repeating an identical greedy prompt in
   the same process then lets such a method "draft" its own previous answer,
   which looks like a huge speedup but isn't. Repetition is done across
   separate processes instead (the suite's --reps), which start with empty
   memory every time.

4. Warmup uses separate warmup prompts that are never timed, for the same
   reason, and prefix caching is OFF by default so every request pays its own
   prompt-processing cost in every method.

5. Nothing is silently trusted. Each run records the mean number of tokens
   produced per target-model verification step ("accept length"). The
   analysis uses it for two sanity checks: a method whose accept length is
   ~1.0 isn't actually speculating, and a speedup larger than the accept
   length is physically impossible (each verify step costs at least one
   ordinary decode step), so it points at a measurement problem.

6. Failures are recorded, not fatal. If an engine refuses a configuration,
   the error is written to the JSON and the suite moves on to the next run.

===============================================================================
ENGINE-SPECIFIC ASSUMPTIONS -- FLAGGED, NOT ASSUMED SAFE
===============================================================================

vLLM: speculative settings go in one nested speculative_config dict. The
  method names "draft_model", "eagle", "ngram" and the spec_decode counters are
  the ones already confirmed working in benchmark_vllm_spec_methods.py.
  "eagle3" and enable_prefix_caching=False are NOT yet confirmed on this
  machine -- if either fails, the error lands in the run's JSON.

SGLang: speculative settings are flat keyword arguments (see
  benchmark_sglang_spec_methods.py). Per-request accept length is read from
  meta_info["spec_verify_ct"] (number of verify steps for that request) if the
  installed version provides it -- NOT yet confirmed on this machine. The raw
  meta_info keys of the first request are printed and saved so this can be
  checked. If it's missing, the coarse server-cumulative average is recorded
  instead and the analysis marks accept length as "coarse".
  disable_radix_cache=True (prefix caching off) is also not yet confirmed to
  be compatible with every speculative algorithm; use --prefix-cache on if an
  engine refuses to start because of it (and apply it to ALL runs of that
  engine, so the comparison stays like-for-like).

--engine mock runs the whole pipeline without a GPU (simulated timings). It
exists only to test the suite and analysis plumbing. Mock numbers mean
nothing.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SCOUT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_EAGLE_DIR = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
DEFAULT_EAGLE3_DIR = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

METHODS = ["baseline", "draft", "eagle", "eagle3", "ngram"]

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Prompts
# ============================================================================
# The three reference prompts are copied here (not imported) so this file can
# run in --engine mock without bench_common's GPU dependencies. A check in
# main() confirms they still match bench_common.REFERENCE_PROMPTS for real runs.
REFERENCE_PROMPTS = [
    ("ref_poem", "reference", "Write an original Chant Royal poem in English celebrating mathematics."),
    ("ref_physics", "reference", "Explain the physics of semiconductor memory bandwidth and the memory wall."),
    ("ref_code", "reference", "Write a Python implementation of a binary search tree with type annotations."),
]

# 24 original prompts, 3 in each of the 8 categories used by MT-Bench (the
# benchmark the EAGLE papers report on). Written for this project, NOT the
# MT-Bench questions themselves -- for direct comparability with published
# figures, pass the real MT-Bench set via --prompts-file instead.
EXTENDED_PROMPTS = [
    ("writing_1", "writing", "Write a short story, about 300 words, about a lighthouse keeper who discovers the light has been signalling to someone."),
    ("writing_2", "writing", "Draft a persuasive op-ed arguing that small towns should invest in public libraries rather than new parking."),
    ("writing_3", "writing", "Write a heartfelt letter from a retiring teacher to the students she taught in her first year."),
    ("roleplay_1", "roleplay", "Pretend you are a 19th-century ship's navigator. Describe to a new sailor how you find your position using the stars."),
    ("roleplay_2", "roleplay", "You are a museum tour guide at a natural history museum. Introduce the dinosaur hall to a group of ten-year-olds."),
    ("roleplay_3", "roleplay", "Act as a patient barista explaining to a first-time customer the difference between a latte, a cappuccino and a flat white."),
    ("reasoning_1", "reasoning", "A farmer has 17 sheep. All but 9 run away. How many are left? Explain your reasoning step by step, then discuss why people often get this wrong."),
    ("reasoning_2", "reasoning", "Three friends split a restaurant bill unevenly. Anna paid twice what Ben paid, and Cara paid 10 more than Ben. The total was 90. Work out what each person paid, showing every step."),
    ("reasoning_3", "reasoning", "If all bloops are razzies and some razzies are lazzies, can we conclude that some bloops are lazzies? Explain carefully."),
    ("math_1", "math", "Solve the equation 3x^2 - 12x + 9 = 0 and explain each step, including how to check the answers."),
    ("math_2", "math", "Explain how to compute the sum of the first 100 positive integers, and prove the general formula by induction."),
    ("math_3", "math", "A circle has radius 5. Find the area of the square inscribed in it, and explain the geometry involved."),
    ("coding_1", "coding", "Write a Python function that returns the longest palindromic substring of a string, with comments and a few test cases."),
    ("coding_2", "coding", "Write a SQL query that finds, for each department, the employee with the highest salary, and explain how it works."),
    ("coding_3", "coding", "Implement a simple LRU cache class in Python with get and put methods, both in O(1) time, and explain the design."),
    ("extraction_1", "extraction",
     "Extract every person's name, their role, and the year mentioned from this text, and return them as a JSON list: "
     "'In 2019, Maria Lopez became chief engineer at the plant. Two years earlier, in 2017, her mentor David Chen had "
     "retired as operations director. The current plant manager, Aisha Rahman, joined in 2021.'"),
    ("extraction_2", "extraction",
     "From this product review, list every feature mentioned and whether the reviewer liked it, as a table: "
     "'The battery lasts two full days, which is great. The camera is sharp in daylight but noisy at night. "
     "I dislike the slippery back, but the screen is bright and easy to read outdoors.'"),
    ("extraction_3", "extraction",
     "Summarise the key dates and events in this passage as a bullet list: 'The bridge project was approved in March. "
     "Construction began in June after a two-month delay. The first span was completed in November, and the bridge "
     "opened to traffic the following April.'"),
    ("stem_1", "stem", "Explain how vaccines train the immune system, including the roles of antibodies and memory cells."),
    ("stem_2", "stem", "Describe how a refrigerator moves heat from inside to outside, and why this does not violate the second law of thermodynamics."),
    ("stem_3", "stem", "Explain why the sky is blue during the day and red at sunset."),
    ("humanities_1", "humanities", "Compare the causes of the French Revolution and the American Revolution in a structured essay."),
    ("humanities_2", "humanities", "Explain the main ideas of utilitarianism and give two common criticisms of it."),
    ("humanities_3", "humanities", "Discuss how the printing press changed the spread of ideas in 15th and 16th century Europe."),
]

# Never timed. Used for the closed-loop warmup and to keep the GPU busy
# between timed batches, so no timed prompt is ever seen before it's timed.
WARMUP_PROMPTS = [
    "Describe the process of glassblowing, from gathering molten glass through shaping, annealing and finishing.",
    "Explain how a bicycle stays upright while moving, in plain language.",
    "Write a short guide to caring for a sourdough starter.",
]


def load_prompts(prompt_set, prompts_file):
    if prompts_file:
        prompts = []
        with open(prompts_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts.append((obj.get("label", f"p{i}"), obj.get("category", "file"), obj["prompt"]))
        return prompts
    if prompt_set == "reference":
        return list(REFERENCE_PROMPTS)
    if prompt_set == "extended":
        return list(REFERENCE_PROMPTS) + list(EXTENDED_PROMPTS)
    raise ValueError(prompt_set)


def file_sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


# ============================================================================
# Engine adapters -- each exposes: build(), generate(texts, max_tokens),
# spec_snapshot(), version(), engine_kwargs
# ============================================================================

class VllmAdapter:
    name = "vllm"

    def __init__(self, a):
        self.a = a
        self.llm = None
        self.engine_kwargs = None

    def build(self):
        from vllm import LLM
        a = self.a
        kw = dict(
            model=TARGET_MODEL_ID,
            dtype="bfloat16",
            gpu_memory_utilization=a.gpu_memory_utilization,
            max_model_len=a.max_model_len,
            # Required for llm.get_metrics() to report the spec_decode
            # counters at all (found 2026-09-19).
            disable_log_stats=False,
            enable_prefix_caching=(a.prefix_cache == "on"),
            seed=0,
        )
        if a.no_async_scheduling:
            # "Engineering-matched" runs (added 2026-09-24): vLLM 0.27.1 disables
            # async scheduling for draft_model (K5) on its own, but not for EAGLE3.
            # Turning it off here puts EAGLE3 on the same footing. Only matches
            # async scheduling -- the V1-vs-V2 model runner difference vLLM also
            # reports for draft_model is NOT matched by this.
            kw["async_scheduling"] = False
        if a.method == "draft":
            kw["speculative_config"] = {"method": "draft_model", "model": SCOUT_MODEL_ID,
                                         "num_speculative_tokens": a.k,
                                         "max_model_len": a.max_model_len}
        elif a.method == "eagle":
            kw["speculative_config"] = {"method": "eagle", "model": a.eagle_dir,
                                         "num_speculative_tokens": a.k}
        elif a.method == "eagle3":
            kw["speculative_config"] = {"method": "eagle3", "model": a.eagle3_dir,
                                         "num_speculative_tokens": a.k}
        elif a.method == "ngram":
            kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": a.k,
                                         "prompt_lookup_max": a.prompt_lookup_max,
                                         "prompt_lookup_min": a.prompt_lookup_min}
        self.engine_kwargs = kw
        self.llm = LLM(**kw)

    def version(self):
        import vllm
        return getattr(vllm, "__version__", "unknown")

    def generate(self, texts, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=self.a.ignore_eos)
        outs = self.llm.generate(texts, sp, use_tqdm=False)
        res = []
        for o in outs:
            ids = list(o.outputs[0].token_ids)
            res.append({"ids": ids, "text": o.outputs[0].text, "n_tokens": len(ids),
                        "verify_steps": None, "meta_keys": None})
        return res

    def spec_snapshot(self):
        """(num_drafts, num_draft_tokens, num_accepted) -- cumulative counters."""
        drafts = draft_tokens = accepted = 0
        try:
            for m in self.llm.get_metrics():
                if m.name == "vllm:spec_decode_num_drafts":
                    drafts += m.value
                elif m.name == "vllm:spec_decode_num_draft_tokens":
                    draft_tokens += m.value
                elif m.name == "vllm:spec_decode_num_accepted_tokens":
                    accepted += m.value
        except Exception:
            return None
        return {"num_drafts": drafts, "num_draft_tokens": draft_tokens, "num_accepted": accepted}


class SglangAdapter:
    name = "sglang"

    def __init__(self, a):
        self.a = a
        self.llm = None
        self.engine_kwargs = None

    def build(self):
        import sglang as sgl
        a = self.a
        kw = dict(
            model_path=TARGET_MODEL_ID,
            mem_fraction_static=a.gpu_memory_utilization,
            context_length=a.max_model_len,
            dtype="bfloat16",  # explicit -- see benchmark_sglang_spec_methods.py (2026-09-21)
            disable_radix_cache=(a.prefix_cache == "off"),
            random_seed=0,
        )
        topk = a.topk
        ndt = a.num_draft_tokens
        if a.method in ("draft", "eagle", "eagle3"):
            # Chain drafting (topk=1): k draft steps -> k drafted tokens plus
            # the position they are verified from = k + 1 draft tokens.
            if ndt is None:
                ndt = a.k + 1 if topk == 1 else None
            if ndt is None:
                raise ValueError("--num-draft-tokens is required when --topk > 1 (tree drafting).")
            algo = {"draft": "STANDALONE", "eagle": "EAGLE", "eagle3": "EAGLE3"}[a.method]
            path = {"draft": SCOUT_MODEL_ID, "eagle": a.eagle_dir, "eagle3": a.eagle3_dir}[a.method]
            kw.update(speculative_algorithm=algo, speculative_draft_model_path=path,
                      speculative_num_steps=a.k, speculative_eagle_topk=topk,
                      speculative_num_draft_tokens=ndt)
        elif a.method == "ngram":
            # SGLang's own test fixture uses 16 draft tokens and no num_steps
            # (see benchmark_sglang_spec_methods.py, 2026-09-20).
            kw.update(speculative_algorithm="NGRAM",
                      speculative_num_draft_tokens=ndt if ndt is not None else 16)
        self.engine_kwargs = kw
        self.llm = sgl.Engine(**kw)

    def version(self):
        import sglang
        return getattr(sglang, "__version__", "unknown")

    def generate(self, texts, max_tokens):
        sp = {"temperature": 0.0, "max_new_tokens": max_tokens, "ignore_eos": self.a.ignore_eos}
        outs = self.llm.generate(texts, sp)
        if isinstance(outs, dict):
            outs = [outs]
        res = []
        for o in outs:
            meta = o.get("meta_info", {}) or {}
            ids = o.get("output_ids") or meta.get("output_ids")
            n = meta.get("completion_tokens")
            if n is None and ids is not None:
                n = len(ids)
            res.append({"ids": list(ids) if ids is not None else None, "text": o.get("text", ""),
                        "n_tokens": n, "verify_steps": meta.get("spec_verify_ct"),
                        "meta_keys": sorted(meta.keys())})
        return res

    def spec_snapshot(self):
        """Server-cumulative average accept length (coarse fallback only)."""
        for name in ("get_server_info", "get_internal_state"):
            fn = getattr(self.llm, name, None)
            if fn is None:
                continue
            try:
                info = fn()
            except Exception:
                continue
            states = info.get("internal_states") if isinstance(info, dict) else None
            if states and isinstance(states, list):
                st = states[0]
                if isinstance(st, dict) and isinstance(st.get("decode"), dict):
                    st = st["decode"]
                if isinstance(st, dict) and st.get("avg_spec_accept_length") is not None:
                    return {"avg_spec_accept_length_cumulative": float(st["avg_spec_accept_length"])}
        return None


class MockAdapter:
    """No GPU. Simulates plausible timings so the suite and analysis can be
    tested end to end. Numbers are meaningless."""
    name = "mock"

    ACCEPT_BY_CATEGORY = {"reference": 0.6, "writing": 0.45, "roleplay": 0.5, "reasoning": 0.6,
                          "math": 0.65, "coding": 0.8, "extraction": 0.75, "stem": 0.55,
                          "humanities": 0.5, "file": 0.6, "warmup": 0.5}

    def __init__(self, a):
        self.a = a
        self.engine_kwargs = {"mock": True, "method": a.method, "k": a.k,
                              "async_scheduling": not a.no_async_scheduling}
        self.category_of = {}
        self._counters = {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted": 0}
        self._rng = random.Random(1234 + a.rep)

    def build(self):
        if self.a.mock_fail:
            raise RuntimeError("mock engine refused this configuration (--mock-fail)")

    def version(self):
        return "mock-0"

    def generate(self, texts, max_tokens):
        a = self.a
        res, total_time = [], 0.0
        step_cost = 0.004  # "seconds" per target step (scaled down)
        for t in texts:
            cat = self.category_of.get(t, "warmup")
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            base_ids = [(h + i * 7919) % 32000 for i in range(max_tokens)]
            if a.method == "baseline":
                tau, draft_cost = 1.0, 0.0
            else:
                p = self.ACCEPT_BY_CATEGORY.get(cat, 0.5)
                if a.method == "eagle3":
                    p = min(0.95, p + 0.05)
                if a.method == "ngram":
                    p = p * 0.4
                k = a.k
                tau = (1 - p ** (k + 1)) / (1 - p)
                draft_cost = {"draft": 0.12, "eagle": 0.05, "eagle3": 0.05, "ngram": 0.0}[a.method] * k
            steps = max_tokens / tau
            if a.no_async_scheduling:
                draft_cost += 0.1
            elapsed = steps * step_cost * (1.0 + draft_cost) * (1 + self._rng.uniform(-0.02, 0.02))
            total_time += elapsed
            ids = list(base_ids)
            if a.method != "baseline" and max_tokens > 40 and self._rng.random() < 0.3:
                d = self._rng.randrange(20, max_tokens)
                ids = ids[:d] + [(x + 1) % 32000 for x in ids[d:]]
            if a.method != "baseline":
                n_drafts = int(round(steps))
                self._counters["num_drafts"] += n_drafts
                self._counters["num_draft_tokens"] += n_drafts * a.k
                self._counters["num_accepted"] += max(0, max_tokens - n_drafts)
            res.append({"ids": ids, "text": None, "n_tokens": max_tokens,
                        "verify_steps": int(round(steps)) if a.method != "baseline" else None,
                        "meta_keys": ["mock"]})
        time.sleep(total_time * a.mock_time_scale)
        return res

    def spec_snapshot(self):
        return dict(self._counters) if self.a.method != "baseline" else None


class FakeMonitor:
    handle = None
    poll_errors = 0

    def start(self): pass
    def stop(self): pass
    def close(self): pass
    def read_energy_j(self): return time.perf_counter() * 350.0
    def device_info(self): return {"name": "mock-gpu", "energy_counter_supported": True}
    def throttle_reasons(self): return None

    def window_stats(self, t0, t1):
        return {"mean_w": 350.0, "temp_end_c": 60, "energy_j_sampled": (t1 - t0) * 350.0}


# ============================================================================
# Measurement
# ============================================================================

def spec_delta(before, after):
    if before is None or after is None:
        return None
    if "num_drafts" in after:
        return {k: after[k] - before.get(k, 0) for k in after}
    return None


def run_batch(adapter, monitor, texts, max_tokens):
    before = adapter.spec_snapshot()
    e0 = monitor.read_energy_j()
    t0 = time.perf_counter()
    outs = adapter.generate(texts, max_tokens)
    t1 = time.perf_counter()
    e1 = monitor.read_energy_j()
    after = adapter.spec_snapshot()
    st = monitor.window_stats(t0, t1)
    energy = (e1 - e0) if (e0 is not None and e1 is not None) else st.get("energy_j_sampled")
    return outs, t1 - t0, energy, st, spec_delta(before, after), after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vllm", "sglang", "mock"], required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--k", type=int, default=5,
                    help="vLLM: num_speculative_tokens. SGLang: speculative_num_steps.")
    ap.add_argument("--topk", type=int, default=1, help="SGLang only: speculative_eagle_topk (1 = chain, >1 = tree).")
    ap.add_argument("--num-draft-tokens", type=int, default=None,
                    help="SGLang only: speculative_num_draft_tokens. Default k+1 for chain drafting, 16 for NGRAM.")
    ap.add_argument("--prompt-lookup-max", type=int, default=5, help="vLLM ngram only.")
    ap.add_argument("--prompt-lookup-min", type=int, default=2, help="vLLM ngram only.")
    ap.add_argument("--no-async-scheduling", action="store_true",
                    help="vLLM only: turn off async scheduling (vLLM already does this itself for "
                         "draft_model). Used for the 'engineering-matched' EAGLE3 runs.")
    ap.add_argument("--eagle-dir", default=DEFAULT_EAGLE_DIR)
    ap.add_argument("--eagle3-dir", default=DEFAULT_EAGLE3_DIR)
    ap.add_argument("--prompt-set", choices=["reference", "extended"], default="extended")
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL with one {\"label\", \"category\", \"prompt\"} per line. Overrides --prompt-set.")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="Rotate the prompt order by this many places (the suite uses the rep number), "
                         "so no prompt is always first.")
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Requests sent together per timed batch. 1 = single-request, like every earlier run.")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Force exactly --tokens tokens per request. Off by default: text generated past "
                         "the natural end tends to be repetitive, which flatters some methods.")
    ap.add_argument("--prefix-cache", choices=["on", "off"], default="off")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                    help="vLLM gpu_memory_utilization / SGLang mem_fraction_static.")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--warmup-seconds", type=float, default=150.0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock-time-scale", type=float, default=0.02)
    ap.add_argument("--mock-fail", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.method == "baseline":
        a.k = None

    record = {
        "status": "started",
        "started_at": datetime.now().isoformat(),
        "engine": a.engine,
        "method": a.method,
        "tag": a.tag,
        "rep": a.rep,
        "args": vars(a),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "worker_sha256": file_sha256(os.path.abspath(__file__)),
            "bench_common_sha256": file_sha256(os.path.join(HERE, "bench_common.py")),
        },
    }

    def write(rec):
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, default=_json_default)
        os.replace(tmp, a.out)

    prompts = load_prompts(a.prompt_set, a.prompts_file)
    if a.prompt_offset:
        off = a.prompt_offset % len(prompts)
        prompts = prompts[off:] + prompts[:off]
    record["prompts"] = [{"label": l, "category": c} for l, c, _ in prompts]

    if a.engine == "mock":
        monitor = FakeMonitor()
        adapter = MockAdapter(a)
        fmt = lambda texts: list(texts)
        bench_common = None
    else:
        os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        sys.path.insert(0, HERE)
        import bench_common
        import torch
        from transformers import AutoTokenizer
        import transformers
        ref_texts = [t for _, _, t in REFERENCE_PROMPTS]
        if ref_texts != list(bench_common.REFERENCE_PROMPTS):
            print("[!] REFERENCE_PROMPTS here no longer match bench_common.REFERENCE_PROMPTS -- update this file.")
            record["warning_reference_prompts_mismatch"] = True
        record["environment"].update({"torch": torch.__version__, "transformers": transformers.__version__})
        tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
        fmt = lambda texts: bench_common.vllm_chat_prompts(tok, texts)
        monitor = bench_common.NVMLPowerMonitor(device_index=0)
        adapter = VllmAdapter(a) if a.engine == "vllm" else SglangAdapter(a)
        try:
            import pynvml
            used = pynvml.nvmlDeviceGetMemoryInfo(monitor.handle).used / 1024 ** 2
            record["gpu_memory_used_mb_at_start"] = round(used, 0)
            if used > 3000:
                print(f"[!] {used:.0f} MB of GPU memory already in use before this run -- "
                      f"another process may be polluting power/energy readings.")
        except Exception:
            pass

    record["device"] = monitor.device_info()
    monitor.start()

    # ---- build engine ----------------------------------------------------
    print(f"[*] {a.engine} / {a.method} {a.tag} -- building engine...")
    try:
        adapter.build()
        record["engine_version"] = adapter.version()
        record["engine_kwargs"] = adapter.engine_kwargs
    except Exception as exc:
        record.update(status="error", stage="build", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(), engine_kwargs=adapter.engine_kwargs,
                      finished_at=datetime.now().isoformat())
        write(record)
        monitor.close()
        print(f"[!] Engine build failed: {exc}")
        sys.exit(2)

    try:
        timed_texts = fmt([t for _, _, t in prompts])
        warm_texts = fmt(WARMUP_PROMPTS)
        if a.engine == "mock":
            for (label, cat, _), txt in zip(prompts, timed_texts):
                adapter.category_of[txt] = cat

        def warm_batch(i):
            return [warm_texts[(i + j) % len(warm_texts)] for j in range(a.concurrency)]

        # ---- closed-loop warmup, on warmup prompts only ------------------
        if a.engine == "mock":
            warm = {"converged": None, "note": "mock engine, warmup skipped"}
        else:
            def step(i):
                adapter.generate(warm_batch(i), a.tokens)
                return f"warm{i % len(warm_texts)}"
            w = bench_common.warm_to_steady_state(monitor, step, min_sec=a.warmup_seconds,
                                                  max_sec=max(a.warmup_seconds * 3, bench_common.MAX_WARMUP_SEC))
            warm = {k: w[k] for k in ("converged", "elapsed_sec", "iterations")}
            warm["final_power_drift_w_by_label"] = w.get("final_power_drift_w_by_label")
            warm["final_temp_drift_c_by_label"] = w.get("final_temp_drift_c_by_label")
        record["warmup"] = warm

        # ---- timed batches: every prompt exactly once --------------------
        batches = []
        idxs = list(range(len(prompts)))
        for b0 in range(0, len(idxs), a.concurrency):
            batch_idx = idxs[b0:b0 + a.concurrency]
            # keep-warm: short, untimed, on a warmup prompt (never the timed one)
            adapter.generate(warm_batch(b0)[:1], 8)
            outs, latency, energy, st, sdelta, safter = run_batch(
                adapter, monitor, [timed_texts[i] for i in batch_idx], a.tokens)
            ntoks = [o["n_tokens"] for o in outs]
            total = sum(n for n in ntoks if n)
            vsteps = [o["verify_steps"] for o in outs]
            entry = {
                "batch_index": len(batches),
                "labels": [prompts[i][0] for i in batch_idx],
                "categories": [prompts[i][1] for i in batch_idx],
                "n_tokens": ntoks,
                "latency_sec": latency,
                "tokens_per_sec": total / latency if latency > 0 else None,
                "energy_j": energy,
                "joules_per_token": energy / total if (energy is not None and total) else None,
                "power_mean_w": st.get("mean_w"),
                "temp_end_c": st.get("temp_end_c"),
                "throttle_reasons": monitor.throttle_reasons(),
                "token_ids": [o["ids"] for o in outs],
                "texts": [o["text"] for o in outs] if all(o["ids"] is None for o in outs) else None,
                "spec_counters_delta": sdelta,
                "spec_server_state_after": safter if (safter and "num_drafts" not in safter) else None,
                "verify_steps": vsteps,
            }
            # accept length = output tokens per target verification step
            tau, tau_source = None, None
            if a.method == "baseline":
                pass  # no speculation -> accept length not applicable
            elif all(v for v in vsteps) and total:
                tau, tau_source = total / sum(vsteps), "per_request_verify_steps"
            elif sdelta and sdelta.get("num_drafts"):
                tau = (sdelta["num_accepted"] + sdelta["num_drafts"]) / sdelta["num_drafts"]
                tau_source = "counter_delta"
            elif entry["spec_server_state_after"]:
                tau = entry["spec_server_state_after"]["avg_spec_accept_length_cumulative"]
                tau_source = "server_cumulative_coarse"
            entry["accept_length"] = tau
            entry["accept_length_source"] = tau_source
            if len(batches) == 0:
                record["first_request_meta_keys"] = outs[0].get("meta_keys")
                print(f"    meta keys of first request: {outs[0].get('meta_keys')}")
            batches.append(entry)
            tau_s = f"  accept_len={tau:.2f} ({tau_source})" if tau else ""
            print(f"  [{len(batches):>3}/{-(-len(idxs) // a.concurrency)}] {','.join(entry['labels'])[:40]:<40} "
                  f"{entry['tokens_per_sec']:.1f} tok/s  J/tok={entry['joules_per_token']}{tau_s}")

        record["batches"] = batches

        # drift over the run (chronological), same statistic as the rest of the repo
        if bench_common is not None and len(batches) >= 3:
            idx = list(range(len(batches)))
            record["drift"] = {
                "power_w": bench_common.fit_drift(idx, [b["power_mean_w"] for b in batches]),
                "temp_c": bench_common.fit_drift(idx, [b["temp_end_c"] for b in batches]),
            }
        record["status"] = "ok"
    except Exception as exc:
        record.update(status="error", stage="run", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        print(f"[!] Run failed: {exc}")
    finally:
        record["poll_errors"] = getattr(monitor, "poll_errors", None)
        record["finished_at"] = datetime.now().isoformat()
        write(record)
        monitor.close()
        shutdown = getattr(getattr(adapter, "llm", None), "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    print(f"[*] Wrote {a.out} (status={record['status']})")
    sys.exit(0 if record["status"] == "ok" else 3)


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
