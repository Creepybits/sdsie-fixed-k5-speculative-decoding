"""
likeforlike_suite.py

Runs the full like-for-like comparison: every configuration of every method,
on vLLM and/or SGLang, repeated --reps times, each run in its own fresh
process (see likeforlike_worker.py for the design rules). Then runs the
analysis.

Written 2026-09-24.

WHAT IT DOES, IN ORDER
  1. Builds a plan: for each repetition and each engine, a baseline run, then
     every other configuration in a RANDOM order (seed recorded), then a
     closing baseline run. The two baselines bracket the block, so drift
     between the start and end of a block is measured, not assumed away.
     Engine blocks alternate order between repetitions.
  2. Launches each run as a separate process with the right Python for that
     engine (vLLM and SGLang live in different venvs -- pass their paths).
  3. After each run, waits until GPU memory is released before starting the
     next one.
  4. Skips runs that already finished (status ok) -- so an interrupted
     overnight suite can be resumed by re-running the same command with
     --resume <suite folder>.
  5. Runs likeforlike_analyze.py on the folder at the end.

Everything goes into telemetry/likeforlike/<timestamp>/ -- new files only,
nothing earlier is overwritten.

TYPICAL USE
  # 1. Plumbing test, no GPU, ~1 minute:
  python likeforlike_suite.py --engines mock --preset quick --reps 2

  # 2. Short real smoke test (reference prompts, few configs):
  python likeforlike_suite.py --engines vllm,sglang --preset quick \\
      --vllm-python ~/sdsie/repos/.vllmvenv/bin/python \\
      --sglang-python ~/sdsie/repos/.sglangvenv/bin/python

  # 3. The real comparison (several hours -- overnight):
  python likeforlike_suite.py --engines vllm,sglang --preset full --reps 2 \\
      --vllm-python ... --sglang-python ...

  (The venv paths above are examples -- use your actual ones.)
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
WORKER = os.path.join(HERE, "likeforlike_worker.py")
ANALYZE = os.path.join(HERE, "likeforlike_analyze.py")

# Configuration grids. Each entry: (method, {worker options}).
# "K5" is method=draft with k=5 (and topk=1 on SGLang).
# Starting points, not gospel: the SGLang EAGLE3 tree preset (steps 5, topk 8,
# 32 draft tokens) is the kind of setting SGLang's docs show for this model
# family -- check it against the docs for the installed SGLang version.
GRIDS = {
    "full": {
        "vllm": [
            ("draft", {"k": 3}), ("draft", {"k": 5}), ("draft", {"k": 7}),
            ("eagle3", {"k": 2}), ("eagle3", {"k": 3}), ("eagle3", {"k": 5}),
            ("eagle", {"k": 3}), ("eagle", {"k": 5}),
            ("ngram", {"k": 5}),
            # engineering-matched EAGLE3 (async scheduling off, like draft_model)
            ("eagle3", {"k": 3, "no_async_scheduling": True}),
            ("eagle3", {"k": 5, "no_async_scheduling": True}),
        ],
        "sglang": [
            ("draft", {"k": 3}), ("draft", {"k": 5}), ("draft", {"k": 7}),
            ("eagle3", {"k": 3}), ("eagle3", {"k": 5}),
            ("eagle3", {"k": 5, "topk": 8, "num_draft_tokens": 32}),
            ("eagle", {"k": 5}),
            ("ngram", {"num_draft_tokens": 16}),
        ],
        "mock": [
            ("draft", {"k": 3}), ("draft", {"k": 5}), ("draft", {"k": 7}),
            ("eagle3", {"k": 3}), ("eagle3", {"k": 5}),
            ("ngram", {"k": 5}),
        ],
    },
    "quick": {
        "vllm": [("draft", {"k": 5}), ("eagle3", {"k": 3}),
                 ("eagle3", {"k": 3, "no_async_scheduling": True})],
        "sglang": [("draft", {"k": 5}), ("eagle3", {"k": 3})],
        "mock": [("draft", {"k": 5}), ("eagle3", {"k": 3}), ("eagle3", {"k": 5}),
                 ("eagle3", {"k": 3, "no_async_scheduling": True})],
    },
}

PRESET_DEFAULTS = {"full": {"prompt_set": "extended"}, "quick": {"prompt_set": "reference"}}


def config_tag(method, opts):
    if method == "baseline":
        return "baseline"
    parts = [method]
    if "k" in opts:
        parts.append(f"k{opts['k']}")
    if opts.get("topk", 1) != 1:
        parts.append(f"topk{opts['topk']}")
    if "num_draft_tokens" in opts:
        parts.append(f"ndt{opts['num_draft_tokens']}")
    if opts.get("no_async_scheduling"):
        parts.append("matched")
    return "_".join(parts)


def build_plan(engines, preset, reps, seed):
    rng = random.Random(seed)
    plan = []
    for rep in range(reps):
        order = list(engines) if rep % 2 == 0 else list(reversed(engines))
        for eng in order:
            configs = list(GRIDS[preset][eng])
            rng.shuffle(configs)
            block = [("baseline", {}, "open")] + [(m, o, "") for m, o in configs] + [("baseline", {}, "close")]
            for pos, (m, o, role) in enumerate(block):
                tag = config_tag(m, o)
                plan.append({
                    "rep": rep, "engine": eng, "method": m, "opts": o, "tag": tag,
                    "role": role, "position": pos,
                    "run_id": f"r{rep:02d}_{eng}_{pos:02d}_{tag}" + (f"_{role}" if role else ""),
                })
    return plan


def gpu_used_mb():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 ** 2
    except Exception:
        return None


def wait_for_gpu(threshold_mb, timeout_s):
    t0 = time.time()
    while True:
        used = gpu_used_mb()
        if used is None or used < threshold_mb:
            return used
        if time.time() - t0 > timeout_s:
            print(f"[!] GPU still shows {used:.0f} MB in use after {timeout_s}s -- continuing anyway.")
            return used
        time.sleep(3)


def run_status(path):
    try:
        with open(path) as f:
            return json.load(f).get("status")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="vllm,sglang", help="Comma list of: vllm, sglang, mock")
    ap.add_argument("--preset", choices=list(GRIDS), default="quick")
    ap.add_argument("--reps", type=int, default=2,
                    help="Repetitions of every run, each in a fresh process. 2 is the minimum that "
                         "says anything about run-to-run variation; 3 is better.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for run order (recorded; random if omitted).")
    ap.add_argument("--vllm-python", default=sys.executable)
    ap.add_argument("--sglang-python", default=sys.executable)
    ap.add_argument("--resume", default=None, help="Existing suite folder to continue.")
    ap.add_argument("--prompt-set", choices=["reference", "extended"], default=None)
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--prefix-cache", choices=["on", "off"], default="off")
    ap.add_argument("--warmup-seconds", type=float, default=150.0)
    ap.add_argument("--timeout-per-run", type=float, default=3600.0)
    ap.add_argument("--gpu-free-mb", type=float, default=3000.0,
                    help="Wait until GPU memory in use drops below this between runs "
                         "(WSL2's own display stack holds ~1-1.5 GB at idle).")
    ap.add_argument("--mock-fail-one", action="store_true",
                    help="Mock only: make one run fail on purpose, to test that failures are handled.")
    a = ap.parse_args()

    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    python_for = {"vllm": a.vllm_python, "sglang": a.sglang_python, "mock": sys.executable}

    if a.resume:
        suite_dir = os.path.abspath(a.resume)
        with open(os.path.join(suite_dir, "plan.json")) as f:
            saved = json.load(f)
        plan, settings = saved["plan"], saved["settings"]
        print(f"[*] Resuming {suite_dir}")
    else:
        seed = a.seed if a.seed is not None else random.randrange(1_000_000)
        prompt_set = a.prompt_set or PRESET_DEFAULTS[a.preset]["prompt_set"]
        settings = {"engines": engines, "preset": a.preset, "reps": a.reps, "seed": seed,
                    "prompt_set": prompt_set, "prompts_file": a.prompts_file, "tokens": a.tokens,
                    "concurrency": a.concurrency, "prefix_cache": a.prefix_cache,
                    "warmup_seconds": a.warmup_seconds, "created_at": datetime.now().isoformat()}
        plan = build_plan(engines, a.preset, a.reps, seed)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_dir = os.path.join(REPO_ROOT, "telemetry", "likeforlike", f"{stamp}_{a.preset}")
        os.makedirs(os.path.join(suite_dir, "runs"), exist_ok=True)
        os.makedirs(os.path.join(suite_dir, "logs"), exist_ok=True)
        with open(os.path.join(suite_dir, "plan.json"), "w") as f:
            json.dump({"settings": settings, "plan": plan}, f, indent=1)
        print(f"[*] Suite folder: {suite_dir}")

    used = gpu_used_mb()
    if used is not None and used > a.gpu_free_mb:
        print(f"[!] {used:.0f} MB of GPU memory already in use before the suite starts. "
              f"If another program is using the GPU, stop it -- it will pollute every energy reading.")

    n = len(plan)
    t_suite = time.time()
    failed_once = False
    for i, run in enumerate(plan, 1):
        out = os.path.join(suite_dir, "runs", run["run_id"] + ".json")
        if run_status(out) == "ok":
            print(f"[{i}/{n}] {run['run_id']} -- already done, skipping")
            continue
        cmd = [python_for[run["engine"]], WORKER,
               "--engine", run["engine"], "--method", run["method"],
               "--rep", str(run["rep"]), "--prompt-offset", str(run["rep"] * 7),
               "--tag", run["tag"], "--out", out,
               "--tokens", str(settings["tokens"]), "--concurrency", str(settings["concurrency"]),
               "--prefix-cache", settings["prefix_cache"],
               "--warmup-seconds", str(settings["warmup_seconds"])]
        if settings.get("prompts_file"):
            cmd += ["--prompts-file", settings["prompts_file"]]
        else:
            cmd += ["--prompt-set", settings["prompt_set"]]
        for key, val in run["opts"].items():
            if val is True:
                cmd += ["--" + key.replace("_", "-")]      # on/off switch, no value
            elif val is not False:
                cmd += ["--" + key.replace("_", "-"), str(val)]
        if run["engine"] == "mock" and a.mock_fail_one and not failed_once and run["method"] == "eagle3":
            cmd += ["--mock-fail"]
            failed_once = True

        elapsed = time.time() - t_suite
        print(f"\n[{i}/{n}] {run['run_id']}   (suite running {elapsed / 60:.1f} min)")
        log_path = os.path.join(suite_dir, "logs", run["run_id"] + ".log")
        t0 = time.time()
        with open(log_path, "w") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, cwd=HERE, bufsize=1)
                for line in proc.stdout:
                    log.write(line)
                    if line.startswith(("[*]", "[!]", "  [")) or "Steady state" in line:
                        print("    " + line.rstrip())
                    if time.time() - t0 > a.timeout_per_run:
                        proc.kill()
                        print(f"[!] Timed out after {a.timeout_per_run:.0f}s -- killed.")
                        break
                proc.wait()
            except Exception as exc:
                print(f"[!] Could not launch: {exc}")
        status = run_status(out)
        print(f"    -> status={status}  ({(time.time() - t0) / 60:.1f} min)  log: {log_path}")
        if status is None:
            with open(out, "w") as f:
                json.dump({"status": "error", "stage": "process", "engine": run["engine"],
                           "method": run["method"], "tag": run["tag"], "rep": run["rep"],
                           "error": "worker exited without writing results -- see log"}, f)
        if run["engine"] != "mock":
            wait_for_gpu(a.gpu_free_mb, timeout_s=180)

    print(f"\n[*] Suite finished in {(time.time() - t_suite) / 60:.1f} min. Running analysis...")
    subprocess.run([sys.executable, ANALYZE, suite_dir], cwd=HERE)


if __name__ == "__main__":
    main()
