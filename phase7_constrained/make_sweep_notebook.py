"""
make_sweep_notebook.py - generate wordle_phase7_sweep_kaggle.ipynb.

Phase 7 used ADAPTIVE_THRESHOLD = 50, which was a guess. The two endpoints
disagree about what is best:

    consistent (always filter)  3.9472 mean, 241/246 solved
    adaptive   (filter <= 50)   3.7846 mean, 239/246 solved

Fewer guesses but two more losses. That is a frontier, not a winner, and it is
worth mapping. This notebook sweeps the threshold and reports both axes, plus
the no-model control at every point so the model's contribution stays separable.

No training. Loads the Phase 7 adapter and runs evaluations only. ~45 min.

    python make_sweep_notebook.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_phase7_sweep_kaggle.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


md(r"""
# Phase 7b — adaptive-threshold sweep

`ADAPTIVE_THRESHOLD` decides when the decoder stops probing and starts
restricting itself to feedback-consistent words. Phase 7 used **50** on the
reasoning that admissible-set medians go 12,972 → ~211 → ~7 → ~2, so 50 sits
between "still probing" and "resolving". That was a reasonable guess, not a
measurement.

The endpoints disagree:

| threshold | mean | solved |
|---|---:|---:|
| ∞ (always filter) | 3.9472 | **241** |
| 50 | **3.7846** | 239 |

Filtering less wins **faster** but loses **more**. That is a trade-off curve and
this maps it.

## Also swept: the no-model control

At every threshold the same games are played by picking **uniformly at random
among admissible words**. Without that, a threshold that merely narrows the
admissible set would look like a model improvement. Phase 7 measured the two
endpoints as 4.5203 (always filter) and 5.0027 (threshold 50) — the model's
margin was **−0.57** and **−1.22** respectively, so the margin itself varies
with the threshold and needs its own curve.

No training. ~45 min.

---

## How to run

1. **GPU T4 x2**
2. Add Input: the v2 SFT package **and** the Phase 7 adapter dataset
3. Set `PREV_RUN_DIR` to the adapter path
4. Run All, download `wordle_sweep_results.zip`
""")

md(r"""
---
# 1. Setup
""")

code(r'''
import os, sys, json, time, random, subprocess, importlib, shutil, glob
from collections import Counter

def _pip(p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
for mod, pkg in [("transformers", "transformers>=4.44"), ("peft", "peft>=0.11"),
                 ("accelerate", "accelerate>=0.30")]:
    try: importlib.import_module(mod)
    except ImportError: _pip(pkg)
import torch, transformers, peft
import numpy as np

def fix_torchao_peft_conflict():
    try: import peft.import_utils as piu
    except Exception as e: return f"unavailable ({e})"
    try: piu.is_torchao_available(); return "no conflict"
    except ImportError: pass
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                   check=False)
    importlib.invalidate_caches()
    try: piu.is_torchao_available(); return "resolved"
    except ImportError: pass
    piu.is_torchao_available = lambda *a, **k: False
    try:
        import peft.tuners.lora.torchao as t
        t.is_torchao_available = lambda *a, **k: False
    except Exception: pass
    return "patched"
print("torchao:", fix_torchao_peft_conflict())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# ============================ EDIT THIS =====================================
DATASET_DIR  = None
PREV_RUN_DIR = None      # the Phase 7 adapter dataset
# ============================================================================

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_NAME = "tree_salet_endgame"
MAX_GUESSES, SEED = 6, 20260817
CONSTRAINED_CHUNK, CONSTRAINED_PRUNE, LENGTH_NORMALISE = 512, True, False

# 0 = never filter (pure legal-word decoding)
# a huge value = always filter (the "consistent" decoder)
THRESHOLDS = [0, 2, 5, 10, 20, 50, 100, 250, 1000, 10**9]
CONTROL_SEEDS = 3          # no-model random-admissible runs per threshold

RESULTS_ROOT = "/kaggle/working/results_sweep"
RESULTS_ZIP  = "/kaggle/working/wordle_sweep_results.zip"
os.makedirs(RESULTS_ROOT, exist_ok=True)

PHASE7 = {"consistent": 3.9472, "adaptive50": 3.7846,
          "control_consistent": 4.5203, "control_adaptive50": 5.0027,
          "classical_entropy": 3.4431, "classical_random": 4.0203,
          "classical_frequency": 3.7927}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("thresholds:", THRESHOLDS)
''')

md(r"""
---
# 2. Dataset and adapter
""")

code(r'''
REQ = ["sft_package/eval/val_answers.jsonl", "code/wordle_solver.py",
       "code/generate_trajectories.py", "artifacts/feedback_matrix.npy",
       "artifacts/answers.txt", "artifacts/valid_guesses.txt"]

def _has(d):
    try: return all(os.path.exists(os.path.join(d, f)) for f in REQ)
    except OSError: return False

def find_root():
    for root in ([DATASET_DIR] if DATASET_DIR else []) + ["/kaggle/input", "."]:
        if not root or not os.path.isdir(root): continue
        if _has(root): return root
        for dp, dn, _ in os.walk(root):
            dn[:] = [d for d in dn if not d.startswith(".")]
            if _has(dp): return dp
    raise FileNotFoundError("SFT package not found under /kaggle/input")

DATA_ROOT = find_root()
SFT_DIR = os.path.join(DATA_ROOT, "sft_package")
sys.path.insert(0, os.path.join(DATA_ROOT, "code"))
print("dataset:", DATA_ROOT)

from wordle_solver import (load_artifacts, SolverConfig, make_solver, play_game,
                           feedback_code, code_to_pattern, ALL_GREEN)
from generate_trajectories import derive_constraints, render_prompt

BUNDLE = load_artifacts(os.path.join(DATA_ROOT, "artifacts"), mmap=True)
VOCAB = BUNDLE.vocab
LEGAL_LOWER = [g.lower() for g in VOCAB.guesses]
LEGAL_GUESSES = set(w.upper() for w in LEGAL_LOWER)
VAL_ANSWERS = [json.loads(l)["answer"].upper()
               for l in open(os.path.join(SFT_DIR, "eval/val_answers.jsonl"),
                             encoding="utf-8")]
assert len(LEGAL_GUESSES) == 12972 and len(VAL_ANSWERS) == 246

def find_adapter():
    for base in ([PREV_RUN_DIR] if PREV_RUN_DIR else []) + ["/kaggle/input",
                                                            "/kaggle/working"]:
        if not base or not os.path.isdir(base): continue
        for dp, _, fs in os.walk(base):
            if "adapter_config.json" in fs and "checkpoint-" not in dp:
                return dp
    raise FileNotFoundError(
        "No adapter found. Attach the Phase 7 adapter dataset and set "
        "PREV_RUN_DIR, or re-run the Phase 7 notebook to train one.")

ADAPTER_DIR = find_adapter()
print("adapter:", ADAPTER_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if TOKENIZER.pad_token is None: TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "left"

def load_model():
    b = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                             trust_remote_code=True)
    b.config.use_cache = False
    m = PeftModel.from_pretrained(b, ADAPTER_DIR); m.config.use_cache = True
    return m.eval().cuda()
''')

md(r"""
---
# 3. Decoder
""")

with open(os.path.join(_paths.CORE, "constrained_decode.py"), encoding="utf-8") as fh:
    SRC = fh.read().rstrip()

code(SRC + r'''


LEGAL_WORDS_SORTED = sorted(LEGAL_GUESSES)
SCORER = None
def build_scorer():
    global SCORER
    if SCORER is None:
        SCORER = LegalWordScorer(TOKENIZER, LEGAL_WORDS_SORTED, device="cuda",
                                 chunk=CONSTRAINED_CHUNK,
                                 length_normalise=LENGTH_NORMALISE)
    return SCORER

class GameState:
    __slots__ = ("answer","history","cands","guesses","patterns","remaining",
                 "forced","n_allowed","done","solved","filt","win_forced")
    def __init__(self, answer, n_answers, filt):
        self.answer, self.filt = answer, filt
        self.history = []; self.cands = np.arange(n_answers, dtype=np.int32)
        self.guesses, self.patterns, self.remaining = [], [], []
        self.forced, self.n_allowed = [], []
        self.done = self.solved = False; self.win_forced = None
    def prompt(self, turn):
        h = [(g.lower(), p) for g, p in self.history]
        return render_prompt(turn=turn, history=h,
                             constraints=derive_constraints(h),
                             n_candidates=len(self.cands),
                             guesses_remaining=MAX_GUESSES-turn+1,
                             max_guesses=MAX_GUESSES,
                             candidates=None, show_candidate_count=False)

@torch.no_grad()
def play(model, scorer, answers, threshold):
    """threshold: filter only when |admissible| <= threshold. 0 = never."""
    games = [GameState(a, VOCAB.n_answers,
                       HardModeFilter(LEGAL_LOWER, feedback_code)) for a in answers]
    t0 = time.perf_counter()
    for turn in range(1, MAX_GUESSES+1):
        active = [g for g in games if not g.done]
        if not active: break
        for g in active:
            allowed = g.filt.indices(scorer) if len(g.filt) <= threshold else None
            r = scorer.select(model, g.prompt(turn), allowed_idx=allowed,
                              prune=CONSTRAINED_PRUNE)
            w = r["word"]
            g.forced.append(r["forced"]); g.n_allowed.append(r["n_allowed"])
            c = feedback_code(w.lower(), g.answer.lower())
            g.cands = BUNDLE.fb.filter_indices(g.cands, w.lower(), c)
            g.filt.refine(w.lower(), c)
            assert g.filt.contains(g.answer), "answer left the admissible set"
            g.history.append((w, code_to_pattern(c)))
            g.guesses.append(w); g.patterns.append(code_to_pattern(c))
            g.remaining.append(int(len(g.cands)))
            if c == ALL_GREEN:
                g.solved = g.done = True; g.win_forced = r["forced"]
            elif turn == MAX_GUESSES:
                g.done = True
    return games, time.perf_counter()-t0

def summarize(games, threshold, secs):
    n = len(games)
    sc = [len(g.guesses) if g.solved else MAX_GUESSES+1 for g in games]
    solved = [len(g.guesses) for g in games if g.solved]
    hv = tv = 0
    for g in games:
        seen = []
        for gu, pat in zip(g.guesses, g.patterns):
            tv += 1; greens = {}
            for pg, pp in seen:
                for i,(ch,t) in enumerate(zip(pg,pp)):
                    if t == "G": greens[i] = ch
            if any(gu[i] != ch for i,ch in greens.items()): hv += 1
            seen.append((gu,pat))
    dec = [f for g in games for f in g.forced]
    return {"threshold": threshold, "n_games": n,
            "mean": round(sum(sc)/n, 4), "solved": len(solved),
            "failure_rate_pct": round(100*(n-len(solved))/n, 2),
            "mean_solved_only": round(sum(solved)/len(solved), 4) if solved else None,
            "hard_mode_violation_pct": round(100*hv/max(tv,1), 2),
            "forced_decisions": sum(1 for f in dec if f), "decisions": len(dec),
            "forced_pct": round(100*sum(1 for f in dec if f)/max(len(dec),1), 2),
            "wins_forced": sum(1 for g in games if g.solved and g.win_forced),
            "wins_model": sum(1 for g in games if g.solved and g.win_forced is False),
            "eval_seconds": round(secs, 1)}
print("decoder ready")
''')

md(r"""
---
# 4. The no-model control

Same games, same thresholds, picking uniformly at random among admissible
words. Pure CPU, no GPU, and it runs first so the model curve always has
something to be measured against.
""")

code(r'''
_cache = {}
def after_first(guess, code):
    k = (guess, code)
    if k not in _cache:
        _cache[k] = [w for w in LEGAL_LOWER if feedback_code(guess, w) == code]
    return _cache[k]

def play_random(ans, rng, threshold, opener="salet"):
    g, allowed = opener, LEGAL_LOWER
    for turn in range(1, MAX_GUESSES+1):
        c = feedback_code(g, ans)
        if c == ALL_GREEN: return turn
        allowed = after_first(g, c) if turn == 1 else \
                  [w for w in allowed if feedback_code(g, w) == c]
        if not allowed: return MAX_GUESSES+1
        g = (allowed[rng.randrange(len(allowed))] if len(allowed) <= threshold
             else LEGAL_LOWER[rng.randrange(len(LEGAL_LOWER))])
    return MAX_GUESSES+1

CONTROL = {}
low = [a.lower() for a in VAL_ANSWERS]
print(f"{'thr':>10}{'mean':>9}{'+/-':>7}{'fail%':>8}")
for thr in THRESHOLDS:
    ms, fs = [], []
    for s in range(CONTROL_SEEDS):
        rng = random.Random(1000+s)
        sc = [play_random(a, rng, thr) for a in low]
        ms.append(sum(sc)/len(sc)); fs.append(100*sum(1 for x in sc if x > MAX_GUESSES)/len(sc))
    CONTROL[thr] = {"mean": round(float(np.mean(ms)), 4),
                    "std": round(float(np.std(ms)), 4),
                    "failure_rate_pct": round(float(np.mean(fs)), 2)}
    print(f"{thr:>10}{CONTROL[thr]['mean']:>9.4f}{CONTROL[thr]['std']:>7.3f}"
          f"{CONTROL[thr]['failure_rate_pct']:>8.1f}")
json.dump(CONTROL, open(os.path.join(RESULTS_ROOT, "control.json"), "w"), indent=2)
''')

md(r"""
---
# 5. The sweep
""")

code(r'''
SWEEP = {}
path = os.path.join(RESULTS_ROOT, "sweep.json")
if os.path.exists(path):
    SWEEP = {int(k): v for k, v in json.load(open(path)).items()}
    print("resumed:", sorted(SWEEP))

MODEL = None
for thr in THRESHOLDS:
    if thr in SWEEP:
        print(f"threshold {thr}: done, skipping"); continue
    if MODEL is None:
        MODEL = load_model(); SC = build_scorer()
    print(f"--- threshold {thr} ---", flush=True)
    games, secs = play(MODEL, SC, VAL_ANSWERS, thr)
    SWEEP[thr] = summarize(games, thr, secs)
    r = SWEEP[thr]
    print(f"  mean={r['mean']:.4f}  solved={r['solved']}  "
          f"fail={r['failure_rate_pct']:.1f}%  hardviol={r['hard_mode_violation_pct']:.1f}%  "
          f"forced={r['forced_pct']:.1f}%  ({secs:.0f}s)")
    json.dump(SWEEP, open(path, "w"), indent=2)
if MODEL is not None:
    del MODEL; torch.cuda.empty_cache()
''')

md(r"""
---
# 6. The frontier
""")

code(r'''
import csv
print("=" * 92)
print("ADAPTIVE THRESHOLD SWEEP")
print("=" * 92)
h = (f"{'threshold':>10}{'mean':>9}{'solved':>8}{'fail%':>7}{'hardviol%':>11}"
     f"{'forced%':>9}{'control':>9}{'model gain':>12}")
print(h); print("-"*len(h))
best_mean = best_solved = None
for thr in THRESHOLDS:
    r = SWEEP.get(thr); c = CONTROL.get(thr)
    if not r: continue
    gain = (c["mean"] - r["mean"]) if c else float("nan")
    print(f"{thr:>10}{r['mean']:>9.4f}{r['solved']:>8}{r['failure_rate_pct']:>7.1f}"
          f"{r['hard_mode_violation_pct']:>11.1f}{r['forced_pct']:>9.1f}"
          f"{(c['mean'] if c else float('nan')):>9.4f}{gain:>12.4f}")
    if best_mean is None or r["mean"] < SWEEP[best_mean]["mean"]: best_mean = thr
    if best_solved is None or r["solved"] > SWEEP[best_solved]["solved"]: best_solved = thr

print(f"\nreference: classical entropy {PHASE7['classical_entropy']:.4f}, "
      f"frequency {PHASE7['classical_frequency']:.4f}, "
      f"random {PHASE7['classical_random']:.4f}")
if best_mean is not None:
    b = SWEEP[best_mean]
    print(f"\nBEST MEAN     threshold {best_mean}: {b['mean']:.4f} "
          f"({b['solved']} solved)   Phase 7 @50 was {PHASE7['adaptive50']:.4f}")
if best_solved is not None:
    b = SWEEP[best_solved]
    print(f"BEST SOLVED   threshold {best_solved}: {b['solved']}/246 "
          f"(mean {b['mean']:.4f})")
print("""
Reading it: low threshold = probe more (fewer guesses when it works, more
losses); high threshold = always play a consistent word (safer, slower). The
'model gain' column is the whole point -- it is how much the model beats a
random admissible pick at that threshold, and if it collapses toward zero the
decoder is doing the work, not the model.""")

with open(os.path.join(RESULTS_ROOT, "sweep.csv"), "w", newline="",
          encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["threshold","mean","solved","failure_rate_pct",
                "hard_mode_violation_pct","forced_pct","wins_forced","wins_model",
                "control_mean","model_gain"])
    for thr in THRESHOLDS:
        r = SWEEP.get(thr); c = CONTROL.get(thr)
        if not r: continue
        w.writerow([thr, r["mean"], r["solved"], r["failure_rate_pct"],
                    r["hard_mode_violation_pct"], r["forced_pct"],
                    r["wins_forced"], r["wins_model"],
                    c["mean"] if c else "", round(c["mean"]-r["mean"], 4) if c else ""])

shutil.make_archive(RESULTS_ZIP[:-4], "zip", RESULTS_ROOT)
print(f"\nZIP: {RESULTS_ZIP} ({os.path.getsize(RESULTS_ZIP)/2**20:.2f} MiB)")
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}")
print(f"  {len(cells)} cells ({len(cells)-n_code} markdown, {n_code} code)")
print(f"  {os.path.getsize(OUT)/1024:.1f} KiB")
