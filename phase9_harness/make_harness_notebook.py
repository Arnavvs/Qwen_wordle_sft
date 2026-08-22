"""
make_harness_notebook.py - generate wordle_phase9_harness_kaggle.ipynb.

THE QUESTION

Eight phases of work have moved the mean from 3.79 (SFT+decoder, Phase 6) to
3.76 (Phase 7) to 3.79 (DPO v3, a null). The one intervention that ever mattered
was the decoder: 5.29 -> 3.76, about 1.5 guesses. That is scaffolding, not
learning.

So the honest next experiment is not more training. It is: how much of the
remaining performance is the OTHER piece of scaffolding - the prompt? This
notebook holds the model and the decoder fixed and varies only the text.

TWO PROBES

  A. GAME SWEEP - every variant plays the same held-out answers under the same
     adaptive@20 decoder, on base Qwen and on the SFT adapter. Because the
     answers are shared, variant-vs-baseline is a PAIRED comparison.

  B. FORMAT PROBE - the same variants, no decoder at all, raw greedy generation
     on a fixed set of states. Measures what the model emits when nothing
     catches it: legal-word rate and admissible rate. A prompt can help the
     model without the decoder and be invisible in probe A, or vice versa.

WHAT WOULD CHANGE MY MIND ABOUT THE PROJECT

`raw_history` is the load-bearing variant. Every phase so far has fed the model
`Confirmed letters : p1=A`, computed by the classical solver. If the model does
as well without it, the derived block was decoration. If it collapses, then a
large part of what we have been calling "the model playing Wordle" is the
harness playing Wordle, and the write-up has to say so.

LEAKAGE

`with_count` shows the surviving-candidate count, which a player cannot see.
It is marked leaky, printed in a separate section, and must never be quoted as
a fair result or used to pick a training format.

    python make_harness_notebook.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_phase9_harness_kaggle.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# ===========================================================================
md(r"""
# Phase 9 — does the harness matter more than the model?

## Where this starts

| | mean | solved | note |
|---|---:|---:|---|
| classical entropy solver | 3.4431 | 246/246 | the ceiling |
| SFT + adaptive decoder @20 | **3.7642** | 242/246 | Phase 7, the baseline |
| DPO v3 + same decoder | 3.7886 | 241/246 | a clean null |
| filter only, no model | 5.1762 | 190/246 | decoder with no policy |
| SFT, no decoder | ~5.29 | — | policy with no decoder |

Read the last two rows together. The decoder without a model gets 5.18; the
model without the decoder gets ~5.29; together they get 3.76. Almost all of the
gain is in the *combination*, and the piece nobody has varied is the prompt.

## What runs here

**A. Game sweep** — 12 prompt variants × {base Qwen, SFT} × the same answers,
adaptive@20 decoder throughout. Shared answers means variant-vs-baseline is
paired, so a 0.05 difference is readable where an unpaired SE of 0.064 would
hide it.

**B. Format probe** — the same variants with *no decoder*, raw greedy
generation. Measures legal-word rate and admissible rate: what the model does
when nothing catches it.

## The twelve

| # | variant | what it tests |
|---|---|---|
| 1 | `baseline` | what every earlier phase used |
| 2 | `raw_history` | **no derived constraints — does the model deduce at all?** |
| 3 | `constraints_only` | deductions without the moves that produced them |
| 4 | `minimal` | no instructions whatsoever |
| 5 | `with_count` | 🚩 LEAKY upper bound — shows candidates remaining |
| 6 | `emoji` | 🟩🟨⬛ vs `GYB` — pure notation |
| 7 | `verbose` | feedback spelled out per letter, in prose |
| 8 | `reversed` | most recent guess first (recency vs chronology) |
| 9 | `keyboard` | the letter-status board a human actually reads |
| 10 | `few_shot` | two worked examples first |
| 11 | `guided_prose` | the same constraints as a sentence, not a table |
| 12 | `hard_mode_hint` | states the rule the decoder already enforces |

## Reading the result before it exists

- **Spread < 0.05 across variants** → prompt format is not a lever; stop here
  and go to GRPO or accept the ceiling.
- **Spread > 0.15** → format matters more than four training interventions did.
  The next SFT run should be on the winning format, and the write-up's headline
  changes.
- **`raw_history` ≈ `baseline`** → the derived block is decoration; the model is
  reading the history itself.
- **`raw_history` ≫ worse** → the harness has been doing the deduction. Say so.

Nothing here trains anything. It is measurement only.
""")

# ===========================================================================
md(r"""
---
# 1. Setup
""")

code(r'''
import os, sys, json, time, math, random, subprocess, importlib, shutil, glob
import itertools
from collections import Counter, defaultdict

def _pip(p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)
for mod, pkg in [("transformers", "transformers>=4.44"), ("peft", "peft>=0.11"),
                 ("accelerate", "accelerate>=0.30")]:
    try: importlib.import_module(mod)
    except ImportError: _pip(pkg)
import torch, transformers, peft
import torch.nn.functional as F
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
print("transformers", transformers.__version__, "| peft", peft.__version__)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# ============================ EDIT THIS =====================================
DATASET_DIR  = None      # sft_package dataset; None = search /kaggle/input
PREV_RUN_DIR = None      # dir holding the Phase 7 SFT adapter
# ============================================================================

MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER = "tree_salet_endgame"

# ---- what to run ----------------------------------------------------------
RUN_GAMES   = True       # probe A
RUN_FORMAT  = True       # probe B (cheap: ~2 min for all 12)
ARMS        = ["sft", "base"]     # order matters; sft first so the baseline
                                  # sanity-check happens in the first 10 min

# The full 246 costs ~12x the sweep. 100 games keeps paired detection at ~0.06
# guesses while fitting 12 variants x 2 models in one session. Set 246 for the
# final numbers once the interesting variants are known.
N_GAMES     = 100
GAME_SEED   = 20260822   # which 100 of the 246; fixed so arms are comparable

VARIANTS_TO_RUN = None   # None = all 12; or e.g. ["baseline","raw_history"]

# ---- decoder: FIXED. This is the control, not a variable. -----------------
ADAPTIVE_THRESHOLD = 20
CONSTRAINED_CHUNK, CONSTRAINED_PRUNE = 512, True
ATTN_IMPL = "sdpa"           # eager is 5-10x slower and silently so
USE_DECISION_CACHE = True    # exact: greedy decoder + deterministic model

# ---- probe B --------------------------------------------------------------
FORMAT_PROBE_STATES = 150
FORMAT_MAX_NEW = 8

SEED = 20260822
MAX_GUESSES = 6

PHASE7_OPENER = "SALET"
PHASE7_MEAN   = 3.7642     # baseline+sft must land near this or the run is void
PHASE7 = {"sft_adaptive20": 3.7642, "control_adaptive20": 5.1762,
          "classical_entropy": 3.4431, "dpo_v3": 3.7886}

WORK_DIR     = "/kaggle/working/wordle_phase9"
RESULTS_ROOT = "/kaggle/working/results_phase9"
RESULTS_ZIP  = "/kaggle/working/wordle_phase9_results.zip"
os.makedirs(WORK_DIR, exist_ok=True); os.makedirs(RESULTS_ROOT, exist_ok=True)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); transformers.set_seed(s)
set_seed()
print(f"\narms: {ARMS}  |  games/variant: {N_GAMES}  |  decoder: adaptive @{ADAPTIVE_THRESHOLD}")
''')

# ===========================================================================
md(r"""
---
# 2. Data, solver, models

Two arms. `base` is stock Qwen2.5-0.5B-Instruct with no adapter — every variant
is equally off-distribution for it, which makes it the *clean* test of whether a
prompt helps at all. `sft` is the Phase 7 adapter, which was trained on
`baseline` only, so every other variant is off-distribution for it — that
asymmetry is itself a result worth having.
""")

code(r'''
REQ = ["sft_package/eval/val_answers.jsonl",
       "code/wordle_solver.py", "code/generate_trajectories.py",
       "artifacts/feedback_matrix.npy"]

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
    raise FileNotFoundError("sft_package + artifacts + code not found under /kaggle/input")

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
ALL_VAL = [json.loads(l)["answer"].upper()
           for l in open(os.path.join(SFT_DIR, "eval/val_answers.jsonl"),
                         encoding="utf-8")]
assert len(LEGAL_GUESSES) == 12972 and len(ALL_VAL) == 246

# One fixed subset for every variant and every arm. Sorted after sampling so the
# order is stable regardless of how random.sample happens to enumerate.
_rng = random.Random(GAME_SEED)
ANSWERS = sorted(_rng.sample(ALL_VAL, min(N_GAMES, len(ALL_VAL))))
print(f"answers: {len(ANSWERS)} of {len(ALL_VAL)}  (seed {GAME_SEED})")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if TOKENIZER.pad_token is None: TOKENIZER.pad_token = TOKENIZER.eos_token

def base_model():
    """fp16 + sdpa, asserted. A silent fp32 load costs ~8x on a T4 and the only
    symptom is a slow run, so it is checked rather than hoped for."""
    kw = dict(trust_remote_code=True)
    if ATTN_IMPL: kw["attn_implementation"] = ATTN_IMPL
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 torch_dtype=torch.float16, **kw)
    got = next(m.parameters()).dtype
    assert got == torch.float16, f"expected fp16, got {got}"
    m.config.use_cache = False
    return m

def find_adapter(name):
    """Exact-name match only. A fallback here once loaded the wrong adapter and
    the run looked entirely normal while measuring a different model."""
    hits = []
    for root in ([PREV_RUN_DIR] if PREV_RUN_DIR else []) + [WORK_DIR, "/kaggle/input"]:
        if not root or not os.path.isdir(root): continue
        for dp, _, fs in os.walk(root):
            if "adapter_config.json" in fs and "checkpoint-" not in dp:
                hits.append(dp)
                if os.path.basename(dp.rstrip("/")) == name:
                    print(f"  adapter {name!r} -> {dp}")
                    return dp
    if hits:
        print(f"  NO adapter named {name!r}. Found instead:")
        for h in sorted(set(hits)):
            print(f"      {os.path.basename(h.rstrip(chr(47))):<24} {h}")
        print("  Refusing to fall back - that would measure the wrong model.")
    return None

SFT_PATH = find_adapter(SFT_ADAPTER)
if "sft" in ARMS and SFT_PATH is None:
    raise SystemExit(f"arm 'sft' requested but adapter {SFT_ADAPTER!r} is not attached. "
                     f"Attach it, or set ARMS = ['base'].")
print("SFT adapter:", SFT_PATH or "(not needed)")
''')

# ===========================================================================
md(r"""
---
# 3. Decoder — Phase 7, unchanged

Held fixed across every variant. If this moved, nothing below would be
attributable to the prompt.
""")

with open(os.path.join(_paths.CORE, "constrained_decode.py"), encoding="utf-8") as fh:
    DECODER_SRC = fh.read().rstrip()

code(DECODER_SRC + r'''


LEGAL_WORDS_SORTED = sorted(LEGAL_GUESSES)
SCORER = None
def build_scorer():
    global SCORER
    if SCORER is None:
        SCORER = LegalWordScorer(TOKENIZER, LEGAL_WORDS_SORTED, device="cuda",
                                 chunk=CONSTRAINED_CHUNK)
    return SCORER
print("decoder ready")
''')

# ===========================================================================
md(r"""
---
# 4. The twelve prompt variants

Every variant is `f(turn, history, max_guesses) -> str` ending in `Next guess:`,
so the decoder scores continuations identically across all of them. Nothing here
sees the answer, the answer list, or the candidate set — except `with_count`,
which is marked leaky and reported separately.

The cell prints all twelve rendered on the same state. **Read that output**
before letting the sweep run for an hour; a variant that renders wrongly will
still produce a plausible-looking number.
""")

with open(os.path.join(_paths.CORE, "prompt_variants.py"), encoding="utf-8") as fh:
    PV_SRC = fh.read().rstrip()
# Strip the local-path bootstrap: no __file__ in a notebook, and
# generate_trajectories is already imported above from the dataset's code/ dir.
PV_SRC = re.sub(
    r"^import sys\nimport os\n\nsys\.path\.insert\([^\n]*\n\nfrom generate_trajectories "
    r"import derive_constraints, render_prompt\n",
    "# (sys.path bootstrap and imports supplied by the cells above)\n",
    PV_SRC, count=1, flags=re.M)
assert "sys.path.insert" not in PV_SRC, "prompt_variants bootstrap not stripped"

code(PV_SRC + r'''


# The few-shot exemplars are computed, but "computed" only means self-consistent
# - it does not prove the words are playable. A worked example using an illegal
# word teaches the model to emit illegal words, and the decoder would hide it.
for _ans, _gs, _nxt in _SHOT_GAMES:
    assert _nxt.upper() in LEGAL_GUESSES, f"exemplar next-guess {_nxt!r} is not legal"
    for _g in _gs:
        assert _g.upper() in LEGAL_GUESSES, f"exemplar guess {_g!r} is not legal"
    assert all(feedback_code(_g, _nxt) == feedback_code(_g, _ans) for _g in _gs),         f"exemplar next-guess {_nxt!r} contradicts its own feedback"
print("few-shot exemplars: legal and self-consistent")

NAMES = list(VARIANTS) if VARIANTS_TO_RUN is None else list(VARIANTS_TO_RUN)
for n in NAMES:
    assert n in VARIANTS, f"unknown variant {n!r}"

_demo = [("salet", "BYBBB"), ("crony", "BYBBG")]
print("=" * 78)
print(f"{len(NAMES)} variants, rendered on turn 3 after SALET->BYBBB, CRONY->BYBBG")
print("=" * 78)
for n in NAMES:
    p = render(n, 3, _demo, MAX_GUESSES, n_candidates=17)
    tag = "  [LEAKY]" if VARIANTS[n]["leaky"] else ""
    print(f"\n{'-'*78}\n### {n}{tag}  -- {VARIANTS[n]['note']}"
          f"\n    ({len(TOKENIZER(p)['input_ids'])} tokens)\n{'-'*78}")
    print(p)
print("\n" + "=" * 78)
_tk = {n: len(TOKENIZER(render(n, 3, _demo, MAX_GUESSES, n_candidates=17))["input_ids"])
       for n in NAMES}
print("prompt length (tokens):",
      "  ".join(f"{k}={v}" for k, v in sorted(_tk.items(), key=lambda x: x[1])))
print("NOTE: longer prompts cost proportionally more decoder time.")
''')

# ===========================================================================
md(r"""
---
# 5. The game loop

One `play()` shared by every variant; the *only* thing that changes between
runs is which renderer `GameState.prompt` calls.

The decision cache is keyed on the prompt string, so it is automatically
per-variant and stays exact: a deterministic model plus a greedy decoder means
the same prompt always yields the same word.
""")

code(r'''
class GameState:
    __slots__ = ("answer","history","cands","guesses","patterns","remaining",
                 "forced","n_allowed","done","solved","filt","variant")
    def __init__(self, answer, variant):
        self.answer = answer; self.variant = variant
        self.filt = HardModeFilter(LEGAL_LOWER, feedback_code)
        self.history = []; self.cands = np.arange(VOCAB.n_answers, dtype=np.int32)
        self.guesses, self.patterns, self.remaining = [], [], []
        self.forced, self.n_allowed = [], []
        self.done = self.solved = False
    def prompt(self, turn):
        h = [(g.lower(), p) for g, p in self.history]
        return render(self.variant, turn, h, MAX_GUESSES,
                      n_candidates=len(self.cands))

@torch.no_grad()
def play(model, scorer, answers, variant, threshold=ADAPTIVE_THRESHOLD,
         log_every=0):
    games = [GameState(a, variant) for a in answers]
    cache, hits, seen = {}, 0, 0
    t0 = time.perf_counter()
    for turn in range(1, MAX_GUESSES + 1):
        active = [g for g in games if not g.done]
        if not active: break
        for g in active:
            n_adm = len(g.filt)
            pr = g.prompt(turn)
            seen += 1
            if USE_DECISION_CACHE and pr in cache:
                r = cache[pr]; hits += 1
            else:
                allowed = g.filt.indices(scorer) if n_adm <= threshold else None
                r = scorer.select(model, pr, allowed_idx=allowed,
                                  prune=CONSTRAINED_PRUNE)
                if USE_DECISION_CACHE: cache[pr] = r
            w, forced = r["word"], r["forced"]
            g.forced.append(forced); g.n_allowed.append(n_adm)
            c = feedback_code(w.lower(), g.answer.lower())
            g.cands = BUNDLE.fb.filter_indices(g.cands, w.lower(), c)
            g.filt.refine(w.lower(), c)
            g.history.append((w, code_to_pattern(c)))
            g.guesses.append(w); g.patterns.append(code_to_pattern(c))
            g.remaining.append(int(len(g.cands)))
            if c == ALL_GREEN: g.solved = g.done = True
            elif turn == MAX_GUESSES: g.done = True
        if log_every:
            print(f"      turn {turn}: {sum(1 for x in games if x.done)}/{len(games)}"
                  f"  {time.perf_counter()-t0:.0f}s  cache {100*hits/max(seen,1):.0f}%",
                  flush=True)
    return games, dict(secs=round(time.perf_counter()-t0, 1),
                       decisions=seen, cache_hit_pct=round(100*hits/max(seen,1), 1))

def summarize(games, label, variant):
    n = len(games)
    sc = [len(g.guesses) if g.solved else MAX_GUESSES + 1 for g in games]
    solved = [g for g in games if g.solved]
    hv = tv = 0
    for g in games:
        seen = []
        for gu, pat in zip(g.guesses, g.patterns):
            tv += 1; greens = {}
            for pg, pp in seen:
                for i, (ch, t) in enumerate(zip(pg, pp)):
                    if t == "G": greens[i] = ch
            if any(gu[i] != ch for i, ch in greens.items()): hv += 1
            seen.append((gu, pat))
    dec = [f for g in games for f in g.forced if f is not None]
    return {"arm": label, "variant": variant, "leaky": VARIANTS[variant]["leaky"],
            "n_games": n, "mean": round(sum(sc)/n, 4), "solved": len(solved),
            "failure_rate_pct": round(100*(n-len(solved))/n, 2),
            "hard_mode_violation_pct": round(100*hv/max(tv, 1), 2),
            "forced_pct": round(100*sum(1 for f in dec if f)/max(len(dec), 1), 2),
            "opener": games[0].guesses[0] if games and games[0].guesses else None,
            "per_game": sc,
            "answers": [g.answer for g in games]}
print("game loop ready")
''')

# ===========================================================================
md(r"""
---
# 6. Resumable state

Every `(arm, variant)` result is written to disk the moment it finishes. If the
session dies at variant 9 of 24, re-running this notebook skips the first eight.
""")

code(r'''
SPATH = os.path.join(RESULTS_ROOT, "harness_results.json")
STATE = {"games": {}, "format": {}, "meta": {}}
if os.path.exists(SPATH):
    try:
        STATE = json.load(open(SPATH, encoding="utf-8"))
        STATE.setdefault("games", {}); STATE.setdefault("format", {})
        STATE.setdefault("meta", {})
        print(f"resuming: {len(STATE['games'])} game runs, "
              f"{len(STATE['format'])} format runs already done")
    except Exception as e:
        print("could not read previous state, starting fresh:", e)

STATE["meta"].update(n_games=len(ANSWERS), game_seed=GAME_SEED, seed=SEED,
                     threshold=ADAPTIVE_THRESHOLD, model=MODEL_NAME,
                     sft_adapter=SFT_ADAPTER, phase7=PHASE7,
                     variants=NAMES, arms=ARMS)

def save(tag=""):
    json.dump(STATE, open(SPATH, "w", encoding="utf-8"), indent=2, default=str)
    if tag: print(f"    saved [{tag}]")

MODEL_CACHE = {}
def get_model(arm):
    """One load per arm. The 12 variants share it - reloading per variant would
    cost more than the sweep itself."""
    if arm in MODEL_CACHE: return MODEL_CACHE[arm]
    for k in list(MODEL_CACHE):
        del MODEL_CACHE[k]
    torch.cuda.empty_cache()
    print(f"  loading arm {arm!r} ...", flush=True)
    m = base_model()
    if arm == "sft":
        m = PeftModel.from_pretrained(m, SFT_PATH, is_trainable=False)
    m = m.to("cuda").eval()
    MODEL_CACHE[arm] = m
    return m
save()
print("state ready ->", SPATH)
''')

# ===========================================================================
md(r"""
---
# 7. Sanity gate

`sft` + `baseline` must reproduce Phase 7's opener and land near 3.7642. If it
does not, the adapter, the decoder, or the prompt path is wrong and every number
below is measuring something else. This has already happened once in this
project, so it is checked rather than assumed.

The gate warns rather than aborts — a 100-game subset legitimately differs from
the 246-game figure — but a *large* miss means stop.
""")

code(r'''
if "sft" in ARMS and "baseline" in NAMES:
    sc = build_scorer()
    m = get_model("sft")
    print("gate: sft + baseline ...", flush=True)
    gs, info = play(m, sc, ANSWERS[:40], "baseline", log_every=1)
    s = summarize(gs, "sft", "baseline")
    print(f"\n  opener {s['opener']}  (Phase 7: {PHASE7_OPENER})")
    print(f"  mean   {s['mean']}   on 40 games  (Phase 7 full-set: {PHASE7_MEAN})")
    print(f"  {info['secs']}s, {info['decisions']} decisions, "
          f"cache {info['cache_hit_pct']}%")
    ok_open = s["opener"] == PHASE7_OPENER
    ok_mean = abs(s["mean"] - PHASE7_MEAN) < 0.60
    if ok_open and ok_mean:
        print("\n  GATE PASS")
    else:
        print("\n  *** GATE FAIL ***")
        if not ok_open: print(f"      opener is {s['opener']}, expected {PHASE7_OPENER}")
        if not ok_mean: print(f"      mean {s['mean']} is far from {PHASE7_MEAN}")
        print("      Wrong adapter or wrong decoder. Fix before trusting the sweep.")
    STATE["meta"]["gate"] = {"opener": s["opener"], "mean": s["mean"],
                             "pass": bool(ok_open and ok_mean)}
    # per-decision cost, so section 8 can print a real ETA
    STATE["meta"]["secs_per_decision"] = round(info["secs"]/max(info["decisions"],1), 3)
    save("gate")
else:
    print("gate skipped (needs arm 'sft' and variant 'baseline')")
''')

# ===========================================================================
md(r"""
---
# 8. Probe A — the sweep

This is the long cell. It is resumable and saves after every variant, so
interrupting it loses at most one variant.
""")

code(r'''
sc = build_scorer()
spd = STATE["meta"].get("secs_per_decision", 0.6)
todo = [(a, v) for a in ARMS for v in NAMES if f"{a}|{v}" not in STATE["games"]]
est = len(todo) * len(ANSWERS) * 2.6 * spd / 60
print(f"{len(todo)} runs to do  |  rough ETA {est:.0f} min at {spd}s/decision\n")

if RUN_GAMES:
    for arm in ARMS:
        pending = [v for v in NAMES if f"{arm}|{v}" not in STATE["games"]]
        if not pending: continue
        model = get_model(arm)
        for i, v in enumerate(pending, 1):
            key = f"{arm}|{v}"
            t0 = time.perf_counter()
            print(f"[{arm}] {i}/{len(pending)}  {v:<18}", end="", flush=True)
            try:
                gs, info = play(model, sc, ANSWERS, v)
            except KeyboardInterrupt:
                print("  INTERRUPTED"); save("interrupt"); raise
            r = summarize(gs, arm, v); r.update(info)
            STATE["games"][key] = r
            flag = " [LEAKY]" if r["leaky"] else ""
            print(f"  mean {r['mean']:.4f}  solved {r['solved']}/{r['n_games']}"
                  f"  open {r['opener']}  {time.perf_counter()-t0:.0f}s"
                  f"  cache {info['cache_hit_pct']:.0f}%{flag}", flush=True)
            save()
    print("\nsweep complete")
else:
    print("RUN_GAMES = False, skipped")
''')

# ===========================================================================
md(r"""
---
# 9. Probe B — format compliance, no decoder

Same variants, decoder switched off, raw greedy generation on a fixed set of
mid-game states. Three numbers per variant:

- **legal %** — is the output a real 5-letter Wordle word at all?
- **admissible %** — is it consistent with the feedback so far?
- **parse %** — did the first line even look like a word?

A prompt can raise admissible-% without changing the game mean (the decoder was
already fixing it) or raise the game mean without changing admissible-% (better
*choice* among legal words). Separating the two is the point.
""")

code(r'''
@torch.no_grad()
def format_probe(model, states, variant):
    ok_parse = ok_legal = ok_adm = 0
    ex = []
    for turn, hist, adm_set, ncand in states:
        pr = render(variant, turn, hist, MAX_GUESSES, n_candidates=ncand)
        ids = TOKENIZER(pr, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=FORMAT_MAX_NEW,
                             do_sample=False, num_beams=1,
                             pad_token_id=TOKENIZER.pad_token_id,
                             use_cache=True)
        txt = TOKENIZER.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        tok = txt.strip().split("\n")[0].strip().strip('."\'*` ').upper()
        tok = "".join(ch for ch in tok if ch.isalpha())[:5]
        if len(tok) == 5:
            ok_parse += 1
            if tok in LEGAL_GUESSES:
                ok_legal += 1
                if tok.lower() in adm_set: ok_adm += 1
        if len(ex) < 5: ex.append((txt.replace("\n", "\\n")[:24], tok))
    n = len(states)
    return {"variant": variant, "n": n,
            "parse_pct": round(100*ok_parse/n, 1),
            "legal_pct": round(100*ok_legal/n, 1),
            "admissible_pct": round(100*ok_adm/n, 1),
            "examples": ex}

def _adm_set(filt):
    """HardModeFilter.words IS the surviving list - it is rebuilt in place by
    refine(), so this is a snapshot, not a view."""
    return set(filt.words)

def build_probe_states(k):
    rng = random.Random(SEED)
    solver = make_solver(BUNDLE, SolverConfig(strategy="entropy"))
    out = []
    for ans in ALL_VAL:
        if len(out) >= k: break
        filt = HardModeFilter(LEGAL_LOWER, feedback_code)
        cands = np.arange(VOCAB.n_answers, dtype=np.int32)
        hist = []
        stop = rng.choice([1, 2, 2, 3, 3, 4])     # depth mix, weighted mid-game
        for turn in range(1, MAX_GUESSES + 1):
            w = (solver.opening_guess() if turn == 1
                 else solver.choose(cands, turn)).lower()
            c = feedback_code(w, ans.lower())
            cands = BUNDLE.fb.filter_indices(cands, w, c)
            filt.refine(w, c)
            hist.append((w, code_to_pattern(c)))
            if c == ALL_GREEN: break
            if turn == stop:
                out.append((turn + 1, list(hist), _adm_set(filt), int(len(cands))))
                break
    return out[:k]

if RUN_FORMAT:
    print("building probe states ...", flush=True)
    STATES = build_probe_states(FORMAT_PROBE_STATES)
    print(f"  {len(STATES)} states, turns "
          f"{Counter(s[0] for s in STATES).most_common()}")
    for arm in ARMS:
        pending = [v for v in NAMES if f"{arm}|{v}" not in STATE["format"]]
        if not pending: continue
        model = get_model(arm)
        model.config.use_cache = True
        for v in pending:
            t0 = time.perf_counter()
            r = format_probe(model, STATES, v); r["arm"] = arm
            STATE["format"][f"{arm}|{v}"] = r
            print(f"[{arm}] {v:<18} parse {r['parse_pct']:>5.1f}%  "
                  f"legal {r['legal_pct']:>5.1f}%  adm {r['admissible_pct']:>5.1f}%"
                  f"   {time.perf_counter()-t0:.0f}s", flush=True)
            save()
        model.config.use_cache = False
    print("\nformat probe complete")
else:
    print("RUN_FORMAT = False, skipped")
''')

# ===========================================================================
md(r"""
---
# 10. Results

Paired comparison against `baseline` within each arm — same answers, same
decoder, so the only difference is the text. Paired is what makes a 0.05 gap
readable here; unpaired SE on 100 games is ~0.10 and would hide everything.
""")

code(r'''
def paired(a, b):
    """a, b are per-game scores on the SAME answers, in the same order."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d); mu = sum(d)/n
    if n < 2: return mu, 0.0, 0, 0
    sd = math.sqrt(sum((x-mu)**2 for x in d)/(n-1))
    t = mu/(sd/math.sqrt(n)) if sd > 0 else 0.0
    return mu, t, sum(1 for x in d if x < 0), sum(1 for x in d if x > 0)

def table(arm):
    rows = [STATE["games"][k] for k in STATE["games"] if k.startswith(arm + "|")]
    if not rows: return
    base = next((r for r in rows if r["variant"] == "baseline"), None)
    rows.sort(key=lambda r: r["mean"])
    print("\n" + "=" * 92)
    print(f"ARM: {arm}    ({rows[0]['n_games']} games, adaptive@{ADAPTIVE_THRESHOLD})")
    print("=" * 92)
    print(f"{'variant':<18}{'mean':>8}{'vs base':>9}{'solved':>8}{'fail%':>7}"
          f"{'HMV%':>7}{'forced%':>9}{'better':>8}{'worse':>7}{'t':>7}  opener")
    print("-" * 92)
    for r in rows:
        if base and r["variant"] != "baseline":
            mu, t, bet, wor = paired(r["per_game"], base["per_game"])
            cmpc = f"{mu:+.4f}"; extra = f"{bet:>8}{wor:>7}{t:>7.2f}"
        else:
            cmpc = "     -"; extra = f"{'-':>8}{'-':>7}{'-':>7}"
        flag = " *LEAKY*" if r["leaky"] else ""
        print(f"{r['variant']:<18}{r['mean']:>8.4f}{cmpc:>9}"
              f"{r['solved']:>8}{r['failure_rate_pct']:>7.1f}"
              f"{r['hard_mode_violation_pct']:>7.2f}{r['forced_pct']:>9.1f}"
              f"{extra}  {r['opener']}{flag}")
    print("-" * 92)
    fair = [r for r in rows if not r["leaky"]]
    if fair:
        spread = fair[-1]["mean"] - fair[0]["mean"]
        print(f"fair spread: {spread:.4f} guesses   "
              f"(best {fair[0]['variant']} {fair[0]['mean']:.4f}, "
              f"worst {fair[-1]['variant']} {fair[-1]['mean']:.4f})")
    print(f"reference: classical entropy {PHASE7['classical_entropy']}, "
          f"filter-only {PHASE7['control_adaptive20']}, "
          f"Phase 7 full-set {PHASE7['sft_adaptive20']}")

for arm in ARMS: table(arm)

# ---- format probe -----------------------------------------------------------
for arm in ARMS:
    rows = [STATE["format"][k] for k in STATE["format"] if k.startswith(arm + "|")]
    if not rows: continue
    rows.sort(key=lambda r: -r["admissible_pct"])
    print("\n" + "=" * 72)
    print(f"FORMAT PROBE (no decoder) - arm {arm}, {rows[0]['n']} states")
    print("=" * 72)
    print(f"{'variant':<18}{'parse%':>9}{'legal%':>9}{'admissible%':>13}   sample")
    print("-" * 72)
    for r in rows:
        s = r["examples"][0][1] if r["examples"] else ""
        print(f"{r['variant']:<18}{r['parse_pct']:>9.1f}{r['legal_pct']:>9.1f}"
              f"{r['admissible_pct']:>13.1f}   {s}")

# ---- the two questions this notebook exists to answer ------------------------
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
for arm in ARMS:
    rows = {STATE["games"][k]["variant"]: STATE["games"][k]
            for k in STATE["games"] if k.startswith(arm + "|")}
    fair = [r for r in rows.values() if not r["leaky"]]
    if not fair: continue
    spread = max(r["mean"] for r in fair) - min(r["mean"] for r in fair)
    print(f"\n[{arm}] fair spread across variants: {spread:.4f} guesses")
    if spread < 0.05:
        print("      -> prompt format is NOT a lever on this task/model.")
    elif spread < 0.15:
        print("      -> small but real. Comparable to a full training phase.")
    else:
        print("      -> format matters MORE than four training interventions did.")
        print("         Next SFT run should use the winning format.")
    b, rh = rows.get("baseline"), rows.get("raw_history")
    if b and rh:
        mu, t, bet, wor = paired(rh["per_game"], b["per_game"])
        print(f"      raw_history vs baseline: {mu:+.4f} (t={t:.2f}, "
              f"{bet} better / {wor} worse)")
        if abs(mu) < 0.05:
            print("      -> the solver-derived constraint block is DECORATION.")
        elif mu > 0:
            print("      -> the harness has been doing the deduction, not the model.")
        else:
            print("      -> raw history BEATS the derived block; the block misleads.")
    lk = [r for r in rows.values() if r["leaky"]]
    if lk and b:
        mu, t, _, _ = paired(lk[0]["per_game"], b["per_game"])
        print(f"      [leaky] candidate count is worth {(-mu):+.4f} - an upper "
              f"bound only, not usable.")
save("final")
''')

# ===========================================================================
md(r"""
---
# 11. Save

Publishes to a Kaggle Dataset if the secrets are set, and always writes a zip so
there is a download either way.
""")

code(r'''
import zipfile
SLUG  = "wordle-phase9-harness"
TITLE = "Wordle Phase 9 - prompt/harness sweep"

save("pre-zip")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_game"}
           for k, v in STATE["games"].items()},
          open(os.path.join(RESULTS_ROOT, "summary.json"), "w"), indent=2)

if os.path.exists(RESULTS_ZIP): os.remove(RESULTS_ZIP)
with zipfile.ZipFile(RESULTS_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(RESULTS_ROOT):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, RESULTS_ROOT))
print(f"zip: {RESULTS_ZIP} ({os.path.getsize(RESULTS_ZIP)/2**20:.2f} MiB)")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(RESULTS_ZIP, "/kaggle/working")))
except Exception: pass

try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
    USER = os.environ["KAGGLE_USERNAME"]
except Exception as e:
    print("no credentials - zip only"); raise SystemExit

json.dump({"title": TITLE, "id": f"{USER}/{SLUG}",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(RESULTS_ROOT, "dataset-metadata.json"), "w"), indent=2)
def run(c):
    r = subprocess.run(c, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")
rc, out = run(["kaggle", "datasets", "version", "-p", RESULTS_ROOT,
               "-m", f"phase9 {time.strftime('%Y-%m-%d %H:%M')}", "-r", "zip"])
if rc != 0 and ("404" in out or "not found" in out.lower()):
    rc, out = run(["kaggle", "datasets", "create", "-p", RESULTS_ROOT, "-r", "zip"])
print(f"PUBLISHED -> https://www.kaggle.com/datasets/{USER}/{SLUG}" if rc == 0
      else "PUBLISH FAILED - use the zip\n" + out.strip()[:600])
''')

# ===========================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    if c["source"]:
        c["source"][-1] = c["source"][-1].rstrip("\n")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print(f"wrote {OUT}  ({len(cells)} cells)")
