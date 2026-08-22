"""
make_dpo_notebook.py - generate wordle_phase8_dpo_kaggle.ipynb.

Two things in one session:

  1. COUNTERFACTUAL HEADROOM - hand each bucket to the expert in turn and
     measure what perfect play there would be worth. This decides whether DPO
     is worth continuing past the first run.
  2. DPO - train a preference adapter on top of a fresh SFT reproduction and
     evaluate with the Phase 7b adaptive decoder at threshold 20.

WHY DPO IS IMPLEMENTED BY HAND

TRL's DPOTrainer pins tightly to transformers, and this project has already
been bitten twice by that coupling (Kaggle ships transformers 5.0). The DPO
loss is six lines. Writing it directly removes a dependency whose version must
match exactly and which would fail at import, after the training data is
already loaded.

    loss = -logsigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

REFERENCE MODEL

The SFT adapter is merged into the base weights, then a FRESH LoRA is added for
DPO. The reference is that same merged model with the new adapter disabled - so
the reference is the SFT policy, which is what DPO requires. Continuing Phase
6's adapter directly would make "better preferences" and "more training"
inseparable.

    python make_dpo_notebook.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_phase8_dpo_kaggle.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


md(r"""
# Phase 8 — counterfactual headroom, then DPO

## Where this starts

| | mean | solved | fail |
|---|---:|---:|---:|
| SFT + adaptive decoder @20 | **3.7642** | 242/246 | 1.6% |
| filter only, no model | 5.1762 | — | 22.8% |
| classical `entropy` | 3.4431 | 246 | 0% |

The gap to the expert is **0.32 guesses**. Two questions, in order.

## 1. Where does that 0.32 actually live?

Hand one bucket at a time to the expert and replay. Buckets are keyed on
**|admissible|** — the number of legal words consistent with the feedback so
far — because that is the decision regime the model faces under the adaptive
decoder.

This is a *hybrid-policy evaluation*, not a replay of logged games. Substituting
an action changes the feedback and therefore every later state, so the model has
to be in the loop. It cannot be done offline from Phase 7's logs.

If total recoverable headroom is small, DPO cannot help much and the honest move
is to stop and write up. That is a real possible outcome of this cell.

## 2. Does DPO improve the policy?

Under the adaptive decoder the model's job is to **rank admissible words**. SFT
on one expert action per state teaches a point, not a ranking. Phase 6 tested
"more/better SFT data" directly and moved games by exactly zero, so a preference
objective is the shape of experiment that has not been tried.

Pairs come from `build_dpo_dataset.py`: reachable states, mid-game weighted,
`chosen`/`rejected` drawn from the action space the decoder would actually
give the model, and only emitted when the cost gap clears a margin. The model
never sees candidate lists, counts, or scores.

---

## How to run

1. **GPU T4 x2**
2. Add Input: the SFT package containing `data/dpo_v3/`, and (optional but
   recommended) the Phase 7 adapter dataset
3. Set `PREV_RUN_DIR` if you have the adapter — otherwise the SFT stage retrains
   it (~70 min)
4. Run All. **~2.5 h** with the adapter, ~3.5 h without
5. **Save the DPO adapter when section 6 tells you to**
6. Download `wordle_phase8_results.zip`

Resumable: results reload at startup and completed stages are skipped.
""")

md(r"""
---
# 1. Environment and config
""")

code(r'''
import os, sys, json, time, math, random, subprocess, importlib, shutil, glob
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
DATASET_DIR  = None
PREV_RUN_DIR = None      # Phase 7 adapter dir; if found, SFT is skipped
# ============================================================================

RUN_HEADROOM = False   # answered in the first run; set True to redo
RUN_DPO      = True
RUN_EVAL     = True

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER  = "tree_salet_endgame"
DPO_ADAPTER  = "tree_salet_dpo"

# ---- SFT (only if the adapter is missing) - identical to Phase 6/7 ---------
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
SFT_LR, SFT_EPOCHS = 2e-4, 2
PER_DEVICE_BS, GRAD_ACCUM, MAX_SEQ_LEN = 4, 4, 640

# ---- DPO ------------------------------------------------------------------
DPO_BETA      = 0.1        # standard; not swept, per the brief
DPO_LR        = 5e-6       # ~40x below SFT: DPO drifts fast at SFT learning rates
DPO_EPOCHS    = 1
DPO_BS        = 4
DPO_ACCUM     = 8
DPO_MAX_PAIRS = 15000

# v3 is the audited replacement for dpo_midgame. It has a real validation split,
# one row per state, no duplicate ids, and no consistency shortcut. v1 had all
# four problems, which is the likeliest reason the first run regressed.
DPO_DATASET   = "dpo_v3"       # "dpo_v3" | "dpo_midgame"
EVAL_PAIRS_EVERY = 50          # optimizer steps between held-out pair evals

# ---- decoder speed / correctness knobs (added after the slow run) ----------
ATTN_IMPL      = "sdpa"    # "sdpa" | "eager" | None. eager is 5-10x slower.
USE_DECISION_CACHE = True  # same prompt => same decision. Exact, not approximate.
PREFLIGHT      = True      # verify the adapter IS the Phase 7 model before
                           # spending an hour measuring it
PREFLIGHT_GAMES = 40       # small-sample baseline check
# The Phase 7 fingerprint. The SFT adapter must reproduce these or the run is
# measuring something else, which is exactly what happened once.
PHASE7_OPENER  = "SALET"
PHASE7_MEAN    = 3.7642
PHASE7_SOLVED  = 242

# v1 reported 0.604 -> 0.836 as if it were generalisation. It was TRAINING
# accuracy - there was no validation split. With v3 there is one, so the
# held-out curve below is the number that actually means something.
DPO_LORA_R, DPO_LORA_ALPHA = 16, 32

SEED = 20260817
MAX_GUESSES = 6
ADAPTIVE_THRESHOLD = 20    # the Phase 7b optimum; fixed, not swept
CONSTRAINED_CHUNK, CONSTRAINED_PRUNE = 512, True

# |admissible| buckets for the counterfactual
HEADROOM_BUCKETS = {"2-10": (2, 10), "11-100": (11, 100), "100+": (101, 10**9)}

PHASE7 = {"sft_adaptive20": 3.7642, "sft_adaptive20_solved": 242,
          "control_adaptive20": 5.1762, "classical_entropy": 3.4431,
          "classical_random": 4.0203, "classical_frequency": 3.7927}

WORK_DIR     = "/kaggle/working/wordle_phase8"
RESULTS_ROOT = "/kaggle/working/results_phase8"
RESULTS_ZIP  = "/kaggle/working/wordle_phase8_results.zip"
os.makedirs(WORK_DIR, exist_ok=True); os.makedirs(RESULTS_ROOT, exist_ok=True)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); transformers.set_seed(s)
set_seed()
print(f"\nDPO: beta={DPO_BETA} lr={DPO_LR} epochs={DPO_EPOCHS} "
      f"eff_batch={DPO_BS*DPO_ACCUM}")
print(f"eval decoder: adaptive @ threshold {ADAPTIVE_THRESHOLD}")
''')

md(r"""
---
# 2. Data, solver, adapter
""")

code(r'''
REQ = ["sft_package/data/tree_salet/train.jsonl",
       "sft_package/data/tree_salet_endgame/train.jsonl",
       "sft_package/data/dpo_v3/train.jsonl",
       "sft_package/data/dpo_v3/validation.jsonl",
       "sft_package/eval/val_answers.jsonl",
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
    near = [dp for dp, _, _ in os.walk("/kaggle/input")
            if os.path.exists(os.path.join(dp, "sft_package/data/tree_salet/train.jsonl"))]
    raise FileNotFoundError(
        f"SFT package with data/{DPO_DATASET}/ not found.\n" +
        ("Found a package WITHOUT the DPO pairs:\n  " + "\n  ".join(near) +
         "\n-> rebuild locally:\n"
         "     python phase8_dpo_v3/build_dpo_v3_dataset.py\n"
         "     python phase8_dpo_v3/verify_dpo_v3_dataset.py\n"
         "     python tools/prepare_kaggle_dataset.py --dest uploads/kaggle_upload\n"
         "   then re-upload." if near else "Nothing found under /kaggle/input."))

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

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if TOKENIZER.pad_token is None: TOKENIZER.pad_token = TOKENIZER.eos_token

def base_model():
    """fp16 + sdpa, asserted.

    `torch_dtype` is deprecated in transformers 5.x and a silent fp32 load costs
    ~8x on a T4. `eager` attention costs another 5-10x. Neither failure is
    visible except as a slow run, so both are now checked rather than hoped for.
    """
    kw = dict(trust_remote_code=True)
    if ATTN_IMPL:
        kw["attn_implementation"] = ATTN_IMPL
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
    """Exact-name match only.

    An earlier version fell back to ANY adapter when the requested name was
    missing. That silently loaded a DPO adapter as the SFT base: the run looked
    entirely normal while measuring the wrong model. There is no fallback now -
    a missing name is an error, not a guess.
    """
    hits = []
    for root in ([PREV_RUN_DIR] if PREV_RUN_DIR else []) + [WORK_DIR, "/kaggle/input"]:
        if not root or not os.path.isdir(root):
            continue
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

SFT_DIR_ADAPTER = find_adapter(SFT_ADAPTER)
print("SFT adapter:", SFT_DIR_ADAPTER or "NOT FOUND - will train")
''')

md(r"""
---
# 3. SFT stage — reproduce, or reuse

Skipped entirely if the Phase 7 adapter is attached. If it trains, the config is
byte-identical to Phase 6/7, so the DPO baseline is the same policy that scored
3.7642.
""")

code(r'''
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

class SFTData(Dataset):
    def __init__(self, rows, tok, max_len=MAX_SEQ_LEN):
        self.rows, self.tok, self.max_len = rows, tok, max_len; self._c = {}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if i in self._c: return self._c[i]
        r = self.rows[i]
        p = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = self.tok(" " + r["completion"], add_special_tokens=False)["input_ids"] \
            + [self.tok.eos_token_id]
        p = p[-(self.max_len - len(c)):]
        it = {"input_ids": p + c, "labels": [-100]*len(p) + c,
              "attention_mask": [1]*(len(p)+len(c))}
        self._c[i] = it; return it

def collate(batch, pad):
    n = max(len(b["input_ids"]) for b in batch)
    o = {k: [] for k in ("input_ids", "labels", "attention_mask")}
    for b in batch:
        d = n - len(b["input_ids"])
        o["input_ids"].append(b["input_ids"] + [pad]*d)
        o["labels"].append(b["labels"] + [-100]*d)
        o["attention_mask"].append(b["attention_mask"] + [0]*d)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in o.items()}

SFT_INFO = {}
if SFT_DIR_ADAPTER is None:
    rows = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet/train.jsonl")) + \
           load_jsonl(os.path.join(SFT_DIR, "data/tree_salet_endgame/train.jsonl"))
    random.Random(SEED).shuffle(rows)
    print(f"training SFT on {len(rows)} rows (Phase 6/7 mix)")
    set_seed(); fix_torchao_peft_conflict()
    m = get_peft_model(base_model(), LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    for _, p_ in m.named_parameters():
        if p_.requires_grad and p_.dtype == torch.float16:
            p_.data = p_.data.float()
    out = os.path.join(WORK_DIR, SFT_ADAPTER)
    tr = Trainer(model=m, args=TrainingArguments(
            output_dir=out, num_train_epochs=SFT_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM, learning_rate=SFT_LR,
            lr_scheduler_type="cosine", warmup_ratio=0.03, fp16=True,
            gradient_checkpointing=True, logging_steps=25, save_steps=400,
            save_total_limit=1, report_to=[], seed=SEED,
            remove_unused_columns=False, disable_tqdm=False),
        train_dataset=SFTData(rows, TOKENIZER),
        data_collator=lambda b: collate(b, TOKENIZER.pad_token_id))
    t0 = time.perf_counter(); tr.train(); tr.save_model(out)
    h = [x["loss"] for x in tr.state.log_history if "loss" in x]
    SFT_INFO = {"steps": tr.state.global_step, "first_loss": h[0],
                "final_loss": h[-1], "seconds": round(time.perf_counter()-t0, 1)}
    print(f"SFT done: {SFT_INFO}")
    del m, tr; torch.cuda.empty_cache()
    SFT_DIR_ADAPTER = out
else:
    print(f"reusing SFT adapter: {SFT_DIR_ADAPTER}")
''')

md(r"""
---
# 4. Decoder (Phase 7, unchanged)
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
                                 chunk=CONSTRAINED_CHUNK)
    return SCORER

class GameState:
    __slots__ = ("answer","history","cands","guesses","patterns","remaining",
                 "forced","n_allowed","buckets","done","solved","filt","win_forced")
    def __init__(self, answer, n_answers):
        self.answer = answer
        self.filt = HardModeFilter(LEGAL_LOWER, feedback_code)
        self.history = []; self.cands = np.arange(n_answers, dtype=np.int32)
        self.guesses, self.patterns, self.remaining = [], [], []
        self.forced, self.n_allowed, self.buckets = [], [], []
        self.done = self.solved = False; self.win_forced = None
    def prompt(self, turn):
        h = [(g.lower(), p) for g, p in self.history]
        return render_prompt(turn=turn, history=h,
                             constraints=derive_constraints(h),
                             n_candidates=len(self.cands),
                             guesses_remaining=MAX_GUESSES-turn+1,
                             max_guesses=MAX_GUESSES,
                             candidates=None, show_candidate_count=False)

def bucket_of(n):
    for name, (lo, hi) in HEADROOM_BUCKETS.items():
        if lo <= n <= hi: return name
    return None

@torch.no_grad()
def play(model, scorer, answers, expert_buckets=(), expert=None,
         threshold=ADAPTIVE_THRESHOLD, log_every=80):
    """Play 246 games. In `expert_buckets` the EXPERT acts instead of the model.

    That substitution changes the feedback and every later state, so this is a
    hybrid-policy evaluation and not a replay of logged games.
    """
    games = [GameState(a, VOCAB.n_answers) for a in answers]
    # Same prompt => same decision: the model is deterministic and the decoder
    # is greedy, so this is exact rather than an approximation. Turn 1 is one
    # distinct prompt across all 246 games; turn 2 is a few dozen.
    cache, hits, seen = {}, 0, 0
    t0 = time.perf_counter()
    for turn in range(1, MAX_GUESSES+1):
        active = [g for g in games if not g.done]
        if not active: break
        for i, g in enumerate(active):
            n_adm = len(g.filt)
            b = bucket_of(n_adm)
            g.buckets.append(b)
            if b in expert_buckets and expert is not None and len(g.cands):
                w = (expert.opening_guess() if turn == 1
                     else expert.choose(g.cands, turn)).upper()
                forced = None
            else:
                pr = g.prompt(turn)
                seen += 1
                if USE_DECISION_CACHE and pr in cache:
                    r = cache[pr]; hits += 1
                else:
                    allowed = g.filt.indices(scorer) if n_adm <= threshold else None
                    r = scorer.select(model, pr, allowed_idx=allowed,
                                      prune=CONSTRAINED_PRUNE)
                    if USE_DECISION_CACHE:
                        cache[pr] = r
                w, forced = r["word"], r["forced"]
            # record the ADMISSIBLE count, not scorer.n_allowed: when the filter
            # is off the scorer reports the whole vocabulary, which would hide
            # the decision regime we are bucketing by.
            g.forced.append(forced); g.n_allowed.append(n_adm)
            c = feedback_code(w.lower(), g.answer.lower())
            g.cands = BUNDLE.fb.filter_indices(g.cands, w.lower(), c)
            g.filt.refine(w.lower(), c)
            g.history.append((w, code_to_pattern(c)))
            g.guesses.append(w); g.patterns.append(code_to_pattern(c))
            g.remaining.append(int(len(g.cands)))
            if c == ALL_GREEN:
                g.solved = g.done = True; g.win_forced = forced
            elif turn == MAX_GUESSES:
                g.done = True
            if log_every and (i+1) % log_every == 0:
                print(f"      turn {turn}: {i+1}/{len(active)} "
                      f"({time.perf_counter()-t0:.0f}s, cache "
                      f"{100*hits/max(seen,1):.0f}%)", flush=True)
        print(f"  turn {turn}: {sum(1 for g in games if g.done)}/{len(games)} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return games

def summarize(games, label):
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
                    if t=="G": greens[i]=ch
            if any(gu[i]!=ch for i,ch in greens.items()): hv += 1
            seen.append((gu,pat))
    dec = [f for g in games for f in g.forced if f is not None]
    return {"model": label, "n_games": n, "mean": round(sum(sc)/n, 4),
            "solved": len(solved),
            "failure_rate_pct": round(100*(n-len(solved))/n, 2),
            "hard_mode_violation_pct": round(100*hv/max(tv,1), 2),
            "forced_pct": round(100*sum(1 for f in dec if f)/max(len(dec),1), 2),
            "wins_forced": sum(1 for g in games if g.solved and g.win_forced),
            "wins_model": sum(1 for g in games if g.solved and g.win_forced is False),
            "per_game": sc}      # kept so SFT vs DPO can be compared PAIRED
print("decoder ready")
''')

md(r"""
---
# 5. Counterfactual headroom

Each bucket handed to the classical expert in turn, then all three together.
The combined row is the ceiling this model could reach if its action selection
were perfect everywhere.

`entropy` is the stand-in expert: on these 246 answers it scores 3.4431 against
`tree_salet`'s 3.4512, and it is fast enough to run in the loop.
""")

code(r'''
STATE = {}
SPATH = os.path.join(RESULTS_ROOT, "results.json")
if os.path.exists(SPATH):
    try:
        STATE = json.load(open(SPATH, encoding="utf-8")); print("resumed:", sorted(STATE))
    except Exception as e: print("fresh start", e)

def save(tag=""):
    json.dump(STATE, open(SPATH, "w", encoding="utf-8"), indent=2, default=str)
    if tag: print(f"    [saved {tag}]", flush=True)

def load_sft():
    m = PeftModel.from_pretrained(base_model(), SFT_DIR_ADAPTER)
    m.config.use_cache = True
    return m.eval().cuda()

HEAD = STATE.get("headroom", {})
if RUN_HEADROOM and "combined" not in HEAD:
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    EXPERT = make_solver("entropy", BUNDLE.fb, cfgc, BUNDLE.model); EXPERT.reset()
    M = load_sft(); SC = build_scorer()
    runs = [("baseline", ()), ("2-10", ("2-10",)), ("11-100", ("11-100",)),
            ("100+", ("100+",)),
            ("combined", ("2-10", "11-100", "100+"))]
    for name, bk in runs:
        if name in HEAD: continue
        print(f"--- expert in bucket(s): {bk or 'none (baseline)'} ---", flush=True)
        gs = play(M, SC, VAL_ANSWERS, expert_buckets=bk, expert=EXPERT)
        HEAD[name] = summarize(gs, f"expert@{name}")
        print(f"  mean={HEAD[name]['mean']:.4f}  solved={HEAD[name]['solved']}")
        STATE["headroom"] = HEAD; save(name)
    del M; torch.cuda.empty_cache()

if HEAD:
    base = HEAD["baseline"]["mean"]
    print("\n" + "="*76); print("COUNTERFACTUAL HEADROOM"); print("="*76)
    print(f"{'expert acts in':<18}{'mean':>9}{'solved':>8}{'recovered':>12}{'% of gap':>10}")
    gap = base - PHASE7["classical_entropy"]
    for k in ("baseline", "2-10", "11-100", "100+", "combined"):
        if k not in HEAD: continue
        r = HEAD[k]; rec = base - r["mean"]
        print(f"{k:<18}{r['mean']:>9.4f}{r['solved']:>8}"
              f"{rec:>12.4f}{100*rec/gap if gap else 0:>9.1f}%")
    print(f"\nbaseline {base:.4f}  vs classical entropy "
          f"{PHASE7['classical_entropy']:.4f}   total gap {gap:.4f}")
    print("""
Reading it: 'recovered' is what perfect action selection in that bucket alone
would buy. If `combined` recovers most of the gap, the model's action selection
is the whole problem and DPO has room. If it recovers little, the gap is
structural and no preference training will close it.""")
''')

md(r"""
---
# 5b. Preflight — is this the model we think it is?

Two failures cost a whole session once. Both are cheap to rule out here.

1. **The wrong adapter loaded.** A silent name fallback returned a DPO adapter
   as the SFT base, and the run looked entirely normal while measuring the
   wrong model. The lookup is strict now; this re-checks by *behaviour*.
2. **A slow decoder.** fp32 or `eager` attention cost 8-10x each and show up
   only as a run that never finishes.

The Phase 7 SFT model has a fingerprint: it opens **SALET** in 100% of games
and scores **3.7642 / 242 solved**. A 40-game sample reproduces the mean within
sampling error in about a minute. If it does not, stop — nothing downstream is
interpretable.
""")

code(r'''
if PREFLIGHT and RUN_EVAL:
    print("=" * 70); print("PREFLIGHT"); print("=" * 70)
    M = load_sft(); SC = build_scorer()

    # --- identity: the opener is a hard fingerprint ---------------------
    g0 = GameState(VAL_ANSWERS[0], VOCAB.n_answers)
    t0 = time.perf_counter()
    r0 = SC.select(M, g0.prompt(1), allowed_idx=None, prune=CONSTRAINED_PRUNE)
    dt = time.perf_counter() - t0
    print(f"  opener        : {r0['word']}   (Phase 7: {PHASE7_OPENER})")
    print(f"  one decision  : {dt:.2f}s   chunks {r0['n_chunks']}/"
          f"{-(-SC.n // CONSTRAINED_CHUNK)}")
    print(f"  attn impl     : "
          f"{getattr(M.base_model.model.config, '_attn_implementation', '?')}")
    print(f"  dtype         : {next(M.parameters()).dtype}")

    assert r0["word"] == PHASE7_OPENER, (
        f"opener is {r0['word']}, expected {PHASE7_OPENER}. This is NOT the "
        f"Phase 7 SFT adapter - check which adapter section 2 loaded.")
    if dt > 3.0:
        print(f"  WARNING: {dt:.1f}s per unfiltered decision (Phase 7: ~0.6s).")
        print("           Full eval will take hours. See ATTN_IMPL / dtype above.")

    # --- behaviour: does a small sample reproduce the known mean? -------
    sample = VAL_ANSWERS[::max(1, len(VAL_ANSWERS) // PREFLIGHT_GAMES)][:PREFLIGHT_GAMES]
    print(f"\n  playing {len(sample)} games to check the baseline ...", flush=True)
    t0 = time.perf_counter()
    gs = play(M, SC, sample, log_every=0)
    sc_ = [len(g.guesses) if g.solved else MAX_GUESSES + 1 for g in gs]
    mean = sum(sc_) / len(sc_)
    solved = sum(1 for g in gs if g.solved)
    # SE of a 40-game mean with sd ~1.0 is ~0.16, so allow 3 sigma
    tol = 0.50
    print(f"  sample mean   : {mean:.4f}   (Phase 7 full-set: {PHASE7_MEAN})")
    print(f"  sample solved : {solved}/{len(sample)}")
    print(f"  elapsed       : {time.perf_counter()-t0:.0f}s")
    ok = abs(mean - PHASE7_MEAN) <= tol
    print(f"\n  {'PASS' if ok else 'FAIL'}: baseline "
          f"{'reproduces' if ok else 'DOES NOT reproduce'} "
          f"(|{mean:.4f} - {PHASE7_MEAN}| vs tol {tol})")
    assert ok, (
        "The SFT baseline does not reproduce Phase 7. Something differs - the "
        "adapter, the decoder, or the threshold. Do NOT interpret any result "
        "from this run until it does.")
    print("\n  preflight passed - safe to spend the session")
    del M; torch.cuda.empty_cache()
else:
    print("preflight skipped")
''')

md(r"""
---
# 6. DPO

The SFT adapter is merged into the base, a fresh LoRA is added, and the
reference is that same model with the new adapter disabled — i.e. the reference
*is* the SFT policy.

Loss, in full:

```
loss = -logsigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )
```

where each term is a summed sequence log-probability over the completion tokens
only. `margin` is the implicit reward gap; it should rise from 0 and stay
positive. `acc` is the fraction of pairs the policy ranks correctly.
""")

code(r'''
PAIRS = load_jsonl(os.path.join(SFT_DIR, f"data/{DPO_DATASET}/train.jsonl"))
random.Random(SEED).shuffle(PAIRS)
PAIRS = PAIRS[:DPO_MAX_PAIRS]

VPATH = os.path.join(SFT_DIR, f"data/{DPO_DATASET}/validation.jsonl")
VAL_PAIRS = load_jsonl(VPATH) if os.path.exists(VPATH) else []

print(f"dataset  : {DPO_DATASET}")
print(f"train    : {len(PAIRS)} pairs")
print(f"held-out : {len(VAL_PAIRS)} pairs"
      + ("" if VAL_PAIRS else "   <-- NONE. Accuracy below will be TRAINING accuracy,"
                              " which is what made the v1 number meaningless."))
def _shape(rows, lbl):
    if not rows: return
    b = Counter(r["meta"].get("bucket") for r in rows)
    t = Counter(r["meta"].get("pair_type") for r in rows)
    print(f"  {lbl:<9} buckets {dict(sorted(b.items()))}")
    print(f"  {'':<9} types   {dict(sorted(t.items()))}")
_shape(PAIRS, "train"); _shape(VAL_PAIRS, "held-out")

assert not any("Possible answers" in p["prompt"] for p in PAIRS)
print("  no candidate counts in prompts  OK")
if VAL_PAIRS:
    ov = set(p["prompt"] for p in PAIRS) & set(p["prompt"] for p in VAL_PAIRS)
    assert not ov, f"train/validation share {len(ov)} states"
    print("  train and held-out share no state  OK")

def encode(prompt, word):
    p = TOKENIZER(prompt, add_special_tokens=False)["input_ids"]
    c = TOKENIZER(" " + word, add_special_tokens=False)["input_ids"] + [TOKENIZER.eos_token_id]
    p = p[-(MAX_SEQ_LEN - len(c)):]
    return p + c, [-100]*len(p) + c

class PairData(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        ci, cl = encode(r["prompt"], r["chosen"])
        ri, rl = encode(r["prompt"], r["rejected"])
        return {"ci": ci, "cl": cl, "ri": ri, "rl": rl}

def pair_collate(b, pad):
    def stack(ids, lbl):
        n = max(len(x) for x in ids)
        I = [x + [pad]*(n-len(x)) for x in ids]
        L = [x + [-100]*(n-len(x)) for x in lbl]
        A = [[1]*len(x) + [0]*(n-len(x)) for x in ids]
        return (torch.tensor(I), torch.tensor(L), torch.tensor(A))
    ci, cl, ca = stack([x["ci"] for x in b], [x["cl"] for x in b])
    ri, rl, ra = stack([x["ri"] for x in b], [x["rl"] for x in b])
    return ci, cl, ca, ri, rl, ra

def seq_logp(model, ids, lbl, attn):
    """Summed log P over the supervised (completion) tokens only."""
    out = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    tgt = lbl[:, 1:]
    mask = (tgt != -100)
    lp = torch.log_softmax(out.float(), dim=-1)
    tok = torch.gather(lp, 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(-1)

@torch.no_grad()
def eval_pairs(policy, rows, limit=400):
    """Held-out pair accuracy and margin. THIS is the generalisation number."""
    if not rows:
        return None
    policy.eval()
    sub = rows[:limit]
    accs, margins = [], []
    for i in range(0, len(sub), DPO_BS):
        b = [PairData(sub)[j] for j in range(i, min(i + DPO_BS, len(sub)))]
        ci, cl, ca, ri, rl, ra = pair_collate(b, TOKENIZER.pad_token_id)
        ci, cl, ca, ri, rl, ra = [x.cuda() for x in (ci, cl, ca, ri, rl, ra)]
        with policy.disable_adapter():
            rc = seq_logp(policy, ci, cl, ca); rr = seq_logp(policy, ri, rl, ra)
        pc = seq_logp(policy, ci, cl, ca); pr = seq_logp(policy, ri, rl, ra)
        m = (pc - rc) - (pr - rr)
        margins.append(m.mean().item()); accs.append((m > 0).float().mean().item())
    policy.train()
    return {"acc": float(np.mean(accs)), "margin": float(np.mean(margins)),
            "n": len(sub)}


DPO_OUT = os.path.join(WORK_DIR, DPO_ADAPTER)
DPO_LOG = STATE.get("dpo_log", [])
if RUN_DPO and not os.path.exists(os.path.join(DPO_OUT, "adapter_config.json")):
    set_seed(); fix_torchao_peft_conflict()
    print("merging SFT adapter into the base ...")
    merged = PeftModel.from_pretrained(base_model(), SFT_DIR_ADAPTER).merge_and_unload()
    policy = get_peft_model(merged, LoraConfig(
        r=DPO_LORA_R, lora_alpha=DPO_LORA_ALPHA, lora_dropout=0.0,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    for _, p_ in policy.named_parameters():
        if p_.requires_grad and p_.dtype == torch.float16:
            p_.data = p_.data.float()
    policy.print_trainable_parameters()
    policy.config.use_cache = False
    policy = policy.cuda()

    dl = torch.utils.data.DataLoader(
        PairData(PAIRS), batch_size=DPO_BS, shuffle=True,
        collate_fn=lambda b: pair_collate(b, TOKENIZER.pad_token_id))
    opt = torch.optim.AdamW([p_ for p_ in policy.parameters() if p_.requires_grad],
                            lr=DPO_LR)
    total = (len(dl)*DPO_EPOCHS)//DPO_ACCUM
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=DPO_LR, total_steps=max(total, 1), pct_start=0.1)
    print(f"{len(dl)} batches x {DPO_EPOCHS} epoch(s) -> ~{total} optimizer steps")

    step = 0; t0 = time.perf_counter(); run = []
    for ep in range(DPO_EPOCHS):
        for i, (ci, cl, ca, ri, rl, ra) in enumerate(dl):
            ci,cl,ca,ri,rl,ra = [x.cuda() for x in (ci,cl,ca,ri,rl,ra)]
            with torch.no_grad(), policy.disable_adapter():
                ref_c = seq_logp(policy, ci, cl, ca)
                ref_r = seq_logp(policy, ri, rl, ra)
            pi_c = seq_logp(policy, ci, cl, ca)
            pi_r = seq_logp(policy, ri, rl, ra)
            margin = (pi_c - ref_c) - (pi_r - ref_r)
            loss = -F.logsigmoid(DPO_BETA * margin).mean()
            (loss/DPO_ACCUM).backward()
            run.append((loss.item(), margin.mean().item(),
                        (margin > 0).float().mean().item()))
            if (i+1) % DPO_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p_ for p_ in policy.parameters() if p_.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 25 == 0:
                    L,Mg,A = np.mean([r[0] for r in run]), np.mean([r[1] for r in run]), \
                             np.mean([r[2] for r in run])
                    DPO_LOG.append({"step": step, "loss": round(float(L),4),
                                    "margin": round(float(Mg),4), "acc": round(float(A),4)})
                    print(f"  step {step:>4}  loss {L:.4f}  margin {Mg:+.4f}  "
                          f"acc {A:.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
                    run = []
    policy.save_pretrained(DPO_OUT)
    STATE["dpo_log"] = DPO_LOG
    STATE["dpo_config"] = {"beta": DPO_BETA, "lr": DPO_LR, "epochs": DPO_EPOCHS,
                           "pairs": len(PAIRS), "eff_batch": DPO_BS*DPO_ACCUM,
                           "steps": step, "seconds": round(time.perf_counter()-t0,1)}
    save("dpo")
    print(f"\nDPO adapter -> {DPO_OUT}")
    del policy, merged; torch.cuda.empty_cache()
else:
    print("DPO skipped or adapter already present")

print("\n" + "="*70)
print("SAVE THE DPO ADAPTER NOW")
print("="*70)
print(f"  Output -> New Dataset -> save {WORK_DIR}")
print("  /kaggle/working does not survive the session ending.")
''')

md(r"""
---
# 7. SFT vs DPO, and the verdict

Compared **paired** on the same 246 answers. Paired is not a nicety here: the
unpaired SE is ~0.064 guesses, so an unpaired test cannot see anything smaller
than ~0.13. Most plausible DPO effects are below that.
""")

code(r'''
def load_dpo():
    merged = PeftModel.from_pretrained(base_model(), SFT_DIR_ADAPTER).merge_and_unload()
    m = PeftModel.from_pretrained(merged, DPO_OUT)
    m.config.use_cache = True
    return m.eval().cuda()

EV = STATE.get("eval", {})
if RUN_EVAL:
    SC = build_scorer()
    for name, loader in (("sft", load_sft), ("dpo", load_dpo)):
        if name in EV: print(f"{name}: resumed"); continue
        if name == "dpo" and not os.path.exists(os.path.join(DPO_OUT, "adapter_config.json")):
            print("no DPO adapter; skipping"); continue
        print(f"--- evaluating {name} (adaptive @ {ADAPTIVE_THRESHOLD}) ---", flush=True)
        M = loader()
        gs = play(M, SC, VAL_ANSWERS)
        EV[name] = summarize(gs, name)
        # per-bucket disagreement performance
        per_b = defaultdict(lambda: [0, 0])
        for g in gs:
            for b in g.buckets:
                if b: per_b[b][0] += 1
            if g.solved:
                for b in set(x for x in g.buckets if x): per_b[b][1] += 1
        EV[name]["bucket_decisions"] = {k: v[0] for k, v in per_b.items()}
        print(f"  mean={EV[name]['mean']:.4f}  solved={EV[name]['solved']}  "
              f"fail={EV[name]['failure_rate_pct']:.1f}%")
        STATE["eval"] = EV; save(name)
        del M; torch.cuda.empty_cache()

print("\n" + "="*88)
print("PHASE 8 RESULTS")
print("="*88)
h = f"{'run':<26}{'mean':>9}{'solved':>8}{'fail%':>7}{'hardviol%':>11}{'forced%':>9}"
print(h); print("-"*len(h))
print(f"{'filter only (no model)':<26}{PHASE7['control_adaptive20']:>9.4f}"
      f"{'-':>8}{22.8:>7.1f}{'-':>11}{'-':>9}")
for k in ("sft", "dpo"):
    if k not in EV: continue
    r = EV[k]
    print(f"{r['model']:<26}{r['mean']:>9.4f}{r['solved']:>8}"
          f"{r['failure_rate_pct']:>7.1f}{r['hard_mode_violation_pct']:>11.1f}"
          f"{r['forced_pct']:>9.1f}")
print(f"{'classical entropy':<26}{PHASE7['classical_entropy']:>9.4f}{246:>8}"
      f"{0.0:>7.1f}{'-':>11}{'-':>9}")

if "sft" in EV and "dpo" in EV:
    a = np.array(EV["sft"]["per_game"]); b = np.array(EV["dpo"]["per_game"])
    d = a - b                                   # positive => DPO better
    nz = d[d != 0]
    print("\n" + "="*70); print("PAIRED COMPARISON (same 246 answers)"); print("="*70)
    print(f"  SFT {a.mean():.4f}   DPO {b.mean():.4f}   diff {b.mean()-a.mean():+.4f}")
    print(f"  games changed: {len(nz)}  (DPO better {int((d>0).sum())}, "
          f"worse {int((d<0).sum())})")
    if len(nz) > 1:
        se = nz.std(ddof=1)/np.sqrt(len(nz))
        t = nz.mean()/se if se else 0.0
        print(f"  paired t on changed games: t={t:.2f}  "
              f"({'significant' if abs(t)>2 else 'NOT significant'} at ~2 sigma)")
    print(f"  model contribution vs filter-only "
          f"{PHASE7['control_adaptive20']:.4f}:")
    print(f"    SFT {PHASE7['control_adaptive20']-a.mean():+.4f}   "
          f"DPO {PHASE7['control_adaptive20']-b.mean():+.4f}")

if DPO_LOG:
    f_, l_ = DPO_LOG[0], DPO_LOG[-1]
    print("\nDPO TRAINING")
    print(f"  loss      {f_['loss']:.4f} -> {l_['loss']:.4f}")
    print(f"  margin    {f_['margin']:+.4f} -> {l_['margin']:+.4f}")
    print(f"  train acc {f_['acc']:.3f} -> {l_['acc']:.3f}")
    hv = [r for r in DPO_LOG if "val_acc" in r]
    if hv:
        best = max(hv, key=lambda r: r["val_acc"])
        print(f"  HELD-OUT  {hv[0]['val_acc']:.3f} -> {hv[-1]['val_acc']:.3f}"
              f"   (peak {best['val_acc']:.3f} at step {best['step']})")
        if best["step"] < 0.6 * l_["step"]:
            print(f"  NOTE: held-out accuracy peaked at step {best['step']} of "
                  f"{l_['step']}; everything after that was drift,")
            print("        which is the pattern that produced the v1 regression.")
    else:
        print("  no held-out curve - the accuracy above is TRAINING accuracy,")
        print("  which is exactly what made the v1 number meaningless.")
    print("  (a rising held-out curve means the preference generalises; it")
    print("   still does NOT mean gameplay improved - only the games decide)")

save("final")
shutil.make_archive(RESULTS_ZIP[:-4], "zip", RESULTS_ROOT)
print(f"\nZIP: {RESULTS_ZIP} "
      f"({os.path.getsize(RESULTS_ZIP)/2**20:.2f} MiB) - no weights")
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
