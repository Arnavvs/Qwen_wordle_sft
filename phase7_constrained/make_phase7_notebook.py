"""
make_phase7_notebook.py - generate wordle_phase7_kaggle.ipynb.

Retrains tree_salet_endgame (Phase 6 config + dataset, so the reproduction is
comparable) and evaluates three decoders on the same 246 held-out answers:

    unconstrained          free generation, non-words cost the turn
    legal                  argmax over the 12,972 legal guesses
    consistent             argmax over the FEEDBACK-CONSISTENT subset

The `adaptive` decoder exists because of a measurement, not a hunch: the
expert's own target is feedback-INCONSISTENT 59.4% of the time at turn 2 (it
probes with a word that cannot be the answer, to split the space). An always-on
filter forbids that. See section 2.

RESUMABILITY - the Phase 6 session died and took the weights with it, so:

  * training is skipped if an adapter already exists in WORK_DIR or PREV_RUN_DIR
  * the adapter is saved and the notebook prints explicit snapshot instructions
    immediately after training, before any evaluation runs
  * results.json is reloaded at startup, so completed evaluations are skipped
    on a re-run instead of repeated
  * every evaluation writes state before starting the next one

    python make_phase7_notebook.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_phase7_kaggle.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# =============================================================================
md(r"""
# Phase 7 — feedback-consistent constrained decoding

Retrain `tree_salet_endgame` (identical Phase 6 config and dataset), then
compare four decoders on the same 246 held-out answers:

| decoder | admissible set |
|---|---|
| `unconstrained` | anything the model emits |
| `legal` | the 12,972 legal guesses |
| `consistent` | legal **and** consistent with all feedback, every turn |
| `adaptive` | consistent only once the admissible set is small |

## Where Phase 6 left this

| | mean | fail | solved |
|---|---:|---:|---:|
| `tree_salet` [noban] | 5.6870 | 56.1% | 108 |
| `tree_salet_endgame` [noban] | 5.6992 | 56.1% | 108 |
| `tree_salet_endgame` [ban] | 5.4797 | 44.7% | 136 |
| classical `random` | 4.0203 | 0.8% | 244 |

The endgame data changed games not at all; banning repeats did the work. But it
*did* teach something real: on words never trained as targets, k=1 accuracy went
15.25% → 25.42% (paired McNemar, 6 gained / 0 lost, p = 0.031) with zero change
on trained words.

The model reaches ~1.2 candidates by turn 3 and then cannot name the word. Two
measurements say the decoder is the lever:

* **hard-mode violations 31%** — a third of guesses contradict feedback already
  received
* at k=1, a **median of 2** legal words are consistent with the revealed
  feedback, against a pool of 12,972

## What the consistent decoder does

A word is admissible iff it would have produced **exactly** the feedback already
observed, for every guess so far:

```python
allowed = [w for w in allowed if feedback_code(guess, w) == observed_code]
```

Two sets are easy to conflate, and the difference is the entire justification:

| set | pool | uses the answer list? |
|---|---|---|
| candidate set | 2,315 answers | **yes** — privileged, never used |
| **hard-mode set** | 12,972 legal guesses | no — a function of the prompt |

The model never sees the admissible set, its size, the answer, or the answer
list. This is a decoder-side restriction of the same kind as the legal-word
constraint — it is what hard mode enforces and what a human sees on their board.

## Read forced and model-chosen separately

Measured on the expert's own games: at k=1 states the filter **alone** leaves
exactly one word **38.9%** of the time. In those the decoder has solved the
game and the model contributed nothing.

So every decision is tagged:

* `forced` — one admissible word. **Not** a model decision.
* `model_chosen` — several admissible; the model ranked them.

A headline that merged the two would let a decoder win masquerade as a model
win. The unfiltered k=1 probe stays as the measure of what the model knows.

---

## How to run

**STEP 1** — Accelerator: **GPU T4 x2**.

**STEP 2** — Add Input: the **v2** SFT package (with
`data/tree_salet_endgame/`). Optionally also an adapters dataset — if it
contains `tree_salet_endgame`, training is skipped entirely.

**STEP 3** — Run All. ~2.5 h from scratch, ~1 h if the adapter already exists.

**STEP 4 — SAVE THE ADAPTER THE MOMENT SECTION 7 FINISHES.** The notebook
stops and tells you how. Do not skip it; this is why Phase 6 had to be retrained.

**STEP 5** — Download `wordle_phase7_results.zip` (small, no weights).
""")

# =============================================================================
md(r"""
---
# 1. Environment
""")

code(r'''
import os, sys, json, time, math, random, subprocess, platform, shutil
import importlib
from collections import Counter

def _pip(p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)

try:
    import torch
except ImportError:
    _pip("torch"); import torch
for mod, pkg in [("transformers", "transformers>=4.44"), ("peft", "peft>=0.11"),
                 ("accelerate", "accelerate>=0.30")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        _pip(pkg)
import torch, transformers, peft, accelerate
import numpy as np

def fix_torchao_peft_conflict():
    """peft's torchao probe RAISES on an outdated torchao instead of returning
    False, which kills get_peft_model. Kaggle ships an old one; we never use it."""
    try:
        import peft.import_utils as piu
    except Exception as e:
        return f"unavailable ({e})"
    try:
        piu.is_torchao_available(); return "no conflict"
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torchao"], check=False)
    importlib.invalidate_caches()
    try:
        piu.is_torchao_available(); return "resolved: uninstalled torchao"
    except ImportError:
        pass
    piu.is_torchao_available = lambda *a, **k: False
    try:
        import peft.tuners.lora.torchao as t
        t.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
    return "resolved: patched probe"

_FIX = fix_torchao_peft_conflict()
print("=" * 66); print("ENVIRONMENT"); print("=" * 66)
for k, v in [("python", platform.python_version()), ("torch", torch.__version__),
             ("transformers", transformers.__version__), ("peft", peft.__version__),
             ("numpy", np.__version__), ("torchao fix", _FIX)]:
    print(f"{k:<14} {v}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"\nGPU {p.name}  {p.total_memory/2**30:.1f} GiB  x{torch.cuda.device_count()}")
else:
    print("\n!! NO GPU !! Session options -> Accelerator -> GPU T4 x2")
''')

# =============================================================================
md(r"""
---
# 2. Configuration

Training config is byte-identical to Phase 6 so the retrain is a reproduction.
""")

code(r'''
# ============================ EDIT THIS =====================================
DATASET_DIR  = None    # auto-detect; must contain data/tree_salet_endgame/
PREV_RUN_DIR = None    # optional: adapters dataset. If it has
                       # tree_salet_endgame, training is SKIPPED.
# ============================================================================

FORCE_RETRAIN  = False      # True only to deliberately overwrite an adapter
RUN_EVALUATION = True
RUN_BASELINES  = True
RUN_K1_PROBE   = True
RUN_BASE_CONTROL = False    # base Qwen: ~4h in legal mode (pruning cannot help
                            # an untrained model). Leave False unless you have
                            # the session budget; Phase 5 already measured it
                            # at 7.0000 / 100% fail in both modes.

# ---- identical to Phase 6 --------------------------------------------------
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_NAME = "tree_salet_endgame"
USE_NATURAL, USE_ENDGAME, ENDGAME_REPEAT = True, True, 1
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE, NUM_EPOCHS = 2e-4, 2
PER_DEVICE_BS, GRAD_ACCUM = 4, 4
MAX_SEQ_LEN, WARMUP_RATIO, WEIGHT_DECAY = 640, 0.03, 0.0
LR_SCHEDULER, FP16, GRAD_CHECKPOINT = "cosine", True, True
LOGGING_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT = 25, 200, 2
SEED = 20260817

# ---- evaluation ------------------------------------------------------------
MAX_GUESSES, GEN_MAX_NEW_TOKENS, EVAL_BATCH = 6, 8, 16
CONSTRAINED_CHUNK, CONSTRAINED_PRUNE, LENGTH_NORMALISE = 512, True, False

# Applying the consistency filter at EVERY turn is not obviously right, and the
# training data says so. The expert's own target is feedback-inconsistent most
# of the time early on:
#
#     turn 2:  40.6% consistent   <- the expert PROBES: it deliberately plays a
#     turn 3:  91.8% consistent      word that cannot be the answer, to split
#     turn 4: 100.0% consistent      the space
#
# An always-on filter therefore forbids the expert's own turn-2 policy in ~59%
# of games. That is the known reason hard mode scores worse than free mode. So
# we also test an ADAPTIVE filter: probe freely while uncertainty is high, become
# consistent once it is low. The trigger is the admissible-set size, which is
# derivable from the prompt like the filter itself (measured medians: turn 2
# ~211, turn 3 ~7, turn 4 ~2, so 50 separates probing from resolving).
ADAPTIVE_THRESHOLD = 50

# (decoder, ban_repeats). Banning is reported separately from the decoder so the
# two effects never get merged.
EVAL_MATRIX = [
    ("unconstrained", False),
    ("legal",         False),
    ("legal",         True),
    ("consistent",    False),
    ("consistent",    True),
    ("adaptive",      True),
]

K1_MAX_STATES = 80

PHASE6 = {"endgame_legal_noban": 5.6992, "endgame_legal_ban": 5.4797,
          "salet_legal_noban": 5.6870, "k1_top1": 27.5, "k1_median_rank": 4.0,
          "k1_seen": 33.33, "k1_unseen": 25.42, "hard_mode_viol": 31.18,
          "classical_random": 4.0203, "classical_entropy": 3.4431}

WORK_DIR     = "/kaggle/working/wordle_phase7"
RESULTS_ROOT = "/kaggle/working/results_phase7"
RESULTS_ZIP  = "/kaggle/working/wordle_phase7_results.zip"
os.makedirs(WORK_DIR, exist_ok=True); os.makedirs(RESULTS_ROOT, exist_ok=True)

def set_seed_everywhere(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

set_seed_everywhere()
_ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"adapter {ADAPTER_NAME}")
print(f"effective batch {PER_DEVICE_BS} x {GRAD_ACCUM} x {_ngpu} GPU(s) = "
      f"{PER_DEVICE_BS*GRAD_ACCUM*_ngpu}   (Phase 6 ran the same way)")
print(f"eval matrix: {EVAL_MATRIX}")
''')

# =============================================================================
md(r"""
---
# 3. Dataset
""")

code(r'''
import glob

REQUIRED_FILES = [
    "sft_package/data/tree_salet/train.jsonl",
    "sft_package/data/tree_salet_endgame/train.jsonl",
    "sft_package/eval/val_answers.jsonl",
    "sft_package/eval/train_answers.jsonl",
    "code/wordle_solver.py", "code/tree_search.py",
    "code/generate_trajectories.py",
    "artifacts/answers.txt", "artifacts/valid_guesses.txt",
    "artifacts/feedback_matrix.npy",
]

def _has_all(d):
    try:
        return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES)
    except OSError:
        return False

def _search(root, max_depth=7):
    if not os.path.isdir(root):
        return None
    for depth in range(0, 6):
        pat = os.path.join(root, *(["*"] * depth)) if depth else root
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and _has_all(d):
                return d
    base = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    best = None
    for dp, dn, _ in os.walk(root):
        if os.path.abspath(dp).rstrip(os.sep).count(os.sep) - base > max_depth:
            dn[:] = []; continue
        dn[:] = [d for d in dn if not d.startswith(".")]
        if _has_all(dp):
            best = dp if best is None or len(dp) < len(best) else best
            dn[:] = []
    return best

def locate_dataset(explicit=None):
    for c in ([explicit, os.path.join(explicit or "", "kaggle_upload")]
              if explicit else []):
        if _has_all(c):
            return c
    for root in ([explicit] if explicit else []) + ["/kaggle/input", ".",
                                                    "/kaggle/working"]:
        hit = _search(root)
        if hit:
            return hit
    near = [dp for dp, _, _ in os.walk("/kaggle/input")
            if os.path.exists(os.path.join(dp, "sft_package/data/tree_salet/train.jsonl"))]
    raise FileNotFoundError("\n".join(
        ["Dataset not found."] +
        (["Found a package WITHOUT the endgame file:"] + [f"  {n}" for n in near]
         + ["", "Missing: sft_package/data/tree_salet_endgame/train.jsonl",
            "Attach the v2 package (rebuilt after Phase 6)."] if near
         else ["Nothing resembling the SFT package under /kaggle/input."])))

DATA_ROOT = locate_dataset(DATASET_DIR)
SFT_DIR = os.path.join(DATA_ROOT, "sft_package")
ARTIFACTS = os.path.join(DATA_ROOT, "artifacts")
CODE_DIR = os.path.join(DATA_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
print(f"dataset root: {DATA_ROOT}")

from wordle_solver import (load_artifacts, SolverConfig, make_solver, play_game,
                           feedback_code, code_to_pattern, ALL_GREEN)
from generate_trajectories import derive_constraints, render_prompt

BUNDLE = load_artifacts(ARTIFACTS, mmap=True)
VOCAB = BUNDLE.vocab
LEGAL_LOWER = [g.lower() for g in VOCAB.guesses]
LEGAL_GUESSES = set(w.upper() for w in LEGAL_LOWER)
VAL_ANSWERS = [json.loads(l)["answer"].upper()
               for l in open(os.path.join(SFT_DIR, "eval/val_answers.jsonl"),
                             encoding="utf-8")]
assert len(LEGAL_GUESSES) == 12972 and len(VAL_ANSWERS) == 246
print(f"legal {len(LEGAL_GUESSES)}   held-out {len(VAL_ANSWERS)}")
''')

# =============================================================================
md(r"""
---
# 4. Model, mix, and training

Training is **skipped** if an adapter already exists. That is the resumability
guarantee: re-running this notebook after a session dies costs an evaluation,
not a retrain.
""")

code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "right"

def load_base_model():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True)
    m.config.use_cache = False
    return m

def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

NATURAL = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet/train.jsonl"))
ENDGAME = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet_endgame/train.jsonl"))
ROWS = (NATURAL if USE_NATURAL else []) + (ENDGAME * ENDGAME_REPEAT if USE_ENDGAME else [])
random.Random(SEED).shuffle(ROWS)

k1 = [r for r in ROWS if r["meta"]["n_candidates"] == 1]
per = Counter(r["completion"] for r in k1)
DATA_STATS = {"n_rows": len(ROWS), "natural": len(NATURAL), "endgame": len(ENDGAME),
              "k1_rows": len(k1), "k1_words": len(per),
              "k1_paths_per_word": round(len(k1)/max(len(per), 1), 3),
              "turn1_pct": round(100*sum(1 for r in ROWS if r["meta"]["turn"] == 1)/len(ROWS), 2)}
print(f"mix: {DATA_STATS['n_rows']} rows  k=1 paths/word "
      f"{DATA_STATS['k1_paths_per_word']}  turn-1 {DATA_STATS['turn1_pct']}%")
assert abs(DATA_STATS["k1_paths_per_word"] - 3.752) < 0.01, \
    "mix differs from Phase 6 - the retrain would not be a reproduction"
print("mix matches Phase 6 exactly  OK")

class WordleSFTDataset(Dataset):
    def __init__(self, rows, tok, max_len=MAX_SEQ_LEN):
        self.rows, self.tok, self.max_len = rows, tok, max_len
        self.n_truncated = 0; self._c = {}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if i in self._c: return self._c[i]
        r = self.rows[i]
        p = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = self.tok(" " + r["completion"], add_special_tokens=False)["input_ids"]
        c = c + [self.tok.eos_token_id]
        keep = self.max_len - len(c)
        if len(p) > keep:
            p = p[-keep:]; self.n_truncated += 1
        it = {"input_ids": p + c, "labels": [-100]*len(p) + c,
              "attention_mask": [1]*(len(p)+len(c))}
        self._c[i] = it; return it

def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        d = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id]*d)
        out["labels"].append(b["labels"] + [-100]*d)
        out["attention_mask"].append(b["attention_mask"] + [0]*d)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}

def adapter_path(name):
    for base in (WORK_DIR, PREV_RUN_DIR or ""):
        p = os.path.join(base, name)
        if os.path.exists(os.path.join(p, "adapter_config.json")):
            return p
    return None

TRAIN_CONFIG = {}
existing = adapter_path(ADAPTER_NAME)
out_dir = os.path.join(WORK_DIR, ADAPTER_NAME)

if existing and not FORCE_RETRAIN:
    print(f"\nADAPTER FOUND at {existing} - training SKIPPED")
    cp = os.path.join(existing, "training_config.json")
    TRAIN_CONFIG = json.load(open(cp)) if os.path.exists(cp) else {"name": ADAPTER_NAME}
else:
    print(f"\n{'='*66}\nTRAINING {ADAPTER_NAME} on {len(ROWS)} rows\n{'='*66}")
    set_seed_everywhere()
    fix_torchao_peft_conflict()
    base = load_base_model()
    model = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    n = 0
    for _, prm in model.named_parameters():
        if prm.requires_grad and prm.dtype == torch.float16:
            prm.data = prm.data.float(); n += 1
    if n: print(f"  cast {n} trainable tensors fp16 -> fp32 (amp requirement)")
    model.print_trainable_parameters()
    ds = WordleSFTDataset(ROWS, TOKENIZER)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir, num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BS,
            gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER, warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY, fp16=FP16, bf16=False,
            gradient_checkpointing=GRAD_CHECKPOINT, logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS, save_total_limit=SAVE_TOTAL_LIMIT,
            save_strategy="steps", report_to=[], seed=SEED, data_seed=SEED,
            optim="adamw_torch", max_grad_norm=1.0, dataloader_num_workers=2,
            remove_unused_columns=False, disable_tqdm=False),
        train_dataset=ds,
        data_collator=lambda b: collate(b, TOKENIZER.pad_token_id))
    t0 = time.perf_counter(); trainer.train(); secs = time.perf_counter()-t0
    trainer.save_model(out_dir)
    hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    TRAIN_CONFIG = {"name": ADAPTER_NAME, "n_train_examples": len(ROWS),
                    "n_truncated": ds.n_truncated, "training_seconds": round(secs, 1),
                    "global_steps": trainer.state.global_step,
                    "first_loss": hist[0] if hist else None,
                    "final_loss": hist[-1] if hist else None,
                    "data_stats": DATA_STATS, "seed": SEED,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    json.dump(TRAIN_CONFIG, open(os.path.join(out_dir, "training_config.json"),
                                 "w", encoding="utf-8"), indent=2, default=str)
    print(f"\ntrained in {secs/60:.1f} min, {trainer.state.global_step} steps, "
          f"loss {hist[0]:.3f} -> {hist[-1]:.4f}")
    del model, trainer; torch.cuda.empty_cache()

ADAPTER_DIR = adapter_path(ADAPTER_NAME)
assert ADAPTER_DIR, "no adapter available"
print(f"\nadapter: {ADAPTER_DIR}")
''')

# =============================================================================
md(r"""
---
# 5. ⚠ SAVE THE ADAPTER NOW

Phase 6's weights were lost to a session teardown and had to be retrained. Do
this before running anything else.
""")

code(r'''
print("=" * 70)
print("SAVE THE ADAPTER BEFORE CONTINUING")
print("=" * 70)
print(f"  path: {os.path.join(WORK_DIR, ADAPTER_NAME)}")
tot = 0
for r, _, fs in os.walk(WORK_DIR):
    for f in fs:
        tot += os.path.getsize(os.path.join(r, f))
print(f"  size: {tot/2**20:.1f} MiB")
print("""
  1. Right sidebar -> Output
  2. New Dataset  ->  save  /kaggle/working/wordle_phase7
  3. Title it  wordle-phase7-adapter

Next session: Add Input that dataset and set

    PREV_RUN_DIR = "/kaggle/input/.../wordle_phase7"

and section 4 will skip training entirely.

/kaggle/working does NOT survive a session ending. This is the step that was
missed in Phase 6.""")
''')

# =============================================================================
md(r"""
---
# 6. The decoders

`constrained_decode.py` verbatim, then the three-way harness. The
`HardModeFilter` docstring states precisely why the consistent set is not
privileged information.
""")

with open(os.path.join(_paths.CORE, "constrained_decode.py"), encoding="utf-8") as fh:
    SRC = fh.read().rstrip()

code(SRC + r'''


# ---------------------------------------------------------------------------
LEGAL_WORDS_SORTED = sorted(LEGAL_GUESSES)
SCORER = None
_VERIFIED = False

def build_scorer(device="cuda"):
    global SCORER
    if SCORER is None:
        t0 = time.perf_counter()
        SCORER = LegalWordScorer(TOKENIZER, LEGAL_WORDS_SORTED, device=device,
                                 chunk=CONSTRAINED_CHUNK,
                                 length_normalise=LENGTH_NORMALISE)
        print(f"scorer over {SCORER.n} words in {time.perf_counter()-t0:.1f}s")
    return SCORER

def verify_scorer(model):
    """Fatal on failure. Every check here has caught a real bug at least once."""
    global _VERIFIED
    sc = build_scorer()
    probe = GameState("CRANE", VOCAB.n_answers).prompt(1, MAX_GUESSES)
    d = sc.self_test(model, probe)
    print(f"  cache vs naive [{d['dtype']}]: max|delta|={d['max_abs_dev']:.4f} "
          f"nats (tol {d['atol']}), corr={d['corr']:.6f}  OK")
    w, nch = sc.verify_against_full(model, probe)
    print(f"  pruned argmax == full argmax ({w})  OK")

    # ban must mask SCORES, not just the pruning bound
    full = sc.score_all(model, probe)
    want = [sc.words[i] for i in torch.argsort(full, descending=True)[:3].tolist()]
    got, ban = [], []
    for _ in range(3):
        r = sc.select(model, probe, banned=ban or None)
        assert r["word"] not in ban
        got.append(r["word"]); ban.append(r["word"])
    assert got == want, f"banning broke the ranking: {got} != {want}"
    print(f"  sequential banning walks the global ranking {got}  OK")

    # allowed_idx must ALSO mask scores: pick a subset excluding the global best
    gbest = sc.words[int(full.argmax())]
    sub = [i for i in range(sc.n) if sc.words[i] != gbest][:400]
    idx = np.array(sub, dtype=np.int64)
    r = sc.select(model, probe, allowed_idx=idx)
    assert r["word"] != gbest, "allowed_idx leaked a non-admissible word"
    best_sub = max(sub, key=lambda i: full[i].item())
    assert r["word"] == sc.words[best_sub], "allowed_idx did not pick the best admissible"
    print(f"  allowed_idx excludes {gbest}, returns best admissible "
          f"({r['word']})  OK")

    one = np.array([sc.index[gbest]], dtype=np.int64)
    r1 = sc.select(model, probe, allowed_idx=one)
    assert r1["forced"] and not r1["model_chosen"] and r1["n_chunks"] == 0
    print(f"  single admissible word -> forced, no model call  OK")
    _VERIFIED = True
    return d

print("decoders ready.")
''')

# =============================================================================
md(r"""
---
# 7. Evaluation harness

One `GameState` per game carries its own `HardModeFilter`. Turn 1 is a no-op
(the whole legal pool); after that the filter refines on the observed feedback.

An eval-only invariant asserts the true answer stays admissible. It is a check
on the filter, never an input to a decision.
""")

code(r'''
import re
from peft import PeftModel

WORD_RE = re.compile(r"^[A-Za-z]{5}$")

def extract_guess(text):
    line = text.strip().split("\n")[0]
    toks = [t.strip(".,:;!?\"'()[]*_-") for t in line.split()]
    toks = [t for t in toks if t]
    if not toks:
        return None, "invalid_format"
    if WORD_RE.match(toks[0]):
        return toks[0].upper(), "ok"
    return None, "invalid_format"

class GameState:
    __slots__ = ("answer", "history", "cands", "guesses", "patterns", "remaining",
                 "statuses", "forced", "n_allowed", "done", "solved", "mode",
                 "filt", "win_forced")
    def __init__(self, answer, n_answers, mode="legal", filt=None):
        self.answer, self.mode, self.filt = answer, mode, filt
        self.history = []
        self.cands = np.arange(n_answers, dtype=np.int32)
        self.guesses, self.patterns, self.remaining = [], [], []
        self.statuses, self.forced, self.n_allowed = [], [], []
        self.done = self.solved = False
        self.win_forced = None
    def prompt(self, turn, max_guesses):
        h = [(g.lower(), p) for g, p in self.history]
        return render_prompt(turn=turn, history=h,
                             constraints=derive_constraints(h),
                             n_candidates=len(self.cands),
                             guesses_remaining=max_guesses - turn + 1,
                             max_guesses=max_guesses,
                             candidates=None, show_candidate_count=False)

def _apply(g, word, status, turn, max_guesses, forced=None, n_allowed=None):
    g.statuses.append(status); g.forced.append(forced); g.n_allowed.append(n_allowed)
    if status != "ok":
        g.guesses.append(word or "<INVALID>"); g.patterns.append(None)
        g.remaining.append(int(len(g.cands)))
    else:
        code_ = feedback_code(word.lower(), g.answer.lower())
        g.cands = BUNDLE.fb.filter_indices(g.cands, word.lower(), code_)
        if g.filt is not None:
            g.filt.refine(word.lower(), code_)
            assert g.filt.contains(g.answer), (
                f"INVARIANT BROKEN: answer {g.answer} left the admissible set")
        g.history.append((word, code_to_pattern(code_)))
        g.guesses.append(word); g.patterns.append(code_to_pattern(code_))
        g.remaining.append(int(len(g.cands)))
        if code_ == ALL_GREEN:
            g.solved = g.done = True
            g.win_forced = forced
    if turn == max_guesses and not g.solved:
        g.done = True

@torch.no_grad()
def play_games(model, answers, decoder="legal", ban_repeats=False, scorer=None,
               max_guesses=MAX_GUESSES, batch=EVAL_BATCH, log_every_games=60):
    assert decoder in ("unconstrained", "legal", "consistent", "adaptive")
    model.eval()
    tok = TOKENIZER; old = tok.padding_side; tok.padding_side = "left"
    games = [GameState(a, VOCAB.n_answers, mode=decoder,
                       filt=HardModeFilter(LEGAL_LOWER, feedback_code)
                       if decoder in ("consistent", "adaptive") else None)
             for a in answers]
    t0 = time.perf_counter()
    for turn in range(1, max_guesses + 1):
        active = [g for g in games if not g.done]
        if not active:
            break
        if decoder == "unconstrained":
            for s in range(0, len(active), batch):
                ch = active[s:s+batch]
                enc = tok([g.prompt(turn, max_guesses) for g in ch],
                          return_tensors="pt", padding=True, truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)
                out = model.generate(**enc, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None,
                                     top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
                for g, txt in zip(ch, tok.batch_decode(
                        out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    w, st = extract_guess(txt)
                    if w is not None and w not in LEGAL_GUESSES:
                        st = "invalid_word"
                    _apply(g, w, st, turn, max_guesses)
        else:
            for i, g in enumerate(active):
                p = g.prompt(turn, max_guesses)
                banned = list(dict.fromkeys(g.guesses)) if ban_repeats else None
                allowed = None
                if decoder == "consistent":
                    allowed = g.filt.indices(scorer)
                elif decoder == "adaptive":
                    # probe freely while the admissible set is large; become
                    # consistent once the state is nearly resolved
                    if len(g.filt) <= ADAPTIVE_THRESHOLD:
                        allowed = g.filt.indices(scorer)
                r = scorer.select(model, p, banned=banned, allowed_idx=allowed,
                                  prune=CONSTRAINED_PRUNE)
                _apply(g, r["word"], "ok", turn, max_guesses,
                       forced=r["forced"], n_allowed=r["n_allowed"])
                if log_every_games and (i+1) % log_every_games == 0:
                    print(f"      turn {turn}: {i+1}/{len(active)}  "
                          f"{time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  turn {turn}: {sum(1 for g in games if g.done)}/{len(games)} "
              f"done ({time.perf_counter()-t0:.0f}s)", flush=True)
    tok.padding_side = old
    return games

def score_games(games, label, decoder, ban, max_guesses=MAX_GUESSES):
    n = len(games)
    scores = [len(g.guesses) if g.solved else max_guesses+1 for g in games]
    solved = [len(g.guesses) for g in games if g.solved]
    dist = {k: sum(1 for g in games if g.solved and len(g.guesses) == k)
            for k in range(1, max_guesses+1)}
    cum = lambda m: 100*sum(dist[i] for i in range(1, m+1))/n
    st = [s for g in games for s in g.statuses]
    rate = lambda x: 100*sum(1 for s in st if s == x)/max(len(st), 1)
    rep = sum(1 for g in games if len(g.guesses) != len(set(g.guesses)))
    hv = tv = 0
    for g in games:
        seen = []
        for gu, pat in zip(g.guesses, g.patterns):
            if pat is None: continue
            tv += 1
            greens = {}
            for pg, pp in seen:
                for i, (c, t) in enumerate(zip(pg, pp)):
                    if t == "G": greens[i] = c
            if any(gu[i] != c for i, c in greens.items()): hv += 1
            seen.append((gu, pat))
    dec = [f for g in games for f in g.forced if f is not None]
    n_forced = sum(1 for f in dec if f)
    wins_forced = sum(1 for g in games if g.solved and g.win_forced is True)
    wins_model = sum(1 for g in games if g.solved and g.win_forced is False)
    na = [x for g in games for x in g.n_allowed if x is not None]
    return {
        "model": label, "decoder": decoder, "ban_repeats": ban, "n_games": n,
        "mean_failures_as_7": round(sum(scores)/n, 4),
        "mean_solved_only": round(sum(solved)/len(solved), 4) if solved else None,
        "median": float(np.median(solved)) if solved else None,
        "max": max(solved) if solved else None,
        "pct_le3": round(cum(3), 2), "pct_le4": round(cum(4), 2),
        "pct_le5": round(cum(5), 2), "distribution": dist,
        "failures": n-len(solved),
        "failure_rate_pct": round(100*(n-len(solved))/n, 2),
        "solved": len(solved),
        "invalid_format_rate_pct": round(rate("invalid_format"), 2),
        "invalid_word_rate_pct": round(rate("invalid_word"), 2),
        "repeated_guess_game_rate_pct": round(100*rep/n, 2),
        "hard_mode_violation_pct": round(100*hv/max(tv, 1), 2),
        # the split that stops a decoder win looking like a model win
        "decisions": len(dec), "forced_decisions": n_forced,
        "model_decisions": len(dec)-n_forced,
        "forced_decision_pct": round(100*n_forced/max(len(dec), 1), 2),
        "wins_forced": wins_forced, "wins_model_chosen": wins_model,
        "median_n_allowed": float(np.median(na)) if na else None,
        "avg_candidates_after_turn": {
            str(t): round(float(np.mean([g.remaining[t-1] for g in games
                                         if len(g.remaining) >= t])), 2)
            for t in range(1, max_guesses+1)
            if any(len(g.remaining) >= t for g in games)},
    }

def load_adapter(path):
    m = PeftModel.from_pretrained(load_base_model(), path)
    m.config.use_cache = True
    return m.eval().cuda()

print("harness ready. candidate list/count/answer shown: False (never)")
''')

# =============================================================================
md(r"""
---
# 8. Result containers — with resume

`results.json` is reloaded if present, so a re-run skips evaluations that
already completed rather than repeating them.
""")

code(r'''
EVAL_ROWS, GAMES_ALL, BASELINES = {}, {}, []
K1 = {}; K1_ROWS = []

RESULTS_PATH = os.path.join(RESULTS_ROOT, "results.json")
if os.path.exists(RESULTS_PATH):
    try:
        prev = json.load(open(RESULTS_PATH, encoding="utf-8"))
        EVAL_ROWS = prev.get("eval_rows", {})
        BASELINES = prev.get("classical_baselines", [])
        K1 = prev.get("k1_probe", {})
        print(f"RESUMED: {sorted(EVAL_ROWS)}")
    except Exception as e:
        print(f"could not resume ({e}); starting fresh")

def key(dec, ban):
    return f"{dec}__{'ban' if ban else 'noban'}"

def save_state(tag=""):
    json.dump({
        "phase": 7, "adapter": ADAPTER_NAME, "model_name": MODEL_NAME,
        "training": TRAIN_CONFIG, "data_stats": DATA_STATS,
        "eval_rows": EVAL_ROWS, "classical_baselines": BASELINES,
        "k1_probe": K1, "phase6_reference": PHASE6,
        "eval_settings": {
            "legal_words": len(LEGAL_WORDS_SORTED), "chunk": CONSTRAINED_CHUNK,
            "exact_pruning": CONSTRAINED_PRUNE,
            "candidate_list_shown": False, "candidate_count_shown": False,
            "answer_shown": False,
            "consistent_decoder": "legal words w with feedback_code(g,w)==observed "
                                  "for every past guess g; a function of the "
                                  "prompt, never the answer list"},
        "environment": {"torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }, open(RESULTS_PATH, "w", encoding="utf-8"), indent=2, default=str)
    if tag: print(f"    [saved: {tag}]", flush=True)

save_state("init")
print(f"results -> {RESULTS_PATH}")
''')

# =============================================================================
md(r"""
---
# 9. The three-way comparison
""")

code(r'''
if RUN_EVALUATION:
    MODEL = None
    for dec, ban in EVAL_MATRIX:
        k = key(dec, ban)
        if k in EVAL_ROWS:
            print(f"{k}: already done (resumed), skipping"); continue
        print("=" * 66)
        print(f"EVALUATING  decoder={dec}  ban_repeats={ban}")
        print("=" * 66)
        if MODEL is None:
            MODEL = load_adapter(ADAPTER_DIR)
        sc = None
        if dec != "unconstrained":
            sc = build_scorer()
            if not _VERIFIED:
                verify_scorer(MODEL)
        t0 = time.perf_counter()
        gs = play_games(MODEL, VAL_ANSWERS, decoder=dec, ban_repeats=ban, scorer=sc)
        row = score_games(gs, f"{ADAPTER_NAME} [{dec}/{'ban' if ban else 'noban'}]",
                          dec, ban)
        row["eval_seconds"] = round(time.perf_counter()-t0, 1)
        GAMES_ALL[k] = gs; EVAL_ROWS[k] = row
        print(f"  mean={row['mean_failures_as_7']:.4f}  "
              f"fail={row['failure_rate_pct']:.1f}%  solved={row['solved']}  "
              f"invalid={row['invalid_word_rate_pct']:.1f}%  "
              f"repeat={row['repeated_guess_game_rate_pct']:.1f}%  "
              f"hardviol={row['hard_mode_violation_pct']:.1f}%")
        if dec in ("consistent", "adaptive"):
            print(f"  forced {row['forced_decisions']}/{row['decisions']} "
                  f"({row['forced_decision_pct']:.1f}%)   "
                  f"wins: forced {row['wins_forced']} / "
                  f"model-chosen {row['wins_model_chosen']}")
        if dec == "legal" and not ban:
            d = row["mean_failures_as_7"] - PHASE6["endgame_legal_noban"]
            print(f"  PHASE 6 REPRODUCTION: {row['mean_failures_as_7']:.4f} vs "
                  f"{PHASE6['endgame_legal_noban']:.4f} ({d:+.4f})"
                  f"{'  OK' if abs(d) < 0.08 else '  <-- MISMATCH'}")
        print(); save_state(k)
    if MODEL is not None:
        del MODEL; torch.cuda.empty_cache()
else:
    print("evaluation skipped")
''')

# =============================================================================
md(r"""
---
# 10. Classical baselines
""")

code(r'''
if RUN_BASELINES and not BASELINES:
    from tree_search import TreeSearchConfig, TreeSearchSolver
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    lower = [a.lower() for a in VAL_ANSWERS]
    def summarize(label, games, n=len(VAL_ANSWERS)):
        sc_ = [g.score for g in games]; so = [g.n_guesses for g in games if g.solved]
        dist = {k: sum(1 for g in games if g.solved and g.n_guesses == k) for k in range(1, 7)}
        cum = lambda m: 100*sum(dist[i] for i in range(1, m+1))/n
        return {"model": label, "n_games": n,
                "mean_failures_as_7": round(sum(sc_)/n, 4),
                "pct_le3": round(cum(3), 2), "pct_le4": round(cum(4), 2),
                "failure_rate_pct": round(100*(n-len(so))/n, 2),
                "solved": len(so), "classical": True}
    for lbl in ["random", "frequency", "entropy"]:
        sv = make_solver(lbl, BUNDLE.fb, cfgc, BUNDLE.model); sv.reset()
        op = sv.opening_guess() if sv.deterministic else None
        BASELINES.append(summarize(lbl, [play_game(sv, a, first_guess=op) for a in lower]))
        print(f"  {lbl:<12} {BASELINES[-1]['mean_failures_as_7']:.4f}")
    tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                          endgame_threshold=10, opening_guess="salet",
                          max_guesses=MAX_GUESSES)
    sv = TreeSearchSolver(BUNDLE.fb, cfgc, tc)
    BASELINES.append(summarize("tree_salet", [play_game(sv, a, first_guess="salet") for a in lower]))
    print(f"  {'tree_salet':<12} {BASELINES[-1]['mean_failures_as_7']:.4f}")
    save_state("baselines")
else:
    print("baselines skipped or resumed")
''')

# =============================================================================
md(r"""
---
# 11. k=1 probe — unfiltered

Deliberately run **without** the consistency filter. This measures what the
*model* knows, and stays comparable to Phase 5/6 (20.0% → 27.5%). Filtering it
would measure the decoder instead.
""")

code(r'''
def terminal_states(answers, k_target=1, cap=K1_MAX_STATES):
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    sv = make_solver("entropy", BUNDLE.fb, cfgc, BUNDLE.model); sv.reset()
    opener = sv.opening_guess()
    out = []
    for ans in answers:
        low = ans.lower(); cands = np.arange(VOCAB.n_answers, dtype=np.int32)
        hist = []
        for turn in range(1, MAX_GUESSES+1):
            n = int(len(cands))
            if n == k_target:
                out.append({"answer": ans, "turn": turn,
                            "candidates": [VOCAB.answers[i].upper() for i in cands],
                            "prompt": render_prompt(
                                turn=turn, history=list(hist),
                                constraints=derive_constraints(hist), n_candidates=n,
                                guesses_remaining=MAX_GUESSES-turn+1,
                                max_guesses=MAX_GUESSES, candidates=None,
                                show_candidate_count=False)})
                break
            if n <= 1: break
            g = opener if turn == 1 else sv.choose(cands, turn)
            c = feedback_code(g, low); hist.append((g, code_to_pattern(c)))
            cands = BUNDLE.fb.filter_indices(cands, g, c)
            if c == ALL_GREEN: break
        if len(out) >= cap: break
    return out

if RUN_K1_PROBE and "tree_salet_endgame" not in K1:
    sc = build_scorer()
    M = load_adapter(ADAPTER_DIR)
    if not _VERIFIED:
        verify_scorer(M)
    TRAIN_TARGETS = set()
    for f in ("data/tree_salet/train.jsonl", "data/tree_salet_endgame/train.jsonl"):
        p = os.path.join(SFT_DIR, f)
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                TRAIN_TARGETS.add(json.loads(l)["completion"].upper())
    ST = terminal_states(VAL_ANSWERS, 1)
    print(f"{len(ST)} k=1 probe states")
    hit = 0; ranks = []
    t0 = time.perf_counter()
    for i, s in enumerate(ST):
        sc_all = sc.score_all(M, s["prompt"])
        top1 = sc.words[int(sc_all.argmax().item())]
        r = sc.rank_of(sc_all, [s["answer"]]).get(s["answer"])
        hit += int(top1 == s["answer"]); ranks.append(r)
        K1_ROWS.append({"answer": s["answer"], "top1": top1,
                        "correct": top1 == s["answer"], "rank": r,
                        "in_train_targets": s["answer"] in TRAIN_TARGETS})
        if (i+1) % 10 == 0:
            el = time.perf_counter()-t0
            print(f"  {i+1}/{len(ST)}  {el:.0f}s  ~{el/(i+1)*(len(ST)-i-1):.0f}s left",
                  flush=True)
    K1["tree_salet_endgame"] = {
        "n_states": len(ST), "top1_accuracy_pct": round(100*hit/len(ST), 2),
        "median_rank_of_answer": float(np.median([r for r in ranks if r])),
        "chance_top1_pct": round(100/sc.n, 4)}
    print(f"\n  k=1 top-1 {K1['tree_salet_endgame']['top1_accuracy_pct']}%  "
          f"median rank {K1['tree_salet_endgame']['median_rank_of_answer']}  "
          f"[Phase 6: {PHASE6['k1_top1']}%, rank {PHASE6['k1_median_rank']}]")
    del M; torch.cuda.empty_cache()
    save_state("k1")
else:
    print("k=1 probe skipped or resumed")
''')

# =============================================================================
md(r"""
---
# 12. Comparison, decomposition, verdict
""")

code(r'''
import csv

print("=" * 108)
print(f"PHASE 7 - {len(VAL_ANSWERS)} held-out answers")
print("=" * 108)
h = (f"{'run':<40}{'mean':>8}{'fail%':>7}{'solved':>8}{'invalid%':>9}"
     f"{'repeat%':>8}{'hardviol%':>10}{'forced%':>8}")
print(h); print("-"*len(h))
for b in BASELINES:
    print(f"{'classical '+b['model']:<40}{b['mean_failures_as_7']:>8.4f}"
          f"{b['failure_rate_pct']:>7.1f}{b['solved']:>8}{0.0:>9.1f}{0.0:>8.1f}"
          f"{0.0:>10.1f}{'-':>8}")
for k in [key(d, b) for d, b in EVAL_MATRIX]:
    r = EVAL_ROWS.get(k)
    if not r: continue
    fp = (f"{r['forced_decision_pct']:>8.1f}"
          if r["decoder"] in ("consistent", "adaptive") else f"{'-':>8}")
    print(f"{r['model']:<40}{r['mean_failures_as_7']:>8.4f}"
          f"{r['failure_rate_pct']:>7.1f}{r['solved']:>8}"
          f"{r['invalid_word_rate_pct']:>9.1f}{r['repeated_guess_game_rate_pct']:>8.1f}"
          f"{r['hard_mode_violation_pct']:>10.1f}{fp}")

def m(d, b):
    r = EVAL_ROWS.get(key(d, b)); return r["mean_failures_as_7"] if r else None

print("\n" + "="*70); print("DECOMPOSITION"); print("="*70)
u, ln, lb, cn, cb = m("unconstrained", False), m("legal", False), m("legal", True), \
                    m("consistent", False), m("consistent", True)
if u and ln: print(f"  legal-word constraint     : {u:.4f} -> {ln:.4f}  ({ln-u:+.4f})")
if ln and cn: print(f"  + feedback consistency    : {ln:.4f} -> {cn:.4f}  ({cn-ln:+.4f})")
if ln and lb: print(f"  repeat banning (legal)    : {ln:.4f} -> {lb:.4f}  ({lb-ln:+.4f})")
if cn and cb: print(f"  repeat banning (consistent): {cn:.4f} -> {cb:.4f}  ({cb-cn:+.4f})")
ad = m("adaptive", True)
if cb and ad:
    print(f"  adaptive vs always-on      : {cb:.4f} -> {ad:.4f}  ({ad-cb:+.4f})")
    print("    negative => filtering the midgame was costing us the expert's probe")

print("\n" + "="*70); print("WHO WON THE GAMES?"); print("="*70)
for k in (key("consistent", False), key("consistent", True), key("adaptive", True)):
    r = EVAL_ROWS.get(k)
    if not r: continue
    tot = r["wins_forced"] + r["wins_model_chosen"]
    print(f"  {r['model']}")
    print(f"    solved {r['solved']}  =  forced {r['wins_forced']} "
          f"+ model-chosen {r['wins_model_chosen']}")
    if tot:
        print(f"    -> {100*r['wins_forced']/tot:.1f}% of wins were decided by the "
              f"filter alone, not the model")

def verdict():
    ev = []
    best = min([x for x in (cn, cb, ln, lb, m("adaptive", True)) if x is not None],
               default=None)
    if best is None:
        return "INCONCLUSIVE", ["no evaluations completed"]
    rnd = next((b["mean_failures_as_7"] for b in BASELINES if b["model"] == "random"),
               PHASE6["classical_random"])
    ent = next((b["mean_failures_as_7"] for b in BASELINES if b["model"] == "entropy"),
               PHASE6["classical_entropy"])
    ev.append(f"best Phase 7 = {best:.4f}  (Phase 6 best {PHASE6['endgame_legal_ban']:.4f})")
    ev.append(f"classical random {rnd:.4f}, entropy {ent:.4f}")
    r = EVAL_ROWS.get(key("consistent", True)) or EVAL_ROWS.get(key("consistent", False))
    if r:
        tot = r["wins_forced"] + r["wins_model_chosen"]
        ev.append(f"forced decisions {r['forced_decision_pct']:.1f}%; "
                  f"wins forced {r['wins_forced']} vs model {r['wins_model_chosen']}")
        if tot and r["wins_forced"]/tot > 0.6:
            ev.append("most wins came from the filter, not the model")
    if K1.get("tree_salet_endgame"):
        k1 = K1["tree_salet_endgame"]
        ev.append(f"unfiltered k=1 top-1 {k1['top1_accuracy_pct']}% "
                  f"(Phase 6 {PHASE6['k1_top1']}%) - model capability, unchanged by decoding")
    if best <= ent + 0.35: return "SOLVED", ev
    if best < rnd: return "BEATS RANDOM", ev + ["the decoder closed the gap"]
    return "STILL SHORT", ev + ["even a feedback-consistent decoder does not "
                                "reach random elimination"]

V, EV = verdict()
print("\n" + "="*70); print(f"VERDICT: {V}"); print("="*70)
for e in EV: print("  " + e)

FIELDS = ["model", "decoder", "ban_repeats", "n_games", "mean_failures_as_7",
          "failure_rate_pct", "solved", "invalid_word_rate_pct",
          "repeated_guess_game_rate_pct", "hard_mode_violation_pct",
          "decisions", "forced_decisions", "model_decisions",
          "forced_decision_pct", "wins_forced", "wins_model_chosen",
          "median_n_allowed", "pct_le3", "pct_le4", "eval_seconds"]
with open(os.path.join(RESULTS_ROOT, "comparison.csv"), "w", newline="",
          encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in EVAL_ROWS.values(): w.writerow(r)
if K1_ROWS:
    with open(os.path.join(RESULTS_ROOT, "k1_probe.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(K1_ROWS[0].keys()))
        w.writeheader()
        for r in K1_ROWS: w.writerow(r)
json.dump({"verdict": V, "evidence": EV},
          open(os.path.join(RESULTS_ROOT, "verdict.json"), "w", encoding="utf-8"),
          indent=2)
save_state("final")
base = RESULTS_ZIP[:-4]
shutil.make_archive(base, "zip", RESULTS_ROOT)
print(f"\nRESULTS ZIP: {base}.zip "
      f"({os.path.getsize(base+'.zip')/2**20:.2f} MiB) - no weights")
print(f"ADAPTER still at {os.path.join(WORK_DIR, ADAPTER_NAME)} - "
      f"snapshot it as a Dataset if you have not already")
''')

# =============================================================================
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
