"""
make_endgame_notebook.py - generate wordle_endgame_sft_kaggle.ipynb (Phase 6).

Trains ONE adapter (tree_salet_endgame) on natural + synthetic endgame data and
evaluates it against the Phase 5 tree_salet adapter under the constrained
decoder, as a 2x2 over {new, old} x {ban repeats, no ban}.

Structural fixes carried over from the Phase 5 run, all of which cost real time
there:

  1. Every result container is initialised in its own cell BEFORE the loops.
     In Phase 5 `EVAL`/`GAMES`/`PRIMARY` were assigned after the model loop, so
     one KeyboardInterrupt left them undefined and the save cell died with
     NameError.
  2. State is written to disk after EVERY evaluation, not once at the end. A
     Kaggle session teardown destroyed /kaggle/working/results in Phase 5 before
     the zip could be downloaded.
  3. Every result is also PRINTED, so it survives in the notebook itself even if
     the disk is lost.
  4. Every long loop prints progress and an ETA. The Phase 5 terminal probe and
     base-model run looked hung for 45+ minutes because they printed nothing.
  5. The results zip contains no model weights. The Phase 5 zip was 657 MB and
     the browser could not download it.
  6. The self-test tolerance is dtype-aware and paired with a ranking
     correlation check. A single fp32-calibrated threshold aborted the first
     Phase 5 constrained run on ordinary fp16 rounding.
  7. Repeat banning is verified before use, since Phase 6 turns it on.

    python make_endgame_notebook.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_endgame_sft_kaggle.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# =============================================================================
md(r"""
# Phase 6 — endgame-heavy SFT

One adapter, one question:

> Does giving each answer **several distinct constraint paths** to itself fix
> the endgame failure that Phase 5 identified?

## What Phase 5 established

| | |
|---|---|
| best constrained SFT (`tree_salet`) | **5.6870** mean, 56.1% fail |
| classical `random` | 4.0203 |
| classical `entropy` | 3.4431 |
| base Qwen (constrained) | 7.0000, 100% fail |
| k=1 top-1 (state determines the answer) | **20.0%** |
| k=1 median rank of the answer | 7.5 of 12,972 |
| policy transfer at turn 2 | 90% agreement with its own expert |

Verdict **D**: removing invalid words entirely bought 0.49 guesses and the model
still lost to random elimination. The policy transferred; the finish did not.

## The actual defect in the data

Not a shortage of endgame states — a shortage of *paths per word*:

```
tree_salet natural:  1,612 k=1 rows / 1,612 distinct words = 1.00 paths per word
```

Every winning move is demonstrated exactly once, by one memorised route. That
teaches a 1,612-way lookup, not the procedure "read the constraints, produce the
word that satisfies them" — which is what evaluation demands on words whose
single route was never shown.

## The intervention

`build_endgame_dataset.py` rolls out **deliberately imperfect** games (SALET
opener, then randomised continuations) and labels the states with the
`tree_salet` expert — DAgger-style noise injection.

```
endgame synthetic:  6,148 k=1 rows / 2,068 words = 2.97 paths per word
MIXED:              7,760 k=1 rows / 2,068 words = 3.75 paths per word
```

Total training rows go 7,067 → 19,212, and the turn-1 constant drops from 29.3%
of the data to 10.8%.

The model still never sees a candidate list, a candidate count, or the answer.
Held-out answers are never used to generate a state. `verify_endgame_dataset.py`
asserts all of it.

## The hypothesis, and how it can fail

**Hypothesis:** many paths per word teaches constraint-following as a
*procedure*, so it generalises to held-out words; the constrained decoder then
supplies the lexicon.

**Failure mode to watch:** it may simply memorise the training answers harder.
Phase 5 already measured 33.3% top-1 on *seen* words vs 15.3% on unseen. If
Phase 6 moves seen sharply and unseen barely, the intervention did not teach a
procedure — it bought more memorisation, and the approach is wrong rather than
under-trained. The terminal probe reports that split directly and it is the
number to read first.

---

## How to run

**STEP 1** — Accelerator: **GPU T4 x2**.

**STEP 2** — Add Input, both datasets:
* the SFT package **rebuilt after Phase 6** (must contain
  `sft_package/data/tree_salet_endgame/train.jsonl`)
* the adapters dataset holding the Phase 5 `tree_salet` adapter

**STEP 3** — Set `PREV_RUN_DIR` in the config cell to the adapters path.

**STEP 4** — Run All. Budget **~3.5 h**:

| Stage | Time |
|---|---|
| train `tree_salet_endgame` (19,212 rows, 2 epochs) | ~70 min |
| 2x2 constrained evaluation | ~60 min |
| classical baselines | ~10 min |
| terminal probe, 2 models, capped | ~50 min |

**STEP 5** — Download `wordle_endgame_results.zip` (small — no weights) **and**
snapshot `/kaggle/working/wordle_endgame` as a Dataset if you want to keep the
adapter.
""")

# =============================================================================
md(r"""
---
# 1. Environment
""")

code(r'''
import os, sys, json, time, math, random, subprocess, platform, shutil, hashlib
import importlib
from collections import Counter

def _pip(pkg):
    print(f"installing {pkg} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

try:
    import torch
except ImportError:
    _pip("torch"); import torch
for mod, pkg in [("transformers", "transformers>=4.44"),
                 ("peft", "peft>=0.11"),
                 ("accelerate", "accelerate>=0.30")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        _pip(pkg)

import torch, transformers, peft, accelerate
import numpy as np


def fix_torchao_peft_conflict():
    """peft.import_utils.is_torchao_available() RAISES on an outdated torchao
    instead of returning False, which kills get_peft_model. Kaggle ships
    torchao 0.10 while peft wants >= 0.16. We never use torchao."""
    try:
        import peft.import_utils as piu
    except Exception as e:
        return f"peft.import_utils unavailable ({e})"
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
        import peft.tuners.lora.torchao as plt_ao
        plt_ao.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
    return "resolved: patched is_torchao_available -> False"


_TORCHAO_FIX = fix_torchao_peft_conflict()

print("=" * 66); print("ENVIRONMENT"); print("=" * 66)
for k, v in [("python", platform.python_version()), ("torch", torch.__version__),
             ("transformers", transformers.__version__), ("peft", peft.__version__),
             ("accelerate", accelerate.__version__), ("numpy", np.__version__),
             ("torchao fix", _TORCHAO_FIX)]:
    print(f"{k:<14} {v}")
print(f"\nCUDA available    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU               {p.name}")
    print(f"VRAM              {p.total_memory/2**30:.1f} GiB")
    print(f"bf16 supported    {torch.cuda.is_bf16_supported()}  (we use fp16)")
else:
    print("\n!! NO GPU !! Session options -> Accelerator -> GPU T4 x2, then re-run.")
''')

# =============================================================================
md(r"""
---
# 2. Configuration

Everything except the dataset is identical to Phase 4/5, so the comparison
isolates endgame coverage.
""")

code(r'''
# ============================ EDIT THIS =====================================
DATASET_DIR  = None    # auto-detect; must contain data/tree_salet_endgame/
PREV_RUN_DIR = None    # e.g. "/kaggle/input/datasets/<user>/wordle-sft-adapters/kaggle_adapters_upload"
# ============================================================================

RUN_TRAINING   = True
RUN_EVALUATION = True
RUN_BASELINES  = True
RUN_TERMINAL_PROBE = True

# ---- the mix ---------------------------------------------------------------
USE_NATURAL = True     # sft_package/data/tree_salet/train.jsonl        (7,067)
USE_ENDGAME = True     # sft_package/data/tree_salet_endgame/train.jsonl (12,145)
ENDGAME_REPEAT = 1     # oversample the synthetic half (1 = plain mix)

# ---- model / LoRA / optimisation: IDENTICAL to Phase 4 ----------------------
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_NAME = "tree_salet_endgame"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 2
PER_DEVICE_BS = 4
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 640
WARMUP_RATIO  = 0.03
WEIGHT_DECAY  = 0.0
LR_SCHEDULER  = "cosine"
FP16          = True
GRAD_CHECKPOINT = True
LOGGING_STEPS = 25
SAVE_STEPS    = 400
SAVE_TOTAL_LIMIT = 1
SEED          = 20260817

# ---- evaluation ------------------------------------------------------------
MAX_GUESSES        = 6
GEN_MAX_NEW_TOKENS = 8
EVAL_BATCH         = 16
SHOW_CANDIDATE_COUNT = False        # never, in training or evaluation

CONSTRAINED_CHUNK = 512
CONSTRAINED_PRUNE = True
LENGTH_NORMALISE  = False

# The 2x2. Phase 5's tree_salet number (5.6870) was measured with ban=False, so
# the no-ban cells are what make the comparison like-for-like; the ban cells
# measure the decoder change separately.
EVAL_MATRIX = [
    ("tree_salet_endgame", True),
    ("tree_salet_endgame", False),
    ("tree_salet",         True),
    ("tree_salet",         False),   # reproduces Phase 5: expect 5.6870
]

TERMINAL_MAX_STATES = {1: 80, 2: 30, 3: 27}
TERMINAL_MODELS = ["tree_salet_endgame", "tree_salet"]
PROBE_LOG_EVERY = 10

# ---- reference numbers from Phase 5, for the comparison table --------------
PHASE5 = {
    "tree_salet_constrained_noban": 5.6870,
    "tree_salet_constrained_noban_fail": 56.10,
    "tree_salet_unconstrained": 6.1748,
    "base_qwen_constrained": 7.0000,
    "classical_entropy": 3.4431,
    "classical_random": 4.0203,
    "k1_top1": 20.0,
    "k1_seen": 33.33,
    "k1_unseen": 15.25,
    "k1_median_rank": 7.5,
}

# ---- paths -----------------------------------------------------------------
WORK_DIR     = "/kaggle/working/wordle_endgame"
RESULTS_ROOT = "/kaggle/working/results_endgame"
RESULTS_ZIP  = "/kaggle/working/wordle_endgame_results.zip"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

def set_seed_everywhere(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

set_seed_everywhere()
print(f"adapter        {ADAPTER_NAME}")
print(f"mix            natural={USE_NATURAL} endgame={USE_ENDGAME} "
      f"(endgame x{ENDGAME_REPEAT})")
_ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"effective batch {PER_DEVICE_BS} x {GRAD_ACCUM} x {_ngpu} GPU(s) = "
      f"{PER_DEVICE_BS*GRAD_ACCUM*_ngpu}")
if _ngpu > 1:
    print(f"  NOTE: Trainer uses all {_ngpu} GPUs, so the real effective batch "
          f"is {PER_DEVICE_BS*GRAD_ACCUM*_ngpu}. Phase 4 ran the same way "
          f"(7,173 rows -> 448 steps confirms it), so this stays comparable.")
print(f"expected steps ~{2*len(range(19212))//(PER_DEVICE_BS*GRAD_ACCUM*_ngpu)} "
      f"at ~3.3 s/step -> see section 6 for the real count")
print(f"eval matrix    {EVAL_MATRIX}")
print(f"work dir       {WORK_DIR}")
''')

# =============================================================================
md(r"""
---
# 3. Locate and validate the dataset

Fails loudly, and specifically names the endgame file if it is missing — that is
the one new requirement versus Phase 5.
""")

code(r'''
import glob

REQUIRED_FILES = [
    "sft_package/data/tree_salet/train.jsonl",
    "sft_package/data/tree_salet_endgame/train.jsonl",   # NEW in Phase 6
    "sft_package/data/disagreement/contrast.jsonl",
    "sft_package/eval/val_answers.jsonl",
    "sft_package/eval/train_answers.jsonl",
    "code/wordle_solver.py",
    "code/tree_search.py",
    "code/generate_trajectories.py",
    "artifacts/answers.txt",
    "artifacts/valid_guesses.txt",
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
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.abspath(dirpath).rstrip(os.sep).count(os.sep) - base > max_depth:
            dirnames[:] = []; continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _has_all(dirpath):
            if best is None or len(dirpath) < len(best):
                best = dirpath
            dirnames[:] = []
    return best

def locate_dataset(explicit=None):
    if explicit:
        for c in (explicit, os.path.join(explicit, "kaggle_upload")):
            if _has_all(c):
                return c
        hit = _search(explicit)
        if hit:
            return hit
    for root in ("/kaggle/input", ".", "/kaggle/working"):
        hit = _search(root)
        if hit:
            return hit
    msg = ["Could not find the dataset package.", ""]
    eg = "sft_package/data/tree_salet_endgame/train.jsonl"
    near = []
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        if os.path.exists(os.path.join(dirpath, "sft_package/data/tree_salet/train.jsonl")):
            near.append(dirpath)
    if near:
        msg += ["Found a package WITHOUT the Phase 6 endgame file:"] + \
               [f"    {n}" for n in near] + \
               ["", f"Missing: {eg}", "",
                "Rebuild it locally and re-upload:",
                "    python build_endgame_dataset.py",
                "    python verify_endgame_dataset.py",
                "    python prepare_kaggle_dataset.py --dest uploads/kaggle_upload"]
    else:
        msg.append("Nothing that looks like the SFT package under /kaggle/input.")
    raise FileNotFoundError("\n".join(msg))

DATA_ROOT = locate_dataset(DATASET_DIR)
SFT_DIR = os.path.join(DATA_ROOT, "sft_package")
ARTIFACTS = os.path.join(DATA_ROOT, "artifacts")
CODE_DIR = os.path.join(DATA_ROOT, "code")
print(f"dataset root : {DATA_ROOT}")
for f in REQUIRED_FILES:
    p = os.path.join(DATA_ROOT, f)
    print(f"  {os.path.getsize(p)/1024:>10.1f} KiB  {f}")

def ensure_code_on_path():
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)

ensure_code_on_path()
from wordle_solver import (load_artifacts, SolverConfig, make_solver, play_game,
                           feedback_code, code_to_pattern, ALL_GREEN)
from generate_trajectories import derive_constraints, render_prompt

BUNDLE = load_artifacts(ARTIFACTS, mmap=True)
VOCAB = BUNDLE.vocab
LEGAL_GUESSES = set(w.upper() for w in VOCAB.guesses)
VAL_ANSWERS = [json.loads(l)["answer"].upper()
               for l in open(os.path.join(SFT_DIR, "eval/val_answers.jsonl"),
                             encoding="utf-8")]
print(f"\nlegal guesses {len(LEGAL_GUESSES)}   held-out answers {len(VAL_ANSWERS)}")
assert len(LEGAL_GUESSES) == 12972
assert len(VAL_ANSWERS) == 246
''')

# =============================================================================
md(r"""
---
# 4. Model and tokenizer
""")

code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer

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

print(f"model {MODEL_NAME}\nvocab {len(TOKENIZER)}  eos {TOKENIZER.eos_token_id}")
''')

# =============================================================================
md(r"""
---
# 5. The mix

The whole experiment is this cell. Read the printed `paths per word` line: the
natural data is 1.00 and that is the defect being corrected.
""")

code(r'''
from torch.utils.data import Dataset

def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

NATURAL = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet/train.jsonl"))
ENDGAME = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet_endgame/train.jsonl"))

ROWS = []
if USE_NATURAL:
    ROWS += NATURAL
if USE_ENDGAME:
    ROWS += ENDGAME * ENDGAME_REPEAT
random.Random(SEED).shuffle(ROWS)

def bucket(n):
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 \
        else "4-10" if n <= 10 else ">10"

def describe(label, rows):
    n = len(rows)
    bc = Counter(bucket(r["meta"]["n_candidates"]) for r in rows)
    tc = Counter(r["meta"]["turn"] for r in rows)
    k1 = [r for r in rows if r["meta"]["n_candidates"] == 1]
    per = Counter(r["completion"] for r in k1)
    ppw = len(k1) / max(len(per), 1)
    print(f"\n{label}  ({n} rows)")
    print(f"  candidates : {dict(sorted(bc.items()))}")
    print(f"  turns      : {dict(sorted(tc.items()))}")
    print(f"  turn-1 share: {100*tc.get(1,0)/n:.1f}%")
    print(f"  k=1        : {len(k1)} rows / {len(per)} words "
          f"= {ppw:.2f} paths per word")
    return {"n_rows": n, "buckets": dict(bc), "turns": dict(tc),
            "turn1_pct": round(100*tc.get(1,0)/n, 2),
            "k1_rows": len(k1), "k1_words": len(per),
            "k1_paths_per_word": round(ppw, 3)}

print("=" * 66); print("DATASET COMPOSITION"); print("=" * 66)
DATA_STATS = {
    "natural": describe("natural (Phase 4/5)", NATURAL),
    "endgame": describe("endgame synthetic (Phase 6)", ENDGAME),
    "mixed":   describe("MIXED -> training set", ROWS),
}
DATA_STATS["endgame_repeat"] = ENDGAME_REPEAT

print("\n" + "=" * 66)
print(f"paths per word at k=1:  natural {DATA_STATS['natural']['k1_paths_per_word']}"
      f"  ->  mixed {DATA_STATS['mixed']['k1_paths_per_word']}")
print("=" * 66)

# The prompt must be unchanged from Phase 4/5, or nothing is comparable.
assert not any("Possible answers remaining" in r["prompt"] for r in ROWS)
assert all(r["prompt"].rstrip().endswith("Next guess:") for r in ROWS[:2000])
print("\nprompt format unchanged from Phase 4/5  OK")


class WordleSFTDataset(Dataset):
    """prompt -> completion, prompt masked out of the loss."""
    def __init__(self, rows, tokenizer, max_len=MAX_SEQ_LEN):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len
        self.n_truncated = 0
        self._cache = {}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        if i in self._cache:
            return self._cache[i]
        r = self.rows[i]
        p_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = self.tok(" " + r["completion"],
                         add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [self.tok.eos_token_id]
        keep = self.max_len - len(c_ids)
        if len(p_ids) > keep:
            p_ids = p_ids[-keep:]          # drop OLDEST prompt tokens
            self.n_truncated += 1
        item = {"input_ids": p_ids + c_ids,
                "labels": [-100] * len(p_ids) + c_ids,
                "attention_mask": [1] * (len(p_ids) + len(c_ids))}
        self._cache[i] = item
        return item

def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        d = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * d)
        out["labels"].append(b["labels"] + [-100] * d)
        out["attention_mask"].append(b["attention_mask"] + [0] * d)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}

_lens = []
for r in ROWS[:1500]:
    _lens.append(len(TOKENIZER(r["prompt"], add_special_tokens=False)["input_ids"])
                 + len(TOKENIZER(" " + r["completion"],
                                 add_special_tokens=False)["input_ids"]) + 1)
_lens = np.array(_lens)
print(f"token lengths: mean {_lens.mean():.0f}  p95 {np.percentile(_lens,95):.0f}"
      f"  max {_lens.max()}   MAX_SEQ_LEN={MAX_SEQ_LEN} -> "
      f"{100*(_lens>MAX_SEQ_LEN).mean():.2f}% truncated")
''')

# =============================================================================
md(r"""
---
# 6. Train `tree_salet_endgame`

Identical hyperparameters to Phase 4. The dataset is ~2.7x larger, so 2 epochs
is ~2.7x the steps — that is a consequence of more data, not a changed setting.
""")

code(r'''
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments

def make_lora_model():
    fix_torchao_peft_conflict()
    base = load_base_model()
    m = get_peft_model(base, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    # amp keeps fp32 master weights and refuses to unscale fp16 grads.
    n = 0
    for _, p in m.named_parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float(); n += 1
    if n:
        print(f"  cast {n} trainable tensors fp16 -> fp32 (amp requirement)")
    return m

TRAIN_CONFIG = {}
out_dir = os.path.join(WORK_DIR, ADAPTER_NAME)

if RUN_TRAINING:
    print("=" * 66); print(f"TRAINING {ADAPTER_NAME} on {len(ROWS)} rows")
    print("=" * 66)
    set_seed_everywhere()
    ds = WordleSFTDataset(ROWS, TOKENIZER)
    model = make_lora_model()
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE, lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY,
        fp16=FP16, bf16=False, gradient_checkpointing=GRAD_CHECKPOINT,
        logging_steps=LOGGING_STEPS, save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT, save_strategy="steps",
        report_to=[], seed=SEED, data_seed=SEED, optim="adamw_torch",
        max_grad_norm=1.0, dataloader_num_workers=2,
        remove_unused_columns=False, disable_tqdm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, TOKENIZER.pad_token_id))
    t0 = time.perf_counter()
    trainer.train()
    secs = time.perf_counter() - t0
    trainer.save_model(out_dir)
    hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    TRAIN_CONFIG = {
        "name": ADAPTER_NAME, "n_train_examples": len(ROWS),
        "n_truncated": ds.n_truncated, "model_name": MODEL_NAME,
        "learning_rate": LEARNING_RATE, "num_epochs": NUM_EPOCHS,
        "per_device_batch_size": PER_DEVICE_BS,
        "gradient_accumulation": GRAD_ACCUM,
        "effective_batch_size": PER_DEVICE_BS * GRAD_ACCUM,
        "max_seq_len": MAX_SEQ_LEN, "lr_scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO, "fp16": FP16,
        "gradient_checkpointing": GRAD_CHECKPOINT, "seed": SEED,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "training_seconds": round(secs, 1),
        "global_steps": trainer.state.global_step,
        "first_loss": hist[0] if hist else None,
        "final_loss": hist[-1] if hist else None,
        "data_stats": DATA_STATS,
    }
    json.dump(TRAIN_CONFIG, open(os.path.join(out_dir, "training_config.json"),
                                 "w", encoding="utf-8"), indent=2, default=str)
    print(f"\ntrained in {secs/60:.1f} min, {trainer.state.global_step} steps, "
          f"loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    del model, trainer; torch.cuda.empty_cache()
else:
    p = os.path.join(out_dir, "training_config.json")
    if os.path.exists(p):
        TRAIN_CONFIG = json.load(open(p))
        print(f"training skipped; existing adapter at {out_dir}")
    else:
        print("training skipped and no existing adapter")
''')

# =============================================================================
md(r"""
---
# 7. Constrained decoding

Unchanged from Phase 5 so the comparison holds. The module docstring below is
the full specification; the short version is that the model never emits free
text — all 12,972 legal words are scored and the argmax is played.

Phase 6 turns **repeat banning on**, so the verification cell also checks that
sequential banning walks the global ranking exactly. That check caught a real
bug in Phase 5 (the ban masked the pruning bound but not the final scores).
""")

with open(os.path.join(_paths.CORE, "constrained_decode.py"), encoding="utf-8") as fh:
    CONSTRAINED_SRC = fh.read().rstrip()

code(CONSTRAINED_SRC + r'''


# ---------------------------------------------------------------------------
LEGAL_WORDS_SORTED = sorted(LEGAL_GUESSES)
assert len(LEGAL_WORDS_SORTED) == 12972

SCORER = None
_SCORER_VERIFIED = False

def build_scorer(device="cuda"):
    global SCORER
    if SCORER is None:
        t0 = time.perf_counter()
        SCORER = LegalWordScorer(TOKENIZER, LEGAL_WORDS_SORTED, device=device,
                                 chunk=CONSTRAINED_CHUNK,
                                 length_normalise=LENGTH_NORMALISE)
        print(f"scorer over {SCORER.n} words (max {SCORER.max_len} tokens) "
              f"in {time.perf_counter()-t0:.1f}s")
    return SCORER

def verify_scorer(model):
    """Fatal on failure, deliberately."""
    global _SCORER_VERIFIED
    sc = build_scorer()
    probe = GameState("CRANE", VOCAB.n_answers).prompt(1, MAX_GUESSES)
    d = sc.self_test(model, probe)
    print(f"  cache vs naive [{d['dtype']}]: max|delta| = {d['max_abs_dev']:.4f} "
          f"nats (tol {d['atol']}) over a {d['spread']:.1f}-nat spread, "
          f"ranking corr = {d['corr']:.6f}  OK")
    if CONSTRAINED_PRUNE:
        w, nch = sc.verify_against_full(model, probe)
        print(f"  pruned argmax == full argmax ({w}), {nch}/"
              f"{-(-sc.n // CONSTRAINED_CHUNK)} chunks  OK")
    full = sc.score_all(model, probe)
    want = [sc.words[i] for i in torch.argsort(full, descending=True)[:4].tolist()]
    got, ban = [], []
    for _ in range(4):
        w2, _, _, _ = sc.argmax(model, probe, banned=ban or None)
        assert w2 not in ban, f"banned word {w2} returned"
        got.append(w2); ban.append(w2)
    assert got == want, f"banning broke the ranking: {got} != {want}"
    print(f"  sequential banning walks the global ranking {got}  OK")
    _SCORER_VERIFIED = True
    return d

print("constrained decoder ready.")
''')

# =============================================================================
md(r"""
---
# 8. Evaluation harness

Identical protocol to Phase 5. The only new knob is `ban_repeats`, which is a
decoder setting, not a change to what the model sees.
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
    __slots__ = ("answer", "history", "cands", "guesses", "patterns",
                 "remaining", "statuses", "margins", "done", "solved", "mode")
    def __init__(self, answer, n_answers, mode="constrained"):
        self.answer = answer; self.mode = mode
        self.history = []
        self.cands = np.arange(n_answers, dtype=np.int32)
        self.guesses, self.patterns, self.remaining = [], [], []
        self.statuses, self.margins = [], []
        self.done = self.solved = False
    def prompt(self, turn, max_guesses):
        return render_prompt(
            turn=turn,
            history=[(g.lower(), p) for g, p in self.history],
            constraints=derive_constraints([(g.lower(), p) for g, p in self.history]),
            n_candidates=len(self.cands),   # analysis only
            guesses_remaining=max_guesses - turn + 1,
            max_guesses=max_guesses,
            candidates=None, show_candidate_count=False)

def _apply(g, word, status, turn, max_guesses, margin=None):
    g.statuses.append(status); g.margins.append(margin)
    if status != "ok":
        g.guesses.append(word or "<INVALID>")
        g.patterns.append(None); g.remaining.append(int(len(g.cands)))
    else:
        code_ = feedback_code(word.lower(), g.answer.lower())
        pat = code_to_pattern(code_)
        g.cands = BUNDLE.fb.filter_indices(g.cands, word.lower(), code_)
        g.history.append((word, pat))
        g.guesses.append(word); g.patterns.append(pat)
        g.remaining.append(int(len(g.cands)))
        if code_ == ALL_GREEN:
            g.solved = g.done = True
    if turn == max_guesses and not g.solved:
        g.done = True

PLAY_STATS = {}

@torch.no_grad()
def play_games(model, answers, mode="constrained", ban_repeats=False,
               scorer=None, max_guesses=MAX_GUESSES, batch=EVAL_BATCH):
    model.eval()
    tok = TOKENIZER
    old = tok.padding_side
    tok.padding_side = "left"
    games = [GameState(a, VOCAB.n_answers, mode=mode) for a in answers]
    chunks = []
    t0 = time.perf_counter()
    for turn in range(1, max_guesses + 1):
        active = [g for g in games if not g.done]
        if not active:
            break
        if mode == "constrained":
            for i, g in enumerate(active):
                p = g.prompt(turn, max_guesses)
                banned = list(dict.fromkeys(g.guesses)) if ban_repeats else None
                w, s, nch, margin = scorer.argmax(model, p, banned=banned,
                                                  prune=CONSTRAINED_PRUNE)
                chunks.append(nch)
                _apply(g, w, "ok", turn, max_guesses, margin)
                if (i + 1) % 60 == 0:
                    el = time.perf_counter() - t0
                    print(f"      turn {turn}: {i+1}/{len(active)} games  "
                          f"{el:.0f}s  avg {np.mean(chunks):.1f} chunks",
                          flush=True)
        else:
            for s0 in range(0, len(active), batch):
                chunk = active[s0:s0 + batch]
                enc = tok([g.prompt(turn, max_guesses) for g in chunk],
                          return_tensors="pt", padding=True, truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)
                out = model.generate(**enc, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None,
                                     top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
                for g, txt in zip(chunk, texts):
                    w, st = extract_guess(txt)
                    if w is not None and w not in LEGAL_GUESSES:
                        st = "invalid_word"
                    _apply(g, w, st, turn, max_guesses)
        done = sum(1 for g in games if g.done)
        print(f"  turn {turn}: {done}/{len(games)} finished "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    tok.padding_side = old
    PLAY_STATS.clear()
    if chunks:
        PLAY_STATS.update(avg_chunks_per_decision=round(float(np.mean(chunks)), 2),
                          unpruned_chunks=-(-len(LEGAL_GUESSES)//CONSTRAINED_CHUNK))
    return games

def score_games(games, label, max_guesses=MAX_GUESSES):
    n = len(games)
    scores = [len(g.guesses) if g.solved else max_guesses + 1 for g in games]
    solved = [len(g.guesses) for g in games if g.solved]
    dist = {k: sum(1 for g in games if g.solved and len(g.guesses) == k)
            for k in range(1, max_guesses + 1)}
    cum = lambda m: 100 * sum(dist[i] for i in range(1, m + 1)) / n
    st = [s for g in games for s in g.statuses]
    rate = lambda x: 100 * sum(1 for s in st if s == x) / max(len(st), 1)
    rep = sum(1 for g in games if len(g.guesses) != len(set(g.guesses)))
    hv = tv = 0
    for g in games:
        seen = []
        for gu, pat in zip(g.guesses, g.patterns):
            if pat is None:
                continue
            tv += 1
            greens = {}
            for pg, pp in seen:
                for i, (ch, t) in enumerate(zip(pg, pp)):
                    if t == "G":
                        greens[i] = ch
            if any(gu[i] != ch for i, ch in greens.items()):
                hv += 1
            seen.append((gu, pat))
    margins = [m for g in games for m in g.margins if m is not None]
    return {
        "model": label, "mode": games[0].mode, "n_games": n,
        "mean_failures_as_7": round(sum(scores)/n, 4),
        "mean_solved_only": round(sum(solved)/len(solved), 4) if solved else None,
        "median": float(np.median(solved)) if solved else None,
        "max": max(solved) if solved else None,
        "pct_le3": round(cum(3), 2), "pct_le4": round(cum(4), 2),
        "pct_le5": round(cum(5), 2), "pct_le6": round(cum(6), 2),
        "distribution": dist, "failures": n - len(solved),
        "failure_rate_pct": round(100*(n-len(solved))/n, 2),
        "turns_generated": len(st),
        "invalid_format_rate_pct": round(rate("invalid_format"), 2),
        "invalid_word_rate_pct": round(rate("invalid_word"), 2),
        "repeated_guess_game_rate_pct": round(100*rep/n, 2),
        "avg_unique_guesses_per_game": round(
            float(np.mean([len(set(g.guesses)) for g in games])), 3),
        "hard_mode_violation_pct": round(100*hv/max(tv, 1), 2),
        "avg_candidates_after_turn": {
            str(t): round(float(np.mean([g.remaining[t-1] for g in games
                                         if len(g.remaining) >= t])), 2)
            for t in range(1, max_guesses+1)
            if any(len(g.remaining) >= t for g in games)},
        "avg_decision_margin": round(float(np.mean(margins)), 4) if margins else None,
    }

def load_adapter(name):
    path = os.path.join(WORK_DIR, name)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        alt = os.path.join(PREV_RUN_DIR or "", name)
        assert os.path.exists(os.path.join(alt, "adapter_config.json")), (
            f"no adapter for {name} in {path} or {alt}. For '{ADAPTER_NAME}' "
            f"run section 6; for 'tree_salet' point PREV_RUN_DIR at the Phase 5 "
            f"adapters dataset.")
        path = alt
    m = PeftModel.from_pretrained(load_base_model(), path)
    m.config.use_cache = True
    return m.eval().cuda()

print("harness ready.  candidate count/list/answer shown: False (never)")
''')

# =============================================================================
md(r"""
---
# 9. Result containers

Their own cell, before any loop. In Phase 5 these lived at the end of the
evaluation cell and a single interrupt left them undefined, which broke the save
step. `save_state()` is called after **every** evaluation so a session teardown
cannot cost more than one cell.
""")

code(r'''
EVAL_ROWS = {}          # (adapter, ban) -> metrics
GAMES_ALL = {}          # (adapter, ban) -> [GameState]
BASELINES = []
TERMINAL = {}
TERMINAL_ROWS = []
TERMINAL_SPLIT = {}
OPENERS = {}
DISAGREE = {}
EXAMPLES = {}
TRAIN_TARGET_WORDS = set()

def _key(a, ban):
    return f"{a}__{'ban' if ban else 'noban'}"

def save_state(tag=""):
    """Write everything known so far. Cheap, and called constantly."""
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "peft": peft.__version__,
           "numpy": np.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "fp16": FP16, "seed": SEED,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload = {
        "phase": 6, "note": f"{len(VAL_ANSWERS)} HELD-OUT answers.",
        "adapter": ADAPTER_NAME, "model_name": MODEL_NAME,
        "training": TRAIN_CONFIG, "data_stats": DATA_STATS,
        "eval_rows": EVAL_ROWS, "classical_baselines": BASELINES,
        "terminal_probe": TERMINAL, "terminal_seen_vs_unseen": TERMINAL_SPLIT,
        "openers": OPENERS, "disagreement": DISAGREE, "examples": EXAMPLES,
        "phase5_reference": PHASE5, "environment": env,
        "eval_settings": {
            "constrained_legal_words": len(LEGAL_WORDS_SORTED),
            "chunk": CONSTRAINED_CHUNK, "exact_pruning": CONSTRAINED_PRUNE,
            "length_normalise": LENGTH_NORMALISE,
            "candidate_list_shown": False, "candidate_count_shown": False,
            "answer_shown": False,
            "score": "sum log P(token | prompt, prefix) over ' '+WORD tokens "
                     "and EOS; argmax over the legal set."},
    }
    with open(os.path.join(RESULTS_ROOT, "results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    if tag:
        print(f"    [saved: {tag}]", flush=True)

save_state("init")
print(f"containers ready; results -> {RESULTS_ROOT}/results.json")
''')

# =============================================================================
md(r"""
---
# 10. The 2×2

`tree_salet` + `noban` reproduces the Phase 5 measurement (**5.6870**). If it
does not come back within rounding, the environment differs and nothing else in
this notebook should be trusted.
""")

code(r'''
if RUN_EVALUATION:
    for adapter, ban in EVAL_MATRIX:
        k = _key(adapter, ban)
        if k in EVAL_ROWS:
            print(f"{k}: already done, skipping"); continue
        print("=" * 66)
        print(f"EVALUATING {adapter}  ban_repeats={ban}  on {len(VAL_ANSWERS)} games")
        print("=" * 66)
        try:
            model = load_adapter(adapter)
        except AssertionError as e:
            print(f"  SKIPPED: {e}\n"); continue
        if not _SCORER_VERIFIED:
            verify_scorer(model)
        sc = build_scorer()
        t0 = time.perf_counter()
        gs = play_games(model, VAL_ANSWERS, mode="constrained",
                        ban_repeats=ban, scorer=sc)
        row = score_games(gs, f"{adapter} [{'ban' if ban else 'noban'}]")
        row.update(PLAY_STATS)
        row["adapter"] = adapter
        row["ban_repeats"] = ban
        row["eval_seconds"] = round(time.perf_counter() - t0, 1)
        GAMES_ALL[k] = gs
        EVAL_ROWS[k] = row
        print(f"  mean={row['mean_failures_as_7']:.4f}  "
              f"fail={row['failure_rate_pct']:.1f}%  "
              f"solved={100-row['failure_rate_pct']:.1f}%  "
              f"repeat={row['repeated_guess_game_rate_pct']:.1f}%  "
              f"invalid={row['invalid_word_rate_pct']:.1f}%  "
              f"({row['eval_seconds']:.0f}s)")
        if adapter == "tree_salet" and not ban:
            d = row["mean_failures_as_7"] - PHASE5["tree_salet_constrained_noban"]
            print(f"  PHASE 5 REPRODUCTION CHECK: {row['mean_failures_as_7']:.4f} "
                  f"vs 5.6870 (delta {d:+.4f})"
                  f"{'  OK' if abs(d) < 0.05 else '  <-- MISMATCH, investigate'}")
        print()
        save_state(k)
        del model; torch.cuda.empty_cache()
else:
    print("evaluation skipped")
''')

# =============================================================================
md(r"""
---
# 11. Classical baselines on the identical 246 answers
""")

code(r'''
if RUN_BASELINES and not BASELINES:
    ensure_code_on_path()
    from tree_search import TreeSearchConfig, TreeSearchSolver
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    lower = [a.lower() for a in VAL_ANSWERS]

    def summarize(label, games, n=len(VAL_ANSWERS)):
        scores = [g.score for g in games]
        solved = [g.n_guesses for g in games if g.solved]
        dist = {k: sum(1 for g in games if g.solved and g.n_guesses == k)
                for k in range(1, 7)}
        cum = lambda m: 100*sum(dist[i] for i in range(1, m+1))/n
        return {"model": label, "n_games": n,
                "mean_failures_as_7": round(sum(scores)/n, 4),
                "median": float(np.median(solved)) if solved else None,
                "max": max(solved) if solved else None,
                "pct_le3": round(cum(3), 2), "pct_le4": round(cum(4), 2),
                "pct_le5": round(cum(5), 2),
                "failure_rate_pct": round(100*(n-len(solved))/n, 2),
                "classical": True}

    for lbl in ["random", "frequency", "entropy"]:
        sv = make_solver(lbl, BUNDLE.fb, cfgc, BUNDLE.model); sv.reset()
        op = sv.opening_guess() if sv.deterministic else None
        BASELINES.append(summarize(lbl, [play_game(sv, a, first_guess=op)
                                         for a in lower]))
        print(f"  {lbl:<12} {BASELINES[-1]['mean_failures_as_7']:.4f}")
    tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                          endgame_threshold=10, opening_guess="salet",
                          max_guesses=MAX_GUESSES)
    sv = TreeSearchSolver(BUNDLE.fb, cfgc, tc)
    BASELINES.append(summarize("tree_salet", [play_game(sv, a, first_guess="salet")
                                              for a in lower]))
    print(f"  {'tree_salet':<12} {BASELINES[-1]['mean_failures_as_7']:.4f}")
    save_state("baselines")
else:
    print("baselines skipped or already computed")
''')

# =============================================================================
md(r"""
---
# 12. Terminal-state probe

The decisive measurement. The classical solver — not the model — is rolled to
states with exactly `k` candidates. At `k=1` the constraints determine the
answer; nothing remains but naming it.

**Read the seen-vs-unseen split first.** If `tree_salet_endgame` improves on
seen words but not unseen, it memorised harder rather than learning the
procedure, and the intervention failed even if the mean improved.
""")

code(r'''
def terminal_states(answers, ks, solver_label="entropy"):
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    sv = make_solver(solver_label, BUNDLE.fb, cfgc, BUNDLE.model); sv.reset()
    opener = sv.opening_guess()
    out = {k: [] for k in ks}
    want = set(ks)
    for ans in answers:
        low = ans.lower()
        cands = np.arange(VOCAB.n_answers, dtype=np.int32)
        hist, seen = [], set()
        for turn in range(1, MAX_GUESSES + 1):
            n = int(len(cands))
            if n in want and n not in seen:
                seen.add(n)
                out[n].append({
                    "answer": ans, "k": n, "turn": turn,
                    "candidates": [VOCAB.answers[i].upper() for i in cands],
                    "prompt": render_prompt(
                        turn=turn, history=list(hist),
                        constraints=derive_constraints(hist), n_candidates=n,
                        guesses_remaining=MAX_GUESSES - turn + 1,
                        max_guesses=MAX_GUESSES, candidates=None,
                        show_candidate_count=False)})
            if n <= 1 or not want - seen:
                break
            g = opener if turn == 1 else sv.choose(cands, turn)
            c = feedback_code(g, low)
            hist.append((g, code_to_pattern(c)))
            cands = BUNDLE.fb.filter_indices(cands, g, c)
            if c == ALL_GREEN:
                break
    return out

@torch.no_grad()
def probe_terminal(model, states, label, caps):
    sc = build_scorer()
    rows, per_k = [], {}
    plan = {k: (states[k] if caps.get(k) is None else states[k][:caps[k]])
            for k in sorted(states)}
    total = sum(len(v) for v in plan.values())
    print(f"  scoring {total} states "
          f"({', '.join(f'k={k}:{len(v)}' for k, v in plan.items())})")
    t0, done = time.perf_counter(), 0
    for k, use in plan.items():
        hit1 = hitc = 0
        ranks, ranks_in = [], []
        for st in use:
            scores = sc.score_all(model, st["prompt"])
            top1 = sc.words[int(scores.argmax().item())]
            r = sc.rank_of(scores, [st["answer"]]).get(st["answer"])
            cs = sorted(((sc.words[sc.index[c]], scores[sc.index[c]].item())
                         for c in st["candidates"] if c in sc.index),
                        key=lambda x: -x[1])
            r_in = next((i+1 for i, (w, _) in enumerate(cs)
                         if w == st["answer"]), None)
            hit1 += int(top1 == st["answer"])
            hitc += int(top1 in set(st["candidates"]))
            if r: ranks.append(r)
            if r_in: ranks_in.append(r_in)
            rows.append({"model": label, "k": k, "answer": st["answer"],
                         "top1": top1, "correct": top1 == st["answer"],
                         "top1_is_candidate": top1 in set(st["candidates"]),
                         "rank_of_answer": r, "rank_within_candidates": r_in,
                         "answer_in_train_targets":
                             st["answer"] in TRAIN_TARGET_WORDS})
            done += 1
            if PROBE_LOG_EVERY and done % PROBE_LOG_EVERY == 0:
                el = time.perf_counter() - t0
                print(f"    {done}/{total}  {el:.0f}s elapsed, "
                      f"~{el/done*(total-done):.0f}s left", flush=True)
        n = max(len(use), 1)
        per_k[k] = {"n_states": len(use),
                    "top1_accuracy_pct": round(100*hit1/n, 2),
                    "top1_is_candidate_pct": round(100*hitc/n, 2),
                    "median_rank_of_answer": float(np.median(ranks)) if ranks else None,
                    "mean_rank_within_candidates": round(float(np.mean(ranks_in)), 3)
                        if ranks_in else None,
                    "chance_top1_pct": round(100/sc.n, 4)}
    return per_k, rows

if RUN_TERMINAL_PROBE:
    if not TRAIN_TARGET_WORDS:
        for f in ("data/tree_salet/train.jsonl",
                  "data/tree_salet_endgame/train.jsonl"):
            p = os.path.join(SFT_DIR, f)
            if os.path.exists(p):
                for l in open(p, encoding="utf-8"):
                    TRAIN_TARGET_WORDS.add(json.loads(l)["completion"].upper())
    print(f"distinct training targets: {len(TRAIN_TARGET_WORDS)}")

    if "STATES" not in globals():
        STATES = terminal_states(VAL_ANSWERS, sorted(TERMINAL_MAX_STATES))
    for k in sorted(STATES):
        print(f"  k={k}: {len(STATES[k])} states available")

    for adapter in TERMINAL_MODELS:
        if adapter in TERMINAL:
            continue
        print(f"\n--- terminal probe: {adapter} ---", flush=True)
        try:
            model = load_adapter(adapter)
        except AssertionError as e:
            print(f"  SKIPPED: {e}"); continue
        if not _SCORER_VERIFIED:
            verify_scorer(model)
        t0 = time.perf_counter()
        per_k, rows = probe_terminal(model, STATES, adapter, TERMINAL_MAX_STATES)
        TERMINAL[adapter] = per_k
        TERMINAL_ROWS.extend(rows)
        for k, r in sorted(per_k.items()):
            print(f"  k={k}  n={r['n_states']:>3}  "
                  f"top1={r['top1_accuracy_pct']:>6.2f}%  "
                  f"in_candidates={r['top1_is_candidate_pct']:>6.2f}%  "
                  f"median_rank={r['median_rank_of_answer']}")
        print(f"  ({time.perf_counter()-t0:.0f}s)")
        save_state(f"terminal:{adapter}")
        del model; torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("k=1 TOP-1, SPLIT BY WHETHER THE ANSWER WAS EVER A TRAINING TARGET")
    print("=" * 70)
    hdr = f"{'model':<24}{'seen n':>8}{'seen':>9}{'unseen n':>10}{'unseen':>9}"
    print(hdr); print("-" * len(hdr))
    for label in sorted({r["model"] for r in TERMINAL_ROWS}):
        sub = [r for r in TERMINAL_ROWS if r["model"] == label and r["k"] == 1]
        seen = [r for r in sub if r["answer_in_train_targets"]]
        uns = [r for r in sub if not r["answer_in_train_targets"]]
        f = lambda xs: round(100*sum(x["correct"] for x in xs)/len(xs), 2) if xs else None
        TERMINAL_SPLIT[label] = {"seen_n": len(seen), "seen_acc_pct": f(seen),
                                 "unseen_n": len(uns), "unseen_acc_pct": f(uns)}
        print(f"{label:<24}{len(seen):>8}{str(f(seen)):>9}"
              f"{len(uns):>10}{str(f(uns)):>9}")
    print(f"\nPhase 5 tree_salet was: seen {PHASE5['k1_seen']}%  "
          f"unseen {PHASE5['k1_unseen']}%  (overall {PHASE5['k1_top1']}%)")
    save_state("terminal-split")
else:
    print("terminal probe skipped")
''')

# =============================================================================
md(r"""
---
# 13. Comparison and verdict
""")

code(r'''
import csv

print("=" * 96)
print(f"PHASE 6 vs PHASE 5  -  {len(VAL_ANSWERS)} held-out answers")
print("=" * 96)
hdr = (f"{'model':<34}{'mean':>9}{'fail%':>8}{'solved%':>9}{'repeat%':>9}"
       f"{'<=3':>7}{'<=4':>7}")
print(hdr); print("-" * len(hdr))
for b in BASELINES:
    print(f"{'classical ' + b['model']:<34}{b['mean_failures_as_7']:>9.4f}"
          f"{b['failure_rate_pct']:>8.1f}{100-b['failure_rate_pct']:>9.1f}"
          f"{0.0:>9.1f}{b['pct_le3']:>7.1f}{b['pct_le4']:>7.1f}")
print(f"{'PHASE 5 tree_salet [noban]':<34}"
      f"{PHASE5['tree_salet_constrained_noban']:>9.4f}"
      f"{PHASE5['tree_salet_constrained_noban_fail']:>8.1f}"
      f"{100-PHASE5['tree_salet_constrained_noban_fail']:>9.1f}"
      f"{30.5:>9.1f}{17.1:>7.1f}{30.1:>7.1f}")
for k, r in EVAL_ROWS.items():
    print(f"{r['model']:<34}{r['mean_failures_as_7']:>9.4f}"
          f"{r['failure_rate_pct']:>8.1f}{100-r['failure_rate_pct']:>9.1f}"
          f"{r['repeated_guess_game_rate_pct']:>9.1f}"
          f"{r['pct_le3']:>7.1f}{r['pct_le4']:>7.1f}")

# ---- the two effects, separated -------------------------------------------
def get(a, ban):
    r = EVAL_ROWS.get(_key(a, ban))
    return r["mean_failures_as_7"] if r else None

new_nb, new_b = get(ADAPTER_NAME, False), get(ADAPTER_NAME, True)
old_nb, old_b = get("tree_salet", False), get("tree_salet", True)

print("\n" + "=" * 70); print("EFFECT DECOMPOSITION"); print("=" * 70)
if new_nb is not None and old_nb is not None:
    print(f"  endgame data alone (noban): {old_nb:.4f} -> {new_nb:.4f}  "
          f"({new_nb-old_nb:+.4f})")
if old_b is not None and old_nb is not None:
    print(f"  repeat banning alone (old): {old_nb:.4f} -> {old_b:.4f}  "
          f"({old_b-old_nb:+.4f})")
if new_b is not None and old_nb is not None:
    print(f"  both combined             : {old_nb:.4f} -> {new_b:.4f}  "
          f"({new_b-old_nb:+.4f})")

# ---- verdict ---------------------------------------------------------------
def verdict():
    ev = []
    best = min([v for v in (new_nb, new_b) if v is not None], default=None)
    if best is None:
        return "INCONCLUSIVE", ["the new adapter was not evaluated"]
    rnd = next((b["mean_failures_as_7"] for b in BASELINES
                if b["model"] == "random"), PHASE5["classical_random"])
    ent = next((b["mean_failures_as_7"] for b in BASELINES
                if b["model"] == "entropy"), PHASE5["classical_entropy"])
    ev.append(f"best Phase 6 = {best:.4f}   Phase 5 = "
              f"{PHASE5['tree_salet_constrained_noban']:.4f}")
    ev.append(f"classical random = {rnd:.4f}, entropy = {ent:.4f}")
    sp = TERMINAL_SPLIT.get(ADAPTER_NAME)
    if sp:
        ev.append(f"k=1 seen {sp['seen_acc_pct']}% (n={sp['seen_n']}) vs "
                  f"unseen {sp['unseen_acc_pct']}% (n={sp['unseen_n']})  "
                  f"[Phase 5: {PHASE5['k1_seen']}% vs {PHASE5['k1_unseen']}%]")
    t = TERMINAL.get(ADAPTER_NAME, {}).get(1)
    if t:
        ev.append(f"k=1 top-1 {t['top1_accuracy_pct']}% "
                  f"[Phase 5: {PHASE5['k1_top1']}%], median rank "
                  f"{t['median_rank_of_answer']} "
                  f"[Phase 5: {PHASE5['k1_median_rank']}]")

    gen = None
    if sp and sp["unseen_acc_pct"] is not None:
        gen = sp["unseen_acc_pct"] - PHASE5["k1_unseen"]

    if best <= ent + 0.35:
        return "SOLVED", ev + ["-> reaches the classical expert"]
    if best < rnd:
        return "MAJOR", ev + ["-> now beats random elimination; endgame "
                              "coverage was the binding constraint"]
    if gen is not None and gen >= 10:
        return "PARTIAL-GENERALISING", ev + [
            "-> still short of random, but unseen-word accuracy moved "
            f"{gen:+.1f} pts: the procedure is transferring. More/denser "
            "paths per word is the obvious next lever."]
    if gen is not None and gen < 3:
        return "MEMORISED", ev + [
            "-> unseen-word accuracy barely moved. The extra paths bought "
            "memorisation, not a procedure. More of the same data will not "
            "help; the task framing needs to change (rank the candidate set "
            "rather than emit from memory)."]
    return "INCONCLUSIVE", ev + ["-> no clean reading; inspect the tables"]

V, EVIDENCE = verdict()
print("\n" + "=" * 70); print(f"VERDICT: {V}"); print("=" * 70)
for e in EVIDENCE:
    print("  " + e)

# ---- write artifacts -------------------------------------------------------
FIELDS = ["model", "adapter", "ban_repeats", "n_games", "mean_failures_as_7",
          "failure_rate_pct", "repeated_guess_game_rate_pct",
          "invalid_word_rate_pct", "pct_le3", "pct_le4", "pct_le5",
          "median", "max", "avg_chunks_per_decision", "eval_seconds"]
with open(os.path.join(RESULTS_ROOT, "comparison.csv"), "w", newline="",
          encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in EVAL_ROWS.values():
        w.writerow(r)
if TERMINAL_ROWS:
    with open(os.path.join(RESULTS_ROOT, "terminal_probe.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TERMINAL_ROWS[0].keys()))
        w.writeheader()
        for r in TERMINAL_ROWS:
            w.writerow(r)
json.dump({"verdict": V, "evidence": EVIDENCE},
          open(os.path.join(RESULTS_ROOT, "verdict.json"), "w",
               encoding="utf-8"), indent=2)
save_state("final")

base = RESULTS_ZIP[:-4] if RESULTS_ZIP.endswith(".zip") else RESULTS_ZIP
shutil.make_archive(base, "zip", RESULTS_ROOT)
print(f"\nRESULTS ZIP: {base}.zip "
      f"({os.path.getsize(base + '.zip')/2**20:.2f} MiB)  -- no weights inside")
print(f"adapter kept at {WORK_DIR}/{ADAPTER_NAME} "
      f"(snapshot it as a Dataset to reuse next session)")
''')

# =============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}")
print(f"  {len(cells)} cells ({len(cells)-n_code} markdown, {n_code} code)")
print(f"  {os.path.getsize(OUT)/1024:.1f} KiB")
