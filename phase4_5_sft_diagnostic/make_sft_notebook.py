"""
make_sft_notebook.py — generate wordle_qwen_sft_kaggle.ipynb.

    python make_sft_notebook.py
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_qwen_sft_kaggle.ipynb")

cells = []


def md(t: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# =============================================================================
md(r"""
# Wordle SFT — Qwen 0.5B on classical expert trajectories

Fine-tunes **three** LoRA adapters on three different expert policies, then
evaluates all three by **actually playing 246 held-out Wordle games**, not by
token accuracy. No reinforcement learning — this notebook is strictly
`SFT -> evaluation -> analysis`.

| Adapter | Training data | Expert mean (2315 games) | Opener |
|---|---|---:|---|
| `qwen_entropy` | `entropy` | 3.4644 | SOARE |
| `qwen_tree_soare` | `tree_soare` | 3.4544 | SOARE |
| `qwen_tree_salet` | `tree_salet` | 3.4212 (proven optimal) | SALET |

Everything except the dataset is held identical across the three runs.

`tree_soare` exists so the comparison is interpretable: `entropy` and
`tree_salet` differ in **both** opener and policy, and the opener is 77% of the
gap between them. `entropy` vs `tree_soare` isolates the decision policy;
`tree_soare` vs `tree_salet` isolates the opening word.

---

## HOW TO RUN

**STEP 1 — Create the Kaggle Notebook.** New Notebook, Python.

**STEP 2 — Enable the GPU.** Right sidebar -> Session options -> Accelerator ->
**GPU T4 x2** or **GPU P100**. Both work; the code uses one GPU. Confirm
"GPU" shows before continuing — on CPU the training cells will not finish.

**STEP 3 — Add the Wordle SFT Dataset.** Right sidebar -> Input -> Add Input ->
Datasets -> your uploaded dataset. See *Expected dataset layout* below. If you
have not uploaded it yet, run `prepare_kaggle_dataset.py` locally and upload the
resulting `kaggle_upload/` folder.

**STEP 4 — Hugging Face token: NOT required.** `Qwen/Qwen2.5-0.5B-Instruct` is
public. Only if you switch `MODEL_NAME` to a gated model: Add-ons -> Secrets ->
add `HF_TOKEN`, then set `USE_HF_TOKEN = True` in the config cell. **Never paste
a token into a code cell.**

**STEP 5 — Set `DATASET_DIR`** in the CONFIG cell to your dataset's mount path.
Run the cell below it to auto-detect and validate; it prints the exact path and
fails loudly with the available options if wrong.

**STEP 6 — Run the environment and setup cells** (sections 1–5).

**STEP 7 — Run the smoke test** (section 6). ~2 minutes. Do not skip it: it
verifies forward/backward, loss behaviour, checkpoint save, adapter reload and
generation before you spend GPU hours.

**STEP 8 — Set `RUN_ENTROPY = True`** and run section 7.

**STEP 9 — Set `RUN_TREE_SOARE = True`** and run section 7.

**STEP 10 — Set `RUN_TREE_SALET = True`** and run section 7.

> All three can be left `True` for a single session — the estimated total is
> ~2.5–3.5 h including evaluation, inside Kaggle's 9 h GPU session limit. Turn
> them off individually to resume across sessions.

**STEP 11 — Run evaluation and analysis** (sections 8–12).

**STEP 12 — Download `wordle_sft_results.zip`.** The path is printed at the end;
also visible under Output in the right sidebar.

---

## DIAGNOSTIC RUN — no training at all

This is the mode to use now. It reuses the adapters already trained and answers
one question: **is the bottleneck vocabulary generation or Wordle reasoning?**

1. **Add Input** both datasets: the SFT package, and the dataset holding the
   trained adapters (see *Resuming across sessions* below).
2. In the CONFIG cell set:

```python
PREV_RUN_DIR   = "/kaggle/input/wordle-sft-adapters/wordle_sft"
RUN_ENTROPY    = False
RUN_TREE_SOARE = False
RUN_TREE_SALET = False
RUN_SMOKE_TEST = False          # nothing is trained, so nothing to smoke-test
EVAL_MODE      = "both"
RUN_BASE_CONTROL   = True
RUN_TERMINAL_PROBE = True
```

3. Run every cell. Nothing trains; the adapters are loaded from the input
   dataset. Budget **~1 h on a T4**, roughly: unconstrained games ~4 min total,
   constrained games ~15 min, terminal probe ~25 min, baselines ~10 min.
4. Download **`wordle_diagnostic_results.zip`**. The original
   `wordle_sft_results.zip` is not touched and not regenerated.

What comes out: sections 9–12 in both modes, the base-Qwen control row, the
terminal probe (section 12b), the comparison table (13b) and a mechanical
A/B/C/D verdict (13c).

---

## Resuming across sessions

`/kaggle/working` **does not survive a session restart.** To continue later:

1. After a run, open the **Output** tab -> **New Dataset**, and save
   `/kaggle/working/wordle_sft` as a Kaggle Dataset (e.g. `wordle-sft-adapters`).
2. In the next session, **Add Input** that dataset alongside the data dataset.
3. Set `PREV_RUN_DIR = "/kaggle/input/wordle-sft-adapters/wordle_sft"`.
4. Set the `RUN_*` flag to `False` for anything already trained.

The notebook then loads the existing adapter and evaluates it **without
retraining**. `PREV_RUN_DIR = None` disables this.

---

## Expected dataset layout

```
/kaggle/input/<your-slug>/
    sft_package/
        data/entropy/train.jsonl        data/entropy/val.jsonl
        data/tree_soare/train.jsonl     data/tree_soare/val.jsonl
        data/tree_salet/train.jsonl     data/tree_salet/val.jsonl
        data/disagreement/contrast.jsonl
        eval/val_answers.jsonl
        manifest.json
    code/
        wordle_solver.py  tree_search.py  benchmark.py  generate_trajectories.py
    artifacts/
        answers.txt  valid_guesses.txt  feedback_matrix.npy
        metadata.json  frequency_model.json  solver_config.json
```

Only `DATASET_DIR` needs changing. Everything else is derived from it.
""")

# =============================================================================
md(r"""
---
# 1. Environment
""")

code(r'''
import os, sys, json, time, math, random, subprocess, platform, shutil, hashlib
import importlib

def _pip(pkg):
    print(f"installing {pkg} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

# Install ONLY what is missing. Kaggle images ship torch + transformers; peft and
# accelerate are the usual gaps. TRL is deliberately not used: plain SFT with a
# completion-only label mask is a few lines and avoids a heavy dependency whose
# version must match transformers exactly.
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
    """Work around peft raising on an outdated torchao.

    `peft.import_utils.is_torchao_available()` RAISES ImportError when torchao
    is present but older than peft's minimum, instead of returning False. LoRA
    injection calls it through `dispatch_torchao`, so `get_peft_model` dies with

        ImportError: Found an incompatible version of torchao.

    Kaggle images ship torchao 0.10.0 while current peft wants >= 0.16.0. We
    never use torchao (plain fp16 LoRA, no quantization), so the dependency is
    removed if possible, and the probe is neutralised if not. Upgrading torchao
    instead would risk dragging in a different torch build and breaking CUDA.
    """
    try:
        import peft.import_utils as piu
    except Exception as e:
        return f"peft.import_utils unavailable ({e})"
    try:
        piu.is_torchao_available()
        return "no conflict"
    except ImportError:
        pass

    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torchao"], check=False)
    importlib.invalidate_caches()
    try:
        piu.is_torchao_available()
        return "resolved: uninstalled torchao (unused here)"
    except ImportError:
        pass

    # Last resort: neutralise the probe. It must be patched in BOTH modules --
    # peft.tuners.lora.torchao does `from peft.import_utils import
    # is_torchao_available`, so it holds its own reference to the original.
    piu.is_torchao_available = lambda *a, **k: False
    try:
        import peft.tuners.lora.torchao as plt_ao
        plt_ao.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
    return "resolved: patched is_torchao_available -> False"


_TORCHAO_FIX = fix_torchao_peft_conflict()

print("=" * 66)
print("ENVIRONMENT")
print("=" * 66)
print(f"python        {platform.python_version()}")
print(f"platform      {platform.platform()}")
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")
print(f"peft          {peft.__version__}")
print(f"accelerate    {accelerate.__version__}")
print(f"numpy         {np.__version__}")
try:
    import importlib.metadata as _im
    print(f"torchao       {_im.version('torchao')}")
except Exception:
    print("torchao       (not installed)")
print(f"torchao fix   {_TORCHAO_FIX}")
print()
print(f"CUDA available    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"CUDA version      {torch.version.cuda}")
    print(f"GPU               {p.name}")
    print(f"VRAM              {p.total_memory/2**30:.1f} GiB")
    print(f"capability        sm_{p.major}{p.minor}")
    print(f"device count      {torch.cuda.device_count()}")
    # T4 is sm_75: no bf16. This project uses fp16 throughout, as specified.
    print(f"bf16 supported    {torch.cuda.is_bf16_supported()}  "
          f"(we use fp16 regardless)")
else:
    print("\n!! NO GPU DETECTED !!")
    print("Session options -> Accelerator -> GPU T4 x2 or P100, then re-run.")
    print("Training on CPU will not finish in a Kaggle session.")
''')

# =============================================================================
md(r"""
---
# 2. Configuration

**The only line you normally need to change is `DATASET_DIR`.**

Every hyperparameter below is shared by all three runs. The single controlled
variable is the training file.
""")

code(r'''
# ============================ EDIT THIS =====================================
DATASET_DIR = None      # e.g. "/kaggle/input/wordle-sft-package"
                        # Leave None to auto-detect a single /kaggle/input entry.

PREV_RUN_DIR = None     # e.g. "/kaggle/input/wordle-sft-adapters/wordle_sft"
                        # Set when resuming, to evaluate adapters trained in an
                        # earlier session without retraining them.
# ============================================================================

# ---- which experiments to run this session --------------------------------
RUN_ENTROPY     = True
RUN_TREE_SOARE  = True
RUN_TREE_SALET  = True

RUN_SMOKE_TEST  = True
RUN_EVALUATION  = True
RUN_BASELINES   = True

# ---- model ----------------------------------------------------------------
MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
USE_HF_TOKEN = False     # True only for gated models; reads Kaggle Secret HF_TOKEN

# ---- LoRA (identical for all three runs) ----------------------------------
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# ---- optimisation (identical for all three runs) --------------------------
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 2
PER_DEVICE_BS   = 4
GRAD_ACCUM      = 4          # effective batch size 16
MAX_SEQ_LEN     = 640
WARMUP_RATIO    = 0.03
WEIGHT_DECAY    = 0.0
LR_SCHEDULER    = "cosine"
FP16            = True       # NOT bf16 - T4 (sm_75) has no usable bf16
GRAD_CHECKPOINT = True
LOGGING_STEPS   = 25
SAVE_STEPS      = 200        # periodic checkpoints; Kaggle sessions can die
SAVE_TOTAL_LIMIT = 2
SEED            = 20260817

# ---- evaluation -----------------------------------------------------------
MAX_GUESSES       = 6
GEN_MAX_NEW_TOKENS = 8
EVAL_BATCH        = 16        # games advanced in parallel per generate() call

# ---- DIAGNOSTIC RUN (no training) -----------------------------------------
# "unconstrained" reproduces the original run exactly: the model emits free
# text and anything that is not a legal word costs a turn.
# "constrained" scores all 12,972 legal words and plays the argmax, so an
# invalid word is impossible by construction. The model still chooses.
# "both" runs the two back to back and compares them.
EVAL_MODE = "both"            # "unconstrained" | "constrained" | "both"

RUN_BASE_CONTROL   = True     # untrained Qwen, same 246 games, both modes
RUN_TERMINAL_PROBE = True     # can it name the word once the state is solved?
TERMINAL_KS        = [1, 2, 3]   # candidate-set sizes to probe

CONSTRAINED_CHUNK  = 512      # legal words scored per forward pass
CONSTRAINED_PRUNE  = True     # exact branch-and-bound; verified, not heuristic
LENGTH_NORMALISE   = False    # sensitivity check only; NOT the default
# Banning repeats would be a policy change rather than a vocabulary
# constraint, so it is off. The repeat rate is measured and reported instead.
BAN_REPEATS_IN_CONSTRAINED = False

# The surviving-candidate COUNT is NOT in the prompt, in training or at
# evaluation. It is a solver-side quantity a player cannot see, and it leaks how
# far the current constraints already narrow the answer. It stays in record
# metadata for analysis only. There is deliberately no flag to re-enable it:
# doing so would silently desynchronise evaluation from the training data.
SHOW_CANDIDATE_COUNT = False

# ---- paths ----------------------------------------------------------------
WORK_DIR     = "/kaggle/working/wordle_sft"
RESULTS_ZIP  = "/kaggle/working/wordle_sft_results.zip"

# Diagnostic output lives in its own tree and its own zip. The original
# evaluation is never overwritten.
RESULTS_ROOT = "/kaggle/working/results"
DIAG_ZIP     = "/kaggle/working/wordle_diagnostic_results.zip"
RESULT_DIRS  = {
    "unconstrained_sft": os.path.join(RESULTS_ROOT, "unconstrained_sft"),
    "constrained_sft":   os.path.join(RESULTS_ROOT, "constrained_sft"),
    "base_qwen":         os.path.join(RESULTS_ROOT, "base_qwen"),
}
for _d in RESULT_DIRS.values():
    os.makedirs(_d, exist_ok=True)

EXPERIMENTS = {
    "entropy":    {"dataset": "entropy",    "expected_opener": "SOARE",
                   "expert_mean_2315": 3.4644, "run": RUN_ENTROPY},
    "tree_soare": {"dataset": "tree_soare", "expected_opener": "SOARE",
                   "expert_mean_2315": 3.4544, "run": RUN_TREE_SOARE},
    "tree_salet": {"dataset": "tree_salet", "expected_opener": "SALET",
                   "expert_mean_2315": 3.4212, "run": RUN_TREE_SALET},
}

def set_seed_everywhere(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

assert EVAL_MODE in ("unconstrained", "constrained", "both"), EVAL_MODE
EVAL_MODES = (["unconstrained", "constrained"] if EVAL_MODE == "both"
              else [EVAL_MODE])

set_seed_everywhere()
os.makedirs(WORK_DIR, exist_ok=True)
print(f"seed {SEED}, work dir {WORK_DIR}")
print(f"effective batch size = {PER_DEVICE_BS} x {GRAD_ACCUM} = "
      f"{PER_DEVICE_BS * GRAD_ACCUM}")
print(f"eval modes    : {EVAL_MODES}")
print(f"base control  : {RUN_BASE_CONTROL}")
print(f"terminal probe: {RUN_TERMINAL_PROBE} (k = {TERMINAL_KS})")
if not any([RUN_ENTROPY, RUN_TREE_SOARE, RUN_TREE_SALET]):
    print("\nNO TRAINING THIS SESSION - adapters must come from PREV_RUN_DIR")
    print(f"  PREV_RUN_DIR = {PREV_RUN_DIR}")
''')

# =============================================================================
md(r"""
---
# 3. Locate and validate the dataset

Fails loudly and tells you exactly what to fix, rather than half-running.
""")

code(r'''
import glob

REQUIRED_FILES = [
    "sft_package/data/entropy/train.jsonl",
    "sft_package/data/entropy/val.jsonl",
    "sft_package/data/tree_soare/train.jsonl",
    "sft_package/data/tree_soare/val.jsonl",
    "sft_package/data/tree_salet/train.jsonl",
    "sft_package/data/tree_salet/val.jsonl",
    "sft_package/data/disagreement/contrast.jsonl",
    "sft_package/eval/val_answers.jsonl",
    "code/wordle_solver.py",
    "code/tree_search.py",
    "code/generate_trajectories.py",
    "artifacts/answers.txt",
    "artifacts/valid_guesses.txt",
    "artifacts/feedback_matrix.npy",
]

def _has_all(d):
    """True if `d` is the package root (contains every required file)."""
    try:
        return all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES)
    except OSError:
        return False

def _search(root, max_depth=7):
    """Find the package root at ANY depth below `root`.

    Kaggle does not guarantee a fixed mount depth: a dataset can appear as
    /kaggle/input/<slug>/ or nested like
    /kaggle/input/datasets/<user>/<slug>/kaggle_upload/. Globbing a fixed
    number of levels breaks the moment the layout changes, so this tries cheap
    shallow globs first and then falls back to a bounded walk.
    """
    if not os.path.isdir(root):
        return None
    for depth in range(0, 6):                      # fast path: depth 0..5
        pat = os.path.join(root, *(["*"] * depth)) if depth else root
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and _has_all(d):
                return d
    base = os.path.abspath(root).rstrip(os.sep).count(os.sep)
    best = None
    for dirpath, dirnames, _ in os.walk(root):     # bounded fallback
        if os.path.abspath(dirpath).rstrip(os.sep).count(os.sep) - base > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _has_all(dirpath):
            if best is None or len(dirpath) < len(best):
                best = dirpath
            dirnames[:] = []                       # do not descend past a hit
    return best

def locate_dataset(explicit=None):
    # An explicit path is honoured, but we still search beneath it: pointing at
    # the dataset root when the files sit in a subfolder is the common mistake,
    # and it should just work rather than fail.
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
    if explicit:
        msg.append(f"DATASET_DIR was set to: {explicit}")
        msg.append("Missing there:")
        for f in REQUIRED_FILES:
            if not os.path.exists(os.path.join(explicit, f)):
                msg.append(f"    {f}")
        msg.append("")
    msg.append("Searched /kaggle/input to depth 7. Found these directories:")
    seen = 0
    for dirpath, dirnames, _ in os.walk("/kaggle/input"):
        d = os.path.abspath(dirpath).count(os.sep)
        if d - os.path.abspath("/kaggle/input").count(os.sep) > 4:
            dirnames[:] = []
            continue
        msg.append(f"    {dirpath}")
        seen += 1
        if seen > 40:
            msg.append("    ...")
            break
    if not seen:
        msg.append("    (nothing - did you Add Input?)")
    msg += ["", "Fix: sidebar -> Input -> Add Input -> your dataset.",
            "Leave DATASET_DIR = None to auto-detect, or set it to any parent",
            "of the folder containing sft_package/ code/ artifacts/."]
    raise FileNotFoundError("\n".join(msg))

DATASET_DIR = locate_dataset(DATASET_DIR)
SFT_DIR   = os.path.join(DATASET_DIR, "sft_package")
CODE_DIR  = os.path.join(DATASET_DIR, "code")
ARTIFACTS = os.path.join(DATASET_DIR, "artifacts")

def ensure_code_on_path():
    """Put CODE_DIR on sys.path. Idempotent, and safe to call repeatedly.

    Called here AND again in the import cell, so importing still works if cells
    are run out of order or the kernel is restarted mid-notebook.
    """
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    return CODE_DIR

ensure_code_on_path()
print(f"DATASET_DIR = {DATASET_DIR}")
print(f"CODE_DIR    = {CODE_DIR}")
print(f"SFT_DIR     = {SFT_DIR}")
print(f"ARTIFACTS   = {ARTIFACTS}")
print(f"sys.path[0] = {sys.path[0]}\n")
assert os.path.exists(os.path.join(CODE_DIR, "wordle_solver.py")), \
    f"wordle_solver.py not found in {CODE_DIR}"

def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

DATASET_HASHES = {}
print(f"{'file':<48}{'lines':>8}{'MiB':>8}  sha256[:12]")
print("-" * 82)
for f in REQUIRED_FILES:
    p = os.path.join(DATASET_DIR, f)
    n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) \
        if f.endswith((".jsonl", ".txt", ".py")) else 0
    hh = sha256(p)
    DATASET_HASHES[f] = {"sha256": hh, "bytes": os.path.getsize(p), "lines": n}
    print(f"{f:<48}{n:>8}{os.path.getsize(p)/2**20:>8.1f}  {hh[:12]}")

ensure_code_on_path()
print("\nall required files present.")
''')

code(r'''
# Re-assert the import path here as well: this cell must work even if it is run
# on its own, or after a kernel restart, without re-running the locate cell.
ensure_code_on_path()
print(f"importing project modules from {CODE_DIR}")

# The exact Wordle environment used to build the data is reused for evaluation,
# so the prompt the model sees at eval is byte-identical in format to training.
from wordle_solver import (
    ALL_GREEN, feedback, feedback_code, code_to_pattern, pattern_to_code,
    load_artifacts, Vocabulary, SolverConfig, make_solver, play_game,
)
from generate_trajectories import derive_constraints, render_prompt, RULES

BUNDLE = load_artifacts(ARTIFACTS, mmap=True)
VOCAB  = BUNDLE.vocab
LEGAL_GUESSES = set(w.upper() for w in VOCAB.guesses)
ANSWER_SET    = set(w.upper() for w in VOCAB.answers)
print(BUNDLE.describe())

VAL_ANSWERS = [json.loads(l)["answer"].upper()
               for l in open(os.path.join(SFT_DIR, "eval/val_answers.jsonl"),
                             encoding="utf-8")]
print(f"\nheld-out evaluation answers: {len(VAL_ANSWERS)}")
print(f"  first 8: {VAL_ANSWERS[:8]}")

# Sanity: the feedback function must be the audited one.
assert feedback("salet", "tribe") == "BBBYY"
assert feedback("added", "dread") == "YYBYG"     # duplicate-letter case
print("\nfeedback function verified on known duplicate-letter cases.")
''')

# =============================================================================
md(r"""
---
# 4. Model and tokenizer

`Qwen/Qwen2.5-0.5B-Instruct` is public — no token needed. `USE_HF_TOKEN` reads
the Kaggle Secret `HF_TOKEN` only if you switch to a gated model.
""")

code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = None
if USE_HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("loaded HF_TOKEN from Kaggle Secrets")
    except Exception as e:
        raise RuntimeError(
            "USE_HF_TOKEN=True but the secret could not be read.\n"
            "Add-ons -> Secrets -> add a secret named exactly HF_TOKEN, "
            "attach it to this notebook, then re-run.\n"
            f"underlying error: {e}")

tok_kwargs = {"trust_remote_code": True}
if HF_TOKEN:
    tok_kwargs["token"] = HF_TOKEN

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, **tok_kwargs)
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "right"     # right for training; flipped for generation

print(f"model      {MODEL_NAME}")
print(f"vocab size {len(TOKENIZER)}")
print(f"eos        {TOKENIZER.eos_token!r} (id {TOKENIZER.eos_token_id})")
print(f"pad        {TOKENIZER.pad_token!r} (id {TOKENIZER.pad_token_id})")

def load_base_model():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,      # fp16 as specified
        device_map=None,
        trust_remote_code=True,
        **({"token": HF_TOKEN} if HF_TOKEN else {}),
    )
    m.config.use_cache = False
    return m

_probe = load_base_model()
N_PARAMS = sum(p.numel() for p in _probe.parameters())
print(f"\nparameters {N_PARAMS/1e6:.1f} M")
print(f"fp16 weights ~{N_PARAMS*2/2**30:.2f} GiB")
del _probe
torch.cuda.empty_cache()
''')

# =============================================================================
md(r"""
---
# 5. Dataset construction

**Completion-only loss.** The prompt tokens are masked to `-100` so the model is
trained to produce the guess, not to reproduce the prompt. Without this the
5-token target is drowned out by ~450 tokens of state description.

**Left truncation of the prompt.** If a sequence exceeds `MAX_SEQ_LEN` the
*oldest* prompt tokens are dropped. Truncating from the right would remove the
`Next guess:` instruction and the most recent feedback — the parts that matter
most.
""")

code(r'''
from torch.utils.data import Dataset

def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

class WordleSFTDataset(Dataset):
    """prompt -> completion, with the prompt masked out of the loss."""

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
        # Tokenize the two halves separately, so the boundary is exact and does
        # not depend on BPE merging across it.
        p_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = self.tok(" " + r["completion"],
                         add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [self.tok.eos_token_id]

        keep = self.max_len - len(c_ids)
        if len(p_ids) > keep:
            p_ids = p_ids[-keep:]           # drop OLDEST prompt tokens
            self.n_truncated += 1

        input_ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        item = {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}
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

# Length audit — confirms MAX_SEQ_LEN is adequate before any GPU time is spent.
_probe_rows = load_jsonl(os.path.join(SFT_DIR, "data/entropy/train.jsonl"))[:800]
_lens = []
for r in _probe_rows:
    _lens.append(len(TOKENIZER(r["prompt"], add_special_tokens=False)["input_ids"])
                 + len(TOKENIZER(" " + r["completion"],
                                 add_special_tokens=False)["input_ids"]) + 1)
_lens = np.array(_lens)
print(f"token lengths over {len(_lens)} sampled training rows:")
print(f"  min {_lens.min()}  mean {_lens.mean():.0f}  p95 {np.percentile(_lens,95):.0f}"
      f"  max {_lens.max()}")
print(f"  MAX_SEQ_LEN = {MAX_SEQ_LEN} -> "
      f"{100*(_lens > MAX_SEQ_LEN).mean():.2f}% would be truncated")
if (_lens > MAX_SEQ_LEN).mean() > 0.02:
    print("  NOTE: >2% truncation. Consider raising MAX_SEQ_LEN to "
          f"{int(np.percentile(_lens, 99)) + 16}.")
''')

# =============================================================================
md(r"""
---
# 6. Smoke test

Verifies forward pass, backward pass, loss behaviour, checkpoint creation,
adapter reload and generation on ~40 examples. **Run this before the full
training.** Roughly 2 minutes.
""")

code(r'''
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments

def make_lora_model():
    # Re-apply defensively: this cell must work even if the environment cell
    # was skipped or the kernel was restarted.
    fix_torchao_peft_conflict()

    base = load_base_model()
    cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    )
    m = get_peft_model(base, cfg)

    # The base is fp16 to save memory, but the TRAINABLE LoRA params must be
    # fp32. torch.cuda.amp keeps fp32 master weights and refuses to unscale
    # fp16 gradients -- with fp16 adapters, training dies on the first
    # optimizer step with "Attempting to unscale FP16 gradients."
    # Recent peft casts adapters via autocast_adapter_dtype=True, but that is
    # version-dependent, so it is enforced here explicitly.
    n_cast = 0
    for _, p in m.named_parameters():
        if p.requires_grad and p.dtype == torch.float16:
            p.data = p.data.float()
            n_cast += 1
    if n_cast:
        print(f"  cast {n_cast} trainable tensors fp16 -> fp32 (amp requirement)")
    dtypes = {str(p.dtype) for p in m.parameters() if p.requires_grad}
    assert dtypes <= {"torch.float32"}, f"trainable params must be fp32, got {dtypes}"
    return m

def training_args(out_dir, n_examples, **over):
    steps_per_epoch = max(1, n_examples // (PER_DEVICE_BS * GRAD_ACCUM))
    kw = dict(
        output_dir=out_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16, bf16=False,
        gradient_checkpointing=GRAD_CHECKPOINT,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_strategy="steps",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        optim="adamw_torch",
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
disable_tqdm=False,
    )
    kw.update(over)
    return TrainingArguments(**kw)

if RUN_SMOKE_TEST:
    print("=" * 66); print("SMOKE TEST"); print("=" * 66)
    set_seed_everywhere()
    rows = load_jsonl(os.path.join(SFT_DIR, "data/entropy/train.jsonl"))[:40]
    ds = WordleSFTDataset(rows, TOKENIZER)
    ex = ds[0]
    n_lab = sum(1 for x in ex["labels"] if x != -100)
    print(f"example 0: {len(ex['input_ids'])} tokens, {n_lab} supervised "
          f"(the completion + EOS)")
    print(f"  decoded target: "
          f"{TOKENIZER.decode([x for x in ex['labels'] if x != -100])!r}")
    assert n_lab < 12, "completion mask looks wrong - too many supervised tokens"

    smoke_dir = os.path.join(WORK_DIR, "_smoke")
    model = make_lora_model()
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=training_args(smoke_dir, len(rows), num_train_epochs=3,
                           logging_steps=1, save_steps=6,
                           save_total_limit=1),
        train_dataset=ds,
        data_collator=lambda b: collate(b, TOKENIZER.pad_token_id),
    )
    t0 = time.perf_counter()
    out = trainer.train()
    print(f"\nsmoke training finished in {time.perf_counter()-t0:.0f}s")

    hist = [h for h in trainer.state.log_history if "loss" in h]
    losses = [h["loss"] for h in hist]
    print(f"loss: first {losses[0]:.4f} -> last {losses[-1]:.4f} "
          f"over {len(losses)} logged steps")
    assert all(math.isfinite(l) for l in losses), "non-finite loss (fp16 blow-up)"
    if losses[-1] >= losses[0]:
        print("  WARNING: loss did not decrease. Check LR / data before scaling up.")
    else:
        print("  loss decreased: forward+backward path is working.")

    trainer.save_model(smoke_dir)
    assert os.path.exists(os.path.join(smoke_dir, "adapter_model.safetensors")) \
        or os.path.exists(os.path.join(smoke_dir, "adapter_model.bin")), \
        "adapter file not written"
    print(f"  checkpoint written to {smoke_dir}")

    del model, trainer; torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base_model(), smoke_dir)
    reloaded.eval().cuda()
    print("  adapter reloaded OK")

    p = rows[0]["prompt"]
    enc = TOKENIZER(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = reloaded.generate(**enc, max_new_tokens=GEN_MAX_NEW_TOKENS,
                              do_sample=False,
                              pad_token_id=TOKENIZER.pad_token_id)
    txt = TOKENIZER.decode(g[0][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True)
    print(f"  generation OK -> {txt!r}  (expert said {rows[0]['completion']})")
    del reloaded; torch.cuda.empty_cache()
    shutil.rmtree(smoke_dir, ignore_errors=True)
    print("\nSMOKE TEST PASSED - safe to run the full experiments.")
else:
    print("smoke test skipped (RUN_SMOKE_TEST = False)")
''')

# =============================================================================
md(r"""
---
# 7. Train the three adapters

Identical in every respect except the training file. Each run writes to its own
directory and records its own config, so a session that dies mid-way loses only
the run in flight.
""")

code(r'''
def train_one(name, spec):
    out_dir = os.path.join(WORK_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(SFT_DIR, f"data/{spec['dataset']}/train.jsonl")
    rows = load_jsonl(path)
    print("=" * 66)
    print(f"TRAINING {name}  ({len(rows)} examples from {spec['dataset']})")
    print("=" * 66)

    set_seed_everywhere()                     # identical init for every run
    ds = WordleSFTDataset(rows, TOKENIZER)
    model = make_lora_model()
    model.print_trainable_parameters()

    args = training_args(out_dir, len(rows))
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, TOKENIZER.pad_token_id))
    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0

    trainer.save_model(out_dir)
    TOKENIZER.save_pretrained(out_dir)

    hist = [h for h in trainer.state.log_history if "loss" in h]
    cfg = {
        "name": name,
        "dataset": spec["dataset"],
        "dataset_path": path,
        "n_train_examples": len(rows),
        "n_truncated": ds.n_truncated,
        "model_name": MODEL_NAME,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                 "target_modules": LORA_TARGETS},
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "per_device_batch_size": PER_DEVICE_BS,
        "gradient_accumulation": GRAD_ACCUM,
        "effective_batch_size": PER_DEVICE_BS * GRAD_ACCUM,
        "max_seq_len": MAX_SEQ_LEN,
        "lr_scheduler": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "fp16": FP16, "bf16": False,
        "gradient_checkpointing": GRAD_CHECKPOINT,
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "training_seconds": round(wall, 1),
        "global_steps": trainer.state.global_step,
        "final_loss": hist[-1]["loss"] if hist else None,
        "first_loss": hist[0]["loss"] if hist else None,
        "loss_curve": [{"step": h["step"], "loss": h["loss"]} for h in hist],
        "expected_opener": spec["expected_opener"],
        "expert_mean_2315": spec["expert_mean_2315"],
    }
    with open(os.path.join(out_dir, "training_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    print(f"\n{name}: {wall/60:.1f} min, {trainer.state.global_step} steps, "
          f"loss {cfg['first_loss']} -> {cfg['final_loss']}")
    print(f"  saved to {out_dir}")
    if ds.n_truncated:
        print(f"  NOTE {ds.n_truncated} examples were left-truncated")

    del model, trainer; torch.cuda.empty_cache()
    return cfg

TRAIN_CONFIGS = {}
for name, spec in EXPERIMENTS.items():
    out_dir = os.path.join(WORK_DIR, name)
    have = os.path.exists(os.path.join(out_dir, "adapter_config.json"))
    if spec["run"]:
        TRAIN_CONFIGS[name] = train_one(name, spec)
    elif have:
        print(f"{name}: RUN flag off, adapter already in {out_dir} - will evaluate")
        TRAIN_CONFIGS[name] = json.load(
            open(os.path.join(out_dir, "training_config.json")))
    elif PREV_RUN_DIR and os.path.exists(
            os.path.join(PREV_RUN_DIR, name, "adapter_config.json")):
        src = os.path.join(PREV_RUN_DIR, name)
        print(f"{name}: RUN flag off, loading previous adapter from {src}")
        shutil.copytree(src, out_dir, dirs_exist_ok=True)
        cp = os.path.join(out_dir, "training_config.json")
        TRAIN_CONFIGS[name] = json.load(open(cp)) if os.path.exists(cp) else {"name": name}
    else:
        print(f"{name}: SKIPPED (RUN flag off and no existing adapter found)")

print(f"\nadapters available: {sorted(TRAIN_CONFIGS)}")
''')

# =============================================================================
md(r"""
---
# 8. Evaluation by actual gameplay

The model plays 246 held-out games. Each turn it sees only the visible state —
history, deduced constraints, turn number — in exactly the format it was trained
on. **The answer is never in the prompt** (it is not even a parameter of
`render_prompt`), and **neither the candidate list nor the candidate count is
ever shown.** The count is tracked per turn for analysis and reported in the
metrics, but it is not part of the model's input.

## Output handling, defined before any results are seen

1. Take the generated text up to the first newline.
2. Strip surrounding punctuation from the **first** whitespace token.
3. If it is exactly 5 ASCII letters -> that is the guess.
4. Otherwise -> `invalid_format`. **No scanning of later tokens by default.**
5. If the word is not in the 12,972-word legal list -> `invalid_word`.

`LENIENT_OUTPUT_PARSING` (default **False**) would also scan later tokens for a
5-letter word. It is off because it silently corrupts results: on
`"I guess SLATE"` it extracts **GUESS**, not SLATE — "guess" is itself a valid
5-letter Wordle word. That converts a formatting failure into a plausible-looking
wrong guess, which is precisely the kind of flattering repair to avoid. The
notebook still reports `would_recover_pct` so you can see how much output the
lenient rule *would* have salvaged, without that rule affecting any score.

An `invalid_format` or `invalid_word` turn **is consumed and yields no
feedback** — the state is unchanged and the model is re-prompted. Because
generation is greedy this will usually repeat, so such a model burns its turns
and fails the game. That is the honest outcome; nothing is silently repaired.

Generation is greedy (`do_sample=False`), so results are deterministic.
""")

code(r'''
import re
from peft import PeftModel

WORD_RE = re.compile(r"^[A-Za-z]{5}$")

LENIENT_OUTPUT_PARSING = False   # see the markdown above before enabling

def extract_guess(text):
    """Returns (word_or_None, status, would_recover). Policy fixed above.

    Strict by default: only the FIRST token counts. Scanning later tokens turns
    "I guess SLATE" into GUESS -- a valid Wordle word, so the error would be
    scored as an ordinary wrong guess instead of a formatting failure.
    `would_recover` reports what the lenient rule would have found, purely for
    diagnostics; it never affects a score unless LENIENT_OUTPUT_PARSING is on.
    """
    line = text.strip().split("\n")[0]
    toks = [t.strip(".,:;!?\"'()[]*_-") for t in line.split()]
    toks = [t for t in toks if t]
    if not toks:
        return None, "invalid_format", None
    if WORD_RE.match(toks[0]):
        return toks[0].upper(), "ok", None
    later = next((t.upper() for t in toks[1:] if WORD_RE.match(t)), None)
    if LENIENT_OUTPUT_PARSING and later:
        return later, "recovered_by_extraction", later
    return None, "invalid_format", later

class GameState:
    __slots__ = ("answer", "history", "cands", "guesses", "patterns",
                 "remaining", "statuses", "would_recover", "raw",
                 "done", "solved", "margins", "mode")
    def __init__(self, answer, n_answers, mode="unconstrained"):
        self.answer = answer
        self.mode = mode
        self.history = []
        self.cands = np.arange(n_answers, dtype=np.int32)
        self.guesses, self.patterns, self.remaining, self.statuses = [], [], [], []
        self.would_recover, self.raw = [], []
        self.margins = []
        self.done = self.solved = False

    def prompt(self, turn, max_guesses):
        c = derive_constraints([(g.lower(), p) for g, p in self.history])
        return render_prompt(
            turn=turn,
            history=[(g.lower(), p) for g, p in self.history],
            constraints=c,
            n_candidates=len(self.cands),    # tracked for analysis only
            guesses_remaining=max_guesses - turn + 1,
            max_guesses=max_guesses,
            candidates=None,                 # NEVER shown
            show_candidate_count=False,      # NEVER shown
        )

PLAY_STATS = {}      # filled by play_games; constrained-mode pruning stats

def _apply(g, word, status, would, raw, turn, max_guesses, margin=None):
    """Advance one game by one turn. Identical bookkeeping in both modes."""
    g.statuses.append(status)
    g.would_recover.append(would)
    g.raw.append(raw)
    g.margins.append(margin)
    if status in ("invalid_format", "invalid_word"):
        # Turn consumed, no information gained, state unchanged.
        g.guesses.append(word or "<INVALID>")
        g.patterns.append(None)
        g.remaining.append(int(len(g.cands)))
    else:
        code_ = feedback_code(word.lower(), g.answer.lower())
        pat = code_to_pattern(code_)
        g.cands = BUNDLE.fb.filter_indices(g.cands, word.lower(), code_)
        g.history.append((word, pat))
        g.guesses.append(word)
        g.patterns.append(pat)
        g.remaining.append(int(len(g.cands)))
        if code_ == ALL_GREEN:
            g.solved = g.done = True
    if turn == max_guesses and not g.solved:
        g.done = True


@torch.no_grad()
def play_games(model, answers, mode="unconstrained", scorer=None,
               max_guesses=MAX_GUESSES, batch=EVAL_BATCH, log_every=50):
    """Play `answers` to completion.

    mode="unconstrained": the original protocol. The model generates free text;
      a non-word costs the turn and returns no feedback.
    mode="constrained": every legal word is scored and the argmax is played.
      `invalid_word` and `invalid_format` are impossible by construction. The
      model still chooses which of the 12,972 legal words to play, and the
      candidate list and answer remain hidden.
    """
    assert mode in ("unconstrained", "constrained")
    if mode == "constrained":
        assert scorer is not None, "constrained mode needs a LegalWordScorer"
    model.eval()
    tok = TOKENIZER
    old_side = tok.padding_side
    tok.padding_side = "left"                # required for batched generation
    games = [GameState(a, VOCAB.n_answers, mode=mode) for a in answers]
    t0 = time.perf_counter()
    n_chunks_used = []

    for turn in range(1, max_guesses + 1):
        active = [g for g in games if not g.done]
        if not active:
            break

        if mode == "constrained":
            for g in active:
                p = g.prompt(turn, max_guesses)
                banned = (list(dict.fromkeys(g.guesses))
                          if BAN_REPEATS_IN_CONSTRAINED else None)
                word, sc, nch, margin = scorer.argmax(
                    model, p, banned=banned, prune=CONSTRAINED_PRUNE)
                n_chunks_used.append(nch)
                # The scorer only ever returns members of LEGAL_GUESSES.
                _apply(g, word, "ok", None, f"<constrained:{sc:.3f}>",
                       turn, max_guesses, margin=margin)
        else:
            for s in range(0, len(active), batch):
                chunk = active[s:s + batch]
                prompts = [g.prompt(turn, max_guesses) for g in chunk]
                enc = tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
                out = model.generate(**enc, max_new_tokens=GEN_MAX_NEW_TOKENS,
                                     do_sample=False, temperature=None,
                                     top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                texts = tok.batch_decode(gen, skip_special_tokens=True)
                for g, txt in zip(chunk, texts):
                    word, status, would = extract_guess(txt)
                    if word is not None and word not in LEGAL_GUESSES:
                        status = "invalid_word"
                    _apply(g, word, status, would, txt, turn, max_guesses)

        if log_every:
            n_done = sum(1 for g in games if g.done)
            extra = ""
            if n_chunks_used:
                extra = f"  [avg {np.mean(n_chunks_used):.1f} chunks/decision]"
            print(f"  turn {turn}: {n_done}/{len(games)} finished "
                  f"({time.perf_counter()-t0:.0f}s){extra}", flush=True)

    tok.padding_side = old_side
    PLAY_STATS.clear()
    if n_chunks_used:
        PLAY_STATS["avg_chunks_per_decision"] = round(float(np.mean(n_chunks_used)), 2)
        PLAY_STATS["max_chunks_per_decision"] = int(max(n_chunks_used))
        PLAY_STATS["unpruned_chunks"] = -(-len(LEGAL_GUESSES) // CONSTRAINED_CHUNK)
    return games

def load_adapter(name):
    """Load a trained adapter. Falls back to PREV_RUN_DIR (Kaggle input)."""
    path = os.path.join(WORK_DIR, name)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        alt = os.path.join(PREV_RUN_DIR or "", name)
        assert os.path.exists(os.path.join(alt, "adapter_config.json")), (
            f"no adapter for {name} in {path} or {alt}. Set PREV_RUN_DIR to the "
            f"Kaggle input dataset holding the trained adapters.")
        path = alt
    base = load_base_model()
    m = PeftModel.from_pretrained(base, path)
    m.config.use_cache = True
    return m.eval().cuda()

def load_base_control():
    """Untrained Qwen — the control that says what SFT actually contributed."""
    m = load_base_model()
    m.config.use_cache = True
    return m.eval().cuda()

print("evaluation harness ready.")
print(f"  candidate COUNT shown to model : {SHOW_CANDIDATE_COUNT} (never)")
print(f"  candidate LIST shown to model  : False (never)")
print(f"  answer shown to model          : False (never)")

# Eval prompts must be byte-identical in format to the training prompts.
_probe = load_jsonl(os.path.join(SFT_DIR, "data/entropy/train.jsonl"))[:200]
_bad = sum(1 for r in _probe if "Possible answers remaining" in r["prompt"])
_g = GameState("CRANE", VOCAB.n_answers)
assert "Possible answers remaining" not in _g.prompt(1, MAX_GUESSES)
print(f"\n  training prompts containing the count: {_bad}/200")
print(f"  evaluation prompt contains the count : False")
assert _bad == 0, "training data still contains the candidate count"
''')

# =============================================================================
md(r"""
---
# 8b. Constrained decoding over the legal word list

The unconstrained run showed ~30% of generated guesses were not legal Wordle
words, and that an invalid guess — which costs a turn and returns no feedback —
repeats under greedy decoding until the game is lost. That confounds two very
different abilities:

* **vocabulary generation** — can it spell a 5-letter English word it was never
  trained to emit?
* **Wordle reasoning** — does it know *which* word to play?

Constrained decoding removes the first so the second can be measured.

**This is not post-hoc filtering.** The model never emits free text in this
mode. Every one of the 12,972 legal guesses is scored under the model, and the
argmax is played. The question put to the model is "which of these words should
I play?", and its answer is read off a ranking it produced.

What the model still does **not** get: the candidate-answer list, the candidate
count, the answer. The constraint is purely lexical — it is the Wordle
keyboard, which a human player also has.

Full mechanics, the exact scoring definition, and the correctness argument for
the pruning are in the module docstring in the next cell.
""")

with open(os.path.join(_paths.CORE, "constrained_decode.py"), encoding="utf-8") as _fh:
    CONSTRAINED_SRC = _fh.read().rstrip()

code(CONSTRAINED_SRC + r'''


# ---------------------------------------------------------------------------
# Wiring into this notebook
# ---------------------------------------------------------------------------
LEGAL_WORDS_SORTED = sorted(LEGAL_GUESSES)      # all 12,972, deterministic order
assert len(LEGAL_WORDS_SORTED) == 12972, len(LEGAL_WORDS_SORTED)

SCORER = None
_SCORER_VERIFIED = False

def build_scorer(device="cuda"):
    global SCORER
    if SCORER is None:
        t0 = time.perf_counter()
        SCORER = LegalWordScorer(TOKENIZER, LEGAL_WORDS_SORTED, device=device,
                                 chunk=CONSTRAINED_CHUNK,
                                 length_normalise=LENGTH_NORMALISE)
        print(f"scorer built over {SCORER.n} legal words "
              f"(max {SCORER.max_len} tokens incl. EOS) "
              f"in {time.perf_counter()-t0:.1f}s")
    return SCORER

def verify_scorer(model):
    """Runs once per session. Failing here is fatal and it should be."""
    global _SCORER_VERIFIED
    sc = build_scorer()
    probe = GameState("CRANE", VOCAB.n_answers).prompt(1, MAX_GUESSES)
    dev = sc.self_test(model, probe)
    print(f"  cache-reuse vs naive scoring [{dev['dtype']}]: "
          f"max |delta| = {dev['max_abs_dev']:.4f} nats "
          f"(tol {dev['atol']}), over a {dev['spread']:.1f}-nat spread, "
          f"ranking corr = {dev['corr']:.6f}  OK")
    if CONSTRAINED_PRUNE:
        w, nch = sc.verify_against_full(model, probe)
        print(f"  pruned argmax == full argmax ({w}), {nch} chunks vs "
              f"{-(-sc.n // CONSTRAINED_CHUNK)} unpruned  OK")
    if BAN_REPEATS_IN_CONSTRAINED:
        # Sequential banning must walk the global ranking exactly.
        full = sc.score_all(model, probe)
        want = [sc.words[i] for i in
                torch.argsort(full, descending=True)[:4].tolist()]
        got, ban = [], []
        for _ in range(4):
            w2, _, _, _ = sc.argmax(model, probe, banned=ban or None)
            assert w2 not in ban, f"banned word {w2} was returned"
            got.append(w2); ban.append(w2)
        assert got == want, f"banning broke the ranking: {got} != {want}"
        print(f"  sequential banning walks the global ranking {got}  OK")
    _SCORER_VERIFIED = True
    return dev

print("constrained decoder defined.")
print(f"  legal words        : {len(LEGAL_WORDS_SORTED)}")
print(f"  chunk              : {CONSTRAINED_CHUNK}")
print(f"  exact pruning      : {CONSTRAINED_PRUNE}")
print(f"  length normalise   : {LENGTH_NORMALISE} (heuristic; off by default)")
print(f"  repeats banned     : {BAN_REPEATS_IN_CONSTRAINED} (policy change; off)")
print("  candidate list / count / answer shown: False (unchanged)")
''')

# =============================================================================
md(r"""
---
# 9. Metrics
""")

code(r'''
def score_games(games, label, max_guesses=MAX_GUESSES):
    n = len(games)
    scores = [len(g.guesses) if g.solved else max_guesses + 1 for g in games]
    solved = [len(g.guesses) for g in games if g.solved]
    dist = {k: sum(1 for g in games if g.solved and len(g.guesses) == k)
            for k in range(1, max_guesses + 1)}
    cum = lambda m: 100 * sum(dist[i] for i in range(1, m + 1)) / n

    all_status = [s for g in games for s in g.statuses]
    n_turns = len(all_status)
    def rate(x):
        return 100 * sum(1 for s in all_status if s == x) / max(n_turns, 1)

    # A repeated guess is strictly wasteful. A guess that is not a surviving
    # candidate is NOT an error -- that is exactly what an information-seeking
    # solver does -- so it is reported separately and neutrally.
    rep = sum(1 for g in games
              if len([x for x in g.guesses]) != len(set(g.guesses)))
    uniq = float(np.mean([len(set(g.guesses)) for g in games]))
    hardmode_viol = 0
    total_valid = 0
    for g in games:
        seen = []
        for gu, pat in zip(g.guesses, g.patterns):
            if pat is None:
                continue
            total_valid += 1
            greens = {}
            for pg, pp in seen:
                for i, (ch, t) in enumerate(zip(pg, pp)):
                    if t == "G":
                        greens[i] = ch
            if any(gu[i] != ch for i, ch in greens.items()):
                hardmode_viol += 1
            seen.append((gu, pat))

    margins = [m for g in games for m in g.margins if m is not None]

    return {
        "model": label,
        "mode": games[0].mode if games else None,
        "n_games": n,
        "mean_failures_as_7": round(sum(scores) / n, 4),
        "mean_solved_only": round(sum(solved) / len(solved), 4) if solved else None,
        "median": float(np.median(solved)) if solved else None,
        "std": round(float(np.std(solved, ddof=1)), 4) if len(solved) > 1 else 0.0,
        "max": max(solved) if solved else None,
        "pct_le3": round(cum(3), 2), "pct_le4": round(cum(4), 2),
        "pct_le5": round(cum(5), 2), "pct_le6": round(cum(6), 2),
        "distribution": dist,
        "failures": n - len(solved),
        "failure_rate_pct": round(100 * (n - len(solved)) / n, 2),
        "turns_generated": n_turns,
        "invalid_format_rate_pct": round(rate("invalid_format"), 2),
        "invalid_word_rate_pct": round(rate("invalid_word"), 2),
        "recovered_by_extraction_pct": round(rate("recovered_by_extraction"), 2),
        # Diagnostic only: how much malformed output a lenient parser WOULD
        # have salvaged. Does not affect any score above.
        "would_recover_pct": round(
            100 * sum(1 for g in games for w in g.would_recover if w)
            / max(n_turns, 1), 2),
        "lenient_parsing_enabled": LENIENT_OUTPUT_PARSING,
        "repeated_guess_game_rate_pct": round(100 * rep / n, 2),
        "avg_unique_guesses_per_game": round(uniq, 3),
        "hard_mode_violation_pct": round(
            100 * hardmode_viol / max(total_valid, 1), 2),
        "avg_candidates_after_turn": {
            str(t): round(float(np.mean([g.remaining[t-1] for g in games
                                         if len(g.remaining) >= t])), 2)
            for t in range(1, max_guesses + 1)
            if any(len(g.remaining) >= t for g in games)
        },
        # Constrained mode only: log-prob gap between the played word and the
        # runner-up. Small margins mean the ranking is nearly a coin flip.
        "avg_decision_margin": (round(float(np.mean(margins)), 4)
                                if margins else None),
    }

def print_scores(rows, title):
    print(title)
    hdr = (f"{'model':<20}{'mean':>8}{'med':>5}{'max':>5}{'<=3':>8}{'<=4':>8}"
           f"{'<=5':>8}{'fail%':>7}{'bad%':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        bad = r.get("invalid_format_rate_pct", 0) + r.get("invalid_word_rate_pct", 0)
        print(f"{r['model']:<20}{r['mean_failures_as_7']:>8.4f}"
              f"{(r['median'] or 0):>5.0f}{(r['max'] or 0):>5}"
              f"{r['pct_le3']:>8.2f}{r['pct_le4']:>8.2f}{r['pct_le5']:>8.2f}"
              f"{r['failure_rate_pct']:>7.2f}{bad:>7.2f}")
''')

code(r'''
GAMES_ALL = {m: {} for m in EVAL_MODES}      # mode -> name -> [GameState]
EVAL_ALL  = {m: {} for m in EVAL_MODES}      # mode -> name -> metrics
TERMINAL  = {}                               # filled in by section 12b

def _report(row, t0):
    row["eval_seconds"] = round(time.perf_counter() - t0, 1)
    bad = row["invalid_format_rate_pct"] + row["invalid_word_rate_pct"]
    print(f"  mean={row['mean_failures_as_7']:.4f}  "
          f"fail={row['failure_rate_pct']:.1f}%  "
          f"solved={100-row['failure_rate_pct']:.1f}%  "
          f"invalid={bad:.1f}%  "
          f"repeat={row['repeated_guess_game_rate_pct']:.1f}%  "
          f"({row['eval_seconds']:.0f}s)\n")

def evaluate(model, label, name, mode):
    t0 = time.perf_counter()
    sc = build_scorer() if mode == "constrained" else None
    gs = play_games(model, VAL_ANSWERS, mode=mode, scorer=sc)
    GAMES_ALL[mode][name] = gs
    row = score_games(gs, label)
    row.update(PLAY_STATS)
    EVAL_ALL[mode][name] = row
    _report(row, t0)
    return row

if RUN_EVALUATION:
    for mode in EVAL_MODES:
        for name in EXPERIMENTS:
            if name not in TRAIN_CONFIGS:
                print(f"{name}: no adapter, skipping"); continue
            print("=" * 66)
            print(f"EVALUATING {name}  [{mode}]  on {len(VAL_ANSWERS)} games")
            print("=" * 66)
            model = load_adapter(name)
            if mode == "constrained" and not _SCORER_VERIFIED:
                verify_scorer(model)
            evaluate(model, f"qwen_{name}", name, mode)
            del model; torch.cuda.empty_cache()

    # ---- the missing control -------------------------------------------
    # Untrained Qwen on exactly the same games and settings. Without this row
    # the SFT numbers cannot be attributed to SFT.
    if RUN_BASE_CONTROL:
        for mode in EVAL_MODES:
            print("=" * 66)
            print(f"EVALUATING base Qwen (NO SFT)  [{mode}]")
            print("=" * 66)
            model = load_base_control()
            if mode == "constrained" and not _SCORER_VERIFIED:
                verify_scorer(model)
            evaluate(model, "base_qwen", "base_qwen", mode)
            del model; torch.cuda.empty_cache()
else:
    print("evaluation skipped (RUN_EVALUATION = False)")

# Back-compat aliases so the original analysis cells below keep working. The
# primary mode is whichever was run first.
PRIMARY = EVAL_MODES[0]
GAMES = GAMES_ALL[PRIMARY]
EVAL  = {k: v for k, v in EVAL_ALL[PRIMARY].items() if k in EXPERIMENTS}
''')

# =============================================================================
md(r"""
---
# 10. Classical baselines on the identical 246 answers

These are **not** the full 2,315-game numbers and are labelled separately
throughout. Comparing a 246-game result against a 2,315-game result would be
meaningless.
""")

code(r'''
BASELINES = []
if RUN_BASELINES:
    ensure_code_on_path()
    from tree_search import TreeSearchConfig, TreeSearchSolver
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    lower = [a.lower() for a in VAL_ANSWERS]

    def summarize_classical(label, games, n=len(VAL_ANSWERS)):
        scores = [g.score for g in games]
        solved = [g.n_guesses for g in games if g.solved]
        dist = {k: sum(1 for g in games if g.solved and g.n_guesses == k)
                for k in range(1, 7)}
        cum = lambda m: 100 * sum(dist[i] for i in range(1, m + 1)) / n
        return {"model": label, "n_games": n,
                "mean_failures_as_7": round(sum(scores)/n, 4),
                "median": float(np.median(solved)) if solved else None,
                "max": max(solved) if solved else None,
                "pct_le3": round(cum(3), 2), "pct_le4": round(cum(4), 2),
                "pct_le5": round(cum(5), 2), "pct_le6": round(cum(6), 2),
                "failure_rate_pct": round(100*(n-len(solved))/n, 2),
                "distribution": dist, "classical": True}

    for lbl in ["random", "frequency", "entropy"]:
        sv = make_solver(lbl, BUNDLE.fb, cfgc, BUNDLE.model); sv.reset()
        op = sv.opening_guess() if sv.deterministic else None
        BASELINES.append(summarize_classical(
            lbl, [play_game(sv, a, first_guess=op) for a in lower]))
        print(f"  {lbl:<12} {BASELINES[-1]['mean_failures_as_7']:.4f}")
    for lbl, op in [("tree_soare", "soare"), ("tree_salet", "salet")]:
        tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                              endgame_threshold=10, opening_guess=op,
                              max_guesses=MAX_GUESSES)
        sv = TreeSearchSolver(BUNDLE.fb, cfgc, tc)
        BASELINES.append(summarize_classical(
            lbl, [play_game(sv, a, first_guess=op) for a in lower]))
        print(f"  {lbl:<12} {BASELINES[-1]['mean_failures_as_7']:.4f}")

ALL_ROWS = list(BASELINES)
for _m in EVAL_MODES:
    for _k, _r in EVAL_ALL[_m].items():
        r = dict(_r)
        r["model"] = f"{_r['model']} [{_m[:6]}]"
        ALL_ROWS.append(r)
if ALL_ROWS:
    print()
    print_scores(ALL_ROWS,
                 f"ALL SOLVERS - {len(VAL_ANSWERS)} HELD-OUT ANSWERS ONLY\n"
                 f"(NOT comparable to the full 2315-game benchmark)")
''')

# =============================================================================
md(r"""
---
# 11. What did the models actually learn?

Two questions that matter far more than training loss.

**Opening word.** Turn 1 carries no information, so the expert's opener is a
constant the model can simply memorize. High opener accuracy proves memorization
worked; it says nothing about search.

**Disagreement states.** The 86 states where `entropy` and `tree_soare` choose
differently are the only place their policies are distinguishable. For each such
state reached during evaluation we record which expert the model agreed with —
or neither.
""")

code(r'''
from collections import Counter

OPENERS = {m: {} for m in EVAL_MODES}
for mode in EVAL_MODES:
    print(f"--- {mode} ---")
    for name, gs in GAMES_ALL[mode].items():
        first = [g.guesses[0] for g in gs if g.guesses]
        want = (EXPERIMENTS[name]["expected_opener"]
                if name in EXPERIMENTS else None)
        c = Counter(first)
        OPENERS[mode][name] = {
            "expected": want,
            "pct_expected": (round(100 * c.get(want, 0) / max(len(first), 1), 2)
                             if want else None),
            "most_common": c.most_common(5),
            "n_distinct": len(c),
        }
        pe = OPENERS[mode][name]["pct_expected"]
        print(f"  {name:<14} expected {str(want):<6}: "
              f"{('%6.2f%%' % pe) if pe is not None else '   n/a':>8}   "
              f"top: {c.most_common(3)}")
''')

code(r'''
CONTRAST = [json.loads(l) for l in open(
    os.path.join(SFT_DIR, "data/disagreement/contrast.jsonl"), encoding="utf-8")]
BY_HIST = {" ".join(f"{h['guess']}:{h['feedback']}" for h in c["meta"]["history"]): c
           for c in CONTRAST}
print(f"{len(CONTRAST)} known entropy-vs-tree disagreement states\n")

def disagreement_for(gs):
    hits = []
    for g in gs:
        hist = []
        for i, (gu, pat) in enumerate(zip(g.guesses, g.patterns)):
            k = " ".join(f"{a}:{b}" for a, b in hist)
            c = BY_HIST.get(k)
            if c is not None and pat is not None:
                hits.append({
                    "history": k,
                    "n_candidates": c["meta"]["n_candidates"],
                    "entropy_action": c["completions"]["entropy"],
                    "tree_action": c["completions"]["tree_soare"],
                    "model_action": gu,
                    "agrees": ("entropy" if gu == c["completions"]["entropy"]
                               else "tree" if gu == c["completions"]["tree_soare"]
                               else "neither"),
                })
            if pat is not None:
                hist.append((gu, pat))
    n = len(hits)
    return {
        "n_disagreement_states_encountered": n,
        "pct_agree_entropy": round(100*sum(1 for h in hits if h["agrees"]=="entropy")/max(n,1), 2),
        "pct_agree_tree":    round(100*sum(1 for h in hits if h["agrees"]=="tree")/max(n,1), 2),
        "pct_neither":       round(100*sum(1 for h in hits if h["agrees"]=="neither")/max(n,1), 2),
        "examples": hits[:15],
    }

DISAGREE = {m: {} for m in EVAL_MODES}
for mode in EVAL_MODES:
    print(f"--- {mode} ---")
    for name, gs in GAMES_ALL[mode].items():
        DISAGREE[mode][name] = d = disagreement_for(gs)
        print(f"  {name:<14} encountered {d['n_disagreement_states_encountered']:>4}  "
              f"entropy {d['pct_agree_entropy']:>6.2f}%  "
              f"tree {d['pct_agree_tree']:>6.2f}%  "
              f"neither {d['pct_neither']:>6.2f}%")

if len(EVAL_MODES) == 2:
    print("\nPOLICY TRANSFER, unconstrained vs constrained")
    hdr = f"{'model':<16}{'mode':<15}{'entropy%':>10}{'tree%':>8}{'neither%':>10}{'n':>6}"
    print(hdr); print("-" * len(hdr))
    for name in sorted(set(DISAGREE[EVAL_MODES[0]]) | set(DISAGREE[EVAL_MODES[1]])):
        for mode in EVAL_MODES:
            d = DISAGREE[mode].get(name)
            if d:
                print(f"{name:<16}{mode:<15}{d['pct_agree_entropy']:>10.2f}"
                      f"{d['pct_agree_tree']:>8.2f}{d['pct_neither']:>10.2f}"
                      f"{d['n_disagreement_states_encountered']:>6}")

print("\nReading this: a model trained on tree_soare should agree with 'tree'")
print("more than the entropy-trained model does. If all three land near the")
print("same split, SFT did not transfer the policy difference -- only surface")
print("format and the opening word.")
print("If constrained decoding RAISES agreement, part of the apparent policy")
print("loss was really spelling failure. If it does not move, the policy")
print("result already stood on its own.")
''')

# =============================================================================
md(r"""
---
# 12. Example games
""")

code(r'''
def show_game(g, title):
    print(f"--- {title} --- (answer {g.answer}, evaluation metadata only)")
    for i, (gu, pat) in enumerate(zip(g.guesses, g.patterns), 1):
        st = g.statuses[i-1]
        print(f"  Turn {i}  Guess {gu:<9} Feedback {pat or '(no feedback: '+st+')'}"
              f"   [candidates left: {g.remaining[i-1]}]")
    print(f"  => {'SOLVED in ' + str(len(g.guesses)) if g.solved else 'FAILED'}\n")

EXAMPLES = {}
for mode in EVAL_MODES:
  EXAMPLES[mode] = {}
  for name, gs in GAMES_ALL[mode].items():
    print("=" * 66); print(f"EXAMPLE GAMES - {name}  [{mode}]"); print("=" * 66)
    picks = {}
    for g in gs:
        n = len(g.guesses)
        if g.solved and n <= 3 and "le3" not in picks: picks["le3"] = ("solved in <=3", g)
        if g.solved and n == 4 and "eq4" not in picks: picks["eq4"] = ("solved in 4", g)
        if g.solved and n >= 5 and "ge5" not in picks: picks["ge5"] = ("solved in 5-6", g)
        if not g.solved and "fail" not in picks: picks["fail"] = ("FAILED", g)
        if any(s in ("invalid_format", "invalid_word") for s in g.statuses) \
           and "bad" not in picks: picks["bad"] = ("malformed output", g)
        if len(g.guesses) != len(set(g.guesses)) and "rep" not in picks:
            picks["rep"] = ("repeated a guess", g)
    for k, (title, g) in picks.items():
        show_game(g, title)
    EXAMPLES[mode][name] = {k: {"answer": g.answer, "guesses": g.guesses,
                                "patterns": g.patterns, "statuses": g.statuses,
                                "remaining": g.remaining, "solved": g.solved}
                            for k, (t, g) in picks.items()}
''')

# =============================================================================
md(r"""
---
# 12b. The terminal-state probe

This is the diagnostic the whole session exists for.

The classical solver is rolled forward — **the model plays no part in reaching
these states** — until exactly `k` candidate answers remain, for `k = 1, 2, 3`.
At `k = 1` the visible constraints already determine the answer uniquely: no
search, no information gathering, nothing left to decide. The only thing left is
to name the word.

At that state we ask the model to rank all 12,972 legal words and record:

* **top-1 accuracy** — is the model's first choice the answer?
* **top-1 in candidate set** — is it at least *a* word still consistent?
* **rank of the answer** out of 12,972, and its rank restricted to the `k`
  candidates.

Chance is `1/12972` for the vocabulary and `1/k` within the candidate set.

Each state is also tagged with whether that answer ever appeared as a training
target. If accuracy is high on seen answers and collapses on unseen ones, the
bottleneck is lexical recall. If it is low on both, the model cannot do the
reasoning even when spelling is free.
""")

code(r'''
def terminal_states(answers, ks, solver_label="entropy"):
    """Roll the CLASSICAL solver forward to states with exactly k candidates.

    The model is not involved. These are expert-reached states, so the
    constraints are exactly the ones a competent player would hold.
    """
    cfgc = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool="full")
    sv = make_solver(solver_label, BUNDLE.fb, cfgc, BUNDLE.model)
    sv.reset()
    opener = sv.opening_guess()
    out = {k: [] for k in ks}
    want = set(ks)
    for ans in answers:
        low = ans.lower()
        cands = np.arange(VOCAB.n_answers, dtype=np.int32)
        hist, seen = [], set()
        # turn is the guess number the model would be asked to produce next
        for turn in range(1, MAX_GUESSES + 1):
            n = int(len(cands))
            if n in want and n not in seen:
                seen.add(n)
                out[n].append({
                    "answer": ans,
                    "k": n,
                    "turn": turn,
                    "candidates": [VOCAB.answers[i].upper() for i in cands],
                    "prompt": render_prompt(
                        turn=turn, history=list(hist),
                        constraints=derive_constraints(hist),
                        n_candidates=n,
                        guesses_remaining=MAX_GUESSES - turn + 1,
                        max_guesses=MAX_GUESSES,
                        candidates=None, show_candidate_count=False),
                })
            if n <= 1 or not want - seen:
                break
            g = opener if turn == 1 else sv.choose(cands, turn)
            code_ = feedback_code(g, low)
            hist.append((g, code_to_pattern(code_)))
            cands = BUNDLE.fb.filter_indices(cands, g, code_)
            if code_ == ALL_GREEN:
                break
    return out

@torch.no_grad()
def probe_terminal(model, states, label, max_states=None):
    sc = build_scorer()
    rows, per_k = [], {}
    for k, sts in sorted(states.items()):
        use = sts if max_states is None else sts[:max_states]
        hit1 = hit_cand = 0
        ranks, ranks_in_cand = [], []
        for st in use:
            scores = sc.score_all(model, st["prompt"])
            top1 = sc.words[int(scores.argmax().item())]
            r = sc.rank_of(scores, [st["answer"]]).get(st["answer"])
            cand_scores = [(sc.words[sc.index[c]], scores[sc.index[c]].item())
                           for c in st["candidates"] if c in sc.index]
            cand_scores.sort(key=lambda x: -x[1])
            r_in = next((i + 1 for i, (w, _) in enumerate(cand_scores)
                         if w == st["answer"]), None)
            hit1 += int(top1 == st["answer"])
            hit_cand += int(top1 in set(st["candidates"]))
            if r: ranks.append(r)
            if r_in: ranks_in_cand.append(r_in)
            rows.append({"model": label, "k": k, "answer": st["answer"],
                         "top1": top1, "correct": top1 == st["answer"],
                         "top1_is_candidate": top1 in set(st["candidates"]),
                         "rank_of_answer": r, "rank_within_candidates": r_in,
                         "answer_in_train_targets":
                             st["answer"] in TRAIN_TARGET_WORDS})
        n = max(len(use), 1)
        per_k[k] = {
            "n_states": len(use),
            "top1_accuracy_pct": round(100 * hit1 / n, 2),
            "top1_is_candidate_pct": round(100 * hit_cand / n, 2),
            "median_rank_of_answer": (float(np.median(ranks)) if ranks else None),
            "mean_rank_within_candidates": (round(float(np.mean(ranks_in_cand)), 3)
                                            if ranks_in_cand else None),
            "chance_top1_pct": round(100 / sc.n, 4),
            "chance_within_candidates_pct": round(100 / k, 2),
        }
    return per_k, rows

TERMINAL, TERMINAL_ROWS = {}, []
if RUN_TERMINAL_PROBE and RUN_EVALUATION:
    # Which answers were ever a training target? Cross-tabulating against this
    # is the direct test of the "held-out vocabulary" hypothesis.
    TRAIN_TARGET_WORDS = set()
    for _ds in {EXPERIMENTS[n]["dataset"] for n in EXPERIMENTS}:
        p = os.path.join(SFT_DIR, f"data/{_ds}/train.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                TRAIN_TARGET_WORDS.add(json.loads(l)["completion"].upper())
    print(f"distinct training targets across datasets: {len(TRAIN_TARGET_WORDS)}")

    STATES = terminal_states(VAL_ANSWERS, TERMINAL_KS)
    for k in TERMINAL_KS:
        print(f"  k={k}: {len(STATES[k])} probe states")

    targets = [(n, load_adapter, f"qwen_{n}") for n in EXPERIMENTS
               if n in TRAIN_CONFIGS]
    if RUN_BASE_CONTROL:
        targets.append(("base_qwen", lambda _n: load_base_control(), "base_qwen"))

    for name, loader, label in targets:
        print(f"\n--- terminal probe: {label} ---")
        model = loader(name)
        if not _SCORER_VERIFIED:
            verify_scorer(model)
        t0 = time.perf_counter()
        per_k, rows = probe_terminal(model, STATES, label)
        TERMINAL[label] = per_k
        TERMINAL_ROWS.extend(rows)
        for k, r in sorted(per_k.items()):
            print(f"  k={k}  top1={r['top1_accuracy_pct']:>6.2f}%  "
                  f"(chance {r['chance_top1_pct']}%)   "
                  f"top1_is_candidate={r['top1_is_candidate_pct']:>6.2f}%  "
                  f"median_rank_of_answer={r['median_rank_of_answer']}")
        print(f"  ({time.perf_counter()-t0:.0f}s)")
        del model; torch.cuda.empty_cache()

    # Seen vs unseen answer words -- the hypothesis test.
    print("\n" + "=" * 66)
    print("TOP-1 ACCURACY AT k=1, SPLIT BY WHETHER THE ANSWER WAS EVER A "
          "TRAINING TARGET")
    print("=" * 66)
    hdr = f"{'model':<18}{'seen n':>8}{'seen acc':>10}{'unseen n':>10}{'unseen acc':>12}"
    print(hdr); print("-" * len(hdr))
    TERMINAL_SPLIT = {}
    for label in sorted({r["model"] for r in TERMINAL_ROWS}):
        sub = [r for r in TERMINAL_ROWS if r["model"] == label and r["k"] == 1]
        seen = [r for r in sub if r["answer_in_train_targets"]]
        unseen = [r for r in sub if not r["answer_in_train_targets"]]
        f = lambda xs: (round(100*sum(x["correct"] for x in xs)/len(xs), 2)
                        if xs else None)
        TERMINAL_SPLIT[label] = {"seen_n": len(seen), "seen_acc_pct": f(seen),
                                 "unseen_n": len(unseen), "unseen_acc_pct": f(unseen)}
        print(f"{label:<18}{len(seen):>8}{str(f(seen)):>10}"
              f"{len(unseen):>10}{str(f(unseen)):>12}")
else:
    TERMINAL_SPLIT = {}
    print("terminal probe skipped")
''')

# =============================================================================
md(r"""
---
# 13. Save everything and build the ZIP
""")

code(r'''
env = {
    "python": platform.python_version(), "platform": platform.platform(),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "peft": peft.__version__, "accelerate": accelerate.__version__,
    "numpy": np.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "cuda": torch.version.cuda,
    "fp16": FP16, "seed": SEED,
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
results = {
    "note": f"All model results are on the {len(VAL_ANSWERS)} HELD-OUT answers. "
            f"NOT comparable to the full 2315-game benchmark.",
    "n_eval_answers": len(VAL_ANSWERS),
    "model_name": MODEL_NAME,
    "shared_hyperparameters": {
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGETS, "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS, "per_device_batch_size": PER_DEVICE_BS,
        "grad_accum": GRAD_ACCUM, "max_seq_len": MAX_SEQ_LEN,
        "fp16": FP16, "seed": SEED,
    },
    "training": TRAIN_CONFIGS,
    "model_eval": EVAL,                     # primary mode, back-compat
    "model_eval_by_mode": EVAL_ALL,         # every mode run this session
    "classical_baselines_same_246": BASELINES,
    "opening_analysis": OPENERS,
    "disagreement_analysis": DISAGREE,
    "terminal_probe": TERMINAL,
    "terminal_probe_seen_vs_unseen": TERMINAL_SPLIT,
    "example_games": EXAMPLES,
    "eval_settings": {
        "greedy": True, "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "candidate_count_shown": SHOW_CANDIDATE_COUNT,
        "candidate_list_shown": False, "answer_shown": False,
        "modes_run": EVAL_MODES,
        "constrained": {
            "legal_words": len(LEGAL_WORDS_SORTED),
            "score": "sum log P(token | prompt, prefix) over ' '+WORD tokens "
                     "and EOS; argmax over the legal set. Not post-hoc "
                     "filtering: the model never emits free text in this mode.",
            "chunk": CONSTRAINED_CHUNK,
            "exact_pruning": CONSTRAINED_PRUNE,
            "length_normalise": LENGTH_NORMALISE,
            "repeats_banned": BAN_REPEATS_IN_CONSTRAINED,
            "candidate_list_shown": False,
            "answer_shown": False,
        },
        "output_policy": "first 5-letter token; else first later 5-letter token "
                         "(flagged); else invalid. Invalid turns are consumed "
                         "with no feedback. No silent repair.",
    },
}
for path, obj in [("results.json", results), ("environment.json", env),
                  ("dataset_hashes.json", DATASET_HASHES),
                  ("training_config.json", results["shared_hyperparameters"])]:
    with open(os.path.join(WORK_DIR, path), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    print(f"wrote {os.path.join(WORK_DIR, path)}")

base = RESULTS_ZIP[:-4] if RESULTS_ZIP.endswith(".zip") else RESULTS_ZIP
shutil.make_archive(base, "zip", WORK_DIR)
print(f"\n{'='*66}")
print(f"ZIP READY: {base}.zip  ({os.path.getsize(base + '.zip')/2**20:.1f} MiB)")
print(f"{'='*66}")
print("Download it from the right sidebar -> Output, or the Data tab.")
print("\nTO CONTINUE IN A LATER SESSION:")
print("  1. Output tab -> New Dataset -> save /kaggle/working/wordle_sft")
print("  2. Next session: Add Input that dataset")
print("  3. Set PREV_RUN_DIR to its path and turn off the RUN_* flags")
print("     for whatever is already trained.")
''')

# =============================================================================
md(r"""
---
# 13b. Diagnostic results tree

The original evaluation is **not** overwritten. Diagnostic output goes to its
own tree and its own ZIP:

```
results/
  unconstrained_sft/    the three adapters, free generation
  constrained_sft/      the three adapters, legal-word argmax
  base_qwen/            untrained Qwen, both modes
  comparison.csv        one row per (model, mode)
  comparison.md         the table below
  terminal_probe.csv    one row per probe state
```
""")

code(r'''
import csv

def _dump(d, path, obj):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, path), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)

for mode in EVAL_MODES:
    for name, row in EVAL_ALL[mode].items():
        tgt = (RESULT_DIRS["base_qwen"] if name == "base_qwen"
               else RESULT_DIRS[f"{mode}_sft"])
        _dump(tgt, f"{name}__{mode}.json", {
            "metrics": row,
            "opening_analysis": OPENERS[mode].get(name),
            "disagreement_analysis": DISAGREE[mode].get(name),
        })

_dump(RESULTS_ROOT, "environment.json", env)
_dump(RESULTS_ROOT, "eval_settings.json", results["eval_settings"])
_dump(RESULTS_ROOT, "classical_baselines_same_246.json", BASELINES)
if TERMINAL:
    _dump(RESULTS_ROOT, "terminal_probe.json",
          {"per_model": TERMINAL, "seen_vs_unseen": TERMINAL_SPLIT})

# ---- comparison table ------------------------------------------------------
FIELDS = ["model", "mode", "mean_failures_as_7", "failure_rate_pct",
          "solved_pct", "invalid_pct", "invalid_format_rate_pct",
          "invalid_word_rate_pct", "repeated_guess_game_rate_pct",
          "hard_mode_violation_pct", "pct_le3", "pct_le4", "pct_le5",
          "pct_le6", "median", "max", "avg_decision_margin"]

def _rowify(r, model, mode):
    inv = r.get("invalid_format_rate_pct", 0) + r.get("invalid_word_rate_pct", 0)
    out = {k: r.get(k) for k in FIELDS}
    out.update(model=model, mode=mode,
               solved_pct=round(100 - r["failure_rate_pct"], 2),
               invalid_pct=round(inv, 2))
    return out

TABLE = []
for b in BASELINES:
    TABLE.append(_rowify({**b, "invalid_format_rate_pct": 0,
                          "invalid_word_rate_pct": 0,
                          "repeated_guess_game_rate_pct": 0,
                          "hard_mode_violation_pct": 0},
                         f"classical {b['model']}", "symbolic"))
for mode in EVAL_MODES:
    for name, r in EVAL_ALL[mode].items():
        TABLE.append(_rowify(r, r["model"], mode))

with open(os.path.join(RESULTS_ROOT, "comparison.csv"), "w",
          newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for t in TABLE:
        w.writerow(t)

lines = ["| Model | Mode | Mean | Fail | Solved | Invalid | Repeat | <=3 | <=4 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for t in TABLE:
    lines.append(
        f"| {t['model']} | {t['mode']} | {t['mean_failures_as_7']:.4f} | "
        f"{t['failure_rate_pct']:.1f}% | {t['solved_pct']:.1f}% | "
        f"{t['invalid_pct']:.1f}% | {t['repeated_guess_game_rate_pct']:.1f}% | "
        f"{t['pct_le3']:.1f} | {t['pct_le4']:.1f} |")
md_table = "\n".join(lines)
with open(os.path.join(RESULTS_ROOT, "comparison.md"), "w",
          encoding="utf-8") as fh:
    fh.write("# Constrained-decoding diagnostic\n\n" + md_table + "\n")

if TERMINAL_ROWS:
    with open(os.path.join(RESULTS_ROOT, "terminal_probe.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TERMINAL_ROWS[0].keys()))
        w.writeheader()
        for r in TERMINAL_ROWS:
            w.writerow(r)

print(md_table)

_dbase = DIAG_ZIP[:-4] if DIAG_ZIP.endswith(".zip") else DIAG_ZIP
shutil.make_archive(_dbase, "zip", RESULTS_ROOT)
print(f"\nDIAGNOSTIC ZIP: {_dbase}.zip "
      f"({os.path.getsize(_dbase + '.zip')/2**20:.2f} MiB)")
print("The original wordle_sft_results.zip is untouched.")
''')

# =============================================================================
md(r"""
---
# 13c. Verdict

Which of these do the numbers support?

**A.** SFT learned the policy; vocabulary generation was the main bottleneck.
**B.** SFT learned some policy but still cannot reason well enough.
**C.** SFT barely improved over base Qwen.
**D.** Something else.

The cell below applies the decision rule mechanically so the reading is not
retrofitted to the result. It prints the evidence alongside the verdict; the
evidence is what matters, and a borderline case should be argued, not accepted
because a threshold fired.
""")

code(r'''
def verdict():
    if not EVAL_ALL.get("constrained"):
        return "INCONCLUSIVE", ["constrained mode was not run"]

    ev, sft = [], [n for n in EXPERIMENTS if n in EVAL_ALL["constrained"]]
    if not sft:
        return "INCONCLUSIVE", ["no SFT adapters were evaluated"]

    con = min(EVAL_ALL["constrained"][n]["mean_failures_as_7"] for n in sft)
    unc = (min(EVAL_ALL["unconstrained"][n]["mean_failures_as_7"] for n in sft)
           if EVAL_ALL.get("unconstrained") else None)
    base_c = EVAL_ALL["constrained"].get("base_qwen", {}).get("mean_failures_as_7")
    ent = next((b["mean_failures_as_7"] for b in BASELINES
                if b["model"] == "entropy"), 3.44)
    rnd = next((b["mean_failures_as_7"] for b in BASELINES
                if b["model"] == "random"), 4.02)

    ev.append(f"best SFT constrained mean   = {con:.4f}")
    if unc is not None:
        ev.append(f"best SFT unconstrained mean = {unc:.4f} "
                  f"(constrained gains {unc - con:+.4f})")
    ev.append(f"base Qwen constrained mean  = "
              f"{'n/a' if base_c is None else format(base_c, '.4f')}")
    ev.append(f"classical entropy = {ent:.4f}, classical random = {rnd:.4f}")
    if TERMINAL:
        for lbl, per_k in TERMINAL.items():
            if 1 in per_k:
                ev.append(f"terminal k=1 top-1 [{lbl}] = "
                          f"{per_k[1]['top1_accuracy_pct']:.1f}% "
                          f"(chance {per_k[1]['chance_top1_pct']}%)")
    for lbl, s in TERMINAL_SPLIT.items():
        ev.append(f"k=1 seen vs unseen [{lbl}]: "
                  f"{s['seen_acc_pct']}% (n={s['seen_n']}) vs "
                  f"{s['unseen_acc_pct']}% (n={s['unseen_n']})")

    # C first: if SFT is not meaningfully better than base, nothing else matters.
    if base_c is not None and con >= base_c - 0.25:
        return "C", ev + ["-> constrained SFT is not meaningfully better "
                          "than constrained base Qwen"]
    if con <= ent + 0.35:
        return "A", ev + ["-> constrained SFT lands near the classical expert: "
                          "vocabulary generation was the binding constraint"]
    if con < rnd:
        return "B", ev + ["-> constrained SFT beats base and random but stays "
                          "well short of the expert: real but insufficient "
                          "policy"]
    return "D", ev + ["-> constrained SFT still loses to random elimination; "
                      "the failure is not primarily vocabulary"]

V, EVIDENCE = verdict()
print("=" * 70)
print(f"VERDICT: {V}")
print("=" * 70)
for e in EVIDENCE:
    print("  " + e)
print("""
A = SFT learned the policy; vocabulary generation was the main bottleneck.
B = SFT learned some policy but still cannot reason well enough.
C = SFT barely improved over base Qwen.
D = Something else.

Do not start GRPO on this alone. If A, the next move is cheap: keep constrained
decoding and re-measure. If B or D, the training distribution is the problem
(6 fully-determined examples in 7,173) and more RL on the same data will not
fix it.""")
_dump(RESULTS_ROOT, "verdict.json", {"verdict": V, "evidence": EVIDENCE})
''')

# =============================================================================
md(r"""
---
# 14. Summary
""")

code(r'''
print("=" * 78)
print(f"SFT RESULTS - {len(VAL_ANSWERS)} HELD-OUT ANSWERS")
print("(classical full-2315 means shown for reference only - DIFFERENT SET)")
print("=" * 78)
if ALL_ROWS:
    print_scores(ALL_ROWS, "")
print()
for name in EXPERIMENTS:
    if name not in EVAL:
        print(f"{name:<14} NOT RUN"); continue
    e, o, d = EVAL[name], OPENERS.get(name, {}), DISAGREE.get(name, {})
    tc = TRAIN_CONFIGS.get(name, {})
    print(f"{name}")
    print(f"  mean {e['mean_failures_as_7']:.4f} | fail {e['failure_rate_pct']:.1f}% "
          f"| invalid {e['invalid_format_rate_pct']+e['invalid_word_rate_pct']:.1f}% "
          f"| expert (2315 games) {EXPERIMENTS[name]['expert_mean_2315']:.4f}")
    print(f"  opener {o.get('expected')} used {o.get('pct_expected')}% of games")
    print(f"  disagreement: entropy {d.get('pct_agree_entropy')}%  "
          f"tree {d.get('pct_agree_tree')}%  neither {d.get('pct_neither')}%")
    print(f"  trained {tc.get('training_seconds','?')}s, "
          f"final loss {tc.get('final_loss','?')}")
print("\nNo GRPO was run. Decide on RL only after reading these numbers.")
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
