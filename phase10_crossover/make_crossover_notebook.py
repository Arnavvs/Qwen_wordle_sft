"""
make_crossover_notebook.py - generate wordle_phase10_crossover_kaggle.ipynb.

Trains ONE adapter, `tree_salet_endgame_rawhist`, on exactly the Phase 6 data
re-rendered into the `raw_history` prompt format. Nothing else changes: same
rows, same completions, same hyperparameters, same seed, same shuffle.

WHY

Every prompt result in Phase 9 carries the same confound. The adapter was
trained on `baseline` alone, so "baseline wins by 0.52 guesses" and "raw_history
emits 1.4% admissible words against baseline's 22.3%" may both be measuring
format lock-in rather than anything about the formats. The Phase 9 `base` arm
was supposed to break that tie and could not - stock Qwen scores 6.71-6.88
under every prompt, so it cannot rank prompts at all.

A second TRAINED adapter can. This notebook produces it. The 2x2 evaluation is
then run by the existing Phase 9 harness with

    ARMS            = ["sft", "sft_rawhist"]
    VARIANTS_TO_RUN = ["baseline", "raw_history"]

which already has the fixed decoder, the paired answer set, the sanity gate and
the per-decision logging. This file deliberately does NOT re-implement any of
that.

READING THE RESULT

    row = the adapter, column = the prompt it is evaluated under

                        eval baseline   eval raw_history
    trained baseline       3.7642            4.2805        <- both measured
    trained raw_history      ?                 ?           <- this run

  * each adapter best on its own format, similar margins
        -> LOCK-IN. Format is a robustness problem; the Phase 9 ranking says
           nothing about which prompt is intrinsically better, and the honest
           write-up is "this model is brittle to format", not "this format is
           better".
  * raw_history adapter recovers toward 3.76 on raw_history
        -> the deduction the solver hands the model is LEARNABLE. The constraint
           block was a crutch for this adapter, not a requirement, and the
           "the harness is the model" worry mostly dissolves.
  * raw_history adapter stays near 4.28 on raw_history
        -> the deduction genuinely needs the solver at 0.5B. The constraint
           block is honest scaffolding and stays, and Phase 9 finding 2 is
           confirmed rather than merely consistent.

Pre-registered before the run, so the outcome cannot be re-read to suit
whatever comes back. Phase 9's decision table failed to anticipate its own
outcome shape and that is what left the GRPO-vs-harness fork open.

    python phase10_crossover/make_crossover_notebook.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.path.join(ROOT, "core")
OUT = os.path.join(HERE, "wordle_phase10_crossover_kaggle.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n")})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# =============================================================================
md(r"""
# Phase 10 — the format crossover (training half)

One adapter, one question:

> When Phase 9 measured `baseline` beating `raw_history` by 0.52 guesses, was it
> measuring the **format**, or was it measuring that the adapter had only ever
> seen `baseline`?

## Why the Phase 9 control failed

Phase 9 ran a `base` arm — stock Qwen2.5-0.5B, for which all twelve prompts are
equally unfamiliar — precisely to answer this. It could not: all nine runs
landed between 6.71 and 6.88 mean with 91–97% failure and a spread of 0.17. A
model that cannot play Wordle under *any* prompt cannot rank prompts. The arm
consumed 84.5% of the session's wall time and settled nothing.

The tie needs a model with non-trivial skill under a second format. That means
training one.

## What this notebook does

Takes the **exact** Phase 6 training set — 19,212 rows, the same completions,
the same expert actions — and rewrites only the prompt text into `raw_history`
format: guesses and feedback, no solver-derived `Confirmed letters` block.
Then trains with hyperparameters identical to Phase 6 in every respect,
including the seed and the shuffle order.

The only difference between `tree_salet_endgame` and
`tree_salet_endgame_rawhist` is the prompt string. That is the whole
experiment.

## What it does NOT do

No evaluation. The 2×2 is run by the Phase 9 harness, which already holds the
fixed decoder, the paired 246-answer set, the sanity gate and the per-decision
`|admissible|` logging. Re-implementing any of that here would fork the
measurement path, and a forked measurement path is exactly what voided Phase
8 v3.

## The pre-registered reading

| | eval `baseline` | eval `raw_history` |
|---|---|---|
| adapter trained on `baseline` | **3.7642** (measured) | **4.2805** (measured) |
| adapter trained on `raw_history` | ? | **?** — the cell that decides |

- Both adapters best on their own format, similar margins → **lock-in**. Format
  is a robustness problem and the Phase 9 ranking is not a quality ranking.
- Raw-history adapter recovers toward 3.76 → the deduction is **learnable**;
  the constraint block is a crutch, not a requirement.
- It stays near 4.28 → the deduction genuinely needs the solver at 0.5B, and
  the constraint block is honest scaffolding that stays.
""")

# =============================================================================
md(r"""
---
# 1. Setup
""")

code(r'''
import os, sys, json, math, time, random, zipfile, shutil, glob, re
import numpy as np
import torch, transformers, peft
from transformers import AutoTokenizer, AutoModelForCausalLM

def fix_torchao_peft_conflict():
    try:
        import peft.tuners.lora.torchao as t
        t.is_torchao_available = lambda *a, **k: False
    except Exception: pass
    return "patched"
print("torchao:", fix_torchao_peft_conflict())
print("transformers", transformers.__version__, "| peft", peft.__version__)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# ============================ EDIT THIS =====================================
DATASET_DIR = None       # sft_package dataset; None = search /kaggle/input
# ============================================================================

# ---- the experiment --------------------------------------------------------
TRAIN_VARIANT = "raw_history"    # the format to re-render into
ADAPTER_NAME  = "tree_salet_endgame_rawhist"

# ---- model / LoRA / optimisation: IDENTICAL to Phase 6, which was identical
# ---- to Phase 4. Do not tune anything here. If a hyperparameter differs, the
# ---- crossover is comparing two things at once and answers nothing.
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
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
SEED          = 20260817          # Phase 6's seed, so the shuffle matches

USE_NATURAL, USE_ENDGAME, ENDGAME_REPEAT = True, True, 1

RUN_TRAINING = True

# Phase 6's own numbers, so the run can be checked against them
PHASE6 = {"n_rows": 19212, "natural": 7067, "endgame": 12145,
          "global_steps": 1202, "first_loss": 5.8861, "final_loss": 0.7410,
          "training_seconds": 4541.1}
# Phase 9's measured cells of the 2x2, for the write-up
PHASE9 = {"baseline_on_baseline": 3.7642, "baseline_on_rawhistory": 4.2805}

WORK_DIR     = "/kaggle/working/wordle_phase10"
RESULTS_ROOT = "/kaggle/working/results_phase10"
RESULTS_ZIP  = "/kaggle/working/wordle_phase10_results.zip"
os.makedirs(WORK_DIR, exist_ok=True); os.makedirs(RESULTS_ROOT, exist_ok=True)

def set_seed_everywhere(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
set_seed_everywhere()

_ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"\nadapter         {ADAPTER_NAME}")
print(f"train variant   {TRAIN_VARIANT}")
print(f"effective batch {PER_DEVICE_BS} x {GRAD_ACCUM} x {_ngpu} GPU(s)"
      f" = {PER_DEVICE_BS*GRAD_ACCUM*_ngpu}")
print(f"expect ~{PHASE6['global_steps']} steps and ~{PHASE6['training_seconds']/60:.0f} min "
      f"(Phase 6 on the same rows)")
''')

# =============================================================================
md(r"""
---
# 2. Locate the dataset

Fails loudly and names the missing file. Every path this notebook needs is
asserted here rather than discovered halfway through training.
""")

code(r'''
REQUIRED_FILES = [
    "sft_package/data/tree_salet/train.jsonl",
    "sft_package/data/tree_salet_endgame/train.jsonl",
    "code/wordle_solver.py",
    "code/generate_trajectories.py",
    "artifacts/answers.txt",
    "artifacts/valid_guesses.txt",
]

def find_data_root():
    cands = ([DATASET_DIR] if DATASET_DIR else []) + sorted(
        glob.glob("/kaggle/input/*/kaggle_upload")) + sorted(
        glob.glob("/kaggle/input/*")) + ["."]
    for root in cands:
        if root and all(os.path.exists(os.path.join(root, f))
                        for f in REQUIRED_FILES):
            return root
    lines = ["could not locate the sft_package dataset. Looked in:"]
    for root in cands:
        if not root: continue
        miss = [f for f in REQUIRED_FILES
                if not os.path.exists(os.path.join(root, f))]
        if miss and os.path.isdir(root):
            lines.append(f"  {root}  (missing {len(miss)}, e.g. {miss[0]})")
    raise SystemExit("\n".join(lines))

DATA_ROOT = find_data_root()
SFT_DIR   = os.path.join(DATA_ROOT, "sft_package")
sys.path.insert(0, os.path.join(DATA_ROOT, "code"))
print("data root:", DATA_ROOT)

from wordle_solver import feedback_code, code_to_pattern      # noqa: E402
from generate_trajectories import derive_constraints, render_prompt  # noqa: E402
print("solver code imported")
''')

# =============================================================================
md(r"""
---
# 3. The prompt variants

`core/prompt_variants.py`, inlined verbatim so the notebook cannot drift from
the module the Phase 9 evaluation uses. If the renderer here differed from the
renderer there by so much as a space, the adapter would be trained on one
format and scored on another — a silent, plausible, entirely wrong result.
""")

with open(os.path.join(CORE, "prompt_variants.py"), encoding="utf-8") as fh:
    PV_SRC = fh.read().rstrip()
import re as _re
PV_SRC = _re.sub(
    r"^import sys\nimport os\n\nsys\.path\.insert\([^\n]*\n\nfrom generate_trajectories "
    r"import derive_constraints, render_prompt\n",
    "# (sys.path bootstrap and imports supplied by the cells above)\n",
    PV_SRC, count=1, flags=_re.M)
assert "sys.path.insert" not in PV_SRC, "prompt_variants bootstrap not stripped"

code(PV_SRC + r'''


assert TRAIN_VARIANT in VARIANTS, f"unknown variant {TRAIN_VARIANT!r}"
assert not VARIANTS[TRAIN_VARIANT]["leaky"], \
    f"{TRAIN_VARIANT!r} is leaky - never train on it"
print(f"variant {TRAIN_VARIANT!r}: {VARIANTS[TRAIN_VARIANT]['note']}")
''')

# =============================================================================
md(r"""
---
# 4. Re-render, with a proof

The rows do not store the game history as data — only as text inside the
already-rendered `baseline` prompt. So the history is parsed back out of that
text, and then, before a single row is used:

```
v_baseline(turn, parsed_history, max_guesses) == stored_prompt   byte-for-byte
```

If the parsed history reproduces the stored prompt exactly, the parse is
provably lossless, and the only difference in the re-rendered prompt is the
variant. Any mismatch aborts.

Verified locally on 2026-08-23 against all 19,212 rows: 19,212 exact
round-trips, zero parse failures. It is re-checked here because the dataset
this notebook attaches could differ from the one on the laptop.

`completion` and every `meta` field are carried through untouched, so the
expert action, split, turn and candidate count are identical between the two
training sets.
""")

code(r'''
HIST_BLOCK = re.compile(r"^History:\n((?:  \d+\. .+\n)+)", re.M)
HIST_LINE  = re.compile(r"\s*\d+\.\s+([A-Za-z]{5})\s*->\s*([GYB]{5})\s*$")
TURN_LINE  = re.compile(r"^Guess (\d+) of (\d+)", re.M)

def parse_state(prompt):
    tm = TURN_LINE.search(prompt)
    if not tm: return None
    turn, max_guesses = int(tm.group(1)), int(tm.group(2))
    if "History: (none" in prompt:
        return turn, max_guesses, []
    bm = HIST_BLOCK.search(prompt)
    if not bm: return None
    hist = []
    for line in bm.group(1).rstrip("\n").split("\n"):
        lm = HIST_LINE.match(line)
        if not lm: return None
        hist.append((lm.group(1).lower(), lm.group(2)))
    return turn, max_guesses, hist

def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

NATURAL = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet/train.jsonl"))
ENDGAME = load_jsonl(os.path.join(SFT_DIR, "data/tree_salet_endgame/train.jsonl"))

BASE_ROWS = []
if USE_NATURAL: BASE_ROWS += NATURAL
if USE_ENDGAME: BASE_ROWS += ENDGAME * ENDGAME_REPEAT
print(f"rows: natural={len(NATURAL):,}  endgame={len(ENDGAME):,}  "
      f"total={len(BASE_ROWS):,}   (Phase 6 saw {PHASE6['n_rows']:,})")
assert len(BASE_ROWS) == PHASE6["n_rows"], \
    f"row count {len(BASE_ROWS)} != Phase 6's {PHASE6['n_rows']} - different data"

n_ok = n_parse_fail = n_mismatch = 0
ROWS, EXAMPLES = [], []
for r in BASE_ROWS:
    st = parse_state(r["prompt"])
    if st is None:
        n_parse_fail += 1; continue
    turn, mx, hist = st
    if v_baseline(turn, hist, mx) != r["prompt"]:
        n_mismatch += 1; continue
    n_ok += 1
    nr = dict(r)
    nr["prompt"] = render(TRAIN_VARIANT, turn, hist, mx,
                          n_candidates=r.get("meta", {}).get("n_candidates", 0))
    assert nr["completion"] == r["completion"]
    assert nr.get("meta") == r.get("meta")
    assert nr["prompt"].rstrip().endswith("Next guess:")
    ROWS.append(nr)
    if len(EXAMPLES) < 2 and hist:
        EXAMPLES.append((r["prompt"], nr["prompt"], r["completion"]))

print(f"round-trip: ok={n_ok:,}  parse-fail={n_parse_fail}  mismatch={n_mismatch}")
if n_parse_fail or n_mismatch:
    raise SystemExit("re-render is not provably lossless - refusing to train. "
                     "A wrong prompt here trains cleanly and means nothing.")

random.Random(SEED).shuffle(ROWS)     # Phase 6's seed, same row list, same order
print(f"\n{len(ROWS):,} rows re-rendered as {TRAIN_VARIANT!r} and shuffled")

b, a, comp = EXAMPLES[0]
print("\n" + "=" * 70 + "\nBEFORE (baseline - what Phase 6/7 trained on)\n" + "=" * 70)
print(b)
print("\n" + "=" * 70 + f"\nAFTER ({TRAIN_VARIANT} - what this run trains on)\n" + "=" * 70)
print(a)
print(f"\ncompletion (unchanged): {comp}")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if TOKENIZER.pad_token is None: TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "right"

_bl = np.array([len(TOKENIZER(r["prompt"], add_special_tokens=False)["input_ids"])
                for r in BASE_ROWS[:1500]])
_nl = np.array([len(TOKENIZER(r["prompt"], add_special_tokens=False)["input_ids"])
                for r in ROWS[:1500]])
print(f"\nprompt tokens  baseline mean {_bl.mean():.0f}  ->  "
      f"{TRAIN_VARIANT} mean {_nl.mean():.0f}   "
      f"({100*(_nl.mean()-_bl.mean())/_bl.mean():+.0f}%)")
print("NOTE: a shorter prompt is expected - the constraint block is what was "
      "removed. It does not on its own make training cheaper; step count is "
      "fixed by row count.")

DATA_STATS = {"n_rows": len(ROWS), "natural": len(NATURAL),
              "endgame": len(ENDGAME) * ENDGAME_REPEAT,
              "variant": TRAIN_VARIANT, "round_trip_ok": n_ok,
              "baseline_prompt_tokens_mean": round(float(_bl.mean()), 1),
              "variant_prompt_tokens_mean": round(float(_nl.mean()), 1)}
''')

# =============================================================================
md(r"""
---
# 5. Dataset and collator

Identical to Phase 6. The prompt is masked out of the loss, so the model is
trained to produce the expert's word given the board — never to reproduce the
board itself. Over-long prompts drop their **oldest** tokens, which keeps the
`Next guess:` anchor and the most recent feedback.
""")

code(r'''
from torch.utils.data import Dataset

class WordleSFTDataset(Dataset):
    """prompt -> completion, prompt masked out of the loss."""
    def __init__(self, rows, tokenizer, max_len=MAX_SEQ_LEN):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len
        self.n_truncated = 0
        self._cache = {}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        if i in self._cache: return self._cache[i]
        r = self.rows[i]
        p_ids = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = self.tok(" " + r["completion"],
                         add_special_tokens=False)["input_ids"]
        c_ids = c_ids + [self.tok.eos_token_id]
        keep = self.max_len - len(c_ids)
        if len(p_ids) > keep:
            p_ids = p_ids[-keep:]
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

_lens = np.array([
    len(TOKENIZER(r["prompt"], add_special_tokens=False)["input_ids"])
    + len(TOKENIZER(" " + r["completion"], add_special_tokens=False)["input_ids"]) + 1
    for r in ROWS[:1500]])
print(f"token lengths: mean {_lens.mean():.0f}  p95 {np.percentile(_lens,95):.0f}"
      f"  max {_lens.max()}   MAX_SEQ_LEN={MAX_SEQ_LEN} -> "
      f"{100*(_lens>MAX_SEQ_LEN).mean():.2f}% truncated")
''')

# =============================================================================
md(r"""
---
# 6. Train

Phase 6 took 1,202 steps and ~76 minutes on a T4 for these rows. The step count
is fixed by the row count and batch size, so it should land at 1,202 again — a
different number means the data is not what it should be, and the run is worth
stopping.
""")

code(r'''
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments

def load_base_model():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=None,
        trust_remote_code=True)
    m.config.use_cache = False
    return m

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
    if n: print(f"  cast {n} trainable tensors fp16 -> fp32 (amp requirement)")
    return m

TRAIN_CONFIG = {}
out_dir = os.path.join(WORK_DIR, ADAPTER_NAME)

if RUN_TRAINING:
    print("=" * 70)
    print(f"TRAINING {ADAPTER_NAME} on {len(ROWS):,} rows rendered as {TRAIN_VARIANT!r}")
    print("=" * 70, flush=True)
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
        "name": ADAPTER_NAME, "train_variant": TRAIN_VARIANT,
        "n_train_examples": len(ROWS), "n_truncated": ds.n_truncated,
        "model_name": MODEL_NAME, "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS, "per_device_batch_size": PER_DEVICE_BS,
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
        "phase6_reference": PHASE6,
    }
    json.dump(TRAIN_CONFIG,
              open(os.path.join(out_dir, "training_config.json"), "w",
                   encoding="utf-8"), indent=2, default=str)
    print(f"\ntrained in {secs/60:.1f} min, {trainer.state.global_step} steps, "
          f"loss {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"Phase 6 for comparison: {PHASE6['global_steps']} steps, "
          f"loss {PHASE6['first_loss']:.4f} -> {PHASE6['final_loss']:.4f}, "
          f"{PHASE6['training_seconds']/60:.1f} min")
    if trainer.state.global_step != PHASE6["global_steps"]:
        print(f"  *** step count differs from Phase 6 "
              f"({trainer.state.global_step} vs {PHASE6['global_steps']}). "
              f"Same rows and same batch size should give the same steps - "
              f"check the data before trusting the crossover.")
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
# 7. Smoke test — does it play at all?

Not the evaluation. Twenty raw greedy generations on `raw_history` boards, no
decoder, purely to catch a dead adapter before it is uploaded and a whole
evaluation session is spent on it.

The bar is deliberately low. Phase 9 measured the *baseline* adapter emitting
1.4% admissible words under `raw_history`; anything meaningfully above that
means this adapter learned something the other one had not. A parse rate near
zero means the run is broken.
""")

code(r'''
LEGAL = set(w.strip().upper() for w in
            open(os.path.join(DATA_ROOT, "artifacts/valid_guesses.txt"))
            if w.strip())
LEGAL |= set(w.strip().upper() for w in
             open(os.path.join(DATA_ROOT, "artifacts/answers.txt")) if w.strip())

SMOKE = [r for r in ROWS if r["meta"]["turn"] >= 3][:20]
m = load_base_model()
m = PeftModel.from_pretrained(m, out_dir, is_trainable=False).to("cuda").eval()
m.config.use_cache = True

ok_parse = ok_legal = 0
print(f"{'emitted':<12}{'legal':<8}expert")
print("-" * 40)
with torch.no_grad():
    for r in SMOKE:
        ids = TOKENIZER(r["prompt"], return_tensors="pt").to("cuda")
        out = m.generate(**ids, max_new_tokens=8, do_sample=False, num_beams=1,
                         pad_token_id=TOKENIZER.pad_token_id, use_cache=True)
        txt = TOKENIZER.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        tok = "".join(c for c in txt.strip().split("\n")[0].upper()
                      if c.isalpha())[:5]
        lg = tok in LEGAL
        ok_parse += len(tok) == 5; ok_legal += lg
        print(f"{tok:<12}{str(lg):<8}{r['completion']}")
n = len(SMOKE)
print(f"\nparse {100*ok_parse/n:.0f}%   legal {100*ok_legal/n:.0f}%   (n={n})")
print("This is a liveness check, not a result. The real numbers come from the "
      "Phase 9 harness 2x2.")
SMOKE_RESULT = {"n": n, "parse_pct": round(100*ok_parse/n, 1),
                "legal_pct": round(100*ok_legal/n, 1)}
del m; torch.cuda.empty_cache()
''')

# =============================================================================
md(r"""
---
# 8. Package the adapter

The adapter is the deliverable. It is written to `/kaggle/working` and zipped on
its own, so it can be downloaded and published as a Dataset — which is what the
Phase 9 harness will attach for the 2×2.

Phase 9's `tree_salet_endgame` adapter existed for weeks only as an ad-hoc
attachment no fresh session could reproduce, and that blocked every re-run.
Publishing this one immediately is the fix for that, not an afterthought.
""")

code(r'''
json.dump(TRAIN_CONFIG, open(os.path.join(RESULTS_ROOT, "training_config.json"),
                             "w", encoding="utf-8"), indent=2, default=str)
json.dump({"smoke": SMOKE_RESULT, "data_stats": DATA_STATS,
           "phase9_measured": PHASE9,
           "next": "run the Phase 9 harness with "
                   "ARMS=['sft','sft_rawhist'] and "
                   "VARIANTS_TO_RUN=['baseline','raw_history']"},
          open(os.path.join(RESULTS_ROOT, "summary.json"), "w",
               encoding="utf-8"), indent=2, default=str)

ADAPTER_ZIP = f"/kaggle/working/{ADAPTER_NAME}.zip"
if os.path.exists(ADAPTER_ZIP): os.remove(ADAPTER_ZIP)
with zipfile.ZipFile(ADAPTER_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(out_dir):
        if "checkpoint-" in root: continue
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.join(ADAPTER_NAME, os.path.relpath(p, out_dir)))
print(f"adapter zip: {ADAPTER_ZIP} "
      f"({os.path.getsize(ADAPTER_ZIP)/2**20:.1f} MiB)")

base = RESULTS_ZIP[:-4] if RESULTS_ZIP.endswith(".zip") else RESULTS_ZIP
shutil.make_archive(base, "zip", RESULTS_ROOT)
print(f"results zip: {base}.zip "
      f"({os.path.getsize(base + '.zip')/2**20:.2f} MiB)  -- no weights inside")

try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(ADAPTER_ZIP, "/kaggle/working")))
    display(FileLink(os.path.relpath(base + ".zip", "/kaggle/working")))
except Exception: pass

print(f"""
NEXT STEPS
  1. Download {ADAPTER_NAME}.zip
  2. Add it to the arnavyrr/wordle-adapters-v2 dataset as a new version,
     keeping the folder name {ADAPTER_NAME}/ -- the harness matches adapters
     by directory basename and refuses to substitute.
  3. Run the Phase 9 harness with:
         ARMS            = ["sft", "sft_rawhist"]
         VARIANTS_TO_RUN = ["baseline", "raw_history"]
     The gate (sft + baseline near 3.7642) is the crossover's control.
""")
print("RUN COMPLETE - adapter trained and packaged.")
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
