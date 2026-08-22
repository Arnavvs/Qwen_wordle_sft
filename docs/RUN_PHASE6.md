# Running Phase 6 on Kaggle — endgame-heavy SFT

Trains one adapter and evaluates it as a 2×2 against the Phase 5 model.
**~3.5 h on a T4.** The dataset is already built, audited, and staged.

---

## Step 1 — Upload the rebuilt SFT package

The Phase 5 dataset does **not** contain the endgame file. You need a new
upload.

Local folder, already rebuilt and byte-verified against the audited source:

```
C:\Users\HP\OneDrive\Desktop\wordle\uploads\kaggle_upload\      (64 MB)
```

It now contains `sft_package/data/tree_salet_endgame/train.jsonl`
(12,145 rows, 12,112,712 bytes, sha256 `f8e8c29fa5afced8…`).

1. <https://www.kaggle.com/datasets> → **New Dataset**
2. Drag in the **`kaggle_upload` folder**
3. Title: **`wordle-sft-package-v2`** (a new dataset — do not overwrite v1)
4. **Create**

## Step 2 — Import the notebook

<https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook** →
`wordle_endgame_sft_kaggle.ipynb`

## Step 3 — GPU on

Session options → **Accelerator** → **GPU T4 x2**.

Both GPUs are used, so the real effective batch is 4 × 4 × 2 = **32**. Phase 4
ran the same way (7,067 rows → 448 steps confirms it), so the comparison holds.
The notebook prints this explicitly.

## Step 4 — Attach both datasets

**Add Input**:

- `wordle-sft-package-v2` (from Step 1)
- your existing **adapters** dataset with the Phase 5 `tree_salet` adapter

## Step 5 — Find the adapter path

```python
import os
for r, d, f in os.walk("/kaggle/input"):
    if "adapter_config.json" in f:
        print("PREV_RUN_DIR =", os.path.dirname(r))
```

Last time this was:

```
/kaggle/input/datasets/arnavyrr/wordle-sft-adapters/kaggle_adapters_upload
```

## Step 6 — Edit the config cell (section 2)

Only **one** line has to change:

```python
PREV_RUN_DIR = "/kaggle/input/datasets/arnavyrr/wordle-sft-adapters/kaggle_adapters_upload"
```

Leave `DATASET_DIR = None` — auto-detect now requires the endgame file, so it
cannot silently pick up the old v1 package. If v1 is what it finds, it fails
with a message naming the missing file and the commands to rebuild.

Everything else is already correct: `RUN_TRAINING`, `RUN_EVALUATION`,
`RUN_BASELINES`, `RUN_TERMINAL_PROBE` all `True`; hyperparameters identical to
Phase 4.

## Step 7 — Run All

| Stage | Time | What to watch |
|---|---|---|
| setup + dataset stats | ~3 min | `paths per word: natural 1.00 → mixed 3.75` |
| train `tree_salet_endgame` | ~70 min | ~1,200 steps, loss ~4.7 → ~0.4 |
| 2×2 constrained eval | ~60 min | the reproduction check, below |
| classical baselines | ~10 min | entropy 3.4431, random 4.0203 |
| terminal probe, 2 models | ~50 min | seen vs unseen split |

## Step 8 — The three lines that must look right

**8a. Dataset composition** (section 5):

```
paths per word at k=1:  natural 1.00  ->  mixed 3.75
```

**8b. Decoder verification** (first constrained eval):

```
cache vs naive [torch.float16]: max|delta| = 0.0xxx nats (tol 0.4) ... corr = 0.99999x  OK
pruned argmax == full argmax (...), N/26 chunks  OK
sequential banning walks the global ranking [...]  OK
```

The third line is new for Phase 6 — repeat banning is on. It caught a real bug
in Phase 5.

**8c. Phase 5 reproduction check:**

```
PHASE 5 REPRODUCTION CHECK: 5.6870 vs 5.6870 (delta +0.0000)  OK
```

If this says `MISMATCH`, the environment differs from Phase 5 and **nothing
else in the run is comparable**. Stop and tell me the number.

## Step 9 — Download

- `wordle_endgame_results.zip` — small, **no weights inside** (the Phase 5 zip
  was 657 MB and the browser could not download it)
- To keep the adapter: **Output → New Dataset** on `/kaggle/working/wordle_endgame`

`save_state()` runs after every evaluation, so a session teardown costs at most
one cell. Every result is printed as well as written, so the notebook itself is
a complete record even if the disk is lost — which is exactly what happened in
Phase 5.

---

## What the result means

Read the **seen vs unseen** split before the mean.

| Outcome | Reading |
|---|---|
| `SOLVED` | reaches the classical expert (≤ 3.79) |
| `MAJOR` | beats random elimination (< 4.02) — endgame coverage was the binding constraint |
| `PARTIAL-GENERALISING` | still short of random, but unseen-word accuracy up ≥10 pts — the procedure is transferring; add more paths per word |
| `MEMORISED` | unseen accuracy moved < 3 pts — the extra paths bought memorisation, not a procedure. More of this data will not help; the task framing has to change |

Phase 5 reference: **20.0%** k=1 top-1 overall, **33.3%** seen, **15.3%** unseen,
median rank 7.5 of 12,972.

`MEMORISED` is a real possible outcome and is not a failed run — it would tell
us the approach is wrong rather than under-scaled, which is worth a session to
learn.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `FileNotFoundError` naming `tree_salet_endgame/train.jsonl` | you attached the v1 package; attach `wordle-sft-package-v2` |
| `no adapter for tree_salet` | `PREV_RUN_DIR` wrong — redo Step 5. The new adapter still trains and evaluates; only the 2 old-adapter rows are skipped |
| CUDA OOM during eval | `CONSTRAINED_CHUNK = 256` |
| running out of session time | set `TERMINAL_MODELS = ["tree_salet_endgame"]` and drop the two `tree_salet` rows from `EVAL_MATRIX` — but keep `("tree_salet", False)`, it is the reproduction check |
