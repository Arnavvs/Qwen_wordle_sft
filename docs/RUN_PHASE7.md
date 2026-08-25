# Running Phase 7 on Kaggle — feedback-consistent decoding

Retrains `tree_salet_endgame` (Phase 6 config, so it reproduces) and compares
three decoders. **~2.5 h from scratch, ~1 h if the adapter already exists.**

---

## Step 1 — Attach the v2 SFT package

`uploads/kaggle_upload` (64 MB) — the one with
`sft_package/data/tree_salet_endgame/train.jsonl`. If you already uploaded it
for Phase 6 as `wordle-sft-package-v2`, just attach that.

Optionally attach an adapters dataset too. If it contains
`tree_salet_endgame`, training is skipped entirely.

## Step 2 — Import + GPU

Import `phase7_constrained/wordle_phase7_kaggle.ipynb`.
Session options → Accelerator → **GPU T4 x2**.

## Step 3 — Config

If you have no saved adapter, change nothing — it trains from scratch.

If you do:

```python
PREV_RUN_DIR = "/kaggle/input/<...>/wordle_phase7"
```

## Step 4 — Run All

| Stage | Time |
|---|---|
| train (skipped if adapter exists) | ~70 min |
| unconstrained eval | ~2 min |
| legal eval x2 (ban / noban) | ~30 min |
| consistent eval x2 | ~6 min (the filter makes it fast) |
| adaptive eval | ~5 min |
| baselines | ~10 min |
| k=1 probe | ~25 min |

## Step 5 — ⚠ SAVE THE ADAPTER

Section 5 stops and tells you to do this. **Do it.** Phase 6's weights were lost
to a session teardown and had to be retrained — that is the only reason this
phase includes a training step at all.

Output tab → New Dataset → save `/kaggle/working/wordle_phase7`.

## Step 6 — Download `wordle_phase7_results.zip`

Small, no weights.

---

## Resumability

`results.json` is reloaded at startup. A re-run skips completed evaluations
instead of repeating them, and skips training if the adapter is present. If the
session dies mid-way, re-running costs only what had not finished.

## What to check

**Phase 6 reproduction** — `legal/noban` must return ≈ 5.6992:

```
PHASE 6 REPRODUCTION: 5.6992 vs 5.6992 (+0.0000)  OK
```

If it says MISMATCH, the retrain differs from Phase 6 and the comparison is
void.

**Decoder verification** — five assertions, each of which has caught a real bug:
cache-vs-naive, pruned==full argmax, sequential banning walks the ranking,
`allowed_idx` excludes the global best and returns the best *admissible* word,
and a single admissible word returns `forced` with no model call.

**Hard-mode violations** should be **0.0%** in `consistent` mode by
construction. If not, the filter is broken.

## The adaptive decoder — why it is there

Measured on the training data, the expert's own target is feedback-**inconsistent**
most of the time early:

```
turn 2:  40.6% consistent   <- the expert probes with a word that cannot win
turn 3:  91.8% consistent
turn 4: 100.0% consistent
```

An always-on filter forbids the expert's turn-2 policy in ~59% of games. That is
the known reason hard mode scores worse than free mode. `adaptive` filters only
once the admissible set is <= 50 words, so the model can still probe while
uncertainty is high.

If `adaptive` beats `consistent`, midgame filtering was costing us the probe.
The decomposition block prints exactly that line.

## Reading the result

The number that matters is not the mean. It is this block:

```
solved N  =  forced X + model-chosen Y
-> Z% of wins were decided by the filter alone, not the model
```

At k=1 states the filter alone determines the word 38.9% of the time. Those are
decoder wins. The unfiltered k=1 probe (section 11) stays as the measure of
model capability and is directly comparable to Phase 5 (20.0%) and Phase 6
(27.5%).
