# Running Phase 9 — the prompt/harness sweep

**Notebook:** `phase9_harness/wordle_phase9_harness_kaggle.ipynb`
**Trains nothing.** Measurement only. Safe to interrupt at any point.

---

## What it answers

Every phase so far varied the model. This varies the *text*, with the model and
the decoder held fixed, because the only intervention that ever moved the number
was scaffolding:

| | mean |
|---|---:|
| decoder alone, no model | 5.1762 |
| model alone, no decoder | ~5.29 |
| both | **3.7642** |
| four training interventions after that | ±0.03 |

So: how much of the remaining 3.76 is the prompt?

---

## Before you upload

Nothing to rebuild — the notebook inlines `core/constrained_decode.py` and
`core/prompt_variants.py` at generation time. If you edit either, regenerate:

```bash
python phase9_harness/make_harness_notebook.py
```

## On Kaggle

1. New notebook → **GPU T4 x2** (one GPU is used; T4 is enough).
2. **Add Data** → your existing `sft_package` dataset (the same one Phase 7/8
   used — it needs `eval/val_answers.jsonl`, `code/`, `artifacts/`).
3. **Add Data** → the dataset holding the Phase 7 SFT adapter
   `tree_salet_endgame`.
4. Upload the `.ipynb`, or paste the cells.
5. Cell 1 — leave `DATASET_DIR` and `PREV_RUN_DIR` as `None` unless the search
   fails, in which case set them to the printed paths.
6. Run all.

If you only have the base model and no adapter, set `ARMS = ["base"]` in cell 1.
The notebook will refuse to run the `sft` arm rather than substitute a different
adapter — that substitution already cost this project one wrong result.

---

## What to watch, in order

**Cell 4 (the twelve rendered).** Read this output. All twelve prompts are
printed on the same state. A variant that renders wrongly still produces a
plausible number, and you will not catch it later.

**Cell 7 (the gate).** `sft` + `baseline` on 40 games must open with `SALET` and
land near 3.7642. On 40 games some drift is normal; the gate allows 0.60. A
wrong opener means the wrong adapter — stop.

**Cell 8 (the sweep).** ~24 runs, one line each, saved after every one.

---

## Cost and knobs

At the Phase 7 decoder speed (~0.6 s/decision) the full sweep is roughly
**60–90 minutes**. If your session is slower — the unexplained ~21 s/decision
regression from the Phase 8 run has never been diagnosed — cut it down:

| knob | default | to go faster |
|---|---|---|
| `N_GAMES` | 100 | 60 |
| `ARMS` | `["sft","base"]` | `["sft"]` |
| `VARIANTS_TO_RUN` | `None` (all 12) | `["baseline","raw_history","minimal","emoji"]` |
| `RUN_FORMAT` | `True` | keep — it is ~2 min and answers a different question |

**It resumes.** State lives in `results_phase9/harness_results.json` and every
`(arm, variant)` is written the moment it completes. Re-running the notebook
skips what is already there. Interrupting costs at most one variant.

To force a redo of one variant, delete its key (e.g. `"sft|emoji"`) from that
JSON and re-run cell 8.

---

## Reading the output

Cell 10 prints one table per arm, sorted best-first, plus a `VERDICT` block.

- **`vs base`** is a *paired* difference — same answers, same decoder, so the
  only difference is the text. Paired is what makes 0.05 readable; unpaired SE
  on 100 games is ~0.10 and would hide everything real here.
- **`fair spread`** excludes `with_count`.
- **`t`** is a paired t. |t| > 2 is worth believing on 100 games; anything
  smaller is a coin flip, regardless of how good the mean looks.

### The one row that matters

`raw_history`. Every phase has fed the model `Confirmed letters : p1=A`,
computed by the classical solver. If `raw_history` matches `baseline`, that
block was decoration. If it collapses, the harness has been doing the deduction
and the write-up has to say so.

### `with_count` is not a result

It shows the surviving-candidate count, which a player cannot see. It is flagged
`*LEAKY*` in the table and excluded from the spread. It exists to bound the
question "how much is that hint worth?" — never quote it, and never train on it.

---

## What each outcome means for the next step

| observed | conclusion |
|---|---|
| fair spread < 0.05 | Prompt format is not a lever. Stop optimising the harness; go to GRPO or accept ~3.76. |
| fair spread 0.05–0.15 | Real but small — about one training phase's worth, for an afternoon's work. |
| fair spread > 0.15 | Format matters more than four training interventions did. Re-run SFT on the winning format; the headline finding changes. |
| base arm spread ≫ sft arm spread | The SFT adapter has *specialised* to `baseline` formatting. That is a fine result, but it means the reported 3.7642 is format-bound. |
| format probe `admissible %` moves but game mean doesn't | The decoder was already repairing what the prompt fixed. Good to know, no action. |

---

## Outputs

- `results_phase9/harness_results.json` — everything, including per-game scores
  (needed for any later paired re-analysis)
- `results_phase9/summary.json` — the same without per-game arrays
- `wordle_phase9_results.zip` — download link in the last cell
- Published as `wordle-phase9-harness` if the Kaggle secrets are set
