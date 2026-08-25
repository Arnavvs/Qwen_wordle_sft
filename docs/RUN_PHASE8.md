# Running Phase 8 on Kaggle — headroom, then DPO

Two experiments in one session: measure where the remaining 0.32 guesses live,
then train the first DPO adapter. **~2.5 h** with the Phase 7 adapter attached,
~3.5 h without.

---

## Step 1 — Rebuild and upload the SFT package

The package needs `sft_package/data/dpo_midgame/train.jsonl`, which did not
exist before Phase 8. Locally:

```bash
python phase8_dpo/build_dpo_dataset.py
```

```bash
python phase8_dpo/verify_dpo_dataset.py
```

```bash
python tools/prepare_kaggle_dataset.py --dest uploads/kaggle_upload
```

Then upload `uploads/kaggle_upload` as a **new** dataset, e.g.
`wordle-sft-package-v3`. The notebook fails loudly and names the missing file if
you attach v2 by mistake.

## Step 2 — Attach inputs

- `wordle-sft-package-v3`
- the **Phase 7 adapter** dataset, if you saved it. With it, the SFT stage is
  skipped and you save ~70 minutes. Without it, the notebook retrains the same
  configuration from scratch.

## Step 3 — GPU and config

Accelerator → **GPU T4 x2**. Then in section 1:

```python
PREV_RUN_DIR = "/kaggle/input/<...>/wordle_phase7"    # or leave None
```

Everything else is already set: `beta=0.1`, `lr=5e-6`, 1 epoch, 15k pairs,
adaptive decoder at threshold 20. **Do not sweep these on the first run.**

## Step 4 — Run All

| Stage | Time |
|---|---|
| SFT (skipped if adapter attached) | ~70 min |
| counterfactual headroom, 5 hybrid runs | ~35 min |
| DPO training, 15k pairs | ~50 min |
| SFT + DPO evaluation | ~15 min |

## Step 5 — ⚠ Save the DPO adapter

Section 6 stops and tells you. Output → New Dataset → save
`/kaggle/working/wordle_phase8`.

## Step 6 — Download `wordle_phase8_results.zip`

Small, no weights.

---

## Reading the output

### The headroom table decides whether to continue

```
expert acts in         mean   solved   recovered   % of gap
baseline             3.7642      242           -          -
2-10                      ?        ?           ?          ?
11-100                    ?        ?           ?          ?
100+                      ?        ?           ?          ?
combined                  ?        ?           ?          ?
```

`recovered` is what perfect action selection **in that bucket alone** would buy.

- **`combined` recovers most of the 0.32** → action selection is the whole
  problem, DPO has room, and the per-bucket rows say where to aim the next
  dataset.
- **`combined` recovers little** → the gap is structural, no preference
  training will close it, and the honest move is to stop and write up. That is a
  real possible outcome and not a failed session.

### DPO training

```
loss   0.69 -> ?      margin +0.00 -> ?      acc 0.50 -> ?
```

Margin rising and accuracy above 0.5 means the preference was learned. **It does
not mean gameplay improved** — those are different claims and the table below is
the one that settles it.

### SFT vs DPO is compared paired

The unpaired SE on 246 games is ~0.064 guesses, so an unpaired test cannot see
anything smaller than ~0.13 — and most plausible DPO effects are below that.
The notebook therefore reports:

```
games changed: N  (DPO better X, worse Y)
paired t on changed games
```

A DPO gain of 0.05 with 30 games changed and 25/5 in its favour is real. The
same gain with 6 games changed and 4/2 is noise. Read the split, not the mean.

### Model contribution

```
model contribution vs filter-only 5.1762:
  SFT -1.41    DPO -?
```

This is the number that says whether the **model** improved rather than the
decoder. Phase 7 showed the decoder does most of the work at high thresholds, so
a mean that improves while contribution falls would be a warning, not a win.

---

## What is deliberately not in this run

- **No hyperparameter sweep.** One beta, one learning rate, one epoch.
- **No GRPO.** DPO first; if it moves nothing, GRPO on the same data will not
  either.
- **No new decoder work.** Threshold fixed at 20 so the comparison is clean.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `FileNotFoundError` naming `dpo_midgame` | you attached v2; attach v3 |
| no SFT adapter found | leave `PREV_RUN_DIR = None`, it retrains |
| CUDA OOM during DPO | `DPO_BS = 2`, `DPO_ACCUM = 16` |
| margin goes negative and stays | `beta` too low or `lr` too high; stop and report — do not sweep blindly |
| session runs short | set `RUN_HEADROOM = False` on a re-run; results reload and only the missing stages run |
