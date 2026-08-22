# Phase 8 v3 — the corrected DPO run

**Read this first.** The previous attempt produced numbers that could not be
interpreted, for a reason worth understanding before repeating it.

## What went wrong last time

The SFT baseline came back at **4.8862 / 210 solved**. It should have
reproduced Phase 7 exactly: **3.7642 / 242**. A 1.12-guess discrepancy on a
run using the same adapter, decoder and threshold means the run was measuring
something else, and the DPO row from it is equally void.

Prime suspect, now fixed: `find_adapter` fell back to **any** adapter when it
could not find one named `tree_salet_endgame`. If the attached dataset held
`tree_salet_dpo` from the earlier run, that loaded as the SFT base — and
everything downstream trained and evaluated against the wrong model while
looking entirely normal.

A second, separate problem: unfiltered decisions ran at ~21 s against Phase 7's
~0.6 s. Cause still unknown; the preflight now measures it rather than letting
it eat a session.

## What is different in this notebook

| | |
|---|---|
| **Strict adapter lookup** | no fallback. A missing name prints what it *did* find and refuses to substitute |
| **Preflight (section 5b)** | verifies by behaviour, not by path — opener must be SALET, and a 40-game sample must reproduce 3.7642 within 0.50. Takes ~1 min and **asserts**, so a wrong setup stops there |
| **fp16 + sdpa asserted** | `torch_dtype` is deprecated in transformers 5.x; a silent fp32 load costs ~8x, `eager` attention another 5–10x. Both are now checked, not hoped for |
| **Decision cache** | same prompt ⇒ same decision. Exact, since the model is deterministic and the decoder greedy. Turn 1 collapses from 246 decisions to 1 |
| **Held-out pair accuracy** | v3 has a real validation split; the earlier 0.604 → 0.836 was *training* accuracy and meant nothing |

## Steps

1. Upload `uploads/kaggle_upload` as **`wordle-sft-package-v4`** (contains
   `data/dpo_v3/` with train 6,000 + validation 800)
2. Attach it **plus a dataset containing `tree_salet_endgame`** — the Phase 7
   SFT adapter, not a DPO one. This is the thing that broke last time.
3. GPU T4 x2. Import the notebook. Set `PREV_RUN_DIR`.
4. **Run `CELL_autosave_daemon.py` before the eval**, not after.
5. Run All.

## The gate

Section 5b either prints

```
preflight passed - safe to spend the session
```

or it raises. **If it raises, fix the setup — do not continue.** Every number
after an unverified baseline is uninterpretable, which is the whole lesson from
last time.

Watch the reported per-decision time too. Phase 7 was ~0.6 s unfiltered. If
preflight reports >3 s it warns; run `kaggle_cells/CELL_debug_slow_eval.py`
before committing to the full evaluation.

## Reading the result

**Held-out pair accuracy, not training accuracy.** If it peaks early while the
margin keeps climbing, that is the v1 failure repeating and the notebook says so.

**The paired split, not the mean:**

```
games changed: N (better X, worse Y)   paired t
```

The v1 run was 70 worse against 38 better, t = −3.21. Unpaired SE on 246 games
is 0.064, so anything under ~0.13 is invisible without pairing.

**Model contribution vs filter-only (5.1762).** If the mean improves while
contribution falls, the decoder gained and the model drifted.

## An honest prior

v3's own README says it is built so a corrected retry is a *fair* experiment,
"not because the result is expected to be positive". Total headroom is 0.32
guesses, three quarters of it in states with 2–10 admissible words. A null
result is informative and is a perfectly good outcome.
