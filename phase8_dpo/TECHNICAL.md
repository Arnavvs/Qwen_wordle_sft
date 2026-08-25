# `wordle_phase8_dpo_kaggle.ipynb` — technical reference

Counterfactual headroom analysis, then DPO on top of the Phase 7 SFT policy.

- **Generator:** `make_dpo_notebook.py` (the notebook is a build artifact — edit the generator, never the `.ipynb`)
- **Operator guides:** `../docs/RUN_PHASE8.md`, `../docs/RUN_PHASE8_V3.md`
- **Structure:** 17 cells, 8 code
- **Status:** executed. v1 harmful, v3 a clean null. See §8.

---

## 1. What it does

Two experiments in one session:

1. **Counterfactual headroom** — hand each `|admissible|` bucket to the
   classical expert in turn and measure what perfect play *there* would be
   worth. This decides whether DPO is worth continuing past the first run.
2. **DPO** — train a preference adapter on top of a fresh SFT reproduction and
   evaluate with the Phase 7b adaptive decoder at threshold 20.

Baseline entering the notebook:

| | mean | solved | fail |
|---|---:|---:|---:|
| SFT + adaptive decoder @20 | **3.7642** | 242/246 | 1.6% |
| filter only, no model | 5.1762 | — | 22.8% |
| classical entropy solver | 3.4431 | 246/246 | 0% |

---

## 2. Cell map

| # | kind | contents |
|---|---|---|
| 0 | md | motivation, baseline table |
| 2 | code | deps, `fix_torchao_peft_conflict()`, all configuration |
| 4 | code | dataset discovery, solver bundle, `base_model()`, `find_adapter()` |
| 6 | code | SFT stage — reproduce Phase 6/7 config, or reuse the adapter |
| 8 | code | `core/constrained_decode.py` inlined, `GameState`, `play()`, `summarize()` |
| 10 | code | counterfactual headroom |
| 12 | code | preflight gate, `load_sft()`, resumable `STATE` |
| 14 | code | DPO training loop (hand-written), held-out pair eval |
| 16 | code | `load_dpo()`, final evaluation, save, publish |

`core/constrained_decode.py` is inlined at *generation* time so the notebook is
self-contained on Kaggle. Editing it requires regenerating:

```bash
python phase8_dpo/make_dpo_notebook.py
```

---

## 3. Configuration

```python
RUN_HEADROOM = False      # answered in the first run
RUN_DPO      = True
RUN_EVAL     = True

MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER  = "tree_salet_endgame"
DPO_ADAPTER  = "tree_salet_dpo"
DPO_DATASET  = "dpo_v3"            # "dpo_v3" | "dpo_midgame" (v1)

# SFT, only if the adapter is missing — byte-identical to Phase 6/7
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
SFT_LR, SFT_EPOCHS               = 2e-4, 2
PER_DEVICE_BS, GRAD_ACCUM, MAX_SEQ_LEN = 4, 4, 640

# DPO
DPO_BETA      = 0.1     # standard; not swept, per the brief
DPO_LR        = 5e-6    # ~40x below SFT — DPO drifts fast at SFT rates
DPO_EPOCHS    = 1
DPO_BS, DPO_ACCUM = 4, 8          # effective batch 32
DPO_MAX_PAIRS = 15000
DPO_LORA_R, DPO_LORA_ALPHA = 16, 32
EVAL_PAIRS_EVERY = 50             # optimizer steps between held-out evals

ADAPTIVE_THRESHOLD = 20           # the Phase 7b optimum; fixed, not swept
CONSTRAINED_CHUNK, CONSTRAINED_PRUNE = 512, True
ATTN_IMPL = "sdpa"
USE_DECISION_CACHE = True
PREFLIGHT, PREFLIGHT_GAMES = True, 40
HEADROOM_BUCKETS = {"2-10": (2,10), "11-100": (11,100), "100+": (101, 10**9)}
SEED = 20260817
```

---

## 4. DPO, implemented by hand

TRL's `DPOTrainer` pins tightly to `transformers`, and this project has been
bitten twice by that coupling (Kaggle ships transformers 5.0). The loss is six
lines; writing it directly removes a dependency whose version must match exactly
and which would fail *at import*, after the training data is already loaded.

```
loss = -logsigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )
```

where each term is a sequence log-probability `Σ log P(token | prompt, prefix)`.

### Reference model

The SFT adapter is **merged into the base weights**, then a *fresh* LoRA is
added for DPO. The reference is that same merged model with the new adapter
disabled — so the reference is exactly the SFT policy, which is what DPO
requires.

Continuing Phase 6's adapter directly would make "better preferences" and "more
training" inseparable, so it is not done.

---

## 5. Evaluation path

Identical to Phase 7 so the comparison is like-for-like: `play()` over the same
246 held-out answers, adaptive decoder at threshold 20, chunk 512, pruning on.

`summarize()` returns `per_game`, which is what makes SFT-vs-DPO a **paired**
comparison rather than two independent means. On 246 games the unpaired SE is
~0.064 — large enough to hide the effects at stake. Paired resolves far finer.

### Decision cache

Keyed on the prompt string, inside `play()`. This is **exact, not an
approximation**: the model is deterministic and the decoder is greedy, so the
same prompt always yields the same word. Turn 1 is one distinct prompt across
all 246 games; turn 2 is a few dozen.

### Bucket accounting

`g.n_allowed` records the **admissible** count, not `scorer.n_allowed`. When the
filter is off the scorer reports the whole vocabulary, which would hide the
decision regime being bucketed by.

### Hybrid-policy caveat

In `expert_buckets`, substituting the expert changes the feedback and every
later state. The headroom numbers are therefore a **hybrid-policy evaluation**,
not a replay of logged games. Stated in the docstring because it is easy to
misread as counterfactual replay.

---

## 6. Guards, and the bugs that motivated them

Each of these exists because the corresponding failure actually occurred.

### `find_adapter` — exact match, no fallback

An earlier version fell back to *any* adapter when the requested name was
missing. It silently loaded a DPO adapter as the SFT base: the run looked
entirely normal while measuring the wrong model, and produced a baseline of
4.8862 instead of 3.7642. It now prints what it found and refuses to substitute.

### `base_model()` — fp16 and sdpa, asserted

`torch_dtype` is deprecated in transformers 5.x, and a silent fp32 load costs
~8× on a T4. `eager` attention costs another 5–10×. Neither failure is visible
except as a slow run, so both are asserted rather than hoped for.

### Preflight gate (cell 12)

Before spending an hour measuring a model, verify it *is* the Phase 7 model:
opener must be `SALET`, and the 40-game mean within 0.50 of 3.7642.

### Decoder self-test

`LegalWordScorer.self_test()` uses dtype-aware tolerance (0.02 fp32 / 0.40 fp16)
**plus** a ranking-correlation check. A tolerance calibrated on fp32 aborted a
valid fp16 run at 0.0369. Fault injection on real weights confirmed genuine
corruption gives correlation 0.44–0.72 against a 0.999 threshold, so the added
check has real discriminating power.

### Mask applied to scores, not only bounds

In `argmax`, `ban_mask` is applied to the bound *and* to the scores. Masking only
the bound is a bug that shipped once: the excluded word still received a genuine
score and could win from the tail of a live chunk.

### Containers initialised before loops

`EVAL` / `GAMES` / `PRIMARY` were once assigned at the *end* of an interruptible
cell, producing `NameError` on any `KeyboardInterrupt` and losing the run.

---

## 7. The v1 → v3 dataset correction

`dpo_midgame` (v1) had four defects, any one of which is sufficient to explain a
regression:

| defect | consequence |
|---|---|
| no validation split | the reported 0.604 → 0.836 was **training** accuracy, reported as generalisation |
| duplicate ids, multiple rows per state | effective dataset far smaller than 15k |
| consistency shortcut | `rejected` often distinguishable without solving anything |
| arbitrary `−0.5·is_candidate` term | 78.9% of "competitive" pairs separated by nothing but that constant |

The last one invalidated my own recommendation to "use competitive pairs only" —
verified against the raw file.

`dpo_v3` (built by a separate run, independently audited here via
`independent_check_v3.py`) has a real validation split, one row per state, no
duplicate ids, and no shortcut.

### Action-space bug

The tree expert and entropy solver pick from the full pool, so `chosen` /
`rejected` could be **inadmissible** under the restricted decoder — training the
policy to prefer words it can never play. Fixed across two passes: 19% → 77 → 0
violations.

The first audit reported "206 violations" from a *sample*, disguising a 19%
rate. The check is now exhaustive with a prefix cache.

---

## 8. Results

| run | mean | solved | vs SFT | verdict |
|---|---:|---:|---:|---|
| SFT (reproduced) | 3.7642 | 242/246 | — | reproduces Phase 7 exactly — run is valid |
| DPO v1 (`dpo_midgame`) | worse | — | t = −3.21 | **harmful** |
| DPO v3 | 3.7886 | 241/246 | +0.0244, t = −0.69 | **clean null** |

v3 detail: 53 games changed — 22 better, 31 worse. Not significant.

**Interpretation.** The clean dataset removed v1's harm but produced no benefit.
DPO at this scale, on this task, with this preference signal, does nothing. The
headroom analysis had localised 74.7% of remaining loss to states with 2–10
admissible words; the preference data targeted exactly that and still moved
nothing.

### Known gaps in the v3 run

- **The held-out curve was never recorded.** No `val_acc` rows, despite 800
  validation pairs loading. The one number that would distinguish "DPO learned
  nothing" from "DPO learned something that didn't transfer" is missing. This is
  a defect in the training loop's logging, not in the data.
- **Weights absent from the results zip.** `RESULTS_ZIP` archives `RESULTS_ROOT`
  only, by design; adapters live under `WORK_DIR`. Use
  `../kaggle_cells/CELL_save_everything.py`, which stages both.

### Unresolved

The evaluation ran at ~21 s per unfiltered decision against Phase 7's ~0.6 s.
dtype and device were ruled out by direct check (fp16, cuda:0) — my fp32
hypothesis was wrong. Phase 9 later established that off-distribution prompts
defeat the decoder's branch-and-bound pruning and cost ~13× (see
`../phase9_harness/TECHNICAL.md` §5). That is a strong candidate for part of
this, but 21 s/decision is ~10× beyond what a full 26-chunk scan accounts for.
**Not closed.**

---

## 9. Files

```
phase8_dpo/
  make_dpo_notebook.py               generator — edit this
  wordle_phase8_dpo_kaggle.ipynb     build artifact — do not edit
  TECHNICAL.md                       this file
  build_dpo_dataset.py               v1 generator (superseded)
  verify_dpo_dataset.py              v1 audit
  independent_check_v3.py            second-opinion audit of v3
  _paths.py                          puts core/ on sys.path, chdir to root

phase8_dpo_v3/                       v3 generator, verifier, AUDIT.md
core/constrained_decode.py           the decoder (inlined at build)
docs/RUN_PHASE8.md, RUN_PHASE8_V3.md operator guides
kaggle_cells/CELL_save_everything.py stages adapters AND results
```

---

## 10. If this is picked up again

DPO is a measured null, so a third dataset iteration needs a reason beyond "the
last one was flawed" — v3 was not flawed. The two things worth doing first:

1. **Fix the `val_acc` logging** and re-run. Without the held-out curve there is
   no way to tell whether the preference signal was learnable at all, and that
   determines whether GRPO is worth attempting on the same signal.
2. **Read Phase 9 before training anything else.** Prompt format moved the mean
   0.83–2.31 guesses; every training intervention here moved it ±0.03. The
   evidence points away from more preference optimisation and toward the
   harness.
