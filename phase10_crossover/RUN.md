# Phase 10 — the format crossover

**Status: prepared, not run.** Everything below is written before any result
exists, deliberately.

---

## 1. The question

Phase 9 measured, on 246 held-out answers with the decoder held fixed:

| variant | mean | vs baseline | solved |
|---|---:|---:|---:|
| `baseline` | **3.7642** | — | 242/246 |
| `raw_history` | 4.2805 | +0.516 (t=7.07) | 232/246 |

and with the decoder switched **off** (probe B, 148 stratified states):

| variant | parse% | legal% | admissible% |
|---|---:|---:|---:|
| `baseline` | 99.3 | 84.5 | **22.3** |
| `raw_history` | 100.0 | **95.9** | **1.4** |

`raw_history` is `baseline` minus the solver-derived constraint block —
`Confirmed letters : p1=A`, `Letters absent : E, L, O` and so on. Without it the
model emits *more* legal words and almost no admissible ones. It keeps the rules
of Wordle and loses the deduction.

The obvious reading is "the harness has been doing the deduction the write-up
credits to the model". **That reading is not yet earned.** The adapter was
trained on `baseline` alone, so every one of those numbers is equally consistent
with "the model recognises the only format it has ever seen". Format lock-in
and an intrinsic property of the format predict exactly the same table.

Phase 9 ran a stock-Qwen `base` arm to break that tie, on the reasoning that
all twelve formats are equally unfamiliar to an untrained model. It could not:
nine runs landed at 6.71–6.88 mean with 91–97% failure and a spread of 0.17. A
model that cannot play Wordle under any prompt cannot rank prompts. The arm cost
84.5% of the session and settled nothing.

Breaking the tie needs a model with real skill under a *second* format. That
means training one.

---

## 2. The design

Train one adapter, `tree_salet_endgame_rawhist`, on the **exact** Phase 6
training set with only the prompt text rewritten into `raw_history` format.
Then evaluate the 2×2.

|  | eval `baseline` | eval `raw_history` |
|---|---|---|
| trained on `baseline` (`tree_salet_endgame`) | **3.7642** measured | **4.2805** measured |
| trained on `raw_history` (`..._rawhist`) | ? | **?** ← the cell that decides |

Two of the four cells are already measured, which is what makes this cheap: one
training run and two new evaluations complete the square.

### What is held identical

Everything except the prompt string:

| | value |
|---|---|
| rows | the same 19,212 (7,067 natural + 12,145 synthetic endgame) |
| completions | untouched — same expert action per row |
| `meta` | untouched — same turn, split, candidate count |
| LoRA | r=16, α=32, dropout=0.05, same 7 target modules |
| optimisation | lr 2e-4, 2 epochs, batch 4×4, cosine, warmup 0.03, fp16 |
| seed | 20260817 — Phase 6's, so the shuffle order matches |
| decoder at eval | adaptive@20, chunk 512, pruning on — the control, never varied |
| eval answers | the same 246, paired |

If any hyperparameter differed, the crossover would be varying two things at
once and would answer nothing.

### The safety argument for the re-render

The SFT rows do not store the game history as data — only as text inside the
already-rendered `baseline` prompt. So the history is parsed back out, and
before a single row is used:

```
v_baseline(turn, parsed_history, max_guesses) == stored_prompt    byte-for-byte
```

If the parsed history reproduces the stored prompt exactly, the parse is
provably lossless and the only difference in the output is the variant. Any
mismatch aborts the run rather than writing a row.

Verified locally 2026-08-23 against all 19,212 rows plus the 853-row val split:
**20,065 exact round-trips, zero parse failures**. The notebook re-checks it,
because the dataset it attaches could differ from the local copy.

Worked example of the transformation:

```
BEFORE (baseline)                      AFTER (raw_history)
History:                               History:
  1. SALET -> BYBBB                      1. SALET -> BYBBB
  2. COBRA -> YBYBY                      2. COBRA -> YBYBY
                                       
Deduced so far:                        Guess 3 of 6 (4 remaining)
  Confirmed letters : (none)           
  Letters present   : A, B, C          Next guess:
  Letters absent    : E, L, O, R, S, T 
  Ruled-out spots   : A not at [1, 4]  
                                       
Guess 3 of 6 (4 remaining)             
                                       
Next guess:                            

completion: ABACK                      completion: ABACK   (unchanged)
```

---

## 3. The pre-registered reading

Written before the run. Phase 9's decision table failed to anticipate its own
outcome shape — large spread, all negative — which is exactly how a fork stays
open, so this one enumerates the outcomes including the awkward ones.

**A — Lock-in.** Each adapter is best on its own format, by roughly similar
margins (`rawhist` on `raw_history` ≈ 3.8–3.9, and worse on `baseline`).
→ Format is a **robustness** problem, not a quality ranking. Phase 9's ordering
of the twelve variants says nothing about which prompt is intrinsically better,
and the honest write-up is "this model is brittle to prompt format". The
constraint block stays because it is what the model was trained on, not because
it is doing hidden work. *Follow-up: mixed-format SFT — one adapter trained on
all non-leaky variants — with the pre-registered expectation that hard-mode
violations fall.*

**B — Learnable.** The `rawhist` adapter recovers toward 3.76 on `raw_history`.
→ The deduction the solver hands the model **can be learned**; the constraint
block was a crutch for the Phase 7 adapter, not a requirement. The "the harness
is the model" worry mostly dissolves, and the interesting question becomes why
the block was ever needed. *Follow-up: is a block-free model actually better —
does it generalise where the scaffolded one does not?*

**C — Intrinsic.** The `rawhist` adapter stays near 4.28 on `raw_history`
despite being trained on it.
→ The deduction genuinely exceeds what a 0.5B model learns at this data scale.
The constraint block is **honest scaffolding** and stays, and Phase 9's finding
2 is confirmed rather than merely consistent with the data. This is the cleanest
outcome for the write-up: the harness does part of the work, it has to, and the
paper says so with a controlled experiment behind it.

**D — Both worse than expected.** The `rawhist` adapter underperforms on *both*
formats.
→ Training is the suspect, not the format. Check the step count against Phase
6's 1,202 and the loss curve against 5.886 → 0.741 before drawing any
conclusion about prompts. The notebook prints both comparisons for this reason.

**Not a valid reading in any branch:** treating a single 246-game cell as
significant on its own. Cells are compared *paired* on identical answers, and
failure counts are read on 246 only — the Phase 9 100-game subset was actively
misleading about failures (see `phase9_harness/TECHNICAL.md` §8c).

---

## 4. How to run it

Two stages, because the adapter has to become a Dataset in between.

### Stage 1 — train (~90 min on a T4)

```bash
.conda/python.exe phase10_crossover/make_crossover_notebook.py   # regenerate
.conda/python.exe tools/kaggle_run.py push phase10               # detach
.conda/python.exe tools/kaggle_run.py runs                       # check later
```

**Gate:** cell 4 must print `round-trip: ok=19,212  parse-fail=0  mismatch=0`,
and cell 6 must land on **1,202 steps** with loss falling from ~5.89. A
different step count on the same rows and batch size means the data is not what
it should be — stop rather than interpret.

The notebook writes `tree_salet_endgame_rawhist.zip` to `/kaggle/working` and
runs a 20-generation liveness smoke test. The smoke test is not a result; it
only catches a dead adapter before a whole evaluation session is spent on it.

### Stage 2 — publish the adapter, then run the 2×2 (~15 min)

Download the adapter zip, add it to `arnavyrr/wordle-adapters-v2` as a new
version **keeping the folder name** `tree_salet_endgame_rawhist/` — the harness
matches adapters by directory basename and refuses to substitute a different
one. Then in `phase9_harness/make_harness_notebook.py`:

```python
ARMS            = ["sft", "sft_rawhist"]
VARIANTS_TO_RUN = ["baseline", "raw_history"]
N_GAMES         = 246
```

and push `phase9`. The `sft` + `baseline` gate — which must reproduce 3.7642 —
is the crossover's control: if the known cell of the 2×2 does not come back
where it was measured, nothing else in the square is interpretable.

The harness already holds the fixed decoder, the paired answer set, the
per-decision `|admissible|` logging and the copy-rate check. **Nothing about
the evaluation is re-implemented here.** Forking the measurement path is what
voided Phase 8 v3.

---

## 5. Cost

| stage | cost |
|---|---|
| training | ~90 min T4 (Phase 6 took 76 min on the same rows) |
| 2×2 evaluation | ~15 min — two new cells at ~6 min each, plus the gate |
| already measured | two of the four cells |

Against a ~30 h/week GPU budget, roughly two hours to close the confound that
sits under every prompt result in the project.

---

## 6. Files

| file | what |
|---|---|
| `make_crossover_notebook.py` | generates the training notebook |
| `wordle_phase10_crossover_kaggle.ipynb` | generated — do not edit directly |
| `rerender_rows.py` | the re-render with the round-trip proof, runnable locally |
| `RUN.md` | this file |

Local dry run, no GPU needed:

```bash
.conda/python.exe phase10_crossover/rerender_rows.py --variant raw_history --dry-run
```
