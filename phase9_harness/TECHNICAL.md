# `wordle_phase9_harness_kaggle.ipynb` — technical reference

Prompt/harness sweep. Measurement only; trains nothing.

- **Generator:** `make_harness_notebook.py` (the notebook is a build artifact — edit the generator, never the `.ipynb`)
- **Operator guide:** `../docs/RUN_PHASE9.md`
- **Structure:** 23 cells, 11 code
- **First executed:** 2026-08-22, 21 of 24 runs completed

---

## 1. What it measures and why

Every prior phase varied the model. This varies the *prompt text*, with model and
decoder held fixed.

The justification is the project's own record. Between Phase 6 and Phase 8, four
training interventions moved the mean by less than noise:

| | mean | verdict |
|---|---:|---|
| Phase 6 SFT + decoder | 3.7886 | — |
| Phase 7 (endgame data) | 3.7642 | marginal |
| Phase 7b (threshold sweep) | 3.7642 | flat frontier |
| Phase 8 DPO v1 | worse, t = −3.21 | harmful (bad dataset) |
| Phase 8 DPO v3 | 3.7886, t = −0.69 | clean null |

Against that, the constrained decoder alone was worth ~1.5 guesses. Scaffolding
dominates training at this model size, and exactly one piece of scaffolding had
never been varied.

---

## 2. Cell map

| # | kind | contents |
|---|---|---|
| 0 | md | motivation, variant table, pre-registered read of the outcome |
| 2 | code | deps, `fix_torchao_peft_conflict()`, all configuration |
| 4 | code | dataset discovery, solver bundle, `base_model()`, `find_adapter()` |
| 6 | code | `core/constrained_decode.py`, inlined verbatim |
| 8 | code | `core/prompt_variants.py`, inlined; renders all 12; exemplar guard |
| 10 | code | `GameState`, `play()`, `summarize()` |
| 12 | code | resumable `STATE`, `save()`, `get_model()` |
| 14 | code | sanity gate — SFT+baseline must reproduce Phase 7 |
| 16 | code | **probe A** — the sweep |
| 18 | code | **probe B** — format compliance, no decoder |
| 20 | code | paired statistics, tables, verdict |
| 22 | code | zip + optional Kaggle Dataset publish |

Two modules are inlined at *generation* time, not imported at runtime, so the
notebook is self-contained on Kaggle. Editing either requires regenerating:

```bash
python phase9_harness/make_harness_notebook.py
```

The inliner strips `prompt_variants.py`'s `sys.path` bootstrap (no `__file__` in
a notebook) and asserts the strip succeeded.

---

## 3. The variant contract

Every variant is `f(turn, history, max_guesses) -> str`, ending in `Next guess:`
so the decoder scores continuations identically across all of them. `history` is
`[(guess_lower, pattern)]` with patterns over `GYB`.

`render(name, turn, history, max_guesses, n_candidates=None)` dispatches;
`n_candidates` is consumed only by `with_count`.

| variant | isolates | tokens |
|---|---|---:|
| `baseline` | what every earlier phase used | 169 |
| `raw_history` | **no derived constraints — does the model deduce?** | 105 |
| `constraints_only` | deductions without the moves | 145 |
| `minimal` | no instructions at all | 15 |
| `with_count` | 🚩 **leaky** — surviving-candidate count | 177 |
| `emoji` | 🟩🟨⬛ vs `GYB` — pure notation | 110 |
| `verbose` | feedback spelled out per letter | 181 |
| `reversed` | most recent guess first | 167 |
| `keyboard` | letter-status board | 179 |
| `few_shot` | two worked examples first | 336 |
| `guided_prose` | constraints as a sentence | 158 |
| `hard_mode_hint` | states the rule the decoder enforces | 180 |

Token counts are measured with the Qwen2.5 tokenizer on turn 3 after
`SALET->BYBBB, CRONY->BYBBG`. Note `emoji` (110) is *shorter* than
`baseline` (169) yet evaluates slower — see §5, cost is driven by model
confidence, not prompt length.

### Leakage

`with_count` shows the surviving-candidate count — a solver-side quantity a
player cannot see. It carries `leaky=True`, is excluded from the reported
spread, prints with a `*LEAKY*` flag, and **must never be quoted as a result or
used to select a training format**. Every other variant honours the project's
standing rule: no answer, no answer list, no candidate set, no candidate count.

### Few-shot exemplars

Computed by `_build_shots()` from `_SHOT_GAMES` using the same
`derive_constraints` / `feedback_code` the live prompts use, so an exemplar
cannot disagree with the game it illustrates. Cell 8 additionally asserts each
exemplar word is legal and consistent with its own feedback.

The first draft was hand-written and listed `B` as an absent letter in a game
where `B` appeared in neither guess — a worked example teaching a false
deduction. See §7 for the defect this *didn't* catch.

---

## 4. Experimental design

**The decoder is the control, not a variable.** Adaptive threshold 20, chunk
512, pruning on, for every run. If it moved, nothing would be attributable to
the prompt.

**Shared answers.** One fixed subset — `random.Random(20260822).sample(ALL_VAL, 100)`,
then sorted — is used by every variant and every arm. This makes
variant-vs-baseline a **paired** comparison, which is what makes the design
work: unpaired SE on 100 games is ~0.10 and would hide every effect of interest.
Paired resolves ~0.05. Cell 20 asserts the answer order is identical before
pairing.

**Two arms.** `sft` is the Phase 7 adapter, trained on `baseline` only — so the
other eleven variants are off-distribution for it. `base` is stock
Qwen2.5-0.5B-Instruct, for which all twelve are *equally* off-distribution;
it was included specifically to separate "this prompt is better" from "this
prompt is what the adapter memorised." See §7 — it failed to do that.

**Two probes.** Probe A plays games under the decoder. Probe B runs the same
variants with **no decoder**, raw greedy generation on fixed mid-game states,
reporting parse / legal / admissible rates. A prompt can improve what the model
emits without changing the game mean, because the decoder was already repairing
it. Those are different findings and are measured separately.

**Sanity gate (cell 14).** `sft` + `baseline` on 40 games must open `SALET` and
land within 0.60 of 3.7642. Measured 2026-08-22: `SALET`, 3.575 — pass. The gate
warns rather than aborts, since a 40-game subset legitimately differs from the
246-game figure. It also records `secs_per_decision` so cell 16 can print a real
ETA.

**`find_adapter` is exact-match with no fallback.** An earlier version fell back
to any adapter it found and silently measured the wrong model. It now prints
what it found and refuses.

---

## 5. Cost model — why runtime varies 13×

This is the most useful thing the run produced, and it is not obvious.

`LegalWordScorer.argmax` is exact branch-and-bound. Words are sorted by an upper
bound — the first-token logprob — and the scan **breaks early** once the
incumbent beats the next chunk's bound (`constrained_decode.py:260`):

```python
if best_s >= bound[rows[0]].item():
    break                # no unvisited word can beat the incumbent
```

Cost therefore depends on **how peaked the model's first-token distribution is**,
not on prompt length:

- **Peaked** (model confident): winner found in chunk 1, loop breaks → 1–2 chunks scored.
- **Flat** (model uncertain): nothing prunes → all 12,972 words scored → ⌈12972/512⌉ = **26 chunks**.

Predicted worst/best ratio 26 ÷ 2 = **13×**. Measured 1.95 ÷ 0.148 = **13.2×**.
The arithmetic matches, so this is the mechanism rather than a hypothesis.

Two regimes result:

| regime | s/decision | runs |
|---|---:|---|
| in-distribution | 0.15–0.28 | 10 of 12 `sft` runs |
| off-distribution | 1.3–2.4 | all 9 `base` runs, plus `sft\|minimal` |

The `base` arm is penalised twice: 13× slower per decision **and** 5.82
decisions/game instead of 3.72, because it fails 91–97% of games and so plays
all six turns. Net: **84.5% of total wall time** (167.6 min of 198.3) for the
nine `base` runs.

**Practical consequence:** an off-distribution prompt is intrinsically ~13×
more expensive to evaluate under this decoder. Budget by expected model
confidence, not by prompt length. `minimal` is the *shortest* prompt and among
the slowest.

This mechanism is also a strong candidate for part of the undiagnosed Phase 8
slowdown, but does not fully explain it: Phase 8 saw ~21 s/decision, roughly 10×
beyond what a full scan alone accounts for. Not closed.

---

## 6. State and resumability

`results_phase9/harness_results.json`, keyed `"{arm}|{variant}"`, written after
every completed run. Re-running skips finished keys. Interrupting costs at most
one run. `KeyboardInterrupt` inside the sweep saves before re-raising.

To force a redo, delete that key and re-run cell 16.

`get_model()` holds one arm at a time and frees the previous — the twelve
variants share a load, since reloading per variant would cost more than the
sweep.

`STATE["meta"]` records `n_games`, both seeds, threshold, model name, adapter
name, arms, variant list, gate result, and `secs_per_decision`.

---

## 7. Known defects

### `few_shot` is invalid as written — exemplar copying

The variant opened **`BOOBY`** on every game. The second exemplar ends
`Next guess: BOBBY`; the model copies it rather than generalising from it. The
variant therefore measures verbatim exemplar copying, not few-shot learning, and
its 6.06 mean is an artifact.

The cell-8 guard checks exemplars are *correct*. It does not check they are
*non-copyable*. Fix: exemplars that don't terminate in a guess the model can
lift, or a neutral opener the copy would be harmless.

### The `base` arm is ceiling-bound and answers nothing

All nine runs land at 6.71–6.88 with 91–97% failure and a spread of 0.17. Base
Qwen cannot play Wordle under any prompt, so the arm cannot discriminate between
prompts. Its intended job — separating format-effect from
distribution-shift-effect — is **not accomplished**, and every §8 result carries
that confound.

A discriminating control needs a model with non-trivial performance under
multiple formats. Options: a larger base model, or an SFT checkpoint trained on
mixed formats.

### Probe B never ran

Cell 18 sits after the sweep, and the sweep did not finish. `format` is empty.
It should run before the expensive cell, or in its own session — it is ~2 min
and answers a question probe A cannot.

### Three runs missing

`base|few_shot`, `base|guided_prose`, `base|hard_mode_hint`. Given the ceiling
effect, not worth the ~55 min.

---

## 8. Measured results — 2026-08-22

100 games, adaptive@20, paired against `baseline` on identical answers.

### `sft` arm

| variant | mean | vs base | t | solved | HMV% | opener |
|---|---:|---:|---:|---:|---:|---|
| baseline | 3.75 | — | — | 97 | 6.18 | SALET |
| with_count 🚩 | 3.78 | +0.03 | 0.69 | 97 | 6.13 | SALET |
| hard_mode_hint | 3.79 | +0.04 | 1.42 | 97 | 5.85 | SALET |
| reversed | 3.80 | +0.05 | 1.52 | 96 | 6.12 | SALET |
| guided_prose | 3.98 | +0.23 | 2.91 | 95 | 9.92 | SALET |
| constraints_only | 4.00 | +0.25 | 2.73 | 95 | 8.35 | SALET |
| emoji | 4.08 | +0.33 | 2.70 | 97 | 11.36 | SALET |
| **raw_history** | **4.15** | **+0.40** | **3.46** | 98 | 12.35 | SALET |
| verbose | 4.26 | +0.51 | 4.77 | 97 | 12.77 | SALET |
| keyboard | 4.58 | +0.83 | 6.29 | 88 | 15.02 | SALET |
| minimal | 5.49 | +1.74 | 9.51 | 61 | 23.33 | PRANK |
| few_shot ⚠ | 6.06 | +2.31 | 13.90 | 49 | 6.85 | BOOBY |

Fair spread **2.31** guesses; **0.83** excluding the two broken variants
(`few_shot`, `minimal`).

### `base` arm

Range 6.71–6.88, 91–97% failure, spread 0.17, no significant differences. See §7.

### Findings

1. **Prompt format is worth 25–75× what training was.** Fair spread 0.83–2.31
   against ±0.03 for every training intervention since Phase 6. This is the
   headline and it survives the §7 confound in magnitude, if not in
   attribution.

2. **The solver-derived constraint block is load-bearing.** `raw_history` costs
   +0.40 guesses at t = 3.46. Removing the pre-computed
   `Confirmed letters : p1=A` measurably hurts, so the harness has been doing
   part of the deduction the write-up attributes to the model.

3. **Leaking the candidate count buys nothing.** `with_count` is a null
   (+0.03, t = 0.69). The standing no-leak constraint costs no performance, and
   the model is not bottlenecked on knowing how many candidates remain. This is
   the cleanest result in the table — it has no distribution-shift confound,
   since `with_count` is `baseline` plus one line.

4. **The opener is format-robust; the midgame is not.** Ten of twelve variants
   still open `SALET`. The two that don't are `minimal` (instructions removed →
   `PRANK`) and `few_shot` (exemplar copied → `BOOBY`).

5. **Hard-mode violations track the mean.** 6.18% at `baseline` → 23.33% at
   `minimal`. Worse prompts produce more feedback-inconsistent guesses in
   exactly the regime the adaptive filter leaves unguarded (|admissible| > 20).

### The confound, stated plainly

The `sft` adapter was trained on `baseline` alone. "`baseline` wins" is
therefore partly format lock-in, not evidence it is intrinsically the best
prompt. Findings 1, 2, 4 and 5 all inherit this. Finding 3 does not.

Separating the two requires a control arm that is not ceiling-bound (§7).

### 8b. Post-hoc decomposition (added 2026-08-23, from the saved per-game arrays)

Each variant's paired degradation vs `baseline`, split into **new failures**
(games baseline solves and the variant does not, scored at the cap of 7) and
**slower solves** (both solve, variant takes more guesses):

| variant | Δmean | new fails | fail cost | slow cost |
|---|---:|---:|---:|---:|
| with_count 🚩 | +0.03 | 0 | 0 | 10 |
| hard_mode_hint | +0.04 | 0 | 0 | 6 |
| reversed | +0.05 | 1 | 1 | 6 |
| guided_prose | +0.23 | 2 | 2 | 34 |
| constraints_only | +0.25 | 2 | 4 | 38 |
| emoji | +0.33 | 2 | 4 | 59 |
| **raw_history** | **+0.40** | **1** | **1** | **64** |
| verbose | +0.51 | 3 | 5 | 63 |
| keyboard | +0.83 | 10 | 28 | 70 |
| minimal | +1.74 | 38 | 126 | 66 |
| few_shot ⚠ | +2.31 | 48 | 166 | 76 |

Two distinct failure regimes: the mid-table variants (`guided_prose` through
`verbose`) degrade almost purely through **friction** — extra probing turns,
nearly no lost games — while `keyboard`/`minimal`/`few_shot` degrade through
**outright failure**.

**This softens finding 2.** `raw_history`'s +0.40 is 64/65 slower solves and
one lost game; it actually solved 98/100, one more than baseline. Without the
solver's constraint block the model still *wins* — it just probes ~0.4 turns
longer. So the block accelerates the midgame rather than gating solvability.
"The harness has been doing part of the deduction" stands, but the honest
version is "part of the *efficient* deduction", not "the deduction".

> **RETRACTED 2026-08-23 by the 246-game re-run — see §8c.** The paragraph
> above is a sampling artifact. `raw_history` beats baseline on failures only
> on this particular 100; on the other 146 it loses 12 games to baseline's 1.
> Finding 2 stands in its original strong form. The paragraph is kept because
> the reasoning was sound given the data and the data was the problem.

**A framing correction to finding 1.** No variant beat `baseline` — the best
alternatives are nulls (+0.03, +0.04) and everything else is worse. The
measured spread is entirely downside. "Prompt format is worth 25–75× what
training was" therefore conflates *brittleness* (a bad prompt can break a
single-format 0.5B adapter — expected) with *leverage* (a better prompt could
improve it — for which the measured evidence within this family is a null).
The pre-registered decision table in RUN_PHASE9.md did not anticipate this
outcome shape — large spread, all negative — so by the project's own rules the
GRPO-vs-harness fork is still open, pending §10 item C.

### 8c. The 246-game re-run (2026-08-23)

`ARMS=["sft"]`, all 246 held-out answers, four variants, rotating non-copyable
few-shot exemplars, probe B moved ahead of the sweep.

| variant | mean | vs baseline | solved | t |
|---|---:|---:|---:|---:|
| baseline | **3.7642** | — | 242/246 | — |
| with_count 🚩 | 3.8049 | +0.041 | 241 | 1.55 |
| raw_history | 4.2805 | +0.516 | 232 | 7.07 |
| few_shot | 5.0041 | +1.240 | 215 | 13.06 |

Fair spread 1.24 (excluding the leaky `with_count`). Still entirely downside:
no variant beat `baseline`, so the framing correction to finding 1 in §8b
survives unchanged.

**The 100 was lying about failures.** The 100 answers are a subset of the 246,
so the two runs can be split:

| games | baseline | raw_history |
|---|---:|---:|
| the shared 100 | 3.750 (3 fail) | 4.150 (**2 fail**) |
| the other 146 | 3.774 (1 fail) | 4.370 (**12 fail**) |
| all 246 | 3.764 (4 fail) | 4.281 (**14 fail**) |

The subset happened to contain almost none of the games `raw_history` loses.
§8b's decomposition — 1 new failure, 64 slower solves — was arithmetically
correct on the games it had and wrong about the population. Means were stable
across the two sample sizes (3.750 → 3.764); **failure counts were not**, and
failures are what the +7 cap makes expensive. Report failure rates on 246 only.

**Determinism is settled.** Both runs share those 100 answers, and all 300
shared per-game scores (baseline, raw_history, with_count) are **bit-identical**
across two independent Kaggle sessions on different notebook versions. The
Phase 8b baseline that would not reproduce was the `find_adapter` fallback
loading the wrong adapter, not latent nondeterminism. That doubt is retired.

**Probe B, finally measured.** Decoder off, 148 states stratified by |A|:

| variant | parse% | legal% | admissible% |
|---|---:|---:|---:|
| baseline | 99.3 | 84.5 | **22.3** |
| with_count | 100.0 | 81.8 | 14.2 |
| few_shot | 99.3 | 94.6 | 12.2 |
| raw_history | 100.0 | **95.9** | **1.4** |

`raw_history` emits *more* legal words than `baseline` and almost no admissible
ones. The model retains the rules of Wordle without the constraint block and
loses the deduction — which is finding 2 measured directly on the model rather
than inferred from game outcomes. It does **not** separate lock-in from
intrinsic: the adapter never saw this format. That is §10C's job.

`few_shot` is now valid — `shot_copy_pct` = 0.0 with the rotating pool, against
a variant that previously opened `BOOBY` on every game. Its 5.00 is a real
measurement of few-shot learning failing, not an artifact.

**The failures have one shape.** All four baseline losses narrow correctly and
then cycle rhyming decoys while the answer stays admissible throughout:

```
JUDGE  |A| 12972 -> 715 -> 18 -> 8 -> 4 -> 3
       SALET  DRONE  HEDGE  BUDGE  PUDGE  MUDGE      (never JUDGE)
KRILL  |A| 12972 -> 263 ->  9 -> 4 -> 3 -> 2
       SALET  COURD  FRILL  GRILL  BRILL  PRILL      (never KRILL)
PITCH  SALET  NORTH  BUTCH  DITCH  WITCH  HITCH      (never PITCH)
WATER  SALET  TAMER  EATER  CATER  HATER  RATER      (never WATER)
```

Every guess after turn 2 is admissible; the model is not violating feedback, it
is ranking near-neighbours badly. Of the 222 games that reach |A| <= 10, 1.8%
are still lost (`raw_history` 5.7%, `few_shot` 8.2%). This is a discrimination
failure in the band that already holds 79% of the training rows — not a
coverage gap.

---

## 9. Validation performed before execution

- Every code cell `ast.parse`d.
- All twelve variants asserted to end in `Next guess:`.
- Exemplars verified legal, in the answer list, and self-consistent.
- `v_baseline` wrapping assumptions checked directly: `RULES` is literally
  present (so `few_shot`'s strip works and rules are not duplicated), and
  `Next guess:` occurs exactly once (so `with_count` / `hard_mode_hint`
  insertion is unambiguous).
- Full pipeline dry-run against a stub scorer mimicking `LegalWordScorer`'s
  surface (`.index`, `.words`, `.n`, `.select`): all twelve variants, the
  decision cache, the hard-mode filter, `summarize`, the paired statistics and
  the verdict block. The stub asserts it is never handed an empty allowed pool.
- Repository suite: 63 tests pass.

Two bugs were caught this way — the `green_pattern` placeholder (`.` is not
falsy, so two variants rendered `position 1 is .`) and the hand-written
exemplars. Exemplar *copying* (§7) was not caught, because the dry-run scorer
picks by hash and cannot copy.

---

## 10. Recommended next actions (revised 2026-08-23)

### A. Housekeeping — done or one command away

| action | status |
|---|---|
| Preserve the per-game record in the repo | **done** — `results/phase9/harness_results.json` (recovered from Downloads; it existed nowhere else. The Aug-22 session was interactive, so its output is not retrievable via the Kaggle API — run future sweeps as committed runs through `tools/kaggle_run.py` so results are always pullable) |
| Upload `uploads/phase7_adapter/tree_salet_endgame` as a Kaggle dataset | **blocker for any API re-run.** It is in no current dataset; the Aug-22 session found it through an ad-hoc attachment that a fresh session cannot reproduce |
| `ARMS = ["sft"]` | keep — the base arm is ceiling-bound (§7) and cost 84.5% of wall time |

### B. One cheap GPU session (~25 min total)

| action | cost | note |
|---|---|---|
| Probe B, before the sweep this time | ~2 min | `format` is empty; this is the only measurement of what the model *emits* vs what the decoder repairs, and it has no lock-in confound |
| `few_shot` with non-copyable exemplars | ~3 min | closes the §7 defect |
| Top 3–4 variants at `N_GAMES = 246` | ~15 min | for the write-up, report as *tightened precision*, not replication — the 100 games are a subset of the 246, and re-measuring selected winners is not an independent confirmation |

### C. The decisive experiment — a format crossover (one SFT run)

The §8 confound (lock-in vs intrinsic) and the `raw_history` question
(can the model learn the deduction the solver block does for it?) are **the
same experiment**: train the identical SFT recipe on `raw_history`-rendered
rows (`phase2_trajectories/rerender_prompts.py` already exists for exactly
this), then evaluate the 2×2:

| | eval `baseline` | eval `raw_history` |
|---|---|---|
| adapter trained on `baseline` | 3.75 (measured) | 4.15 (measured) |
| adapter trained on `raw_history` | ? | **?** — the cell that decides |

- Both adapters best on their own format by a similar margin → **lock-in**;
  format is a robustness problem, not a quality ranking.
- `raw_history`-adapter on `raw_history` recovers to ≈3.75 → the deduction is
  **learnable**; the solver block was a crutch for this adapter, not a
  requirement — and the "harness is the model" worry mostly dissolves.
- It stays ≈4.15 → the deduction genuinely needs the solver at 0.5B, and the
  constraint block is honest scaffolding to keep.

Cost: one Phase-6-style SFT run (~4 h T4) + 4 paired evals. This is the item
the old list deferred as "design work"; it is now the highest-value run in the
project.

### D. Conditional, after C

- **If lock-in:** mixed-format SFT — one adapter trained on all non-leaky
  variants. Pre-registered expectation: hard-mode violations drop (finding 5:
  HMV% tracks format quality in exactly the |admissible| > 20 regime the
  adaptive filter leaves unguarded), mean holds or improves. This is the first
  training intervention since Phase 6 with a mechanism for why it should move
  the number.
- **If intrinsic / learnable-but-not-learned:** the prompt thread closes with a
  clean conclusion, and the GRPO-vs-accept-3.76 decision resurfaces with the
  scaffolding half finally measured.

### E. Instrumentation for whichever run is next

Log `|admissible|` per decision in `play()`. The Phase 8 counterfactual showed
74.7% of the classical gap lives at 2–10 admissible; nothing in Phase 9 can
currently say whether prompt degradation concentrates there too, because only
per-game totals were saved. One extra field per decision closes that.

---

## 11. Files

```
phase9_harness/
  make_harness_notebook.py            generator — edit this
  wordle_phase9_harness_kaggle.ipynb  build artifact — do not edit
  TECHNICAL.md                        this file
  _paths.py                           puts core/ on sys.path, chdir to root

core/prompt_variants.py               the twelve renderers (inlined at build)
core/constrained_decode.py            the decoder (inlined at build)
docs/RUN_PHASE9.md                    operator guide
```

Outputs: `results_phase9/harness_results.json` (full, with per-game arrays
needed for any later paired re-analysis), `results_phase9/summary.json`,
`wordle_phase9_results.zip`, and Kaggle Dataset `wordle-phase9-harness` when
secrets are present.
