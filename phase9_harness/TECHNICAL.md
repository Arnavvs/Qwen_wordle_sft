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

## 10. Recommended next actions

| priority | action | cost |
|---|---|---|
| 1 | `ARMS = ["sft"]` — drop the ceiling-bound arm | saves ~85% of runtime |
| 2 | Re-run `few_shot` with non-copyable exemplars | ~3 min |
| 3 | Run probe B (move cell 18 before cell 16) | ~2 min |
| 4 | Re-run the top 3–4 variants at `N_GAMES = 246` for publishable numbers | ~15 min |
| 5 | A discriminating control arm to break the §8 confound | design work |

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
