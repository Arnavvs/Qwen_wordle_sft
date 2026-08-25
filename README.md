# Wordle: distilling a classical solver into a 0.5B LLM

Eleven experiments on one question: **how much of a symbolic solver's skill can
you put inside a small language model, and what stops you?**

A classical solver plays Wordle in **3.4431** guesses. `Qwen2.5-0.5B-Instruct`,
untrained, cannot play it at all — 91–97% failure under every prompt tried.
Distilled from that solver and decoded under a feedback-consistency constraint,
it reaches **3.7642 guesses, 242/246 solved, 1.6% failure** on a held-out set.

That beats classical `random` (4.0203) and `frequency` (3.7927), and sits
**0.32 guesses** short of the best classical solver. Four separate attempts to
close that last 0.32 all failed, and the interesting part of this project is
*why* — the gap turned out not to be the thing anyone assumed.

**Everything here is reproducible.** Every number comes from a script or a
notebook in this repository, seeds fixed, and the two headline training runs
were re-executed on different machines and produced **byte-identical** weights.

---

## The result

| | mean guesses | solved / 246 | failure |
|---|---:|---:|---:|
| classical `entropy` | **3.4431** | 246 | 0% |
| classical `frequency` | 3.7927 | 242 | 1.6% |
| **0.5B LLM + adaptive decoder** | **3.7642** | **242** | **1.6%** |
| classical `random` | 4.0203 | 244 | 0.8% |
| 0.5B LLM, no decoder | 5.9634 | 83 | 66.3% |
| stock Qwen2.5-0.5B | 6.71–6.88 | — | 91–97% |

---

## Three findings worth the reader's time

### 1. The decoder was worth more than all the training

Restricting each guess to words *consistent with the feedback already received*
moved the model from 5.4837 to 3.9472 — **−1.54 guesses**, more than everything
learned in Phases 4–6 combined. Filtering only once the candidate set is small
(the "adaptive" decoder) bought another 0.16, because the expert's own turn-2
policy is feedback-*inconsistent* 59% of the time: it probes with a word that
cannot win, and an always-on filter forbids that.

The model is not a passenger. Playing the same games by picking *at random*
among admissible words scores 4.52, so the model contributes **−0.57** there
and **−1.22** under the adaptive decoder.

### 2. The last 0.32 is not an action-selection problem

Two post-training methods, opposite failure modes, same answer:

| | what happened | effect |
|---|---|---:|
| Phase 8 — DPO | drifted away from the SFT policy | **−0.1708** |
| Phase 11 — GRPO | held its ground, changed nothing | **+0.0041** (n.s.) |

Phase 11's *decision budget* explains both **in advance**. Pricing all 922 of
the model's real decisions against the best action available at each state:
perfect action selection across the entire regime the decoder restricts is
worth only **0.0698 guesses** — because the model is already optimal in 85.6%
of those decisions, and 83% of the remainder are exact ties it could not have
got wrong.

DPO had less upside than its own drift. GRPO had upside it could not find
because there was almost none left. Neither was a botched run; the ceiling was
measured first and both results landed where it predicted.

### 3. Prompt-format results were mostly lock-in, not quality

Stripping the solver-derived constraint block from the prompt costs **+0.52
guesses** — which looks like proof the harness is doing the model's reasoning.
It isn't. Training a second adapter on the *same 19,212 rows* with only the
prompt re-rendered gives a clean 2×2:

| | eval `baseline` | eval `raw_history` |
|---|---|---|
| trained `baseline` | **3.7642** | 4.2805 |
| trained `raw_history` | 4.1098 | **3.8089** |

Each adapter is best on its own format (both off-format penalties significant,
t = 7.07 and 4.92), and the diagonal is a statistical tie (t = 1.08). So the
penalty was **format lock-in**.

But the decoder-off probe cuts the other way and is the more interesting half:
the `raw_history` model emits **0% admissible words unaided** on its own format
while still playing 3.8089. It reaches equivalent scores having learned
essentially no feedback consistency — leaning entirely on the decoder. The
constraint block is replaceable *for score* and load-bearing *for what the
model knows*.

---

## So what is the gap?

The capability limit measured back in Phases 4–6, which nothing since has
moved: **given a state with exactly one possible answer, free spelling, and
nothing left to decide, the best model names it 20% of the time** — 33% even
when it was trained on that exact word.

Better action selection cannot fix an inability to produce the right word. The
cause is structural: a 3.46-guess expert almost never visits the endgame, so
distilling it supplies thin, early-skewed coverage. The next lever is a better
model or better endgame coverage, not a better objective.

---

## The eleven phases

| | question | outcome |
|---|---|---|
| 1 | How far does classical Wordle solving get? | `entropy` 3.4644, 0% fail |
| 2 | Can an expert produce clean demonstrations? | 3 policies × 2,315 games |
| 3 | Can they become leak-free SFT data? | 7,067–7,173 rows/policy |
| 4 | Does SFT on them work? | No — 76–79% failure |
| 5 | Is the failure vocabulary or reasoning? | Neither, mostly |
| 6 | Does endgame-heavy data fix it? | Generalises (p=0.031), games unchanged |
| 7 | Does a feedback-consistent decoder fix it? | **Yes — 3.78, beats `random`** |
| 7b | What is the right filter threshold? | plateau at 10–50; **3.7642** |
| 8 | Where is the gap, and does DPO close it? | 74.7% in 2–10 words; DPO **regressed** |
| 9 | Is the harness doing the model's work? | spread 1.24 guesses, all downside |
| 10 | Is that lock-in or the format? | **lock-in**; own-format parity |
| 11 | Is the rest of the gap action selection? | **No.** GRPO +0.004, n.s. |

Full narrative with every result, including the corrections:
**[PROJECT_README.md](PROJECT_README.md)**.

---

## Methodology notes

The parts that took the most care, and that a reader may want to check:

- **One measurement path.** Every phase scores through the same Phase 9
  harness, with a control cell that must reproduce 3.7642 before any other
  number is read. Forking the measurement path is what voided an earlier run.
- **Paired tests only.** The unpaired SE on 246 games is ~0.064, which cannot
  resolve the effect sizes post-training produces. Every comparison is paired
  on identical answers.
- **Ceilings before methods.** `phase11_grpo/decision_budget.py` prices what a
  perfect policy would buy *before* a method is chosen. It cancelled one
  planned training run whose downside exceeded the entire available upside.
- **Pre-registered readings.** Phases 10 and 11 wrote their outcome
  interpretations *before* the run — see the `RUN.md` in each folder, where
  sections written beforehand are left untouched.
- **Leakage control.** Answer-keyed splits, no answer / candidate list /
  candidate count in any prompt, and audit scripts that re-derive every state
  from its own rendered prompt rather than trusting metadata.

---

## Layout

```
core/                    solver, feedback engine, lookahead, decoder
phase1_classical/        the classical solver + its own README
phase2_trajectories/     expert rollouts -> step records
phase3_sft_package/      step records -> leak-free SFT data
phase4_5_sft_diagnostic/ train adapters, then the constrained diagnostic
phase6_endgame/          endgame-heavy SFT
phase7_constrained/      feedback-consistent decoding + threshold sweep
phase8_dpo/              DPO (negative result)
phase8_dpo_v3/           audited DPO rebuild + the audit that cancelled it
phase9_harness/          the prompt-format harness = the measurement path
phase10_crossover/       the format crossover
phase11_grpo/            decision budget, GRPO tasks, GRPO run
docs/                    MATH.md (every formula, worked), run guides
tools/                   Kaggle packaging + API run driver
results/                 the experimental record, as JSON
tests/                   63 tests
```

Large regenerable artifacts (model weights, trajectory dumps, Kaggle staging)
are excluded from git — see `.gitignore`, which says what each one is and how
to rebuild it. The JSON results in `results/` are the experimental record and
are tracked.

## Reproducing

```bash
conda env create -f environment.yml
python phase1_classical/build_artifacts.py      # vocabulary + feedback matrix
python -m pytest                                 # 63 tests
```

GPU work runs on Kaggle through `tools/kaggle_run.py` (see
`.claude/skills/kaggle-run/`), which drives the whole lifecycle through the API.

## Credits

Base model `Qwen/Qwen2.5-0.5B-Instruct`. Word lists in `data/` with provenance
in `data/PROVENANCE.json`. Trained and evaluated on Kaggle T4s.
