# Wordle: classical solver → LLM distillation

Full project log: classical solver, distillation into a 0.5B LLM, why that
failed, and the diagnostic run that identified the real cause.

**Where this stands after Phase 9.** The headline number is unchanged at
**3.7642**, and it now has a much stronger claim to being real: two independent
Kaggle sessions on different notebook versions produced **bit-identical**
per-game results across 300 shared games. The Phase 8b baseline that would not
reproduce was a `find_adapter` fallback loading the wrong adapter, not latent
nondeterminism — so the doubt that hung over every paired comparison in this
project is retired.

Phase 9 varied the *prompt* with the decoder held fixed, across twelve formats.
The spread is 1.24 guesses and **every variant is worse than the format the
model was trained on** — the two nearest are statistical nulls and the rest
degrade. That is a brittleness result, not a leverage result: it says a bad
prompt can break a single-format 0.5B adapter, which is unsurprising, and says
nothing about whether a better prompt exists. Removing the solver-derived
constraint block (`raw_history`) costs 0.52 guesses and ten solved games, and
with the decoder switched off that adapter emits **1.4%** admissible words
against `baseline`'s 22.3% — while emitting *more* legal words. It keeps the
rules and loses the deduction.

All of it carries one confound: the adapter only ever saw `baseline`. **Phase
10 is built to break that tie** and is ready to run — see
[phase10_crossover/](phase10_crossover/).

A training-data audit run alongside it found something that reframes the
endgame work: 19,212 training rows are only **13,872 distinct boards**, and
13,814 of those sit at ≤10 candidates. Above 10 candidates the entire corpus
holds **58 distinct boards** — 1 opening, 11 midgame, 46 late-midgame — because
a good solver opens the same word every game and reaches a decisive position in
three moves. The endgame is the data-*rich* half.

**Where this stood after Phase 8.** A 0.5B LLM distilled from a classical
expert, decoded under a feedback-consistency constraint, plays Wordle at
**3.7642 mean / 1.6% failure** on 246 held-out answers — or **243/246 solved at
1.2% failure** if tuned for wins rather than speed. That beats classical
`random` (4.0203) and `frequency` (3.7927), and sits **0.32** off the best
classical solver (`entropy`, 3.4431).

Almost all of that came from the **decoder**, not the model. Restricting each
guess to words consistent with the feedback already received was worth ~1.5
guesses; everything learned in Phases 4–6 was worth a fraction of that. But the
model is not a passenger: playing the same games by picking *at random* among
admissible words scores 4.52, so the model contributes **−0.57** there and
**−1.22** under the adaptive decoder, where its learned probing policy replaces
a random probe.

The earlier reading — that endgame retrieval was the binding constraint — was
wrong. It was a real weakness (Phase 6 improved unseen-word k=1 accuracy
15.25% → 25.42%, p = 0.031) but not the one that governed games.

**Phase 8 located the remaining gap and failed to close it.** A counterfactual
analysis shows **74.7%** of the 0.32 lives in states with 2–10 admissible words
— perfect play there alone solves 246/246. A DPO run aimed squarely at those
states learned the preference (accuracy 0.60 → 0.84) and made gameplay
**significantly worse** (3.7642 → 3.9350, paired t = −3.21, 70 games worse
against 38 better). Diagnosis: 64% of the pairs were easy ones the model already
ranked correctly, so it spent the run inflating margins instead of fixing
mistakes, and paid for the drift in real play.

**Two companion documents:**

- **[docs/MATH.md](docs/MATH.md)** — every formula used anywhere in the project,
  each with a real worked example printed from the actual data files: feedback
  encoding, entropy and expected-remaining, the SFT loss and its token mask,
  constrained-decoding scores and the branch-and-bound proof, the hard-mode
  filter, the DPO objective, and the GRPO objective planned next.
- **[docs/site.html](docs/site.html)** — the same story written for someone not
  working on the project, published as a web page.

`README.md` (unchanged) is the deliverable doc for **Phase 1** — the classical
solver on its own. This file covers the whole arc, Phases 1–10.

**Phase-level technical notes** live next to their code:
[phase9_harness/TECHNICAL.md](phase9_harness/TECHNICAL.md) (§8c holds the
246-game re-run, the §8b retraction, probe B, and the failure analysis) and
[phase10_crossover/RUN.md](phase10_crossover/RUN.md) (the crossover's
pre-registered design and how to run it).

---

## How to read this

Nine runs across eight phases, chronological. Each has a **design** section (what was built and
why) and a **results** section (what happened, including where a conclusion was
later corrected).

| Phase | Question | Outcome |
|---|---|---|
| 1 | How far does classical Wordle solving get? | `entropy` 3.4644, 0% fail |
| 2 | Can an expert produce clean demonstrations? | 3 policies x 2,315 games |
| 3 | Can they become leak-free SFT data? | 7,067–7,173 rows/policy |
| 4 | Does SFT on them work? | No — 76–79% failure |
| 5 | Is the failure vocabulary or reasoning? | Verdict D: neither, mostly |
| 6 | Does endgame-heavy data fix it? | Generalises (p=0.031), games unchanged |
| 7 | Does a feedback-consistent decoder fix it? | **Yes — 3.78, beats `random`** |
| 7b | What is the right filter threshold? | plateau at 10–50; **3.7642**, 243/246 |
| 8 | Where is the gap, and does DPO close it? | 74.7% in 2–10 words; DPO **regressed** (t=−3.21) |
| 8b | Does a clean dataset change that? | run void — wrong adapter silently loaded; cause found, see Phase 9 |
| 9 | Is the harness the model? | spread 1.24 guesses, **all downside**; no prompt beat `baseline` |
| 10 | Is that lock-in or the format? | **lock-in.** Own-format parity (t=1.08); both off-format penalties significant |
| 11 | Is the rest of the gap action selection? | **no.** GRPO 3.7602 vs 3.7642, t=−0.28; 233/246 games identical |

Phases 1–3 were built before the numbering existed and are labelled
retrospectively; the work is unchanged.

---

# Phase 1 — classical solver

Built a symbolic + information-theoretic solver, no ML anywhere, as the
reference policy to distil from. Full detail in `README.md`.

Full 2,315-answer benchmark, `guess_pool="full"`, `max_guesses=6`:

| Solver | Opener | Mean | Fail |
|---|---|---:|---:|
| `random` | — | 4.0259 | 1.56% |
| `frequency` | STARE | 3.7459 | 1.38% |
| **`entropy`** | **SOARE** | **3.4644** | **0%** |
| `expected` | ROATE | 3.4812 | 0% |
| `minimax` | ARISE | 3.5732 | 0% |
| `hybrid` | RAISE | 3.4812 | 0% |

Reproduces the published figures for this vocabulary (SOARE 5.886 bits, ROATE
60.42 expected remaining, minimax worst case 168) — external validation that the
feedback function and entropy computation are correct.

Supporting experiments, all logged:

- `tiebreak_experiment.py` → `tiebreak_log.txt`. Tie-break rule is worth up to
  0.065 guesses (3.4631–3.5279). Not noise; the shipped rule is the best one.
- `tree_search.py` / `run_tree_benchmark.py` → `tree_scaling.txt`. Depth-6
  lookahead with `top_k=100`: mean 3.4300. Confirms the greedy solvers sit
  ~0.03 above true lookahead — the "greedy is not optimal" caveat, quantified.
- `verify_salet_tree.py` → `verify_salet.txt`. SALET tree policy: 3.4212,
  matching the known optimum for this word list.

---

# Phase 2 — expert trajectory generation

`generate_trajectories.py` rolled out three expert policies over all 2,315
answers, recording every `(state → action)` step.

| Policy | Mean (2315) | Steps | Opener |
|---|---:|---:|---|
| `entropy` | 3.4644 | 8020 | SOARE |
| `tree_soare` | 3.4544 | 7997 | SOARE |
| `tree_salet` | 3.4212 (optimal) | 7920 | SALET |

`tree_soare` exists to make the comparison interpretable: `entropy` vs
`tree_salet` differ in **both** opener and policy, and the opener accounts for
77% of the gap. So:

- `entropy` vs `tree_soare` → isolates the **decision policy** (shared opener)
- `tree_soare` vs `tree_salet` → isolates the **opening word** (shared policy)

---

# Phase 3 — SFT package

`build_sft_package.py` → `sft_package/`, audited by `verify_no_leakage.py`.

Prompt format — history, derived constraints, turn counter, nothing else:

```
You are playing Wordle. Deduce the hidden 5-letter word.
...
History:
  1. SOARE -> BYYBB
  2. CLOOT -> BGGBG
  3. ABAFT -> GBBGG

Deduced so far:
  Confirmed letters : p1=A p2=L p3=O p4=F p5=T
  Letters present   : A, F, L, O, T
  Exact counts      : Ax1, Ox1
  Letters absent    : B, C, E, R, S
  Ruled-out spots   : A not at [2], O not at [1, 3]

Guess 4 of 6 (3 remaining)

Next guess:
```

Completion: `ALOFT`. Loss on the completion only.

Leakage controls that were deliberately built in:

- The **answer** never appears — not a parameter of `render_prompt`, and
  structurally impossible in history (a correct guess ends the game).
- The **candidate count** is never shown. It is a solver-side quantity a player
  cannot see, and it leaks how far the constraints already narrow the space.
  Kept in `meta.n_candidates` for analysis only. There is no flag to re-enable it.
- The **candidate list** is never shown.
- Greens render position-by-position (`p1=A p2=L ...`), never concatenated —
  once all five are known, the concatenation would *be* the answer.
- Answer-keyed split `sha256(salt|answer)`, identical across all three solvers:
  **2,069 train / 246 held-out**.
- `game_id` and `_holdout` stripped; ids hashed.

Result: 7,067–7,173 train records per policy, ~850 val.

---

# Phase 4 — Kaggle SFT run

`make_sft_notebook.py` → `wordle_qwen_sft_kaggle.ipynb`, run on Kaggle T4.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA | r=16, α=32, dropout=0.05, all 7 proj modules |
| Optimisation | lr 2e-4 cosine, 2 epochs, eff. batch 16, fp16, seq 640 |
| Runs | 3 (one per expert policy), identical hyperparameters |
| Train time | ~24 min each on a T4 |
| Eval | 246 held-out answers, greedy, `max_new_tokens=8` |
| Final loss | 0.508 / 0.485 / 0.481 (from ~4.7) |
| Truncated examples | 0 |

Evaluation rules, as recorded in `eval_settings`: greedy decoding; strict
parsing (first token only); an invalid guess **consumes the turn and returns no
feedback**; no silent repair.

---

## The two result folders this produced

Both under `results/`, from the same Kaggle run (`generated_utc 2026-08-17T22:18:35Z`).

| Folder | Size | Contents |
|---|---:|---|
| `wordle_sft_results/` | 740 MB | complete run output — 3 LoRA adapters, 2 checkpoints each, optimizer/scheduler state, tokenizer, `results.json`, `environment.json`, `dataset_hashes.json` |
| `wordle_sft_important_results/` | 8.7 KB | curated extract — same `results.json`, plus CSV summaries, adapter configs, `README.txt` |

`results.json` is **byte-identical between the two**. The big folder adds only
weights and training state. Nothing was lost in the curated copy.

---

## Phase 4 results — and they are bad

On the same 246 held-out answers (failures scored as 7):

| | Mean | Median | ≤3 | Fail % | Invalid fmt % | Invalid word % |
|---|---:|---:|---:|---:|---:|---:|
| `qwen_entropy` | 6.3902 | 4.0 | 8.54 | **79.3** | 8.13 | 28.61 |
| `qwen_tree_soare` | 6.3211 | 4.0 | 9.35 | **78.5** | 8.74 | 27.83 |
| `qwen_tree_salet` | **6.1748** | 3.0 | 13.41 | **76.4** | 7.81 | 29.75 |
| — classical `entropy` | 3.4431 | 3.0 | 56.1 | 0.0 | — | — |
| — classical `random` | 4.0203 | 4.0 | 27.64 | 0.81 | — | — |

The best SFT model loses to **random guessing with perfect constraint
elimination** by 2.15 guesses, and fails 3 games in 4.

---

## What actually happened

Four findings, in order of how much they explain.

## 1. The model learned the policy. It just cannot finish.

This is the part that gets missed by looking at the mean. The policy transfer
worked, and worked well:

- **Opener: 100% exact**, all three models, all 246 games. One distinct opener
  each — SOARE, SOARE, SALET as trained.
- **Turn 2 on disagreement states: 90.1% exact match** with its own expert. On
  the 111 states where the `entropy` and `tree` experts choose differently, the
  entropy-trained model picks the entropy action 90.1% of the time and the tree
  action **0.0%** of the time. The tree-trained model mirrors it: 89.9% tree,
  0.0% entropy. The two models are cleanly separated by their teacher.
- It reproduces obscure expert probes verbatim — CLOOT, BARCA, RIYAL, MARON,
  CANAL, DIRER. That is not pattern-matching; that is the policy.

And the information gathering is at expert parity:

| Mean candidates remaining | after g1 | g2 | g3 |
|---|---:|---:|---:|
| classical `entropy` (2315 games) | 62.30 | 3.35 | 1.07 |
| `qwen_entropy` (246 games) | 62.24 | **3.19** | 1.45 |

After two guesses the model has narrowed 2,315 words to **3.19** — slightly
better than the expert it copied. Then it stalls at ~1.34 for the remaining
four turns and dies.

**The model is excellent at narrowing the space and cannot name the word it has
narrowed to.**

## 2. Root cause: the winning move is a vocabulary lookup, and the answers were held out

The last move of a Wordle game is not a decision. It is retrieval: emit the one
English word consistent with the constraints. Check what the training data
taught about that:

```
train answers appearing as a training target: 2069 / 2069  (100%)
val   answers appearing as a training target:   41 /  246  (16.7%)
```

Every training answer was demonstrated as a winning guess at least once. **83%
of the evaluation answers were never once produced as a target.** The split is
answer-keyed and clean — that is exactly what it was designed to do — but it
means the SFT task decomposes into:

- a **policy** part (which probe splits the space best) → generalises, learned, 90% match
- a **lexicon** part (which word is it) → cannot generalise, because the eval
  words were held out by construction

So the model does what a model with the constraints but not the lexicon entry
does — it generates a phonotactically plausible non-word that satisfies the
pattern:

| Answer | Model emitted | Constraint state |
|---|---|---|
| ABOVE | **ABOBE** ×3 | `AB..E`, 1 candidate left |
| ABOVE | **ABHIE** ×3 | `AB..E`, 1 candidate left |
| AGING | **AMING** ×2 | `A..NG`, 2 candidates left |

Every one is a near-miss on a fully-determined state. The deduction is
finished; the word is not in the model's reach. That is the whole 28–30%
invalid-word rate.

## 3. The training distribution spent 58% of its budget on 125 states

Distribution of the 7,173 entropy training examples:

| Turn | Examples | Distinct prompts | Median candidates | Fully determined |
|---:|---:|---:|---:|---:|
| 1 | 2,069 | **1** | 2,315 | 0 |
| 2 | 2,069 | **124** | 42 | 0 |
| 3 | 2,029 | 1,197 | 2 | 4 |
| 4 | 946 | 899 | 1 | 2 |
| 5 | 59 | 58 | 1 | 0 |
| 6 | 1 | 1 | 1 | 0 |

- **28.8% of all gradient goes to one constant** (turn 1 → SOARE).
- **57.7% goes to 125 distinct states** (turns 1–2), each seen 17–2069 times.
- Turns 4–6 — where games are actually won — get **14%**, almost all singletons.
- **`fully_determined` states: 6 out of 7,173.** The finishing move, the one
  the model fails at, has six examples in the entire dataset.

This is the natural visitation distribution of an expert that averages 3.46
guesses: good experts spend almost no time in the endgame, so distilling them
teaches almost nothing about it. The duplication was known and recorded
(`duplication_factor` ~3.2 in the manifest, and `dedup` configs were built) but
the baseline runs used `dedup: false`.

The loss curve corroborates it: 4.62 → 1.76 within 50 steps (that is the
constant opener plus output format), then a slow crawl to 0.51. Since ~58% of
examples are near-zero-loss memorised states, the effective loss on the states
that decide games is roughly 1.2/token — the model never learned that part at all.

## 4. The scoring rule turns one hallucination into a guaranteed loss

From `play_games` in the notebook: on `invalid_word` or `invalid_format`, the
turn is consumed, `g.history` is **not** appended, and no feedback is returned.
The next prompt is therefore identical apart from the turn counter — and
decoding is greedy. So the model re-emits the same invalid word, and again, and
again, until the turns run out.

The measurements confirm the spiral:

| | repeated-guess games | unique guesses / game | hard-mode violations |
|---|---:|---:|---:|
| `qwen_entropy` | 67.1% | 4.31 / 6 | 30.9% |
| `qwen_tree_soare` | 64.6% | 4.26 / 6 | 26.9% |
| `qwen_tree_salet` | 63.0% | 4.11 / 6 | 26.2% |

One hallucination at turn 3 costs turns 3, 4, 5 and 6 — not one turn. This is
the multiplier that takes a ~30% per-turn error rate to a ~78% game failure
rate.

**This is not a bug** — it is the honest scoring rule, chosen deliberately
("no silent repair"). But it means the headline number measures vocabulary
recall under a compounding penalty, not policy quality.

### Ruled out

- **Not a parsing bug.** `would_recover_pct = 0.0` for all three models — the
  lenient parser would have rescued exactly nothing. The 8% `invalid_format`
  cases contain no 5-letter token anywhere in the output.
- **Not train/eval prompt drift.** The notebook asserts prompt-format parity
  against the training file before evaluating, and asserts the candidate count
  is absent from both.
- **Not leakage or a broken split.** `verify_no_leakage.py` passes; the split is
  answer-keyed and consistent across solvers.
- **Not undertrained to the point of nonsense.** 100% opener accuracy and 90%
  expert match at turn 2 prove the pipeline works end to end.

## What the three-way comparison shows

The experiment those three runs were designed to answer is **underpowered at
this failure rate**. From `power_log.txt`, on the *classical* policies:

```
entropy - tree_soare    +0.0081  t=0.32  INDISTINGUISHABLE
entropy - tree_salet    -0.0244  t=0.54  INDISTINGUISHABLE
tree_soare - tree_salet -0.0325  t=0.71  INDISTINGUISHABLE
needs ~1024 paired games to resolve the true 0.0432 gap; you have 246
```

The expert policies differ by 0.04 guesses. The models differ from the experts
by 2.7. The teacher signal is two orders of magnitude below the noise floor.
`tree_salet` "winning" at 6.17 is not evidence that the optimal policy distils
better — it is 188 vs 195 failures on 246 games.

## Also missing

There is **no base-model control**. `classical_baselines_same_246` contains
only the five classical solvers. Untrained Qwen2.5-0.5B-Instruct was never
evaluated on the same 246 games, so the results cannot say how much the SFT
helped — only that the result is worse than `random`. That row should be added;
it is cheap.

---

---

---

# Phase 5 — the diagnostic run

Before any retraining or GRPO, one question has to be settled: **is the
bottleneck vocabulary generation or Wordle reasoning?** The unconstrained
numbers cannot separate them, because a spelling failure and a bad decision
both show up as a lost game.

Nothing here trains anything. The existing adapters are reloaded from the Kaggle
input dataset.

## Constrained decoding — `constrained_decode.py`

The honest version of "restrict to legal words". It is **not** a post-hoc filter
on generated text; the model never emits free text in this mode.

For a prompt `p` and each of the 12,972 legal guesses `w`, tokenized exactly as
the SFT data was (`" " + WORD`, then EOS):

```
score(w) = log P(w | p) = Σ log P(t_i | p, t_0..t_{i-1})
```

and the argmax over the legal set is played. EOS is included so the scores are
probabilities of *complete* strings — without it, a word whose tokens are a
prefix of a longer word's is systematically over-ranked. No length
normalisation: `score(w)` is exactly the probability the model assigns to
playing `w`, which is the quantity to argmax. (`LENGTH_NORMALISE=True` exists as
a sensitivity check; it is a heuristic and off by default.)

The question the model answers is *"which of these 12,972 words should I
play?"*. It still gets no candidate list, no candidate count, and no answer.
The constraint is purely lexical — it is the Wordle keyboard, which a human
player also has.

**Cost.** Naively 12,972 forward passes per turn. Instead the prompt is run once
with `use_cache=True`; its next-token distribution gives every word's first-token
log-probability for free, and the remaining tokens are teacher-forced through a
KV cache expanded to a batch of 512. Words are ≤ 4 tokens (+EOS), so the padded
matrix is `[12972, 5]`.

**Exact pruning.** Every per-token log-probability is ≤ 0, so
`score(w) ≤ lp0[first_token(w)]` is a valid upper bound. Words are visited in
descending order of that bound and the scan stops once the incumbent beats the
bound of everything unvisited. This is branch-and-bound argmax, not a heuristic
shortlist — `verify_against_full()` asserts the pruned answer equals the
full-scan answer.

**Guards that make the run fail loudly rather than quietly**, all executed once
per session before any number is produced:

- `self_test()` compares cache-reuse scoring against naive full-sequence
  scoring. Probe words are spread evenly across the whole list, not taken from
  the front — a cache mutated in place by the first chunk would only corrupt
  *later* chunks, and a front-loaded probe would miss it.
- `verify_against_full()` re-checks the pruning bound.
- If repeat-banning is enabled, sequential banning must reproduce the global
  ranking exactly.

The cache check is **two-sided**: a dtype-aware absolute tolerance (0.02 nats
fp32, 0.40 fp16) *and* a ranking-correlation test. The first run on a T4 showed
why both are needed — fp16 produced a 0.037-nat deviation against a single
fp32-calibrated 0.02 threshold and aborted the run. The two scoring paths use
different matmul shapes, so fp16 reduction order differs; that is arithmetic
noise, not a broken cache. The discrepancy is a *validation* artifact in any
case: all 12,972 words go through the same fast path, so the ranking the argmax
is read off is internally consistent.

Verified locally on CPU:

| Check | Result |
|---|---|
| cache-reuse vs naive, real Qwen2.5-0.5B weights | max &#124;Δ&#124; `2.2e-05`, corr `1.000000` |
| cache-reuse vs naive, full 12,972 words, probes across all 26 chunks | max &#124;Δ&#124; `4.8e-06` |
| pruned argmax == full-scan argmax | identical |
| sequential banning vs global ranking | identical |

**Fault injection**, real weights — deliberately corrupting `expand_cache`:

| Injected fault | Ranking corr | Caught? |
|---|---:|---|
| keys halved | `0.715` | yes |
| keys rolled 3 positions | `0.435` | yes |

Detection comes from the **correlation** check, not the absolute tolerance —
which is why loosening the fp16 tolerance to 0.40 opens no blind spot. A
corrupted cache lands at corr 0.4–0.7, nowhere near the 0.999 threshold.

(An earlier attempt at this injection on a randomly-initialised model showed
almost no effect — a random model barely uses its context, so it cannot
demonstrate cache corruption at all. The test only means something on trained
weights.)

The ban check earned its keep immediately: it caught a real bug where `banned`
masked the pruning *bound* but not the final *scores*, so a banned word could
still be returned. Fixed and re-verified. (The flag is off by default, so this
never affected a published number — but it was live code.)

## What the run produces

| Section | Output |
|---|---|
| 8b | constrained decoder + its two correctness guards |
| 9–11 | every metric in **both** modes: mean, solved %, invalid-word, repeated-guess, hard-mode violation, ≤3/≤4/≤5/≤6 |
| — | **base Qwen control**, untrained, same 246 games, both modes |
| 12b | **terminal-state probe** |
| 11 | disagreement analysis re-run in both modes |
| 13b | `results/` tree + comparison table |
| 13c | mechanical A/B/C/D verdict |

## The terminal-state probe (section 12b)

The classical solver — not the model — is rolled forward until exactly `k`
candidates remain, `k ∈ {1,2,3}`. At `k=1` the visible constraints already
determine the answer; nothing is left to decide except naming it. Verified
locally against the real artifacts:

| k | probe states (of 246 answers) | reached at turn |
|---:|---:|---|
| 1 | 203 | mostly 3–4 |
| 2 | 64 | mostly 3 |
| 3 | 27 | mostly 3 |

The first `k=1` state generated is exactly the ABOVE case from the post-mortem —
expert history `SOARE→BYYBG, BUNDT→YBBBB`, one candidate left, and the
unconstrained model answered `ABOBE`.

At each state the model ranks all 12,972 words; recorded are top-1 accuracy,
whether top-1 is at least *a* consistent candidate, the answer's rank out of
12,972, and its rank within the `k` candidates. Chance is `1/12972` and `1/k`.

Every state is tagged with **whether that answer was ever a training target**.
High accuracy on seen answers collapsing on unseen ones is the direct
confirmation of the held-out-vocabulary hypothesis.

## Verdict rule (section 13c)

Applied mechanically so the reading is not retrofitted, and printed alongside
its evidence:

- **C** if constrained SFT is not meaningfully better than constrained base Qwen
- **A** if constrained SFT lands within 0.35 of the classical expert
- **B** if it beats base and random but stays short of the expert
- **D** if it still loses to random elimination

## How to run it

Attach both datasets, then in the CONFIG cell:

```python
PREV_RUN_DIR   = "/kaggle/input/wordle-sft-adapters/wordle_sft"
RUN_ENTROPY    = False
RUN_TREE_SOARE = False
RUN_TREE_SALET = False
RUN_SMOKE_TEST = False
EVAL_MODE      = "both"
RUN_BASE_CONTROL   = True
RUN_TERMINAL_PROBE = True
```

Run every cell. ~1 h on a T4. Download `wordle_diagnostic_results.zip`;
`wordle_sft_results.zip` is neither touched nor regenerated.

```
results/
  unconstrained_sft/    three adapters, free generation
  constrained_sft/      three adapters, legal-word argmax
  base_qwen/            untrained Qwen, both modes
  comparison.csv / .md  one row per (model, mode)
  terminal_probe.csv    one row per probe state
  verdict.json
```

---

## Phase 5 results — verdict D

Ran on a Kaggle T4. No training. 246 held-out answers, both modes, plus the
base control and the terminal probe.

| Model | Mode | Mean | Fail | Invalid | Repeat |
|---|---|---:|---:|---:|---:|
| classical `entropy` | symbolic | **3.4431** | 0.0% | 0% | 0% |
| classical `random` | symbolic | **4.0203** | 0.8% | 0% | 0% |
| qwen_entropy | unconstrained | 6.3902 | 79.3% | 36.7% | 67.1% |
| qwen_tree_soare | unconstrained | 6.3211 | 78.5% | 36.6% | 64.6% |
| qwen_tree_salet | unconstrained | 6.1748 | 76.4% | 37.6% | 63.0% |
| base Qwen | unconstrained | 7.0000 | 100.0% | 1.5% | 100.0% |
| qwen_entropy | constrained | 5.9715 | 62.2% | **0.0%** | 36.2% |
| qwen_tree_soare | constrained | 5.9309 | 63.4% | **0.0%** | 37.8% |
| **qwen_tree_salet** | **constrained** | **5.6870** | **56.1%** | **0.0%** | 30.5% |
| base Qwen (n=62) | constrained | 7.0000 | 100.0% | 0.0% | 100.0% |

Constrained decoding did exactly what it was built to do — invalid words 37% →
0%, repeats 63% → 30.5%, failures 76% → 56%, a real gain of **0.49 guesses**.

**And the best constrained model still loses to random guessing with perfect
elimination by 1.67 guesses.** Vocabulary generation was a genuine contributor,
not the binding constraint. The Phase 4 hypothesis was wrong in emphasis.

## The terminal-state probe settles it

At `k=1` the visible constraints determine the answer uniquely. No search left,
spelling handled externally. 80 states, best model:

| | k=1 top-1 | median rank of answer |
|---|---:|---:|
| qwen_tree_salet | **20.0%** | **7.5** / 12,972 |
| base Qwen | 0.0% | 1388 / 12,972 |
| chance | 0.008% | 6486 |

Two findings at once:

- **SFT did something large and real.** It moved the answer from rank 1388 to
  rank 7.5, and base Qwen never gets a single one right in either mode. Verdict
  C is firmly excluded — SFT is not a marginal improvement over base.
- **20% top-1 on a state with exactly one possible answer is the failure.** The
  model reaches the right neighbourhood and cannot close. It degrades fast:
  k=2 → 6.7%, k=3 → 3.7%, with the answer's median rank falling to 18 then 80.

## The vocabulary hypothesis, honestly

| qwen_tree_salet, k=1 top-1 | seen as a training target | unseen |
|---|---:|---:|
| accuracy | 33.3% (n=21) | 15.3% (n=59) |

Directionally exactly as predicted — 2.2× better on words it was trained to
emit. But at those sample sizes the difference is **~1.6σ**: suggestive,
underpowered, not established.

The more damning number is the 33.3% itself. On answers the model was
*explicitly trained to output as the winning guess*, in states that uniquely
determine them, it fails two times in three. Held-out vocabulary cannot explain
that.

## What did transfer, and it is not nothing

| Model | Mode | agrees entropy | agrees tree |
|---|---|---:|---:|
| entropy | unconstrained | 90.09% | 0% |
| entropy | constrained | 90.18% | 0% |
| tree_soare | unconstrained | 0% | 89.91% |
| tree_soare | constrained | 0% | **93.75%** |

Openers 100% in both modes. Policy transfer is intact and slightly *better*
under constrained decoding. So the model reproduces its expert's early-game
policy almost perfectly and still loses 56% of games.

The failure is concentrated entirely in the endgame — precisely where the
training data had **6 fully-determined examples out of 7,173**.

## Verdict

```
VERDICT: D
  best SFT constrained mean   = 5.6870
  best SFT unconstrained mean = 6.1748 (constrained gains +0.4878)
  base Qwen constrained mean  = 7.0000
  classical entropy = 3.4431, classical random = 4.0203
  terminal k=1 top-1 [qwen_tree_salet] = 20.0% (chance 0.0077%)
  terminal k=1 top-1 [base_qwen]       = 0.0%
  -> constrained SFT still loses to random elimination;
     the failure is not primarily vocabulary
```

Read plainly: **not A** (vocabulary was a contributor, not the bottleneck),
**not C** (SFT beat base by a mile), and worse than B (it does not merely fall
short of the expert — it loses to random elimination).

---

---

# Phase 6 — endgame-heavy SFT

## A correction to the Phase 4/5 diagnosis

Earlier writeups said the endgame had "6 fully-determined examples out of
7,173". That conflated two different things. `fully_determined` means *all five
greens are known* — a much narrower condition than "the state determines the
answer". The real numbers for `tree_salet`:

| | |
|---|---|
| `n_candidates == 1` rows | **1,612** |
| distinct words covered | **1,612** |
| paths per word | **exactly 1.00, for every word** |
| `fully_determined` (all greens) | 4 |

So the endgame is not short of *states*. It is a 1,612-way retrieval task with
**exactly one example per class** — every winning move reachable by one
memorised route. That teaches a lookup table, not the procedure "read the
constraints, produce the word that satisfies them", which is what evaluation
demands on words whose single route was never shown.

This sharpens the fix: what is needed is **many distinct constraint paths per
word**, not simply more endgame rows.

## The intervention

`build_endgame_dataset.py` rolls out deliberately imperfect games — SALET
opener (so states are reachable at evaluation time), then randomised
continuations mixing surviving candidates and arbitrary legal words — and
labels each state with the `tree_salet` expert. Standard DAgger-style noise
injection: natural trajectories are on-policy and never show the states a
fallible model actually reaches.

Descent uses only cheap policies; the expert is called once per accepted state.
That matters — `choose()` costs 0.1–18 ms at 1–10 candidates but ~1.4 s at 40.
Whole dataset: **32 seconds**.

| | rows | k=1 rows | words | paths/word | turn-1 share |
|---|---:|---:|---:|---:|---:|
| natural | 7,067 | 1,612 | 1,612 | 1.00 | 29.3% |
| endgame synthetic | 12,145 | 6,148 | 2,068 | 2.97 | 0% |
| **mixed → training set** | **19,212** | **7,760** | **2,068** | **3.75** | **10.8%** |

Candidate buckets in the synthetic half: k=1 6,148 · k=2 2,041 · k=3 1,989 ·
k=4–10 1,967.

`verify_endgame_dataset.py` asserts, and all pass: no candidate count or list in
any prompt, greens never concatenated, no held-out answer used as a k=1 target,
every completion a legal guess, at k=1 the expert plays the surviving candidate,
prompt format byte-identical to Phase 4/5, and turn-1 share strictly below the
natural 29%.

One check needed refining: matching the answer against the *whole* prompt flags
AFTER and GUESS, which are real Wordle answers that also occur in the fixed
instruction text ("After each guess…", "Next guess:"). The same collision exists
in the Phase 4 data, which was independently audited clean. The check now scans
only the dynamic region.

## The hypothesis, and how it can fail

**Hypothesis:** many paths per word teaches constraint-following as a
procedure, so it generalises to held-out words; the constrained decoder supplies
the lexicon.

**Failure mode:** it may just memorise the training answers harder. Phase 5
measured 33.3% top-1 on seen words vs 15.3% on unseen. If Phase 6 moves seen
sharply and unseen barely, the intervention bought memorisation, not a
procedure — and the framing must change rather than the data scaling up. The
terminal probe reports that split directly and it is the number to read first.

## Design

One adapter, `tree_salet_endgame`, hyperparameters identical to Phase 4.
Evaluated as a **2×2** over {new, old} × {ban repeats, no ban}, because Phase
5's 5.6870 was measured with `ban=False` — without the no-ban cells the dataset
effect and the decoder effect would be confounded. `tree_salet` + `noban` also
serves as a reproduction check: it must return 5.6870.

`make_endgame_notebook.py` → `wordle_endgame_sft_kaggle.ipynb`. Every Phase 5
failure mode is fixed: result containers initialised before the loops (a single
interrupt cost the whole save step last time), `save_state()` after every
evaluation, every result printed as well as written, progress and ETA in every
long loop, and a results zip with no weights in it.

Budget ~3.5 h on a T4: train ~70 min, 2×2 eval ~60 min, baselines ~10 min,
capped terminal probe ~50 min.

---

## Phase 6 results — the intervention worked, and it was not enough

Ran on a T4. 19,212 rows, 1,202 steps, 64.5 min, loss 5.886 → 0.742.
**The Phase 5 reproduction check returned 5.6870 exactly**, so the comparison is
valid.

## Game outcomes: essentially unchanged

| adapter | ban repeats | mean | fail | solved | repeat |
|---|---|---:|---:|---:|---:|
| `tree_salet` (Phase 5) | no | 5.6870 | 56.1% | 108 | 30.5% |
| `tree_salet_endgame` | no | **5.6992** | 56.1% | **108** | 32.1% |
| `tree_salet` (Phase 5) | yes | 5.5122 | 46.3% | 132 | 0% |
| `tree_salet_endgame` | yes | **5.4797** | 44.7% | **136** | 0% |
| classical `random` | — | 4.0203 | 0.8% | 244 | — |

Effect decomposition:

```
endgame data alone (noban):  5.6870 -> 5.6992   +0.012   nothing
repeat banning alone (old):  5.6870 -> 5.5122   -0.175   the decoder
both:                        5.6870 -> 5.4797   -0.207
```

At the game level the endgame data did **nothing** — 108 solved before, 108
after, identical. Every game-level gain came from the decoder change. The
combined best still loses to random elimination by 1.46 guesses.

## But the terminal probe moved, and moved in the right place

Identical probe states, both adapters, same run:

| | k=1 top-1 | median rank | k=2 top-1 | k=3 top-1 |
|---|---:|---:|---:|---:|
| `tree_salet` | 20.0% | 7.5 | 6.67% | 3.70% |
| `tree_salet_endgame` | **27.5%** | **4.0** | **16.67%** | **7.41%** |

## The verdict cell was wrong, and the correction matters

The notebook printed **MEMORISED**, on the grounds that unseen-word accuracy
fell from 15.25% to 5.56%. That comparison is invalid: the seen/unseen partition
changed between phases. Phase 5 built its "seen" set from the three natural
datasets (21 seen / 59 unseen of the 80 probe states); Phase 6 built it from
`tree_salet` + endgame completions, whose broader vocabulary reclassified most
of the probe answers as seen (62 / 18). The rule compared a number against a
differently-defined number.

Recomputed under the **Phase 5 definition** — which reproduces Phase 5's figures
exactly for the old adapter, confirming the method:

| adapter | seen (n=21) | unseen (n=59) |
|---|---:|---:|
| `tree_salet` | 33.33% | 15.25% |
| `tree_salet_endgame` | **33.33%** | **25.42%** |

Paired McNemar on identical states:

| subset | n | salet | endgame | gained | lost | p |
|---|---:|---:|---:|---:|---:|---:|
| k=1 **unseen** | 59 | 9 | 15 | **6** | **0** | **0.031** |
| k=1 seen | 21 | 7 | 7 | 3 | 3 | 1.000 |

**Improvement is concentrated entirely on words the model was never trained to
emit — 6 gained, 0 lost, p = 0.031 — and is exactly zero on words it was.**
That is the signature of a transferable procedure, and the precise opposite of
the memorisation the verdict claimed. The rule fired on a bug, not on evidence.

## Why it did not show up in games

The candidate trajectory is identical across all four runs:

```
after g1  74.72   g2 3.62   g3 1.25   g4 1.09   g5 1.03   g6 1.02
```

The model reaches ~1.2 candidates by turn 3 — effectively solved — then spends
three turns failing to name the word. Game outcome is therefore governed almost
entirely by the k=1 hit rate. At 27.5% per attempt with banning, three attempts
give 1 − 0.725³ ≈ 62%, against the observed 55% solve rate. The arithmetic
closes: **the endgame hit rate is the whole game.** Moving it 20% → 27.5% is
real and far too small.

## The next lever, quantified

Hard-mode violations are **31%**: a third of guesses contradict feedback the
model has already received. And at the k=1 probe states, the set of legal words
consistent with the *revealed feedback* has a median size of **2** (mean 3.4,
p90 7) versus the full pool of 12,972 — a **6,486x** reduction.

That constraint set is derivable from the prompt alone. It is what hard-mode
Wordle enforces and what a human player sees; it is not the answer list and not
privileged information. Restricting the decoder to it is the same class of move
as restricting to legal words, and the model currently ranks the answer 4th out
of 12,972 while a median of only 2 words are even admissible.

**Phase 7 should be a hard-mode constrained decoder, not more data.** It costs no
training, and the measurement above says the headroom is large.

---

---

# Phase 7 — feedback-consistent decoding

Phase 6 said the decoder was the lever, on two measurements: the model violated
its own known constraints **31%** of the time, and at k=1 only a **median of 2**
legal words were consistent with the revealed feedback against a pool of 12,972.

Phase 7 retrains `tree_salet_endgame` (identical Phase 6 config, so the numbers
are comparable) and runs four decoders on the same 246 answers.

## The four decoders

| decoder | admissible set |
|---|---|
| `unconstrained` | anything the model emits |
| `legal` | the 12,972 legal guesses |
| `consistent` | legal **and** consistent with all feedback, every turn |
| `adaptive` | consistent only once the admissible set is <= 50 |

A word is admissible iff it would have produced exactly the feedback already
observed:

```python
allowed = [w for w in allowed if feedback_code(guess, w) == observed_code]
```

**This is not the candidate set.** The candidate set is answers consistent with
feedback and uses the 2,315-word answer list — privileged, never used here. The
hard-mode set is *legal guesses* consistent with feedback: a pure function of
the prompt plus the public word list, exactly what a hard-mode player computes
from their own board.

## Why `adaptive` exists

Not a hunch — a measurement. The expert's own training target is
feedback-**inconsistent** most of the time early on:

```
turn 2:  40.6% consistent   <- the expert PROBES with a word that cannot win
turn 3:  91.8% consistent
turn 4: 100.0% consistent
```

An always-on filter forbids the expert's turn-2 policy in ~59% of games. That is
the known reason hard mode scores worse than free mode, so `adaptive` probes
freely while uncertainty is high and becomes consistent once it is low.

## Forced vs model-chosen

At k=1 states the filter alone leaves exactly one word **38.9%** of the time —
the decoder has solved the game and the model contributed nothing. Every
decision is therefore tagged `forced` or `model_chosen`, and a **no-model
control** (pick uniformly at random among admissible words) is run separately.
Without both, a decoder win would masquerade as a model win.

## Phase 7 results — the decoder was the bottleneck

Retrained `tree_salet_endgame` and it **reproduced Phase 6 exactly**: 1,202
steps, final loss 0.7410 vs 0.7419, unfiltered k=1 top-1 27.5% and median rank
4.0 — identical. Same model, four decoders.

| decoder | mean | fail | solved | invalid | repeat | hard-mode viol |
|---|---:|---:|---:|---:|---:|---:|
| unconstrained | 5.9634 | 66.3% | 83 | 21.2% | 43.1% | 25.3% |
| legal | 5.6951 | 55.7% | 109 | 0% | 30.9% | 29.4% |
| legal + ban | 5.4837 | 44.3% | 137 | 0% | 0% | 31.1% |
| **consistent** | **3.9472** | **2.0%** | **241** | 0% | 0% | **0%** |
| **adaptive** | **3.7846** | 2.9% | 239 | 0% | 0% | 6.8% |
| — classical `random` | 4.0203 | 0.8% | 244 | | | |
| — classical `frequency` | 3.7927 | 1.6% | 242 | | | |
| — classical `entropy` | 3.4431 | 0% | 246 | | | |

**The feedback-consistency filter is worth ~1.5 guesses** — far more than
everything else in Phases 4–6 combined. The system now beats classical `random`
and edges past `frequency`, and sits **0.34** off the best classical solver.

```
unconstrained -> legal        5.9634 -> 5.6951   -0.27
legal -> legal+ban            5.6951 -> 5.4837   -0.21
legal+ban -> consistent       5.4837 -> 3.9472   -1.54   <- the whole story
consistent -> adaptive        3.9472 -> 3.7846   -0.16
```

## The prediction held

The adaptive decoder was added because the expert's own target is
feedback-**inconsistent** 59.4% of the time at turn 2 — it probes with a word
that cannot win. An always-on filter forbids that, which is the known reason
hard mode scores worse than free mode. Filtering only once the admissible set
is ≤ 50 recovers **0.16 guesses**, and the hard-mode violation rate rises from
0% to 6.8% exactly as it should: the model is probing again.

## How much is the model and how much is the filter?

The run reports 40.2% of wins as `forced` (one admissible word, no model
involvement). That is necessary but not sufficient, so the missing control was
computed separately: **play the same 246 games picking uniformly at random
among admissible words, no model at all.**

| | no model | with model | model contributes |
|---|---:|---:|---:|
| consistent | 4.5203 ± 0.02 | 3.9472 | **−0.573** |
| adaptive | 5.0027 ± 0.02 | 3.7846 | **−1.218** |

The filter alone gets 4.52. The model is doing real work on top of it — and
**much more in adaptive mode**, because there its learned probing policy (90%
expert agreement at turn 2) replaces a random probe. A better model is worth
more under the adaptive decoder than under the always-on one.

## An elegant self-check

`consistent/ban` and `consistent/noban` returned **identical** results down to
the last game. That is correct and worth stating: under the consistency filter a
repeat is impossible by construction, since a previously played word is
inconsistent with its own non-winning feedback. Repeat-banning becomes a no-op,
which is independent confirmation the filter is applied correctly.

## What did not change

Unfiltered k=1 top-1 stayed at 27.5%, median rank 4.0 — identical to Phase 6.
Decoding does not alter model capability, and the probe correctly reports that.
Every gain here is the decoder plus the model's existing policy being allowed to
express itself.

---

# Phase 7b — adaptive-threshold sweep

Phase 7's threshold of 50 was a guess. This sweeps it, and runs the no-model
control at every point so the model's margin stays separable.

## Validation first

Both endpoints reproduce Phase 7 exactly, from a separate run:

| | sweep | Phase 7 |
|---|---|---|
| threshold 0 (never filter) | 5.6951 | `legal/noban` **5.6951** |
| threshold 1e9 (always filter) | 3.9472 | `consistent` **3.9472** |

## Results

| threshold | mean | solved | fail | hard-mode viol | model gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.6951 | 109 | 55.7% | 29.4% | 1.30 |
| 2 | 4.1707 | 228 | 7.3% | 16.1% | **1.79** |
| 5 | 3.8618 | 240 | 2.4% | 12.1% | 1.70 |
| 10 | 3.7764 | **243** | **1.2%** | 10.2% | 1.58 |
| **20** | **3.7642** | 242 | 1.6% | 9.1% | 1.41 |
| 50 *(Phase 7)* | 3.7886 | 239 | 2.9% | 6.8% | 1.21 |
| 100 | 3.8171 | 239 | 2.9% | 4.5% | 1.07 |
| 250 | 3.8943 | 239 | 2.9% | 2.4% | 0.83 |
| 1000 / 1e9 | 3.9472 | 241 | 2.0% | 0.0% | 0.57 |

`model gain` = control mean − model mean, i.e. how far the model beats a random
pick among admissible words at that threshold.

**Best mean 3.7642 at threshold 20; best solve rate 243/246 (1.2% failure) at
threshold 10.** Against classical `frequency` 3.7927 and `entropy` 3.4431.

## Two things worth not over-reading

**It is a plateau, not a peak.** SE of the mean is ~0.064 guesses, and
thresholds 10 / 20 / 50 / 100 all fall within 1 SE of one another. Threshold 20
is not meaningfully better than 10 or 50. What *is* outside noise is that both
extremes are worse — threshold 0 by 30σ and 1e9 by 2.9σ.

**Thresholds ≥ 769 are all the same thing.** The admissible set never exceeds
769 after turn 1 (max, measured), so any threshold above that filters at every
turn. 1000 and 1e9 producing byte-identical results is correct behaviour, not a
bug; anything else would be the bug.

## The model's share shrinks as the decoder's grows

`model gain` decays monotonically, 1.79 → 0.57, as filtering increases. More
decoder means less model. At the plateau it sits at ~1.4–1.6, so the model is
still doing real work there — a better place to operate than the high-threshold
end, where the filter does nearly everything and the model contributes 0.57.

Eval time also falls 805s → 268s across the sweep, confirming the scan-pool fix
(a 3-word admissible set previously cost more than an unfiltered decision
because both pushed a full 512-row chunk through the model; now 209x faster).

---

# Phase 8 — counterfactual headroom, then DPO

Two experiments. The first was built as a stop-gate: measure where the remaining
0.32 guesses actually live before spending a phase trying to close them.

## Counterfactual headroom

Hand one decision regime at a time to the classical expert and replay. Buckets
are keyed on |admissible| — the number of legal words consistent with the
feedback — because that is what the adaptive decoder actually presents.

This is a **hybrid-policy evaluation**, not a replay: substituting an action
changes the feedback and every later state, so the model has to be in the loop.

| expert acts in | mean | solved | recovered | % of gap |
|---|---:|---:|---:|---:|
| baseline (model everywhere) | 3.7642 | 242 | — | — |
| **2–10 admissible** | **3.5244** | **246** | **0.2398** | **74.7%** |
| 11–100 admissible | 3.6707 | 244 | 0.0935 | 29.1% |
| 100+ admissible | 4.1626 | 231 | **−0.3984** | −124.1% |
| combined (expert everywhere) | 3.4431 | 246 | 0.3211 | 100.0% |

The `combined` row returning **exactly** classical `entropy` (3.4431) is a
definitional check that the harness is wired correctly — expert everywhere *is*
the expert.

**Three quarters of the entire gap lives in states with 2–10 admissible words.**
Perfect play there alone would solve 246/246.

### The negative row is the interesting one

Handing the expert the **100+** bucket makes things *worse* by 0.40. That looks
impossible until you see what the bucket is: |admissible| > 100 means turns 1–2.
So this run has the entropy expert play the opening (SOARE) and the model play
from turn 3 on — and the model was trained on `tree_salet` trajectories, which
open SALET.

It is a **policy-mismatch** cost, not evidence the model out-plays the expert.
The handoff drops the model into states its training never covered. The same
effect shows up in the arithmetic: the three buckets sum to −0.065 while the
combined run recovers +0.321. Strongly non-additive, because the policies
interact at every handoff.

## DPO

14,923 preference pairs, β=0.1, lr 5e-6, 1 epoch, 466 steps, 44 min. Reference
model is the SFT policy (merged, fresh LoRA on top, adapter-disabled as ref).

**Training worked:**

```
loss    0.6922 -> 0.3988
margin  +0.018 -> +11.128
acc      0.604 ->   0.836
```

**Gameplay regressed:**

| | mean | solved | fail | hard-mode viol | model contribution |
|---|---:|---:|---:|---:|---:|
| SFT | **3.7642** | **242** | 1.63% | 9.11% | **+1.4120** |
| DPO | 3.9350 | 240 | 2.44% | 9.25% | +1.2412 |

Paired on the same 246 answers:

```
games changed 108   DPO better 38   DPO worse 70
paired t = -3.21    SIGNIFICANT
failures    SFT 4   DPO 6
```

Not noise. The most common movements were `3 -> 4` (28 games) and `4 -> 5` (22),
against `4 -> 3` (18) — the model is systematically taking one guess longer.

### Why it failed

The margin climbed to **+11** while accuracy plateaued at **0.836 by step 75**
and never moved again. That combination is diagnostic: the model spent 390 steps
growing more confident about pairs it *already ranked correctly*, not fixing the
ones it got wrong.

Which pairs were those? **9,556 of 14,923 (64%) are `clear` pairs with a median
cost gap of 3.07** — comparisons like RUMBO (4.48) versus AVERT (25.07), where
the right answer is obvious. They deliver large margins cheaply and dominate the
gradient. The `competitive` pairs, median gap 0.50, are where the actual signal
is, and they are the minority.

So DPO optimised a proxy it had already mastered, drifted from the SFT policy to
do it, and paid for the drift in real play. Model contribution fell 1.412 →
1.241, confirming the model itself got worse rather than the decoder changing.

<span></span>

**Verdict: the first DPO run is a clean negative.** The preference objective was
learned and did not transfer. That is worth knowing before GRPO, which optimises
against the same style of proxy with far more compute and less stability.

### What would be worth trying, if anything

1. **Competitive pairs only.** Drop the 64% easy majority; keep gaps under ~1.0.
2. **Stop far earlier.** Accuracy saturates at step 75 of 466; everything after
   is drift. Early-stop on *gameplay*, not loss.
3. **Lower β or fewer epochs** to keep the policy near the SFT reference.

But the honest reading is that the headroom analysis already told us where the
gap is (2–10 admissible words, 74.7%), and one targeted preference run aimed
squarely at it made things measurably worse. That is evidence about the approach,
not just the hyperparameters.

---

# Phase 8 v3 — attempted, void, retry prepared

The v3 dataset (audited, 6,000 pairs, real validation split) trained cleanly:
loss 0.681 → 0.557, margin +0.32 → +5.11, train accuracy 0.525 → 0.716 over
187 steps.

**The run is not interpretable.** Its SFT baseline returned **4.8862 / 210
solved** where Phase 7 — same adapter, same decoder, same threshold — measured
**3.7642 / 242**. A 1.12-guess discrepancy on a control that should reproduce
exactly means the session was measuring something other than what it claimed,
so the DPO row is void along with it.

## Cause, and the fix

`find_adapter` fell back to **any** adapter when the requested name was absent.
With a DPO adapter in the attached dataset, that silently loaded as the SFT
base — training and evaluating against the wrong model with no visible symptom.
The lookup is now strict: it prints what it found and refuses to substitute.

This is the second silent fallback in the project to produce a plausible wrong
answer (the first masked a 19% data-corruption rate behind a sampled count).
Both were introduced for convenience; both cost a session.

## What was added so it cannot recur

- **Preflight (§5b)** verifies the model by *behaviour* before any long run:
  the opener must be SALET, and a 40-game sample must reproduce 3.7642 within
  0.50. It asserts, so a wrong setup stops in ~1 minute rather than an hour.
- **fp16 and sdpa are asserted**, not assumed. `torch_dtype` is deprecated in
  transformers 5.x and a silent fp32 load costs ~8x on a T4; `eager` attention
  costs another 5–10x. Neither is visible except as a slow run.
- **A decision cache**, exact rather than approximate: the model is
  deterministic and the decoder greedy, so identical prompts reuse the result.
  Turn 1 collapses from 246 decisions to 1.

## Still unexplained

Unfiltered decisions ran at ~21 s against Phase 7's ~0.6 s — a separate problem
from the adapter, and not yet diagnosed. `kaggle_cells/CELL_debug_slow_eval.py`
times each stage of one decision (prompt pass, cache expansion, chunk scoring,
filter refinement) to locate it instead of guessing. Preflight now reports
per-decision time and warns above 3 s.

---

# Phase 9 — is the harness the model?

Prepared, not yet run. `phase9_harness/wordle_phase9_harness_kaggle.ipynb`,
generated by `make_harness_notebook.py`. Instructions: `docs/RUN_PHASE9.md`.

## Why this and not GRPO

Four interventions after Phase 7 have moved the mean by less than the noise:

| | mean | verdict |
|---|---:|---|
| Phase 6 SFT + decoder | 3.7886 | — |
| Phase 7 (endgame data) | 3.7642 | marginal |
| Phase 7b (threshold sweep) | 3.7642 | flat frontier |
| Phase 8 DPO v1 | worse, t = −3.21 | harmful, bad dataset |
| Phase 8 DPO v3 | 3.7886, t = −0.69 | clean null |

Against that, the decoder alone was worth about 1.5 guesses. The measured
lesson of this project so far is that scaffolding dominates training on this
task at this model size — and exactly one piece of scaffolding has never been
varied: the prompt.

Starting GRPO now would be a fifth attempt at the half that has not moved,
before measuring the half that has.

## Design

Twelve prompt variants × {base Qwen, SFT} × the same answers, decoder pinned at
adaptive@20 so nothing below is attributable to anything but the text. Shared
answers make every variant-vs-baseline comparison **paired**, which matters: the
unpaired SE on 100 games is ~0.10 and would hide every effect this is looking
for.

A second, cheap probe runs the same variants with **no decoder at all** — raw
greedy generation — and reports legal-word and admissible rates. A prompt can
improve what the model emits without changing the game mean, because the decoder
was already repairing it. Those are different findings and are measured
separately.

## The load-bearing variant

`raw_history`. Every phase so far has fed the model `Confirmed letters : p1=A`,
computed by the classical solver. That is the solver doing the deduction and
handing over the answer to the hard part. If the model plays as well without it,
the block was decoration. If it collapses, then a large share of what this
write-up has been calling "the model playing Wordle" is the harness playing
Wordle, and the honest headline is different.

Either result is worth having. Only one of them is comfortable.

## Leakage

One variant, `with_count`, shows the surviving-candidate count. That violates the
standing rule for every other artefact in this project — the model never sees the
answer list, the candidate set, or the candidate count. It is marked `leaky`,
excluded from the reported spread, printed with a `*LEAKY*` flag, and exists only
to bound "what would that hint be worth?". It must never be quoted as a result or
used to select a training format.

## Two bugs caught before the run

- The few-shot exemplars were hand-written and listed `B` as an absent letter in
  a game where `B` appeared in neither guess — a worked example teaching a false
  deduction. They are now computed by the same `derive_constraints` /
  `feedback_code` used by the live prompts, so an exemplar cannot disagree with
  the game it illustrates, and the notebook asserts each exemplar word is legal
  and consistent with its own feedback.
- `green_pattern` uses `.` for unknown positions, not `""`. Two variants
  filtered on truthiness and rendered `position 1 is .`. Fixed and covered by
  the render check in cell 4.

The whole pipeline was dry-run locally against a stub scorer — all twelve
variants, the decision cache, the hard-mode filter, the paired statistics and
the verdict block — before any GPU time was spent.

---

# Phase 10 — the format crossover

Phase 9 measured a **+0.516** penalty for `raw_history` (the prompt minus the
solver-derived constraint block) and read it as evidence the harness was doing
deduction the write-up credits to the model. That reading was not earned: the
adapter had only ever seen `baseline`, so **format lock-in predicts the same
table**. Phase 9's stock-Qwen arm was meant to break the tie and could not — a
model that cannot play Wordle under any prompt cannot rank prompts.

Breaking it needed a second *trained* adapter. `tree_salet_endgame_rawhist` is
the **same 19,212 rows** with only the prompt re-rendered: identical LoRA, lr
2e-4, 2 epochs, batch 4x4, seed 20260817. It landed on **1,202 steps**, exactly
Phase 6's, and the re-render was proved lossless first — every row round-trips
back to its stored prompt byte-for-byte (`ok=19,212  mismatch=0`).

## The square

246 held-out answers, decoder fixed at adaptive@20, every cell opens SALET.
`sft`+`baseline` reproduced **3.7642** exactly, which is the control that makes
the rest interpretable.

|  | eval `baseline` | eval `raw_history` |
|---|---|---|
| trained `baseline` (`sft`) | **3.7642** · 242/246 | 4.2805 · 232/246 |
| trained `raw_history` (`sft_rawhist`) | 4.1098 · 227/246 | **3.8089** · 240/246 |

Paired on identical answers:

```
diagonal - each adapter on its own format   +0.0447   t=+1.08   NOT significant
sft moved off its format                    +0.5163   t=+7.07   SIGNIFICANT
sft_rawhist moved off its format            +0.3008   t=+4.92   SIGNIFICANT
```

**Outcome A (lock-in), as pre-registered.** Each adapter is best on its own
format; both off-format penalties are significant; `rawhist` on `raw_history`
lands at 3.8089, inside the 3.8-3.9 band written down before the run.

**Outcome C is refuted.** C predicted `rawhist` would stay near 4.28 on
`raw_history` despite training on it. It recovered to statistical parity with
baseline. So **Phase 9's +0.516 was mostly lock-in, not a property of the
format** — a correction to the Phase 9 reading.

## The probe disagrees with the score, and both are right

Decoder **off**, 148 stratified states:

| arm | eval format | parse% | legal% | admissible% |
|---|---|---:|---:|---:|
| `sft` | baseline | 99.3 | 84.5 | **22.3** |
| `sft` | raw_history | 100.0 | 95.9 | 1.4 |
| `sft_rawhist` | baseline | 99.3 | 93.2 | **6.1** |
| `sft_rawhist` | raw_history | 100.0 | 99.3 | **0.0** |

`sft_rawhist`, on the format it was trained on, emits **0.0% admissible words
unaided** — and still plays 3.8089. It reaches baseline-equivalent gameplay
having learned essentially no feedback consistency. On matched `baseline`
prompts it manages 6.1% against `sft`'s 22.3%.

Two true statements that point opposite ways:

- **On final score**, the constraint block is replaceable. Train without it and
  the decoder covers the difference.
- **On what the model knows**, the block is what does the teaching. Removing it
  costs roughly all of the model's unaided deduction.

So the "is the harness the model?" worry is not dissolved — for `sft_rawhist`
it is **sharpened**. That adapter is close to a pure decoder-driven player, and
the original `sft` at 22.3% admissible is the one doing more of its own work.
What this rules out is the stronger claim that the block encodes deduction a
0.5B model cannot learn at this scale.

**Not claimable from this:** that either format is intrinsically better. The
diagonal is a tie.

---

# Phase 11 — the decision budget, and what is left for RL

Before choosing a method to close the last **0.3211**, `decision_budget.py`
prices where that gap can and cannot be recovered. It takes the model's own
**922 decisions** across the 246 games and prices each against the best action
at *that exact state*. Phase 8's counterfactual answered the same question by
substituting the expert and replaying, which is non-additive — its three
buckets summed to −0.065 against a combined +0.321. This is additive and
attributable.

## Where the decisions are

```
|adm| 0-1    forced/solved      11.4%
|adm| 2-10   restricted         25.4%
|adm| 11-20  restricted          5.3%
|adm| 21-100 UNrestricted       11.0%
|adm| 100+   UNrestricted       47.0%   <- turn-2 probes
```

**58% of the model's decisions are unrestricted**, and nearly half are turn-2
probes where it picks freely from 12,972 words.

## What perfect action selection is worth

Exact adaptive-decoder tree, every restricted decision priced:

```
restricted decisions with a real choice   153   (0.62 per game)
model already optimal                     131   (85.6%)
  ...where the optimum was a TIE          127   (83.0%)
genuine mistakes                           22   (14.4%)

expected guesses lost                  0.0698 per game
```

**Fixing every restricted-regime mistake — perfectly, with zero drift — buys
0.0698 guesses.** 3.7642 → 3.6944, or 22% of the gap. Phase 8's DPO drift cost
**0.1708**: the downside was 2.4x the entire available upside.

That killed the planned DPO retry. The corrected dataset (`phase8_dpo_v3/`,
6,000 pairs, all hard checks passing) was built, audited and **never trained
on**, because measuring the ceiling first showed it could not win. The same
measurement rules out on-policy mistake mining independently: 0.089 mistakes
per game over 2,069 answers is ~185 pairs, and 83% of the decisions the model
gets right were ties it could not have got wrong.

## Two structural facts this exposes

**The optimum is usually a set.** 83% of correct restricted decisions were
ties, and 89.5% of the built GRPO tasks have more than one exactly-optimal
action. A pairwise preference cannot represent that; group-relative advantage
can, which is the substantive reason to prefer GRPO here over DPO.

**The unrestricted regime is not exactly priceable.** One lookahead valuation
of a single turn-2 state costs ~13.5s at 83 candidates and grows sharply; a
10-decision sample did not finish in 30 minutes. That is why Phase 11 rewards
that regime with a one-ply proxy and labels every such task `exact: false`.

## The run, and what it settled

GRPO trained on 3,150 exactly-rewarded states, 393 steps. The proxy moved the
right way — held-out optimal-action rate **67.6% → 68.9%** — and every state
carried gradient signal (`zero_adv` 0.00 throughout, vindicating the decision to
drop std-normalisation given 89.5% of states have tied optima).

Gameplay did not move. 246 held-out answers, decoder fixed, `sft` + `baseline`
control reproducing **3.7642** exactly:

| arm | mean | solved | fail |
|---|---:|---:|---:|
| `sft` | **3.7642** | 242/246 | 1.63% |
| `sft_grpo` | 3.7602 | 241/246 | 2.03% |

```
diff -0.0041   paired t = -0.28   NOT significant
better 7   worse 6   unchanged 233
```

**233 of 246 games were identical.** The change is 6% of the 0.0698 ceiling and
inside noise. Hard-mode violations and forced-move rate did not move either.

The run was executed twice — interactively and headless through the API — and
the two adapters are **byte-identical**. The pipeline is deterministic.

### The stop rule was tested, not assumed

RUN.md pre-registered early-stopping on the paired rollout rather than on loss.
The API re-run recovered the mid-training checkpoints, so all three were scored
on the same 246 answers:

| arm | mean | solved | vs `sft` | t | games changed |
|---|---:|---:|---:|---:|---:|
| `sft` (control) | **3.7642** | 242 | — | — | — |
| `sft_grpo_150` | 3.7602 | 242 | −0.0041 | −0.38 | 7 |
| `sft_grpo_300` | 3.7520 | 241 | −0.0122 | −0.90 | 11 |
| `sft_grpo` (393) | 3.7602 | 241 | −0.0041 | −0.28 | 13 |

None is significant. Step 300 has the lowest mean, but at t = −0.90 across
three comparisons that is selection on noise, so it is reported and not
adopted. The only real pattern is `games changed` rising 7 → 11 → 13 while the
mean stays flat: the policy moves and buys nothing. **Outcome C holds for the
whole trajectory, not just its endpoint.**

### Two methods, opposite failure modes, one conclusion

| | what happened | effect |
|---|---|---:|
| Phase 8 DPO | drifted from the SFT policy | **−0.1708** |
| Phase 11 GRPO | stayed put | **+0.0041**, n.s. |

The decision budget predicted both before either ran. Perfect restricted-regime
action selection is worth 0.0698 guesses, and the model was already optimal in
85.6% of those decisions — 83% of the remainder being ties it could not have got
wrong. DPO had less upside than its own drift; GRPO had upside it could not
find because there was almost none left.

**The remaining 0.32 is not an action-selection problem.** It is the capability
limit measured back in Phases 4–6: given a state with exactly one possible
answer, the best model names it 20% of the time. Ranking cannot fix an
inability to produce.

### What this does not establish

The unrestricted regime was trained against a **one-ply proxy** over a top-48
menu, not the full vocabulary — so this run could teach the model to rank good
turn-2 probes but not to avoid bad ones. "GRPO cannot help at turn 2" is not
shown; only "this run did not".

Design and pre-registered readings: [phase11_grpo/RUN.md](phase11_grpo/RUN.md).
Results: `results/phase11/`.

---

# Where this leaves it

The distillation worked and was not enough. The models reproduce their expert's
opener 100% of the time and its turn-2 policy 90%, narrowing 2,315 candidates to
3.19 — expert parity — and SFT moves the answer from rank 1388 (base Qwen) to
rank 7.5 out of 12,972 in states that uniquely determine it. But given a state
with exactly one possible answer, free spelling, and nothing left to decide, the
best model names it 20% of the time, and 33% even when it was trained on that
exact word. Removing invalid words entirely bought 0.49 guesses and left the
model losing to random elimination. The bottleneck is not vocabulary and not the
decoder: it is that a 3.46-guess expert almost never visits the endgame, so
distilling it supplies thin endgame coverage — 1,612 of 7,173 rows sit at
1.00 paths per word.

(An earlier draft said "six fully-determined examples out of 7,173". That
conflated the all-greens flag with endgame coverage and understated the real
figure by two orders of magnitude. The conclusion survives — coverage is thin
and skewed early — but the number was wrong and is corrected here.)

**Phases 8–11 then established what the last 0.32 is not.** Two post-training
methods were tried against it, with opposite failure modes and the same answer:
DPO drifted and lost 0.1708; GRPO held its ground and gained 0.0041 (n.s.,
233/246 games unchanged). Phase 11's decision budget explains both in advance —
perfect action selection across the entire restricted regime is worth **0.0698
guesses**, because the model is already optimal in 85.6% of those decisions and
83% of the rest are ties it could not have got wrong.

So the gap is not preference, not ranking, and not the prompt format either:
Phase 10's crossover showed the `raw_history` penalty was mostly format lock-in,
and that a model can reach 3.76-equivalent play without the constraint block —
but only by leaning harder on the decoder, emitting **0% admissible words
unaided** on its own format.

What is left is the capability limit Phases 4–6 measured directly, and nothing
since has moved it: given a state with exactly one possible answer, the best
model names it 20% of the time. Better action selection cannot fix an inability
to produce the right word, and three phases of trying is reasonable evidence
that the next lever is a better model or better endgame coverage, not a better
objective.

---

---

## Layout

Organised by phase (`organize_by_phase.py`). The library lives in `core/` but is
still imported by **bare top-level name** — `from wordle_solver import ...` —
because `prepare_kaggle_dataset.py` copies those modules into `code/` inside the
Kaggle dataset and the notebooks import them that way. Renaming them to
`core.wordle_solver` would break every notebook on Kaggle.

Each phase folder carries a small `_paths.py`:

```python
import _paths   # puts core/ + root on sys.path, chdir to root
```

The `chdir` is what lets every script keep its relative data paths
(`"artifacts"`, `"sft_package/..."`) untouched. Only entry-point scripts import
it; the library modules never do. `conftest.py` does the same for pytest.

```
core/                       shared library, imported everywhere
  wordle_solver.py            solver + feedback engine
  tree_search.py              lookahead solver
  benchmark.py                evaluation harness
  generate_trajectories.py    prompt rendering + expert rollouts
  constrained_decode.py       exact legal-word argmax (Phase 5+)

phase1_classical/           the classical solver and its ablations
  build_artifacts.py  run_full_benchmark.py  run_tree_benchmark.py
  tiebreak_experiment.py  hybrid_sweep.py  verify_salet_tree.py
  baselines_val246.py  todays_wordle.py
  make_notebook.py -> wordle_classical_solver.ipynb

phase2_trajectories/        expert rollouts -> step records
  merge_trajectories.py  rerender_prompts.py  consolidate_results.py

phase3_sft_package/         step records -> SFT package
  build_sft_package.py  build_weighted_dataset.py
  verify_no_leakage.py  inspect_examples.py  power_analysis.py

phase4_5_sft_diagnostic/    train 3 adapters, then the constrained diagnostic
  make_sft_notebook.py -> wordle_qwen_sft_kaggle.ipynb
  rebuild_diagnostic_results.py

phase6_endgame/             endgame-heavy SFT (runs next)
  build_endgame_dataset.py    synthesise many paths per word
  verify_endgame_dataset.py   leakage + policy audit
  audit_endgame_dataset.py    deep audit: re-derives each state from its prompt
  make_endgame_notebook.py -> wordle_endgame_sft_kaggle.ipynb

phase7_constrained/         feedback-consistent decoding
  make_phase7_notebook.py     retrain + 4 decoders, resumable
  make_sweep_notebook.py      adaptive-threshold frontier sweep

tools/                      cross-phase utilities
  prepare_kaggle_dataset.py  validate_notebook.py  organize_project.py

data/                       source word lists - never modified
artifacts/                  generated solver bundle
sft_package/                SFT data, incl. data/tree_salet_endgame/
tests/                      63 tests
trajectories/  trajectories_with_count/
docs/                       RUN_DIAGNOSTIC.md, RUN_PHASE6.md
logs/                       run logs from every phase
kaggle_cells/               paste-ready notebook cells (Phase 5 patches)
uploads/                    Kaggle dataset staging
results/                    output from every phase
  phase6/  phase7/            results.json, adapter zip
uploads/phase7_adapter/     the trained adapter, extracted and reusable
```

Verified after restructuring: 63 tests pass, all three notebook generators run,
the Kaggle packaging reproduces the audited dataset byte-for-byte, and the
Phase 3 and Phase 6 audits pass.
