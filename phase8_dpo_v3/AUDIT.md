# Phase-8 DPO audit — the v1 dataset and pipeline

Every number below is printed by `phase8_dpo_v3/audit_legacy_dpo.py` from the
shipped file `sft_package/data/dpo_midgame/train.jsonl`. Nothing is taken from
v1's own metadata on trust: each of the 9,893 distinct states is re-derived from
its rendered prompt, and every label is re-scored twice — once with v1's own
cost function, once with an exact value function.

---

## Bottom line

**v1's preference labels are not backwards, and its prompts do not leak the
answer.** Both are worth stating up front, because both were plausible causes
and neither is the cause. Re-scored against an exact adaptive-decoder tree,
v1's labels in the high-value regime are correct on 99.6% of rows and backwards
on **zero**.

What is wrong is almost everything else. In order of how much I think each
contributed to the regression:

| # | Problem | Scale |
|---|---|---|
| 1 | v1's difficulty axis does not measure difficulty | `competitive` pairs have a median **true** gap of a full guess |
| 2 | 75.9% of `competitive` pairs are separated by nothing but an arbitrary constant | 4,072 / 5,367 rows |
| 3 | The budget is aimed away from the headroom | 33.5% of rows in the bucket holding 74.7% of the gap; 20.1% in the bucket measured at **−124%** |
| 4 | 57% of pairs are winnable by a feedback-consistency check alone | +44.5 pt asymmetry on 8,479 unrestricted rows |
| 5 | One state contributes 177 rows | 6,668 rows sit in repeated states; 1,867 ids are duplicates |
| 6 | No validation split exists | the reported 0.604 → 0.836 is **training** accuracy |
| 7 | Held-out answers appear as the *preferred* action | 694 rows |

---

## 1. The `competitive` pairs are not competitive

This is the finding that reframes the whole run, and it contradicts the
remedy the project README proposes.

v1 ranks actions by

```
cost(w) = E[remaining candidates after w] − 0.5 · (w is a candidate)
```

and calls a pair `competitive` when that gap is small. Re-scoring the same pairs
with an **exact** value function — the finite game the adaptive decoder actually
plays, solved to full remaining depth in integer arithmetic — gives this:

All 4,999 rows in that regime, over 4,872 distinct states:

```
chosen truly better :   4980  (99.6%)
exactly equal value :     19  (0.4%)
BACKWARDS           :      0  (0.0%)
```

| v1 pair type | rows | median **true** gap | p90 | truly close (≤ 0.25 guesses) |
|---|---:|---:|---:|---:|
| `clear` | 1,500 | 1.75 guesses | 4.67 | **0.0%** |
| `competitive` | 3,499 | **1.00 guesses** | 7.00 | **4.1%** |

A median true gap of **one full guess** is not a close decision, and only 4.1%
of the pairs v1 labels `competitive` are actually within a quarter-guess. v1's
`competitive` label selects pairs whose *proxy* gap is small, and the proxy gap
turns out to be uncorrelated with value — a `competitive` pair has a p90 true
gap of 7.0 guesses, which is a lost game.

The consequence is direct: PROJECT_README's first recommendation — *"Competitive
pairs only. Drop the 64% easy majority; keep gaps under ~1.0"* — **would not have
worked.** It filters on the axis that does not measure difficulty. Keeping only
v1's `competitive` pairs would have retained 3,499 rows of which **95.9% are not
close decisions**, and discarded nothing that was making the run easy. The
dataset does not contain a reservoir of hard pairs waiting to be isolated; by
true value it is ~96% easy, not 64%.

## 2. Where the `competitive` gap actually comes from

Decomposing the 5,367 `competitive` pairs into `E[remaining]` and the `0.5`
bonus:

```
pairs whose chosen and rejected have IDENTICAL E[remaining]   4,072   (75.9%)
  -> the entire recorded preference IS the 0.5 constant
pairs where the REJECTED action is strictly better on E[rem]    336    (6.3%)
  -> preferred only because chosen happens to be a candidate
(chosen_is_candidate, rejected_is_candidate) = (True, False)  4,685   (87.3%)
```

In the two highest-value buckets it is total:

| bucket | `competitive` rows | identical `E[remaining]` |
|---|---:|---:|
| 2–3 admissible | 1,500 | **100.0%** |
| 4–10 admissible | 1,999 | **91.0%** |

So in the regime carrying three quarters of the remaining gap, v1's
"fine discrimination the mid-game actually needs" teaches exactly one rule:
**prefer a word that could be the answer over one that could not.** That is the
`CANDIDATE_BONUS = 0.5` line, not tree search, not entropy, not game value.

That rule is also the hard-mode bias the Phase-7 decoder was specifically
designed to avoid. Phase 7 measured that the expert probes with a
*non*-candidate 59.4% of the time at turn 2, and Phase 7b measured that
permitting probing (threshold 20 rather than always-filter) is worth 0.16–0.18
guesses. 4,685 pairs pushing the model toward naming candidates is a coherent
mechanism for a systematic **+1 guess** drift, which is exactly the observed
failure shape: `3 → 4` in 28 games and `4 → 5` in 22, against `4 → 3` in 18.

## 3. The budget is aimed away from the headroom

Phase 8's own counterfactual analysis measured where the remaining 0.32 guesses
live. v1's quotas were then set almost inversely to it:

| |admissible| | share of headroom | share of v1 rows |
|---|---:|---:|
| 2–10 | **74.7%** | 33.5% (4,999) |
| 11–100 | 29.1% | 46.4% (6,924) |
| 100+ | **−124.1%** | 20.1% (3,000) |

A fifth of the gradient went to the one regime the project had already measured
as actively harmful to hand over, and turn 2 alone accounts for 3,486 rows.

## 4. Trivial pairs, and what they are trivial against

```
clear        9,556 rows (64.0%)   median gap  9.98   p90 68.59   max 215.09
competitive  5,367 rows (36.0%)   median gap  0.50   p90  0.50   max   6.08
```

The README quotes 3.07 as the `clear` median. 3.07 is the median of **all**
rows; the `clear`-only median is **9.98**. A pair with a gap of 215 cannot
teach anything a 0.5B model does not already know.

Worse, in the unrestricted regime the `rejected` word contradicts feedback that
is printed in the prompt:

```
unrestricted rows                                     8,479
  ... whose `rejected` is inadmissible                8,415  (99.2%)
clear rows whose `rejected` is inadmissible           6,883 / 9,556  (72.0%)
```

These are legal actions, so this is not a validity bug. And inadmissible actions
are *normal* in the unrestricted regime — the strongest probes deliberately use
fresh letters, which contradicts the yellows already on the board. So the raw
rate is not the tell. **The asymmetry is:**

| unrestricted regime | v1 | v3 |
|---|---:|---:|
| `chosen` contradicts visible feedback | 54.8% | 87.4% |
| `rejected` contradicts visible feedback | **99.2%** | 87.4% |
| **asymmetry** | **+44.5 pts** | **+0.0 pts** |

A 44.5-point gap means the model can win 8,479 of v1's pairs by checking
feedback consistency alone — a rule it already knows and the decoder already
enforces below the threshold — without engaging any Wordle strategy. That is a
free, learnable shortcut sitting on 57% of the dataset, and it is the mechanism
behind the diagnostic pattern in the README: margin climbing to +11 while
accuracy sat at 0.836 from step 75.

## 5. Duplication

```
rows                              14,923
distinct states                    9,893
max rows for a single state          177   <- SALET->BBBBB, turn 2, 221 candidates
rows sitting in repeated states    6,668
duplicate `id` values              1,867   <- ids are not unique, so a row
                                              cannot be audited by id
```

v1's dedup key is `(prompt, rejected)` while `rejected` is re-sampled from a
fresh random 120-word subset on every visit, so revisiting a state mints new
rows indefinitely. The five most repeated states are all turn-2 and contribute
750 rows (5.0% of the dataset) between them.

## 6. No validation split

`sft_package/data/dpo_midgame/` contains exactly one file, `train.jsonl`, and
every row carries `meta.split == "train"`. The notebook loads it, shuffles it,
truncates to 15,000 and trains on all of it. `acc` is computed inside the
training loop as `(margin > 0).mean()` **on the batch currently being
optimised**.

So `acc 0.604 → 0.836` is a *training* accuracy. There is no held-out
preference measurement for v1 at all, and therefore no evidence that the
preference generalised — only that it was fitted. "DPO improved preference
accuracy but made gameplay worse" is more precisely: *DPO fitted its training
objective, and the only held-out measurement anyone took (gameplay) got worse.*

## 7. Leakage

| | |
|---|---:|
| rows whose **chosen** action is a held-out answer | **694** |
| rows whose **rejected** action is a held-out answer | 193 |
| states whose candidate set contains a held-out answer | 3,507 / 9,893 (35.4%) |
| rows in such states | 7,385 (49.5%) |
| states whose candidates are *entirely* held-out answers | 0 |

v1 is answer-keyed at the **source** only — rollouts start from a training
answer — but nothing stops a held-out answer surviving in a candidate set, or
becoming the preferred action. 694 rows actively raise the probability of
emitting a specific held-out answer word. The 246-answer benchmark is the only
held-out measurement in the project, so this is worth removing even though the
effect is indirect.

## 8. What is *not* wrong

Stated explicitly, because a review should be able to rule these out:

- **Prompt hygiene is clean.** No prompt contains the answer, a candidate list,
  a candidate count, an admissible count or a score. (A naive substring scan
  flags 9 rows for "score"/"cost" — all false positives: the words `SCORE`,
  `COSTA` and `COSTS` appear as *guesses* in the history.)
- **Every state re-derives exactly from its own prompt.** 0/14,923 mismatches on
  `n_candidates`, `n_admissible` and `turn`.
- **Every action is legal and correctly scoped.** `chosen` and `rejected` are
  always distinct 5-letter legal guesses, and in the restricted regime both are
  always in the admissible set. 0 violations.
- **The label direction is self-consistent.** 0/14,923 rows are backwards under
  v1's own cost function, and 0 under an exact tree in the 2–10 regime.
- **The action space matches the decoder.** v1 correctly switches between the
  admissible set and the full legal pool at the Phase-7b threshold of 20.
- **The `E[remaining]` machinery is correct.** `FeedbackMatrix.partition_stats`
  reproduces exactly, and an independently rebuilt 12,972 × 12,972 feedback
  table agrees byte-for-byte with the shipped one on all answer columns.

---

## The structural finding, and the skeptical reading

While designing the replacement I measured something that is a fact about
Wordle rather than about any dataset. Over 1,200 distinct reachable states with
2–10 admissible words, scored exactly:

```
actions available per state                     median 6
actions that are EXACTLY optimal                median 2
states with more than one optimal action              95.3%
states where EVERY action is optimal                  11.9%
gap to the best strictly-worse action           median 0.50 guesses
                                                only 19.2% are within 0.25
                                                       10.9% are over 1.0
```

Reproduce with `python phase8_dpo_v3/measure_decision_structure.py`
(-> `decision_structure.json`). The population is the one the generator
labels: reachable, 2-10 admissible, at least two candidates left.

Three consequences, and they cut against the method:

1. **The optimum is usually a set, not a point.** In **95.3%** of the states
   that carry the headroom, two or more actions are exactly equally good — a
   median of 2 optimal actions out of 6 available. A pairwise preference cannot
   represent "these two are equal". Any generator that simply takes the top two
   actions by score spends almost all of its rows asserting an ordering that
   does not exist — and in **11.9%** of states every action is optimal and there
   is nothing whatsoever to teach.

2. **Close decisions are unevenly distributed, and absent where the endgame
   is tightest.** Gaps are quantised to multiples of `1/n_candidates`, so the
   decision tends to be *wide*: a median 0.50-guess step to the next-best move,
   and 10.9% of states where even the closest alternative is over a full guess
   worse. Worse, the resource thins exactly where the game is decided:

   | \|admissible\| | every action tied | has a near-miss (≤ 0.5) |
   |---|---:|---:|
   | 2–3 | **50.0%** | 20.3% |
   | 4–6 | 4.2% | 60.1% |
   | 7–10 | 0.0% | 83.7% |

   At 2–3 admissible words — the last decision before the game ends — half the
   states have nothing to teach at all, and only a fifth contain a close call.

3. **DPO does not address the measured bottleneck.** Phases 4–6 established
   that given a state with exactly one possible answer and nothing left to
   decide, the best model names it 20% of the time — 33% even when it was
   trained on that word. That is a retrieval failure. Ranking two words the
   model can already produce does not fix an inability to produce the right
   word at all.

**So: is DPO fundamentally mismatched here?** My honest read is **yes, weakly
mismatched — not merely badly fed.** The v1 data was genuinely bad and the
regression is fully explained by it, so a corrected retry is a legitimate
experiment. But three structural facts cap the upside: the optimum is
set-valued where the headroom is, the objective is a static pairwise proxy for a
six-step sequential score, and the binding constraint measured across three
phases is retrieval rather than ranking. I would not expect a large gain, and I
would not run a third DPO variant if this one also fails a paired rollout test.

**What the same machinery supports better.** The exact adaptive tree yields, for
every state, the *complete set* of optimal actions. That is a natural fit for a
set-aware objective and a poor fit for pairwise DPO:

- **Filtered / rejection-sampling SFT** on the optimal set — cross-entropy over
  "any optimal action", which represents ties correctly instead of breaking them
  arbitrarily. This is the option I would run first, and it reuses this dataset's
  labels directly.
- **A ranking loss with an explicit tie class**, if a preference form is wanted.
- **Early stopping on gameplay, not loss.** v1's accuracy saturated at step 75
  of 466; everything after was drift. Whatever the objective, the stopping
  criterion has to be the paired 246-answer rollout.

The corrected dataset is built so that all of these are available from it: every
row records the full optimal-action set size, exact values, and the regime it
was scored in.
