# Phase-8 DPO audit (pre-training)

## Bottom line

The legacy DPO run is a valid negative result: its loss and pair accuracy rose
while paired rollout performance fell (3.7642 to 3.9350).  The data did not
have an observed reversed one-ply preference label, but it was poorly matched
to the decision problem and heavily duplicated.  The corrected generator is a
cleaner experiment, not evidence that DPO will improve Wordle play.

## Findings in the legacy dataset

The supplied `train.jsonl` has 14,923 rows and 9,893 distinct prompts:

| Measure | Legacy result | Consequence |
|---|---:|---|
| clear pairs | 9,556 / 14,923 (64.0%) | easy gradients dominate |
| clear median gap | 9.9773 | not close discrimination |
| clear maximum gap | 215.0882 | extremely trivial comparisons exist |
| competitive median / max gap | 0.5000 / 6.0814 | `competitive` is not consistently close |
| duplicate state rows | 5,030 | repeated states dominate updates |
| maximum rows for one prompt | 177 | strong state-frequency distortion |
| duplicate opaque ids | 1,867 | ids cannot uniquely audit a row |
| desired bucket quotas met | no | generator ended at 14,923 rather than its 15,000 requested total |
| one-ply label direction failures | 0 / 14,923 | no reversal found under its own scalar cost |
| distinct legacy states compatible with a held-out answer | 3,507 | train-source splitting is not strict state leakage control |

The README statement that the *clear* median gap is 3.07 is inaccurate for the
checked file: 3.0714 is the **overall** median.  The clear-only median is
9.9773.

## Why better DPO metrics made games worse

The loss rewards making already-correct pair margins larger.  With 64% clear
pairs, the run reached 0.836 pair accuracy by roughly step 75 and then grew its
margin to +11.1.  That is exactly the pattern expected when DPO drifts away from
the SFT policy to optimize easy static preferences.  The paired rollout result
(70 games worse, 38 better; t=-3.21) and lower model contribution show this was
not a decoder artifact.

There is an additional objective mismatch.  The legacy selected a `tree_salet`
action when possible, but judged it against alternatives with the different
one-ply scalar `expected_remaining - 0.5 * is_candidate`.  The 0.5 bonus is an
unstated heuristic rather than a tree value.  Above 12 candidates it selected
the preferred action from a random 120-word subset, not the actual action
space.  Therefore its “expert” is neither consistently a tree-optimal action
nor the full-pool one-ply optimum.  The old verifier checks the same scalar,
so it cannot validate this substitution.

## Decoder alignment and legality

The legacy design correctly recognized the immediate Phase-7b rule: it uses
the feedback-consistent legal set at <=20 admissible words and the full legal
pool above 20.  Its code also prevents an obviously inadmissible selected tree
or entropy action from being emitted in the restricted regime.  However, the
tree value itself is over the unconstrained future game, so its labels are not
values of the adaptive decoder's sequential decision process.  The replacement
propagates the admissible set through every branch in its high-value regime.

The legacy verifier has meaningful gaps: it samples only 2,500 labels for score
direction and state reconstruction; it checks one prompt-hygiene condition only
over the first 500 rows; it carries no DPO validation split; and its source
answer test only looks for held-out targets at one-candidate states.  It cannot
prove that a multi-candidate row was not generated from, or reachable by, a
held-out answer.

## Remaining fundamental limitation

DPO is not a natural exact objective for Wordle.  The deployed policy chooses
the top model-scored action from a state-dependent constrained set, while DPO
sees isolated textual comparisons and does not directly optimize six-turn game
score, failure risk, or the decoder's next constrained set.  A local preference
can be correct and still degrade a rollout after policy drift.  The data
correction makes one small, well-controlled retry defensible; it does **not**
justify forcing DPO if paired rollout validation again fails.
