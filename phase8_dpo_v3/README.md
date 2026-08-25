# Phase 8 v3 — corrected DPO preference data

An audited replacement for `sft_package/data/dpo_midgame/`, the dataset behind
the Phase-8 DPO run that learned its objective and made gameplay worse
(3.7642 → 3.9350, paired t = −3.21).

**Read [`AUDIT.md`](AUDIT.md) first.** It documents what was wrong with v1, and
it also argues — with measurements — that DPO is a weak fit for this task even
with clean data. This dataset is built so that a corrected retry is a fair
experiment, not because the result is expected to be positive.

```
sft_package/data/dpo_v3/
  train.jsonl        6,000 pairs   (11.4 MB)
  validation.jsonl     800 pairs   ( 1.5 MB)
  manifest.json      generation config, quotas, rejections, summary stats
  audit.json         output of the verifier
```

---

## What changed, in one table

| | v1 `dpo_midgame` | v3 `dpo_v3` |
|---|---|---|
| rows / distinct states | 14,923 / 9,893 | 6,000 / **6,000** |
| max rows from one state | **177** | **1** |
| duplicate `id`s | 1,867 | 0 |
| value function (2–10 words) | `E[remaining] − 0.5·is_candidate` | **exact adaptive-decoder tree, full depth, integer costs** |
| alternatives scored per state | 120 random + expert | **every action in the space** |
| share in the 2–10 bucket (74.7% of the headroom) | 33.5% | **75.0%** |
| share in the 100+ bucket (−124.1% of the headroom) | 20.1% | **6.7%** |
| hard-pair share | 36.0% (and not actually hard — see `AUDIT.md`) | **74.7%** |
| consistency-shortcut asymmetry (unrestricted regime) | **+44.5 pts** | **+0.0 pts** |
| pairs between two exactly-tied optimal actions | not checked | **forbidden, verified 0** |
| train rows naming a held-out answer | 887 | **0** |
| rows whose candidates touch a held-out answer | 7,385 (49.5%) | 1,525 (25.4%) |
| validation split | **none** | 800 rows, answer- and state-disjoint |

Full old-vs-new statistics: `python phase8_dpo_v3/compare_datasets.py`.

---

## Row format

One JSON object per line. `prompt`, `chosen` and `rejected` are the only fields
a trainer should read; everything else is audit metadata.

```jsonc
{
  "id": "dpo3-3d053403bc5bcdf7",
  "prompt": "You are playing Wordle. ...\n\nNext guess:",
  "chosen":   "CORNU",        // uppercase, 5 letters, always a legal guess
  "rejected": "TOURS",
  "meta": { ... }
}
```

The prompt is produced by the same `render_prompt` call as every earlier phase.
It contains the guess history, the constraints derivable from it, and the turn
counter — **and nothing else**. No answer, no candidate list, no candidate
count, no admissible count, no score. The verifier re-checks this on every row.

### `meta` fields

**Identity and provenance**

| field | meaning |
|---|---|
| `format_version` | `3` |
| `split` | `"train"` or `"validation"` |
| `state_key` | `"salet:BBBBB\|crony:BYBBG"` — the full history, the dedup key |
| `state_hash`, `source_answer_hash` | salted hashes for leakage checks without revealing the answer |
| `turn`, `guesses_left` | position in the game |
| `reachable` | always `true`; every state came from an actually-played game |

**Decision regime** — which problem the model faces here

| field | meaning |
|---|---|
| `n_candidates` | surviving answers (solver-side; never in the prompt) |
| `n_admissible` | legal guesses consistent with all feedback |
| `bucket` | `"2-10"`, `"11-100"`, `"100+"`, keyed on `n_admissible` |
| `action_space` | `"admissible"` if the decoder would filter here, else `"legal"` |
| `adaptive_threshold` | `20`, the Phase-7b setting |
| `n_actions` | size of the action space at this state |

**Why this pair got this label** — the audit trail

| field | meaning |
|---|---|
| `score_family` | `exact_adaptive_decoder_tree` or `full_legal_pool_one_ply_expected_remaining` |
| `score_units` | `mean_further_guesses` or `expected_remaining_candidates` |
| `exact` | `true` only for the tree-scored rows |
| `horizon`, `tree_depth`, `failure_cost` | scoring parameters |
| `chosen_value`, `rejected_value` | lower is better |
| `preference_gap` | `rejected_value − chosen_value`, **in `score_units`** |
| `preference_gap_relative` | the same gap over `chosen_value` — the only form comparable across regimes |
| `chosen_total_cost`, `rejected_total_cost` | the exact **integer** numerators |
| `best_value`, `chosen_rank`, `rejected_rank` | position in the full action ranking |
| `n_optimal_actions` | **how many actions are exactly optimal here** |
| `min_possible_positive_gap` | `1/n_candidates`; the smallest real difference at this state |
| `pair_type` | `competitive` or `contrastive` (see below) |
| `chosen_is_admissible` / `_candidate`, `rejected_is_…` | action properties |

> **`preference_gap` is not comparable across `score_family` values.** One is in
> guesses, the other in candidates. Pool with `preference_gap_relative`, or
> filter to one family first.

---

## How the labels are made

### The action space is the decoder's, not the vocabulary's

Under the Phase-7b adaptive decoder the model's options depend on the board:

- `|admissible| ≤ 20` → the decoder restricts it to feedback-consistent legal
  words. Both `chosen` and `rejected` are drawn from that set.
- `|admissible| > 20` → any legal word. Both are drawn from all 12,972.

The verifier confirms per row that both actions lie inside the space the
deployed decoder would actually offer.

### Values: exact where it matters

**Restricted regime (`|admissible| ≤ 20`, 3,818 + 1,125 = 4,943 train rows).**
The label is the exact solution of the finite game the adaptive decoder plays,
solved to *full* remaining depth. At every node the action set is refined by the
feedback just received, exactly as the decoder refines it at run time. This is
closed and cheap because filtering only ever removes words: once a game is below
the threshold, every successor is too.

Costs are **integers** — total guesses summed over every candidate — so ties are
exact and a gap of `1/n` is a real one-guess difference on one candidate, not a
rounding artefact. The evaluator is checked against a no-memo, no-pruning
reference implementation on a random sample every time the verifier runs.

**Unrestricted regime (`|admissible| > 20`, 1,057 train rows).** No exact
multi-ply value is affordable over 12,972 actions, so the label is an exhaustive
one-ply `E[remaining candidates]` over the whole legal pool — and every such row
is marked `exact: false`, `horizon: "one_ply_proxy"`. The integer numerator is
recovered so ties are still exact.

There is **no candidate bonus** anywhere. v1's `− 0.5·is_candidate` was the
entire preference signal for 75.9% of its "competitive" pairs; in the exact
regime the tree already values winning-now correctly, because guessing a
candidate genuinely ends some branches early.

### Pairs: `chosen` is optimal, `rejected` is strictly worse

`chosen` is always an **exactly optimal** action. Where several tie, it is
spread over them by a hash of the state key rather than taken alphabetically.

`rejected` is always **strictly worse** — never a tied optimum. This matters
more than it sounds: **89.4% of the emitted states have more than one exactly
optimal action** (95.3% among 2–10 states measured on their own — see
`measure_decision_structure.py`). Any generator that takes the top two by score
spends most of its rows asserting an ordering that does not exist.

| `pair_type` | rule | train rows |
|---|---|---:|
| `competitive` | exact regime: the **closest strictly-worse tier** — the hardest correct discrimination the state admits.<br>proxy regime: a relative band of 5–20% over the best value. | 4,483 (74.7%) |
| `contrastive` | exact regime: 1.0–2.5 mean guesses worse.<br>proxy regime: 30–100% relative. | 1,517 (25.3%) |

`contrastive` has an **upper** bound as well as a lower one. A pair nobody could
get wrong teaches nothing — v1 had gaps up to 215.

States where *every* action is optimal are dropped (2,214 in this run); there is
nothing to teach there.

### How close the competitive pairs actually are

"The closest strictly-worse action" is as hard as the state allows, but that is
not the same as *always hard*. For the 3,818 exact-regime competitive rows:

| true gap (mean further guesses) | rows | share |
|---|---:|---:|
| ≤ 0.25 | 699 | 18.3% |
| ≤ 0.50 | 2,756 | **72.2%** |
| ≤ 1.00 | 3,203 | 83.9% |
| ≥ 2.00 | 579 | 15.2% |

Gaps are quantised to multiples of `1/n_candidates`, so 0.333 (793 rows) and
0.25 (364) are common exact values rather than approximations.

The 579-row tail where even the closest alternative is ≥ 2 guesses worse is
concentrated at `guesses_left ∈ {1, 2}` with 2–3 candidates: positions where the
alternative simply loses the game, and the failure cost of 7 dominates. Those
are correct labels but easy ones. `preference_gap` is in the metadata, so filter
them if you want a uniformly hard set:

```python
hard = [r for r in train if r["meta"]["exact"] and r["meta"]["preference_gap"] <= 1.0]
```

### A worked example

```
History:
  1. SALET -> BYYYG

Deduced so far:
  Confirmed letters : p5=T
  Letters present   : A, E, L, T
  Letters absent    : S
  Ruled-out spots   : A not at [1], E not at [3], L not at [2]

Guess 2 of 6 (5 remaining)
```

`chosen: LEAPT`, `rejected: PLEAT` — both admissible, both anagrams of the same
letters, and the entire difference is one guess on one of the eight surviving
candidates:

```
chosen_total_cost      17      rejected_total_cost      18
chosen_value        2.125      rejected_value         2.25
preference_gap      0.125  ==  min_possible_positive_gap
n_actions               9      n_optimal_actions          1
```

This is the shape of pair the exercise is supposed to produce, and the shape v1
contained almost none of.

### Sampling: reachable and on-distribution

Every state comes from an actually-played game, so reachability is structural.
Plausibility is handled by the continuation policy:

- turn 1 — always `SALET` (the SFT policy reproduces its opener 100% of the time)
- turn 2 — the entropy expert's probe 70% of the time (the SFT policy agrees
  with its expert at turn 2 90% of the time), else an admissible word
- turn 3+ — an admissible word in the restricted regime, because that *is* the
  decoder's action space; a candidate/admissible mix with a small arbitrary-probe
  tail otherwise

v1 chose a uniformly random legal word 45% of the time, which is nothing like
the deployed policy and kept the admissible set artificially large.

Quotas follow the measured headroom: **75% of rows in the 2–10 bucket** that
holds 74.7% of the remaining gap, 18.3% in 11–100, and 6.7% in 100+ which was
measured at −124% and is retained only for coverage.

### Leakage controls

1. **Prompt** — structurally cannot contain the answer (`render_prompt` has no
   answer parameter), the candidate list, the count, or any score. Re-checked
   per row.
2. **Source answers** — train rows come only from the 2,069 training answers,
   validation rows only from the 246 held-out ones. Verified via salted hashes.
3. **Actions** — no training row names a held-out answer as `chosen` or
   `rejected` (v1 had 887). 989 candidate states were dropped for this; the
   held-out set is chosen by `sha256(salt|answer)`, independent of every game
   property, so the filter cannot preferentially remove hard or easy states.
4. **States** — train and validation share no state key and no prompt.
5. **Candidate sets** — a large state's candidate set will contain held-out
   answers; that is unavoidable and is *reported*, not hidden. Pass
   `--strict-candidate-partition` to force it to zero at the cost of losing
   nearly all large states.

---

## Reproducing

Everything depends only on the shipped `artifacts/` bundle and
`sft_package/eval/*_answers.jsonl` — no network, no extra data files, and no
precomputed 12,972 × 12,972 table (feedback rows are computed on demand and
cached, and were checked byte-identical against a full table).

Fixed seed, deterministic: two runs at the same seed and quotas produce
**byte-identical** `train.jsonl` and `validation.jsonl`. Verified.

```bash
python phase8_dpo_v3/build_dpo_v3_dataset.py      # ~8 min, writes sft_package/data/dpo_v3/
python phase8_dpo_v3/verify_dpo_v3_dataset.py     # gate: exits non-zero on any hard failure
python phase8_dpo_v3/audit_legacy_dpo.py          # reproduces every number in AUDIT.md
python phase8_dpo_v3/compare_datasets.py          # old vs new statistics
python phase8_dpo_v3/measure_decision_structure.py   # the structural argument in AUDIT.md
```

Useful variants:

```bash
# 2-10 states only — the bucket carrying 74.7% of the headroom
python phase8_dpo_v3/build_dpo_v3_dataset.py --train-11-100 0 --train-100+ 0 \
    --val-11-100 0 --val-100+ 0 --out sft_package/data/dpo_v3_endgame_only

# competitive pairs only, no easy signal at all
python phase8_dpo_v3/build_dpo_v3_dataset.py --competitive-frac 1.0

# strictest possible answer partition
python phase8_dpo_v3/build_dpo_v3_dataset.py --strict-candidate-partition
```

---

## Using it on Kaggle

### 1. Package

```bash
python tools/prepare_kaggle_dataset.py --clean
```

This writes `kaggle_upload/` containing `sft_package/`, `code/` (the exact
Wordle environment), `artifacts/` (vocabulary + feedback matrix), and this
folder's `README.md` / `AUDIT.md` / audit JSON, plus a
`DATASET_MANIFEST.json` with a SHA-256 for every file. Upload the whole folder
as a Kaggle Dataset. No credentials are included.

### 2. Load

```python
import json, os
DATASET_DIR = "/kaggle/input/<your-dataset-slug>"
DPO_DIR = os.path.join(DATASET_DIR, "sft_package/data/dpo_v3")

def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

train = load_jsonl(os.path.join(DPO_DIR, "train.jsonl"))
val   = load_jsonl(os.path.join(DPO_DIR, "validation.jsonl"))
manifest = json.load(open(os.path.join(DPO_DIR, "manifest.json")))

# NEVER concatenate these two.
assert not ({r["meta"]["state_key"] for r in train}
            & {r["meta"]["state_key"] for r in val})
```

### 3. Guard rails to keep in the notebook

```python
# the prompt must stay clean
assert not any("Possible answers" in r["prompt"] for r in train + val)
# no training row may name a held-out answer
holdout = {json.loads(l)["answer"].lower() for l in
           open(os.path.join(DATASET_DIR, "sft_package/eval/val_answers.jsonl"))}
assert not any(r["chosen"].lower() in holdout or r["rejected"].lower() in holdout
               for r in train)
```

### 4. Encoding

Unchanged from the v1 notebook — the row format is drop-in compatible, so
`phase8_dpo/make_dpo_notebook.py`'s `encode`/`PairData`/`seq_logp` work as-is.
Only the path changes:

```python
PAIRS = load_jsonl(os.path.join(SFT_DIR, "data/dpo_v3/train.jsonl"))
```

### 5. Report the metric v1 could not

v1's `acc 0.604 → 0.836` was a **training** accuracy computed on the batch being
optimised. With a validation split, evaluate on `validation.jsonl` at intervals:

```python
@torch.no_grad()
def pref_accuracy(policy, rows):
    """Fraction of held-out pairs the policy ranks correctly."""
    ok = 0
    for r in rows:
        ci, cl = encode(r["prompt"], r["chosen"])
        ri, rl = encode(r["prompt"], r["rejected"])
        ...  # margin = (pi_c - ref_c) - (pi_r - ref_r)
        ok += margin > 0
    return ok / len(rows)
```

Report it **broken down by `score_family`**: the `exact_adaptive_decoder_tree`
rows are the ones whose labels are trustworthy, and they are the ones that
correspond to the measured headroom.

---

## Recommended first run, and the stopping rule

Given the audit, if a DPO run happens at all:

1. **Train on the `2-10` bucket only** (`bucket == "2-10"`, 4,500 rows) — those
   are exactly-labelled and carry 74.7% of the headroom. Keep the other buckets
   as an ablation, not the default.
2. **Small β, ≤1 epoch, low LR.** v1's accuracy saturated at step 75 of 466 and
   the remaining 390 steps were pure drift away from the SFT reference.
3. **Early-stop on gameplay, not loss.** The only metric that has ever detected
   this failure is the paired 246-answer rollout. Checkpoint often and evaluate.
4. **Pre-register the decision rule.** v1 is already one clean negative. If a
   corrected, exactly-labelled, competitive-pair run also fails a paired rollout
   test, that is evidence about the *method*, and `AUDIT.md` sets out what to
   do instead — chiefly filtered SFT over the optimal action *set*, which this
   dataset's `n_optimal_actions` and exact values already support and which
   represents ties correctly instead of breaking them arbitrarily.

---

## Files

```
phase8_dpo_v3/
  dpo_core.py                  Board, AdaptiveTree, OnePlyScorer — shared by all three scripts
  build_dpo_v3_dataset.py      the generator
  verify_dpo_v3_dataset.py     independent verification; exits non-zero on failure
  audit_legacy_dpo.py          reproduces every number in AUDIT.md from the v1 file
  compare_datasets.py          old vs new statistics -> comparison.json
  measure_decision_structure.py  how close the decisions really are; backs the
                               "is DPO the right objective" section of AUDIT.md
  AUDIT.md                     findings, and the argument about method fit
  README.md                    this file
  generation.log  verification.log
```

`phase8_dpo_corrected/` is an earlier partial draft of this work (its output,
`dpo_adaptive_competitive_v2/`, is a 75-row smoke test with a depth-3 truncated
tree). It is superseded by this folder and can be deleted.
