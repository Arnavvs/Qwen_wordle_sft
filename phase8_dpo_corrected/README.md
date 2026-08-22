# Corrected Phase-8 DPO dataset

This is a replacement generator for `phase8_dpo/build_dpo_dataset.py`.  It is
deliberately a new pipeline, not a patch to the legacy file.  Do **not** train
the existing Phase-8 DPO adapter from this directory.

## What it produces

`build_corrected_dpo_dataset.py` writes:

```
sft_package/data/dpo_adaptive_competitive_v2/
  train.jsonl
  validation.jsonl
  manifest.json
```

Every row has `prompt`, `chosen`, `rejected`, and audit-only `meta` fields.
Only the first three fields should be read by a DPO loader.  In particular,
never concatenate `meta` into model input.

The prompt contains no hidden answer, candidate set/count, admissible set/count,
or classical score.  Metadata records state and candidate-set digests, action
space, both recomputable scores, score direction, gap bounds, split, and the
label scorer.  The source answer is represented only by a non-reversible hash.

## Labeling policy

The deployed Phase-7b decoder uses a feedback-consistent action set only when
the admissible set has at most 20 words.  The dataset follows that exact rule.

| State regime | Current action space | Label value |
|---|---|---|
| 2--20 admissible | all feedback-consistent legal words | exact adaptive hard-mode tree, depth 3 |
| >20 admissible | all 12,972 legal guesses | exhaustive one-ply tree value, expected remaining candidates |

The restricted evaluator propagates the hard-mode action set down each feedback
branch.  It therefore does not make the legacy mistake of asking a full-pool
tree for an action and then projecting it into a different decoder space.

There is one pair per state.  The rejected action is sampled reproducibly only
from actions with a strict, bounded score gap from the best action.  Obvious
worst actions are excluded.  The default mix is 75% `2-10`, 18% `11-100`, and
7% `100+`; the generated local set may be smaller if the strict split barrier
cannot fill a quota.

## Leakage barrier

Source answers follow the existing 2,069/246 train/validation split, and the
generator rejects every exact feedback state already assigned to the other
split.  This prevents source-answer, game, exact-state, and exact-pair leakage.
The verifier checks those barriers exhaustively.

The candidate set is not model input, and it naturally contains answers from
both source partitions at many early states.  If an even stronger forensic
partition is needed, `--strict-candidate-partition` retains only states whose
entire candidate set falls within its source split.  It is intentionally
optional: with a 246-answer validation split it makes 100+ candidate states
vanishingly rare and cannot provide a useful validation distribution.

## Local reproduction

From the project root, use the supplied Conda environment (the system Python
may not include NumPy):

```powershell
.\.conda\python.exe phase8_dpo_corrected\build_corrected_dpo_dataset.py
.\.conda\python.exe phase8_dpo_corrected\verify_corrected_dpo_dataset.py `
  --report sft_package\data\dpo_adaptive_competitive_v2\audit.json
```

The seed is recorded in `manifest.json`.  A generator run may finish below a
requested quota rather than fabricating a pair; inspect `manifest.json` before
training.

## Kaggle use

1. Upload the project bundle containing `core/`, `artifacts/`, `sft_package/`,
   and `phase8_dpo_corrected/`.
2. Run the generator and verifier cells before any training cell.
3. Stop if the verifier does not print `"ok": true`.
4. Load `train.jsonl` for optimization and `validation.jsonl` only for
   preference validation/early stopping.  Do not fold validation into training.
5. Keep Phase-7 adaptive decoding at threshold 20 when evaluating.  A DPO
   dataset labelled for this action space cannot validate a different decoder.

The legacy Phase-8 notebook is not automatically compatible: it hard-codes
`data/dpo_midgame/train.jsonl`.  Point a new training notebook at this directory
and make it consume only `prompt`, `chosen`, and `rejected`.

## Important limitation

This dataset removes identified data and objective-alignment defects, but does
not prove that DPO is the right optimization method.  DPO still optimizes static
pairwise local values while Wordle is a sequential rollout problem, and the
decoder already determines much of the outcome.  Any future run should be
short, use the validation pairs and paired game rollouts for early stopping,
and be treated as a falsifiable experiment rather than a presumed improvement.
