"""
build_sft_package.py — assemble a training/evaluation package from the three
trajectory datasets. Does NOT train anything.

    python build_sft_package.py

Produces `sft_package/`:

    data/<solver>/train.jsonl        model-facing SFT records, no answers
    data/<solver>/val.jsonl
    data/disagreement/contrast.jsonl the 86 entropy-vs-tree_soare states,
                                     BOTH expert actions on one record
    data/disagreement/<policy>.jsonl the same states in plain SFT form, so each
                                     policy's action can be oversampled
    eval/val_answers.jsonl           held-out answers for rollout scoring
                                     (NOT model input - clearly separated)
    index/id_map.jsonl               id -> game/answer, for analysis only
    configs/*.json                   oversampling recipes
    manifest.json

MODEL-FACING RECORDS CARRY NO ANSWER, DIRECTLY OR INDIRECTLY
------------------------------------------------------------
`_holdout` is dropped. So is `game_id`: it encodes the answer's row index, so
anyone holding the answer list could invert it. Records get an opaque hashed
`id` instead, and `index/id_map.jsonl` (analysis only, never fed to a model)
keeps the mapping for evaluation.

DUPLICATION IS EXPOSED, NOT DECIDED
-----------------------------------
Many games pass through the same state, so the same (prompt, completion) pair
recurs -- the turn-1 record is literally identical in all 2315 games. Whether to
train on the natural visitation distribution or on distinct states is a real
modelling choice, so the full per-step files are written as-is and every record
carries `meta.duplicate_count`. `build_weighted_dataset.py --dedup` collapses
them at build time. Nothing is baked in.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from typing import Dict, List, Tuple

SOLVERS = ["entropy", "tree_soare", "tree_salet"]
SRC = "trajectories"
DEST = "sft_package"


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def history_key(step: Dict) -> str:
    return "|".join(f"{h['guess']}:{h['feedback']}"
                    for h in step["state"]["history"])


def opaque_id(solver: str, game_id: str, turn: int) -> str:
    h = hashlib.sha256(f"{solver}|{game_id}|{turn}".encode()).hexdigest()
    return f"{solver[:2]}-{h[:14]}"


def to_sft(step: Dict, dup_counts: Dict[Tuple[str, str], int],
           disagreement_keys: set) -> Dict:
    """Model-facing record. Only `prompt` and `completion` go to the model."""
    k = (step["prompt"], step["completion"])
    return {
        "id": opaque_id(step["solver"], step["game_id"], step["turn"]),
        "prompt": step["prompt"],
        "completion": step["completion"],
        "meta": {
            "solver": step["solver"],
            "turn": step["turn"],
            "n_candidates": step["state"]["n_candidates"],
            "bits_remaining": step["state"]["bits_remaining"],
            "action_is_candidate": step["action"]["was_in_candidate_set"],
            "action_is_possible_answer": step["action"]["is_possible_answer"],
            "information_gain_bits": step["action"]["information_gain_bits"],
            "fully_determined": step["state"]["constraints"]["fully_determined"],
            "is_disagreement_state": history_key(step) in disagreement_keys,
            "duplicate_count": dup_counts[k],
            "split": step["split"],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--clean", action="store_true",
                    help="remove the destination directory first")
    args = ap.parse_args(argv)

    if args.clean and os.path.isdir(args.dest):
        shutil.rmtree(args.dest)

    steps = {s: load_jsonl(os.path.join(args.src, f"{s}_steps.jsonl"))
             for s in SOLVERS}
    games = {s: load_jsonl(os.path.join(args.src, f"{s}_games.jsonl"))
             for s in SOLVERS}
    for s in SOLVERS:
        print(f"loaded {s}: {len(steps[s])} steps, {len(games[s])} games")

    # ---- disagreement states: entropy vs tree_soare (shared opener) --------
    ent_by_key = {history_key(x): x for x in steps["entropy"]}
    soa_by_key = {history_key(x): x for x in steps["tree_soare"]}
    shared = set(ent_by_key) & set(soa_by_key)
    dis_keys = {k for k in shared
                if ent_by_key[k]["completion"] != soa_by_key[k]["completion"]}
    print(f"\nshared states: {len(shared)}   disagreement states: {len(dis_keys)}")

    # Which answers pass through each disagreement state, and their splits.
    # A state is reached by many games, so its split is a composition, not a
    # single label -- this is reported rather than silently collapsed.
    reach: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for st in steps["entropy"]:
        k = history_key(st)
        if k in dis_keys:
            reach[k][st["split"]] += 1

    # ---- per-solver SFT files --------------------------------------------
    manifest: Dict = {"solvers": {}, "disagreement": {}, "notes": {}}
    for s in SOLVERS:
        dup = collections.Counter((x["prompt"], x["completion"])
                                  for x in steps[s] if x["split"] == "train")
        rows = [to_sft(x, dup, dis_keys) for x in steps[s]]
        tr = [r for r in rows if r["meta"]["split"] == "train"]
        va = [r for r in rows if r["meta"]["split"] == "val"]
        write_jsonl(os.path.join(args.dest, "data", s, "train.jsonl"), tr)
        write_jsonl(os.path.join(args.dest, "data", s, "val.jsonl"), va)

        distinct_tr = len({(r["prompt"], r["completion"]) for r in tr})
        manifest["solvers"][s] = {
            "train_records": len(tr),
            "val_records": len(va),
            "distinct_train_pairs": distinct_tr,
            "duplication_factor": round(len(tr) / max(distinct_tr, 1), 2),
            "disagreement_train_records": sum(
                1 for r in tr if r["meta"]["is_disagreement_state"]),
        }
        print(f"  {s:<12} train={len(tr):>5} val={len(va):>4} "
              f"distinct_train={distinct_tr:>5} "
              f"(x{len(tr)/max(distinct_tr,1):.2f} duplication)")

    # ---- contrast file: both actions on one record ------------------------
    contrast = []
    for k in sorted(dis_keys):
        e, t = ent_by_key[k], soa_by_key[k]
        assert e["prompt"] == t["prompt"], "same state must render identically"
        comp = reach[k]
        contrast.append({
            "id": "dis-" + hashlib.sha256(k.encode()).hexdigest()[:14],
            "prompt": e["prompt"],
            "completions": {
                "entropy": e["completion"],
                "tree_soare": t["completion"],
            },
            "meta": {
                "history": [{"guess": h["guess"], "feedback": h["feedback"]}
                            for h in e["state"]["history"]],
                "turn": e["turn"],
                "n_candidates": e["state"]["n_candidates"],
                "entropy_action_is_candidate":
                    e["action"]["was_in_candidate_set"],
                "tree_action_is_candidate": t["action"]["was_in_candidate_set"],
                "reached_by_train_games": comp.get("train", 0),
                "reached_by_val_games": comp.get("val", 0),
            },
        })
    write_jsonl(os.path.join(args.dest, "data", "disagreement",
                             "contrast.jsonl"), contrast)

    # Plain SFT form of the same states, per policy, TRAIN-REACHABLE ONLY.
    # A state reached by any val game is excluded from the oversampling pool so
    # that inflating it cannot pull held-out states into training.
    pool_keys = [k for k in sorted(dis_keys) if reach[k].get("val", 0) == 0]
    for label, by_key in (("entropy", ent_by_key), ("tree_soare", soa_by_key)):
        dup = collections.Counter()
        rows = []
        for k in pool_keys:
            st = by_key[k]
            rows.append(to_sft(st, collections.Counter({
                (st["prompt"], st["completion"]): 1}), dis_keys))
        write_jsonl(os.path.join(args.dest, "data", "disagreement",
                                 f"{label}.jsonl"), rows)
    manifest["disagreement"] = {
        "pair": "entropy vs tree_soare (shared opener SOARE)",
        "shared_states": len(shared),
        "disagreement_states": len(dis_keys),
        "oversampling_pool_states": len(pool_keys),
        "excluded_touch_val": len(dis_keys) - len(pool_keys),
        "by_turn": dict(sorted(collections.Counter(
            ent_by_key[k]["turn"] for k in dis_keys).items())),
    }
    print(f"\ncontrast states written: {len(contrast)}")
    print(f"  oversampling pool (no val game passes through): {len(pool_keys)}")
    print(f"  excluded because a val game reaches them: "
          f"{len(dis_keys) - len(pool_keys)}")

    # ---- evaluation answers (NOT model input) -----------------------------
    val_answers = sorted({g["_holdout"]["answer"] for g in games["entropy"]
                          if g["split"] == "val"})
    train_answers = sorted({g["_holdout"]["answer"] for g in games["entropy"]
                            if g["split"] == "train"})
    write_jsonl(os.path.join(args.dest, "eval", "val_answers.jsonl"),
                [{"answer": a, "split": "val"} for a in val_answers])
    write_jsonl(os.path.join(args.dest, "eval", "train_answers.jsonl"),
                [{"answer": a, "split": "train"} for a in train_answers])
    print(f"\neval answers: {len(val_answers)} val, {len(train_answers)} train")

    # ---- id map (analysis only) ------------------------------------------
    idmap = []
    for s in SOLVERS:
        for x in steps[s]:
            idmap.append({
                "id": opaque_id(s, x["game_id"], x["turn"]),
                "solver": s, "game_id": x["game_id"], "turn": x["turn"],
                "answer": x["_holdout"]["answer"], "split": x["split"],
            })
    write_jsonl(os.path.join(args.dest, "index", "id_map.jsonl"), idmap)
    print(f"id map: {len(idmap)} entries (analysis only, never model input)")

    # ---- oversampling configs --------------------------------------------
    cfgdir = os.path.join(args.dest, "configs")
    recipes = {
        "baseline_entropy.json": {
            "name": "baseline_entropy",
            "description": "Plain SFT on the greedy entropy policy.",
            "seed": 20260817,
            "dedup": False,
            "shuffle": True,
            "components": [
                {"source": "data/entropy/train.jsonl", "repeat": 1},
            ],
        },
        "baseline_tree_soare.json": {
            "name": "baseline_tree_soare",
            "description": "Plain SFT on tree search with entropy's opener. "
                           "Controlled comparison against baseline_entropy: "
                           "same opener, so only the policy differs.",
            "seed": 20260817,
            "dedup": False,
            "shuffle": True,
            "components": [
                {"source": "data/tree_soare/train.jsonl", "repeat": 1},
            ],
        },
        "baseline_tree_salet.json": {
            "name": "baseline_tree_salet",
            "description": "Plain SFT on the optimal policy. NOTE: differs from "
                           "baseline_entropy in BOTH opener and policy.",
            "seed": 20260817,
            "dedup": False,
            "shuffle": True,
            "components": [
                {"source": "data/tree_salet/train.jsonl", "repeat": 1},
            ],
        },
        "tree_soare_disagreement_x20.json": {
            "name": "tree_soare_disagreement_x20",
            "description": "Tree policy with its disagreement states amplified "
                           "20x, so the ~1% of decisions that separate it from "
                           "entropy are not swamped by the 99% both agree on.",
            "seed": 20260817,
            "dedup": False,
            "shuffle": True,
            "components": [
                {"source": "data/tree_soare/train.jsonl", "repeat": 1},
                {"source": "data/disagreement/tree_soare.jsonl", "repeat": 20},
            ],
        },
        "dedup_tree_soare.json": {
            "name": "dedup_tree_soare",
            "description": "Distinct states only. Removes the natural "
                           "visitation skew (turn 1 alone repeats 2315x).",
            "seed": 20260817,
            "dedup": True,
            "shuffle": True,
            "components": [
                {"source": "data/tree_soare/train.jsonl", "repeat": 1},
            ],
        },
    }
    os.makedirs(cfgdir, exist_ok=True)
    for name, cfg in recipes.items():
        with open(os.path.join(cfgdir, name), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    print(f"\nwrote {len(recipes)} oversampling recipes to {cfgdir}")

    manifest["notes"] = {
        "model_facing_fields": ["prompt", "completion"],
        "answer_visibility": "No answer appears in any model-facing file. "
                             "_holdout and game_id are both dropped; ids are "
                             "hashed. Answers live only in eval/ and index/.",
        "split": "answer-keyed sha256(salt|answer), identical across solvers",
        "duplication": "Full per-step files preserve the natural visitation "
                       "distribution; meta.duplicate_count and --dedup let you "
                       "choose otherwise at build time.",
        "training": "This package contains NO training code. Nothing was trained.",
    }
    with open(os.path.join(args.dest, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {args.dest}/manifest.json")

    # The README is generated here rather than hand-written, because --clean
    # removes the whole directory and a hand-written file would be lost.
    write_readme(args.dest, manifest)
    print(f"wrote {args.dest}/README.md")
    return 0


def write_readme(dest: str, manifest: Dict) -> None:
    s = manifest["solvers"]
    d = manifest["disagreement"]
    txt = f"""# Wordle SFT / evaluation package

Training and evaluation data from three classical solvers.
**No model has been trained; this package contains no training code.**

Generated by `build_sft_package.py`. Audited by `verify_no_leakage.py`.
Browsed with `inspect_examples.py`. Mixtures built by
`build_weighted_dataset.py`.

## Policies

| Dataset | Mean (2315 games) | Total | Opener |
|---|---:|---:|---|
| `entropy` | 3.4644 | 8020 | SOARE |
| `tree_soare` | 3.4544 | 7997 | SOARE |
| `tree_salet` | 3.4212 (proven optimal) | 7920 | SALET |

`tree_soare` makes the comparison interpretable: `entropy` and `tree_salet`
differ in **both** opener and policy, and the opener is 77% of the gap.

| Comparison | Isolates |
|---|---|
| `entropy` vs `tree_soare` | decision policy (same opener) |
| `tree_soare` vs `tree_salet` | opening word (same policy) |

## Contents

| Dataset | Train | Val | Distinct pairs | Duplication |
|---|---:|---:|---:|---:|
| entropy | {s['entropy']['train_records']} | {s['entropy']['val_records']} | {s['entropy']['distinct_train_pairs']} | {s['entropy']['duplication_factor']}x |
| tree_soare | {s['tree_soare']['train_records']} | {s['tree_soare']['val_records']} | {s['tree_soare']['distinct_train_pairs']} | {s['tree_soare']['duplication_factor']}x |
| tree_salet | {s['tree_salet']['train_records']} | {s['tree_salet']['val_records']} | {s['tree_salet']['distinct_train_pairs']} | {s['tree_salet']['duplication_factor']}x |

```
data/<solver>/train.jsonl        model-facing SFT records
data/<solver>/val.jsonl
data/disagreement/contrast.jsonl {d['disagreement_states']} states, BOTH expert actions
data/disagreement/*.jsonl        oversampling pool ({d['oversampling_pool_states']} val-free states)
eval/val_answers.jsonl           246 held-out answers for rollout scoring
index/id_map.jsonl               id -> answer. ANALYSIS ONLY.
configs/*.json                   oversampling recipes
```

**Only `prompt` and `completion` go to the model.** `meta` is for filtering
and weighting.

## What the model never sees

- The **answer**: not a parameter of `render_prompt`, and structurally
  impossible in history (a correct guess ends the game).
- The **candidate list**.
- The **candidate count**. It is a solver-side quantity a player cannot see and
  it leaks how far the constraints already narrow the answer. Kept in
  `meta.n_candidates` for analysis only.
- `_holdout` and `game_id` are stripped; ids are hashed, since `game_id`
  encodes the answer's row index.

Greens render position by position (`p1=A p2=L ...`), never concatenated:
once all five are known the concatenation would *be* the answer, reachable by
deduction without guessing it. Those states are flagged `fully_determined`.

## Evaluation

75.6% of val records share a `(prompt, completion)` pair with training. This is
inherent to an answer-keyed split — one early-turn state is reached by many
answers — and cannot be fixed by reshuffling.

> Score by **rollout on `eval/val_answers.jsonl`**, not token accuracy.

A paired power analysis on those 246 games gives SE ~0.025-0.046 against true
policy gaps of 0.008-0.043: the set **cannot rank the three experts**. Use it
for large effects (model vs random), opener accuracy, and disagreement-state
agreement, all of which it resolves fine.

## Disagreement set

`entropy` and `tree_soare` share 1394 states and disagree on **{d['disagreement_states']}**
({d['by_turn']} by turn). Those states cover {s['entropy']['disagreement_train_records']} of
{s['entropy']['train_records']} training records (13.1%) and touch 45.3% of games.

The oversampling pool holds **{d['oversampling_pool_states']}** states — those no val game
passes through. The other {d['excluded_touch_val']} are excluded so amplifying them cannot
pull held-out states into training.

## Usage

```bash
python inspect_examples.py --stats
python inspect_examples.py --disagreement --n 10
python verify_no_leakage.py
python build_weighted_dataset.py --list
```

Seed 20260817. Split key `sha256("wordle-split-v1|" + answer)`, identical
across all three datasets.
"""
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    raise SystemExit(main())
