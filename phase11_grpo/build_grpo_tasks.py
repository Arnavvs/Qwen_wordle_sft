"""build_grpo_tasks.py — the reward table GRPO trains against.

WHY A PRECOMPUTED TABLE
-----------------------
GRPO needs a reward for every action it samples. Computing one on the GPU would
mean shipping the solver into the training loop and paying a tree search per
sample per step — and it would put the measurement path inside the training
path, which is the mistake that voided Phase 8 v3.

Instead every reward is computed here, once, exactly, and shipped as data. The
Kaggle notebook then does only what a GPU is for: sample, look up, update. It
never scores anything itself, so the reward cannot drift from the one the
evaluation harness would assign.

WHAT A TASK IS
--------------
    prompt      the rendered state, identical to every other phase
    actions     the action space the deployed decoder offers at that state
    values      expected further guesses for each action (LOWER IS BETTER)

Reward is `-value`, and GRPO normalises within the group anyway, so the scale
is irrelevant — only the ordering and the spread inside one state matter.

TWO REGIMES, AND WHY THE SPLIT IS WHERE IT IS
---------------------------------------------
`|admissible| <= 20` — the adaptive decoder restricts the model here, so the
action space IS the admissible set, and `dpo_core.AdaptiveTree` solves the
resulting game exactly to full depth in integer arithmetic. Every value is
exact; ties are exact ties.

`|admissible| > 20` — the model may emit any legal word, and nothing exact is
affordable: one lookahead valuation of a single turn-2 state costs ~13.5s at 83
candidates and grows sharply from there, so pricing top-48 actions over even
150 states runs to tens of hours. This regime therefore gets an exhaustive
*one-ply* E[remaining] over the full legal pool, top-k kept, and every such task
is labelled `exact: false`.

That is a real weakening and it is stated rather than hidden. It is also not a
weak signal: greedily minimising E[remaining] is the classical `expected`
solver, 3.4812 over 2,315 games against `entropy`'s 3.4431 — both far ahead of
the model's 3.7642.

The two regimes report values in different units (further guesses vs remaining
candidates). That is safe here and only here: GRPO normalises advantages within
a group, a group is a single state, so the currencies never meet inside one
update. Any statistic that pooled values across states would be meaningless.

    python phase11_grpo/build_grpo_tasks.py
"""
import _paths  # noqa: F401

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from dpo_core import (ADAPTIVE_THRESHOLD, MAX_GUESSES, AdaptiveTree, Board,
                      Vocab, bucket_of, load_answer_splits, sha)
from wordle_solver import ALL_GREEN

SEED = 20260825
OPENER = "salet"
FORMAT_VERSION = 1

DEFAULT_TRAIN = {"2-10": 2500, "11-100": 500, "100+": 150}
DEFAULT_VAL = {"2-10": 300, "11-100": 60, "100+": 20}


def price_unrestricted(V, board, top_k):
    """One-ply E[remaining candidates] for the top-k legal actions.

    WHY NOT LOOKAHEAD. A full lookahead valuation of one turn-2 state costs
    ~13.5s at 83 candidates and grows sharply with the candidate count;
    pricing top-48 actions across even 150 states is tens of hours. Measured,
    not assumed — see `decision_budget.py`.

    So this regime gets an exhaustive one-ply value instead, and is labelled
    `exact: false`. That is a weaker signal but not a weak one: greedily
    minimising E[remaining] is exactly the classical `expected` solver, which
    scores 3.4812 over 2,315 games against `entropy`'s 3.4431 — both far ahead
    of the model's 3.7642. A policy that followed this proxy well at turn 2
    would be a large improvement, so it is a useful thing to reward.

    UNITS DIFFER FROM THE RESTRICTED REGIME, AND THAT IS SAFE. Values here are
    expected remaining *candidates*; restricted values are further *guesses*.
    GRPO normalises advantages within a group, and a group is one state, so the
    two currencies never meet inside a single update. Anything that pooled
    values across states would be wrong — nothing here does.
    """
    S = np.sort(V.answer_col[board.candidates].astype(np.int32))
    stats = V.bundle.fb.partition_stats(np.arange(V.n, dtype=np.int32), S)
    er = stats["expected_remaining"]
    # Recover the exact integer numerator so ties are exact rather than
    # float-approximate, as in the Phase 8 v3 generator.
    n = len(S)
    sumsq = np.rint(er * n).astype(np.int64)
    order = np.argsort(sumsq, kind="stable")[:top_k]
    return {int(g): float(sumsq[int(g)]) / n for g in order}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft_package/data/grpo_tasks")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-rollouts", type=int, default=400000)
    ap.add_argument("--unrestricted-top-k", type=int, default=48)
    ap.add_argument("--max-per-answer", type=int, default=6)
    for b in ("2-10", "11-100", "100+"):
        f = b.replace("-", "_").replace("+", "plus")
        ap.add_argument(f"--train-{b}", dest=f"train_{f}", type=int,
                        default=DEFAULT_TRAIN[b])
        ap.add_argument(f"--val-{b}", dest=f"val_{f}", type=int,
                        default=DEFAULT_VAL[b])
    a = ap.parse_args(argv)

    def q(pre, tbl):
        return {b: getattr(a, f"{pre}_{b.replace('-','_').replace('+','plus')}")
                for b in tbl}
    quotas = {"train": q("train", DEFAULT_TRAIN), "validation": q("val", DEFAULT_VAL)}

    V = Vocab()
    tree = AdaptiveTree(V)
    train_answers, val_answers = load_answer_splits()
    holdout = set(val_answers)
    opener = V.index[OPENER]

    print("=" * 72)
    print("GRPO TASK SET — exact rewards, precomputed")
    print("=" * 72)
    print(f"seed={a.seed}  threshold={ADAPTIVE_THRESHOLD}  "
          f"unrestricted_top_k={a.unrestricted_top_k}")
    print(f"quotas: {quotas}")

    out = {}
    # A state reachable from a training answer is often reachable from a
    # held-out one too, so `seen` carries forward: validation may only claim
    # states training did not already take. Without this the two splits share
    # states and the validation number is not held out at all.
    claimed = set()
    for split, answers in (("train", train_answers), ("validation", val_answers)):
        rng = np.random.default_rng(a.seed + (0 if split == "train" else 1))
        rows, got, per_ans, rej = [], Counter(), Counter(), Counter()
        seen = set(claimed)
        want = quotas[split]
        t0, rollouts = time.perf_counter(), 0
        print(f"\n-- {split} --")
        while any(got[b] < want[b] for b in want) and rollouts < a.max_rollouts:
            rollouts += 1
            ans = answers[int(rng.integers(len(answers)))]
            if per_ans[ans] >= a.max_per_answer:
                continue
            ai = V.index[ans]
            board = Board(V)
            g = opener
            for _ in range(MAX_GUESSES - 1):
                code = V.code_against(g, ai)
                if code == ALL_GREEN:
                    break
                board = board.play(g, code)
                na = len(board.admissible)
                b = bucket_of(na)
                key = board.state_key()
                if (b and got[b] < want[b] and key not in seen
                        and len(board.candidates) >= 2 and board.turn <= MAX_GUESSES):
                    rec = _task(V, tree, board, ans, split, holdout,
                                a.unrestricted_top_k, rej)
                    if rec is not None:
                        rows.append(rec); seen.add(key)
                        got[b] += 1; per_ans[ans] += 1
                        break
                if len(board.candidates) == 1:
                    break
                adm = board.admissible
                g = (int(adm[rng.integers(len(adm))]) if na <= 40
                     else int(board.candidates[rng.integers(len(board.candidates))]))
            if rollouts % 2000 == 0:
                print(f"    {rollouts:6d} rollouts  {len(rows):5d} tasks  "
                      f"{dict(sorted(got.items()))}  {time.perf_counter()-t0:.0f}s",
                      flush=True)
        out[split] = rows
        claimed |= {r["meta"]["state_key"] for r in rows}
        print(f"  {split}: {len(rows)} tasks in {time.perf_counter()-t0:.0f}s  "
              f"rejects={dict(rej)}")

    tk = {r["meta"]["state_key"] for r in out["train"]}
    vk = {r["meta"]["state_key"] for r in out["validation"]}
    assert not tk & vk, "train/validation state leakage"
    assert not ({r["prompt"] for r in out["train"]}
                & {r["prompt"] for r in out["validation"]}), "prompt leakage"

    os.makedirs(a.out, exist_ok=True)
    for split, rows in out.items():
        p = os.path.join(a.out, f"{split}.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"wrote {len(rows):5d} -> {p}")

    man = {
        "format_version": FORMAT_VERSION, "seed": a.seed,
        "adaptive_threshold": ADAPTIVE_THRESHOLD, "opener": OPENER,
        "unrestricted_top_k": a.unrestricted_top_k,
        "counts": {k: len(v) for k, v in out.items()},
        "buckets": {k: dict(Counter(r["meta"]["bucket"] for r in v))
                    for k, v in out.items()},
        "reward": "value = expected further guesses, LOWER IS BETTER; "
                  "reward = -value; GRPO normalises within the group",
        "value_families": {
            "restricted": "exact adaptive-decoder tree, full remaining depth "
                          "(units: mean further guesses)",
            "unrestricted": f"exhaustive one-ply E[remaining] over the legal "
                            f"pool, top-{a.unrestricted_top_k} kept "
                            "(units: expected remaining candidates)",
            "units_differ_by_regime": "safe: GRPO normalises within a group, "
                                      "and a group is one state",
        },
        "leakage": {"train_actions_exclude_holdout_answers": True,
                    "train_val_state_disjoint": True},
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("\n" + json.dumps(man["buckets"], sort_keys=True))
    return 0


def _task(V, tree, board, answer, split, holdout, top_k, rej):
    na = len(board.admissible)
    if na <= ADAPTIVE_THRESHOLD:
        totals = tree.action_costs(board)
        n = len(board.candidates)
        vals = {g: t / n for g, t in totals.items()}
        family, exact = "exact_adaptive_decoder_tree", True
    else:
        vals = price_unrestricted(V, board, top_k)
        family, exact = "one_ply_expected_remaining_top_k", False
    if len(vals) < 2:
        rej["too_few_actions"] += 1
        return None
    best = min(vals.values())
    if all(abs(v - best) < 1e-12 for v in vals.values()):
        rej["all_actions_equal"] += 1      # nothing for a gradient to prefer
        return None
    # Held-out answers must never be reinforced. HOW that is enforced differs
    # by regime, and the difference matters:
    #
    # restricted - the action space IS the admissible set, so removing a word
    #   would train the policy against a different action set than the decoder
    #   presents at deployment, and could leave a suboptimal action sitting at
    #   rank 1. So the whole task is DROPPED instead. The action space stays
    #   faithful or the task does not exist.
    #
    # unrestricted - the menu is already a top-k slice of 12,972, so excluding
    #   a word changes nothing structurally; it just shortens the menu.
    if split == "train":
        if na <= ADAPTIVE_THRESHOLD:
            if any(V.words[g] in holdout for g in vals):
                rej["holdout_in_admissible_set"] += 1
                return None
        else:
            vals = {g: v for g, v in vals.items() if V.words[g] not in holdout}
            if len(vals) < 2:
                rej["holdout_filtered_out"] += 1
                return None
    words = sorted(vals, key=lambda g: (vals[g], V.words[g]))
    return {
        "id": "grpo-" + sha(split, board.state_key())[:16],
        "prompt": board.prompt(),
        "actions": [V.words[g].upper() for g in words],
        "values": [round(float(vals[g]), 8) for g in words],
        "meta": {
            "format_version": FORMAT_VERSION, "split": split,
            "state_key": board.state_key(),
            "state_hash": sha("state", board.state_key())[:20],
            "source_answer_hash": sha("phase11-source", answer)[:16],
            "turn": board.turn, "guesses_left": board.guesses_left,
            "n_candidates": int(len(board.candidates)),
            "n_admissible": int(na),
            "bucket": bucket_of(na),
            "action_space": board.action_space,
            "n_actions": len(words),
            "value_family": family, "exact": exact,
            "value_units": "mean_further_guesses",
            "value_direction": "lower_is_better",
            "best_value": round(float(min(vals.values())), 8),
            "worst_value": round(float(max(vals.values())), 8),
            "value_spread": round(float(max(vals.values()) - min(vals.values())), 8),
            "n_optimal_actions": sum(1 for v in vals.values()
                                     if abs(v - min(vals.values())) < 1e-12),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
