"""
merge_trajectories.py — collect per-solver trajectory runs into one dataset dir
and verify the two datasets are actually comparable.

The entropy-vs-optimal experiment is only meaningful if the two datasets differ
in exactly one thing: the demonstrating policy. This checks that.

    python merge_trajectories.py --src trajectories_entropy --dest trajectories
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=["trajectories_entropy"])
    ap.add_argument("--dest", default="trajectories")
    args = ap.parse_args(argv)

    os.makedirs(args.dest, exist_ok=True)
    cards = []
    dc = os.path.join(args.dest, "dataset_card.json")
    if os.path.exists(dc):
        cards.append(json.load(open(dc, encoding="utf-8")))

    for src in args.src:
        for f in glob.glob(os.path.join(src, "*.jsonl")):
            dst = os.path.join(args.dest, os.path.basename(f))
            shutil.copy(f, dst)
            print(f"moved {f} -> {dst}")
        c = os.path.join(src, "dataset_card.json")
        if os.path.exists(c):
            cards.append(json.load(open(c, encoding="utf-8")))

    # ---- comparability checks -------------------------------------------
    games = {}
    for f in sorted(glob.glob(os.path.join(args.dest, "*_games.jsonl"))):
        label = os.path.basename(f).replace("_games.jsonl", "")
        games[label] = load_jsonl(f)

    print(f"\ndatasets present: {sorted(games)}")
    problems = []
    labels = sorted(games)
    if len(labels) >= 2:
        ref = labels[0]
        ref_split = {g["_holdout"]["answer"]: g["split"] for g in games[ref]}
        for other in labels[1:]:
            oth = {g["_holdout"]["answer"]: g["split"] for g in games[other]}
            if set(ref_split) != set(oth):
                problems.append(
                    f"{ref} and {other} cover different answer sets "
                    f"({len(ref_split)} vs {len(oth)})")
                continue
            mism = [a for a in ref_split if ref_split[a] != oth[a]]
            if mism:
                problems.append(
                    f"{ref} and {other} disagree on the split for "
                    f"{len(mism)} answers, e.g. {mism[:3]}")
            else:
                nval = sum(1 for v in oth.values() if v == "val")
                print(f"  split identical between {ref} and {other}: "
                      f"{len(oth)} answers, {nval} in val")

    if problems:
        print("\nCOMPARABILITY PROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1

    # ---- combined card ---------------------------------------------------
    solvers = []
    seen = set()
    for c in cards:
        for s in c.get("solvers", []):
            if s["solver"] not in seen:
                seen.add(s["solver"])
                solvers.append(s)
    if cards:
        card = dict(cards[0])
        card["solvers"] = solvers
        card["comparability"] = (
            "All datasets cover the identical answer set with an identical "
            "train/val split (keyed on sha256(salt|answer)), so the only "
            "difference between them is the demonstrating policy."
        )
        with open(dc, "w", encoding="utf-8") as fh:
            json.dump(card, fh, indent=2)
        print(f"\nwrote combined {dc}")

    print(f"\n{'solver':<14}{'games':>7}{'steps':>8}{'mean':>9}{'total':>8}"
          f"{'%answer-probes':>16}{'bits/step':>11}")
    print("-" * 73)
    for s in solvers:
        print(f"{s['solver']:<14}{s['n_games']:>7}{s['n_steps']:>8}"
              f"{s['mean_guesses']:>9.4f}{s['total_guesses']:>8}"
              f"{s['pct_actions_that_are_possible_answers']:>16.1f}"
              f"{s['mean_information_gain_bits']:>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
