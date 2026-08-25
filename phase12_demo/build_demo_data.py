"""build_demo_data.py — the 20-game data blob for the playable demo.

Picks 20 of the 246 held-out answers and, for each, records how every solver in
the project actually played it — not just the score, but the full guess
sequence, so the demo can replay a solver move by move.

WHY 20, AND WHICH 20
--------------------
Taking the first 20, or 20 at random, would be dominated by 3- and 4-guess
games and would flatter every solver equally. Instead the 20 are stratified by
how hard the SFT model found them: a fixed spread across its per-game scores,
so the set contains games it solved in 2 and games it failed outright. That
makes the comparison legible rather than a wall of near-ties.

The selection is deterministic (fixed seed, sorted keys) so the demo is
reproducible.

SOURCES
-------
Model arms come from the harness results already in `results/` — the same
per-game records the paired tests were computed from, so nothing is re-measured
and nothing can disagree with the published numbers.

Classical solvers have no per-game record on disk (only aggregates in
`artifacts/val246_baselines.json`), so they are replayed here against the exact
same answers through `core/wordle_solver.play_game`.

    python phase12_demo/build_demo_data.py
"""
import _paths  # noqa: F401

import json
import os

import numpy as np

from wordle_solver import (SolverConfig, load_artifacts, make_solver,
                           play_game)
from tree_search import TreeSearchConfig, TreeSearchSolver

N_GAMES = 20
SEED = 20260826
OUT = "phase12_demo/demo_data.json"

# (label, source file, key in games{})  -- the model arms, already measured
MODEL_ARMS = [
    ("SFT + adaptive decoder", "results/phase11/grpo_checkpoints_246.json",
     "sft|baseline", "the published 3.7642 policy"),
    ("SFT + GRPO", "results/phase11/grpo_checkpoints_246.json",
     "sft_grpo|baseline", "Phase 11: RL on exact rewards, no significant change"),
    ("SFT, no constraint block", "results/phase10/crossover_2x2_harness_results.json",
     "sft|raw_history", "Phase 10: same model, prompt stripped of deductions"),
]

CLASSICAL = [
    ("Classical entropy", "entropy", "maximise information gain; the best solver here"),
    ("Classical frequency", "frequency", "rank surviving words by letter frequency"),
    ("Classical random", "random", "pick any surviving word at random"),
]


def pick_games(answers, scores, n):
    """Stratified by difficulty so the set is not all 3s and 4s."""
    order = sorted(range(len(answers)), key=lambda i: (scores[i], answers[i]))
    # even slice across the difficulty-sorted list, then a stable shuffle
    idx = [order[round(i * (len(order) - 1) / (n - 1))] for i in range(n)]
    idx = sorted(set(idx))
    while len(idx) < n:                       # ties collapsed a slot; backfill
        for j in order:
            if j not in idx:
                idx.append(j); break
        idx = sorted(set(idx))
    rng = np.random.default_rng(SEED)
    idx = list(rng.permutation(idx))[:n]
    return [int(i) for i in idx]


def main():
    bundle = load_artifacts("artifacts", mmap=True)
    ph11 = json.load(open(MODEL_ARMS[0][1], encoding="utf-8"))["games"]
    ph10 = json.load(open(MODEL_ARMS[2][1], encoding="utf-8"))["games"]
    src = {MODEL_ARMS[0][1]: ph11, MODEL_ARMS[2][1]: ph10}

    ref = ph11["sft|baseline"]
    answers, scores = ref["answers"], ref["per_game"]
    # both files must be the same 246 in the same order, or rows would misalign
    assert ph10["sft|baseline"]["answers"] == answers, "answer order differs"
    picks = pick_games(answers, scores, N_GAMES)
    # the harness stores answers uppercase; the solver vocabulary is lowercase
    chosen = [answers[i].upper() for i in picks]
    print(f"picked {len(chosen)} games, SFT scores "
          f"{sorted(scores[i] for i in picks)}")

    solvers = {}
    for label, path, key, note in MODEL_ARMS:
        g = src[path][key]
        solvers[label] = {
            "note": note, "kind": "model",
            "mean246": g["mean"], "solved246": g["solved"],
            "games": [{"guesses": [w.upper() for w in g["per_guess"][i]],
                       "score": g["per_game"][i]} for i in picks],
        }

    cfg = SolverConfig(max_guesses=6, seed=SEED, guess_pool="full")
    base = json.load(open("artifacts/val246_baselines.json", encoding="utf-8"))
    agg = {r["solver"]: r for r in base["rows"]}
    for label, name, note in CLASSICAL:
        sv = make_solver(name, bundle.fb, cfg, bundle.model)
        sv.reset()
        games = []
        for a in chosen:
            r = play_game(sv, a.lower())
            games.append({"guesses": [w.upper() for w in r.guesses],
                          "score": r.score})
        row = agg.get(name, {})
        solvers[label] = {"note": note, "kind": "classical",
                          "mean246": row.get("mean_failures_as_7"),
                          "solved246": row.get("n_games", 246) - round(
                              row.get("failure_rate_pct", 0) / 100 * 246),
                          "games": games}
        print(f"  {label:24s} on these 20: "
              f"{sum(g['score'] for g in games)/len(games):.2f}")

    # A perfect-play reference: the lookahead tree, so the demo can show the
    # true floor rather than implying `entropy` is optimal.
    tree = TreeSearchSolver(bundle.fb, cfg, TreeSearchConfig(
        depth=6, top_k=100, endgame_top_k=60, endgame_threshold=10,
        opening_guess="salet", max_guesses=6))
    games = []
    for a in chosen:
        r = play_game(tree, a.lower())
        games.append({"guesses": [w.upper() for w in r.guesses],
                      "score": r.score})
    solvers["Lookahead tree (best known)"] = {
        "note": "depth-6 search; the practical floor for this word list",
        "kind": "classical", "mean246": None, "solved246": None, "games": games}
    print(f"  {'Lookahead tree':24s} on these 20: "
          f"{sum(g['score'] for g in games)/len(games):.2f}")

    legal = sorted(w.upper() for w in bundle.vocab.guesses)
    data = {
        "n_games": N_GAMES, "seed": SEED,
        "answers": list(chosen),
        "solvers": solvers,
        "legal": "".join(legal),          # 5-char fixed width, no separators
        "n_legal": len(legal),
        "provenance": {
            "held_out_set": 246,
            "source_model": "Qwen2.5-0.5B-Instruct + LoRA, adaptive decoder @20",
            "note": "Model rows are the exact per-game records the paired tests "
                    "used; classical rows were replayed on the same answers.",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB, "
          f"{len(legal)} legal words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
