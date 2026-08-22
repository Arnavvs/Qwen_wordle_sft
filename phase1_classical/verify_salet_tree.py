"""
verify_salet_tree.py — independent verification of the SALET tree-search result.

The run reported a total of 7920 guesses over 2315 answers, which equals the
value Alex Selby proved is the minimum. A claim that strong should not rest on
the same code path that produced it, so this re-derives the result from scratch:

  1. Replays every game with a FRESH solver instance.
  2. Recomputes all feedback with the pure-Python reference `feedback()`,
     never the precomputed matrix.
  3. Verifies every guess is in the 12972-word legal list.
  4. Verifies the solver's guess depends only on the visible history, by
     re-deriving the candidate set independently and feeding it back in.
  5. Recomputes the total and distribution from the replay.

Passing this does NOT prove 7920 is optimal — that is Selby's result, obtained
by exhaustive search. It proves this tree *attains* 7920 legitimately.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import collections
import time

import numpy as np

from wordle_solver import (
    ALL_GREEN, SolverConfig, feedback, filter_history, load_artifacts,
    pattern_to_code,
)
from tree_search import TreeSearchConfig, TreeSearchSolver

OPENER = "salet"


def main() -> int:
    b = load_artifacts("artifacts")
    V = b.vocab
    legal = set(V.guesses)
    cfg = SolverConfig(max_guesses=6, seed=20260817)
    tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                          endgame_threshold=10, opening_guess=OPENER,
                          max_guesses=6)
    solver = TreeSearchSolver(b.fb, cfg, tc)

    total = 0
    dist = collections.Counter()
    worst = 0
    failures = []
    t0 = time.perf_counter()

    for k, answer in enumerate(V.answers, 1):
        history = []                      # [(guess, pattern)] — visible info only
        cands = list(V.answers)           # independent, pure-Python candidate set
        solved = False
        for turn in range(1, 7):
            if turn == 1:
                g = OPENER
            else:
                # Feed the solver a candidate set derived independently of its
                # own bookkeeping, as indices into the answer list.
                idx = np.array(sorted(V.answer_index[w] for w in cands),
                               dtype=np.int32)
                g = solver.choose(idx, turn)

            assert g in legal, f"illegal guess {g!r}"
            pat = feedback(g, answer)                 # pure-Python reference
            history.append((g, pat))
            if pattern_to_code(pat) == ALL_GREEN:
                assert g == answer
                solved = True
                break
            cands = filter_history(V.answers, history)
            assert answer in cands, f"answer eliminated for {answer!r}"
            assert cands, "candidate set emptied"

        n = len(history)
        if not solved:
            failures.append(answer)
            n = 7
        total += n
        dist[n] += 1
        worst = max(worst, n if solved else 7)

        if k % 500 == 0:
            print(f"  {k}/{len(V.answers)}  running total={total}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    n_games = len(V.answers)
    mean = total / n_games
    print("\nINDEPENDENT REPLAY RESULT")
    print(f"  games            {n_games}")
    print(f"  total guesses    {total}")
    print(f"  mean             {mean:.6f}")
    print(f"  worst case       {worst}")
    print(f"  failures         {len(failures)} {failures[:5]}")
    print(f"  distribution     {dict(sorted(dist.items()))}")
    cum = lambda m: 100 * sum(dist[i] for i in range(1, m + 1)) / n_games
    print(f"  <=3 {cum(3):.2f}%   <=4 {cum(4):.2f}%   <=5 {cum(5):.2f}%")
    print(f"  wall             {time.perf_counter()-t0:.0f}s")

    print(f"\n  Selby proven minimum total = 7920  (7920/2315 = {7920/2315:.6f})")
    if total == 7920:
        print("  RESULT: this tree ATTAINS the proven minimum.")
        print("  Optimality of the tree follows from Selby's lower bound,")
        print("  NOT from this search, which used top_k=100 pruning.")
    elif total > 7920:
        print(f"  RESULT: {total - 7920} guesses above the proven minimum "
              f"({mean - 7920/2315:+.6f} per game).")
    else:
        print("  RESULT: BELOW the proven minimum — this indicates a BUG "
              "(illegal play or leakage), not a discovery.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
