"""
todays_wordle.py — run every solver against one specific answer and dump traces.

    python todays_wordle.py --answer tribe --puzzle 1885 --date 2026-08-17

Writes artifacts/todays_wordle.json containing, per solver, the full guess trace
(guess, feedback pattern, surviving candidates, bits remaining) plus timing.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import json
import math
import time

import numpy as np

from wordle_solver import (
    SolverConfig, feedback, load_artifacts, make_solver, play_game,
)
from tree_search import TreeSearchConfig, TreeSearchSolver

SEED = 20260817


def trace(result, label, opener, elapsed, extra=None):
    steps = []
    for i, (g, p) in enumerate(zip(result.guesses, result.patterns)):
        rem = result.candidates_remaining[i]
        steps.append({
            "n": i + 1,
            "guess": g.upper(),
            "pattern": p,
            "remaining": rem,
            "bits": round(math.log2(rem), 3) if rem > 0 else 0.0,
        })
    row = {
        "solver": label,
        "opener": opener.upper() if opener else None,
        "solved": result.solved,
        "n_guesses": result.n_guesses if result.solved else None,
        "score": result.score,
        "steps": steps,
        "seconds": round(elapsed, 4),
    }
    if extra:
        row.update(extra)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", required=True)
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--out", default="artifacts/todays_wordle.json")
    args = ap.parse_args(argv)

    answer = args.answer.strip().lower()
    b = load_artifacts("artifacts")
    V = b.vocab
    if answer not in V.answer_index:
        raise SystemExit(
            f"{answer!r} is not in this project's 2315-word answer list; "
            f"the solvers cannot treat it as a hypothesis."
        )

    cfg = SolverConfig(max_guesses=6, seed=SEED, guess_pool="full")
    rows = []

    CLASSIC = ["random", "frequency", "entropy", "expected", "minimax", "hybrid"]
    for strat in CLASSIC:
        sv = make_solver(strat, b.fb, cfg, b.model)
        sv.reset()
        op = sv.opening_guess() if sv.deterministic else None
        t0 = time.perf_counter()
        r = play_game(sv, answer, first_guess=op)
        el = time.perf_counter() - t0
        rows.append(trace(r, strat, op or r.guesses[0], el,
                          {"deterministic": sv.deterministic}))
        print(f"{strat:<12} {r.n_guesses if r.solved else 'FAIL'} guesses  "
              f"{' -> '.join(g.upper() for g in r.guesses)}", flush=True)

    TREES = [
        ("tree depth-2 (SOARE)", "soare", 2),
        ("tree (SALET)", "salet", 6),
    ]
    for label, opener, depth in TREES:
        tc = TreeSearchConfig(depth=depth, top_k=100, endgame_top_k=60,
                              endgame_threshold=10, opening_guess=opener,
                              max_guesses=6)
        sv = TreeSearchSolver(b.fb, cfg, tc)
        t0 = time.perf_counter()
        r = play_game(sv, answer, first_guess=opener)
        el = time.perf_counter() - t0
        rows.append(trace(r, label, opener, el, {
            "deterministic": True,
            "nodes": sv.tree.stats.nodes_evaluated,
            "cache": sv.tree.cache_size,
            "regime": sv.describe_regime(),
        }))
        print(f"{label:<22} {r.n_guesses if r.solved else 'FAIL'} guesses  "
              f"{' -> '.join(g.upper() for g in r.guesses)}", flush=True)

    payload = {
        "answer": answer.upper(),
        "puzzle": args.puzzle,
        "date": args.date,
        "vocab": {"answers": V.n_answers, "guesses": V.n_guesses},
        "seed": SEED,
        "start_bits": round(math.log2(V.n_answers), 3),
        "results": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
