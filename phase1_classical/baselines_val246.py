"""
baselines_val246.py — classical solvers on the 246 held-out answers.

The SFT models are evaluated on exactly this answer set, so their scores must be
compared against classical scores on the SAME set, never against the full
2315-game benchmark. Those numbers are not interchangeable and are labelled
separately everywhere.

    python baselines_val246.py
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import statistics
import time

import numpy as np

from wordle_solver import SolverConfig, load_artifacts, make_solver, play_game
from tree_search import TreeSearchConfig, TreeSearchSolver

SEED = 20260817
VAL = "sft_package/eval/val_answers.jsonl"


def summarize(label, games, opener, wall):
    n = len(games)
    scores = [g.score for g in games]
    solved = [g.n_guesses for g in games if g.solved]
    dist = {k: sum(1 for g in games if g.solved and g.n_guesses == k)
            for k in range(1, 7)}
    cum = lambda m: 100 * sum(dist[i] for i in range(1, m + 1)) / n
    return {
        "solver": label,
        "opener": opener.upper() if opener else None,
        "n_games": n,
        "mean_failures_as_7": round(sum(scores) / n, 4),
        "mean_solved_only": round(sum(solved) / len(solved), 4) if solved else None,
        "median": statistics.median(solved) if solved else None,
        "std": round(statistics.stdev(solved), 4) if len(solved) > 1 else 0.0,
        "max": max(solved) if solved else None,
        "pct_le3": round(cum(3), 2),
        "pct_le4": round(cum(4), 2),
        "pct_le5": round(cum(5), 2),
        "pct_le6": round(cum(6), 2),
        "failures": sum(1 for g in games if not g.solved),
        "failure_rate_pct": round(100 * sum(1 for g in games if not g.solved) / n, 2),
        "distribution": dist,
        "wall_s": round(wall, 1),
        "avg_candidates_after_turn": {
            str(t): round(float(np.mean([g.candidates_remaining[t - 1]
                                         for g in games
                                         if len(g.candidates_remaining) >= t])), 2)
            for t in range(1, 7)
            if any(len(g.candidates_remaining) >= t for g in games)
        },
    }


def main() -> int:
    answers = [json.loads(l)["answer"].lower()
               for l in open(VAL, encoding="utf-8")]
    print(f"evaluating {len(answers)} held-out answers\n")

    b = load_artifacts("artifacts")
    cfg = SolverConfig(max_guesses=6, seed=SEED, guess_pool="full")
    rows = []

    for label in ["random", "frequency", "entropy"]:
        sv = make_solver(label, b.fb, cfg, b.model)
        sv.reset()
        op = sv.opening_guess() if sv.deterministic else None
        t0 = time.perf_counter()
        games = [play_game(sv, a, first_guess=op) for a in answers]
        rows.append(summarize(label, games, op, time.perf_counter() - t0))
        print(f"  {label:<14} mean={rows[-1]['mean_failures_as_7']:.4f} "
              f"opener={rows[-1]['opener']}")

    for label, opener in [("tree_soare", "soare"), ("tree_salet", "salet")]:
        tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                              endgame_threshold=10, opening_guess=opener,
                              max_guesses=6)
        sv = TreeSearchSolver(b.fb, cfg, tc)
        t0 = time.perf_counter()
        games = [play_game(sv, a, first_guess=opener) for a in answers]
        rows.append(summarize(label, games, opener, time.perf_counter() - t0))
        print(f"  {label:<14} mean={rows[-1]['mean_failures_as_7']:.4f} "
              f"opener={rows[-1]['opener']}")

    out = {
        "note": "Classical solvers on the 246 HELD-OUT answers only. NOT "
                "comparable to the full 2315-game benchmark.",
        "n_answers": len(answers),
        "seed": SEED,
        "rows": rows,
    }
    with open("artifacts/val246_baselines.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'solver':<14}{'open':<7}{'mean':>8}{'med':>5}{'max':>5}"
          f"{'<=3':>8}{'<=4':>8}{'<=5':>8}{'fail%':>7}")
    print("-" * 70)
    for r in rows:
        print(f"{r['solver']:<14}{(r['opener'] or '-'):<7}"
              f"{r['mean_failures_as_7']:>8.4f}{r['median']:>5.0f}{r['max']:>5}"
              f"{r['pct_le3']:>8.2f}{r['pct_le4']:>8.2f}{r['pct_le5']:>8.2f}"
              f"{r['failure_rate_pct']:>7.2f}")
    print("\nwrote artifacts/val246_baselines.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
