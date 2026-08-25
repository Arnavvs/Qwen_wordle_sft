"""
hybrid_sweep.py — full-scale lambda / nu / mu sweep for the hybrid solver.

The notebook's sweep runs on a 300-answer subsample, which is too noisy to
settle whether the hybrid's hand-set weights actually beat plain entropy. This
runs the same grid over all 2315 answers so the comparison is exact.

Recall the scoring rule (all four terms in bits):

    score(g) = a*H(g) - lam*log2(worst_case) - nu*log2(E[remaining])
               + mu*log2(1 + p_answer(g))

lam = nu = mu = 0 reduces the hybrid to a pure max-entropy rule, so that row is
the control: it should reproduce the entropy solver's number up to tie-breaking.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import time

from wordle_solver import SolverConfig, load_artifacts
from benchmark import run_benchmark, summarize

GRID = [
    # (lam, nu, mu)  -- lam=nu=mu=0 is the pure-entropy control
    (0.00, 0.00, 0.00),
    (0.00, 0.00, 1.00),
    (0.25, 0.25, 1.00),   # shipped default
    (0.50, 0.25, 1.00),
    (0.25, 0.50, 1.00),
    (0.10, 0.10, 1.00),
    (0.50, 0.50, 1.00),
    (1.00, 1.00, 1.00),
]


def main() -> int:
    b = load_artifacts("artifacts")
    answers = b.vocab.answers
    rows = []
    print(f"full 2315-answer hybrid sweep (seed 20260817, guess_pool=full)\n")
    print(f"{'lam':>6}{'nu':>6}{'mu':>6}{'opener':>8}{'mean':>9}{'max':>5}"
          f"{'<=3':>8}{'<=4':>8}{'fail':>6}{'wall_s':>9}")
    print("-" * 71)
    for lam, nu, mu in GRID:
        cfg = SolverConfig(max_guesses=6, seed=20260817, guess_pool="full",
                           entropy_weight=1.0, minimax_weight=lam,
                           expected_weight=nu, answer_bonus_weight=mu)
        t0 = time.perf_counter()
        res = run_benchmark("hybrid", b.fb, cfg, answers=answers,
                            model=b.model, log=None)
        s = summarize(res, max_guesses=6)
        dc = {int(k): v for k, v in s["distribution_counts"].items()}
        n = s["n_games"]
        cum = lambda m: 100 * sum(dc.get(i, 0) for i in range(1, m + 1)) / n
        row = {"lam": lam, "nu": nu, "mu": mu,
               "opener": s["opening_guess"],
               "mean": s["mean_score_failures_penalized"],
               "max": s["max_guesses"], "le3": cum(3), "le4": cum(4),
               "fails": s["n_failed"], "wall_s": s["wall_s"]}
        rows.append(row)
        print(f"{lam:>6.2f}{nu:>6.2f}{mu:>6.2f}{row['opener']:>8}"
              f"{row['mean']:>9.4f}{row['max']:>5}{row['le3']:>8.2f}"
              f"{row['le4']:>8.2f}{row['fails']:>6}{row['wall_s']:>9.0f}",
              flush=True)

    with open("artifacts/hybrid_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    best = min(rows, key=lambda r: r["mean"])
    print(f"\nbest configuration in grid: lam={best['lam']} nu={best['nu']} "
          f"mu={best['mu']} -> {best['mean']:.4f} (opener {best['opener'].upper()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
