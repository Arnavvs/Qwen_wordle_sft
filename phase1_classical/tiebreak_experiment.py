"""
tiebreak_experiment.py — how much of entropy's 3.4644 is the tie-break?

46% of mid-game states have exact entropy ties (mean ~165 tied guesses), so the
secondary criterion is not a detail. This runs the full 2315-answer benchmark
under four tie-break rules that are all "maximum entropy" solvers, differing
only in how they break exact ties.

    A  candidate-preference, first index   (the shipped EntropySolver)
    B  no preference, first index          (pure argmax)
    C  candidate-preference, last index    (probes index-order sensitivity)
    D  candidate-preference, then minimum expected-remaining as secondary key

D is the only principled one: among equally informative guesses, prefer the one
leaving the smaller expected posterior.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import time

import numpy as np

from wordle_solver import (
    EntropySolver, FeedbackMatrix, SolverConfig, load_artifacts,
)
from benchmark import run_benchmark, summarize


class NoPreferenceEntropy(EntropySolver):
    """B: pure argmax over entropy, no candidate preference."""
    name = "entropy_noprefer"

    def choose(self, cand_idx, turn):
        if len(cand_idx) == 1:
            return self.vocab.answers[int(cand_idx[0])]
        pool = self._pool(cand_idx, turn)
        vals = self.fb.partition_stats(pool, cand_idx)["entropy"]
        return self.vocab.guesses[int(pool[int(np.argmax(vals))])]


class LastIndexEntropy(EntropySolver):
    """C: candidate preference, but resolve ties toward the LAST index."""
    name = "entropy_lastidx"

    def choose(self, cand_idx, turn):
        if len(cand_idx) == 1:
            return self.vocab.answers[int(cand_idx[0])]
        pool = self._pool(cand_idx, turn)
        vals = self.fb.partition_stats(pool, cand_idx)["entropy"].astype(np.float64)
        cand_set = set(int(x) for x in self.fb.answer_pos[cand_idx])
        is_cand = np.fromiter((int(g) in cand_set for g in pool),
                              dtype=bool, count=len(pool))
        adj = vals + 1e-9 * is_cand
        best = adj.max()
        tied = np.flatnonzero(adj >= best - 1e-12)
        return self.vocab.guesses[int(pool[int(tied[-1])])]


class ExpectedTieBreakEntropy(EntropySolver):
    """D: max entropy, ties broken by minimum expected remaining, then candidate."""
    name = "entropy_exp_tiebreak"

    def choose(self, cand_idx, turn):
        if len(cand_idx) == 1:
            return self.vocab.answers[int(cand_idx[0])]
        pool = self._pool(cand_idx, turn)
        st = self.fb.partition_stats(pool, cand_idx)
        ent = st["entropy"].astype(np.float64)
        best = ent.max()
        tied = np.flatnonzero(ent >= best - 1e-12)
        if len(tied) > 1:
            exp = st["expected_remaining"][tied]
            keep = np.flatnonzero(exp <= exp.min() + 1e-12)
            tied = tied[keep]
        if len(tied) > 1:
            cand_set = set(int(x) for x in self.fb.answer_pos[cand_idx])
            is_c = np.fromiter((int(pool[t]) in cand_set for t in tied),
                               dtype=bool, count=len(tied))
            if is_c.any():
                tied = tied[np.flatnonzero(is_c)]
        return self.vocab.guesses[int(pool[int(tied[0])])]


VARIANTS = {
    "A_shipped":        None,                    # the stock EntropySolver
    "B_no_preference":  NoPreferenceEntropy,
    "C_last_index":     LastIndexEntropy,
    "D_expected_break": ExpectedTieBreakEntropy,
}


def main() -> int:
    b = load_artifacts("artifacts")
    cfg = SolverConfig(max_guesses=6, seed=20260817, guess_pool="full")
    answers = b.vocab.answers

    rows = {}
    for name, cls in VARIANTS.items():
        t0 = time.perf_counter()
        if cls is None:
            res = run_benchmark("entropy", b.fb, cfg, answers=answers,
                                model=b.model, log=None)
        else:
            solver = cls(b.fb, cfg)
            solver.reset()
            op = solver.opening_guess()
            from wordle_solver import play_game
            from benchmark import BenchmarkResult
            games = [play_game(solver, a, first_guess=op) for a in answers]
            res = BenchmarkResult(strategy=name, games=games,
                                  wall_s=time.perf_counter() - t0,
                                  config=cfg.to_dict(),
                                  n_answers_evaluated=len(answers),
                                  opening_guess=op)
        s = summarize(res, max_guesses=6)
        dc = {int(k): v for k, v in s["distribution_counts"].items()}
        n = s["n_games"]
        rows[name] = {
            "opener": s["opening_guess"],
            "mean": s["mean_score_failures_penalized"],
            "max": s["max_guesses"],
            "le3": 100 * sum(dc.get(i, 0) for i in range(1, 4)) / n,
            "le4": 100 * sum(dc.get(i, 0) for i in range(1, 5)) / n,
            "fails": s["n_failed"],
            "wall": s["wall_s"],
        }
        print(f"{name:<18} opener={rows[name]['opener']:<7} "
              f"mean={rows[name]['mean']:.4f}  max={rows[name]['max']}  "
              f"<=3={rows[name]['le3']:.2f}%  <=4={rows[name]['le4']:.2f}%  "
              f"fails={rows[name]['fails']}  ({rows[name]['wall']:.0f}s)",
              flush=True)

    with open("artifacts/tiebreak_experiment.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    ms = [r["mean"] for r in rows.values()]
    print(f"\nspread across tie-break rules: {min(ms):.4f} .. {max(ms):.4f} "
          f"(delta {max(ms)-min(ms):.4f} guesses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
