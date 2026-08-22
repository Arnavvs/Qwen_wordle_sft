"""
run_tree_benchmark.py — staged evaluation of the game-tree search solver.

Staged deliberately (100 -> 500 -> 1000 -> 2315) so scaling is measured before
the full run is attempted, and so a configuration that turns out to be
intractable is discovered cheaply.

    python run_tree_benchmark.py --stage depth      # depth 1/2/3/6 on a subset
    python run_tree_benchmark.py --stage topk       # top-K pruning sweep
    python run_tree_benchmark.py --stage scaling    # 100/500/1000 scaling
    python run_tree_benchmark.py --stage full       # all 2315, best config
    python run_tree_benchmark.py --stage keys       # state-key representation
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import json
import os
import time

import numpy as np

from wordle_solver import SolverConfig, load_artifacts, play_game, make_solver
from benchmark import BenchmarkResult, summarize
from tree_search import (
    GameTree, TreeSearchConfig, TreeSearchSolver, compare_state_keys,
)

HERE = _paths.ROOT  # data lives at the repo root
SEED = 20260817


def sample(answers, n, seed=SEED):
    """Fixed random subset — alphabetical prefixes are badly biased."""
    if n <= 0 or n >= len(answers):
        return list(answers)
    rs = np.random.default_rng(seed)
    return [answers[i] for i in sorted(rs.choice(len(answers), n, replace=False))]


def run_tree(fb, cfg, tcfg, answers, label):
    sv = TreeSearchSolver(fb, cfg, tcfg)
    opener = sv.opening_guess()
    t0 = time.perf_counter()
    games = [play_game(sv, a, first_guess=opener) for a in answers]
    wall = time.perf_counter() - t0
    res = BenchmarkResult(strategy=label, games=games, wall_s=wall,
                          config=cfg.to_dict(), n_answers_evaluated=len(answers),
                          opening_guess=opener)
    s = summarize(res, max_guesses=cfg.max_guesses)
    s["tree_stats"] = sv.tree.stats.as_dict()
    s["cache_size"] = sv.tree.cache_size
    s["regime"] = sv.describe_regime()
    return s


def run_baseline(fb, model, cfg, strategy, answers, label=None):
    solver = make_solver(strategy, fb, cfg, model)
    solver.reset()
    op = solver.opening_guess() if solver.deterministic else None
    t0 = time.perf_counter()
    games = [play_game(solver, a, first_guess=op) for a in answers]
    wall = time.perf_counter() - t0
    res = BenchmarkResult(strategy=label or strategy, games=games, wall_s=wall,
                          config=cfg.to_dict(), n_answers_evaluated=len(answers),
                          opening_guess=op)
    return summarize(res, max_guesses=cfg.max_guesses)


def row(s):
    dc = {int(k): v for k, v in s["distribution_counts"].items()}
    n = s["n_games"]
    cum = lambda m: 100 * sum(dc.get(i, 0) for i in range(1, m + 1)) / n
    return {
        "solver": s["strategy"], "n": n,
        "mean": s["mean_score_failures_penalized"],
        "median": s["median_guesses"], "std": s["std_guesses"],
        "max": s["max_guesses"],
        "le3": cum(3), "le4": cum(4), "le5": cum(5), "le6": cum(6),
        "fails": s["n_failed"], "fail_pct": s["failure_rate_pct"],
        "ms_per_game": s["ms_per_game"], "wall_s": s["wall_s"],
        "cache_size": s.get("cache_size"),
        "nodes": (s.get("tree_stats") or {}).get("nodes_evaluated"),
    }


HDR = (f"{'solver':<18}{'n':>6}{'mean':>9}{'med':>5}{'std':>7}{'max':>5}"
       f"{'<=3':>8}{'<=4':>8}{'<=5':>8}{'fail':>6}{'ms/game':>10}"
       f"{'nodes':>10}{'cache':>10}")


def show(rows):
    print(HDR)
    print("-" * len(HDR))
    for r in rows:
        print(f"{r['solver']:<18}{r['n']:>6}{r['mean']:>9.4f}{r['median']:>5.0f}"
              f"{r['std']:>7.3f}{r['max']:>5}{r['le3']:>8.2f}{r['le4']:>8.2f}"
              f"{r['le5']:>8.2f}{r['fails']:>6}{r['ms_per_game']:>10.1f}"
              f"{(r['nodes'] if r['nodes'] is not None else '-'):>10}"
              f"{(r['cache_size'] if r['cache_size'] is not None else '-'):>10}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="depth",
                    choices=["depth", "topk", "scaling", "full", "keys"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--opener", default="soare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    b = load_artifacts("artifacts")
    fb, V = b.fb, b.vocab
    cfg = SolverConfig(max_guesses=6, seed=SEED, guess_pool="full")
    rows = []

    if args.stage == "keys":
        print(compare_state_keys(fb))
        return 0

    if args.stage == "depth":
        answers = sample(V.answers, args.n)
        print(f"DEPTH COMPARISON — {len(answers)} random answers (seed {SEED}), "
              f"opener={args.opener.upper()}, top_k={args.top_k}\n")
        rows.append(row(run_baseline(fb, b.model, cfg, "entropy", answers,
                                     "entropy(greedy)")))
        for d in (1, 2, 3, 6):
            tc = TreeSearchConfig(depth=d, top_k=args.top_k,
                                  opening_guess=args.opener, max_guesses=6)
            rows.append(row(run_tree(fb, cfg, tc, answers, f"tree_depth{d}")))
            show(rows[-1:])
        print()
        show(rows)

    elif args.stage == "topk":
        answers = sample(V.answers, args.n)
        print(f"TOP-K PRUNING SWEEP — {len(answers)} random answers, "
              f"depth={args.depth}, opener={args.opener.upper()}\n")
        rows.append(row(run_baseline(fb, b.model, cfg, "entropy", answers,
                                     "entropy(greedy)")))
        tc = TreeSearchConfig(depth=args.depth, top_k=0, endgame_top_k=0,
                              opening_guess=args.opener, max_guesses=6)
        rows.append(row(run_tree(fb, cfg, tc, answers, "tree_candidates")))
        for k in (10, 25, 50, 100, 250):
            tc = TreeSearchConfig(depth=args.depth, top_k=k,
                                  endgame_top_k=max(k, 60),
                                  opening_guess=args.opener, max_guesses=6)
            rows.append(row(run_tree(fb, cfg, tc, answers, f"tree_topk{k}")))
            show(rows[-1:])
        print()
        show(rows)

    elif args.stage == "scaling":
        print(f"SCALING — depth={args.depth}, top_k={args.top_k}, "
              f"opener={args.opener.upper()}\n")
        for n in (100, 500, 1000):
            answers = sample(V.answers, n)
            tc = TreeSearchConfig(depth=args.depth, top_k=args.top_k,
                                  opening_guess=args.opener, max_guesses=6)
            rows.append(row(run_tree(fb, cfg, tc, answers, f"tree_n{n}")))
            show(rows[-1:])
        print()
        show(rows)

    elif args.stage == "full":
        answers = list(V.answers)
        print(f"FULL BENCHMARK — all {len(answers)} answers, depth={args.depth}, "
              f"top_k={args.top_k}, opener={args.opener.upper()}\n")
        tc = TreeSearchConfig(depth=args.depth, top_k=args.top_k,
                              opening_guess=args.opener, max_guesses=6)
        s = run_tree(fb, cfg, tc, answers, f"tree_{args.opener}_k{args.top_k}")
        print(s["regime"])
        rows.append(row(s))
        show(rows)

    out = args.out or os.path.join(
        "artifacts", f"tree_benchmark_{args.stage}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
