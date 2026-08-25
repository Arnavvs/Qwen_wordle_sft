"""
run_full_benchmark.py — evaluate every solver against every possible answer.

    python run_full_benchmark.py
    python run_full_benchmark.py --strategies entropy hybrid --limit 200

Writes artifacts/benchmark_results.json plus a first-guess analysis CSV.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import json
import os
import sys
import time

import numpy as np

from wordle_solver import SolverConfig, load_artifacts
from benchmark import (
    candidate_trajectory, first_guess_analysis, format_openers, run_benchmark,
    save_benchmark, summarize, summary_table, top_openers,
)

HERE = _paths.ROOT  # data lives at the repo root
ALL = ["random", "frequency", "entropy", "expected", "minimax", "hybrid"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--strategies", nargs="*", default=ALL)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N answers (0 = all)")
    ap.add_argument("--max-guesses", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--guess-pool", default="full",
                    choices=["full", "adaptive", "candidates"])
    ap.add_argument("--skip-first-guess", action="store_true")
    args = ap.parse_args(argv)

    b = load_artifacts(args.artifact_dir)
    print(b.describe(), flush=True)

    cfg = SolverConfig(
        max_guesses=args.max_guesses, seed=args.seed, guess_pool=args.guess_pool,
    )
    answers = b.vocab.answers[:args.limit] if args.limit else b.vocab.answers
    print(f"\nevaluating {len(answers)} answers x {len(args.strategies)} strategies "
          f"(max_guesses={cfg.max_guesses}, guess_pool={cfg.guess_pool}, "
          f"seed={cfg.seed})\n", flush=True)

    summaries, trajectories = [], {}
    grand_t0 = time.perf_counter()
    for strat in args.strategies:
        print(f"=== {strat} ===", flush=True)
        res = run_benchmark(strat, b.fb, cfg, answers=answers, model=b.model,
                            progress_every=500)
        s = summarize(res, max_guesses=cfg.max_guesses)
        summaries.append(s)
        trajectories[strat] = candidate_trajectory(res, max_turns=cfg.max_guesses)
        print(f"  mean={s['mean_guesses_solved_only']:.4f}  "
              f"fail={s['failure_rate_pct']:.2f}%  "
              f"wall={res.wall_s:.1f}s\n", flush=True)

    print("\n" + "=" * 100)
    print(f"FULL BENCHMARK — {len(answers)} answers, guess_pool={cfg.guess_pool}")
    print("=" * 100)
    print(summary_table(summaries, max_guesses=cfg.max_guesses))
    print(f"\ntotal wall time {time.perf_counter() - grand_t0:.1f}s")

    print("\n\nAVERAGE SURVIVING CANDIDATES AFTER EACH GUESS")
    print(f"{'strategy':<11}" + "".join(f"{'g'+str(t):>12}" for t in range(1, cfg.max_guesses + 1)))
    print("-" * (11 + 12 * cfg.max_guesses))
    for strat in args.strategies:
        tr = trajectories[strat]
        row = f"{strat:<11}"
        for t in range(1, cfg.max_guesses + 1):
            row += f"{tr[t]['mean_remaining']:>12.2f}" if t in tr else f"{'-':>12}"
        print(row)

    # ---- first-guess analysis -------------------------------------------
    fga_payload = None
    if not args.skip_first_guess:
        print("\n\nscoring all 12972 openers ...", flush=True)
        t0 = time.perf_counter()
        fga = first_guess_analysis(b.fb, model=b.model)
        print(f"done in {time.perf_counter() - t0:.1f}s\n")

        blocks = [
            ("entropy", True, "TOP 20 OPENERS BY ENTROPY (max expected information gain)"),
            ("expected_remaining", False, "TOP 20 OPENERS BY EXPECTED REMAINING CANDIDATES (min)"),
            ("worst_case", False, "TOP 20 OPENERS BY WORST-CASE BUCKET (minimax)"),
            ("frequency_score", True, "TOP 20 OPENERS BY LETTER-FREQUENCY HEURISTIC"),
        ]
        fga_payload = {}
        for metric, maximize, title in blocks:
            rows = top_openers(fga, metric, 20, maximize=maximize)
            fga_payload[metric] = rows
            print(format_openers(rows, "\n" + title))

        csv_p = os.path.join(args.artifact_dir, "first_guess_analysis.csv")
        order = np.argsort(-fga["entropy"], kind="stable")
        with open(csv_p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("word,entropy_bits,expected_remaining,worst_case_remaining,"
                     "is_possible_answer,frequency_score\n")
            for i in order:
                fh.write(f"{fga['word'][i]},{fga['entropy'][i]:.6f},"
                         f"{fga['expected_remaining'][i]:.4f},{fga['worst_case'][i]},"
                         f"{int(fga['is_possible_answer'][i])},"
                         f"{fga['frequency_score'][i]:.6f}\n")
        print(f"\nwrote {csv_p} (all {len(order)} openers, ranked by entropy)")

    out = os.path.join(args.artifact_dir, "benchmark_results.json")
    save_benchmark(out, summaries, trajectories, extra={
        "n_answers_evaluated": len(answers),
        "config": cfg.to_dict(),
        "first_guess_top20": fga_payload,
        "vocabulary": {"n_answers": b.vocab.n_answers, "n_guesses": b.vocab.n_guesses},
    })
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
