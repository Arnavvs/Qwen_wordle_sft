"""
benchmark.py — evaluation harness for the classical Wordle solvers.

Kept separate from `wordle_solver.py` so the solver module stays free of
evaluation-only code. Depends on NumPy only; produces plain dicts and lists so
the notebook can turn them into tables or plots without this module knowing
anything about pandas or matplotlib.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from wordle_solver import (
    FeedbackMatrix, FrequencyModel, GameResult, Solver, SolverConfig,
    make_solver, play_game,
)

__all__ = [
    "BenchmarkResult", "run_benchmark", "summarize", "summary_table",
    "first_guess_analysis", "candidate_trajectory",
]


# =============================================================================
# Running games
# =============================================================================

@dataclass
class BenchmarkResult:
    """All per-game outcomes for one strategy, plus timing."""

    strategy: str
    games: List[GameResult] = field(default_factory=list)
    wall_s: float = 0.0
    config: Dict = field(default_factory=dict)
    n_answers_evaluated: int = 0
    opening_guess: Optional[str] = None

    @property
    def scores(self) -> np.ndarray:
        """Guesses used per game; failures counted as max_guesses + 1."""
        return np.array([g.score for g in self.games], dtype=np.float64)

    @property
    def solved_mask(self) -> np.ndarray:
        return np.array([g.solved for g in self.games], dtype=bool)


def run_benchmark(
    strategy: str,
    fb: FeedbackMatrix,
    config: SolverConfig,
    *,
    answers: Optional[Sequence[str]] = None,
    model: Optional[FrequencyModel] = None,
    progress_every: int = 250,
    log: Optional[Callable[[str], None]] = print,
) -> BenchmarkResult:
    """Play one game per answer and collect results.

    The opening guess is computed once and reused, since turn 1 sees no
    information and is therefore identical across games. For the stochastic
    `random` strategy the opener is redrawn per game from the seeded RNG, so the
    run stays reproducible without being artificially fixed.
    """
    solver = make_solver(strategy, fb, config, model)
    solver.reset()
    pool = list(answers if answers is not None else fb.vocab.answers)

    fixed_opener: Optional[str] = None
    if solver.deterministic:
        t0 = time.perf_counter()
        fixed_opener = solver.opening_guess()
        if log:
            log(f"  [{strategy}] opener = {fixed_opener.upper()} "
                f"({time.perf_counter() - t0:.1f}s)")

    games: List[GameResult] = []
    t_start = time.perf_counter()
    for i, ans in enumerate(pool, 1):
        games.append(play_game(solver, ans, first_guess=fixed_opener))
        if log and progress_every and i % progress_every == 0:
            el = time.perf_counter() - t_start
            mean_so_far = float(np.mean([g.score for g in games]))
            log(f"  [{strategy}] {i}/{len(pool)}  "
                f"mean={mean_so_far:.4f}  {el:.1f}s  "
                f"(eta {el / i * (len(pool) - i):.0f}s)")
    wall = time.perf_counter() - t_start

    return BenchmarkResult(
        strategy=strategy, games=games, wall_s=wall,
        config=config.to_dict(), n_answers_evaluated=len(pool),
        opening_guess=fixed_opener,
    )


# =============================================================================
# Summarizing
# =============================================================================

def summarize(res: BenchmarkResult, max_guesses: int = 6) -> Dict:
    """Descriptive statistics for one strategy.

    `mean_guesses` counts only SOLVED games, which is the number normally quoted
    for Wordle solvers. `mean_score` counts failures as `max_guesses + 1` and is
    the honest single-number comparison when failure rates differ, so both are
    reported.
    """
    scores = res.scores
    solved = res.solved_mask
    n = len(scores)
    solved_scores = scores[solved]

    dist = {k: int(((scores == k) & solved).sum()) for k in range(1, max_guesses + 1)}
    pct = {k: 100.0 * v / n for k, v in dist.items()}

    out = {
        "strategy": res.strategy,
        "n_games": n,
        "opening_guess": res.opening_guess,
        "mean_guesses_solved_only": float(np.mean(solved_scores)) if len(solved_scores) else float("nan"),
        "mean_score_failures_penalized": float(np.mean(scores)),
        "median_guesses": float(np.median(solved_scores)) if len(solved_scores) else float("nan"),
        "std_guesses": float(np.std(solved_scores, ddof=1)) if len(solved_scores) > 1 else 0.0,
        "min_guesses": int(np.min(solved_scores)) if len(solved_scores) else None,
        "max_guesses": int(np.max(solved_scores)) if len(solved_scores) else None,
        "n_solved": int(solved.sum()),
        "n_failed": int((~solved).sum()),
        "solve_rate_pct": 100.0 * float(solved.mean()),
        "failure_rate_pct": 100.0 * float((~solved).mean()),
        "distribution_counts": dist,
        "distribution_pct": pct,
        "failures": [g.answer for g in res.games if not g.solved][:50],
        "wall_s": res.wall_s,
        "ms_per_game": 1000.0 * res.wall_s / n if n else 0.0,
    }
    return out


def candidate_trajectory(res: BenchmarkResult, max_turns: int = 6) -> Dict[int, Dict]:
    """Average surviving candidates after each guess.

    Only games still in progress at a given turn contribute to that turn's
    average — a solved game has left the pool, so including its (absent) later
    turns would bias the number downward. `n_games_reaching_turn` is reported so
    the average can be read in context.
    """
    out: Dict[int, Dict] = {}
    for t in range(1, max_turns + 1):
        vals = [g.candidates_remaining[t - 1]
                for g in res.games if len(g.candidates_remaining) >= t]
        if not vals:
            continue
        out[t] = {
            "n_games_reaching_turn": len(vals),
            "mean_remaining": float(np.mean(vals)),
            "median_remaining": float(np.median(vals)),
            "max_remaining": int(np.max(vals)),
            "mean_bits_remaining": float(np.mean([math.log2(v) for v in vals if v > 0])),
        }
    return out


def summary_table(summaries: Sequence[Dict], max_guesses: int = 6) -> str:
    """Fixed-width comparison table — no pandas needed."""
    hdr = (f"{'strategy':<11}{'open':<7}{'mean':>7}{'med':>5}{'std':>6}"
           f"{'min':>4}{'max':>4}" +
           "".join(f"{'%'+str(k):>7}" for k in range(1, max_guesses + 1)) +
           f"{'fail%':>7}{'ms/game':>9}")
    lines = [hdr, "-" * len(hdr)]
    for s in summaries:
        row = (
            f"{s['strategy']:<11}"
            f"{(s['opening_guess'] or '-')[:6]:<7}"
            f"{s['mean_guesses_solved_only']:>7.4f}"
            f"{s['median_guesses']:>5.0f}"
            f"{s['std_guesses']:>6.3f}"
            f"{s['min_guesses']:>4}"
            f"{s['max_guesses']:>4}"
            + "".join(f"{s['distribution_pct'].get(k, 0.0):>7.2f}"
                      for k in range(1, max_guesses + 1))
            + f"{s['failure_rate_pct']:>7.2f}"
            + f"{s['ms_per_game']:>9.1f}"
        )
        lines.append(row)
    return "\n".join(lines)


# =============================================================================
# First-guess analysis
# =============================================================================

def first_guess_analysis(
    fb: FeedbackMatrix,
    *,
    model: Optional[FrequencyModel] = None,
    chunk: int = 2048,
) -> Dict[str, np.ndarray]:
    """Score every legal guess as an opener against the full answer list.

    Turn 1 is the one decision where all strategies face an identical, fully
    known state (every answer is possible), so it is the cleanest place to see
    what each objective actually optimises for and where they disagree.

    Returns arrays aligned with `fb.vocab.guesses`.
    """
    vocab = fb.vocab
    all_answers = np.arange(vocab.n_answers, dtype=np.int32)
    all_guesses = np.arange(vocab.n_guesses, dtype=np.int32)
    stats = fb.partition_stats(all_guesses, all_answers, chunk=chunk)

    is_answer = np.zeros(vocab.n_guesses, dtype=bool)
    is_answer[fb.answer_pos] = True

    m = model or FrequencyModel.fit(vocab.answers)
    freq = m.score_many(vocab.guesses)

    return {
        "word": np.array(vocab.guesses, dtype=object),
        "entropy": stats["entropy"],
        "expected_remaining": stats["expected_remaining"],
        "worst_case": stats["worst_case"],
        "is_possible_answer": is_answer,
        "frequency_score": freq,
    }


def top_openers(
    fga: Dict[str, np.ndarray],
    metric: str,
    n: int = 20,
    *,
    maximize: bool = True,
) -> List[Dict]:
    """Top-n openers by one metric, each row carrying all the other metrics too."""
    vals = fga[metric]
    order = np.argsort(-vals if maximize else vals, kind="stable")[:n]
    rows = []
    for rank, i in enumerate(order, 1):
        rows.append({
            "rank": rank,
            "word": str(fga["word"][i]),
            "entropy_bits": float(fga["entropy"][i]),
            "expected_remaining": float(fga["expected_remaining"][i]),
            "worst_case_remaining": int(fga["worst_case"][i]),
            "is_possible_answer": bool(fga["is_possible_answer"][i]),
            "frequency_score": float(fga["frequency_score"][i]),
        })
    return rows


def format_openers(rows: Sequence[Dict], title: str) -> str:
    hdr = (f"{'#':>3} {'word':<7}{'H(bits)':>9}{'E[rem]':>10}"
           f"{'worst':>7}{'answer?':>9}{'freq':>7}")
    out = [title, hdr, "-" * len(hdr)]
    for r in rows:
        out.append(
            f"{r['rank']:>3} {r['word']:<7}{r['entropy_bits']:>9.4f}"
            f"{r['expected_remaining']:>10.2f}{r['worst_case_remaining']:>7}"
            f"{('yes' if r['is_possible_answer'] else 'no'):>9}"
            f"{r['frequency_score']:>7.3f}"
        )
    return "\n".join(out)


# =============================================================================
# Persistence of results
# =============================================================================

def save_benchmark(
    path: str,
    summaries: Sequence[Dict],
    trajectories: Dict[str, Dict],
    *,
    extra: Optional[Dict] = None,
) -> str:
    """Write benchmark summaries to JSON (per-game detail is intentionally omitted)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "summaries": list(summaries),
        "candidate_trajectories": trajectories,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path
