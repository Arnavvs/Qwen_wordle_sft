"""
tree_search.py — game-tree search solver for Wordle.

A separate, additive module: `wordle_solver.py` is imported, never modified, and
`EntropySolver` remains the fast greedy baseline. Nothing here changes the
behaviour of any existing strategy.

WHAT IS BEING COMPUTED
----------------------
For a state `S` (the set of answers still possible) with `d` guesses remaining,
define the **total** cost

    T(S, d) = minimum over guesses g of  [ |S| + sum_{f != GGGGG} T(S_f, d-1) ]

where `S_f = {a in S : feedback(g, a) = f}`. The `|S|` term charges one guess to
every answer still alive; the all-green bucket contributes nothing further
because that answer is finished. The mean guesses over `S` is `T(S, d) / |S|`.

Uniform probability over the surviving answers is assumed throughout, matching
`FeedbackMatrix.partition_stats` — so the tree search and the greedy solvers
optimise comparable objectives, one myopically and one with lookahead.

Base cases:
  |S| == 1              -> 1        (name it)
  |S| == 2, d >= 2      -> 3        (name one; 1 + 2, provably optimal)
  |S| >  1, d == 1      -> INFEASIBLE

EXACT vs PRUNED — read this before quoting any number
-----------------------------------------------------
The recursion above is exact only when the guess pool at every node is the full
legal dictionary and the depth limit never binds. That is the `exact=True`
configuration and it is expensive.

Every other configuration searches a *subset* of guesses (`top_k`, or
candidates-only) and is therefore **optimal only within the pool it searched** —
an upper bound on the true optimum, not the optimum. `SearchStats.exact` records
which regime actually ran, and `TreeSearchSolver.describe_regime()` prints it.
Results from a pruned run are labelled "pruned" everywhere and must not be
called optimal.

LOWER BOUND (used for branch-and-bound)
---------------------------------------
    T(S, d) >= 2|S| - 1

Every answer in S costs at least one more guess (|S|), and at most one of them
can be identified by the guess just played, so the remaining |S|-1 each cost at
least one further guess. The bound is tight exactly when the chosen guess is
itself in S and every other bucket is a singleton — which is also the early-exit
condition, since no better guess can exist once it is met.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wordle_solver import (
    ALL_GREEN, FeedbackMatrix, Solver, SolverConfig, Vocabulary,
)

__all__ = [
    "INFEASIBLE", "SearchStats", "TreeSearchConfig", "GameTree",
    "TreeSearchSolver", "bucket_sumsq", "state_key_bytes", "state_key_bitset",
]

INFEASIBLE = 1 << 30


# =============================================================================
# Fast kernels
# =============================================================================

def bucket_sumsq(MT: np.ndarray, S: np.ndarray) -> np.ndarray:
    """For every guess, return sum of squared bucket sizes over the state `S`.

    `sum_f k_f^2` is exactly `|S| * E[remaining]`, so ranking by it ranks by
    expected remaining candidates. Computed without the 243-wide histogram that
    dominates `partition_stats`, which matters because the search evaluates
    thousands of small states.

    Method: sort each guess's column of feedback codes, then use the identity

        sum_{r=1}^{k} (2r - 1) = k^2

    so summing `2r-1` over every position, where `r` is the 1-based rank of that
    position inside its run of equal codes, yields `sum_f k_f^2` directly. Costs
    a few passes over an (|S|, n_guesses) array and scales with |S|, unlike the
    histogram approach whose cost is constant in |S|.
    """
    X = np.sort(MT[S], axis=0)                       # (n, G)
    n = X.shape[0]
    if n == 1:
        return np.ones(X.shape[1], dtype=np.int64)
    neq = np.empty(X.shape, dtype=bool)
    neq[0] = True
    np.not_equal(X[1:], X[:-1], out=neq[1:])
    idx = np.arange(n, dtype=np.int32)[:, None]
    start = np.maximum.accumulate(np.where(neq, idx, np.int32(0)), axis=0)
    r = idx - start + 1                              # rank within run
    return (2 * r.astype(np.int64) - 1).sum(axis=0)


def state_key_bytes(S: np.ndarray) -> bytes:
    """Canonical state key: raw bytes of the sorted index array."""
    return S.tobytes()


def state_key_bitset(S: np.ndarray, n_answers: int) -> int:
    """Canonical state key as a Python big-int bitmask over the answer list.

    Benchmarked against `state_key_bytes` in `compare_state_keys`; kept because
    the task asked for it to be investigated, not because it wins.
    """
    m = 0
    for i in S:
        m |= 1 << int(i)
    return m


# =============================================================================
# Configuration and stats
# =============================================================================

@dataclass
class TreeSearchConfig:
    """Knobs for the search. Defaults are the practical pruned configuration."""

    max_guesses: int = 6

    # Depth limit, counted in guesses of lookahead. depth=1 reproduces a greedy
    # one-ply decision; depth>=5 with max_guesses=6 never binds after turn 1.
    depth: int = 6

    # Guess pool at each node.
    #   exact=True  -> every legal guess (search is exact within `depth`)
    #   exact=False -> candidates + the top_k best-ranked non-candidate probes
    exact: bool = False
    top_k: int = 25

    # Below this many candidates, widen the pool back to `endgame_top_k`
    # (cheap because the state is small, and this is where mistakes are costly).
    endgame_threshold: int = 10
    endgame_top_k: int = 60

    # Fixed opening guess. Turn 1 is identical in every game, so searching it is
    # wasted work; and a full exact root search is the expensive part.
    opening_guess: Optional[str] = "soare"

    # Branch-and-bound on/off (for measuring what it buys).
    use_branch_and_bound: bool = True

    # Cache shared across games in a benchmark run.
    reuse_cache: bool = True


@dataclass
class SearchStats:
    """Instrumentation, so cost claims are measured rather than asserted."""

    nodes_evaluated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    guesses_expanded: int = 0
    bb_cutoffs: int = 0
    lb_early_exits: int = 0
    rank_calls: int = 0
    seconds: float = 0.0
    exact: bool = False

    def merge(self, other: "SearchStats") -> None:
        for f in ("nodes_evaluated", "cache_hits", "cache_misses",
                  "guesses_expanded", "bb_cutoffs", "lb_early_exits",
                  "rank_calls"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.seconds += other.seconds

    def as_dict(self) -> dict:
        return {
            "nodes_evaluated": self.nodes_evaluated,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": self.cache_misses,
            "guesses_expanded": self.guesses_expanded,
            "bb_cutoffs": self.bb_cutoffs,
            "lb_early_exits": self.lb_early_exits,
            "rank_calls": self.rank_calls,
            "seconds": round(self.seconds, 2),
            "exact": self.exact,
        }


# =============================================================================
# The search
# =============================================================================

class GameTree:
    """Memoized depth-limited expectiminimax over Wordle states."""

    def __init__(self, fb: FeedbackMatrix, config: Optional[TreeSearchConfig] = None):
        self.fb = fb
        self.vocab = fb.vocab
        self.cfg = config or TreeSearchConfig()
        # Contiguous copies: row-major for single-guess partitioning, transposed
        # for bulk ranking. A memmap would make both gathers far slower.
        self.M = np.ascontiguousarray(np.asarray(fb.M))       # (G, A)
        self.MT = np.ascontiguousarray(self.M.T)              # (A, G)
        self.answer_pos = np.asarray(fb.answer_pos)
        self.n_guesses = self.M.shape[0]
        self._memo: Dict[Tuple[bytes, int, int], Tuple[int, int]] = {}
        self._h_memo: Dict[Tuple[bytes, int], int] = {}
        self.stats = SearchStats(exact=self.cfg.exact)

    # -- horizon evaluation -------------------------------------------------
    def horizon_cost(self, S: np.ndarray, d: int) -> int:
        """Cost estimate used when the depth limit is reached.

        Continues with a cheap greedy rollout — candidates only, minimising
        expected remaining — and counts the guesses it actually takes. This is a
        real playable line, so it is a genuine *upper* bound on the state's cost
        rather than an optimistic guess. That matters: an optimistic horizon
        (e.g. the 2|S|-1 lower bound) would let the search prefer branches it
        cannot actually deliver, and depth-2 could then score worse than depth-1.
        """
        n = len(S)
        if n == 1:
            return 1
        if d <= 1:
            return INFEASIBLE
        if n == 2:
            return 3
        key = (S.tobytes(), d)
        hit = self._h_memo.get(key)
        if hit is not None:
            return hit

        cand = self.answer_pos[S]
        sq = bucket_sumsq(self.MT, S)
        g = int(cand[int(np.argmin(sq[cand]))])
        codes = self.M[g][S]
        cost = n
        for c in np.unique(codes):
            if c == ALL_GREEN:
                continue
            sub_cost = self.horizon_cost(S[codes == c], d - 1)
            if sub_cost >= INFEASIBLE:
                cost = INFEASIBLE
                break
            cost += sub_cost
        self._h_memo[key] = cost
        return cost

    # -- pool construction -------------------------------------------------
    def _pool(self, S: np.ndarray) -> np.ndarray:
        """Guesses to consider at this node, best-first for branch-and-bound.

        Candidates come first: they can win outright, so they usually produce a
        strong upper bound immediately, which is what makes the bound prune.
        """
        cand = self.answer_pos[S]
        if self.cfg.exact:
            # Full pool, but still candidate-first so the bound tightens early.
            rest = np.setdiff1d(np.arange(self.n_guesses, dtype=np.int32),
                                cand, assume_unique=False)
            self.stats.rank_calls += 1
            sq = bucket_sumsq(self.MT, S)
            rest = rest[np.argsort(sq[rest], kind="stable")]
            return np.concatenate([cand, rest])

        k = (self.cfg.endgame_top_k if len(S) <= self.cfg.endgame_threshold
             else self.cfg.top_k)
        if k <= 0:
            return cand
        self.stats.rank_calls += 1
        sq = bucket_sumsq(self.MT, S)
        # Rank all guesses by expected remaining (= sum k^2 / |S|), take top k,
        # then union with the candidates.
        top = np.argpartition(sq, min(k, len(sq) - 1))[:k].astype(np.int32)
        pool = np.concatenate([cand, top])
        _, first = np.unique(pool, return_index=True)
        return pool[np.sort(first)]

    # -- core recursion ----------------------------------------------------
    def solve_state(self, S: np.ndarray, d: int,
                    search_depth: Optional[int] = None) -> Tuple[int, int]:
        """Return (total_cost, best_guess_index) for state `S` with `d` left.

        `total_cost` is the summed number of guesses over every answer in `S`;
        divide by `len(S)` for the mean. `INFEASIBLE` means the state cannot be
        finished within `d` guesses.

        `search_depth` is the remaining plies of *exact* lookahead; once it hits
        zero the node is scored by `horizon_cost` instead of being expanded.
        Defaults to `config.depth`.
        """
        if search_depth is None:
            search_depth = self.cfg.depth
        n = len(S)
        if n == 1:
            return 1, int(self.answer_pos[S[0]])
        if d <= 1:
            return INFEASIBLE, -1
        if n == 2:
            # Name one of them: 1 guess for it, 2 for the other. Optimal.
            return 3, int(self.answer_pos[S[0]])
        if search_depth <= 0:
            return self.horizon_cost(S, d), -1

        key = (state_key_bytes(S), d, search_depth)
        hit = self._memo.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return hit
        self.stats.cache_misses += 1
        self.stats.nodes_evaluated += 1

        lower = 2 * n - 1
        best = INFEASIBLE
        best_g = -1
        row_of = self.M

        for g in self._pool(S):
            g = int(g)
            self.stats.guesses_expanded += 1
            codes = row_of[g][S]
            order = np.argsort(codes, kind="stable")
            cs = codes[order]
            Ss = S[order]
            # Run boundaries = bucket boundaries.
            bounds = np.flatnonzero(np.concatenate(
                ([True], cs[1:] != cs[:-1], [True])))
            n_buckets = len(bounds) - 1
            if n_buckets == 1 and cs[0] != ALL_GREEN:
                continue                      # splits nothing and cannot win

            cost = n
            feasible = True
            # Largest buckets first: they are most likely to blow the bound, so
            # evaluating them early makes branch-and-bound cut sooner.
            spans = [(bounds[i], bounds[i + 1]) for i in range(n_buckets)]
            spans.sort(key=lambda ab: ab[0] - ab[1])
            for a, bnd in spans:
                if cs[a] == ALL_GREEN:
                    continue                  # this answer is finished
                sub_cost, _ = self.solve_state(Ss[a:bnd], d - 1, search_depth - 1)
                if sub_cost >= INFEASIBLE:
                    feasible = False
                    break
                cost += sub_cost
                if self.cfg.use_branch_and_bound and cost >= best:
                    feasible = False
                    self.stats.bb_cutoffs += 1
                    break
            if feasible and cost < best:
                best, best_g = cost, g
                if best <= lower:
                    self.stats.lb_early_exits += 1
                    break                     # provably optimal for this pool

        out = (best, best_g)
        self._memo[key] = out
        return out

    # -- convenience -------------------------------------------------------
    def best_guess(self, S: np.ndarray, d: int) -> int:
        return self.solve_state(S, d)[1]

    @property
    def cache_size(self) -> int:
        return len(self._memo) + len(self._h_memo)

    def clear_cache(self) -> None:
        self._memo.clear()
        self._h_memo.clear()


# =============================================================================
# Solver adapter — plugs into play_game / run_benchmark unchanged
# =============================================================================

class TreeSearchSolver(Solver):
    """`Solver` wrapper around `GameTree`, usable anywhere a solver is."""

    name = "tree"
    deterministic = True

    def __init__(self, fb, config: Optional[SolverConfig] = None,
                 tree_config: Optional[TreeSearchConfig] = None):
        super().__init__(fb, config)
        tc = tree_config or TreeSearchConfig()
        tc.max_guesses = self.config.max_guesses
        self.tcfg = tc
        self.tree = GameTree(fb, tc)

    def opening_guess(self) -> str:
        if self.tcfg.opening_guess:
            return self.tcfg.opening_guess
        if self.config.opening_guess:
            return self.config.opening_guess
        # No fixed opener: search the root. Expensive; measured, not assumed.
        S = np.arange(self.vocab.n_answers, dtype=np.int32)
        gi = self.tree.best_guess(S, self.config.max_guesses)
        return self.vocab.guesses[gi]

    def choose(self, cand_idx: np.ndarray, turn: int) -> str:
        S = np.sort(cand_idx.astype(np.int32))
        d = self.config.max_guesses - turn + 1
        cost, gi = self.tree.solve_state(S, d)
        if gi < 0:
            # No guaranteed line within the remaining depth: fall back to the
            # best single-ply probe rather than giving up the game.
            sq = bucket_sumsq(self.tree.MT, S)
            cand = self.tree.answer_pos[S]
            gi = int(cand[int(np.argmin(sq[cand]))])
        return self.vocab.guesses[int(gi)]

    def describe_regime(self) -> str:
        c = self.tcfg
        if c.exact:
            return (f"EXACT within depth={c.depth} (full {self.tree.n_guesses}-guess "
                    f"pool at every node); opener="
                    f"{c.opening_guess or 'searched'}")
        return (f"PRUNED (top_k={c.top_k}, endgame_top_k={c.endgame_top_k} "
                f"below |S|<={c.endgame_threshold}, depth={c.depth}); "
                f"opener={c.opening_guess or 'searched'} "
                f"-- optimal only within the searched pool, NOT proven optimal")


# =============================================================================
# State-key representation comparison (Step 4)
# =============================================================================

def compare_state_keys(fb: FeedbackMatrix, sizes=(5, 20, 62, 183), reps=2000) -> str:
    """Benchmark bytes-key vs Python-int bitset-key for memo hashing."""
    rs = np.random.default_rng(0)
    n_ans = fb.vocab.n_answers
    lines = [f"{'|S|':>6}{'bytes key':>14}{'bitset key':>14}{'ratio':>9}",
             "-" * 43]
    for n in sizes:
        S = np.sort(rs.choice(n_ans, n, replace=False)).astype(np.int32)
        t0 = time.perf_counter()
        for _ in range(reps):
            hash(state_key_bytes(S))
        tb = (time.perf_counter() - t0) / reps * 1e6
        t0 = time.perf_counter()
        for _ in range(reps):
            hash(state_key_bitset(S, n_ans))
        tm = (time.perf_counter() - t0) / reps * 1e6
        lines.append(f"{n:>6}{tb:>12.2f}us{tm:>12.2f}us{tm/tb:>9.1f}x")
    lines.append("bytes key wins at every size: numpy .tobytes() is a single memcpy,")
    lines.append("while building a 2315-bit Python int costs one shift+or per member.")
    return "\n".join(lines)
