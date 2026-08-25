"""
wordle_solver.py — a classical (non-LLM, non-RL) Wordle solver.

This module is deliberately self-contained and depends only on the Python
standard library plus NumPy. It contains no machine-learning models, no neural
networks, no reinforcement learning and no language models. Every decision made
by every solver in this file is an explicit symbolic or information-theoretic
computation over an enumerable hypothesis space.

The implementation is organised into three conceptually distinct layers, and the
separation is intentional — it is the thing being measured:

  1. SYMBOLIC CONSTRAINT SOLVING
     `feedback`, `filter_candidates`, `FeedbackMatrix`. Given a game history,
     deduce exactly which words remain logically possible. This layer is
     complete and exact: it never guesses and never approximates.

  2. INFORMATION-THEORETIC DECISION MAKING
     `EntropySolver`, `ExpectedRemainingSolver`, `MinimaxSolver`. Given the
     surviving hypothesis set, choose the probe that maximises expected
     information gain (or minimises expected / worst-case posterior size).
     No knowledge of English is used — only the partition structure induced by
     the feedback function.

  3. PROBABILITY / FREQUENCY HEURISTICS
     `FrequencySolver`, and the answer-probability term of `HybridSolver`.
     Uses empirical letter statistics of the answer list. This is the only
     layer that encodes anything resembling "prior knowledge" about words.

FEEDBACK ENCODING
-----------------
A feedback pattern is 5 tiles, each one of:

    0 = B  grey   (letter not present, or all its occurrences already accounted for)
    1 = Y  yellow (letter present, wrong position)
    2 = G  green  (letter present, correct position)

Patterns are encoded as a single base-3 integer, little-endian by position:

    code = sum(tile[i] * 3**i  for i in range(5))

so `code` lies in [0, 243). Because 243 < 256 the entire guess x answer
feedback table fits in a `uint8` array — 12972 x 2315 is only ~28.6 MiB, which
is what makes exhaustive precomputation practical on Kaggle or a laptop.

The all-green pattern "GGGGG" is code 242.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "WORD_LEN", "N_PATTERNS", "ALL_GREEN",
    "feedback", "feedback_code", "code_to_pattern", "pattern_to_code",
    "is_consistent", "filter_candidates",
    "Vocabulary", "load_vocabulary",
    "build_feedback_matrix", "FeedbackMatrix",
    "FrequencyModel",
    "SolverConfig", "GameResult",
    "Solver", "RandomSolver", "FrequencySolver", "EntropySolver",
    "ExpectedRemainingSolver", "MinimaxSolver", "HybridSolver",
    "STRATEGIES", "make_solver",
    "play_game", "solve", "interactive_play",
    "save_artifacts", "load_artifacts", "WordleSolverBundle",
]

WORD_LEN = 5
N_PATTERNS = 3 ** WORD_LEN          # 243
ALL_GREEN = N_PATTERNS - 1          # 242, i.e. "GGGGG"
_POW3 = tuple(3 ** i for i in range(WORD_LEN))
_TILE_CHARS = "BYG"


# =============================================================================
# LAYER 1 — SYMBOLIC: the feedback function
# =============================================================================

def feedback(guess: str, answer: str) -> str:
    """Return the Wordle feedback for `guess` against `answer` as e.g. "GYBBB".

    This is the reference implementation: pure Python, no NumPy, no caching.
    Everything else in this module is validated against it.

    Duplicate-letter semantics (this is the part everyone gets wrong):

    Think of the answer as supplying a *budget* of tiles for each letter, equal
    to how many times that letter occurs in the answer. Two passes:

      Pass 1 — greens. Every position where guess and answer agree is green and
               consumes one unit of that letter's budget.
      Pass 2 — yellows, scanned left to right. A non-green guess position is
               yellow only if its letter still has unconsumed budget, and it
               then consumes one unit. Otherwise it is grey.

    Greens therefore always take priority over yellows regardless of position,
    and surplus occurrences in the guess come back grey. Left-to-right ordering
    in pass 2 is what makes the function deterministic when a letter is
    over-guessed.
    """
    if len(guess) != WORD_LEN or len(answer) != WORD_LEN:
        raise ValueError(
            f"both words must be length {WORD_LEN}; got {guess!r} and {answer!r}"
        )

    tiles = [0] * WORD_LEN
    # Pass 1: greens consume from the budget first.
    remaining = Counter(answer)
    for i in range(WORD_LEN):
        if guess[i] == answer[i]:
            tiles[i] = 2
            remaining[guess[i]] -= 1
    # Pass 2: yellows, left to right, from whatever budget survives.
    for i in range(WORD_LEN):
        if tiles[i] == 2:
            continue
        c = guess[i]
        if remaining[c] > 0:
            tiles[i] = 1
            remaining[c] -= 1
    return "".join(_TILE_CHARS[t] for t in tiles)


def feedback_code(guess: str, answer: str) -> int:
    """Same as `feedback` but returns the base-3 integer encoding."""
    return pattern_to_code(feedback(guess, answer))


def pattern_to_code(pattern: str) -> int:
    """Encode a pattern string (chars in B/Y/G, case-insensitive) as an int."""
    p = pattern.strip().upper()
    if len(p) != WORD_LEN:
        raise ValueError(f"pattern must be {WORD_LEN} chars; got {pattern!r}")
    code = 0
    for i, ch in enumerate(p):
        # Accept a few common aliases so interactive input is forgiving.
        if ch in ("B", "_", "0", "."):
            t = 0
        elif ch in ("Y", "1", "?"):
            t = 1
        elif ch in ("G", "2", "+"):
            t = 2
        else:
            raise ValueError(f"bad feedback character {ch!r} in {pattern!r}")
        code += t * _POW3[i]
    return code


def code_to_pattern(code: int) -> str:
    """Decode a base-3 integer back into a pattern string like "GYBBB"."""
    if not 0 <= code < N_PATTERNS:
        raise ValueError(f"code must be in [0,{N_PATTERNS}); got {code}")
    out = []
    for _ in range(WORD_LEN):
        out.append(_TILE_CHARS[code % 3])
        code //= 3
    return "".join(out)


# =============================================================================
# LAYER 1 — SYMBOLIC: candidate filtering
# =============================================================================

def is_consistent(candidate: str, guess: str, pattern: str) -> bool:
    """True iff `candidate` could still be the answer given (guess -> pattern).

    Defined *by* the feedback function rather than by re-deriving constraint
    rules, which is what guarantees the two can never disagree: a word survives
    exactly when it would have produced the observed pattern.
    """
    return feedback(guess, candidate) == pattern.strip().upper()


def filter_candidates(
    candidates: Iterable[str],
    guess: str,
    pattern: str,
) -> List[str]:
    """Eliminate every candidate inconsistent with a single (guess, pattern).

    Applying this once per history entry is equivalent to filtering against the
    complete history, because consistency with each observation is independent
    of the others (each is a pure function of the candidate). `filter_history`
    does that fold for convenience.
    """
    target = pattern.strip().upper()
    return [w for w in candidates if feedback(guess, w) == target]


def filter_history(
    candidates: Iterable[str],
    history: Sequence[Tuple[str, str]],
) -> List[str]:
    """Filter against a whole game history: [(guess, pattern), ...]."""
    out = list(candidates)
    for guess, pattern in history:
        out = filter_candidates(out, guess, pattern)
    return out


# =============================================================================
# Vocabulary loading / validation
# =============================================================================

@dataclass
class Vocabulary:
    """The two word lists the game is played over.

    `answers`   — words that can actually be the hidden solution.
    `guesses`   — every word the game will *accept* as a guess. This is a
                  superset of `answers`; keeping them distinct is essential,
                  because restricting probes to `answers` throws away most of
                  the available information-seeking power.
    """

    answers: List[str]
    guesses: List[str]
    source: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.answer_index = {w: i for i, w in enumerate(self.answers)}
        self.guess_index = {w: i for i, w in enumerate(self.guesses)}
        # Every answer must be a legal guess, or the game is inconsistent.
        missing = [w for w in self.answers if w not in self.guess_index]
        if missing:
            raise ValueError(
                f"{len(missing)} answers are not legal guesses, e.g. {missing[:5]}"
            )

    @property
    def n_answers(self) -> int:
        return len(self.answers)

    @property
    def n_guesses(self) -> int:
        return len(self.guesses)

    def answer_positions(self) -> np.ndarray:
        """Indices, within `guesses`, of the words that are possible answers."""
        return np.array([self.guess_index[w] for w in self.answers], dtype=np.int32)

    def describe(self) -> str:
        return (
            f"Vocabulary(answers={self.n_answers}, legal_guesses={self.n_guesses}, "
            f"non_answer_guesses={self.n_guesses - self.n_answers})"
        )


def normalize_words(
    raw: str,
    word_len: int = WORD_LEN,
    *,
    strict: bool = False,
) -> Tuple[List[str], Dict[str, int]]:
    """Split raw text into normalized words and report what had to be cleaned.

    Normalization: strip whitespace, lowercase, drop empties. Words that are the
    wrong length or contain anything outside a-z are rejected (and counted)
    rather than silently mangled. Duplicates are removed, first occurrence wins,
    original order preserved.
    """
    import unicodedata

    stats = {
        "raw_tokens": 0, "empty": 0, "uppercased": 0, "accented": 0,
        "bad_length": 0, "non_alpha": 0, "duplicates": 0, "kept": 0,
    }
    words: List[str] = []
    seen = set()
    # Split on any whitespace and also on commas, so simple CSV columns work.
    for tok in raw.replace(",", "\n").split():
        stats["raw_tokens"] += 1
        w = tok.strip().strip('"').strip("'")
        if not w:
            stats["empty"] += 1
            continue
        if w != w.lower():
            stats["uppercased"] += 1
            w = w.lower()
        decomposed = unicodedata.normalize("NFKD", w)
        if decomposed != w:
            stats["accented"] += 1
            w = "".join(c for c in decomposed if not unicodedata.combining(c))
        if len(w) != word_len:
            stats["bad_length"] += 1
            continue
        if not (w.isascii() and w.isalpha()):
            stats["non_alpha"] += 1
            continue
        if w in seen:
            stats["duplicates"] += 1
            continue
        seen.add(w)
        words.append(w)
    stats["kept"] = len(words)
    if strict and (stats["bad_length"] or stats["non_alpha"]):
        raise ValueError(f"rejected words during normalization: {stats}")
    return words, stats


def load_vocabulary(
    answers_path: str,
    guesses_path: Optional[str] = None,
    *,
    guesses_are_extra: Optional[bool] = None,
) -> Vocabulary:
    """Load a `Vocabulary` from two text files (one word per line).

    `guesses_are_extra` controls how the guess file is interpreted:
      True   — the file lists only *non-answer* extras; the legal pool is
               answers + extras.
      False  — the file is already the complete legal pool.
      None   — infer it: if the two lists are disjoint, treat the guess file as
               extras; otherwise assume it is already complete.
    This inference exists because both conventions are common in the wild and
    guessing wrong changes the size of the guess pool.
    """
    with open(answers_path, encoding="utf-8") as fh:
        answers, _ = normalize_words(fh.read())
    if guesses_path is None:
        guesses = list(answers)
    else:
        with open(guesses_path, encoding="utf-8") as fh:
            extra, _ = normalize_words(fh.read())
        aset = set(answers)
        if guesses_are_extra is None:
            overlap = len(aset & set(extra))
            guesses_are_extra = (overlap == 0)
        if guesses_are_extra:
            guesses = list(answers) + [w for w in extra if w not in aset]
        else:
            guesses = list(extra)
            missing = [w for w in answers if w not in set(guesses)]
            guesses.extend(missing)
    guesses = sorted(set(guesses))
    return Vocabulary(
        answers=sorted(answers),
        guesses=guesses,
        source={"answers_path": str(answers_path), "guesses_path": str(guesses_path)},
    )


# =============================================================================
# LAYER 1 — SYMBOLIC: precomputed feedback matrix
# =============================================================================

def _letters_to_idx(words: Sequence[str]) -> np.ndarray:
    """(n, WORD_LEN) uint8 array of 0..25 letter indices."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    return (arr - ord("a")).reshape(len(words), WORD_LEN)


def build_feedback_matrix(
    guesses: Sequence[str],
    answers: Sequence[str],
    *,
    chunk: int = 512,
    out: Optional[np.ndarray] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Compute the full feedback table, shape (len(guesses), len(answers)), uint8.

    `M[g, a] == feedback_code(guesses[g], answers[a])`.

    Vectorized restatement of the two-pass rule. Greens are a plain positional
    equality test. Yellows are handled one letter of the alphabet at a time:

      * `ng[g, a, i]`  — position i of the guess holds letter c and the answer
                         does *not* hold c there, i.e. a non-green occurrence.
      * `remaining`    — occurrences of c in the answer minus greens of c, which
                         is exactly the yellow budget after pass 1.
      * `rank`         — exclusive prefix count of non-green occurrences of c to
                         the left of i. The left-to-right greedy rule is then
                         simply `rank < remaining`.

    Chunking over guesses bounds peak memory at roughly
    `chunk * n_answers * WORD_LEN` bytes for the boolean temporaries.
    """
    ng_, na_ = len(guesses), len(answers)
    G = _letters_to_idx(guesses)
    A = _letters_to_idx(answers)

    if out is None:
        out = np.empty((ng_, na_), dtype=np.uint8)
    elif out.shape != (ng_, na_) or out.dtype != np.uint8:
        raise ValueError(f"`out` must be uint8 with shape {(ng_, na_)}")

    pow3 = np.array(_POW3, dtype=np.int16)
    green_w = (2 * pow3).astype(np.int16)

    # Per-letter answer masks and counts, computed once.
    a_eq = [(A == c) for c in range(26)]                       # each (na, 5) bool
    a_cnt = np.stack([m.sum(axis=1) for m in a_eq]).astype(np.int16)   # (26, na)

    for start in range(0, ng_, chunk):
        gl = G[start:start + chunk]
        c_len = gl.shape[0]

        # --- pass 1: greens -------------------------------------------------
        green = (gl[:, None, :] == A[None, :, :])              # (c, na, 5)
        codes = np.tensordot(green.astype(np.int16), green_w, axes=([2], [0]))

        # --- pass 2: yellows, per alphabet letter ---------------------------
        present = np.unique(gl)   # only letters actually used in this chunk
        for c in present:
            gmask = (gl == c)                                  # (c, 5)
            am = a_eq[c]                                       # (na, 5)
            gm3 = gmask[:, None, :]
            # non-green occurrences of c in the guess
            ng_mask = gm3 & ~am[None, :, :]                    # (c, na, 5)
            # greens of c consume budget first
            g_cnt = (gm3 & am[None, :, :]).sum(axis=2, dtype=np.int16)
            budget = a_cnt[c][None, :] - g_cnt                 # (c, na)
            rank = np.cumsum(ng_mask, axis=2, dtype=np.int16) - ng_mask
            yellow = ng_mask & (rank < budget[:, :, None])
            codes += np.tensordot(yellow.astype(np.int16), pow3, axes=([2], [0]))

        out[start:start + c_len] = codes.astype(np.uint8)
        if progress is not None:
            progress(min(start + chunk, ng_), ng_)
    return out


class FeedbackMatrix:
    """Thin wrapper tying a precomputed matrix to its vocabulary.

    Holds the (n_guesses, n_answers) uint8 table plus the index of each answer
    inside the guess list, so a solver can talk about "the guess pool" and "the
    surviving answers" without repeatedly rebuilding lookups.
    """

    def __init__(self, matrix: np.ndarray, vocab: Vocabulary):
        if matrix.shape != (vocab.n_guesses, vocab.n_answers):
            raise ValueError(
                f"matrix shape {matrix.shape} does not match vocabulary "
                f"{(vocab.n_guesses, vocab.n_answers)}"
            )
        if matrix.dtype != np.uint8:
            raise ValueError(f"matrix must be uint8, got {matrix.dtype}")
        self.M = matrix
        self.vocab = vocab
        self.answer_pos = vocab.answer_positions()

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        vocab: Vocabulary,
        *,
        chunk: int = 512,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> "FeedbackMatrix":
        M = build_feedback_matrix(
            vocab.guesses, vocab.answers, chunk=chunk, progress=progress
        )
        return cls(M, vocab)

    @property
    def nbytes(self) -> int:
        return int(self.M.size)

    # -- symbolic filtering, but as array ops ------------------------------
    def filter_indices(
        self,
        cand_idx: np.ndarray,
        guess: str,
        code: int,
    ) -> np.ndarray:
        """Restrict candidate answer indices to those consistent with (guess, code).

        Exactly `filter_candidates`, expressed as a row lookup. Verified against
        the pure-Python path in the test suite.
        """
        gi = self.vocab.guess_index[guess]
        row = self.M[gi]
        return cand_idx[row[cand_idx] == code]

    def partition_stats(
        self,
        guess_idx: np.ndarray,
        cand_idx: np.ndarray,
        *,
        chunk: int = 2048,
    ) -> Dict[str, np.ndarray]:
        """Partition statistics for every guess in `guess_idx`.

        For guess g, the candidate set is partitioned by observed feedback. Let
        `k` be the bucket sizes and `n = len(cand_idx)`. Returns, per guess:

          entropy   H(g) = -sum p log2 p,  with p = k / n
          expected  E[remaining] = sum k * (k/n) = sum k^2 / n
          worst     max k

        PROBABILITY MODEL — stated explicitly because it is the one modelling
        choice here: every surviving candidate is treated as equally likely,
        p = 1/n. This is the correct posterior under a uniform prior over the
        answer list, which is what the real game does (the daily answer is drawn
        from the answer list, not weighted by English word frequency). So
        p(feedback) is just bucket_size / n. No word-frequency prior is used by
        the information-theoretic solvers; that lives only in `FrequencyModel`.
        """
        n = int(len(cand_idx))
        m = int(len(guess_idx))
        ent = np.empty(m, dtype=np.float64)
        exp = np.empty(m, dtype=np.float64)
        worst = np.empty(m, dtype=np.int32)

        for s in range(0, m, chunk):
            gi = guess_idx[s:s + chunk]
            sub = self.M[np.ix_(gi, cand_idx)].astype(np.int32)   # (b, n)
            b = sub.shape[0]
            # Row-wise histogram over the 243 possible patterns via one bincount.
            flat = sub + (np.arange(b, dtype=np.int32) * N_PATTERNS)[:, None]
            hist = np.bincount(
                flat.ravel(), minlength=b * N_PATTERNS
            ).reshape(b, N_PATTERNS).astype(np.float64)

            p = hist / n
            # log2 only where p > 0; empty buckets contribute 0 to the entropy.
            logp = np.zeros_like(p)
            np.log2(p, out=logp, where=(p > 0))
            ent[s:s + b] = -(p * logp).sum(axis=1)
            exp[s:s + b] = (hist * hist).sum(axis=1) / n
            worst[s:s + b] = hist.max(axis=1).astype(np.int32)
        return {"entropy": ent, "expected_remaining": exp, "worst_case": worst}


# =============================================================================
# LAYER 3 — PROBABILITY / FREQUENCY HEURISTICS
# =============================================================================

@dataclass
class FrequencyModel:
    """Empirical letter statistics of the answer list.

    Two tables:
      `letter_prob[c]`         — fraction of answers containing letter c at all
                                 (presence, not multiplicity).
      `positional_prob[i][c]`  — fraction of answers with letter c at position i.

    Deliberately *presence*-based rather than count-based: scoring a word by
    summing per-occurrence letter frequencies rewards words like "eerie" for
    repeating a common letter, even though a repeat conveys strictly less new
    information than a fresh letter would. See `FrequencySolver.score` for how
    this is used.
    """

    letter_prob: Dict[str, float]
    positional_prob: List[Dict[str, float]]
    n_answers: int
    positional_weight: float = 0.5

    @classmethod
    def fit(cls, answers: Sequence[str], positional_weight: float = 0.5) -> "FrequencyModel":
        n = len(answers)
        pres = Counter()
        for w in answers:
            pres.update(set(w))          # set() => presence, not multiplicity
        pos = [Counter() for _ in range(WORD_LEN)]
        for w in answers:
            for i, ch in enumerate(w):
                pos[i][ch] += 1
        letters = [chr(ord("a") + k) for k in range(26)]
        return cls(
            letter_prob={c: pres[c] / n for c in letters},
            positional_prob=[{c: p[c] / n for c in letters} for p in pos],
            n_answers=n,
            positional_weight=positional_weight,
        )

    def score(self, word: str) -> float:
        """Heuristic desirability of `word` as a probe.

        score = sum over the word's DISTINCT letters of presence-probability
              + w * sum over positions of positional-probability

        The first term is the coverage payoff and counts each letter once, so a
        repeated letter earns nothing extra. The second term rewards putting
        common letters where they are commonly found, and is the only term that
        sees position.
        """
        cov = sum(self.letter_prob.get(c, 0.0) for c in set(word))
        pos = sum(self.positional_prob[i].get(word[i], 0.0) for i in range(WORD_LEN))
        return cov + self.positional_weight * pos

    def score_many(self, words: Sequence[str]) -> np.ndarray:
        return np.array([self.score(w) for w in words], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FrequencyModel":
        return cls(
            letter_prob=d["letter_prob"],
            positional_prob=d["positional_prob"],
            n_answers=d["n_answers"],
            positional_weight=d.get("positional_weight", 0.5),
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SolverConfig:
    """Single place to change every knob that affects solver behaviour."""

    # -- game rules ---------------------------------------------------------
    max_guesses: int = 6

    # -- strategy selection -------------------------------------------------
    strategy: str = "hybrid"

    # -- reproducibility ----------------------------------------------------
    seed: int = 20260817

    # -- guess pool policy --------------------------------------------------
    # "full"        : always score all legal guesses (strongest, slowest)
    # "adaptive"    : full pool while len(candidates) > adaptive_threshold,
    #                 candidates only below it
    # "candidates"  : only ever guess a surviving candidate (fastest, weakest)
    guess_pool: str = "full"
    adaptive_threshold: int = 2

    # Once this few candidates remain, guessing a non-candidate can never help:
    # with <= 2 left, probing cannot beat simply naming one of them.
    endgame_candidates: int = 2

    # Reserve the last guess for an actual candidate — a probe on the final turn
    # is guaranteed to lose.
    force_candidate_on_last_guess: bool = True

    # -- fixed opener -------------------------------------------------------
    # Turn 1 has no information, so its choice is the same for every game. Set
    # this to skip recomputing it 2315 times during a benchmark.
    opening_guess: Optional[str] = None

    # -- hybrid weights (all terms are in BITS; see HybridSolver) -----------
    entropy_weight: float = 1.0        # coefficient on H(guess)
    minimax_weight: float = 0.25       # lambda: penalty on log2(worst case)
    expected_weight: float = 0.25      # nu:     penalty on log2(E[remaining])
    answer_bonus_weight: float = 1.0   # mu:     credit for a possible-answer probe

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SolverConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# =============================================================================
# Solvers
# =============================================================================

@dataclass
class GameResult:
    """Outcome of a single game."""

    answer: str
    guesses: List[str]
    patterns: List[str]
    candidates_remaining: List[int]   # after each guess
    solved: bool
    n_guesses: int
    elapsed_s: float = 0.0

    @property
    def score(self) -> int:
        """Guesses used, or max_guesses + 1 as a conventional failure marker."""
        return self.n_guesses if self.solved else len(self.guesses) + 1


class Solver:
    """Base class. Subclasses implement `choose`.

    A solver is a pure function of (surviving candidates, turn number). The
    driver `play_game` owns the constraint bookkeeping so that every strategy
    sees exactly the same, provably-correct hypothesis set — differences between
    strategies are therefore purely differences in *decision rule*.
    """

    name = "base"
    deterministic = True

    def __init__(self, fb: FeedbackMatrix, config: Optional[SolverConfig] = None):
        self.fb = fb
        self.vocab = fb.vocab
        self.config = config or SolverConfig()
        self.rng = random.Random(self.config.seed)
        self._all_guesses = np.arange(self.vocab.n_guesses, dtype=np.int32)
        self._opener_cache: Optional[str] = None

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(self.config.seed if seed is None else seed)

    # -- guess pool policy --------------------------------------------------
    def _pool(self, cand_idx: np.ndarray, turn: int) -> np.ndarray:
        """Which guesses this solver is allowed to score on this turn."""
        cfg = self.config
        n = len(cand_idx)
        cand_as_guess = self.fb.answer_pos[cand_idx]

        # Final turn, or few enough candidates that probing cannot pay off.
        if cfg.force_candidate_on_last_guess and turn >= cfg.max_guesses:
            return cand_as_guess
        if n <= cfg.endgame_candidates:
            return cand_as_guess

        if cfg.guess_pool == "candidates":
            return cand_as_guess
        if cfg.guess_pool == "adaptive" and n <= cfg.adaptive_threshold:
            return cand_as_guess
        return self._all_guesses

    def choose(self, cand_idx: np.ndarray, turn: int) -> str:
        raise NotImplementedError

    # -- opener caching -----------------------------------------------------
    def opening_guess(self) -> str:
        """The turn-1 guess. Identical across games, so computed at most once."""
        if self.config.opening_guess:
            return self.config.opening_guess
        if self._opener_cache is None:
            all_answers = np.arange(self.vocab.n_answers, dtype=np.int32)
            self._opener_cache = self.choose(all_answers, 1)
        return self._opener_cache


class RandomSolver(Solver):
    """Baseline: name a surviving candidate uniformly at random.

    Pure constraint solving with no decision-making at all — it measures how
    much of the score comes from exact elimination alone. Never probes with a
    non-candidate, so it is the "symbolic layer only" reference point.
    """

    name = "random"
    deterministic = False

    def choose(self, cand_idx: np.ndarray, turn: int) -> str:
        i = self.rng.randrange(len(cand_idx))
        return self.vocab.answers[int(cand_idx[i])]

    def opening_guess(self) -> str:
        # Random has no canonical opener; draw one like any other turn.
        if self.config.opening_guess:
            return self.config.opening_guess
        return self.choose(np.arange(self.vocab.n_answers, dtype=np.int32), 1)


class FrequencySolver(Solver):
    """Rank surviving candidates by empirical letter frequency (Layer 3 only).

    Uses no partition structure whatsoever — it never asks what a guess would
    *reveal*, only whether its letters look typical of an answer. Guesses are
    restricted to surviving candidates, which makes it a direct upgrade over
    `RandomSolver` and isolates the value of frequency priors.
    """

    name = "frequency"

    def __init__(self, fb, config=None, model: Optional[FrequencyModel] = None):
        super().__init__(fb, config)
        self.model = model or FrequencyModel.fit(self.vocab.answers)
        self._cand_scores = self.model.score_many(self.vocab.answers)

    def choose(self, cand_idx: np.ndarray, turn: int) -> str:
        scores = self._cand_scores[cand_idx]
        best = int(cand_idx[int(np.argmax(scores))])
        return self.vocab.answers[best]


class _PartitionSolver(Solver):
    """Shared machinery for the three information-theoretic rules."""

    metric = "entropy"
    maximize = True

    def choose(self, cand_idx: np.ndarray, turn: int) -> str:
        n = len(cand_idx)
        if n == 1:
            return self.vocab.answers[int(cand_idx[0])]

        pool = self._pool(cand_idx, turn)
        stats = self.fb.partition_stats(pool, cand_idx)
        vals = stats[self.metric].astype(np.float64)

        # Tie-break toward guesses that could themselves be the answer: among
        # equally informative probes, one that might win outright is strictly
        # better. Without this, solvers waste turns on non-answers.
        cand_set = set(int(x) for x in self.fb.answer_pos[cand_idx])
        is_cand = np.fromiter(
            (int(g) in cand_set for g in pool), dtype=bool, count=len(pool)
        )
        eps = 1e-9
        adj = vals + (eps if self.maximize else -eps) * is_cand

        k = int(np.argmax(adj) if self.maximize else np.argmin(adj))
        return self.vocab.guesses[int(pool[k])]


class EntropySolver(_PartitionSolver):
    """Maximize expected information gain, H(guess) = -sum p log2 p.

    p(feedback) = bucket_size / n_candidates under a uniform posterior over the
    surviving answers (see `FeedbackMatrix.partition_stats`). H is measured in
    bits: it is the expected reduction in log2 of the hypothesis-space size, so
    a guess scoring 5.9 bits is expected to cut ~2315 candidates to ~40.

    This is one-step-greedy — it optimises immediate information, not the true
    minimum-expected-guesses game tree.
    """

    name = "entropy"
    metric = "entropy"
    maximize = True


class ExpectedRemainingSolver(_PartitionSolver):
    """Minimize E[remaining candidates] = sum bucket^2 / n.

    Closely related to entropy but not identical: entropy penalises by log of
    bucket size, this penalises linearly, so it is more tolerant of one large
    bucket if the rest are tiny. Optimises a different moment of the same
    partition.
    """

    name = "expected"
    metric = "expected_remaining"
    maximize = False


class MinimaxSolver(_PartitionSolver):
    """Minimize the largest resulting partition — worst-case, not average.

    Makes no probabilistic assumption at all beyond the candidate set itself:
    it asks only "if an adversary picked the answer, how bad could this get?"
    Tends to have a better maximum guess count and a slightly worse mean than
    the entropy rule.
    """

    name = "minimax"
    metric = "worst_case"
    maximize = False


class HybridSolver(Solver):
    """Combine all three partition statistics plus the answer-probability prior.

    NORMALIZATION — the reason the weights are meaningful rather than arbitrary:
    the four quantities have incompatible units (bits; a candidate count in
    [1, n]; a count in [1, n]; a probability). Rather than z-scoring, which makes
    weights depend on the spread of whatever pool happens to be scored this
    turn, every term is converted into BITS, which are all comparable and stable
    across turns:

        H(g)                         already bits of expected information
        log2(worst_case(g))          bits of hypothesis space left, worst case
        log2(E[remaining](g))        bits of hypothesis space left, in expectation
        p_answer(g) = 1/n if g is a surviving candidate else 0
                                     converted to bits of value as
                                     log2(1 + p_answer), the expected credit for
                                     possibly winning this turn outright

    score = a*H(g) - lam*log2(worst_case) - nu*log2(E[remaining])
            + mu*log2(1 + p_answer(g))

    All four terms are then on a bits scale, so lam = nu = 0.25 genuinely means
    "weight worst-case and expected-size at a quarter of raw information gain",
    and the default mu = 1.0 gives a possible-answer probe a small edge that
    grows as the candidate set shrinks (which is exactly when winning outright
    matters most). Defaults were chosen to be interpretable, then checked
    against the benchmark rather than tuned to it.
    """

    name = "hybrid"

    def __init__(self, fb, config=None, model: Optional[FrequencyModel] = None):
        super().__init__(fb, config)
        self.model = model or FrequencyModel.fit(self.vocab.answers)

    def choose(self, cand_idx: np.ndarray, turn: int) -> str:
        n = len(cand_idx)
        if n == 1:
            return self.vocab.answers[int(cand_idx[0])]

        pool = self._pool(cand_idx, turn)
        stats = self.fb.partition_stats(pool, cand_idx)
        cfg = self.config

        H = stats["entropy"]
        worst = np.log2(np.maximum(stats["worst_case"], 1).astype(np.float64))
        exp = np.log2(np.maximum(stats["expected_remaining"], 1.0))

        cand_set = set(int(x) for x in self.fb.answer_pos[cand_idx])
        is_cand = np.fromiter(
            (int(g) in cand_set for g in pool), dtype=bool, count=len(pool)
        )
        p_answer = is_cand.astype(np.float64) / n
        bonus = np.log2(1.0 + p_answer)

        score = (
            cfg.entropy_weight * H
            - cfg.minimax_weight * worst
            - cfg.expected_weight * exp
            + cfg.answer_bonus_weight * bonus
        )
        k = int(np.argmax(score))
        return self.vocab.guesses[int(pool[k])]


STRATEGIES: Dict[str, type] = {
    "random": RandomSolver,
    "frequency": FrequencySolver,
    "entropy": EntropySolver,
    "expected": ExpectedRemainingSolver,
    "minimax": MinimaxSolver,
    "hybrid": HybridSolver,
}


def make_solver(
    strategy: str,
    fb: FeedbackMatrix,
    config: Optional[SolverConfig] = None,
    model: Optional[FrequencyModel] = None,
) -> Solver:
    """Instantiate a solver by name."""
    key = strategy.lower()
    if key not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    cls = STRATEGIES[key]
    if cls in (FrequencySolver, HybridSolver):
        return cls(fb, config, model)
    return cls(fb, config)


# =============================================================================
# Game driver
# =============================================================================

def play_game(
    solver: Solver,
    answer: str,
    *,
    max_guesses: Optional[int] = None,
    verbose: bool = False,
    first_guess: Optional[str] = None,
) -> GameResult:
    """Play one full game and return a `GameResult`.

    The driver — not the solver — maintains the candidate set, using the
    precomputed matrix for elimination. Every strategy is therefore evaluated
    against an identical, exact constraint solver.
    """
    fb = solver.fb
    vocab = fb.vocab
    cfg = solver.config
    limit = cfg.max_guesses if max_guesses is None else max_guesses

    if answer not in vocab.answer_index:
        raise ValueError(f"{answer!r} is not in the answer list")

    cand_idx = np.arange(vocab.n_answers, dtype=np.int32)
    guesses: List[str] = []
    patterns: List[str] = []
    remaining: List[int] = []
    t0 = time.perf_counter()
    solved = False

    for turn in range(1, limit + 1):
        if turn == 1:
            g = first_guess or solver.opening_guess()
        else:
            g = solver.choose(cand_idx, turn)

        code = feedback_code(g, answer)
        pat = code_to_pattern(code)
        cand_idx = fb.filter_indices(cand_idx, g, code)

        guesses.append(g)
        patterns.append(pat)
        remaining.append(int(len(cand_idx)))

        if verbose:
            print(f"Guess {turn}: {g.upper()}")
            print(f"Feedback: {pat}")
            print(f"Candidates remaining: {len(cand_idx)}"
                  + (f"  -> {[vocab.answers[i] for i in cand_idx[:8]]}"
                     f"{' ...' if len(cand_idx) > 8 else ''}"
                     if 0 < len(cand_idx) <= 30 else ""))
            print()

        if code == ALL_GREEN:
            solved = True
            break
        if len(cand_idx) == 0:
            # Only reachable if the answer is outside the answer list, or the
            # feedback function and matrix disagree. Both are bugs, not states.
            raise RuntimeError(
                f"candidate set emptied while solving {answer!r} — "
                f"history={list(zip(guesses, patterns))}"
            )

    elapsed = time.perf_counter() - t0
    if verbose:
        if solved:
            print(f"Solved in {len(guesses)} guesses. ({elapsed*1000:.0f} ms)")
        else:
            print(f"FAILED in {limit} guesses. Answer was {answer.upper()}.")

    return GameResult(
        answer=answer, guesses=guesses, patterns=patterns,
        candidates_remaining=remaining, solved=solved,
        n_guesses=len(guesses), elapsed_s=elapsed,
    )


def solve(
    answer: str = "crane",
    *,
    bundle: Optional["WordleSolverBundle"] = None,
    strategy: Optional[str] = None,
    verbose: bool = True,
    artifact_dir: str = "artifacts",
) -> GameResult:
    """Convenience entry point: `solve(answer="CRANE", verbose=True)`.

    Loads artifacts from `artifact_dir` on first use if no bundle is supplied.
    """
    b = bundle or load_artifacts(artifact_dir)
    s = b.solver(strategy)
    return play_game(s, answer.strip().lower(), verbose=verbose)


def interactive_play(
    bundle: Optional["WordleSolverBundle"] = None,
    *,
    strategy: Optional[str] = None,
    artifact_dir: str = "artifacts",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> Optional[str]:
    """Manual play: the solver proposes, you type the feedback you saw.

    Enter feedback as 5 characters using G/Y/B (grey may also be typed as `_`,
    `.` or `0`). Type `q` to quit. Useful for driving the real NYT game.
    """
    b = bundle or load_artifacts(artifact_dir)
    s = b.solver(strategy)
    vocab = b.vocab
    cand_idx = np.arange(vocab.n_answers, dtype=np.int32)

    print_fn(f"Interactive Wordle solver — strategy={s.name}")
    print_fn("Enter feedback as 5 chars of G/Y/B (or 'q' to quit).\n")

    for turn in range(1, s.config.max_guesses + 1):
        g = s.opening_guess() if turn == 1 else s.choose(cand_idx, turn)
        print_fn(f"Guess {turn}: {g.upper()}   ({len(cand_idx)} candidates)")
        while True:
            raw = input_fn("  feedback> ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print_fn("Aborted.")
                return None
            try:
                code = pattern_to_code(raw)
                break
            except ValueError as exc:
                print_fn(f"  {exc}")
        if code == ALL_GREEN:
            print_fn(f"\nSolved in {turn} guesses: {g.upper()}")
            return g
        cand_idx = b.fb.filter_indices(cand_idx, g, code)
        if len(cand_idx) == 0:
            print_fn("\nNo candidates left — the feedback entered is inconsistent, "
                     "or the answer is not in this answer list.")
            return None
        if len(cand_idx) <= 15:
            print_fn("  remaining: "
                     + ", ".join(vocab.answers[i] for i in cand_idx))
    print_fn("\nOut of guesses.")
    return None


# =============================================================================
# Artifact persistence
# =============================================================================

ARTIFACT_FILES = {
    "answers": "answers.txt",
    "guesses": "valid_guesses.txt",
    "matrix": "feedback_matrix.npy",
    "metadata": "metadata.json",
    "frequency": "frequency_model.json",
    "config": "solver_config.json",
}


def _portable_path(p: Optional[str]) -> Optional[str]:
    """Reduce a path to something machine-agnostic for recording in metadata.

    Absolute paths from the build machine are useless to anyone else and leak the
    builder's directory layout, so they are stored relative to the working
    directory (forward-slashed) and reduced to a bare filename when a relative
    path would be meaningless (different drive, or far outside the tree).
    """
    if not p:
        return p
    p = str(p)
    try:
        rel = os.path.relpath(p, start=os.getcwd()).replace("\\", "/")
    except ValueError:          # different drive on Windows
        return os.path.basename(p)
    return os.path.basename(p) if rel.startswith("../../") else rel


def _is_memmap_of(arr: np.ndarray, path: str) -> bool:
    """True if `arr` is a numpy memmap backed by the file at `path`."""
    if not isinstance(arr, np.memmap):
        return False
    src = getattr(arr, "filename", None)
    if not src:
        return False
    try:
        return os.path.exists(path) and os.path.samefile(src, path)
    except OSError:
        return os.path.abspath(str(src)) == os.path.abspath(path)


def _matrix_unchanged_on_disk(path: str, M: np.ndarray) -> bool:
    """True if `path` already holds exactly the array `M`.

    Lets `save_artifacts` skip a pointless 28 MiB rewrite. That matters for more
    than speed: on Windows, any live memory map of the destination file — held by
    a *different* array, e.g. a bundle reloaded earlier in the same session —
    locks it, and np.save would fail with OSError(EINVAL). Comparing first turns
    the common build -> save -> reload -> save-again flow into a no-op instead of
    a crash.
    """
    if not os.path.exists(path):
        return False
    existing = None
    try:
        existing = np.load(path, mmap_mode="r")
        if existing.shape != M.shape or existing.dtype != M.dtype:
            return False
        return bool(np.array_equal(np.asarray(existing), np.asarray(M)))
    except Exception:
        return False
    finally:
        # Release the mapping promptly, or we become the thing holding the lock.
        mm = getattr(existing, "_mmap", None) if existing is not None else None
        del existing
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass


def save_artifacts(
    artifact_dir: str,
    vocab: Vocabulary,
    fb: FeedbackMatrix,
    model: FrequencyModel,
    config: SolverConfig,
    *,
    extra_metadata: Optional[dict] = None,
) -> Dict[str, str]:
    """Write a complete, portable solver bundle to `artifact_dir`.

    Everything needed to reconstruct the solver on another machine with only
    Python + NumPy. Paths are relative by design; nothing Kaggle-specific is
    recorded inside the artifacts.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    p = {k: os.path.join(artifact_dir, v) for k, v in ARTIFACT_FILES.items()}

    with open(p["answers"], "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(vocab.answers) + "\n")
    with open(p["guesses"], "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(vocab.guesses) + "\n")

    # Skip rewriting the matrix when the destination already holds exactly these
    # bytes — either because `fb.M` maps that very file, or because an identical
    # matrix is already there. Besides saving a 28 MiB write, this is what makes
    # build -> save -> reload -> save-again work on Windows, where any live
    # memory map of the file (possibly held by another array entirely) locks it
    # and np.save would fail with OSError(EINVAL).
    if _is_memmap_of(fb.M, p["matrix"]) or _matrix_unchanged_on_disk(p["matrix"], fb.M):
        pass
    else:
        try:
            np.save(p["matrix"], fb.M)
        except OSError as exc:
            raise OSError(
                f"could not write {p['matrix']!r}: {exc}. If another array is "
                f"memory-mapping this file, load it with mmap=False before saving."
            ) from exc

    with open(p["frequency"], "w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2)
    with open(p["config"], "w", encoding="utf-8") as fh:
        json.dump(config.to_dict(), fh, indent=2)

    import platform
    meta = {
        "format_version": 1,
        "word_len": WORD_LEN,
        "n_patterns": N_PATTERNS,
        "all_green_code": ALL_GREEN,
        "feedback_encoding": "base3 little-endian per position; 0=B(grey) 1=Y(yellow) 2=G(green); code=sum(tile[i]*3**i)",
        "n_answers": vocab.n_answers,
        "n_guesses": vocab.n_guesses,
        "n_non_answer_guesses": vocab.n_guesses - vocab.n_answers,
        "matrix_shape": list(fb.M.shape),
        "matrix_dtype": str(fb.M.dtype),
        "matrix_bytes": int(fb.M.nbytes),
        "matrix_orientation": "M[guess_index, answer_index]; rows follow valid_guesses.txt, cols follow answers.txt",
        # Recorded relative / basename-only so the bundle carries no absolute
        # paths from the build machine.
        "vocab_source": {k: _portable_path(v) for k, v in vocab.source.items()},
        "built_with": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategies": sorted(STRATEGIES),
        "notes": "Classical solver artifacts. No ML/LLM/RL components.",
    }
    if extra_metadata:
        meta.update(extra_metadata)
    with open(p["metadata"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return p


@dataclass
class WordleSolverBundle:
    """A loaded solver: vocabulary + matrix + frequency model + config."""

    vocab: Vocabulary
    fb: FeedbackMatrix
    model: FrequencyModel
    config: SolverConfig
    metadata: dict = field(default_factory=dict)

    def solver(self, strategy: Optional[str] = None) -> Solver:
        return make_solver(strategy or self.config.strategy, self.fb, self.config, self.model)

    def describe(self) -> str:
        return (
            f"{self.vocab.describe()}\n"
            f"matrix {self.fb.M.shape} {self.fb.M.dtype} "
            f"({self.fb.M.nbytes/2**20:.1f} MiB)\n"
            f"default strategy={self.config.strategy}, "
            f"guess_pool={self.config.guess_pool}, seed={self.config.seed}"
        )


def load_artifacts(
    artifact_dir: str = "artifacts",
    *,
    mmap: bool = True,
) -> WordleSolverBundle:
    """Reload a bundle written by `save_artifacts` — no rebuilding required.

    `mmap=True` memory-maps the feedback matrix, so start-up is instant and the
    ~28 MiB table is paged in on demand rather than copied into RAM.
    """
    p = {k: os.path.join(artifact_dir, v) for k, v in ARTIFACT_FILES.items()}
    for k in ("answers", "guesses", "matrix"):
        if not os.path.exists(p[k]):
            raise FileNotFoundError(
                f"missing artifact {p[k]!r}. Build artifacts first "
                f"(see build_artifacts.py or the notebook)."
            )

    with open(p["answers"], encoding="utf-8") as fh:
        answers = [w for w in fh.read().split() if w]
    with open(p["guesses"], encoding="utf-8") as fh:
        guesses = [w for w in fh.read().split() if w]
    vocab = Vocabulary(answers=answers, guesses=guesses)

    M = np.load(p["matrix"], mmap_mode="r" if mmap else None)
    fb = FeedbackMatrix(M, vocab)

    if os.path.exists(p["frequency"]):
        with open(p["frequency"], encoding="utf-8") as fh:
            model = FrequencyModel.from_dict(json.load(fh))
    else:
        model = FrequencyModel.fit(answers)

    if os.path.exists(p["config"]):
        with open(p["config"], encoding="utf-8") as fh:
            config = SolverConfig.from_dict(json.load(fh))
    else:
        config = SolverConfig()

    metadata = {}
    if os.path.exists(p["metadata"]):
        with open(p["metadata"], encoding="utf-8") as fh:
            metadata = json.load(fh)

    return WordleSolverBundle(vocab=vocab, fb=fb, model=model,
                              config=config, metadata=metadata)


# =============================================================================
# CLI
# =============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="wordle_solver",
        description="Classical Wordle solver (no ML/LLM/RL).",
    )
    ap.add_argument("--artifact-dir", default="artifacts",
                    help="directory holding the built artifacts (default: artifacts)")
    ap.add_argument("--strategy", default=None, choices=sorted(STRATEGIES),
                    help="override the strategy stored in solver_config.json")
    ap.add_argument("--answer", default=None,
                    help="solve this answer verbosely, e.g. --answer crane")
    ap.add_argument("--interactive", action="store_true",
                    help="play against the real game, entering feedback by hand")
    ap.add_argument("--info", action="store_true", help="describe the loaded bundle")
    args = ap.parse_args(argv)

    bundle = load_artifacts(args.artifact_dir)
    if args.info or (not args.answer and not args.interactive):
        print(bundle.describe())
        if not args.answer and not args.interactive:
            return 0
    if args.answer:
        play_game(bundle.solver(args.strategy), args.answer.strip().lower(), verbose=True)
    if args.interactive:
        interactive_play(bundle, strategy=args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
