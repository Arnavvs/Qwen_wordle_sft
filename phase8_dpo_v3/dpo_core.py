"""Shared machinery for the Phase-8 v3 DPO pipeline.

Three things live here, used by the generator, the verifier and the
legacy-audit script so that all three agree by construction:

  1. `Board`      — the game state, and the only place admissible / candidate
                    sets are derived. Everything downstream re-derives from a
                    rendered prompt through this class, so a metadata field can
                    never silently disagree with the prompt it describes.
  2. `AdaptiveTree` — the value function for the restricted regime.
  3. `OnePlyScorer` — the value function for the unrestricted regime.

WHY TWO VALUE FUNCTIONS
-----------------------
Under the Phase-7b adaptive decoder the model's action space depends on the
board:

    |admissible| <= 20   the decoder restricts the model to feedback-consistent
                         legal words
    |admissible| >  20   the model may emit any legal word

Those are genuinely different decision problems and they get genuinely
different labels. In the restricted regime the admissible set is small and —
crucially — *monotone*: filtering can only ever remove words, so once a game is
below the threshold it stays below it. That makes the decoder's remaining
decision process a small, exactly-solvable finite tree, and `AdaptiveTree`
solves it to full depth with no horizon truncation. In the unrestricted regime
the action space is all 12,972 legal words and no exact multi-ply value is
affordable, so `OnePlyScorer` is used and every row it labels is *marked* as a
horizon-1 proxy in its metadata.

NO CANDIDATE BONUS
------------------
The Phase-8 v1 generator ranked actions by

    cost(w) = E[remaining] - 0.5 * (w is a candidate)

The 0.5 is an unmotivated constant, and the audit shows it is not a tie-break
in practice — it is *the entire label* for 75.9% of that dataset's
"competitive" pairs, and it inverts the underlying information measure on 336
of them. Nothing here uses it. In the restricted regime the exact tree already
values winning-now correctly, because guessing a candidate genuinely ends some
branches a turn early; that is the principled version of the same idea and it
is derived rather than assumed.
"""
import functools
import hashlib
import os

import numpy as np

from wordle_solver import (ALL_GREEN, build_feedback_matrix, feedback,
                           load_artifacts, pattern_to_code)
from generate_trajectories import derive_constraints, render_prompt

MAX_GUESSES = 6

# The Phase-7b setting the deployed decoder actually runs at (best mean,
# 3.7642 on the 246 held-out answers). The action space in this file is derived
# from it, not from a guess about it.
ADAPTIVE_THRESHOLD = 20

# A lost game costs this many guesses. Only reachable inside the tree when a
# state cannot be resolved within the turns that genuinely remain; it is a
# scoring convention, so it is recorded in every manifest.
FAILURE_COST = 7


# =============================================================================
# vocabulary + feedback rows
# =============================================================================

class Vocab:
    """Legal-guess indexed view of the shipped artifacts bundle.

    Everything in this pipeline is expressed in *guess indices* (0..12971),
    including candidate answers. Answers are a subset of legal guesses, so one
    index space covers both and there is no answer-index/guess-index confusion
    of the kind that is easy to introduce and hard to see.
    """

    def __init__(self, artifacts="artifacts"):
        self.bundle = load_artifacts(artifacts, mmap=True)
        v = self.bundle.vocab
        self.words = list(v.guesses)
        self.index = dict(v.guess_index)
        self.n = len(self.words)
        self.answers = list(v.answers)
        self.answer_idx = np.array([self.index[w] for w in self.answers],
                                   dtype=np.int32)
        # position of each guess index inside the answer list, -1 if not an answer
        self.answer_col = np.full(self.n, -1, dtype=np.int32)
        self.answer_col[self.answer_idx] = np.arange(len(self.answers),
                                                     dtype=np.int32)
        self.is_answer = self.answer_col >= 0

    @functools.lru_cache(maxsize=4096)
    def _row(self, g_idx):
        """feedback_code(words[g_idx], w) for every legal word w.

        Computed on demand with the vectorised builder and cached. Verified
        byte-identical to a precomputed 12,972 x 12,972 table; computing rows
        lazily keeps the pipeline dependent on nothing but the shipped
        `artifacts/` bundle, which is what makes it reproducible on Kaggle.
        """
        return build_feedback_matrix([self.words[g_idx]], self.words)[0]

    def codes(self, g_idx, word_idx):
        """Feedback codes of guess `g_idx` against an array of word indices."""
        return self._row(int(g_idx))[word_idx]

    def code_against(self, g_idx, target_idx):
        return int(self._row(int(g_idx))[int(target_idx)])


# =============================================================================
# board state
# =============================================================================

class Board:
    """A reachable Wordle position.

    `admissible` is every legal guess consistent with all feedback so far — the
    hard-mode set, a pure function of the prompt plus the public word list, and
    exactly what the deployed decoder computes. `candidates` is the subset of
    those that are also on the answer list; it is solver-side privileged
    information and never reaches the model.
    """

    __slots__ = ("V", "history", "admissible", "candidates")

    def __init__(self, V, history=(), admissible=None, candidates=None):
        self.V = V
        self.history = list(history)
        self.admissible = (np.arange(V.n, dtype=np.int32)
                           if admissible is None else admissible)
        self.candidates = (V.answer_idx.copy() if candidates is None
                           else candidates)

    def play(self, g_idx, code):
        """Return the board after guess `g_idx` produced feedback `code`."""
        row = self.V._row(int(g_idx))
        return Board(
            self.V,
            self.history + [(self.V.words[int(g_idx)], _pattern(code))],
            self.admissible[row[self.admissible] == code],
            self.candidates[row[self.candidates] == code],
        )

    @property
    def turn(self):
        return len(self.history) + 1

    @property
    def guesses_left(self):
        return MAX_GUESSES - self.turn + 1

    @property
    def restricted(self):
        """True iff the deployed decoder would filter the model here."""
        return len(self.admissible) <= ADAPTIVE_THRESHOLD

    @property
    def action_space(self):
        return "admissible" if self.restricted else "legal"

    def actions(self):
        """The action set the deployed decoder actually offers the model."""
        if self.restricted:
            return self.admissible
        return np.arange(self.V.n, dtype=np.int32)

    def state_key(self):
        return "|".join(f"{g}:{p}" for g, p in self.history)

    def prompt(self):
        """The model-facing prompt.

        Identical `render_prompt` call as every earlier phase. The answer is not
        a parameter of that function at all, `candidates=None` withholds the
        candidate list, and `show_candidate_count=False` withholds the count —
        so no answer, no candidate set, no candidate count, no admissible count
        and no score can appear. `n_candidates` is passed only to satisfy the
        signature and is unreachable with the count switched off.
        """
        return render_prompt(
            turn=self.turn,
            history=list(self.history),
            constraints=derive_constraints(self.history),
            n_candidates=0,
            guesses_remaining=self.guesses_left,
            max_guesses=MAX_GUESSES,
            candidates=None,
            show_candidate_count=False,
        )

    @classmethod
    def from_history(cls, V, history):
        """Rebuild a board from `[(guess, pattern), ...]`.

        The verifier uses this to re-derive every state from its own rendered
        prompt, which is what makes the metadata auditable rather than merely
        asserted.
        """
        b = cls(V)
        for g, pat in history:
            b = b.play(V.index[g.lower()], pattern_to_code(pat))
        return b


@functools.lru_cache(maxsize=512)
def _pattern(code):
    from wordle_solver import code_to_pattern
    return code_to_pattern(code)


# =============================================================================
# value functions
# =============================================================================

class AdaptiveTree:
    """Exact expected-guesses value for the *decoder's own* decision process.

    Solves, to full remaining depth, the finite game the adaptive decoder plays
    once it has restricted the model: at every node the action set is refined by
    the feedback just received, exactly as the decoder refines it at run time.

    Costs are integers — the summed number of guesses over every candidate still
    possible in the state — so comparisons between actions are exact. There is
    no floating-point tie-break and therefore no label noise: two actions are
    equal only if they are genuinely equal, and a reported gap of `1/|S|` is a
    real one-guess difference on one candidate rather than a rounding artefact.

    This is well defined only in the restricted regime, and that is not a
    limitation being papered over — it is a property of the decoder. Filtering
    only ever removes words, so a board below the threshold has every successor
    below the threshold too, and the tree it induces is closed.
    """

    def __init__(self, V, failure_cost=FAILURE_COST):
        self.V = V
        self.failure_cost = failure_cost
        self._memo = {}
        self.nodes = 0

    def _cost(self, S, A, depth):
        """Minimum total guesses over all candidates in `S`, playing optimally."""
        n = len(S)
        if n == 0:
            return 0
        # Depth is checked before the singleton shortcut: with no guesses left
        # even a state with one candidate is a loss, and testing in the other
        # order silently awards a free win on the last branch of a deep line.
        if depth <= 0 or len(A) == 0:
            return self.failure_cost * n
        if n == 1:
            # The one surviving candidate is an answer consistent with every
            # observation, hence itself admissible: it can always be named.
            return 1
        key = (S.tobytes(), A.tobytes(), depth)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.nodes += 1
        best = 1 << 30
        for g in _candidates_first(self.V, A):
            total = self._branch(S, A, int(g), depth, bound=best)
            if total < best:
                best = total
        self._memo[key] = best
        return best

    def _branch(self, S, A, g, depth, bound=1 << 30):
        """Total cost of playing `g` in state (S, A), with branch-and-bound."""
        row = self.V._row(g)
        cs = row[S]
        total = len(S)                      # one guess spent on every candidate
        ca = row[A]
        for code in np.unique(cs):
            if int(code) == ALL_GREEN:
                continue                    # that branch won outright
            total += self._cost(S[cs == code], A[ca == code], depth - 1)
            if total >= bound:
                return total                # cannot beat the incumbent
        return total

    def action_costs(self, board):
        """Integer total cost of every action available on `board`.

        Returns `{guess_index: total_cost}`. Divide by `len(candidates)` for the
        mean number of further guesses.
        """
        S, A = board.candidates, board.admissible
        depth = board.guesses_left
        return {int(g): self._branch(S, A, int(g), depth)
                for g in A}


def _candidates_first(V, A):
    """Actions ordered candidates-first, so branch-and-bound prunes early.

    A candidate can win outright, so trying one first almost always yields a
    tight upper bound immediately. Same ordering trick as `core/tree_search.py`.
    """
    is_c = V.is_answer[A]
    return np.concatenate([A[is_c], A[~is_c]])


class OnePlyScorer:
    """Exhaustive one-ply E[remaining candidates] over the whole legal pool.

    Used only where the decoder does not restrict the model and an exact
    multi-ply value is unaffordable. Two deliberate differences from the v1
    generator: the pool is *every* legal word rather than a random 120-word
    sample, so the argmin is the true argmin; and there is no candidate bonus,
    so the number means what it says.
    """

    def __init__(self, V):
        self.V = V
        self._all = np.arange(V.n, dtype=np.int32)

    def action_scores(self, board):
        cols = self.V.answer_col[board.candidates]
        stats = self.V.bundle.fb.partition_stats(self._all, cols)
        return stats["expected_remaining"]


# =============================================================================
# small helpers shared by generator and verifier
# =============================================================================

def sha(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def bucket_of(n_admissible):
    """Buckets keyed on |admissible|, matching the Phase-8 headroom analysis."""
    if n_admissible < 2:
        return None
    if n_admissible <= 10:
        return "2-10"
    if n_admissible <= 100:
        return "11-100"
    return "100+"


BUCKETS = ("2-10", "11-100", "100+")


def load_answer_splits(sft_dir="sft_package"):
    import json
    def read(name):
        p = os.path.join(sft_dir, "eval", name)
        with open(p, encoding="utf-8") as fh:
            return [json.loads(l)["answer"].lower() for l in fh if l.strip()]
    train, val = read("train_answers.jsonl"), read("val_answers.jsonl")
    if set(train) & set(val):
        raise AssertionError("source answer splits overlap")
    return train, val


def check_feedback(V, guess, pattern, answer):
    """Independent confirmation that a history line is what the answer implies."""
    return V.code_against(V.index[guess.lower()],
                          V.index[answer.lower()]) == pattern_to_code(pattern)


__all__ = ["MAX_GUESSES", "ADAPTIVE_THRESHOLD", "FAILURE_COST", "Vocab",
           "Board", "AdaptiveTree", "OnePlyScorer", "sha", "bucket_of",
           "BUCKETS", "load_answer_splits", "check_feedback", "feedback"]
