"""
generate_trajectories.py — expert Wordle trajectories for SFT / GRPO.

Produces one decision-level record per guess and one game-level record per game,
for any solver in the project. The intended use is comparing SFT on *optimal*
demonstrations (tree search with the SALET opener, which attains the proven
minimum of 7920 total guesses) against SFT on *greedy* demonstrations (the
entropy solver, 8020) on identical states and an identical split.

    python generate_trajectories.py --solver tree_salet
    python generate_trajectories.py --solver entropy
    python generate_trajectories.py --solver all --verify

THE ANSWER IS NEVER IN THE MODEL INPUT
--------------------------------------
This is the property the whole dataset depends on, so it is enforced three ways
rather than asserted once:

1. **By construction.** `render_prompt` is a pure function of
   `(turn, history, derived constraints, n_candidates, guesses_remaining)`. The
   answer is not a parameter of it and cannot reach it.

2. **Structurally, it cannot appear in history.** Guessing the answer always
   returns GGGGG and ends the game, so a correct guess is always the *last*
   guess and therefore never appears in the history of a later prompt. There is
   no decision after the winning guess, so no prompt can ever contain it.

3. **By verification.** `--verify` re-reads every emitted record and asserts the
   answer is absent from the prompt as a whole-token match, that the recorded
   candidate count matches an independent pure-Python recomputation from the
   history alone, and that identical history prefixes always yield identical
   actions (no hidden state).

The answer is kept in each record under `_holdout` for analysis and reward
computation. It is namespaced with a leading underscore precisely so a training
pipeline can drop every `_`-prefixed key mechanically. `--strip-holdout` removes
it entirely.

THE CANDIDATE LIST IS OFF BY DEFAULT
------------------------------------
`--include-candidates` adds the surviving answer list to the prompt. It is off
by default because when few candidates remain that list literally contains the
answer string. That is *not* leakage — it is derivable from the visible feedback
by exact filtering, so a perfect reasoner would know it. But it hands the model
the deduction instead of asking for it, and it breaks the guarantee above. Turn
it on only if you specifically want to train the choice given the candidate set
rather than the deduction plus the choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wordle_solver import (
    ALL_GREEN, SolverConfig, feedback, feedback_code, filter_history,
    load_artifacts, make_solver,
)
from tree_search import TreeSearchConfig, TreeSearchSolver

SEED = 20260817
WORD_LEN = 5

SOLVERS = {
    # label            kind      opener   note
    "tree_salet":  ("tree", "salet", "optimal demonstrations (attains proven minimum 7920)"),
    "tree_soare":  ("tree", "soare", "tree search, greedy-entropy opener"),
    "entropy":     ("classic", None, "one-step greedy max-information (8020)"),
    "expected":    ("classic", None, "one-step greedy min expected remaining"),
    "minimax":     ("classic", None, "one-step greedy min worst case"),
    "frequency":   ("classic", None, "letter-frequency heuristic"),
}


# =============================================================================
# Visible game state
# =============================================================================

def derive_constraints(history: Sequence[Tuple[str, str]]) -> Dict:
    """Everything logically deducible from the feedback so far.

    All of this is redundant with `history` — a perfect reasoner could derive it —
    but stating it explicitly is standard for this kind of dataset and lets you
    ablate how much of the task is deduction versus choice.

    Duplicate letters are handled exactly as the feedback rule implies:
      * a G or Y for letter c raises the known MINIMUM count of c
      * a B for c, in a guess that also scored some G/Y for c, pins the EXACT
        count of c (the greys prove there are no more)
      * a B for c with no G/Y for c anywhere in that guess means c is absent
      * any non-green tile at position i rules c out of position i
    """
    greens: List[Optional[str]] = [None] * WORD_LEN
    min_count: Dict[str, int] = {}
    exact_count: Dict[str, int] = {}
    forbidden: Dict[str, set] = {}

    for guess, pattern in history:
        marked = Counter()
        greyed = set()
        for i, (ch, t) in enumerate(zip(guess, pattern)):
            if t == "G":
                greens[i] = ch
                marked[ch] += 1
            elif t == "Y":
                marked[ch] += 1
                forbidden.setdefault(ch, set()).add(i)
            else:
                greyed.add(ch)
                forbidden.setdefault(ch, set()).add(i)
        for ch, k in marked.items():
            min_count[ch] = max(min_count.get(ch, 0), k)
        for ch in greyed:
            # A grey proves the answer holds exactly as many `ch` as were marked
            # in THIS guess (0 if none were marked).
            exact_count[ch] = marked.get(ch, 0)

    absent = sorted(c for c, k in exact_count.items() if k == 0)
    present = sorted(c for c in min_count if min_count[c] > 0)
    return {
        "greens": greens,
        "green_pattern": "".join(g if g else "." for g in greens),
        # True when feedback has greened all five positions. The answer is then
        # fully determined WITHOUT having been guessed -- e.g. CLOOT + ABAFT
        # green every position of ALOFT. That is legitimate deduction, not
        # leakage, but it means any contiguous rendering of the pattern would be
        # the answer itself, so `render_prompt` emits it position by position.
        # These steps also teach nothing but "type the word you already know",
        # so filter on this flag if you want only genuine decisions.
        "fully_determined": all(g is not None for g in greens),
        "present_letters": present,
        "min_counts": {c: min_count[c] for c in present},
        "exact_counts": {c: k for c, k in sorted(exact_count.items()) if k > 0},
        "absent_letters": absent,
        "forbidden_positions": {c: sorted(v) for c, v in sorted(forbidden.items())
                                if c in min_count},
    }


RULES = (
    "You are playing Wordle. Deduce the hidden 5-letter word.\n"
    "After each guess you receive 5 feedback characters:\n"
    "  G = correct letter in the correct position\n"
    "  Y = letter is in the word but in a different position\n"
    "  B = letter is not in the word (or all its copies are already accounted for)\n"
)


def render_prompt(
    turn: int,
    history: Sequence[Tuple[str, str]],
    constraints: Dict,
    n_candidates: int,
    guesses_remaining: int,
    max_guesses: int,
    candidates: Optional[Sequence[str]] = None,
    show_candidate_count: bool = False,
) -> str:
    """Render the visible state. Pure function of its arguments — no answer.

    Deliberately has no `answer` parameter, so leakage is impossible here rather
    than merely unlikely.

    `show_candidate_count` defaults to **False**: the number of surviving answers
    is a solver-side quantity that a player cannot see, and handing it over
    leaks how much the current constraints actually pin down. It stays available
    in `n_candidates` for analysis and in record metadata, but it is not part of
    the model's input. The parameter exists only so the original rendering can
    be reproduced for verification; defaulting it to False means a caller that
    forgets the argument gets the safe behaviour.
    """
    parts = [RULES, ""]
    if history:
        parts.append("History:")
        for i, (g, p) in enumerate(history, 1):
            parts.append(f"  {i}. {g.upper()} -> {p}")
    else:
        parts.append("History: (none - this is the opening guess)")
    parts.append("")

    # Everything is uppercased to match the history and the completion, so the
    # same letter is always the same token wherever it appears.
    c = constraints
    parts.append("Deduced so far:")
    # Position by position, never as a contiguous word: once all five are known
    # the concatenation would BE the answer (see `fully_determined`).
    fixed = " ".join(f"p{i+1}={g.upper()}"
                     for i, g in enumerate(c["greens"]) if g)
    parts.append(f"  Confirmed letters : {fixed if fixed else '(none)'}")
    parts.append(f"  Letters present   : "
                 f"{', '.join(x.upper() for x in c['present_letters']) if c['present_letters'] else '(none)'}")
    if c["exact_counts"]:
        parts.append("  Exact counts      : "
                     + ", ".join(f"{k.upper()}x{v}" for k, v in c["exact_counts"].items()))
    parts.append(f"  Letters absent    : "
                 f"{', '.join(x.upper() for x in c['absent_letters']) if c['absent_letters'] else '(none)'}")
    if c["forbidden_positions"]:
        parts.append("  Ruled-out spots   : "
                     + ", ".join(f"{k.upper()} not at {v}" for k, v in
                                 c["forbidden_positions"].items()))
    parts.append("")
    if show_candidate_count:
        parts.append(f"Possible answers remaining: {n_candidates}")
    if candidates is not None:
        parts.append(f"Candidates: {', '.join(w.upper() for w in candidates)}")
    parts.append(f"Guess {turn} of {max_guesses} ({guesses_remaining} remaining)")
    parts.append("")
    parts.append("Next guess:")
    return "\n".join(parts)


# =============================================================================
# Generation
# =============================================================================

def build_solver(label: str, bundle, cfg: SolverConfig):
    kind, opener, _ = SOLVERS[label]
    if kind == "tree":
        tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                              endgame_threshold=10, opening_guess=opener,
                              max_guesses=cfg.max_guesses)
        sv = TreeSearchSolver(bundle.fb, cfg, tc)
        return sv, opener
    sv = make_solver(label, bundle.fb, cfg, bundle.model)
    sv.reset()
    return sv, (sv.opening_guess() if sv.deterministic else None)


def split_for(answer: str, val_frac: float, salt: str = "wordle-split-v1") -> str:
    """Deterministic per-ANSWER split, identical across solvers.

    Keyed on the answer word rather than a row index so every solver's dataset
    puts the same words in validation. Without that, comparing SFT-on-entropy
    against SFT-on-optimal would confound the demonstration source with the
    evaluation set.
    """
    h = hashlib.sha256((salt + "|" + answer).encode()).hexdigest()
    return "val" if (int(h[:8], 16) / 0xFFFFFFFF) < val_frac else "train"


def generate(
    label: str,
    bundle,
    *,
    max_guesses: int = 6,
    include_candidates: bool = False,
    candidate_cap: int = 30,
    val_frac: float = 0.1,
    limit: int = 0,
    log_every: int = 500,
) -> Tuple[List[Dict], List[Dict]]:
    V = bundle.vocab
    cfg = SolverConfig(max_guesses=max_guesses, seed=SEED, guess_pool="full")
    solver, opener = build_solver(label, bundle, cfg)

    answers = V.answers[:limit] if limit else V.answers
    steps: List[Dict] = []
    games: List[Dict] = []
    t0 = time.perf_counter()

    for gi, answer in enumerate(answers):
        history: List[Tuple[str, str]] = []
        cand_idx = np.arange(V.n_answers, dtype=np.int32)
        game_steps: List[Dict] = []
        solved = False

        for turn in range(1, max_guesses + 1):
            n_before = int(len(cand_idx))
            constraints = derive_constraints(history)
            cand_words = None
            if include_candidates and n_before <= candidate_cap:
                cand_words = [V.answers[i] for i in cand_idx]

            prompt = render_prompt(
                turn=turn, history=history, constraints=constraints,
                n_candidates=n_before, guesses_remaining=max_guesses - turn + 1,
                max_guesses=max_guesses, candidates=cand_words,
            )

            # --- the expert action ---------------------------------------
            if turn == 1 and opener:
                guess = opener
            else:
                guess = solver.choose(cand_idx, turn)

            # Was this probe still a live hypothesis, or a pure information
            # play? This is the field that separates "named a candidate" from
            # "spent a turn splitting the space", which is the behaviour the
            # later model most needs to learn.
            ai = V.answer_index.get(guess, -1)
            was_candidate = bool(ai >= 0 and ai in set(int(x) for x in cand_idx))

            code = feedback_code(guess, answer)
            pattern = feedback(guess, answer)
            cand_idx = bundle.fb.filter_indices(cand_idx, guess, code)
            n_after = int(len(cand_idx))

            game_steps.append({
                "solver": label,
                "game_id": f"{label}/{gi:05d}",
                "turn": turn,
                "prompt": prompt,
                "completion": guess.upper(),
                "state": {
                    "turn": turn,
                    "guesses_remaining": max_guesses - turn + 1,
                    "history": [{"guess": g.upper(), "feedback": p}
                                for g, p in history],
                    "n_candidates": n_before,
                    "bits_remaining": round(math.log2(n_before), 4) if n_before else 0.0,
                    "constraints": constraints,
                    "candidates_shown": ([w.upper() for w in cand_words]
                                         if cand_words else None),
                },
                "action": {
                    "word": guess.upper(),
                    "is_possible_answer": bool(guess in V.answer_index),
                    "was_in_candidate_set": was_candidate,
                    "feedback_received": pattern,
                    "n_candidates_after": n_after,
                    "bits_after": round(math.log2(n_after), 4) if n_after else 0.0,
                    "information_gain_bits": round(
                        math.log2(n_before) - math.log2(n_after), 4)
                    if n_after else round(math.log2(n_before), 4),
                },
                "_holdout": {"answer": answer.upper()},
            })

            history.append((guess, pattern))
            if code == ALL_GREEN:
                solved = True
                break

        total = len(game_steps)
        split = split_for(answer, val_frac)
        outcome = {
            "solved": solved,
            "total_guesses": total if solved else None,
            "score": total if solved else max_guesses + 1,
        }
        for k, st in enumerate(game_steps):
            st["outcome"] = dict(outcome)
            st["outcome"]["turns_after_this"] = total - (k + 1)
            st["split"] = split
        steps.extend(game_steps)

        games.append({
            "game_id": f"{label}/{gi:05d}",
            "solver": label,
            "opener": (opener or game_steps[0]["completion"]).upper(),
            "n_turns": total,
            "solved": solved,
            "score": outcome["score"],
            "guesses": [s["completion"] for s in game_steps],
            "feedback": [s["action"]["feedback_received"] for s in game_steps],
            "candidates_trajectory": [s["state"]["n_candidates"] for s in game_steps],
            "split": split,
            "_holdout": {"answer": answer.upper()},
        })

        if log_every and (gi + 1) % log_every == 0:
            print(f"  [{label}] {gi+1}/{len(answers)} games, {len(steps)} steps "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    return steps, games


# =============================================================================
# Verification
# =============================================================================

def verify(steps: List[Dict], games: List[Dict], bundle,
           include_candidates: bool) -> List[str]:
    """Re-derive everything from the emitted records. Returns a list of failures."""
    V = bundle.vocab
    legal = set(w.upper() for w in V.guesses)
    problems: List[str] = []

    # 1. No answer in any prompt.
    #
    # Scanned against the VARIABLE region only. The constant rules preamble is
    # excluded because it is byte-identical in every record and therefore
    # carries zero information about the answer -- and because ordinary English
    # words in it ("After ...") collide with legitimate answers like AFTER,
    # which is a false positive, not a leak. Check 1b below proves the preamble
    # really is constant, so excluding it cannot hide anything.
    if not include_candidates:
        # Words belonging to the fixed template are not content. They are
        # collected by rendering an empty state rather than hard-coded, so the
        # exclusion stays correct if the template is ever reworded. This matters
        # because ordinary template words like "Guess" and "Exact" are
        # themselves valid Wordle answers and would otherwise fire constantly.
        blank = render_prompt(
            turn=1, history=[],
            constraints=derive_constraints([]),
            n_candidates=0, guesses_remaining=0, max_guesses=6,
        )
        template_tokens = set(
            t.strip(" ,.:->()[]").upper()
            for t in blank.replace("\n", " ").split()
        )

        for s in steps:
            ans = s["_holdout"]["answer"].upper()

            # (a) precise content scan: the only places word-level content can
            #     enter the prompt at all.
            content = {h["guess"].upper() for h in s["state"]["history"]}
            if s["state"].get("candidates_shown"):
                content |= {w.upper() for w in s["state"]["candidates_shown"]}
            if ans in content:
                problems.append(
                    f"ANSWER LEAK (content): {ans} in {s['game_id']} "
                    f"turn {s['turn']}")

            # (b) belt-and-braces full-text scan, template words removed, to
            #     catch anything reaching the prompt by an unexpected route.
            body = s["prompt"].split(RULES, 1)[-1]
            tokens = {
                t.strip(" ,.:->()[]").upper()
                for t in body.replace("\n", " ").split()
            } - template_tokens
            if ans in tokens:
                problems.append(
                    f"ANSWER LEAK (text): {ans} in {s['game_id']} "
                    f"turn {s['turn']}")
            if len(problems) > 5:
                return problems

        # 1b. the excluded preamble is genuinely constant
        for s in steps:
            if not s["prompt"].startswith(RULES):
                problems.append(f"PREAMBLE DRIFT in {s['game_id']} turn {s['turn']}")
                break

        # 1c. Strongest form: the prompt must be a function of the visible
        # history alone. If two games with DIFFERENT answers reach the same
        # history, their prompts must be byte-identical -- otherwise something
        # answer-dependent is reaching the input.
        by_hist: Dict[str, Tuple[str, str]] = {}
        for s in steps:
            key = "|".join(f"{h['guess']}:{h['feedback']}"
                           for h in s["state"]["history"])
            prev = by_hist.get(key)
            if prev is not None and prev[0] != s["prompt"]:
                problems.append(
                    f"ANSWER-DEPENDENT PROMPT after history [{key}]: "
                    f"{prev[1]} vs {s['game_id']} differ")
                if len(problems) > 5:
                    return problems
            by_hist[key] = (s["prompt"], s["game_id"])

    # 2. every action is a legal guess
    for s in steps:
        if s["completion"] not in legal:
            problems.append(f"ILLEGAL GUESS {s['completion']} in {s['game_id']}")

    # 3. candidate counts recomputed independently, pure Python, from history only
    checked = 0
    for s in steps:
        hist = [(h["guess"].lower(), h["feedback"]) for h in s["state"]["history"]]
        expect = len(filter_history(V.answers, hist)) if hist else V.n_answers
        if expect != s["state"]["n_candidates"]:
            problems.append(
                f"COUNT MISMATCH {s['game_id']} turn {s['turn']}: "
                f"recorded {s['state']['n_candidates']} vs recomputed {expect}")
            if len(problems) > 5:
                return problems
        checked += 1
        if checked >= 4000:      # independent recomputation is O(n) per step
            break

    # 4. determinism: identical history prefix must give an identical action
    seen: Dict[str, str] = {}
    for s in steps:
        key = "|".join(f"{h['guess']}:{h['feedback']}" for h in s["state"]["history"])
        prev = seen.get(key)
        if prev is not None and prev != s["completion"]:
            problems.append(
                f"NONDETERMINISM after history [{key}]: {prev} vs {s['completion']}")
            if len(problems) > 5:
                return problems
        seen[key] = s["completion"]

    # 5. the answer must never be a non-final guess
    for g in games:
        ans = g["_holdout"]["answer"]
        for i, w in enumerate(g["guesses"]):
            if w == ans and i != len(g["guesses"]) - 1:
                problems.append(f"ANSWER GUESSED EARLY in {g['game_id']}")

    return problems


# =============================================================================
# IO
# =============================================================================

def write_jsonl(path: str, rows: List[Dict], strip_holdout: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            if strip_holdout:
                r = {k: v for k, v in r.items() if not k.startswith("_")}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(label: str, steps: List[Dict], games: List[Dict]) -> Dict:
    n = len(games)
    scores = [g["score"] for g in games]
    turn_hist = Counter(g["n_turns"] for g in games if g["solved"])
    return {
        "solver": label,
        "note": SOLVERS[label][2],
        "n_games": n,
        "n_steps": len(steps),
        "opener": games[0]["opener"] if games else None,
        "mean_guesses": round(sum(scores) / n, 4),
        "total_guesses": sum(scores),
        "solved": sum(1 for g in games if g["solved"]),
        "failed": sum(1 for g in games if not g["solved"]),
        "turn_distribution": dict(sorted(turn_hist.items())),
        "steps_train": sum(1 for s in steps if s["split"] == "train"),
        "steps_val": sum(1 for s in steps if s["split"] == "val"),
        "pct_actions_that_are_possible_answers": round(
            100 * sum(1 for s in steps if s["action"]["is_possible_answer"])
            / max(len(steps), 1), 2),
        "mean_information_gain_bits": round(
            sum(s["action"]["information_gain_bits"] for s in steps)
            / max(len(steps), 1), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="all",
                    help="one of " + ", ".join(SOLVERS) + ", or 'all', or 'compare'")
    ap.add_argument("--artifact-dir", default="artifacts")
    ap.add_argument("--out-dir", default="trajectories")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-guesses", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--include-candidates", action="store_true",
                    help="put the surviving answer list in the prompt "
                         "(BREAKS the answer-never-in-input guarantee)")
    ap.add_argument("--candidate-cap", type=int, default=30)
    ap.add_argument("--strip-holdout", action="store_true",
                    help="drop the _holdout answer field from the written files")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    if args.solver == "all":
        labels = list(SOLVERS)
    elif args.solver == "compare":
        labels = ["tree_salet", "entropy"]
    else:
        labels = [args.solver]
    for l in labels:
        if l not in SOLVERS:
            raise SystemExit(f"unknown solver {l!r}; choose from {list(SOLVERS)}")

    bundle = load_artifacts(args.artifact_dir)
    print(f"vocabulary: {bundle.vocab.n_answers} answers / "
          f"{bundle.vocab.n_guesses} legal guesses")
    if args.include_candidates:
        print("\n!! --include-candidates is ON: prompts may contain the answer "
              "when few candidates remain. The leak check is disabled.\n")

    summaries = []
    for label in labels:
        print(f"\n=== {label} — {SOLVERS[label][2]} ===", flush=True)
        t0 = time.perf_counter()
        steps, games = generate(
            label, bundle, max_guesses=args.max_guesses,
            include_candidates=args.include_candidates,
            candidate_cap=args.candidate_cap,
            val_frac=args.val_frac, limit=args.limit,
        )
        el = time.perf_counter() - t0

        if args.verify:
            print(f"  verifying {len(steps)} steps ...", flush=True)
            probs = verify(steps, games, bundle, args.include_candidates)
            if probs:
                print("  VERIFICATION FAILED:")
                for p in probs[:10]:
                    print(f"    {p}")
                return 1
            print("  verification passed: no answer in any prompt, all guesses "
                  "legal, candidate counts independently reproduced, "
                  "actions deterministic in the history.")

        sp = os.path.join(args.out_dir, f"{label}_steps.jsonl")
        gp = os.path.join(args.out_dir, f"{label}_games.jsonl")
        write_jsonl(sp, steps, args.strip_holdout)
        write_jsonl(gp, games, args.strip_holdout)

        s = summarize(label, steps, games)
        s["seconds"] = round(el, 1)
        s["files"] = {"steps": sp, "games": gp}
        summaries.append(s)
        print(f"  {s['n_games']} games, {s['n_steps']} steps, "
              f"mean={s['mean_guesses']:.4f}, total={s['total_guesses']}, "
              f"train/val steps={s['steps_train']}/{s['steps_val']}  ({el:.0f}s)")
        print(f"  wrote {sp}")
        print(f"  wrote {gp}")

    card = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "vocabulary": {"answers": bundle.vocab.n_answers,
                       "guesses": bundle.vocab.n_guesses,
                       "source": "original Wordle lists (2315 / 12972)"},
        "max_guesses": args.max_guesses,
        "val_frac": args.val_frac,
        "split_key": "sha256(salt|answer) — identical across solvers by design",
        "include_candidates": args.include_candidates,
        "answer_visibility": (
            "The answer is never a parameter of render_prompt and cannot appear "
            "in history (a correct guess ends the game). Held out under "
            "'_holdout'; strip all '_'-prefixed keys for training."
            if not args.include_candidates else
            "WARNING: --include-candidates was set; prompts may contain the answer."
        ),
        "reference_results": {
            "tree_salet_total": 7920, "tree_salet_mean": 7920 / 2315,
            "entropy_total": 8020, "entropy_mean": 8020 / 2315,
            "proven_optimal_total": 7920,
        },
        "solvers": summaries,
    }
    cp = os.path.join(args.out_dir, "dataset_card.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    print(f"\nwrote {cp}")

    if len(summaries) > 1:
        print(f"\n{'solver':<14}{'games':>7}{'steps':>8}{'mean':>9}{'total':>8}"
              f"{'%answer-probes':>16}{'bits/step':>11}")
        print("-" * 73)
        for s in summaries:
            print(f"{s['solver']:<14}{s['n_games']:>7}{s['n_steps']:>8}"
                  f"{s['mean_guesses']:>9.4f}{s['total_guesses']:>8}"
                  f"{s['pct_actions_that_are_possible_answers']:>16.1f}"
                  f"{s['mean_information_gain_bits']:>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
