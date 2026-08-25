"""
build_endgame_dataset.py - synthesise endgame-heavy SFT data for tree_salet.

WHY
---
Phase 5 verdict D: the models transfer the expert's policy (100% opener, 90%
turn-2 agreement) yet lose 56% of games with vocabulary errors made impossible,
and name the word only 20% of the time when the state determines it uniquely.

The cause is the shape of the endgame data, and it is NOT a shortage of endgame
states. tree_salet's natural trajectories contain:

    n_candidates == 1 :  1,612 examples
    distinct words     :  1,612
    states per word    :  exactly 1, for every single word

So the winning move is demonstrated 1,612 times - but as a 1,612-way retrieval
task with exactly ONE example per class. Each word is reachable by one
memorised path. Nothing in that teaches "read the constraints, produce the word
that satisfies them" as a general procedure, which is what evaluation demands
on words whose single path was never shown.

(The `fully_determined` flag - all five greens known - is 4 in the same file,
but that is a much narrower condition than "the state determines the answer"
and is not the relevant measure.)

The fix this script implements is therefore MANY DISTINCT CONSTRAINT PATHS PER
WORD, not simply more endgame rows.

HOW
---
Natural trajectories are ON-POLICY: they only show states the expert itself
reaches. The model does not play like the expert, so at evaluation it lands in
states the demonstrations never covered - the classic behavioural-cloning
distribution shift. The fix is the standard one (DAgger-style noise injection):
roll out with a deliberately imperfect policy, then label the states reached
with the expert.

    turn 1 is always SALET      - the deployed opener, so sampled states are
                                  reachable at evaluation time
    turns 2+ are randomised     - a mix of surviving candidates and arbitrary
                                  legal words, which is what a fallible model
                                  actually does
    the label is the expert     - tree_salet's choice at that state

Descent uses only cheap policies; the expert is called ONCE per accepted state.
That matters: `choose()` costs ~0.1-18 ms at 1-10 candidates but ~1.4 s at 40,
so labelling is cheap while expert descent would not be.

WHAT THE MODEL SEES
-------------------
Exactly what it saw before. `render_prompt(candidates=None,
show_candidate_count=False)` - no candidate list, no candidate count, no answer.
The candidate set is used internally to pick and verify states and never
reaches the prompt. `verify_endgame_dataset.py` asserts this.

Source answers are the 2,069 TRAINING answers only. The 246 held-out answers
are never used to generate a state, so no evaluation state is manufactured.

    python build_endgame_dataset.py --out sft_package/data/tree_salet_endgame
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = _paths.ROOT  # data lives at the repo root
sys.path.insert(0, HERE)

from wordle_solver import (load_artifacts, SolverConfig, feedback_code,
                           code_to_pattern, feedback, ALL_GREEN)
from tree_search import TreeSearchConfig, TreeSearchSolver
from generate_trajectories import derive_constraints, render_prompt

MAX_GUESSES = 6
SEED = 20260817

# Distinct states to manufacture PER TRAINING ANSWER, per bucket.
#
# Per-answer, not a global total. The whole point is that the natural data has
# exactly one k=1 state per word; giving each word several different constraint
# paths to itself is the intervention being tested. k=1 gets the most because
# that is the state the Phase 5 probe showed failing (20% top-1).
DEFAULT_PER_ANSWER = {"1": 3, "2": 1, "3": 1, "4-10": 1}


def bucket_of(n):
    if n <= 0:
        return None
    if n <= 3:
        return str(n)
    if n <= 10:
        return "4-10"
    return None


def opaque_id(kind, answer, turn, salt):
    h = hashlib.sha256(f"{kind}|{answer}|{turn}|{salt}".encode()).hexdigest()
    return f"eg-{h[:14]}"


def load_train_answers(sft_dir):
    """The 2,069 training answers. Authoritative - not re-derived from a hash."""
    p = os.path.join(sft_dir, "eval", "train_answers.jsonl")
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l)["answer"].lower() for l in fh if l.strip()]


def make_record(answer, history, cand_idx, turn, expert, bundle, vocab):
    """Label one sampled state with the expert action. Returns an SFT record."""
    n_before = int(len(cand_idx))
    guess = expert.choose(cand_idx, turn)

    constraints = derive_constraints(history)
    prompt = render_prompt(
        turn=turn,
        history=history,
        constraints=constraints,
        n_candidates=n_before,          # analysis metadata only
        guesses_remaining=MAX_GUESSES - turn + 1,
        max_guesses=MAX_GUESSES,
        candidates=None,                # NEVER shown
        show_candidate_count=False,     # NEVER shown
    )

    ai = vocab.answer_index.get(guess, -1)
    was_cand = bool(ai >= 0 and ai in set(int(x) for x in cand_idx))
    code = feedback_code(guess, answer)
    n_after = int(len(bundle.fb.filter_indices(cand_idx, guess, code)))

    return {
        "id": opaque_id("endgame", answer, turn,
                        "|".join(f"{g}:{p}" for g, p in history)),
        "prompt": prompt,
        "completion": guess.upper(),
        "meta": {
            "solver": "tree_salet",
            "source": "endgame_synthetic",
            "turn": turn,
            "n_candidates": n_before,
            "bits_remaining": round(math.log2(n_before), 4) if n_before else 0.0,
            "action_is_candidate": was_cand,
            "action_is_possible_answer": bool(guess in vocab.answer_index),
            "information_gain_bits": (
                round(math.log2(n_before) - math.log2(n_after), 4)
                if n_after else round(math.log2(n_before), 4)),
            "fully_determined": constraints["fully_determined"],
            "is_disagreement_state": False,
            "duplicate_count": 1,
            "split": "train",
            "bucket": bucket_of(n_before),
        },
    }


def sample(per_answer, answers, bundle, expert, rng, max_turn=5,
           p_candidate=0.55, max_rollouts_per_answer=40, log_every=250):
    """For EACH training answer, harvest several distinct states per bucket.

    Answer-driven rather than target-driven: the intervention is coverage per
    word, so every training answer gets its own quota. A word whose quota
    cannot be met (some answers are hard to strand at exactly k=3, say) simply
    contributes fewer rows rather than skewing the sample toward easy words.
    """
    vocab = bundle.vocab
    guesses = vocab.guesses
    out = []
    got = Counter()
    rollouts = 0
    t0 = time.perf_counter()

    for ai, answer in enumerate(answers):
        need = Counter(per_answer)
        seen_prompts = set()
        for _ in range(max_rollouts_per_answer):
            if not any(need[b] > 0 for b in need):
                break
            rollouts += 1
            cand_idx = np.arange(vocab.n_answers, dtype=np.int32)
            history = []
            guess = "salet"                  # deployed opener, always turn 1

            for _ in range(max_turn):
                code = feedback_code(guess, answer)
                if code == ALL_GREEN:
                    break                    # solved; no state left to ask about
                history.append((guess, feedback(guess, answer)))
                cand_idx = bundle.fb.filter_indices(cand_idx, guess, code)
                n = int(len(cand_idx))
                turn = len(history) + 1
                if n == 0 or turn > MAX_GUESSES:
                    break

                b = bucket_of(n)
                if b is not None and need[b] > 0:
                    rec = make_record(answer, list(history), cand_idx, turn,
                                      expert, bundle, vocab)
                    if rec["prompt"] not in seen_prompts:
                        seen_prompts.add(rec["prompt"])
                        out.append(rec)
                        need[b] -= 1
                        got[b] += 1

                if n == 1:
                    break                    # only the answer remains
                if rng.random() < p_candidate:
                    guess = vocab.answers[int(cand_idx[rng.integers(n)])]
                else:
                    guess = guesses[rng.integers(len(guesses))]

        if log_every and (ai + 1) % log_every == 0:
            el = time.perf_counter() - t0
            print(f"  {ai+1:5d}/{len(answers)} answers  {len(out):6d} states  "
                  f"{el:5.0f}s  ~{el/(ai+1)*(len(answers)-ai-1):.0f}s left  "
                  f"{dict(got)}", flush=True)

    return out, got, rollouts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--out", default="sft_package/data/tree_salet_endgame")
    ap.add_argument("--seed", type=int, default=SEED)
    for k, v in DEFAULT_PER_ANSWER.items():
        ap.add_argument(f"--per{k.replace('-', '_')}", type=int, default=v)
    ap.add_argument("--limit-answers", type=int, default=0,
                    help="debug: only use the first N training answers")
    a = ap.parse_args(argv)

    per_answer = {"1": a.per1, "2": a.per2, "3": a.per3, "4-10": a.per4_10}

    print("=" * 68)
    print("ENDGAME DATASET SYNTHESIS - tree_salet")
    print("=" * 68)
    print(f"states per answer, per bucket: {per_answer}")

    bundle = load_artifacts(a.artifacts, mmap=True)
    cfg = SolverConfig(max_guesses=MAX_GUESSES, seed=a.seed, guess_pool="full")
    tc = TreeSearchConfig(depth=6, top_k=100, endgame_top_k=60,
                          endgame_threshold=10, opening_guess="salet",
                          max_guesses=MAX_GUESSES)
    expert = TreeSearchSolver(bundle.fb, cfg, tc)

    answers = load_train_answers(a.sft_dir)
    if a.limit_answers:
        answers = answers[:a.limit_answers]
    print(f"source answers: {len(answers)} TRAINING answers "
          f"(the 246 held-out are never used)")
    print(f"upper bound: {len(answers) * sum(per_answer.values())} states")

    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    rows, got, attempts = sample(per_answer, answers, bundle, expert, rng)
    el = time.perf_counter() - t0

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "train.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} records to {path} in {el:.0f}s "
          f"({attempts} rollouts)")
    print(f"per bucket: {dict(sorted(got.items()))}")
    print(f"turn distribution: "
          f"{dict(sorted(Counter(r['meta']['turn'] for r in rows).items()))}")
    print(f"distinct completions: {len(set(r['completion'] for r in rows))}")

    # The metric this whole exercise exists to move.
    k1 = [r for r in rows if r["meta"]["n_candidates"] == 1]
    per = Counter(r["completion"] for r in k1)
    print(f"\nk=1 states: {len(k1)} covering {len(per)} distinct words")
    print(f"  paths per word: {dict(sorted(Counter(per.values()).items()))}")
    print(f"  mean paths per word: {len(k1)/max(len(per),1):.2f}   "
          f"(natural tree_salet data: exactly 1.00)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
