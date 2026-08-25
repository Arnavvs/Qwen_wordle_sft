"""
build_artifacts.py — one-shot precomputation step.

Reads the two word lists from DATA_DIR, builds the full feedback matrix, and
writes a portable artifact bundle to ARTIFACT_DIR. Run once; after that the
solver loads in milliseconds via `wordle_solver.load_artifacts`.

    python build_artifacts.py
    python build_artifacts.py --data-dir data --artifact-dir artifacts

Both directories default to paths relative to this file, so the script behaves
identically on Windows and Linux and contains no Kaggle-specific logic.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import os
import sys
import time

import numpy as np

from wordle_solver import (
    FeedbackMatrix, FrequencyModel, SolverConfig, feedback_code,
    load_vocabulary, save_artifacts,
)

HERE = _paths.ROOT  # data lives at the repo root
DEFAULT_DATA_DIR = os.path.join(HERE, "data")
DEFAULT_ARTIFACT_DIR = os.path.join(HERE, "artifacts")

# Candidate filenames, tried in order. Extended rather than assumed — the
# notebook's discovery layer handles arbitrary Kaggle dataset layouts.
ANSWER_NAMES = [
    "wordle_answers.txt", "answers.txt", "wordle-answers-alphabetical.txt",
    "valid_solutions.txt", "solutions.txt",
]
GUESS_NAMES = [
    "wordle_allowed_guesses.txt", "valid_guesses.txt",
    "wordle-allowed-guesses.txt", "allowed_guesses.txt", "valid_words.txt",
]


def find_first(data_dir: str, names) -> str:
    for n in names:
        p = os.path.join(data_dir, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"none of {names} found in {data_dir!r}. "
        f"Present: {sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else 'directory missing'}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--answers", default=None, help="explicit path to the answer list")
    ap.add_argument("--guesses", default=None, help="explicit path to the guess list")
    ap.add_argument("--chunk", type=int, default=512,
                    help="guesses per vectorized block (memory/speed tradeoff)")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--strategy", default="hybrid")
    ap.add_argument("--verify", type=int, default=20000,
                    help="random cells to check against the reference implementation")
    args = ap.parse_args(argv)

    a_path = args.answers or find_first(args.data_dir, ANSWER_NAMES)
    g_path = args.guesses or find_first(args.data_dir, GUESS_NAMES)
    print(f"answers : {a_path}")
    print(f"guesses : {g_path}")

    vocab = load_vocabulary(a_path, g_path)
    print(vocab.describe())

    print(f"\nbuilding feedback matrix "
          f"{vocab.n_guesses} x {vocab.n_answers} uint8 "
          f"({vocab.n_guesses * vocab.n_answers / 2**20:.1f} MiB) ...")
    t0 = time.perf_counter()
    last = [0.0]

    def prog(done, total):
        now = time.perf_counter()
        if now - last[0] > 2.0 or done == total:
            last[0] = now
            el = now - t0
            print(f"  {done}/{total} guesses  {el:.1f}s  "
                  f"(eta {el / max(done, 1) * (total - done):.0f}s)")

    fb = FeedbackMatrix.build(vocab, chunk=args.chunk, progress=prog)
    build_s = time.perf_counter() - t0
    print(f"built in {build_s:.1f}s")

    # --- verification against the pure-Python reference --------------------
    if args.verify > 0:
        rng = np.random.default_rng(args.seed)
        gi = rng.integers(0, vocab.n_guesses, args.verify)
        ai = rng.integers(0, vocab.n_answers, args.verify)
        bad = 0
        for g, a in zip(gi, ai):
            if fb.M[g, a] != feedback_code(vocab.guesses[g], vocab.answers[a]):
                bad += 1
        if bad:
            print(f"FAIL: {bad}/{args.verify} cells disagree with reference")
            return 1
        print(f"verified {args.verify} random cells against the reference: OK")
        # The diagonal is a cheap global invariant: every answer guessed against
        # itself must be all-green.
        diag = fb.M[fb.answer_pos, np.arange(vocab.n_answers)]
        assert (diag == 242).all(), "diagonal is not all-green"
        print("verified all-green diagonal for every answer: OK")

    model = FrequencyModel.fit(vocab.answers)
    config = SolverConfig(seed=args.seed, strategy=args.strategy)

    paths = save_artifacts(
        args.artifact_dir, vocab, fb, model, config,
        extra_metadata={
            "matrix_build_seconds": round(build_s, 2),
            "matrix_build_chunk": args.chunk,
            "verified_random_cells": args.verify,
        },
    )
    print(f"\nwrote artifacts to {args.artifact_dir}:")
    for k, p in paths.items():
        print(f"  {k:<10} {os.path.basename(p):<22} "
              f"{os.path.getsize(p)/1024:>10.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
