"""
rerender_rows.py - rewrite the `prompt` field of SFT rows into a different
prompt variant, changing nothing else.

    python phase10_crossover/rerender_rows.py --variant raw_history --dry-run
    python phase10_crossover/rerender_rows.py --variant raw_history --out <dir>

WHY THIS EXISTS

Phase 9 measured a 0.52-guess penalty for `raw_history` - the prompt with the
solver-derived constraint block removed - and probe B measured the model
emitting 1.4% admissible words under it against baseline's 22.3%. Both results
are confounded: the adapter was trained on `baseline` alone, so "baseline wins"
may only mean "the adapter recognises baseline". Separating format lock-in from
an intrinsic property of the format needs a second adapter trained on the other
format, which needs this file.

THE SAFETY ARGUMENT

The rows do not store the game history as data - only as text inside the
already-rendered baseline prompt. So the history is parsed back out, and then,
before anything is written:

    v_baseline(turn, parsed_history, max_guesses) == stored_prompt   (byte-for-byte)

If the parsed history re-renders the stored prompt exactly, the parse is
provably lossless and the only difference in the output is the variant. Any
mismatch aborts the whole run rather than writing a single row. Verified
2026-08-23 against all 19,212 rows of tree_salet + tree_salet_endgame: 19,212
exact round-trips, zero parse failures.

`completion` and every `meta` field are copied through untouched, so the
expert action, the split, the turn and the candidate count are identical
between the two training sets. The ONLY difference is the prompt text - which
is the entire point of the experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "core"))

import prompt_variants as pv  # noqa: E402

HIST_BLOCK = re.compile(r"^History:\n((?:  \d+\. .+\n)+)", re.M)
HIST_LINE = re.compile(r"\s*\d+\.\s+([A-Za-z]{5})\s*->\s*([GYB]{5})\s*$")
TURN_LINE = re.compile(r"^Guess (\d+) of (\d+)", re.M)


def parse_state(prompt):
    """(turn, max_guesses, history) recovered from a baseline-rendered prompt,
    or None if it does not parse. history is [(guess_lower, GYB_pattern)]."""
    tm = TURN_LINE.search(prompt)
    if not tm:
        return None
    turn, max_guesses = int(tm.group(1)), int(tm.group(2))
    if "History: (none" in prompt:
        return turn, max_guesses, []
    bm = HIST_BLOCK.search(prompt)
    if not bm:
        return None
    history = []
    for line in bm.group(1).rstrip("\n").split("\n"):
        lm = HIST_LINE.match(line)
        if not lm:
            return None
        history.append((lm.group(1).lower(), lm.group(2)))
    return turn, max_guesses, history


def rerender_row(row, variant):
    """Return the row with `prompt` replaced, or raise if the round-trip fails."""
    st = parse_state(row["prompt"])
    if st is None:
        raise ValueError(f"row {row.get('id')!r}: prompt did not parse")
    turn, max_guesses, history = st
    # The proof: if baseline re-renders byte-for-byte from the parsed state,
    # the parse lost nothing and the variant render is trustworthy.
    if pv.v_baseline(turn, history, max_guesses) != row["prompt"]:
        raise ValueError(
            f"row {row.get('id')!r}: round-trip mismatch - the parsed history "
            f"does not reproduce the stored prompt. Refusing to write.")
    out = dict(row)
    out["prompt"] = pv.render(variant, turn, history, max_guesses,
                              n_candidates=row.get("meta", {}).get(
                                  "n_candidates", 0))
    return out


def rerender_file(src, dst, variant, dry_run=False):
    n = 0
    rows = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            new = rerender_row(row, variant)
            assert new["completion"] == row["completion"], "completion changed"
            assert new.get("meta") == row.get("meta"), "meta changed"
            assert new["prompt"].rstrip().endswith("Next guess:"), \
                "variant does not end in the decoder's anchor"
            rows.append(new)
            n += 1
    if not dry_run:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n, rows


DEFAULT_FILES = [
    "data/tree_salet/train.jsonl",
    "data/tree_salet/val.jsonl",
    "data/tree_salet_endgame/train.jsonl",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="raw_history",
                    help="a name from core/prompt_variants.py VARIANTS")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--out", default=None,
                    help="output root; default sft_package_<variant>")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.variant not in pv.VARIANTS:
        raise SystemExit(f"unknown variant {a.variant!r}; "
                         f"choose from {sorted(pv.VARIANTS)}")
    if pv.VARIANTS[a.variant]["leaky"]:
        raise SystemExit(f"{a.variant!r} is leaky - never train on it.")

    out_root = a.out or f"sft_package_{a.variant}"
    total = 0
    for rel in DEFAULT_FILES:
        src = os.path.join(a.sft_dir, rel)
        if not os.path.exists(src):
            print(f"  skip (absent): {src}")
            continue
        dst = os.path.join(out_root, rel)
        n, rows = rerender_file(src, dst, a.variant, a.dry_run)
        total += n
        print(f"  {rel:44s} {n:>7,} rows -> "
              f"{'(dry run)' if a.dry_run else dst}")
        if rows:
            print("      sample prompt tail: "
                  + repr(rows[0]['prompt'][-70:]))
    print(f"\n{total:,} rows re-rendered as {a.variant!r}; "
          f"all round-trips verified.")


if __name__ == "__main__":
    main()
