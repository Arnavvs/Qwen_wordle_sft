"""
verify_no_leakage.py — audit the SFT package for answer leakage and split hygiene.

    python verify_no_leakage.py
    python verify_no_leakage.py --package sft_package --strict

Exits non-zero if any hard check fails. Soft findings are reported as WARN and
do not fail the run; they are properties you need to know about rather than
defects (see the contamination check).

Hard checks
  1. No `_holdout` key, and no field literally named `answer`, in any
     model-facing file.
  2. No `game_id` in model-facing records (it encodes the answer's row index,
     so it is invertible by anyone holding the answer list).
  3. The true answer never appears as a token in a prompt. Cross-referenced
     through index/id_map.jsonl, so this is checked against the real answer for
     each record rather than assumed.
  4. Every completion is a legal 5-letter guess.
  5. Prompt is a pure function of history: identical history => identical
     prompt, across every solver and split.
  6. Train and validation answer sets are disjoint.
  7. The disagreement oversampling pool touches no validation game.

Soft check
  8. Prompt-level overlap between train and val. With an answer-keyed split this
     is EXPECTED and large: a turn-2 state is reached by many answers, so the
     same prompt legitimately appears on both sides. It is reported because it
     governs which evaluation metrics are meaningful.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List

MODEL_FACING_GLOBS = [
    ("data", "entropy", "train.jsonl"), ("data", "entropy", "val.jsonl"),
    ("data", "tree_soare", "train.jsonl"), ("data", "tree_soare", "val.jsonl"),
    ("data", "tree_salet", "train.jsonl"), ("data", "tree_salet", "val.jsonl"),
    ("data", "disagreement", "entropy.jsonl"),
    ("data", "disagreement", "tree_soare.jsonl"),
]

RULES_MARKER = "B = letter is not in the word"


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _template_tokens() -> set:
    """Words belonging to the fixed prompt scaffolding, not to any game state.

    Derived by rendering an empty state with the real generator, so this stays
    correct if the template is reworded. It matters because template words like
    "Guess" are themselves valid Wordle answers -- without this, every record of
    the game whose answer is GUESS reports a false leak.
    """
    try:
        from generate_trajectories import (RULES, derive_constraints,
                                           render_prompt)
    except Exception:
        return set()
    blank = render_prompt(turn=1, history=[],
                          constraints=derive_constraints([]),
                          n_candidates=0, guesses_remaining=0, max_guesses=6)
    body = blank.split(RULES, 1)[-1]
    return {t.strip(" ,.:->()[]").upper()
            for t in body.replace("\n", " ").split()}


TEMPLATE_TOKENS = _template_tokens()


def tokens_of(prompt: str) -> set:
    body = prompt.split(RULES_MARKER, 1)[-1]
    toks = {t.strip(" ,.:->()[]").upper()
            for t in body.replace("\n", " ").split()}
    return toks - TEMPLATE_TOKENS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="sft_package")
    ap.add_argument("--vocab", default="artifacts/valid_guesses.txt")
    ap.add_argument("--strict", action="store_true",
                    help="treat soft findings as failures too")
    args = ap.parse_args(argv)

    P = args.package
    fails: List[str] = []
    warns: List[str] = []

    files = {}
    for parts in MODEL_FACING_GLOBS:
        p = os.path.join(P, *parts)
        if os.path.exists(p):
            files[os.path.join(*parts)] = load_jsonl(p)
    if not files:
        print(f"no model-facing files found under {P}")
        return 1
    total = sum(len(v) for v in files.values())
    print(f"model-facing files: {len(files)}, records: {total}\n")

    idmap = {}
    imp = os.path.join(P, "index", "id_map.jsonl")
    if os.path.exists(imp):
        for r in load_jsonl(imp):
            idmap[r["id"]] = r

    legal = set()
    if os.path.exists(args.vocab):
        with open(args.vocab, encoding="utf-8") as fh:
            legal = {w.strip().upper() for w in fh if w.strip()}

    # ---- 1 & 2: forbidden fields ----------------------------------------
    bad_fields = collections.Counter()
    for name, rows in files.items():
        for r in rows:
            for k in r:
                if k.startswith("_") or k == "answer":
                    bad_fields[f"{name}:{k}"] += 1
            for k in r.get("meta", {}):
                if k.startswith("_") or k in ("answer", "game_id"):
                    bad_fields[f"{name}:meta.{k}"] += 1
    if bad_fields:
        fails.append(f"forbidden fields present: {dict(bad_fields)}")
    print(f"[{'PASS' if not bad_fields else 'FAIL'}] 1&2 no _holdout / answer / "
          f"game_id fields in model-facing records")

    # ---- 3: answer never in prompt --------------------------------------
    #
    # Only the DYNAMIC region of the prompt counts. The static scaffold
    # ("After each guess you receive...", "Guess 3 of 6", "Next guess:") is
    # byte-identical in every prompt and so carries no information about any
    # answer -- but it contains the English words GUESS and AFTER, which are
    # themselves valid Wordle answers. Matching the whole prompt reports those
    # as leaks; it is a false positive, and it fired on 9 prompts (all GUESS)
    # before this fix. Scanning from "History:" onward, minus the turn counter
    # lines, leaves exactly the parts that vary with the answer.
    # Parenthesised text is also scaffold: the only parenthesised strings in
    # these prompts are the placeholders "(none)" and "(none - this is the
    # opening guess)". That last one contains the word GUESS, which is a valid
    # answer, and it appears in the turn-1 prompt -- a prompt that is
    # byte-identical for all 2,069 training games and therefore cannot encode
    # any answer. Real state (guesses, feedback, letters, positions) is never
    # parenthesised; ruled-out spots use brackets.
    import re as _re

    def dynamic_region(p):
        i = p.find("History:")
        body = p[i:] if i >= 0 else p
        body = "\n".join(ln for ln in body.splitlines()
                         if not ln.startswith("Guess ")
                         and not ln.startswith("Next guess"))
        return _re.sub(r"\([^)]*\)", " ", body)

    checked = leaked = unresolved = 0
    for name, rows in files.items():
        for r in rows:
            info = idmap.get(r["id"])
            if not info:
                unresolved += 1
                continue
            checked += 1
            if info["answer"].upper() in tokens_of(dynamic_region(r["prompt"])):
                leaked += 1
                if leaked <= 3:
                    fails.append(f"ANSWER LEAK {info['answer']} in {name} "
                                 f"id={r['id']}")
    if leaked:
        fails.append(f"{leaked} leaked prompts total")
    print(f"[{'PASS' if not leaked else 'FAIL'}] 3 answer absent from all "
          f"{checked} resolvable prompts"
          + (f" ({unresolved} ids unresolved)" if unresolved else ""))

    # ---- 4: completions legal -------------------------------------------
    illegal = set()
    for name, rows in files.items():
        for r in rows:
            c = r["completion"]
            if len(c) != 5 or not c.isalpha() or (legal and c not in legal):
                illegal.add(c)
    if illegal:
        fails.append(f"illegal completions: {sorted(illegal)[:10]}")
    print(f"[{'PASS' if not illegal else 'FAIL'}] 4 all completions are legal "
          f"5-letter guesses")

    # ---- 5: prompt is a function of history ------------------------------
    # Reconstructed from the prompt's own History block, so this does not
    # depend on any field we might have stripped.
    by_hist: Dict[str, str] = {}
    prompt_conflicts = 0
    for name, rows in files.items():
        for r in rows:
            body = r["prompt"]
            hist = body.split("History:", 1)[-1].split("Deduced so far:", 1)[0]
            key = " ".join(hist.split())
            prev = by_hist.get(key)
            if prev is not None and prev != r["prompt"]:
                prompt_conflicts += 1
                if prompt_conflicts <= 2:
                    fails.append(f"prompt differs for identical history in {name}")
            by_hist[key] = r["prompt"]
    print(f"[{'PASS' if not prompt_conflicts else 'FAIL'}] 5 identical history "
          f"=> identical prompt ({len(by_hist)} distinct histories)")

    # ---- 6: answer split disjoint ---------------------------------------
    tr = {r["answer"] for r in load_jsonl(os.path.join(P, "eval", "train_answers.jsonl"))}
    va = {r["answer"] for r in load_jsonl(os.path.join(P, "eval", "val_answers.jsonl"))}
    overlap = tr & va
    if overlap:
        fails.append(f"train/val answers overlap: {sorted(overlap)[:5]}")
    print(f"[{'PASS' if not overlap else 'FAIL'}] 6 train ({len(tr)}) and val "
          f"({len(va)}) answer sets disjoint")

    # ---- 7: oversampling pool avoids val --------------------------------
    cp = os.path.join(P, "data", "disagreement", "contrast.jsonl")
    pool_bad = 0
    if os.path.exists(cp):
        contrast = load_jsonl(cp)
        pool_prompts = set()
        for f in ("entropy.jsonl", "tree_soare.jsonl"):
            fp = os.path.join(P, "data", "disagreement", f)
            if os.path.exists(fp):
                pool_prompts |= {r["prompt"] for r in load_jsonl(fp)}
        for c in contrast:
            if c["prompt"] in pool_prompts and c["meta"]["reached_by_val_games"]:
                pool_bad += 1
        if pool_bad:
            fails.append(f"{pool_bad} oversampling-pool states are reached by "
                         f"val games")
    print(f"[{'PASS' if not pool_bad else 'FAIL'}] 7 disagreement oversampling "
          f"pool touches no val game")

    # ---- 8: prompt overlap train vs val (SOFT) ---------------------------
    print()
    for solver in ("entropy", "tree_soare", "tree_salet"):
        tr_rows = files.get(f"data{os.sep}{solver}{os.sep}train.jsonl", [])
        vap_rows = files.get(f"data{os.sep}{solver}{os.sep}val.jsonl", [])
        if not tr_rows or not vap_rows:
            continue
        trp = {r["prompt"] for r in tr_rows}
        tr_pairs = {(r["prompt"], r["completion"]) for r in tr_rows}
        shared = sum(1 for r in vap_rows if r["prompt"] in trp)
        # The sharper number: val records whose prompt AND target both already
        # occur in training, i.e. nothing is being held out at all.
        exact = sum(1 for r in vap_rows
                    if (r["prompt"], r["completion"]) in tr_pairs)
        pct, pct_e = 100 * shared / len(vap_rows), 100 * exact / len(vap_rows)
        warns.append(f"{solver}: {pct:.1f}% of val prompts, {pct_e:.1f}% of "
                     f"val (prompt,target) pairs, also appear in train")
        print(f"[WARN] 8 {solver:<12} prompt shared {shared}/{len(vap_rows)} "
              f"({pct:.1f}%)   prompt+target shared {exact}/{len(vap_rows)} "
              f"({pct_e:.1f}%)")
    print("       Expected under an answer-keyed split: one early-turn state is")
    print("       reached by many answers. It means token-level val accuracy is")
    print("       NOT a held-out metric here -- score by rollout on eval/val_answers.jsonl.")

    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails[:12]:
            print(f"  {f}")
        return 1
    print("ALL HARD CHECKS PASSED")
    if warns and args.strict:
        print(f"--strict: {len(warns)} soft findings treated as failure")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
