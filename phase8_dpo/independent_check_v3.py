"""
independent_check_v3.py - a second opinion on sft_package/data/dpo_v3/.

The v3 dataset ships with its own verifier and its own audit. Both pass. This
file exists because a generator checking its own output is not independent
evidence, and v1 shipped with a passing verifier too.

Every check here is written against the raw rows, re-deriving state from the
prompt text, and deliberately overlaps the v3 verifier as little as possible.
It re-runs the four checks that caught real bugs in v1, plus the ones v3's own
audit says v1 lacked:

  1. prompt hygiene and state re-derivation
  2. train / validation disjointness, on answers AND on states
  3. duplicate ids and state over-representation
  4. action space matches the deployed decoder
  5. no pair between two identical actions
  6. the consistency shortcut - can a pair be won without any value reasoning?
  7. held-out answers as the preferred action

    python phase8_dpo/independent_check_v3.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os
import re
from collections import Counter

import numpy as np

from wordle_solver import load_artifacts, feedback_code

TRAIN = "sft_package/data/dpo_v3/train.jsonl"
VALID = "sft_package/data/dpo_v3/validation.jsonl"
VAL_ANSWERS = "sft_package/eval/val_answers.jsonl"
FILTER_THRESHOLD = 20

HIST = re.compile(r"^\s*(\d+)\.\s+([A-Za-z]{5})\s*->\s*([GYB]{5})\s*$")
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def parse_history(prompt):
    out, inside = [], False
    for ln in prompt.splitlines():
        if ln.startswith("History:"):
            inside = True
            continue
        if inside:
            m = HIST.match(ln)
            if m:
                out.append((m.group(2).lower(), m.group(3)))
            elif ln.startswith("Deduced"):
                break
    return tuple(out)


def code_of(pat):
    return sum({"B": 0, "Y": 1, "G": 2}[c] * (3 ** j) for j, c in enumerate(pat))


def main():
    tr, va = load(TRAIN), load(VALID)
    val_answers = {json.loads(l)["answer"].upper()
                   for l in open(VAL_ANSWERS, encoding="utf-8")}
    bundle = load_artifacts("artifacts", mmap=True)
    V = bundle.vocab
    legal = set(w.upper() for w in V.guesses)
    print(f"train {len(tr)}   validation {len(va)}   held-out answers {len(val_answers)}\n")

    # prefix cache: histories share prefixes, so admissibility is derived once
    _pref = {(): [w.lower() for w in V.guesses]}

    def admissible(h):
        if h in _pref:
            return _pref[h]
        base = admissible(h[:-1])
        g, pat = h[-1]
        c = code_of(pat)
        _pref[h] = [w for w in base if feedback_code(g, w) == c]
        return _pref[h]

    print("1. PROMPT HYGIENE AND STATE RE-DERIVATION")
    allrows = tr + va
    banned = ["Possible answers remaining", "candidates remaining", "Candidates:",
              "admissible", "cost", "expected_remaining", "Answer:", "value"]
    hit = [b for b in banned if any(b in r["prompt"] for r in allrows)]
    check("no candidate / admissible / score wording in prompts", not hit, str(hit))
    check("prompts end with 'Next guess:'",
          all(r["prompt"].rstrip().endswith("Next guess:") for r in allrows))
    check("only prompt/chosen/rejected/id/meta at top level",
          all(set(r) <= {"id", "prompt", "chosen", "rejected", "meta"} for r in allrows))
    bad = sum(1 for r in allrows if not parse_history(r["prompt"]))
    check("every prompt has a parseable history", bad == 0, f"{bad} unparseable")

    print("\n2. TRAIN / VALIDATION DISJOINTNESS")
    tr_p = set(r["prompt"] for r in tr)
    va_p = set(r["prompt"] for r in va)
    check("no shared state between train and validation",
          not (tr_p & va_p), f"{len(tr_p & va_p)} shared")
    check("no shared id", not ({r["id"] for r in tr} & {r["id"] for r in va}))

    print("\n3. DUPLICATION (v1 had 1,867 duplicate ids and 177 rows on one state)")
    for label, rows in (("train", tr), ("validation", va)):
        ids = Counter(r["id"] for r in rows)
        dup = sum(v - 1 for v in ids.values() if v > 1)
        pr = Counter(r["prompt"] for r in rows)
        check(f"{label}: no duplicate ids", dup == 0, f"{dup} duplicates")
        check(f"{label}: at most one row per state", max(pr.values()) == 1,
              f"max {max(pr.values())} rows from one state")

    print("\n4. ACTION SPACE MATCHES THE DEPLOYED DECODER")
    restricted = [(r, parse_history(r["prompt"])) for r in allrows
                  if len(admissible(parse_history(r["prompt"]))) <= FILTER_THRESHOLD]
    viol = 0
    for r, h in sorted(restricted, key=lambda x: len(x[1])):
        A = set(w.upper() for w in admissible(h))
        if r["chosen"] not in A or r["rejected"] not in A:
            viol += 1
    check(f"|admissible| <= {FILTER_THRESHOLD}: both words admissible "
          f"(all {len(restricted)} restricted rows)", viol == 0, f"{viol} violations")

    print("\n5. PAIR VALIDITY")
    check("chosen != rejected", all(r["chosen"] != r["rejected"] for r in allrows))
    bad = [w for r in allrows for w in (r["chosen"], r["rejected"]) if w not in legal]
    check("both words legal", not bad, f"{len(bad)} illegal")
    check("both words 5 uppercase letters",
          all(re.fullmatch(r"[A-Z]{5}", r[k]) for r in allrows
              for k in ("chosen", "rejected")))

    print("\n6. CONSISTENCY SHORTCUT  (v1 leaked +44.5 pts here)")
    # If `chosen` is far more often feedback-consistent than `rejected`, a model
    # can win pairs by checking consistency alone and never learn any value.
    # Only meaningful where the decoder does NOT already restrict the action
    # space, i.e. the unrestricted regime.
    unres = [(r, parse_history(r["prompt"])) for r in allrows
             if len(admissible(parse_history(r["prompt"]))) > FILTER_THRESHOLD]
    if unres:
        ch = rj = 0
        for r, h in sorted(unres, key=lambda x: len(x[1])):
            A = set(w.upper() for w in admissible(h))
            ch += r["chosen"] in A
            rj += r["rejected"] in A
        n = len(unres)
        gap = 100 * (ch - rj) / n
        print(f"      unrestricted rows {n}: chosen consistent {100*ch/n:.1f}%  "
              f"rejected consistent {100*rj/n:.1f}%")
        check("consistency asymmetry under 10 pts", abs(gap) < 10,
              f"{gap:+.1f} pts")
    else:
        print("      no unrestricted rows")

    print("\n7. HELD-OUT ANSWERS AS THE PREFERRED ACTION  (v1 had 694)")
    for label, rows in (("train", tr), ("validation", va)):
        n = sum(1 for r in rows if r["chosen"] in val_answers)
        if label == "train":
            check("train: no held-out answer is the chosen action", n == 0, f"{n} rows")
        else:
            print(f"      validation: {n} rows name a held-out answer "
                  f"(expected — validation is built from them)")

    print("\n8. SHAPE")
    for label, rows in (("train", tr), ("validation", va)):
        b = Counter(r["meta"].get("bucket") for r in rows)
        t = Counter(r["meta"].get("pair_type") for r in rows)
        print(f"      {label:<11} buckets {dict(sorted(b.items()))}")
        print(f"      {'':<11} types   {dict(sorted(t.items()))}")

    print("\n" + ("=" * 60))
    print("INDEPENDENT VERDICT: " + ("PASS" if not FAILS else f"FAIL {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
