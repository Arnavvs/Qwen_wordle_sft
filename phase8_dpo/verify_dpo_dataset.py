"""
verify_dpo_dataset.py - audit the preference pairs before training on them.

Preference data has a failure mode SFT data does not: a pair can be *backwards*.
If `rejected` is actually the better action, DPO will confidently learn the
wrong thing, and no loss curve will reveal it. So the central check here
re-derives both costs from the state itself and asserts the ordering.

  A. every state re-derives from its own prompt (as in Phase 6)
  B. cost(chosen) < cost(rejected), recomputed independently of generation
  C. chosen != rejected, both legal, both 5 letters
  D. the action space matches the deployed decoder: when |admissible| <= 20
     BOTH words must be admissible, else the pair trains a distinction the
     decoder already enforces
  E. no candidate list / count / admissible count / answer / score in prompts
  F. no held-out answer used to generate a state
  G. bucket / margin / teacher distributions

    python verify_dpo_dataset.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os
import re
from collections import Counter

import numpy as np

from wordle_solver import load_artifacts, feedback_code

DATA = "sft_package/data/dpo_midgame/train.jsonl"
VAL = "sft_package/eval/val_answers.jsonl"
FILTER_THRESHOLD = 20
CANDIDATE_BONUS = 0.5
HIST = re.compile(r"^\s*(\d+)\.\s+([A-Za-z]{5})\s*->\s*([GYB]{5})\s*$")

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


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
    return out


def code_of(pat):
    return sum({"B": 0, "Y": 1, "G": 2}[c] * (3 ** j) for j, c in enumerate(pat))


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    bundle = load_artifacts("artifacts", mmap=True)
    V = bundle.vocab
    gi = {w: i for i, w in enumerate(V.guesses)}
    legal = set(w.upper() for w in V.guesses)
    val = {json.loads(l)["answer"].upper() for l in open(VAL, encoding="utf-8")}
    print(f"{len(rows)} preference pairs\n")

    print("A. STATE RE-DERIVES FROM ITS OWN PROMPT")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(2500, len(rows)), replace=False)
    bad_c, bad_a, unparsed = 0, 0, 0
    backwards, checked = [], 0
    print("B. COST ORDERING RECOMPUTED INDEPENDENTLY")
    for i in idx:
        r = rows[int(i)]
        h = parse_history(r["prompt"])
        if not h:
            unparsed += 1
            continue
        cands = np.arange(V.n_answers, dtype=np.int32)
        adm = [w.lower() for w in V.guesses]
        for g, pat in h:
            c = code_of(pat)
            cands = bundle.fb.filter_indices(cands, g, c)
            adm = [w for w in adm if feedback_code(g, w) == c]
        if int(len(cands)) != r["meta"]["n_candidates"]:
            bad_c += 1
        if len(adm) != r["meta"]["n_admissible"]:
            bad_a += 1
        # recompute the objective from scratch
        pair = [r["chosen"].lower(), r["rejected"].lower()]
        st = bundle.fb.partition_stats(
            np.array([gi[w] for w in pair], dtype=np.int32), cands)
        cw = {V.answers[j] for j in cands}
        cost = st["expected_remaining"] - np.array(
            [CANDIDATE_BONUS if w in cw else 0.0 for w in pair])
        checked += 1
        if not cost[0] < cost[1]:
            backwards.append((r["id"], r["chosen"], float(cost[0]),
                              r["rejected"], float(cost[1])))
    check(f"history parses ({len(idx)} sampled)", unparsed == 0, f"{unparsed} bad")
    check("re-derived |candidates| matches meta", bad_c == 0, f"{bad_c} bad")
    check("re-derived |admissible| matches meta", bad_a == 0, f"{bad_a} bad")
    check(f"cost(chosen) < cost(rejected) on all {checked} checked",
          not backwards, f"{len(backwards)} BACKWARDS {backwards[:2]}")

    print("\nC. PAIR VALIDITY")
    check("chosen != rejected", all(r["chosen"] != r["rejected"] for r in rows))
    bad = [r["chosen"] for r in rows if r["chosen"] not in legal] + \
          [r["rejected"] for r in rows if r["rejected"] not in legal]
    check("both words are legal guesses", not bad, f"{len(bad)} illegal")
    check("both words are 5 letters",
          all(re.fullmatch(r"[A-Z]{5}", r[k]) for r in rows
              for k in ("chosen", "rejected")))

    print(chr(10) + "D. ACTION SPACE MATCHES THE DEPLOYED DECODER")
    # EVERY restricted row, not a sample: a sampled count once disguised a ~19%
    # violation rate as "206 violations". A prefix cache makes exhaustive
    # checking affordable, since histories share prefixes.
    _pref = {(): [w.lower() for w in V.guesses]}

    def admissible_for(h):
        key = tuple(h)
        if key in _pref:
            return _pref[key]
        base = admissible_for(h[:-1])
        g, pat = h[-1]
        c = code_of(pat)
        _pref[key] = [w for w in base if feedback_code(g, w) == c]
        return _pref[key]

    restricted = [(r, parse_history(r["prompt"])) for r in rows
                  if r["meta"]["n_admissible"] <= FILTER_THRESHOLD]
    viol = bad_chosen = bad_rej = 0
    for r, h in sorted(restricted, key=lambda x: len(x[1])):
        A = set(w.upper() for w in admissible_for(h))
        bc = r["chosen"] not in A
        br = r["rejected"] not in A
        bad_chosen += bc; bad_rej += br; viol += (bc or br)
    check(f"|admissible| <= {FILTER_THRESHOLD}: BOTH words admissible "
          f"(all {len(restricted)} restricted rows)", viol == 0,
          f"{viol} violations (chosen {bad_chosen}, rejected {bad_rej})")
    sp = Counter(r["meta"]["action_space"] for r in rows)
    print(f"      action_space: {dict(sp)}")

    print("\nE. PROMPT HYGIENE")
    banned = ["Possible answers remaining", "candidates remaining", "Candidates:",
              "admissible", "cost", "expected_remaining", "Answer:"]
    hit = [b for b in banned if any(b in r["prompt"] for r in rows)]
    check("no candidate/admissible/score wording in prompts", not hit, str(hit))
    check("prompts end with 'Next guess:'",
          all(r["prompt"].rstrip().endswith("Next guess:") for r in rows))
    check("no meta field names appear in any prompt",
          not any(k in r["prompt"] for r in rows[:500]
                  for k in ("n_admissible", "n_candidates", "cost_gap")))

    print("\nF. SPLIT INTEGRITY")
    # a held-out answer may appear as a probe word; what must not happen is a
    # state generated FROM one, which shows up as a 1-candidate chosen action
    k1 = {r["chosen"] for r in rows if r["meta"]["n_candidates"] == 1}
    leak = sorted(k1 & val)
    check("no held-out answer is a 1-candidate target", not leak,
          f"{len(leak)}: {leak[:5]}")
    check("all rows split=train", all(r["meta"]["split"] == "train" for r in rows))

    print("\nG. DISTRIBUTIONS")
    for f in ("bucket", "pair_type", "teacher", "action_space"):
        print(f"      {f:<12} {dict(sorted(Counter(r['meta'][f] for r in rows).items()))}")
    print(f"      turns        {dict(sorted(Counter(r['meta']['turn'] for r in rows).items()))}")
    g = np.array([r["meta"]["cost_gap"] for r in rows])
    print(f"      cost gap     median {np.median(g):.2f}  p10 {np.percentile(g,10):.2f}"
          f"  p90 {np.percentile(g,90):.2f}")
    comp = np.array([r["meta"]["cost_gap"] for r in rows
                     if r["meta"]["pair_type"] == "competitive"])
    if len(comp):
        print(f"      competitive  median {np.median(comp):.2f}  "
              f"(these are the hard, informative pairs)")
    print(f"      distinct prompts {len(set(r['prompt'] for r in rows))}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
