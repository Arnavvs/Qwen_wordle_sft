"""
verify_endgame_dataset.py - audit the synthetic endgame data before training.

Every check is an assertion. A synthetic dataset is exactly the place a subtle
leak gets introduced, and a leak here would invalidate Phase 6 the same way it
would have invalidated Phase 4.

  1. no candidate count in any prompt
  2. no candidate list in any prompt
  3. the answer never appears in its own prompt (in any casing)
  4. greens are never concatenated into the answer
  5. no held-out answer was used to generate a state
  6. prompt format is byte-identical in shape to the natural records
  7. every completion is a legal Wordle guess
  8. at k=1 the completion IS the unique surviving answer
  9. the mixed dataset is consistent and its head is not re-duplicated

    python verify_endgame_dataset.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os
import re
import sys
from collections import Counter

HERE = _paths.ROOT  # data lives at the repo root
sys.path.insert(0, HERE)

NATURAL = "sft_package/data/tree_salet/train.jsonl"
ENDGAME = "sft_package/data/tree_salet_endgame/train.jsonl"
VAL_ANSWERS = "sft_package/eval/val_answers.jsonl"


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    nat = load(os.path.join(HERE, NATURAL))
    eg = load(os.path.join(HERE, ENDGAME))
    val = {json.loads(l)["answer"].upper()
           for l in open(os.path.join(HERE, VAL_ANSWERS), encoding="utf-8")}
    legal = set()
    for f in ("data/wordle_answers.txt", "data/wordle_allowed_guesses.txt"):
        legal |= {w.strip().upper()
                  for w in open(os.path.join(HERE, f), encoding="utf-8")
                  if w.strip()}

    print(f"natural {len(nat)}   endgame {len(eg)}   "
          f"held-out answers {len(val)}   legal words {len(legal)}")
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails += 1

    print("\n-- prompt hygiene --")
    banned = ("Possible answers remaining", "candidates remaining",
              "Candidates:", "n_candidates", "Answer:", "answer is")
    hits = [b for b in banned if any(b in r["prompt"] for r in eg)]
    check("no candidate count / list / answer phrasing", not hits, str(hits))

    # The answer must not be recoverable from its own prompt. At k=1 the
    # completion IS the answer, so the prompt is checked against it.
    #
    # Only the DYNAMIC region counts. The static scaffold ("After each guess
    # you receive...", "Guess 3 of 6", "Next guess:") is byte-identical in
    # every prompt and so carries no information about any answer - but it
    # contains the English words AFTER and GUESS, which are themselves valid
    # Wordle answers. Matching against the whole prompt flags those as leaks.
    # It is a false positive, and it is present in the natural Phase 4 data
    # too (AFTER), which was audited clean by verify_no_leakage.py.
    def dynamic(p):
        i = p.find("History:")
        body = p[i:] if i >= 0 else p
        return "\n".join(ln for ln in body.splitlines()
                         if not ln.startswith("Guess ")
                         and not ln.startswith("Next guess")).upper()

    bad = []
    for r in eg:
        if r["meta"]["n_candidates"] == 1:
            w = r["completion"].upper()
            if re.search(r"\b" + re.escape(w) + r"\b", dynamic(r["prompt"])):
                bad.append((w, r["id"]))
    check("k=1 answer never in the dynamic part of its own prompt", not bad,
          f"{len(bad)} violations: {bad[:3]}" if bad else "")

    # Greens render position-by-position; five concatenated greens would BE it.
    concat = [r for r in eg if re.search(r"Confirmed letters : [A-Za-z]{5}\b",
                                         r["prompt"])]
    check("greens never concatenated", not concat, f"{len(concat)} hits")

    print("\n-- split integrity --")
    # A held-out answer can legitimately appear as a probe WORD. What must not
    # happen is a state GENERATED FROM a held-out answer, which at k=1 would
    # make the completion that answer.
    k1_words = {r["completion"].upper() for r in eg
                if r["meta"]["n_candidates"] == 1}
    leaked = sorted(k1_words & val)
    check("no held-out answer is a k=1 target", not leaked,
          f"{len(leaked)} leaked: {leaked[:5]}")
    check("every endgame record is split=train",
          all(r["meta"]["split"] == "train" for r in eg))

    print("\n-- format parity with the natural records --")
    def shape(p):
        return [ln.split(":")[0] for ln in p.splitlines() if ":" in ln][:6]
    check("prompt header shape matches", shape(nat[0]["prompt"]) == shape(eg[0]["prompt"]),
          f"{shape(eg[0]['prompt'])}")
    check("prompts end with 'Next guess:'",
          all(r["prompt"].rstrip().endswith("Next guess:") for r in eg))
    check("all records have prompt+completion+meta",
          all({"prompt", "completion", "meta"} <= set(r) for r in eg))

    print("\n-- label validity --")
    illegal = [r["completion"] for r in eg if r["completion"].upper() not in legal]
    check("every completion is a legal guess", not illegal,
          f"{len(illegal)} illegal")
    check("every completion is 5 letters",
          all(re.fullmatch(r"[A-Z]{5}", r["completion"].upper()) for r in eg))
    k1 = [r for r in eg if r["meta"]["n_candidates"] == 1]
    check("at k=1 the expert names a possible answer",
          all(r["meta"]["action_is_possible_answer"] for r in k1),
          f"{sum(1 for r in k1 if not r['meta']['action_is_possible_answer'])} bad")
    check("at k=1 the expert plays the surviving candidate",
          all(r["meta"]["action_is_candidate"] for r in k1),
          f"{sum(1 for r in k1 if not r['meta']['action_is_candidate'])} bad")

    print("\n-- coverage: the point of the exercise --")
    def paths(rows):
        c = Counter(r["completion"] for r in rows
                    if r["meta"]["n_candidates"] == 1)
        return c
    pn, pe = paths(nat), paths(eg)
    nk1 = sum(pn.values())
    ek1 = sum(pe.values())
    print(f"  natural k=1: {nk1} rows / {len(pn)} words "
          f"= {nk1/max(len(pn),1):.2f} paths per word")
    print(f"  endgame k=1: {ek1} rows / {len(pe)} words "
          f"= {ek1/max(len(pe),1):.2f} paths per word")
    merged = pn + pe
    print(f"  MIXED   k=1: {sum(merged.values())} rows / {len(merged)} words "
          f"= {sum(merged.values())/max(len(merged),1):.2f} paths per word")
    check("mixed data gives >1 path per word on average",
          sum(merged.values()) / max(len(merged), 1) > 1.5)

    print("\n-- head duplication (must not get worse) --")
    for lbl, rows in (("natural", nat), ("mixed", nat + eg)):
        t1 = sum(1 for r in rows if r["meta"]["turn"] == 1)
        print(f"  {lbl:<8} turn-1 rows {t1:6d} / {len(rows):6d} "
              f"= {100*t1/len(rows):5.1f}%")
    mixed = nat + eg
    t1frac = sum(1 for r in mixed if r["meta"]["turn"] == 1) / len(mixed)
    check("turn-1 share of the mix is below the natural 29%", t1frac < 0.29,
          f"{100*t1frac:.1f}%")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{fails} CHECK(S) FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
