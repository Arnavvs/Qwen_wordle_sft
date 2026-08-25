"""verify_grpo_tasks.py — independent audit of the GRPO task set.

Same contract as `phase8_dpo_v3/verify_dpo_v3_dataset.py`: nothing is taken
from the task metadata on trust. Every state is re-derived from its own
rendered prompt and every value is recomputed from scratch, so a metadata field
and the prompt it describes can never silently disagree.

  A  structure, ids, duplicate states
  B  prompt hygiene: no answer, no candidate set or count, no values
  C  state re-derivation from the prompt alone
  D  action validity: every action legal, distinct, and inside the action space
     the deployed decoder would actually offer
  E  values recomputed independently, per regime
  F  no task where every action is equal (no gradient), and values sorted
  G  train/validation separation: answers, states, prompts
  H  answer leakage: no training task lists a held-out answer as an action
  I  distribution

Exits non-zero on any hard failure, so it can gate a training run.

    python phase11_grpo/verify_grpo_tasks.py
"""
import _paths  # noqa: F401

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from dpo_core import (ADAPTIVE_THRESHOLD, AdaptiveTree, Board, Vocab,
                      bucket_of, load_answer_splits, sha)

HISTORY_LINE = re.compile(r"^\s+(\d+)\.\s+([A-Z]{5})\s+->\s+([BYG]{5})$", re.M)
FAILS, WARNS = [], []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)


def parse_history(prompt):
    found = HISTORY_LINE.findall(prompt)
    if [int(i) for i, _, _ in found] != list(range(1, len(found) + 1)):
        raise ValueError("history lines are not numbered 1..n")
    return [(g, p) for _, g, p in found]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_package/data/grpo_tasks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    print("=" * 72)
    print("GRPO TASK SET VERIFICATION")
    print("=" * 72)

    V = Vocab()
    tree = AdaptiveTree(V)
    train_answers, val_answers = load_answer_splits()
    HOLDOUT = set(val_answers)
    LEGAL = set(V.words)
    all_idx = np.arange(V.n, dtype=np.int32)

    splits = {}
    for name in ("train", "validation"):
        p = os.path.join(a.data, f"{name}.jsonl")
        splits[name] = ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                        if os.path.exists(p) else [])
    report = {"counts": {k: len(v) for k, v in splits.items()}}

    # ------------------------------------------------------------------ A
    print("\nA. STRUCTURE")
    for name, rows in splits.items():
        check(bool(rows), f"{name} present and non-empty", f"{len(rows)} tasks")
        if not rows:
            continue
        keys = [r["meta"]["state_key"] for r in rows]
        check(len(set(keys)) == len(keys), f"{name}: one task per state",
              f"max {max(Counter(keys).values())}")
        check(len({r["id"] for r in rows}) == len(rows), f"{name}: ids unique")
        check(len({r["prompt"] for r in rows}) == len(rows),
              f"{name}: prompts unique")

    # ------------------------------------------------------------------ B
    print("\nB. PROMPT HYGIENE (every task)")
    banned = {
        "candidate count": re.compile(r"Possible answers remaining", re.I),
        "candidate list": re.compile(r"^Candidates:", re.M),
        "admissible count": re.compile(r"admissible", re.I),
        "values/rewards": re.compile(r"\b(value|reward|expected[_ ]remaining|advantage)\b", re.I),
    }
    hits = defaultdict(int)
    green = 0
    for rows in splits.values():
        for r in rows:
            for lab, rx in banned.items():
                if rx.search(r["prompt"]):
                    hits[lab] += 1
            if any(p == "GGGGG" for _, p in parse_history(r["prompt"])):
                green += 1
    for lab in banned:
        check(hits[lab] == 0, f"prompt never contains: {lab}",
              f"{hits[lab]} tasks" if hits[lab] else "")
    check(green == 0, "no prompt history contains a solved (GGGGG) line")

    # -------------------------------------------------------------- C/D/E/F
    print("\nC-F. RE-DERIVE EVERY TASK FROM ITS OWN PROMPT")
    s = Counter()
    fams = Counter()
    for name, rows in splits.items():
        for r in rows:
            m = r["meta"]
            board = Board.from_history(V, parse_history(r["prompt"]))
            acts = [w.lower() for w in r["actions"]]
            vals = r["values"]

            # C
            s["turn"] += board.turn != m["turn"]
            s["n_cand"] += len(board.candidates) != m["n_candidates"]
            s["n_adm"] += len(board.admissible) != m["n_admissible"]
            s["bucket"] += bucket_of(len(board.admissible)) != m["bucket"]
            s["action_space"] += board.action_space != m["action_space"]
            s["state_key"] += board.state_key() != m["state_key"]
            s["state_hash"] += sha("state", m["state_key"])[:20] != m["state_hash"]

            # D
            s["illegal_action"] += sum(1 for w in acts if w not in LEGAL)
            s["dup_action"] += len(set(acts)) != len(acts)
            s["len_mismatch"] += len(acts) != len(vals)
            s["n_actions"] += len(acts) != m["n_actions"]
            s["not_upper"] += any(w.upper() != aw for w, aw in zip(acts, r["actions"]))
            space = set(board.actions().tolist())
            s["outside_action_space"] += sum(
                1 for w in acts if V.index[w] not in space)

            # E — recompute, per regime
            if board.restricted:
                fams["exact"] += 1
                s["family_wrong"] += m["value_family"] != "exact_adaptive_decoder_tree"
                s["exact_flag_wrong"] += m["exact"] is not True
                tot = tree.action_costs(board)
                n = len(board.candidates)
                for w, v in zip(acts, vals):
                    if abs(tot[V.index[w]] / n - v) > 1e-6:
                        s["VALUE_WRONG"] += 1
                # restricted tasks must expose the WHOLE admissible set
                s["restricted_not_full_space"] += len(acts) != len(board.admissible)
            else:
                fams["one_ply"] += 1
                s["exact_flag_wrong"] += m["exact"] is not False
                S = np.sort(V.answer_col[board.candidates].astype(np.int32))
                er = V.bundle.fb.partition_stats(all_idx, S)["expected_remaining"]
                sq = np.rint(er * len(S)).astype(np.int64)
                for w, v in zip(acts, vals):
                    if abs(float(sq[V.index[w]]) / len(S) - v) > 1e-6:
                        s["VALUE_WRONG"] += 1

            # F
            s["not_sorted"] += any(x > y + 1e-12 for x, y in zip(vals, vals[1:]))
            s["ALL_EQUAL"] += max(vals) - min(vals) < 1e-12
            s["best_value_wrong"] += abs(min(vals) - m["best_value"]) > 1e-6
            n_opt = sum(1 for v in vals if abs(v - min(vals)) < 1e-12)
            s["n_optimal_wrong"] += n_opt != m["n_optimal_actions"]

    hard = ["turn", "n_cand", "n_adm", "bucket", "action_space", "state_key",
            "state_hash", "illegal_action", "dup_action", "len_mismatch",
            "n_actions", "not_upper", "outside_action_space", "family_wrong",
            "exact_flag_wrong", "VALUE_WRONG", "restricted_not_full_space",
            "not_sorted", "ALL_EQUAL", "best_value_wrong", "n_optimal_wrong"]
    labels = {
        "VALUE_WRONG": "every value recomputed independently and matches",
        "ALL_EQUAL": "no task where all actions are equal (no gradient)",
        "outside_action_space": "every action is inside the decoder's action space",
        "restricted_not_full_space": "restricted tasks expose the full admissible set",
        "not_sorted": "values are sorted ascending (best first)",
    }
    for k in hard:
        check(s[k] == 0, labels.get(k, k), f"{s[k]} bad" if s[k] else "")
    print(f"       ({fams['exact']} exact-tree tasks, {fams['one_ply']} one-ply)")
    report["recompute"] = {k: int(s[k]) for k in hard}

    # ------------------------------------------------------------------ G
    print("\nG. TRAIN / VALIDATION SEPARATION")
    tk = {r["meta"]["state_key"] for r in splits["train"]}
    vk = {r["meta"]["state_key"] for r in splits["validation"]}
    check(not tk & vk, "no shared state", f"{len(tk & vk)} shared")
    check(not ({r["prompt"] for r in splits["train"]}
               & {r["prompt"] for r in splits["validation"]}), "no shared prompt")
    th = {r["meta"]["source_answer_hash"] for r in splits["train"]}
    vh = {r["meta"]["source_answer_hash"] for r in splits["validation"]}
    check(not th & vh, "no shared source answer")
    check(th <= {sha("phase11-source", w)[:16] for w in train_answers},
          "every train task came from a TRAIN answer")
    check(vh <= {sha("phase11-source", w)[:16] for w in val_answers},
          "every validation task came from a HELD-OUT answer")

    # ------------------------------------------------------------------ H
    print("\nH. ANSWER LEAKAGE")
    bad = sum(1 for r in splits["train"]
              for w in r["actions"] if w.lower() in HOLDOUT)
    check(bad == 0, "no TRAIN task lists a held-out answer as an action",
          f"{bad} actions")

    # ------------------------------------------------------------------ I
    print("\nI. DISTRIBUTION")
    for name, rows in splits.items():
        if not rows:
            continue
        na = np.array([r["meta"]["n_actions"] for r in rows])
        sp = np.array([r["meta"]["value_spread"] for r in rows])
        print(f"  {name}: {len(rows)} tasks")
        print(f"    buckets   : {dict(sorted(Counter(r['meta']['bucket'] for r in rows).items()))}")
        print(f"    exact     : {sum(1 for r in rows if r['meta']['exact'])}/{len(rows)}")
        print(f"    actions   : median {int(np.median(na))}  mean {na.mean():.1f}")
        print(f"    tied optimum in {100*np.mean([r['meta']['n_optimal_actions']>1 for r in rows]):.1f}% of tasks")
        print(f"    value spread: median {np.median(sp):.4f}  p90 {np.percentile(sp,90):.4f}")

    print("\n" + "=" * 72)
    print("VERDICT: " + ("ALL HARD CHECKS PASSED" if not FAILS
                         else f"{len(FAILS)} HARD CHECK(S) FAILED"))
    for f in FAILS:
        print(f"  - {f}")
    print("=" * 72)
    report["verdict"] = {"passed": not FAILS, "failures": FAILS}
    out = a.out or os.path.join(a.data, "audit.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"audit written to {out}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
