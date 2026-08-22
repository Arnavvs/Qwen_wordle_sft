"""audit_legacy_dpo.py — reproducible audit of the Phase-8 v1 preference set.

Everything asserted in `AUDIT.md` is printed by this script from the shipped
file `sft_package/data/dpo_midgame/train.jsonl`. Nothing is taken from the v1
metadata on trust: states are re-derived from their own prompts and labels are
re-scored, both with v1's own metric and with the exact value function in
`dpo_core`.

The headline result is the last section. v1's preference *direction* is not
wrong — under an exact adaptive-decoder tree its 2-10 labels are correct on
99.6% of rows and backwards on none. What is wrong is that its notion of pair
difficulty is unrelated to value: the pairs it calls `competitive` have a median
*true* gap of a full guess, because 75.9% of them are separated by nothing but
the `- 0.5 * is_candidate` constant in its cost function.

    python phase8_dpo_v3/audit_legacy_dpo.py
    python phase8_dpo_v3/audit_legacy_dpo.py --max-tree-states 800   # faster
"""
import _paths  # noqa: F401

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np

from dpo_core import AdaptiveTree, Board, Vocab, load_answer_splits

HISTORY_LINE = re.compile(r"^\s+\d+\.\s+([A-Z]{5})\s+->\s+([BYG]{5})$", re.M)
CANDIDATE_BONUS = 0.5      # v1's tie-break constant


def h1(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", default="sft_package/data/dpo_midgame/train.jsonl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--max-tree-states", type=int, default=0,
                    help="cap states re-judged by the exact tree (0 = all)")
    ap.add_argument("--out", default="phase8_dpo_v3/legacy_audit.json")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.legacy, encoding="utf-8") if l.strip()]
    V = Vocab(a.artifacts)
    tree = AdaptiveTree(V)
    train_answers, val_answers = load_answer_splits(a.sft_dir)
    HOLDOUT, TRAINA = set(val_answers), set(train_answers)
    LEGAL = set(V.words)
    R = {"file": a.legacy, "rows": len(rows)}

    h1(f"PHASE-8 v1 AUDIT — {len(rows)} rows from {a.legacy}")

    # ---------------------------------------------------------------- states
    boards = {}
    t0 = time.perf_counter()
    for p in {r["prompt"] for r in rows}:
        boards[p] = Board.from_history(V, HISTORY_LINE.findall(p))
    print(f"re-derived {len(boards)} distinct states from their prompts "
          f"in {time.perf_counter()-t0:.0f}s")

    # ---------------------------------------------------------------- 1
    h1("1. DUPLICATION")
    pc = Counter(r["prompt"] for r in rows)
    ids = Counter(r["id"] for r in rows)
    R["duplication"] = {
        "rows": len(rows),
        "distinct_states": len(pc),
        "distinct_triples": len({(r["prompt"], r["chosen"], r["rejected"]) for r in rows}),
        "distinct_ids": len(ids),
        "duplicate_id_rows": len(rows) - len(ids),
        "max_rows_one_state": max(pc.values()),
        "rows_in_repeated_states": sum(v for v in pc.values() if v > 1),
    }
    for k, v in R["duplication"].items():
        print(f"  {k:26s} {v}")
    print("\n  the five most repeated states:")
    for p, n in pc.most_common(5):
        b = boards[p]
        print(f"    {n:4d} rows  turn {b.turn}  |adm|={len(b.admissible):4d} "
              f"|cand|={len(b.candidates):4d}  {b.state_key()}")
    print("\n  NOTE: v1's dedup key is (prompt, rejected), and `rejected` is")
    print("  re-randomised on every visit, so a state can mint new rows forever.")

    # ---------------------------------------------------------------- 2
    h1("2. STATE / METADATA CONSISTENCY  (re-derived from the prompt)")
    bad = Counter()
    for r in rows:
        b, m = boards[r["prompt"]], r["meta"]
        bad["n_candidates"] += len(b.candidates) != m["n_candidates"]
        bad["n_admissible"] += len(b.admissible) != m["n_admissible"]
        bad["turn"] += b.turn != m["turn"]
    for k, v in bad.items():
        print(f"  {'[PASS]' if not v else '[FAIL]'} {k:16s} mismatches: {v}")
    R["metadata_consistency"] = dict(bad)

    # ---------------------------------------------------------------- 3
    h1("3. ACTION VALIDITY")
    ill = Counter()
    for r in rows:
        c, j = r["chosen"].lower(), r["rejected"].lower()
        b = boards[r["prompt"]]
        adm = set(b.admissible.tolist())
        ill["chosen_illegal"] += c not in LEGAL
        ill["rejected_illegal"] += j not in LEGAL
        ill["identical"] += c == j
        sp = r["meta"]["action_space"]
        if sp == "admissible":
            ill["chosen_outside_restricted_space"] += V.index[c] not in adm
            ill["rejected_outside_restricted_space"] += V.index[j] not in adm
        else:
            ill["legal_space_rejected_is_inadmissible"] += V.index[j] not in adm
            ill["legal_space_rows"] += 1
    for k in ("chosen_illegal", "rejected_illegal", "identical",
              "chosen_outside_restricted_space", "rejected_outside_restricted_space"):
        print(f"  {'[PASS]' if not ill[k] else '[FAIL]'} {k:36s} {ill[k]}")
    n_lg = ill["legal_space_rows"]
    print(f"\n  In the UNRESTRICTED regime ({n_lg} rows) the `rejected` word")
    print(f"  contradicts feedback already on the board in "
          f"{ill['legal_space_rejected_is_inadmissible']}/{n_lg} rows "
          f"({100*ill['legal_space_rejected_is_inadmissible']/max(n_lg,1):.1f}%).")
    print("  Those are legal actions, so this is not a validity bug — but it is")
    print("  a triviality one: the comparison is against a word the model can")
    print("  already rule out from its own prompt.")
    R["action_validity"] = dict(ill)

    # ---------------------------------------------------------------- 4
    h1("4. PAIR TYPES AND THE GAPS v1 RECORDED")
    pt = Counter(r["meta"]["pair_type"] for r in rows)
    print(f"  pair_type: {dict(pt)}   clear = "
          f"{100*pt['clear']/len(rows):.1f}% of the dataset")
    print(f"  buckets  : {dict(sorted(Counter(r['meta']['bucket'] for r in rows).items()))}")
    print(f"  turns    : {dict(sorted(Counter(r['meta']['turn'] for r in rows).items()))}")
    gp = {}
    for k in ("clear", "competitive"):
        g = np.array([r["meta"]["cost_gap"] for r in rows
                      if r["meta"]["pair_type"] == k])
        gp[k] = {"n": len(g), "median": float(np.median(g)),
                 "p90": float(np.percentile(g, 90)), "max": float(g.max())}
        print(f"  {k:12s} n={len(g):6d}  median={np.median(g):9.4f}  "
              f"p90={np.percentile(g,90):10.4f}  max={g.max():10.4f}")
    allg = np.array([r["meta"]["cost_gap"] for r in rows])
    print(f"\n  overall median gap = {np.median(allg):.4f}")
    print("  PROJECT_README reports 3.07 as the *clear* median; 3.07 is the")
    print(f"  median of ALL rows. The clear-only median is {gp['clear']['median']:.4f}.")
    R["pair_types"] = {"counts": dict(pt), "gaps": gp,
                       "overall_median_gap": float(np.median(allg))}

    # ---------------------------------------------------------------- 5
    h1("5. WHAT IS THE `competitive` GAP ACTUALLY MADE OF?")
    comp = [r for r in rows if r["meta"]["pair_type"] == "competitive"]
    raw, both = [], Counter()
    for r in comp:
        b = boards[r["prompt"]]
        cols = V.answer_col[b.candidates]
        idx = np.array([V.index[r["chosen"].lower()],
                        V.index[r["rejected"].lower()]], dtype=np.int32)
        er = V.bundle.fb.partition_stats(idx, cols)["expected_remaining"]
        raw.append(float(er[1] - er[0]))
        cand = set(b.candidates.tolist())
        both[(int(idx[0]) in cand, int(idx[1]) in cand)] += 1
    raw = np.array(raw)
    # Count first, derive the percentage from the count. Reconstructing a count
    # by multiplying a float fraction back out floors 4072 to 4071.
    n_ident = int(np.count_nonzero(np.abs(raw) < 1e-9))
    n_neg = int(np.count_nonzero(raw < -1e-9))
    ident, negv = n_ident / len(comp), n_neg / len(comp)
    print(f"  competitive rows: {len(comp)}")
    print(f"  (chosen_is_candidate, rejected_is_candidate): {dict(both)}")
    print(f"\n  Rows where chosen and rejected have IDENTICAL E[remaining],")
    print(f"  i.e. the entire recorded gap is the {CANDIDATE_BONUS} candidate bonus:")
    print(f"      {n_ident} / {len(comp)}  ({100*ident:.1f}%)")
    print(f"  Rows where the rejected action is STRICTLY BETTER on E[remaining]")
    print(f"  and is only 'worse' because of that constant:")
    print(f"      {n_neg} / {len(comp)}  ({100*negv:.1f}%)")
    R["competitive_composition"] = {
        "n": len(comp), "n_identical_expected_remaining": n_ident,
        "n_rejected_better_on_information": n_neg,
        "identical_expected_remaining_frac": round(ident, 4),
        "rejected_better_on_information_frac": round(negv, 4),
        "pairs_by_candidacy": {str(k): v for k, v in both.items()}}

    # ---------------------------------------------------------------- 6
    h1("6. PREFERENCE DIRECTION UNDER v1's OWN METRIC")
    back = 0
    for r in rows:
        b = boards[r["prompt"]]
        cols = V.answer_col[b.candidates]
        ws = [r["chosen"].lower(), r["rejected"].lower()]
        idx = np.array([V.index[w] for w in ws], dtype=np.int32)
        er = V.bundle.fb.partition_stats(idx, cols)["expected_remaining"]
        cand = set(b.candidates.tolist())
        cost = er - np.array([CANDIDATE_BONUS if int(i) in cand else 0.0 for i in idx])
        back += not cost[0] < cost[1]
    print(f"  [{'PASS' if not back else 'FAIL'}] rows backwards under v1's own "
          f"cost function: {back} / {len(rows)}")
    print("  v1 has no sign bug. Its labels are self-consistent.")
    R["direction_own_metric"] = {"backwards": back}

    # ---------------------------------------------------------------- 7
    h1("7. PREFERENCE DIRECTION UNDER AN EXACT VALUE FUNCTION  (2-10 regime)")
    sub = [r for r in rows if r["meta"]["bucket"] in ("2-3", "4-10")]
    if a.max_tree_states:
        keep = {p for p in list({r["prompt"] for r in sub})[:a.max_tree_states]}
        sub = [r for r in sub if r["prompt"] in keep]
    print(f"  re-judging {len(sub)} rows over "
          f"{len({r['prompt'] for r in sub})} states with the exact adaptive tree ...")
    res, gaps, by_kind = Counter(), [], defaultdict(list)
    t0 = time.perf_counter()
    for r in sub:
        b = boards[r["prompt"]]
        if not b.restricted:
            res["state_not_restricted"] += 1
            continue
        tot = tree.action_costs(b)
        ci, ji = V.index[r["chosen"].lower()], V.index[r["rejected"].lower()]
        if ci not in tot or ji not in tot:
            res["action_outside_space"] += 1
            continue
        d = (tot[ji] - tot[ci]) / len(b.candidates)
        gaps.append(d); by_kind[r["meta"]["pair_type"]].append(d)
        res["agrees" if d > 0 else ("tie" if d == 0 else "BACKWARDS")] += 1
    g = np.array(gaps)
    n = max(len(g), 1)
    print(f"  done in {time.perf_counter()-t0:.0f}s\n")
    print(f"    chosen truly better : {res['agrees']:6d}  ({100*res['agrees']/n:.1f}%)")
    print(f"    exactly equal value : {res['tie']:6d}  ({100*res['tie']/n:.1f}%)")
    print(f"    BACKWARDS           : {res['BACKWARDS']:6d}  ({100*res['BACKWARDS']/n:.1f}%)")
    print(f"\n  TRUE gap, in mean further guesses:")
    print(f"    all          median={np.median(g):.4f}  p90={np.percentile(g,90):.4f}")
    for k, v in sorted(by_kind.items()):
        v = np.array(v)
        print(f"    {k:12s} median={np.median(v):.4f}  p90={np.percentile(v,90):.4f}  "
              f"frac truly close (<=0.25): {100*np.mean(v<=0.25):.1f}%")
    print("\n  This is the finding that matters. v1's labels point the right way,")
    print("  but the pairs it calls `competitive` are separated by a median of a")
    print("  FULL GUESS of real value. v1's difficulty axis does not measure")
    print("  difficulty, so 'keep only the competitive pairs' would not have")
    print("  selected close decisions either.")
    R["direction_exact_tree"] = {
        "rows_judged": int(n), "agrees": res["agrees"], "ties": res["tie"],
        "backwards": res["BACKWARDS"],
        "true_gap_median": round(float(np.median(g)), 4),
        "true_gap_median_by_type": {k: round(float(np.median(v)), 4)
                                    for k, v in by_kind.items()},
        "frac_truly_close_by_type": {
            k: round(float(np.mean(np.array(v) <= 0.25)), 4)
            for k, v in by_kind.items()}}

    # ---------------------------------------------------------------- 8
    h1("8. LEAKAGE")
    nchosen = sum(1 for r in rows if r["chosen"].lower() in HOLDOUT)
    nrej = sum(1 for r in rows if r["rejected"].lower() in HOLDOUT)
    touch_states = sum(1 for p, b in boards.items()
                       if {V.words[int(i)] for i in b.candidates} & HOLDOUT)
    touch_rows = sum(1 for r in rows
                     if {V.words[int(i)] for i in boards[r["prompt"]].candidates} & HOLDOUT)
    only_val = sum(1 for p, b in boards.items()
                   if {V.words[int(i)] for i in b.candidates} <= HOLDOUT)
    print(f"  rows whose CHOSEN action is a held-out answer   : {nchosen}")
    print(f"  rows whose REJECTED action is a held-out answer : {nrej}")
    print(f"  states whose candidate set contains a held-out answer : "
          f"{touch_states}/{len(boards)} ({100*touch_states/len(boards):.1f}%)")
    print(f"  rows in such states                                  : "
          f"{touch_rows}/{len(rows)} ({100*touch_rows/len(rows):.1f}%)")
    print(f"  states whose candidates are ENTIRELY held-out answers : {only_val}")
    print("\n  v1 is answer-keyed at the SOURCE only: rollouts start from a train")
    print("  answer, but nothing stops a held-out answer from surviving in the")
    print(f"  candidate set, or from being the preferred action ({nchosen} rows).")
    R["leakage"] = {"chosen_is_holdout": nchosen, "rejected_is_holdout": nrej,
                    "states_touching_holdout": touch_states,
                    "rows_touching_holdout": touch_rows,
                    "states_entirely_holdout": only_val}

    # ---------------------------------------------------------------- 9
    h1("9. SPLITS")
    print(f"  files in {os.path.dirname(a.legacy)}: "
          f"{sorted(os.listdir(os.path.dirname(a.legacy)))}")
    print(f"  meta.split values: {dict(Counter(r['meta']['split'] for r in rows))}")
    print("\n  There is no validation file. The DPO notebook loads train.jsonl,")
    print("  shuffles it and trains on all of it, computing `acc` per batch on")
    print("  the batch it is training on. The reported 0.604 -> 0.836 is a")
    print("  TRAINING accuracy; no held-out preference number exists for v1.")
    R["splits"] = {"files": sorted(os.listdir(os.path.dirname(a.legacy))),
                   "has_validation": False,
                   "reported_accuracy_is": "training accuracy, per batch"}

    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(R, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\naudit written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
