"""verify_dpo_v3_dataset.py — independent audit of a v3 preference set.

The point of this script is that it does not trust the generator. It re-derives
every claim from the *rendered prompt* — the only thing the model actually sees
— using the shared machinery in `dpo_core`, and compares the result against the
metadata. If a metadata field and the prompt it describes ever disagree, that is
a hard failure.

Checks, in order:

  A  file structure, ids, duplicate rows and duplicate states
  B  prompt hygiene: no answer, no candidate set or count, no scores
  C  state re-derivation: |candidates|, |admissible|, turn and bucket all
     recomputed from the prompt text alone
  D  action validity: both words legal, distinct, and inside the action space
     the deployed decoder would actually offer on that board
  E  label direction and magnitude, recomputed from scratch with an independent
     value function (and, in the exact regime, cross-checked against a
     no-memo no-pruning brute force on a sample)
  F  no pair between two exactly-optimal actions
  G  train/validation separation: answers, states, prompts
  H  answer leakage: held-out answers as actions, and in candidate sets
  I  distribution: buckets, pair types, gaps per score family, duplication
  J  selection-bias diagnostic for the held-out-action filter

Exit status is non-zero if any hard check fails, so it can gate a training run.

    python phase8_dpo_v3/verify_dpo_v3_dataset.py
    python phase8_dpo_v3/verify_dpo_v3_dataset.py --data sft_package/data/dpo_v3
"""
import _paths  # noqa: F401

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from dpo_core import (ADAPTIVE_THRESHOLD, AdaptiveTree, Board, OnePlyScorer,
                      Vocab, bucket_of, load_answer_splits, sha)
from wordle_solver import ALL_GREEN

HISTORY_LINE = re.compile(r"^\s+(\d+)\.\s+([A-Z]{5})\s+->\s+([BYG]{5})$", re.M)

FAILS = []
WARNS = []


def check(ok, label, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(label)
    return ok


def warn(ok, label, detail=""):
    if not ok:
        print(f"  [WARN] {label}  {detail}")
        WARNS.append(label)
    else:
        print(f"  [PASS] {label}{('  ' + detail) if detail else ''}")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def parse_history(prompt):
    """The history exactly as the model reads it, with ordering verified."""
    found = HISTORY_LINE.findall(prompt)
    if [int(i) for i, _, _ in found] != list(range(1, len(found) + 1)):
        raise ValueError("history lines are not numbered 1..n in order")
    return [(g, p) for _, g, p in found]


# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_package/data/dpo_v3")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--brute-force-sample", type=int, default=150,
                    help="states to re-solve with a reference implementation")
    ap.add_argument("--out", default=None, help="write the audit JSON here")
    a = ap.parse_args(argv)

    print("=" * 72)
    print("PHASE-8 v3 DPO DATASET VERIFICATION")
    print("=" * 72)

    V = Vocab(a.artifacts)
    tree = AdaptiveTree(V)
    oneply = OnePlyScorer(V)
    train_answers, val_answers = load_answer_splits(a.sft_dir)
    HOLDOUT = set(val_answers)
    LEGAL = set(V.words)

    splits = {}
    for name in ("train", "validation"):
        p = os.path.join(a.data, f"{name}.jsonl")
        splits[name] = load(p) if os.path.exists(p) else []
    manifest_path = os.path.join(a.data, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}

    report = {"data": a.data, "counts": {k: len(v) for k, v in splits.items()}}

    # ---------------------------------------------------------------- A
    print("\nA. STRUCTURE")
    check(bool(splits["train"]), "train.jsonl present and non-empty",
          f"{len(splits['train'])} rows")
    check(bool(splits["validation"]),
          "validation.jsonl present and non-empty "
          "(v1 shipped no validation split at all)",
          f"{len(splits['validation'])} rows")
    for name, rows in splits.items():
        if not rows:
            continue
        ids = [r["id"] for r in rows]
        keys = [r["meta"]["state_key"] for r in rows]
        pcs = Counter(r["prompt"] for r in rows)
        check(len(set(ids)) == len(ids), f"{name}: ids unique",
              f"{len(ids) - len(set(ids))} duplicates")
        check(len(set(keys)) == len(keys), f"{name}: one pair per state",
              f"max rows for one state = {max(Counter(keys).values())}")
        check(max(pcs.values()) == 1, f"{name}: every prompt appears once",
              f"max = {max(pcs.values())}")

    # ---------------------------------------------------------------- B
    print("\nB. PROMPT HYGIENE (every row)")
    banned = {
        "candidate count": re.compile(r"Possible answers remaining", re.I),
        "candidate list": re.compile(r"^Candidates:", re.M),
        "admissible count": re.compile(r"admissible", re.I),
        "expert/preference score": re.compile(
            r"\b(cost[_ ]gap|preference_gap|chosen_value|expected[_ ]remaining|"
            r"entropy\b|score[_ ]family)", re.I),
        "answer field": re.compile(r"\banswer\s*[:=]", re.I),
    }
    hits = defaultdict(int)
    for rows in splits.values():
        for r in rows:
            for label, rx in banned.items():
                if rx.search(r["prompt"]):
                    hits[label] += 1
    for label in banned:
        check(hits[label] == 0, f"prompt never contains: {label}",
              f"{hits[label]} rows" if hits[label] else "")

    # The strongest form of the answer check: a prompt's history can never
    # contain a winning guess, because a correct guess ends the game.
    bad_green = 0
    for rows in splits.values():
        for r in rows:
            if any(pat == "GGGGG" for _, pat in parse_history(r["prompt"])):
                bad_green += 1
    check(bad_green == 0, "no prompt history contains a solved (GGGGG) line",
          f"{bad_green} rows")

    # ---------------------------------------------------------------- C/D/E/F
    print("\nC-F. RE-DERIVE EVERY ROW FROM ITS OWN PROMPT")
    stats = Counter()
    per_family = defaultdict(lambda: defaultdict(list))
    boards = {}
    for name, rows in splits.items():
        for r in rows:
            m = r["meta"]
            hist = parse_history(r["prompt"])
            board = Board.from_history(V, hist)
            boards[(name, r["id"])] = board

            # -- C: state matches metadata ---------------------------------
            stats["turn_mismatch"] += board.turn != m["turn"]
            stats["n_cand_mismatch"] += len(board.candidates) != m["n_candidates"]
            stats["n_adm_mismatch"] += len(board.admissible) != m["n_admissible"]
            stats["bucket_mismatch"] += bucket_of(len(board.admissible)) != m["bucket"]
            stats["action_space_mismatch"] += board.action_space != m["action_space"]
            stats["state_key_mismatch"] += board.state_key() != m["state_key"]
            stats["state_hash_mismatch"] += sha("state", m["state_key"])[:20] != m["state_hash"]
            stats["guesses_left_mismatch"] += board.guesses_left != m["guesses_left"]
            stats["threshold_mismatch"] += m["adaptive_threshold"] != ADAPTIVE_THRESHOLD

            # -- D: actions are valid for THIS board -----------------------
            c, j = r["chosen"].lower(), r["rejected"].lower()
            stats["chosen_illegal"] += c not in LEGAL
            stats["rejected_illegal"] += j not in LEGAL
            stats["same_word"] += c == j
            stats["bad_length"] += (len(c) != 5 or len(j) != 5)
            stats["not_uppercase"] += (r["chosen"] != c.upper()
                                       or r["rejected"] != j.upper())
            space = set(board.actions().tolist())
            ci, ji = V.index.get(c, -1), V.index.get(j, -1)
            stats["chosen_outside_action_space"] += ci not in space
            stats["rejected_outside_action_space"] += ji not in space
            adm = set(board.admissible.tolist())
            cand = set(board.candidates.tolist())
            stats["chosen_admissible_flag_wrong"] += (ci in adm) != m["chosen_is_admissible"]
            stats["rejected_admissible_flag_wrong"] += (ji in adm) != m["rejected_is_admissible"]
            stats["chosen_candidate_flag_wrong"] += (ci in cand) != m["chosen_is_candidate"]
            stats["rejected_candidate_flag_wrong"] += (ji in cand) != m["rejected_is_candidate"]

            # -- E: recompute the label from scratch -----------------------
            n = len(board.candidates)
            if board.restricted:
                totals = tree.action_costs(board)
                vals = {g: t / n for g, t in totals.items()}
                stats["exact_rows"] += 1
                stats["chosen_total_cost_wrong"] += totals[ci] != m["chosen_total_cost"]
                stats["rejected_total_cost_wrong"] += totals[ji] != m["rejected_total_cost"]
                # integer comparison: exact, no tolerance needed
                if not totals[ci] < totals[ji]:
                    stats["BACKWARDS"] += 1
                stats["family_mismatch"] += m["score_family"] != "exact_adaptive_decoder_tree"
                stats["depth_mismatch"] += m["tree_depth"] != board.guesses_left
            else:
                sumsq = np.rint(oneply.action_scores(board) * n).astype(np.int64)
                vals = {i: int(v) / n for i, v in enumerate(sumsq)}
                stats["proxy_rows"] += 1
                if not sumsq[ci] < sumsq[ji]:
                    stats["BACKWARDS"] += 1
                stats["family_mismatch"] += m["score_family"] != \
                    "full_legal_pool_one_ply_expected_remaining"

            best = min(vals.values())
            tol = 1e-12
            n_opt = sum(1 for v in vals.values() if v <= best + tol)
            stats["chosen_not_optimal"] += vals[ci] > best + tol
            stats["n_optimal_mismatch"] += n_opt != m["n_optimal_actions"]
            stats["n_actions_mismatch"] += len(vals) != m["n_actions"]
            # -- F: the pair must not be two tied optima -------------------
            stats["TIED_PAIR"] += abs(vals[ci] - vals[ji]) <= tol
            for k, field in (("chosen_value", ci), ("rejected_value", ji)):
                stats[f"{k}_wrong"] += abs(vals[field] - m[k]) > 1e-6
            gap = vals[ji] - vals[ci]
            stats["gap_wrong"] += abs(gap - m["preference_gap"]) > 1e-6
            per_family[m["score_family"]][m["pair_type"]].append(gap)

    hard = ["turn_mismatch", "n_cand_mismatch", "n_adm_mismatch", "bucket_mismatch",
            "action_space_mismatch", "state_key_mismatch", "state_hash_mismatch",
            "guesses_left_mismatch", "threshold_mismatch", "chosen_illegal",
            "rejected_illegal", "same_word", "bad_length", "not_uppercase",
            "chosen_outside_action_space", "rejected_outside_action_space",
            "chosen_admissible_flag_wrong", "rejected_admissible_flag_wrong",
            "chosen_candidate_flag_wrong", "rejected_candidate_flag_wrong",
            "chosen_total_cost_wrong", "rejected_total_cost_wrong", "BACKWARDS",
            "family_mismatch", "depth_mismatch", "chosen_not_optimal",
            "n_optimal_mismatch", "n_actions_mismatch", "TIED_PAIR",
            "chosen_value_wrong", "rejected_value_wrong", "gap_wrong"]
    labels = {
        "BACKWARDS": "preference direction correct on EVERY row "
                     "(recomputed independently)",
        "TIED_PAIR": "no pair between two exactly-optimal actions",
        "chosen_not_optimal": "`chosen` is an optimal action on every row",
        "chosen_outside_action_space": "`chosen` is inside the deployed "
                                       "decoder's action space",
        "rejected_outside_action_space": "`rejected` is inside the deployed "
                                         "decoder's action space",
    }
    for k in hard:
        check(stats[k] == 0, labels.get(k, k), f"{stats[k]} bad rows" if stats[k] else "")
    print(f"       ({stats['exact_rows']} rows scored by exact tree, "
          f"{stats['proxy_rows']} by one-ply proxy)")
    report["recompute"] = {k: int(stats[k]) for k in hard}

    # ------------------------------------------- E2: brute-force cross-check
    print("\nE2. EXACT VALUE FUNCTION vs A REFERENCE IMPLEMENTATION")
    ref_checked = ref_bad = 0
    rng = np.random.default_rng(0)
    exact_rows = [(n, r) for n, rows in splits.items() for r in rows
                  if r["meta"]["score_family"] == "exact_adaptive_decoder_tree"]
    if exact_rows:
        idx = rng.permutation(len(exact_rows))[:a.brute_force_sample]
        for i in idx:
            name, r = exact_rows[int(i)]
            board = boards[(name, r["id"])]
            for w in (r["chosen"].lower(), r["rejected"].lower()):
                g = V.index[w]
                got = tree._branch(board.candidates, board.admissible, g,
                                   board.guesses_left)
                exp = _brute(V, board.candidates, board.admissible, g,
                             board.guesses_left)
                ref_checked += 1
                ref_bad += got != exp
    check(ref_bad == 0,
          "memoised/pruned tree agrees with a no-memo no-pruning reference",
          f"{ref_checked} action values checked, {ref_bad} mismatches")
    report["brute_force"] = {"checked": ref_checked, "mismatches": ref_bad}

    # ---------------------------------------------------------------- G
    print("\nG. TRAIN / VALIDATION SEPARATION")
    tk = {r["meta"]["state_key"] for r in splits["train"]}
    vk = {r["meta"]["state_key"] for r in splits["validation"]}
    tp = {r["prompt"] for r in splits["train"]}
    vp = {r["prompt"] for r in splits["validation"]}
    check(not tk & vk, "no shared state between train and validation",
          f"{len(tk & vk)} shared")
    check(not tp & vp, "no shared prompt between train and validation",
          f"{len(tp & vp)} shared")
    th = {r["meta"]["source_answer_hash"] for r in splits["train"]}
    vh = {r["meta"]["source_answer_hash"] for r in splits["validation"]}
    check(not th & vh, "no shared source answer between train and validation",
          f"{len(th & vh)} shared")
    # confirm the source-answer hashes really are from the declared splits
    tr_h = {sha("phase8v3-source", w)[:16] for w in train_answers}
    va_h = {sha("phase8v3-source", w)[:16] for w in val_answers}
    check(th <= tr_h, "every train row came from a TRAIN answer")
    check(vh <= va_h, "every validation row came from a HELD-OUT answer")
    report["split"] = {"train_states": len(tk), "val_states": len(vk),
                       "state_overlap": len(tk & vk), "prompt_overlap": len(tp & vp)}

    # ---------------------------------------------------------------- H
    print("\nH. ANSWER LEAKAGE")
    lk = sum(1 for r in splits["train"]
             if r["chosen"].lower() in HOLDOUT or r["rejected"].lower() in HOLDOUT)
    check(lk == 0, "no TRAIN row names a held-out answer as chosen or rejected",
          f"{lk} rows  (v1: 694 chosen + 193 rejected)")
    touch = 0
    for name, rows in (("train", splits["train"]),):
        for r in rows:
            b = boards[(name, r["id"])]
            if {V.words[int(i)] for i in b.candidates} & HOLDOUT:
                touch += 1
    frac = touch / max(len(splits["train"]), 1)
    warn(frac < 0.40,
         "train states whose candidate set contains a held-out answer",
         f"{touch}/{len(splits['train'])} ({100*frac:.1f}%) — unavoidable for "
         f"large states; use --strict-candidate-partition to force 0")
    report["leakage"] = {"train_rows_naming_holdout": lk,
                         "train_states_touching_holdout": touch,
                         "train_states_touching_holdout_frac": round(frac, 4)}

    # ---------------------------------------------------------------- I
    print("\nI. DISTRIBUTION")
    for name, rows in splits.items():
        if not rows:
            continue
        pt = Counter(r["meta"]["pair_type"] for r in rows)
        bk = Counter(r["meta"]["bucket"] for r in rows)
        print(f"  {name}: {len(rows)} rows")
        print(f"    buckets     : {dict(sorted(bk.items()))}")
        print(f"    pair types  : {dict(pt)}  competitive = "
              f"{100*pt['competitive']/len(rows):.1f}%")
        print(f"    turns       : {dict(sorted(Counter(r['meta']['turn'] for r in rows).items()))}")
        print(f"    2-10 share  : {100*bk['2-10']/len(rows):.1f}%  "
              f"(headroom analysis puts 74.7% of the gap there)")
        print(f"    chosen is a candidate: "
              f"{100*np.mean([r['meta']['chosen_is_candidate'] for r in rows]):.1f}%   "
              f"rejected is a candidate: "
              f"{100*np.mean([r['meta']['rejected_is_candidate'] for r in rows]):.1f}%")
        print(f"    states with a tied optimum: "
              f"{100*np.mean([r['meta']['n_optimal_actions']>1 for r in rows]):.1f}%")
    print("\n  gap by score family (recomputed, not read from metadata):")
    for fam, kinds in sorted(per_family.items()):
        units = ("mean further guesses" if fam.startswith("exact")
                 else "expected remaining candidates")
        for kind, g in sorted(kinds.items()):
            g = np.array(g)
            print(f"    {fam[:34]:34s} {kind:12s} n={len(g):5d} "
                  f"median={np.median(g):8.4f} p90={np.percentile(g,90):8.4f} "
                  f"max={g.max():9.4f}  [{units}]")
    report["gap_by_family"] = {
        fam: {kind: {"n": len(g), "median": round(float(np.median(g)), 4),
                     "p90": round(float(np.percentile(g, 90)), 4),
                     "max": round(float(max(g)), 4)}
              for kind, g in kinds.items()}
        for fam, kinds in per_family.items()}

    # ---------------------------------------------------------------- J
    print("\nJ. SELECTION-BIAS DIAGNOSTIC (held-out-action filter)")
    rej = (manifest.get("rejections", {}) or {}).get("train", {})
    dropped = rej.get("action_is_holdout_answer", 0)
    kept = len(splits["train"])
    share = dropped / max(dropped + kept, 1)
    warn(share < 0.25,
         "share of candidate training states dropped for naming a held-out answer",
         f"{dropped} dropped vs {kept} kept ({100*share:.1f}%)")
    print("      Held-out answers are 246/2315 = 10.6% of the answer list, and a")
    print("      pair exposes two actions, so if both were answers the expected")
    print("      drop rate would be 1-(1-0.106)^2 = 20.1%. Not every action is an")
    print("      answer, so anything between ~10% and ~20% is the filter doing")
    print("      its job.")
    print("      The stronger argument is structural: the held-out set is chosen")
    print("      by sha256(salt|answer), which is independent of every game")
    print("      property, so the filter cannot preferentially remove hard or")
    print("      easy states — only a pseudorandom 10.6% of words.")
    report["holdout_filter"] = {"dropped": dropped, "kept": kept,
                                "share": round(share, 4)}

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 72)
    if FAILS:
        print(f"VERDICT: {len(FAILS)} HARD CHECK(S) FAILED")
        for f in FAILS:
            print(f"  - {f}")
    else:
        print("VERDICT: ALL HARD CHECKS PASSED")
    if WARNS:
        print(f"{len(WARNS)} warning(s): " + ", ".join(WARNS))
    print("=" * 72)
    report["verdict"] = {"failures": FAILS, "warnings": WARNS,
                         "passed": not FAILS}

    out = a.out or os.path.join(a.data, "audit.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"audit written to {out}")
    return 1 if FAILS else 0


def _brute(V, S, A, g, depth, fail=7):
    """Reference: cost of playing `g`, with no memo, no pruning, no ordering."""
    row = V._row(g)
    cs, ca = row[S], row[A]
    total = len(S)
    for code in np.unique(cs):
        if int(code) == ALL_GREEN:
            continue
        total += _brute_cost(V, S[cs == code], A[ca == code], depth - 1, fail)
    return total


def _brute_cost(V, S, A, depth, fail):
    n = len(S)
    if n == 0:
        return 0
    if depth <= 0 or len(A) == 0:
        return fail * n
    if n == 1:
        return 1
    return min(_brute(V, S, A, int(g), depth, fail) for g in A)


if __name__ == "__main__":
    sys.exit(main())
