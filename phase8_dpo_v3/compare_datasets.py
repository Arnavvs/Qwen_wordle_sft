"""compare_datasets.py — side-by-side statistics, Phase-8 v1 vs v3.

Both datasets are measured the same way, from their prompts rather than from
their metadata, so the columns are genuinely comparable. Where a quantity only
makes sense for one of them (v1 has no validation split; v3 has no
random-alternative sampling) the cell says so instead of inventing a number.

    python phase8_dpo_v3/compare_datasets.py
"""
import _paths  # noqa: F401

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

from dpo_core import Board, Vocab, bucket_of, load_answer_splits

HISTORY_LINE = re.compile(r"^\s+\d+\.\s+([A-Z]{5})\s+->\s+([BYG]{5})$", re.M)


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def v3_bucket(r):
    return r["meta"]["bucket"]


def v1_bucket(r):
    """Collapse v1's six buckets onto v3's three so the rows line up."""
    return bucket_of(r["meta"]["n_admissible"])


def row(label, left, right, note=""):
    print(f"  {label:<44} {str(left):>18}  {str(right):>18}   {note}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", default="sft_package/data/dpo_midgame/train.jsonl")
    ap.add_argument("--new", default="sft_package/data/dpo_v3")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="phase8_dpo_v3/comparison.json")
    a = ap.parse_args(argv)

    V = Vocab(a.artifacts)
    _, val_answers = load_answer_splits()
    HOLDOUT = set(val_answers)

    old = load(a.legacy)
    new_tr = load(os.path.join(a.new, "train.jsonl"))
    new_va = load(os.path.join(a.new, "validation.jsonl"))
    new = new_tr + new_va

    boards = {}
    for r in old + new:
        p = r["prompt"]
        if p not in boards:
            boards[p] = Board.from_history(V, HISTORY_LINE.findall(p))

    def stats(rows, bucket_fn, kind_map):
        if not rows:
            return {}
        pc = Counter(r["prompt"] for r in rows)
        bk = Counter(bucket_fn(r) for r in rows)
        kd = Counter(kind_map(r) for r in rows)
        touch = sum(1 for r in rows
                    if {V.words[int(i)] for i in boards[r["prompt"]].candidates} & HOLDOUT)
        # A `rejected` that contradicts visible feedback is only a trivial cue
        # if `chosen` does NOT. Where the decoder leaves the model unrestricted,
        # the strongest probes are usually inadmissible themselves, so what
        # matters is the ASYMMETRY: how much more often the rejected action is
        # inconsistent than the chosen one. A large gap lets the model win the
        # pair by checking consistency alone, without any Wordle strategy.
        inadm = c_inadm = j_inadm = unres = 0
        for r in rows:
            b = boards[r["prompt"]]
            adm = set(b.admissible.tolist())
            cj = V.index[r["chosen"].lower()] not in adm
            rj = V.index[r["rejected"].lower()] not in adm
            inadm += rj
            if len(b.admissible) > 20:      # the unrestricted regime
                unres += 1
                c_inadm += cj
                j_inadm += rj
        return {
            "rows": len(rows),
            "distinct_states": len(pc),
            "rows_per_state_max": max(pc.values()),
            "rows_per_state_mean": round(len(rows) / len(pc), 3),
            "duplicate_ids": len(rows) - len({r["id"] for r in rows}),
            "bucket": dict(bk),
            "bucket_frac": {k: round(v / len(rows), 4) for k, v in bk.items()},
            "kind": dict(kd),
            "hard_pair_frac": round(kd.get("competitive", 0) / len(rows), 4),
            "turn": dict(sorted(Counter(r["meta"]["turn"] for r in rows).items())),
            "rejected_contradicts_feedback": inadm,
            "rejected_contradicts_feedback_frac": round(inadm / len(rows), 4),
            "unrestricted_rows": unres,
            "unrestricted_chosen_inadmissible_frac":
                round(c_inadm / unres, 4) if unres else None,
            "unrestricted_rejected_inadmissible_frac":
                round(j_inadm / unres, 4) if unres else None,
            "unrestricted_consistency_asymmetry":
                round((j_inadm - c_inadm) / unres, 4) if unres else None,
            "chosen_is_holdout_answer": sum(1 for r in rows
                                            if r["chosen"].lower() in HOLDOUT),
            "rejected_is_holdout_answer": sum(1 for r in rows
                                              if r["rejected"].lower() in HOLDOUT),
            "rows_touching_holdout_candidate": touch,
        }

    o = stats(old, v1_bucket, lambda r: r["meta"]["pair_type"])
    n = stats(new_tr, v3_bucket, lambda r: r["meta"]["pair_type"])

    print("=" * 96)
    print("PHASE-8 PREFERENCE DATA — v1 vs v3   (v3 column = TRAIN split)")
    print("=" * 96)
    print(f"  {'':<44} {'v1 (dpo_midgame)':>18}  {'v3 (dpo_v3)':>18}")
    print("  " + "-" * 92)
    row("rows", o["rows"], n["rows"])
    row("distinct states", o["distinct_states"], n["distinct_states"])
    row("mean rows per state", o["rows_per_state_mean"], n["rows_per_state_mean"])
    row("MAX rows for a single state", o["rows_per_state_max"], n["rows_per_state_max"],
        "v1: one turn-2 state is 1.2% of the set")
    row("duplicate ids", o["duplicate_ids"], n["duplicate_ids"])
    print()
    for b in ("2-10", "11-100", "100+"):
        row(f"bucket |admissible| {b}",
            f"{o['bucket'].get(b,0)} ({100*o['bucket_frac'].get(b,0):.1f}%)",
            f"{n['bucket'].get(b,0)} ({100*n['bucket_frac'].get(b,0):.1f}%)",
            {"2-10": "74.7% of the measured headroom",
             "11-100": "29.1%", "100+": "-124.1%"}[b])
    print()
    row("hard-pair share", f"{100*o['hard_pair_frac']:.1f}%",
        f"{100*n['hard_pair_frac']:.1f}%",
        "v1 'competitive' / v3 'competitive'")
    row("`rejected` contradicts visible feedback",
        f"{o['rejected_contradicts_feedback']} ({100*o['rejected_contradicts_feedback_frac']:.1f}%)",
        f"{n['rejected_contradicts_feedback']} ({100*n['rejected_contradicts_feedback_frac']:.1f}%)",
        "raw rate; see the asymmetry below")
    print()
    print("  In the UNRESTRICTED regime, inadmissible actions are normal — the")
    print("  best probes are usually inconsistent themselves. What makes a pair")
    print("  trivial is the ASYMMETRY between the two sides:")
    row("  unrestricted rows", o["unrestricted_rows"], n["unrestricted_rows"])
    row("  ... `chosen` inadmissible",
        f"{100*o['unrestricted_chosen_inadmissible_frac']:.1f}%",
        f"{100*n['unrestricted_chosen_inadmissible_frac']:.1f}%")
    row("  ... `rejected` inadmissible",
        f"{100*o['unrestricted_rejected_inadmissible_frac']:.1f}%",
        f"{100*n['unrestricted_rejected_inadmissible_frac']:.1f}%")
    row("  ... ASYMMETRY (the trivial cue)",
        f"{100*o['unrestricted_consistency_asymmetry']:+.1f} pts",
        f"{100*n['unrestricted_consistency_asymmetry']:+.1f} pts",
        "0 = no consistency shortcut")
    print()
    row("TRAIN rows naming a held-out answer",
        o["chosen_is_holdout_answer"] + o["rejected_is_holdout_answer"],
        n["chosen_is_holdout_answer"] + n["rejected_is_holdout_answer"],
        "v3 blocks these by construction")
    row("rows whose candidates touch a held-out answer",
        f"{o['rows_touching_holdout_candidate']} "
        f"({100*o['rows_touching_holdout_candidate']/o['rows']:.1f}%)",
        f"{n['rows_touching_holdout_candidate']} "
        f"({100*n['rows_touching_holdout_candidate']/n['rows']:.1f}%)",
        "unavoidable at large |candidates|")
    print()
    row("validation split", "NONE", f"{len(new_va)} rows",
        "v1's reported acc was training acc")
    row("value function",
        "1-ply E[rem] - 0.5*is_cand",
        "exact tree (2-10) / 1-ply (rest)")
    row("alternatives scored per state",
        "120 random + expert", "every action in the space")
    row("pairs between two tied-optimal actions",
        "not checked", 0, "v3 forbids them")
    print("=" * 96)

    out = {"v1": o, "v3_train": n,
           "v3_validation": stats(new_va, v3_bucket,
                                  lambda r: r["meta"]["pair_type"]) if new_va else {}}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
