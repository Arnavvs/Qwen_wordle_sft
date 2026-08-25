"""
inspect_examples.py — read examples out of the SFT package.

    python inspect_examples.py --dataset entropy --n 3
    python inspect_examples.py --dataset tree_soare --turn 2 --n 5
    python inspect_examples.py --disagreement --n 10
    python inspect_examples.py --disagreement --full --n 3
    python inspect_examples.py --stats

`--show-answer` reveals the held-out answer via index/id_map.jsonl. It is off by
default so that casual inspection cannot accidentally paste an answer into
something the model will read.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from typing import Dict, List

PKG = "sft_package"


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def rule(ch: str = "-", n: int = 78) -> str:
    return ch * n


def show_record(r: Dict, idmap: Dict, show_answer: bool, full: bool) -> None:
    m = r.get("meta", {})
    print(rule("="))
    head = f"id={r['id']}  solver={m.get('solver')}  turn={m.get('turn')}"
    if m.get("n_candidates") is not None:
        head += f"  candidates={m['n_candidates']}"
    if m.get("is_disagreement_state"):
        head += "  [DISAGREEMENT]"
    print(head)
    if show_answer and r["id"] in idmap:
        print(f"answer (held out): {idmap[r['id']]['answer']}")
    print(rule())
    if full:
        print(r["prompt"])
    else:
        # Skip the fixed rules block; show only the part that varies.
        body = r["prompt"].split("B = letter is not in the word", 1)[-1]
        body = body.split("\n", 1)[-1] if "\n" in body else body
        print(body.strip())
    print(f"--> COMPLETION: {r['completion']}")
    extra = {k: v for k, v in m.items()
             if k in ("action_is_candidate", "information_gain_bits",
                      "duplicate_count", "fully_determined", "split")}
    print(f"    meta: {extra}")


def show_contrast(c: Dict, full: bool) -> None:
    m = c["meta"]
    print(rule("="))
    hist = " ".join(f"{h['guess']}:{h['feedback']}" for h in m["history"])
    print(f"id={c['id']}  turn={m['turn']}  candidates={m['n_candidates']}")
    print(f"history: {hist}")
    print(f"reached by {m['reached_by_train_games']} train / "
          f"{m['reached_by_val_games']} val games")
    print(rule())
    if full:
        print(c["prompt"])
    e, t = c["completions"]["entropy"], c["completions"]["tree_soare"]
    print(f"  entropy    -> {e}   (candidate: {m['entropy_action_is_candidate']})")
    print(f"  tree_soare -> {t}   (candidate: {m['tree_action_is_candidate']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=PKG)
    ap.add_argument("--dataset", default="entropy",
                    choices=["entropy", "tree_soare", "tree_salet"])
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--turn", type=int, default=None)
    ap.add_argument("--min-candidates", type=int, default=None)
    ap.add_argument("--only-disagreement", action="store_true")
    ap.add_argument("--disagreement", action="store_true",
                    help="show the entropy-vs-tree contrast file instead")
    ap.add_argument("--full", action="store_true",
                    help="print the whole prompt including the rules block")
    ap.add_argument("--show-answer", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    P = args.package
    idmap = {}
    imp = os.path.join(P, "index", "id_map.jsonl")
    if args.show_answer and os.path.exists(imp):
        idmap = {r["id"]: r for r in load_jsonl(imp)}

    if args.stats:
        print(rule("="))
        print("SFT PACKAGE CONTENTS")
        print(rule("="))
        man = json.load(open(os.path.join(P, "manifest.json"), encoding="utf-8"))
        print(f"{'dataset':<14}{'train':>8}{'val':>7}{'distinct':>10}{'dup':>7}"
              f"{'disagree':>10}")
        print(rule())
        for k, v in man["solvers"].items():
            print(f"{k:<14}{v['train_records']:>8}{v['val_records']:>7}"
                  f"{v['distinct_train_pairs']:>10}"
                  f"{v['duplication_factor']:>7.2f}"
                  f"{v['disagreement_train_records']:>10}")
        d = man["disagreement"]
        print(f"\ndisagreement: {d['disagreement_states']} states "
              f"({d['pair']}), by turn {d['by_turn']}")
        print(f"  oversampling pool {d['oversampling_pool_states']}, "
              f"excluded (touch val) {d['excluded_touch_val']}")
        for s in ("entropy", "tree_soare", "tree_salet"):
            rows = load_jsonl(os.path.join(P, "data", s, "train.jsonl"))
            c = collections.Counter(r["meta"]["turn"] for r in rows)
            pc = collections.Counter(r["meta"]["action_is_candidate"]
                                     for r in rows)
            print(f"\n{s}: turns {dict(sorted(c.items()))}")
            print(f"  actions that were live candidates: "
                  f"{100*pc[True]/max(len(rows),1):.1f}%")
        return 0

    if args.disagreement:
        rows = load_jsonl(os.path.join(P, "data", "disagreement",
                                       "contrast.jsonl"))
        rows.sort(key=lambda c: -c["meta"]["n_candidates"])
        print(f"{len(rows)} disagreement states "
              f"(entropy vs tree_soare, shared opener)\n")
        for c in rows[:args.n]:
            show_contrast(c, args.full)
        return 0

    path = os.path.join(P, "data", args.dataset, f"{args.split}.jsonl")
    rows = load_jsonl(path)
    if args.turn is not None:
        rows = [r for r in rows if r["meta"]["turn"] == args.turn]
    if args.min_candidates is not None:
        rows = [r for r in rows if r["meta"]["n_candidates"] >= args.min_candidates]
    if args.only_disagreement:
        rows = [r for r in rows if r["meta"]["is_disagreement_state"]]
    if not rows:
        print("no records match those filters")
        return 1

    random.Random(args.seed).shuffle(rows)
    print(f"{path}: {len(rows)} matching records, showing {min(args.n, len(rows))}\n")
    for r in rows[:args.n]:
        show_record(r, idmap, args.show_answer, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
