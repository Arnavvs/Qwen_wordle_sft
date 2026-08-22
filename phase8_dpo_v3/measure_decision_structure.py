"""measure_decision_structure.py — how close are the decisions the decoder faces?

This produces the numbers `AUDIT.md` uses to argue that DPO is a weak fit for
this task, so it ships as a script rather than a claim.

The question: in the 2-10 admissible regime that holds 74.7% of the remaining
gap, what does the decision actually look like? Specifically —

  * how many actions are exactly optimal (can a *pairwise* preference even
    express the target?)
  * how far away is the closest strictly-worse action (are there genuinely
    close calls to teach?)

Both are answered exactly, with the same full-depth adaptive-decoder tree the
generator uses for labels. States are sampled by rolling out real games from
training answers, so the distribution is the reachable one.

    python phase8_dpo_v3/measure_decision_structure.py
    python phase8_dpo_v3/measure_decision_structure.py --n-states 400   # faster
"""
import _paths  # noqa: F401

import argparse
import json
import time
from collections import Counter

import numpy as np

from dpo_core import (AdaptiveTree, Board, Vocab, load_answer_splits)
from wordle_solver import ALL_GREEN


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n-states", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="phase8_dpo_v3/decision_structure.json")
    a = ap.parse_args(argv)

    V = Vocab(a.artifacts)
    tree = AdaptiveTree(V)
    train_answers, _ = load_answer_splits()
    rng = np.random.default_rng(a.seed)
    opener = V.index["salet"]

    rows, seen = [], set()
    t0 = time.perf_counter()
    while len(rows) < a.n_states:
        ans = train_answers[int(rng.integers(len(train_answers)))]
        ai = V.index[ans]
        board = Board(V)
        g = opener
        for _ in range(5):
            code = V.code_against(g, ai)
            if code == ALL_GREEN:
                break
            board = board.play(g, code)
            key = board.state_key()
            if (2 <= len(board.admissible) <= 10 and key not in seen
                    and board.turn <= 5 and len(board.candidates) >= 2):
                seen.add(key)
                totals = tree.action_costs(board)
                n = len(board.candidates)
                vals = np.sort(np.array(sorted(totals.values()),
                                        dtype=np.float64)) / n
                best = vals[0]
                worse = vals[vals > best + 1e-12]
                rows.append({
                    "turn": board.turn,
                    "n_admissible": int(len(board.admissible)),
                    "n_candidates": n,
                    "n_actions": len(totals),
                    "n_optimal": int(np.sum(np.abs(vals - best) < 1e-12)),
                    "gap_to_next": float(worse[0] - best) if len(worse) else None,
                    "n_within_half": int(np.sum((vals > best + 1e-12)
                                                & (vals <= best + 0.5))),
                })
                break
            if len(board.candidates) <= 1:
                break
            adm = board.admissible
            g = int(adm[rng.integers(len(adm))]) if len(adm) <= 40 else \
                int(board.candidates[rng.integers(len(board.candidates))])

    el = time.perf_counter() - t0
    print("=" * 72)
    print(f"DECISION STRUCTURE — {len(rows)} distinct reachable 2-10 states "
          f"({el:.0f}s)")
    print("=" * 72)

    na = np.array([r["n_actions"] for r in rows])
    no = np.array([r["n_optimal"] for r in rows])
    nw = np.array([r["n_within_half"] for r in rows])
    gp = np.array([r["gap_to_next"] for r in rows if r["gap_to_next"] is not None])

    print("\nCAN A PAIRWISE PREFERENCE EXPRESS THE TARGET?")
    print(f"  actions available per state       median {int(np.median(na))}  "
          f"mean {na.mean():.1f}")
    print(f"  actions that are EXACTLY optimal  median {int(np.median(no))}  "
          f"mean {no.mean():.2f}")
    print(f"  states with >1 optimal action     {100*np.mean(no > 1):.1f}%")
    print(f"  states where EVERY action is optimal (nothing to teach)  "
          f"{100*np.mean(no == na):.1f}%")

    print("\nARE THERE GENUINELY CLOSE CALLS TO TEACH?")
    print(f"  gap to the best strictly-worse action, in mean further guesses:")
    print(f"    median {np.median(gp):.4f}   mean {gp.mean():.4f}   "
          f"p25 {np.percentile(gp, 25):.4f}   p75 {np.percentile(gp, 75):.4f}")
    for lo, hi in [(0, .125), (.125, .25), (.25, .5), (.5, 1.0), (1.0, 2.0),
                   (2.0, 1e9)]:
        m = (gp > lo) & (gp <= hi)
        hs = "inf" if hi > 1e8 else f"{hi:.3f}"
        print(f"      ({lo:.3f}, {hs:>5}]   {m.sum():5d}   {100*m.mean():5.1f}%")
    print(f"  strictly-worse actions within 0.5 guesses: median "
          f"{int(np.median(nw))}, at least one in {100*np.mean(nw > 0):.1f}% of states")

    print("\nBY |admissible|")
    out_by = {}
    for lo, hi in [(2, 3), (4, 6), (7, 10)]:
        m = [r for r in rows if lo <= r["n_admissible"] <= hi]
        if not m:
            continue
        g = np.array([r["gap_to_next"] for r in m if r["gap_to_next"] is not None])
        o = np.array([r["n_optimal"] for r in m])
        c = np.array([r["n_actions"] for r in m])
        w = np.array([r["n_within_half"] for r in m])
        print(f"  |adm| {lo:2d}-{hi:2d}  n={len(m):4d}  median gap {np.median(g):.3f}  "
              f"all-tied {100*np.mean(o == c):5.1f}%  "
              f">=1 near-miss(<=0.5) {100*np.mean(w > 0):5.1f}%")
        out_by[f"{lo}-{hi}"] = {
            "n": len(m), "median_gap": round(float(np.median(g)), 4),
            "all_tied_frac": round(float(np.mean(o == c)), 4),
            "has_near_miss_frac": round(float(np.mean(w > 0)), 4)}

    print("\nturn distribution:", dict(sorted(Counter(r["turn"] for r in rows).items())))
    print("\nREADING IT")
    print("  The typical position is: two exactly-tied best moves, and")
    print("  everything else a full guess worse. A pairwise preference cannot")
    print("  represent the tie, and there is often no close call to teach.")
    print("  That is a property of Wordle, not of any dataset.")

    result = {
        "n_states": len(rows), "seed": a.seed,
        "actions_per_state_median": int(np.median(na)),
        "optimal_actions_median": int(np.median(no)),
        "states_with_tied_optimum_frac": round(float(np.mean(no > 1)), 4),
        "states_all_optimal_frac": round(float(np.mean(no == na)), 4),
        "gap_to_next_median": round(float(np.median(gp)), 4),
        "gap_to_next_within_0.25_frac": round(float(np.mean(gp <= 0.25)), 4),
        "gap_to_next_within_0.5_frac": round(float(np.mean(gp <= 0.5)), 4),
        "by_admissible": out_by,
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
