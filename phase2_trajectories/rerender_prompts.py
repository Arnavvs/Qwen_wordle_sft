"""
rerender_prompts.py — rebuild the `prompt` field of every trajectory record with
the candidate count removed, changing nothing else.

    python rerender_prompts.py            # verify + rewrite
    python rerender_prompts.py --dry-run  # verify only

WHY RE-RENDER RATHER THAN REGENERATE
------------------------------------
Re-running the solvers would reproduce the same trajectories, but only if every
seed, config and tie-break resolved identically -- and any drift would silently
alter the expert actions. Re-rendering touches exactly one field. Histories,
feedback, expert actions, candidate counts, splits and outcomes are carried
through untouched, and the script asserts that.

THE SAFETY ARGUMENT
-------------------
Before rewriting, each record's prompt is re-rendered with
`show_candidate_count=True` and compared byte-for-byte against the stored
prompt. If those match, the renderer is provably reproducing the original from
the stored state alone, so the only difference in the new prompt is the removed
line. A mismatch aborts the run rather than writing anything.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import json
import os
import shutil
from typing import Dict, List

from generate_trajectories import derive_constraints, render_prompt

SOLVERS = ["entropy", "tree_soare", "tree_salet"]
IMMUTABLE = ["completion", "split", "turn", "solver", "game_id",
             "action", "outcome", "_holdout"]


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def rebuild(rec: Dict, max_guesses: int = 6, show_count: bool = False) -> str:
    hist = [(h["guess"].lower(), h["feedback"]) for h in rec["state"]["history"]]
    turn = rec["state"]["turn"]
    return render_prompt(
        turn=turn,
        history=hist,
        constraints=derive_constraints(hist),
        n_candidates=rec["state"]["n_candidates"],
        guesses_remaining=rec["state"]["guesses_remaining"],
        max_guesses=max_guesses,
        candidates=None,
        show_candidate_count=show_count,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="trajectories")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", default="trajectories_with_count",
                    help="where the originals are copied before rewriting")
    args = ap.parse_args(argv)

    total_ok = total_changed = 0
    all_rows = {}

    for s in SOLVERS:
        path = os.path.join(args.src, f"{s}_steps.jsonl")
        rows = load_jsonl(path)
        reproduced = changed = 0
        for r in rows:
            # 1. prove the renderer reproduces the original exactly
            old = rebuild(r, show_count=True)
            if old != r["prompt"]:
                print(f"ABORT: {s} record {r.get('game_id')} turn {r['turn']} "
                      f"could not be reproduced from its stored state.")
                print("  stored :", repr(r["prompt"][-160:]))
                print("  rebuilt:", repr(old[-160:]))
                return 1
            reproduced += 1
            # 2. build the new prompt with the count removed
            new = rebuild(r, show_count=False)
            assert "Possible answers remaining" not in new
            if new != r["prompt"]:
                changed += 1
            r["_new_prompt"] = new
        print(f"{s:<12} {len(rows):>5} records | reproduced {reproduced} | "
              f"prompts changed {changed}")
        total_ok += reproduced
        total_changed += changed
        all_rows[s] = rows

    print(f"\nreproduced {total_ok} originals exactly; "
          f"{total_changed} prompts will change")

    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    os.makedirs(args.backup, exist_ok=True)
    for s in SOLVERS:
        for suffix in ("_steps.jsonl", "_games.jsonl"):
            src = os.path.join(args.src, s + suffix)
            dst = os.path.join(args.backup, s + suffix)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
    print(f"originals backed up to {args.backup}/")

    for s in SOLVERS:
        rows = all_rows[s]
        path = os.path.join(args.src, f"{s}_steps.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                new = r.pop("_new_prompt")
                r["prompt"] = new
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"rewrote {path}")

    # ---- post-write invariants ------------------------------------------
    print("\npost-write checks")
    for s in SOLVERS:
        new_rows = load_jsonl(os.path.join(args.src, f"{s}_steps.jsonl"))
        old_rows = load_jsonl(os.path.join(args.backup, f"{s}_steps.jsonl"))
        assert len(new_rows) == len(old_rows), f"{s}: record count changed"
        for a, b in zip(old_rows, new_rows):
            for k in IMMUTABLE:
                assert a.get(k) == b.get(k), f"{s}: field {k!r} changed"
            assert a["state"]["history"] == b["state"]["history"]
            assert a["state"]["n_candidates"] == b["state"]["n_candidates"]
            assert "Possible answers remaining" not in b["prompt"]
        print(f"  {s:<12} OK  histories, actions, counts, splits, outcomes "
              f"all unchanged; count absent from every prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
