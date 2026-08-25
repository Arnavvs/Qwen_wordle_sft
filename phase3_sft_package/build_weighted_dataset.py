"""
build_weighted_dataset.py — materialize a training file from an oversampling
recipe. Oversampling lives here and in the config, never in the raw data.

    python build_weighted_dataset.py --config sft_package/configs/baseline_entropy.json
    python build_weighted_dataset.py --config <cfg> --out my_train.jsonl
    python build_weighted_dataset.py --list

Recipe format:

    {
      "name": "tree_soare_disagreement_x20",
      "seed": 20260817,
      "dedup": false,            # collapse identical (prompt, completion) pairs
      "shuffle": true,
      "components": [
        {"source": "data/tree_soare/train.jsonl", "repeat": 1},
        {"source": "data/disagreement/tree_soare.jsonl", "repeat": 20}
      ]
    }

`repeat` may be fractional: 0.25 samples a quarter of the component without
replacement, using the recipe seed, so partial weights are reproducible.

Does NOT train anything. It only writes a JSONL file.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
from typing import Dict, List

PKG = "sft_package"


def load_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--package", default=PKG)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true", help="list available recipes")
    ap.add_argument("--dry-run", action="store_true",
                    help="report composition without writing")
    args = ap.parse_args(argv)

    if args.list or not args.config:
        cfgs = sorted(glob.glob(os.path.join(args.package, "configs", "*.json")))
        print("available recipes:")
        for c in cfgs:
            d = json.load(open(c, encoding="utf-8"))
            print(f"  {os.path.basename(c):<36} {d.get('description','')}")
        return 0

    cfg = json.load(open(args.config, encoding="utf-8"))
    rng = random.Random(cfg.get("seed", 0))
    rows: List[Dict] = []
    breakdown = []

    for comp in cfg["components"]:
        path = comp["source"]
        if not os.path.isabs(path):
            path = os.path.join(args.package, path)
        src = load_jsonl(path)
        rep = float(comp.get("repeat", 1))
        whole, frac = int(rep), rep - int(rep)
        out = src * whole
        if frac > 0:
            k = int(round(len(src) * frac))
            out += rng.sample(src, min(k, len(src)))
        rows.extend(out)
        breakdown.append({
            "source": comp["source"], "records_in_source": len(src),
            "repeat": rep, "records_contributed": len(out),
        })
        print(f"  {comp['source']:<44} {len(src):>6} x{rep:<5g} -> {len(out):>7}")

    if cfg.get("dedup"):
        before = len(rows)
        seen = set()
        ded = []
        for r in rows:
            k = (r["prompt"], r["completion"])
            if k in seen:
                continue
            seen.add(k)
            ded.append(r)
        rows = ded
        print(f"  dedup: {before} -> {len(rows)} distinct (prompt, completion)")

    if cfg.get("shuffle", True):
        rng.shuffle(rows)

    # Composition report — what the model would actually see.
    by_solver = collections.Counter(r["meta"]["solver"] for r in rows)
    by_turn = collections.Counter(r["meta"]["turn"] for r in rows)
    n_dis = sum(1 for r in rows if r["meta"].get("is_disagreement_state"))
    print(f"\ntotal records      {len(rows)}")
    print(f"  by solver        {dict(by_solver)}")
    print(f"  by turn          {dict(sorted(by_turn.items()))}")
    print(f"  disagreement     {n_dis} ({100*n_dis/max(len(rows),1):.2f}%)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = args.out or os.path.join(args.package, "built",
                                   f"{cfg['name']}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "recipe": cfg,
        "records": len(rows),
        "by_solver": dict(by_solver),
        "by_turn": {str(k): v for k, v in sorted(by_turn.items())},
        "disagreement_records": n_dis,
        "components": breakdown,
    }
    with open(out.replace(".jsonl", ".meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {out}")
    print(f"wrote {out.replace('.jsonl', '.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
