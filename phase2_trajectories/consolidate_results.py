"""
consolidate_results.py — merge every measured run into one comparison artifact.

Reads only files produced by actual runs; anything missing is reported as
NOT RUN rather than filled in. Writes artifacts/final_comparison.json and prints
the research comparison table.
"""

from __future__ import annotations

import json
import os

ART = "artifacts"


def load(path):
    p = os.path.join(ART, path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def from_summary(s):
    dc = {int(k): v for k, v in s["distribution_counts"].items()}
    n = s["n_games"]
    cum = lambda m: 100 * sum(dc.get(i, 0) for i in range(1, m + 1)) / n
    return {
        "solver": s["strategy"], "n": n, "opener": s.get("opening_guess"),
        "mean": s["mean_score_failures_penalized"],
        "median": s["median_guesses"], "std": s["std_guesses"],
        "max": s["max_guesses"],
        "le3": cum(3), "le4": cum(4), "le5": cum(5), "le6": cum(6),
        "fails": s["n_failed"], "fail_pct": s["failure_rate_pct"],
        "ms_per_game": s["ms_per_game"], "wall_s": s["wall_s"],
        "total_guesses": round(s["mean_score_failures_penalized"] * n),
        "cache_size": None, "nodes": None,
    }


rows = []

bench = load("benchmark_results.json")
if bench:
    for s in bench["summaries"]:
        rows.append(from_summary(s))

TREE_FILES = [
    ("tree_full_d1.json", "tree depth-1"),
    ("tree_full_d2.json", "tree depth-2"),
    ("tree_full_d3.json", "tree depth-3"),
    ("tree_full_soare.json", "tree full (SOARE)"),
    ("tree_full_salet.json", "tree full (SALET)"),
]
missing = []
for fname, label in TREE_FILES:
    d = load(fname)
    if not d:
        missing.append(label)
        continue
    r = dict(d[0])
    r["solver"] = label
    r["total_guesses"] = round(r["mean"] * r["n"])
    r.setdefault("opener", "salet" if "salet" in fname else "soare")
    rows.append(r)

OPTIMAL_TOTAL, N_ANS = 7920, 2315
OPTIMAL_MEAN = OPTIMAL_TOTAL / N_ANS

HDR = (f"{'Solver':<20}{'Open':<7}{'Avg':>8}{'Worst':>7}{'<=3':>8}{'<=4':>8}"
       f"{'<=5':>8}{'Fail%':>7}{'ms/game':>10}{'Total':>8}{'vs opt':>9}")
print(HDR)
print("-" * len(HDR))
for r in rows:
    if r["n"] != N_ANS:
        continue
    gap = r["mean"] - OPTIMAL_MEAN
    print(f"{r['solver']:<20}{(r.get('opener') or '-'):<7}{r['mean']:>8.4f}"
          f"{r['max']:>7}{r['le3']:>8.2f}{r['le4']:>8.2f}{r['le5']:>8.2f}"
          f"{r['fail_pct']:>7.2f}{r['ms_per_game']:>10.1f}"
          f"{r['total_guesses']:>8}{gap:>+9.4f}")
print("-" * len(HDR))
print(f"{'PROVEN OPTIMAL':<20}{'salet':<7}{OPTIMAL_MEAN:>8.4f}{5:>7}"
      f"{'-':>8}{'-':>8}{'100.00':>8}{0.00:>7.2f}{'-':>10}{OPTIMAL_TOTAL:>8}"
      f"{0.0:>+9.4f}")

if missing:
    print(f"\nNOT RUN / MISSING: {missing}")

with open(os.path.join(ART, "final_comparison.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "rows": rows,
        "reference_optimal": {
            "total": OPTIMAL_TOTAL, "mean": OPTIMAL_MEAN, "worst_case": 5,
            "opener": "salet", "mode": "normal",
            "answer_list": 2315, "guess_pool": 12972,
            "source": "Alex Selby, 'The best strategies for Wordle', "
                      "sonorouschocolate.com (2022), exhaustive search",
        },
        "missing": missing,
    }, fh, indent=2)
print(f"\nwrote {ART}/final_comparison.json")
