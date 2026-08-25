"""decision_budget.py — where does the remaining 0.32 guesses actually live?

Phase 8's counterfactual answered this by *substituting* the expert into a
regime and replaying. That is a hybrid-policy evaluation: swapping an action
changes every later state, so the buckets interact and do not sum (Phase 8's
three buckets summed to -0.065 against a combined +0.321).

This measures the same question the other way round, and additively: take the
model's *own* 922 decisions from the Phase 9/10 run, and price each one against
the best action available at that exact state. No substitution, no replay, so
the numbers add up and each is attributable to a single decision.

Two regimes, two value functions, because there is no single affordable one:

  |admissible| <= 20   the exact adaptive-decoder tree from `dpo_core`. Full
                       remaining depth, integer costs, so a "mistake" is a
                       mistake and a tie is a tie.
  |admissible| >  20   the classical lookahead tree (`core/tree_search.py`)
                       priced against the expert's own choice rather than a
                       true argmin — evaluating 12,972 actions at 200
                       candidates is not affordable, and the expert's action is
                       the counterfactual that matters anyway.

The headline it produces is the ceiling on any intervention that only fixes
action selection: the most such a method can buy, before any drift cost.

    python phase11_grpo/decision_budget.py
    python phase11_grpo/decision_budget.py --unrestricted-sample 60
"""
import _paths  # noqa: F401

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np

from dpo_core import ADAPTIVE_THRESHOLD, AdaptiveTree, Board, Vocab
from wordle_solver import ALL_GREEN, SolverConfig
from tree_search import GameTree, TreeSearchConfig

RESULTS = ".kaggle_runs/phase9_out/results_phase9/harness_results.json"
BASELINE_MEAN = 3.7642
CLASSICAL_ENTROPY = 3.4431


def bucket(n):
    if n < 2:
        return "0-1_forced"
    if n <= 10:
        return "2-10"
    if n <= 20:
        return "11-20"
    if n <= 100:
        return "21-100"
    return "100+"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--cell", default="sft|baseline")
    # Default 0, and that is a finding rather than laziness. Pricing ONE
    # unrestricted decision needs two full lookahead valuations, and a single
    # turn-2 state costs ~13.5s at 83 candidates and grows sharply from there;
    # a 10-sample run did not finish in 30 minutes. The unrestricted regime is
    # not affordably priceable this way, which is exactly why Phase 11 rewards
    # it with a one-ply proxy instead. Raise this only if you are prepared to
    # wait hours for a wide-error-bar estimate.
    ap.add_argument("--unrestricted-sample", type=int, default=0,
                    help="unrestricted decisions to price with the lookahead "
                         "tree; each costs tens of seconds, see the note in "
                         "the source")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="phase11_grpo/decision_budget.json")
    a = ap.parse_args(argv)

    V = Vocab()
    tree = AdaptiveTree(V)
    R = json.load(open(a.results, encoding="utf-8"))
    d = R["games"][a.cell]
    answers, guesses = d["answers"], d["per_guess"]
    n_games = len(answers)

    print("=" * 72)
    print(f"DECISION BUDGET — {a.cell}, {n_games} games")
    print("=" * 72)

    # ---- pass 1: enumerate every decision, price the restricted ones -------
    hist = Counter()
    restricted = {"decisions": 0, "optimal": 0, "optimal_was_tie": 0,
                  "mistakes": 0}
    cost_by_bucket = defaultdict(float)
    mistakes = []
    unrestricted_states = []          # (board, model_guess) for pass 2

    t0 = time.perf_counter()
    for ans, gs in zip(answers, guesses):
        ai = V.index[ans.lower()]
        b = Board(V)
        for w in gs:
            gi = V.index[w.lower()]
            na = len(b.admissible)
            hist[bucket(na)] += 1
            if 2 <= na <= ADAPTIVE_THRESHOLD and len(b.candidates) >= 2:
                totals = tree.action_costs(b)
                if gi in totals:
                    best = min(totals.values())
                    restricted["decisions"] += 1
                    if totals[gi] == best:
                        restricted["optimal"] += 1
                        if sum(1 for v in totals.values() if v == best) > 1:
                            restricted["optimal_was_tie"] += 1
                    else:
                        c = (totals[gi] - best) / len(b.candidates)
                        restricted["mistakes"] += 1
                        cost_by_bucket[bucket(na)] += c
                        mistakes.append(c)
            elif na > ADAPTIVE_THRESHOLD and len(b.candidates) >= 2:
                unrestricted_states.append((b, gi, na))
            code = V.code_against(gi, ai)
            if code == ALL_GREEN:
                break
            b = b.play(gi, code)

    total_dec = sum(hist.values())
    print(f"\n{total_dec} decisions ({total_dec/n_games:.2f} per game), "
          f"priced in {time.perf_counter()-t0:.0f}s\n")
    print(f"  {'|admissible|':<14}{'decisions':>10}{'share':>9}")
    for k in ("0-1_forced", "2-10", "11-20", "21-100", "100+"):
        v = hist.get(k, 0)
        print(f"  {k:<14}{v:>10}{100*v/total_dec:>8.1f}%")

    print(f"\nRESTRICTED REGIME (|adm| <= {ADAPTIVE_THRESHOLD}), exact tree")
    r = restricted
    n = max(r["decisions"], 1)
    print(f"  decisions with a real choice   {r['decisions']:>5}")
    print(f"  model already optimal          {r['optimal']:>5}   "
          f"{100*r['optimal']/n:5.1f}%")
    print(f"    ...where the optimum was tied{r['optimal_was_tie']:>5}   "
          f"{100*r['optimal_was_tie']/n:5.1f}%")
    print(f"  genuine mistakes               {r['mistakes']:>5}   "
          f"{100*r['mistakes']/n:5.1f}%")
    restricted_cost = sum(cost_by_bucket.values())
    print(f"\n  expected guesses lost          {restricted_cost:.2f}")
    print(f"  per game                       {restricted_cost/n_games:.4f}")
    for k in sorted(cost_by_bucket):
        print(f"     {k:<8} {cost_by_bucket[k]/n_games:.4f}/game")

    # ---- pass 2: price a sample of unrestricted decisions ------------------
    print(f"\nUNRESTRICTED REGIME (|adm| > {ADAPTIVE_THRESHOLD}), lookahead tree")
    print(f"  {len(unrestricted_states)} such decisions; pricing a random "
          f"{a.unrestricted_sample} against the expert's own action")
    cfg = SolverConfig(max_guesses=6, seed=20260817, guess_pool="full")
    gt = GameTree(V.bundle.fb, TreeSearchConfig(
        depth=6, top_k=100, endgame_top_k=60, endgame_threshold=10,
        opening_guess="salet", max_guesses=6))
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(unrestricted_states))[:a.unrestricted_sample]
    diffs, by_b = [], defaultdict(list)
    t0 = time.perf_counter()
    for j, i in enumerate(idx):
        b, gi, na = unrestricted_states[int(i)]
        S = np.sort(V.answer_col[b.candidates].astype(np.int32))
        depth = 6 - b.turn + 1
        try:
            _, best_gi = gt.solve_state(S, depth)
            if best_gi < 0:
                continue
            c_model = _price(gt, S, gi, depth, V)
            c_best = _price(gt, S, int(best_gi), depth, V)
            dv = (c_model - c_best) / len(S)
            diffs.append(dv)
            by_b[bucket(na)].append(dv)
        except Exception:
            continue
        if (j + 1) % 20 == 0:
            print(f"    {j+1}/{len(idx)}  {time.perf_counter()-t0:.0f}s", flush=True)
    unres_per_game = None
    if diffs:
        mean_diff = float(np.mean(diffs))
        unres_per_game = mean_diff * len(unrestricted_states) / n_games
        print(f"\n  mean cost per unrestricted decision  {mean_diff:.4f} guesses")
        print(f"  decisions per game                   "
              f"{len(unrestricted_states)/n_games:.2f}")
        print(f"  -> extrapolated cost per game        {unres_per_game:.4f}")
        for k in sorted(by_b):
            print(f"     {k:<8} n={len(by_b[k]):3d}  mean {np.mean(by_b[k]):.4f}")

    # ---- the budget -------------------------------------------------------
    gap = BASELINE_MEAN - CLASSICAL_ENTROPY
    print("\n" + "=" * 72)
    print("BUDGET")
    print("=" * 72)
    print(f"  current mean                       {BASELINE_MEAN:.4f}")
    print(f"  classical entropy                  {CLASSICAL_ENTROPY:.4f}")
    print(f"  gap                                {gap:.4f}\n")
    print(f"  attributable to restricted picks   {restricted_cost/n_games:.4f}"
          f"   ({100*(restricted_cost/n_games)/gap:.0f}% of gap)")
    if unres_per_game is not None:
        print(f"  attributable to unrestricted picks {unres_per_game:.4f}"
              f"   ({100*unres_per_game/gap:.0f}% of gap)")
        acc = restricted_cost/n_games + unres_per_game
        print(f"  accounted for                      {acc:.4f}"
              f"   ({100*acc/gap:.0f}% of gap)")
        print(f"  unaccounted                        {gap-acc:.4f}")
        print("\n  Unaccounted mass is not error: this prices each decision")
        print("  against the best action AT THAT STATE, so it cannot capture")
        print("  the cost of being in a worse state to begin with. Phase 8's")
        print("  substitution measure captures that and is non-additive; this")
        print("  one is additive and misses it. They bound the answer from")
        print("  opposite sides.")

    out = {
        "cell": a.cell, "n_games": n_games, "total_decisions": total_dec,
        "decisions_per_game": round(total_dec/n_games, 3),
        "histogram": dict(hist),
        "restricted": dict(restricted),
        "restricted_cost_per_game": round(restricted_cost/n_games, 4),
        "restricted_cost_by_bucket_per_game": {
            k: round(v/n_games, 4) for k, v in cost_by_bucket.items()},
        "unrestricted_decisions": len(unrestricted_states),
        "unrestricted_sampled": len(diffs),
        "unrestricted_cost_per_decision": round(float(np.mean(diffs)), 4) if diffs else None,
        "unrestricted_cost_per_game": round(unres_per_game, 4) if unres_per_game else None,
        "baseline_mean": BASELINE_MEAN, "classical_entropy": CLASSICAL_ENTROPY,
        "gap": round(gap, 4),
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwritten to {a.out}")
    return 0


def _price(gt, S, g_idx, depth, V):
    """Total guesses over S from playing `g_idx`, then optimally.

    `GameTree.MT` is the transposed table, shape (n_answers, n_guesses), so the
    feedback of one guess against the surviving answers is a *column*.
    """
    row = np.asarray(gt.MT[S, g_idx])
    total = float(len(S))
    for code in np.unique(row):
        if int(code) == ALL_GREEN:
            continue
        child = S[row == code]
        c, _ = gt.solve_state(child, depth - 1)
        total += c
    return total


if __name__ == "__main__":
    raise SystemExit(main())
