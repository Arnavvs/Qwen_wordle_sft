"""build_dpo_v3_dataset.py — corrected Phase-8 preference data.

This is a replacement for `phase8_dpo/build_dpo_dataset.py`, not a patch to its
output. The audit (`audit_legacy_dpo.py`, findings in `AUDIT.md`) found the v1
labels were *directionally* sound but that essentially every other property of
the dataset was wrong: the pair-difficulty labels did not track real value, one
state contributed 177 rows, 64% of pairs were against a word that contradicts
feedback the model can already see, 40% of the budget went to the two buckets
the counterfactual analysis measured as worth <=29% and -124%, and there was no
validation split at all.

WHAT THIS GENERATOR DOES DIFFERENTLY
------------------------------------
1. **Exact values, not a one-ply proxy.**  In the regime that matters the label
   comes from `AdaptiveTree`: the finite game the adaptive decoder actually
   plays, solved to full remaining depth, in exact integer arithmetic. No
   horizon truncation, no `- 0.5 * is_candidate` constant.

2. **"Competitive" means competitive.**  v1 defined difficulty by a threshold on
   a proxy gap, which turned out to be uncorrelated with true value: its
   `competitive` pairs have a median *true* gap of a full guess. Here the
   default `rejected` is drawn from the **best strictly-worse tier** — the
   closest real mistake the state admits — so the pair is as hard as the
   position allows, by construction rather than by threshold.

3. **Exactly-tied actions are never paired against each other.**  95.3% of
   reachable 2-10 states have two or more exactly-optimal admissible actions.
   Asserting a preference between two equally good moves is teaching noise, and
   it is the single easiest way to make a preference set actively harmful.

4. **One pair per distinct state.**  v1's dedup key was `(prompt, rejected)`
   with `rejected` freshly randomised per visit, so revisiting a state minted
   new rows indefinitely.

5. **Budget follows the measured headroom.**  75% of pairs in the 2-10 bucket
   that holds 74.7% of the remaining gap; the 100+ bucket, measured at -124%,
   is kept only for coverage.

6. **A real validation split**, answer-partitioned and state-disjoint, so
   preference accuracy can be reported on held-out states. v1 reported training
   accuracy only — there was no second file.

LEAKAGE
-------
The prompt is the same `render_prompt` call as every earlier phase: no answer,
no candidate list, no candidate count, no admissible count, no score. Beyond
that, training rows never name a held-out answer as `chosen` or `rejected`
(v1 had 694 rows whose preferred action was a held-out answer); the drop is
counted and the verifier checks it did not skew the distribution.

    python phase8_dpo_v3/build_dpo_v3_dataset.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=project root)

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from dpo_core import (ADAPTIVE_THRESHOLD, BUCKETS, FAILURE_COST, MAX_GUESSES,
                      AdaptiveTree, Board, OnePlyScorer, Vocab, bucket_of,
                      load_answer_splits, sha)
from wordle_solver import ALL_GREEN

SEED = 20260821
FORMAT_VERSION = 3
OPENER = "salet"          # the SFT policy reproduces its opener 100% of the time

DEFAULT_TRAIN = {"2-10": 4500, "11-100": 1100, "100+": 400}
DEFAULT_VAL = {"2-10": 600, "11-100": 150, "100+": 50}

# Share of pairs whose `rejected` is drawn from the closest strictly-worse tier
# rather than from a clearly-worse action. The remainder keep some easy signal
# for optimisation stability without being allowed to dominate the gradient.
DEFAULT_COMPETITIVE_FRAC = 0.75

# The two regimes score actions in different units — mean further guesses under
# the exact tree, expected remaining candidates under the one-ply proxy — so a
# single absolute gap threshold cannot serve both. A "1.0" that means one whole
# extra guess in the endgame means a 1.5% difference at 70 candidates. Bands are
# therefore declared per regime, and relative where the units are not guesses.
#
# EXACT regime: any positive gap is a real difference (costs are integers over a
# common denominator), so the closest strictly-worse tier is always meaningful
# and needs no floor. Only the contrastive class needs bounds, and it gets an
# upper one too — a pair nobody could get wrong teaches nothing, which is how v1
# ended up with gaps of 215.
EXACT_CONTRASTIVE_BAND = (1.0, 2.5)          # mean further guesses

# PROXY regime: one-ply E[remaining] is not trustworthy at fine resolution — the
# audit shows its ordering of near-ties does not survive contact with an exact
# value function. So even the "competitive" class takes a floor here, expressed
# as a fraction of the best action's value so it is scale-free.
PROXY_COMPETITIVE_BAND = (0.05, 0.20)        # relative to best value
PROXY_CONTRASTIVE_BAND = (0.30, 1.00)


# =============================================================================
# rollout policy
# =============================================================================

class Rollout:
    """Reachable, on-distribution game continuations.

    Every state is reachable by construction because it is produced by actually
    playing a game. The harder requirement is that it be *plausible* — a state
    the deployed system would really visit. v1 chose a uniformly random legal
    word 45% of the time, which is nothing like the deployed policy and leaves
    the admissible set large: 23% of its rows sat at turn 2, in the bucket its
    own headroom analysis had measured at -124%.

    Here the continuation mirrors the deployed decoder instead:

      turn 1     always SALET
      turn 2     the entropy expert's probe most of the time (the SFT policy
                 agrees with its expert's turn-2 choice 90% of the time), else
                 an admissible word
      turn 3+    restricted regime -> an admissible word, because that is
                 literally the decoder's action space there
                 unrestricted      -> a candidate or an admissible word, with a
                 small tail of arbitrary legal probes for coverage
    """

    def __init__(self, V, rng, expert_frac=0.70, cand_frac=0.50,
                 adm_frac=0.40):
        self.V = V
        self.rng = rng
        self.expert_frac = expert_frac
        self.cand_frac = cand_frac
        self.adm_frac = adm_frac
        self._turn2 = {}          # only ~150 distinct turn-2 states exist

    def _entropy_pick(self, board):
        """The entropy solver's choice, cached by state key."""
        key = board.state_key()
        hit = self._turn2.get(key)
        if hit is None:
            cols = self.V.answer_col[board.candidates]
            stats = self.V.bundle.fb.partition_stats(
                np.arange(self.V.n, dtype=np.int32), cols)
            hit = int(np.argmax(stats["entropy"]))
            self._turn2[key] = hit
        return hit

    def next_guess(self, board):
        adm, cand = board.admissible, board.candidates
        if board.turn == 2 and self.rng.random() < self.expert_frac:
            return self._entropy_pick(board)
        if board.restricted:
            return int(adm[self.rng.integers(len(adm))])
        u = self.rng.random()
        if u < self.cand_frac:
            return int(cand[self.rng.integers(len(cand))])
        if u < self.cand_frac + self.adm_frac:
            return int(adm[self.rng.integers(len(adm))])
        return int(self.rng.integers(self.V.n))


# =============================================================================
# labelling
# =============================================================================

def score_actions(board, tree, oneply):
    """Value every action the decoder would offer on this board.

    Returns `(values, order, meta)` where `values` maps a guess index to a mean
    number of further guesses (lower is better).

    Restricted boards get the exact adaptive-decoder tree. Unrestricted boards
    get an exhaustive one-ply E[remaining] over all 12,972 legal words, which is
    a proxy and is labelled as one in `meta["score_family"]`.
    """
    n = len(board.candidates)
    if board.restricted:
        totals = tree.action_costs(board)          # exact integers
        values = {g: t / n for g, t in totals.items()}
        return values, totals, {
            "score_family": "exact_adaptive_decoder_tree",
            "score_units": "mean_further_guesses",
            "tree_depth": board.guesses_left,
            "horizon": "full_remaining_game",
            "exact": True,
            "failure_cost": FAILURE_COST,
            "competitive_rule": "closest_strictly_worse_tier",
            "contrastive_band_absolute": list(EXACT_CONTRASTIVE_BAND),
        }
    # E[remaining] = sum(bucket^2)/n. Recovering the integer numerator makes
    # ties *exact* instead of float-approximate: without this, two
    # mathematically identical actions can differ by ~1e-15 and get emitted as
    # the "closest competitive pair" in the state, which is pure label noise.
    sumsq = np.rint(oneply.action_scores(board) * n).astype(np.int64)
    totals = {i: int(v) for i, v in enumerate(sumsq)}
    values = {i: v / n for i, v in totals.items()}
    return values, totals, {
        "score_family": "full_legal_pool_one_ply_expected_remaining",
        "score_units": "expected_remaining_candidates",
        "tree_depth": 1,
        "horizon": "one_ply_proxy",
        "exact": False,
        "failure_cost": None,
        "competitive_rule": "relative_band_over_best",
        "competitive_band_relative": list(PROXY_COMPETITIVE_BAND),
        "contrastive_band_relative": list(PROXY_CONTRASTIVE_BAND),
    }


def build_pair(board, values, want_competitive, rng, exact):
    """Pick `chosen` and `rejected` for one board, or explain why we cannot.

    `chosen` is always an exactly-optimal action. `rejected` is always strictly
    worse — never a tied optimum. Pairing two equally good moves is the one
    thing a preference set must not do: it asserts an ordering that does not
    exist, and 95.3% of reachable 2-10 states have a tied optimum, so it is the
    default outcome of any generator that just takes the top two by score.
    """
    items = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    best = items[0][1]
    tol = 1e-12
    optimal = [g for g, v in items if v <= best + tol]
    worse = [(g, v) for g, v in items if v > best + tol]
    if not worse:
        return None, "all_actions_optimal"

    # Spread `chosen` over the tied optima rather than always taking the
    # alphabetically first, which would bake a lexical bias into the target.
    pick = int(sha("chosen", board.state_key())[:12], 16) % len(optimal)
    chosen = optimal[pick]

    if exact:
        lo, hi = EXACT_CONTRASTIVE_BAND
        if want_competitive:
            # Any positive gap is real here, so the closest strictly-worse tier
            # is both the hardest and a genuinely correct discrimination.
            edge = worse[0][1] - best
            tier = [g for g, v in worse if v <= best + edge + tol]
            return (chosen, tier[int(rng.integers(len(tier)))], "competitive",
                    optimal, len(items), best), None
        tier = [g for g, v in worse if lo <= v - best <= hi]
        if not tier:
            return None, "no_contrastive_action"
        return (chosen, tier[int(rng.integers(len(tier)))], "contrastive",
                optimal, len(items), best), None

    # Proxy regime: bands are relative to the best value, because the units are
    # candidates and their scale varies by two orders of magnitude across the
    # bucket. A floor also keeps the pair above the resolution at which a
    # one-ply score is worth believing.
    lo, hi = (PROXY_COMPETITIVE_BAND if want_competitive
              else PROXY_CONTRASTIVE_BAND)
    if best <= 0:
        return None, "degenerate_best_value"
    tier = [g for g, v in worse if lo <= (v - best) / best <= hi]
    if not tier:
        return None, ("no_competitive_action" if want_competitive
                      else "no_contrastive_action")
    kind = "competitive" if want_competitive else "contrastive"
    return (chosen, tier[int(rng.integers(len(tier)))], kind, optimal,
            len(items), best), None


def make_record(V, board, values, extra, chosen, rejected, kind, optimal,
                n_actions, best, split, source_answer, totals):
    n_cand = len(board.candidates)
    gap = values[rejected] - values[chosen]
    ranks = sorted(values.values())
    key = board.state_key()
    cw = set(board.candidates.tolist())
    adm = set(board.admissible.tolist())
    rec = {
        "id": "dpo3-" + sha(split, key, V.words[chosen], V.words[rejected])[:16],
        "prompt": board.prompt(),
        "chosen": V.words[chosen].upper(),
        "rejected": V.words[rejected].upper(),
        "meta": {
            "format_version": FORMAT_VERSION,
            "split": split,
            # ---- state identity, for dedup and leakage auditing -------------
            "state_key": key,
            "state_hash": sha("state", key)[:20],
            "source_answer_hash": sha("phase8v3-source", source_answer)[:16],
            "turn": board.turn,
            "guesses_left": board.guesses_left,
            "reachable": True,
            # ---- decision regime -------------------------------------------
            "n_candidates": n_cand,
            "n_admissible": int(len(board.admissible)),
            "bucket": bucket_of(len(board.admissible)),
            "adaptive_threshold": ADAPTIVE_THRESHOLD,
            "action_space": board.action_space,
            "n_actions": int(n_actions),
            # ---- why this pair was labelled the way it was -----------------
            "pair_type": kind,
            "chosen_value": round(float(values[chosen]), 8),
            "rejected_value": round(float(values[rejected]), 8),
            "preference_gap": round(float(gap), 8),
            # The two regimes report gaps in different units, so anything that
            # pools rows across them must use the relative form.
            "preference_gap_relative": round(float(gap / values[chosen]), 8)
                                        if values[chosen] > 0 else None,
            "score_direction": "lower_value_is_better",
            "best_value": round(float(best), 8),
            "n_optimal_actions": int(len(optimal)),
            "chosen_is_optimal": True,
            "chosen_rank": 1,
            "rejected_rank": int(1 + sum(1 for v in ranks if v < values[rejected] - 1e-12)),
            # Both value families are integers over a common denominator of
            # n_candidates, so this is the smallest real difference that can
            # exist here — the yardstick for calling a gap "close".
            "min_possible_positive_gap": round(float(1.0 / n_cand), 8),
            "chosen_total_cost": int(totals[chosen]) if totals is not None else None,
            "rejected_total_cost": int(totals[rejected]) if totals is not None else None,
            # ---- action validity -------------------------------------------
            "chosen_is_admissible": chosen in adm,
            "rejected_is_admissible": rejected in adm,
            "chosen_is_candidate": chosen in cw,
            "rejected_is_candidate": rejected in cw,
            # ---- prompt hygiene assertions, re-checked by the verifier ------
            "answer_in_prompt": False,
            "candidate_info_in_prompt": False,
            "score_info_in_prompt": False,
        },
    }
    rec["meta"].update(extra)
    return rec


# =============================================================================
# harvest
# =============================================================================

def harvest(V, source_answers, quotas, split, seed, args, holdout,
            forbidden_states=(), log=print):
    rng = np.random.default_rng(seed)
    tree = AdaptiveTree(V)
    oneply = OnePlyScorer(V)
    roll = Rollout(V, rng)
    opener = V.index[OPENER]

    rows = []
    got = Counter()
    comp = Counter()          # competitive rows so far, per bucket
    reject = Counter()
    seen_states = set(forbidden_states)
    per_answer = Counter()
    t0 = time.perf_counter()
    rollouts = 0

    while any(got[b] < quotas[b] for b in quotas) and rollouts < args.max_rollouts:
        rollouts += 1
        answer = source_answers[int(rng.integers(len(source_answers)))]
        if per_answer[answer] >= args.max_per_answer:
            reject["answer_cap"] += 1
            continue
        ai = V.index[answer]
        board = Board(V)
        g = opener
        for _ in range(MAX_GUESSES - 1):
            code = V.code_against(g, ai)
            if code == ALL_GREEN:
                break
            board = board.play(g, code)
            if len(board.candidates) == 0:
                raise AssertionError("reachable rollout emptied its candidate set")
            emitted = _try_state(V, board, answer, split, quotas, got, comp,
                                 rows, seen_states, per_answer, reject, tree,
                                 oneply, rng, args, holdout)
            if emitted or len(board.candidates) == 1 or board.turn > MAX_GUESSES:
                break
            g = roll.next_guess(board)

        if rollouts % 2000 == 0:
            log(f"    {split}: {rollouts:7d} rollouts  {len(rows):5d} rows  "
                f"{dict(sorted(got.items()))}  {time.perf_counter()-t0:.0f}s")

    return rows, got, reject, rollouts, tree.nodes


def _try_state(V, board, answer, split, quotas, got, comp, rows, seen_states,
               per_answer, reject, tree, oneply, rng, args, holdout):
    """Attempt one record from `board`. Returns True if a row was emitted."""
    n_adm = len(board.admissible)
    b = bucket_of(n_adm)
    if b is None or got[b] >= quotas[b]:
        return False
    if board.turn > MAX_GUESSES or len(board.candidates) < 2:
        reject["degenerate_state"] += 1
        return False
    key = board.state_key()
    if key in seen_states:
        reject["duplicate_state"] += 1
        return False
    if args.strict_candidate_partition and holdout:
        cw = {V.words[int(i)] for i in board.candidates}
        if cw & holdout:
            reject["candidate_set_touches_holdout"] += 1
            return False

    values, totals, extra = score_actions(board, tree, oneply)
    # Aim at the configured mix per bucket without letting the easy class
    # overshoot: ask for a competitive pair whenever we are behind the target.
    want_comp = comp[b] < args.competitive_frac * (got[b] + 1)
    built, why = build_pair(board, values, want_comp, rng, extra["exact"])
    if built is None:
        # A state with no action in the requested band can still yield one in
        # the other; retry once before giving up on the state entirely.
        built, why2 = build_pair(board, values, not want_comp, rng, extra["exact"])
        if built is None:
            reject[why2 or why] += 1
            return False
    chosen, rejected, kind, optimal, n_actions, best = built

    if holdout and args.block_holdout_actions:
        if V.words[chosen] in holdout or V.words[rejected] in holdout:
            reject["action_is_holdout_answer"] += 1
            return False

    rec = make_record(V, board, values, extra, chosen, rejected, kind, optimal,
                      n_actions, best, split, answer, totals)
    rows.append(rec)
    seen_states.add(key)
    per_answer[answer] += 1
    got[b] += 1
    if kind == "competitive":
        comp[b] += 1
    return True


# =============================================================================
# main
# =============================================================================

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def _dist(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"n": int(len(v)), "median": round(float(np.median(v)), 4),
            "mean": round(float(v.mean()), 4),
            "p10": round(float(np.percentile(v, 10)), 4),
            "p90": round(float(np.percentile(v, 90)), 4),
            "max": round(float(v.max()), 4)}


def summarise(rows):
    """Per-split statistics.

    Absolute gaps are reported *per score family only*. Pooling them would
    average mean-further-guesses against expected-remaining-candidates and
    produce a number with no unit; the relative gap is the one that is
    comparable across regimes.
    """
    if not rows:
        return {}
    out = {
        "rows": len(rows),
        "distinct_states": len({r["meta"]["state_key"] for r in rows}),
        "distinct_prompts": len({r["prompt"] for r in rows}),
        "distinct_ids": len({r["id"] for r in rows}),
        "max_rows_per_state": max(Counter(r["meta"]["state_key"] for r in rows).values()),
        "bucket": dict(sorted(Counter(r["meta"]["bucket"] for r in rows).items())),
        "pair_type": dict(Counter(r["meta"]["pair_type"] for r in rows)),
        "action_space": dict(Counter(r["meta"]["action_space"] for r in rows)),
        "score_family": dict(Counter(r["meta"]["score_family"] for r in rows)),
        "turn": dict(sorted(Counter(r["meta"]["turn"] for r in rows).items())),
        "relative_gap": _dist([r["meta"]["preference_gap_relative"] for r in rows]),
        "chosen_is_candidate_frac": round(
            float(np.mean([r["meta"]["chosen_is_candidate"] for r in rows])), 4),
        "rejected_is_candidate_frac": round(
            float(np.mean([r["meta"]["rejected_is_candidate"] for r in rows])), 4),
        "states_with_tied_optimum_frac": round(
            float(np.mean([r["meta"]["n_optimal_actions"] > 1 for r in rows])), 4),
        "mean_actions_per_state": round(
            float(np.mean([r["meta"]["n_actions"] for r in rows])), 2),
    }
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["meta"]["score_family"]][r["meta"]["pair_type"]].append(
            r["meta"]["preference_gap"])
    out["absolute_gap_by_family"] = {
        fam: {"units": ("mean_further_guesses"
                        if fam.startswith("exact") else "expected_remaining_candidates"),
              **{kind: _dist(g) for kind, g in kinds.items()}}
        for fam, kinds in by.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--out", default="sft_package/data/dpo_v3")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-rollouts", type=int, default=400000)
    ap.add_argument("--max-per-answer", type=int, default=8,
                    help="cap rows sourced from any single answer")
    ap.add_argument("--competitive-frac", type=float,
                    default=DEFAULT_COMPETITIVE_FRAC)
    ap.add_argument("--block-holdout-actions", type=int, default=1,
                    help="1 = training pairs may never name a held-out answer")
    ap.add_argument("--strict-candidate-partition", action="store_true",
                    help="also drop training states whose candidate set "
                         "contains any held-out answer (much stricter, and "
                         "unsatisfiable for large states)")
    for b in BUCKETS:
        f = b.replace("-", "_").replace("+", "plus")
        ap.add_argument(f"--train-{b}", dest=f"train_{f}", type=int,
                        default=DEFAULT_TRAIN[b])
        ap.add_argument(f"--val-{b}", dest=f"val_{f}", type=int,
                        default=DEFAULT_VAL[b])
    a = ap.parse_args(argv)

    def q(prefix, table):
        return {b: getattr(a, f"{prefix}_{b.replace('-', '_').replace('+', 'plus')}")
                for b in table}
    train_q, val_q = q("train", DEFAULT_TRAIN), q("val", DEFAULT_VAL)

    print("=" * 72)
    print("PHASE-8 v3 DPO DATA — exact adaptive-decoder values, competitive pairs")
    print("=" * 72)
    V = Vocab(a.artifacts)
    train_answers, val_answers = load_answer_splits(a.sft_dir)
    holdout = set(val_answers)
    print(f"seed={a.seed}  adaptive_threshold={ADAPTIVE_THRESHOLD}  "
          f"failure_cost={FAILURE_COST}")
    print(f"source answers: train={len(train_answers)} val={len(val_answers)}")
    print(f"quotas: train={train_q}  val={val_q}")
    print(f"competitive_frac={a.competitive_frac}  "
          f"block_holdout_actions={bool(a.block_holdout_actions)}  "
          f"strict_candidate_partition={a.strict_candidate_partition}")

    t0 = time.perf_counter()
    print("\n-- train --")
    train, tgot, trej, troll, tnodes = harvest(
        V, train_answers, train_q, "train", a.seed, a, holdout)
    print("\n-- validation --")
    # Validation states come from the 246 held-out answers, so preference
    # accuracy is measured on states the gameplay benchmark actually visits.
    # Held-out actions are permitted here: this split is never trained on.
    va = argparse.Namespace(**vars(a))
    va.block_holdout_actions = 0
    va.strict_candidate_partition = False
    val, vgot, vrej, vroll, vnodes = harvest(
        V, val_answers, val_q, "validation", a.seed + 1, va, None,
        forbidden_states={r["meta"]["state_key"] for r in train})

    # -- hard split guarantees ------------------------------------------------
    tk = {r["meta"]["state_key"] for r in train}
    vk = {r["meta"]["state_key"] for r in val}
    tp = {r["prompt"] for r in train}
    vp = {r["prompt"] for r in val}
    assert not tk & vk, "train/validation state leakage"
    assert not tp & vp, "train/validation prompt leakage"
    assert len(tk) == len(train), "duplicate state in train"
    assert len(vk) == len(val), "duplicate state in validation"
    if a.block_holdout_actions:
        bad = [r for r in train
               if r["chosen"].lower() in holdout or r["rejected"].lower() in holdout]
        assert not bad, f"{len(bad)} training rows name a held-out answer"

    write_jsonl(os.path.join(a.out, "train.jsonl"), train)
    write_jsonl(os.path.join(a.out, "validation.jsonl"), val)

    manifest = {
        "format_version": FORMAT_VERSION,
        "generated_by": "phase8_dpo_v3/build_dpo_v3_dataset.py",
        "seed": a.seed,
        "adaptive_threshold": ADAPTIVE_THRESHOLD,
        "failure_cost": FAILURE_COST,
        "opener": OPENER,
        "config": {
            "competitive_frac": a.competitive_frac,
            "exact_contrastive_band_absolute": list(EXACT_CONTRASTIVE_BAND),
            "proxy_competitive_band_relative": list(PROXY_COMPETITIVE_BAND),
            "proxy_contrastive_band_relative": list(PROXY_CONTRASTIVE_BAND),
            "max_per_answer": a.max_per_answer,
            "block_holdout_actions": bool(a.block_holdout_actions),
            "strict_candidate_partition": a.strict_candidate_partition,
        },
        "requested": {"train": train_q, "validation": val_q},
        "produced": {"train": dict(tgot), "validation": dict(vgot)},
        "rollouts": {"train": troll, "validation": vroll},
        "tree_nodes": {"train": tnodes, "validation": vnodes},
        "rejections": {"train": dict(trej), "validation": dict(vrej)},
        "source_answers": {"train": len(train_answers), "validation": len(val_answers)},
        "split_checks": {"state_overlap": 0, "prompt_overlap": 0},
        "summary": {"train": summarise(train), "validation": summarise(val)},
        "label_policy": {
            "restricted_regime": "|admissible| <= 20: exact adaptive-decoder "
                                 "tree, full remaining depth, integer costs",
            "unrestricted_regime": "|admissible| > 20: exhaustive one-ply "
                                   "E[remaining] over all legal words (proxy)",
            "chosen": "an exactly-optimal action, spread over tied optima by a "
                      "state-keyed hash",
            "rejected_competitive": "uniform over the best strictly-worse tier",
            "rejected_contrastive": "exact regime: %s mean guesses worse; "
                                    "proxy regime: %s relative to best"
                                    % (EXACT_CONTRASTIVE_BAND, PROXY_CONTRASTIVE_BAND),
            "never": "two exactly-tied optimal actions paired against each other",
            "pairs_per_state": 1,
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"\nwrote train={len(train)} validation={len(val)} -> {a.out} "
          f"in {manifest['seconds']:.0f}s")
    for name, s in (("train", manifest["summary"]["train"]),
                    ("validation", manifest["summary"]["validation"])):
        print(f"  {name:11s} buckets={s.get('bucket')} types={s.get('pair_type')}")
        print(f"              relative gap median={s['relative_gap']['median']} "
              f"p90={s['relative_gap']['p90']}  "
              f"tied-optimum states={s['states_with_tied_optimum_frac']}")
        for fam, d in s["absolute_gap_by_family"].items():
            for kind in ("competitive", "contrastive"):
                if kind in d:
                    print(f"                {fam[:28]:28s} {kind:12s} "
                          f"n={d[kind]['n']:5d} median={d[kind]['median']:8.4f} "
                          f"max={d[kind]['max']:9.4f} [{d['units']}]")
    if trej:
        print("  train rejections:", dict(trej.most_common(8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
