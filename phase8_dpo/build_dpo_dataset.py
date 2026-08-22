"""
build_dpo_dataset.py - preference pairs for Phase 8.

WHY PREFERENCE DATA RATHER THAN MORE SFT
----------------------------------------
Under the Phase 7 adaptive decoder the model's job is no longer "emit a word".
It is "rank the admissible words". SFT on a single expert action never teaches
a ranking - it teaches one point. Phase 6 tested "more/better SFT data"
directly and moved games by exactly zero, so repeating that shape of experiment
is not the move.

A preference pair teaches the thing the decoder actually asks for:

    given this state, WHY is CRATE a better action than HOUSE?

WHAT A PAIR LOOKS LIKE
----------------------
    prompt    the same rendered state as every other phase
    chosen    a strong action
    rejected  a measurably worse action FROM THE SAME ACTION SPACE

The action space matters and is the easiest thing to get wrong. At deployment
the adaptive decoder restricts the model to feedback-consistent words only when
the admissible set is <= 20. So:

    |admissible| <= 20   pairs are drawn from the ADMISSIBLE set
                         (teaching "prefer a legal word over an inconsistent
                          one" would be worthless - the decoder already
                          guarantees it)
    |admissible| >  20   pairs are drawn from the full legal pool, because
                         that is genuinely the action space there

RANKING OBJECTIVE
-----------------
    cost(w) = E[remaining candidates after playing w] - 0.5 * (w is a candidate)

`E[remaining]` comes from `FeedbackMatrix.partition_stats`, computed exactly
from the precomputed feedback table. The candidate bonus is the standard
tie-break: at small candidate counts every guess has the same E[remaining], so
without it the objective cannot distinguish "a word that might win now" from
"a word that cannot". Lower cost is better.

A pair is only emitted when `cost(rejected) - cost(chosen) >= margin`. If the
alternative is not measurably worse there is nothing to teach, and asserting a
preference anyway would be teaching noise.

TEACHER
-------
`chosen` is the tree_salet action where tree search is affordable, and the
cost-argmin otherwise. Tree search costs ~1.4 s at 40 candidates and grows
quickly, so labelling thousands of mid-game states with it is not practical.
The substitution is cheap in quality terms: tree_salet scores 3.4212 over 2,315
games and the pure expected-remaining solver scores 3.4812 - a 0.06 gap,
against a model currently at 3.76. Every record records which teacher was used.

LEAKAGE
-------
The prompt is produced by the same `render_prompt` call as every previous
phase: no candidate list, no candidate count, no admissible count, no answer,
no scores. Those exist only in `meta`, for analysis and weighting.

    python build_dpo_dataset.py --out sft_package/data/dpo_midgame
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np

from wordle_solver import (load_artifacts, SolverConfig, make_solver,
                           feedback_code, feedback, ALL_GREEN)
from tree_search import TreeSearchConfig, TreeSearchSolver
from generate_trajectories import derive_constraints, render_prompt

MAX_GUESSES = 6
SEED = 20260817
FILTER_THRESHOLD = 20        # the Phase 7b adaptive decoder setting
CANDIDATE_BONUS = 0.5
TREE_MAX_CANDIDATES = 12     # above this, tree search is too slow to label with

# Buckets are keyed on |admissible|, because that is the decision regime the
# model actually faces under the adaptive decoder. |candidates| is a latent
# quantity it never sees.
BUCKETS = [("2-3", 2, 3), ("4-10", 4, 10), ("11-30", 11, 30),
           ("31-100", 31, 100), ("101-300", 101, 300), ("300+", 301, 10 ** 9)]

# Mid-game heavy, per the Phase 7 finding that the endgame is now handled by
# the decoder. k=1 is excluded entirely: the filter forces those states 38.9%
# of the time and collapses the rest to ~2 words.
DEFAULT_PER_BUCKET = {"2-3": 2000, "4-10": 4000, "11-30": 3000,
                      "31-100": 3000, "101-300": 2000, "300+": 1000}


def bucket_of(n_adm):
    for name, lo, hi in BUCKETS:
        if lo <= n_adm <= hi:
            return name
    return None


def opaque_id(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return "dpo-" + h[:14]


def load_train_answers(sft_dir):
    p = os.path.join(sft_dir, "eval", "train_answers.jsonl")
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l)["answer"].lower() for l in fh if l.strip()]


class Scorer:
    """cost(w) over a candidate set, from the precomputed feedback table."""

    def __init__(self, bundle):
        self.fb = bundle.fb
        self.V = bundle.vocab
        self.gi = {w: i for i, w in enumerate(self.V.guesses)}

    def cost(self, words, cand_idx):
        gidx = np.array([self.gi[w] for w in words], dtype=np.int32)
        st = self.fb.partition_stats(gidx, cand_idx)
        cand_words = {self.V.answers[i] for i in cand_idx}
        bonus = np.array([CANDIDATE_BONUS if w in cand_words else 0.0
                          for w in words])
        return st["expected_remaining"] - bonus, st


def make_pairs(answer, history, cand_idx, admissible, turn, sc, tree, ent,
               rng, alt_pool, margin_a, margin_b):
    """Build up to two preference pairs for one state."""
    n_adm = len(admissible)
    n_c = int(len(cand_idx))
    b = bucket_of(n_adm)
    if b is None:
        return []

    # The action space the deployed decoder would give the model here.
    if n_adm <= FILTER_THRESHOLD:
        space = list(admissible)
    else:
        space = list(sc.V.guesses)
    if len(space) < 3:
        return []

    # Sample alternatives to score. Always include the whole space when small.
    if len(space) <= alt_pool:
        alts = list(space)
    else:
        alts = [space[i] for i in rng.choice(len(space), alt_pool, replace=False)]

    # ---- chosen -----------------------------------------------------------
    # The tree expert picks from the FULL guess pool and frequently probes with
    # a word that cannot be the answer. That is correct play, but when the
    # deployed decoder restricts the model to admissible words the model can
    # never play such a word - so training it to prefer one puts probability
    # mass on an unreachable action. When the filter would be active, `chosen`
    # must therefore come from the admissible set too.
    restricted = n_adm <= FILTER_THRESHOLD
    chosen, teacher = None, None
    if n_c <= TREE_MAX_CANDIDATES:
        cand = tree.choose(cand_idx, turn)
        if not restricted or cand in admissible:
            chosen, teacher = cand, "tree_salet"
    if chosen is None:
        c, _ = sc.cost(alts, cand_idx)
        chosen = alts[int(np.argmin(c))]      # alts already respect the space
        teacher = "expected_remaining"
    if chosen not in alts:
        alts.append(chosen)

    costs, _ = sc.cost(alts, cand_idx)
    order = np.argsort(costs)
    ci = alts.index(chosen)
    c_chosen = float(costs[ci])

    out = []
    prompt = render_prompt(
        turn=turn, history=list(history),
        constraints=derive_constraints(history), n_candidates=n_c,
        guesses_remaining=MAX_GUESSES - turn + 1, max_guesses=MAX_GUESSES,
        candidates=None, show_candidate_count=False)   # never shown

    base = dict(bucket=b, n_admissible=n_adm, n_candidates=n_c, turn=turn,
                teacher=teacher, action_space=("admissible" if n_adm <= FILTER_THRESHOLD
                                               else "legal"),
                split="train")

    def emit(rej_i, kind, margin):
        gap = float(costs[rej_i]) - c_chosen
        if alts[rej_i] == chosen or gap < margin:
            return
        out.append({
            "id": opaque_id(answer, turn, kind,
                            "|".join(f"{g}:{p}" for g, p in history)),
            "prompt": prompt,
            "chosen": chosen.upper(),
            "rejected": alts[rej_i].upper(),
            "meta": dict(base, pair_type=kind, cost_gap=round(gap, 4),
                         cost_chosen=round(c_chosen, 4),
                         cost_rejected=round(float(costs[rej_i]), 4)),
        })

    # (1) expert vs a clearly poor action from the same space
    emit(int(order[-1]), "clear", margin_a)

    # (2) expert vs a COMPETITIVE alternative: the entropy solver's own choice
    #     when it differs, else the runner-up by cost. This is the fine
    #     discrimination the mid-game actually needs.
    comp = None
    if n_c <= 400:
        e = ent.choose(cand_idx, turn)
        # The entropy solver also picks from the FULL pool, so its choice can be
        # inadmissible for the same reason the tree expert's can. Under the
        # restricted action space it is not a legal alternative and must not
        # become `rejected` - the runner-up by cost is used instead.
        if restricted and e not in admissible:
            e = chosen
        if e != chosen:
            if e not in alts:
                alts.append(e)
                costs, _ = sc.cost(alts, cand_idx)
                order = np.argsort(costs)
                ci = alts.index(chosen)
                c_chosen = float(costs[ci])
            comp = alts.index(e)
    if comp is None:
        for i in order:
            if alts[int(i)] != chosen:
                comp = int(i)
                break
    if comp is not None:
        emit(comp, "competitive", margin_b)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--out", default="sft_package/data/dpo_midgame")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alt-pool", type=int, default=120)
    ap.add_argument("--margin-clear", type=float, default=2.0)
    ap.add_argument("--margin-competitive", type=float, default=0.25)
    ap.add_argument("--max-rollouts", type=int, default=200000)
    ap.add_argument("--limit-answers", type=int, default=0)
    for name, _, _ in BUCKETS:
        ap.add_argument(f"--n-{name}", type=int,
                        default=DEFAULT_PER_BUCKET.get(name, 0))
    a = ap.parse_args(argv)
    want = {name: getattr(a, f"n_{name}".replace("-", "_"))
            for name, _, _ in BUCKETS}

    print("=" * 70)
    print("DPO PREFERENCE PAIRS - mid-game focused")
    print("=" * 70)
    print(f"target pairs per bucket: {want}   total {sum(want.values())}")

    bundle = load_artifacts(a.artifacts, mmap=True)
    V = bundle.vocab
    LEGAL = [g.lower() for g in V.guesses]
    cfg = SolverConfig(max_guesses=MAX_GUESSES, seed=a.seed, guess_pool="full")
    tree = TreeSearchSolver(bundle.fb, cfg, TreeSearchConfig(
        depth=6, top_k=100, endgame_top_k=60, endgame_threshold=10,
        opening_guess="salet", max_guesses=MAX_GUESSES))
    ent = make_solver("entropy", bundle.fb, cfg, bundle.model); ent.reset()
    sc = Scorer(bundle)

    answers = load_train_answers(a.sft_dir)
    if a.limit_answers:
        answers = answers[:a.limit_answers]
    print(f"source answers: {len(answers)} TRAINING answers "
          f"(the 246 held-out are never used)")

    rng = np.random.default_rng(a.seed)
    got = Counter()
    rows, seen = [], set()
    t0 = time.perf_counter()
    rollouts = 0

    while any(got[b] < want[b] for b in want) and rollouts < a.max_rollouts:
        rollouts += 1
        ans = answers[rng.integers(len(answers))]
        cand_idx = np.arange(V.n_answers, dtype=np.int32)
        admissible = LEGAL
        history = []
        guess = "salet"
        for _ in range(5):
            code = feedback_code(guess, ans)
            if code == ALL_GREEN:
                break
            history.append((guess, feedback(guess, ans)))
            cand_idx = bundle.fb.filter_indices(cand_idx, guess, code)
            admissible = [w for w in admissible if feedback_code(guess, w) == code]
            turn = len(history) + 1
            if not len(cand_idx) or turn > MAX_GUESSES:
                break
            b = bucket_of(len(admissible))
            if b is not None and got[b] < want[b]:
                for r in make_pairs(ans, history, cand_idx, admissible, turn,
                                    sc, tree, ent, rng, a.alt_pool,
                                    a.margin_clear, a.margin_competitive):
                    k = (r["prompt"], r["rejected"])
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append(r)
                    got[b] += 1
                if got[b] % 500 == 0 and got[b]:
                    el = time.perf_counter() - t0
                    print(f"  {len(rows):6d} pairs  {el:5.0f}s  {dict(got)}",
                          flush=True)
            if len(cand_idx) == 1:
                break
            # imperfect continuation: what a fallible model actually does
            if rng.random() < 0.55:
                guess = V.answers[int(cand_idx[rng.integers(len(cand_idx))])]
            else:
                guess = LEGAL[rng.integers(len(LEGAL))]

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "train.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    el = time.perf_counter() - t0
    print(f"\nwrote {len(rows)} pairs to {path} in {el:.0f}s "
          f"({rollouts} rollouts)")
    print(f"  per bucket   : {dict(sorted(got.items()))}")
    print(f"  pair types   : {dict(Counter(r['meta']['pair_type'] for r in rows))}")
    print(f"  teacher      : {dict(Counter(r['meta']['teacher'] for r in rows))}")
    print(f"  action space : {dict(Counter(r['meta']['action_space'] for r in rows))}")
    gaps = np.array([r["meta"]["cost_gap"] for r in rows])
    print(f"  cost gap     : median {np.median(gaps):.2f}  "
          f"p10 {np.percentile(gaps, 10):.2f}  p90 {np.percentile(gaps, 90):.2f}")
    print(f"  distinct prompts: {len(set(r['prompt'] for r in rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
