"""Build a state-deduplicated DPO set aligned to the adaptive decoder.

The original Phase-8 generator sampled alternatives from a different random
subset on every visit to a state.  This generator deliberately emits at most
one pair per state.  It uses only reachable feedback histories, and uses the
same action set as deployment:

  * |A| <= 20: every action is in the feedback-consistent legal-word set A.
  * |A| > 20: every legal Wordle guess is available.

For the high-value restricted regime (2--20 admissible words), labels come
from an exact finite-horizon adaptive hard-mode tree.  Its successor action
sets are refined after every possible feedback result, so this is a value over
the decoder's actual future decision space, rather than a full-pool tree action
projected into hard mode.  For the larger unrestricted regimes, the exact
one-ply tree value E[remaining candidates] is used over the *entire* legal
pool.  Those labels are explicitly marked as horizon-1 in metadata.

All label-only information lives in meta.  The model-facing prompt contains
neither an answer, candidate set/count, admissible set/count, nor a score.

Run from project root:
  .\\.conda\\python.exe phase8_dpo_corrected\\build_corrected_dpo_dataset.py
"""
import _paths  # noqa: F401 -- puts core/ on sys.path

import argparse
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np

from wordle_solver import ALL_GREEN, feedback, feedback_code, load_artifacts
from generate_trajectories import derive_constraints, render_prompt


SEED = 20260821
MAX_GUESSES = 6
ADAPTIVE_THRESHOLD = 20
FAILURE_COST = 7.0
ADAPTIVE_TREE_DEPTH = 3

# The dataset is intentionally small. DPO should not be allowed to overwhelm
# the Phase-7 behavioural-cloning policy with many variants of an easy state.
DEFAULT_TRAIN = {"2-10": 900, "11-100": 220, "100+": 80}
DEFAULT_VAL = {"2-10": 80, "11-100": 20, "100+": 8}


def bucket_of(n_admissible):
    if 2 <= n_admissible <= 10:
        return "2-10"
    if 11 <= n_admissible <= 100:
        return "11-100"
    if n_admissible >= 101:
        return "100+"
    return None


def code_from_pattern(pattern):
    return sum({"B": 0, "Y": 1, "G": 2}[x] * (3 ** i)
               for i, x in enumerate(pattern))


def stable_hash(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def state_key(history):
    return "|".join(f"{g}:{p}" for g, p in history)


def answer_hash(answer):
    return stable_hash("phase8-v2-source", answer)[:16]


class AdaptiveTree:
    """Exact expected game-score evaluator for a restricted hard-mode state.

    S is an answer-index array. A is the current feedback-consistent legal
    action set.  The recurrence is the same total-cost tree recurrence used by
    ``core/tree_search.py``, but it additionally carries A because adaptive
    decoding changes the legal action set after each feedback branch.
    """

    def __init__(self, bundle):
        self.bundle = bundle
        self.vocab = bundle.vocab
        self.fb = bundle.fb
        self.gi = {w: i for i, w in enumerate(self.vocab.guesses)}
        self._memo = {}
        self.nodes = 0

    def refine_actions(self, actions, guess, code):
        return tuple(w for w in actions if feedback_code(guess, w) == code)

    def value(self, S, actions, remaining):
        S = np.asarray(S, dtype=np.int32)
        actions = tuple(actions)
        n = len(S)
        if n == 1:
            # The unique candidate is necessarily a legal consistent word.
            return 1.0
        if remaining <= 0 or not actions:
            return FAILURE_COST * n
        key = (S.tobytes(), actions, remaining)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.nodes += 1
        best = float("inf")
        # Candidate words first give a good upper bound, as in tree_search.
        candidate_words = {self.vocab.answers[int(i)] for i in S}
        ordered = sorted(actions, key=lambda w: (w not in candidate_words, w))
        for guess in ordered:
            gi = self.gi[guess]
            codes = self.fb.M[gi, S]
            total = float(n)  # one guess for every still-possible answer
            for code in np.unique(codes):
                if int(code) == ALL_GREEN:
                    continue
                child = S[codes == code]
                child_actions = self.refine_actions(actions, guess, int(code))
                total += self.value(child, child_actions, remaining - 1)
                if total >= best:
                    break
            if total < best:
                best = total
        self._memo[key] = best
        return best

    def action_values(self, S, actions, remaining):
        """Expected final additional-game score for every current action."""
        S = np.asarray(S, dtype=np.int32)
        n = len(S)
        out = {}
        for guess in actions:
            gi = self.gi[guess]
            codes = self.fb.M[gi, S]
            total = float(n)
            for code in np.unique(codes):
                if int(code) == ALL_GREEN:
                    continue
                child = S[codes == code]
                child_actions = self.refine_actions(actions, guess, int(code))
                total += self.value(child, child_actions, remaining - 1)
            out[guess] = total / n
        return out


def full_pool_horizon_one(bundle, cand_idx):
    """Exact depth-one tree ranking over every legal action (no sampling)."""
    rows = np.arange(bundle.vocab.n_guesses, dtype=np.int32)
    stats = bundle.fb.partition_stats(rows, cand_idx)
    return {bundle.vocab.guesses[int(i)]: float(v)
            for i, v in zip(rows, stats["expected_remaining"])}


def choose_close_pair(values, min_gap, max_gap, salt):
    """Choose a reproducible close, strictly ordered pair from action values."""
    ranked = sorted(values.items(), key=lambda x: (x[1], x[0]))
    if len(ranked) < 2:
        return None
    chosen, chosen_value = ranked[0]
    eligible = [(word, value) for word, value in ranked[1:]
                if min_gap <= value - chosen_value <= max_gap]
    if not eligible:
        return None
    # Avoid repeatedly selecting only the runner-up while remaining wholly
    # deterministic. Every selected alternative is still close in value.
    h = int(stable_hash("pair", salt)[:12], 16)
    rejected, rejected_value = eligible[h % len(eligible)]
    return chosen, rejected, chosen_value, rejected_value


def prompt_for(history, turn):
    return render_prompt(
        turn=turn,
        history=list(history),
        constraints=derive_constraints(history),
        n_candidates=0,                 # API parity only; never rendered
        guesses_remaining=MAX_GUESSES - turn + 1,
        max_guesses=MAX_GUESSES,
        candidates=None,
        show_candidate_count=False,
    )


def make_record(bundle, tree, history, cand_idx, admissible, source_answer,
                split, strict_partition):
    n_adm = len(admissible)
    bucket = bucket_of(n_adm)
    if bucket is None:
        return None
    turn = len(history) + 1
    if turn > MAX_GUESSES or len(cand_idx) < 2:
        return None
    action_space = "admissible" if n_adm <= ADAPTIVE_THRESHOLD else "legal"
    if action_space == "admissible":
        # High-value regime: score every action under the future adaptive tree.
        tree_depth = min(ADAPTIVE_TREE_DEPTH, MAX_GUESSES - turn + 1)
        values = tree.action_values(cand_idx, tuple(admissible), tree_depth)
        score_family = "adaptive_hardmode_tree_depth_3"
        min_gap, max_gap = 0.01, 0.75
    else:
        # Current decoder action space really is the whole legal vocabulary.
        # This is a correct depth-one tree value, evaluated exhaustively rather
        # than over the old generator's random 120-word alternative sample.
        values = full_pool_horizon_one(bundle, cand_idx)
        score_family = "full_legal_tree_horizon_1_expected_remaining"
        min_gap, max_gap = 0.01, (0.75 if bucket == "11-100" else 1.25)
    key = state_key(history)
    pair = choose_close_pair(values, min_gap, max_gap, key)
    if pair is None:
        return None
    chosen, rejected, cscore, rscore = pair
    # Explicit guards: a record must be legal under the *deployed* decoder.
    legal = set(bundle.vocab.guesses)
    assert chosen in legal and rejected in legal and chosen != rejected
    if action_space == "admissible":
        assert chosen in admissible and rejected in admissible
    assert feedback_code(history[-1][0], source_answer) == code_from_pattern(history[-1][1])
    prompt = prompt_for(history, turn)
    candidate_words = tuple(bundle.vocab.answers[int(i)] for i in cand_idx)
    return {
        "id": "dpo2-" + stable_hash(split, key, chosen, rejected)[:16],
        "prompt": prompt,
        "chosen": chosen.upper(),
        "rejected": rejected.upper(),
        "meta": {
            "format_version": 2,
            "split": split,
            "source_answer_hash": answer_hash(source_answer),
            "state_hash": stable_hash("state", key)[:20],
            "history_hash": stable_hash("history", key)[:20],
            "reachable": True,
            "turn": turn,
            "candidate_count": int(len(cand_idx)),
            "candidate_set_hash": stable_hash("candidates", *candidate_words)[:20],
            "admissible_count": n_adm,
            "bucket": bucket,
            "adaptive_threshold": ADAPTIVE_THRESHOLD,
            "action_space": action_space,
            "chosen_is_admissible": chosen in admissible,
            "rejected_is_admissible": rejected in admissible,
            "score_family": score_family,
            "tree_depth": tree_depth if action_space == "admissible" else 1,
            "score_direction": "lower_expected_game_cost_is_preferred",
            "chosen_score": round(float(cscore), 8),
            "rejected_score": round(float(rscore), 8),
            "preference_gap": round(float(rscore - cscore), 8),
            "competitive_gap_min": min_gap,
            "competitive_gap_max": max_gap,
            "strict_answer_partition": strict_partition,
            "hidden_answer_in_prompt": False,
            "candidate_information_in_prompt": False,
        },
    }


def load_split_answers(sft_dir):
    def read(name):
        path = os.path.join(sft_dir, "eval", name)
        return [json.loads(line)["answer"].lower()
                for line in open(path, encoding="utf-8") if line.strip()]
    return read("train_answers.jsonl"), read("val_answers.jsonl")


def harvest(bundle, source_answers, allowed_answers, split, quotas, rng,
            strict_answer_partition, max_rollouts, forbidden_state_hashes=()):
    """Noisy but reachable rollouts; retain one close pair per distinct state."""
    all_answers = set(allowed_answers)
    legal = tuple(bundle.vocab.guesses)
    answer_index = bundle.vocab.answer_index
    tree = AdaptiveTree(bundle)
    rows, seen_states, seen_pairs = [], set(), set()
    forbidden_state_hashes = set(forbidden_state_hashes)
    got, rejects = Counter(), Counter()
    rollouts = 0
    while any(got[b] < quotas[b] for b in quotas) and rollouts < max_rollouts:
        rollouts += 1
        answer = source_answers[int(rng.integers(len(source_answers)))]
        cand_idx = np.arange(bundle.vocab.n_answers, dtype=np.int32)
        admissible = tuple(legal)
        history = []
        guess = "salet"
        for _ in range(MAX_GUESSES - 1):
            code = feedback_code(guess, answer)
            if code == ALL_GREEN:
                break
            pattern = feedback(guess, answer)
            history.append((guess, pattern))
            cand_idx = bundle.fb.filter_indices(cand_idx, guess, code)
            admissible = tuple(w for w in admissible if feedback_code(guess, w) == code)
            if not len(cand_idx):
                raise AssertionError("reachable rollout emptied its candidate set")
            candidate_words = {bundle.vocab.answers[int(i)] for i in cand_idx}
            # Strong split: no label state has even a possible answer from the
            # other split. This is stricter than answer-keyed source splitting.
            if strict_answer_partition and not candidate_words <= all_answers:
                rejects["cross_split_candidate_state"] += 1
            else:
                b = bucket_of(len(admissible))
                key = state_key(history)
                if (b is not None and got[b] < quotas[b] and key not in seen_states
                        and stable_hash("state", key)[:20] not in forbidden_state_hashes):
                    rec = make_record(bundle, tree, history, cand_idx, admissible,
                                      answer, split, strict_answer_partition)
                    if rec is None:
                        rejects["no_close_pair"] += 1
                    else:
                        pair = (key, rec["chosen"], rec["rejected"])
                        if pair in seen_pairs:
                            raise AssertionError("duplicate pair after state dedup")
                        seen_states.add(key); seen_pairs.add(pair)
                        rows.append(rec); got[b] += 1
            if len(cand_idx) == 1:
                break
            # DAgger-style off-policy continuation: reachable yet not limited
            # to expert paths. Candidate guesses keep the game realistic; legal
            # probes cover failures the LLM can actually reach.
            if rng.random() < 0.60:
                guess = bundle.vocab.answers[int(cand_idx[rng.integers(len(cand_idx))])]
            else:
                guess = legal[int(rng.integers(len(legal)))]
        if rollouts % 1000 == 0:
            print(f"  {split}: {rollouts:6d} rollouts, {len(rows):4d} rows, "
                  f"{dict(got)}", flush=True)
    return rows, got, rejects, rollouts, tree.nodes


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--out", default="sft_package/data/dpo_adaptive_competitive_v2")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-rollouts", type=int, default=120000)
    ap.add_argument("--strict-candidate-partition", action="store_true",
                    help="also require every state candidate to belong to its source split; this is much smaller and may not fill 100+ validation")
    for b in DEFAULT_TRAIN:
        suffix = b.replace("-", "_").replace("+", "plus")
        ap.add_argument("--train-" + suffix.replace("_", "-"),
                        dest="train_" + suffix, type=int, default=DEFAULT_TRAIN[b])
        ap.add_argument("--val-" + suffix.replace("_", "-"),
                        dest="val_" + suffix, type=int, default=DEFAULT_VAL[b])
    a = ap.parse_args(argv)
    train_q = {b: getattr(a, "train_" + b.replace("-", "_").replace("+", "plus"))
               for b in DEFAULT_TRAIN}
    val_q = {b: getattr(a, "val_" + b.replace("-", "_").replace("+", "plus"))
             for b in DEFAULT_VAL}
    strict = a.strict_candidate_partition
    bundle = load_artifacts(a.artifacts, mmap=True)
    train_answers, val_answers = load_split_answers(a.sft_dir)
    if set(train_answers) & set(val_answers):
        raise AssertionError("source answer splits overlap")
    print("Corrected Phase-8 DPO generator")
    print(f"seed={a.seed} strict_answer_partition={strict}")
    print(f"source answers: train={len(train_answers)} val={len(val_answers)}")
    print(f"quotas: train={train_q} val={val_q}")
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    train, train_got, train_rej, train_roll, train_nodes = harvest(
        bundle, train_answers, set(train_answers), "train", train_q, rng,
        strict, a.max_rollouts)
    val, val_got, val_rej, val_roll, val_nodes = harvest(
        bundle, val_answers, set(val_answers), "validation", val_q, rng,
        strict, a.max_rollouts,
        forbidden_state_hashes={r["meta"]["state_hash"] for r in train})
    train_keys = {r["meta"]["state_hash"] for r in train}
    val_keys = {r["meta"]["state_hash"] for r in val}
    if train_keys & val_keys:
        raise AssertionError("train/validation state leakage")
    write_jsonl(os.path.join(a.out, "train.jsonl"), train)
    write_jsonl(os.path.join(a.out, "validation.jsonl"), val)
    manifest = {
        "format_version": 2,
        "seed": a.seed,
        "adaptive_threshold": ADAPTIVE_THRESHOLD,
        "strict_answer_partition": strict,
        "source_answers": {"train": len(train_answers), "validation": len(val_answers)},
        "requested": {"train": train_q, "validation": val_q},
        "produced": {"train": dict(train_got), "validation": dict(val_got)},
        "rows": {"train": len(train), "validation": len(val)},
        "state_overlap": len(train_keys & val_keys),
        "rollouts": {"train": train_roll, "validation": val_roll},
        "adaptive_tree_nodes": {"train": train_nodes, "validation": val_nodes},
        "rejections": {"train": dict(train_rej), "validation": dict(val_rej)},
        "label_policy": {
            "2-20_admissible": "exact adaptive hard-mode tree, depth capped at 3",
            "21+_admissible": "exhaustive full-legal one-ply tree expected remaining",
            "pairs_per_state": 1,
            "trivial_pairs": "excluded by a bounded close-value gap",
        },
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote train={len(train)} validation={len(val)} in {time.perf_counter()-t0:.1f}s")
    print(json.dumps(manifest["produced"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
