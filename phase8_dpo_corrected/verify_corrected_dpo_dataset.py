"""Exhaustive audit for the corrected DPO dataset, plus an old/new comparison.

This verifies labels from the rendered prompt, not from trusted generation
metadata.  It is intentionally independent of model training and is suitable
as a Kaggle preflight check.
"""
import _paths  # noqa: F401

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

from wordle_solver import feedback_code, load_artifacts
from build_corrected_dpo_dataset import (
    ADAPTIVE_THRESHOLD, AdaptiveTree, code_from_pattern, full_pool_horizon_one,
    stable_hash, state_key,
)


HIST = re.compile(r"^\s*\d+\.\s+([A-Za-z]{5})\s*->\s*([GYB]{5})\s*$")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def history_from_prompt(prompt):
    out, inside = [], False
    for line in prompt.splitlines():
        if line.startswith("History:"):
            inside = True
            continue
        if inside:
            match = HIST.match(line)
            if match:
                out.append((match.group(1).lower(), match.group(2)))
            elif line.startswith("Deduced so far:"):
                break
    return out


def reconstruct(bundle, history):
    cand = np.arange(bundle.vocab.n_answers, dtype=np.int32)
    admissible = tuple(bundle.vocab.guesses)
    for guess, pattern in history:
        code = code_from_pattern(pattern)
        cand = bundle.fb.filter_indices(cand, guess, code)
        admissible = tuple(w for w in admissible if feedback_code(guess, w) == code)
    return cand, admissible


def reconstruct_candidates(bundle, history):
    """Candidate-only replay for the legacy audit; avoids needless A scans."""
    cand = np.arange(bundle.vocab.n_answers, dtype=np.int32)
    for guess, pattern in history:
        cand = bundle.fb.filter_indices(cand, guess, code_from_pattern(pattern))
    return cand


def q(values, p):
    return round(float(np.percentile(np.asarray(values), p)), 6) if values else None


def audit_corrected(root, bundle, train_answers, val_answers):
    paths = {"train": os.path.join(root, "train.jsonl"),
             "validation": os.path.join(root, "validation.jsonl")}
    rows = {k: load_jsonl(v) for k, v in paths.items()}
    failures = []
    def fail(msg):
        failures.append(msg)

    train_states, val_states = set(), set()
    seen_ids, seen_pairs = set(), set()
    legal = set(bundle.vocab.guesses)
    split_answers = {"train": set(train_answers), "validation": set(val_answers)}
    split_answer_hashes = {k: {stable_hash("phase8-v2-source", answer)[:16] for answer in v}
                           for k, v in split_answers.items()}
    tree = AdaptiveTree(bundle)
    stats = {}
    banned = ("Possible answers remaining", "candidates remaining", "Candidates:",
              "admissible", "candidate_count", "admissible_count", "chosen_score",
              "rejected_score", "preference_gap", "Answer:")
    for split, records in rows.items():
        gaps, labels = [], Counter()
        for index, record in enumerate(records):
            where = f"{split}[{index}]"
            meta = record.get("meta", {})
            if meta.get("split") != split:
                fail(f"{where}: split metadata")
            if meta.get("source_answer_hash") not in split_answer_hashes[split]:
                fail(f"{where}: source answer belongs to the other split")
            if record.get("id") in seen_ids:
                fail(f"{where}: duplicate id")
            seen_ids.add(record.get("id"))
            history = history_from_prompt(record.get("prompt", ""))
            if not history:
                fail(f"{where}: unparseable/empty history")
                continue
            key = state_key(history)
            target = train_states if split == "train" else val_states
            if key in target:
                fail(f"{where}: duplicate state")
            target.add(key)
            pair = (key, record.get("chosen"), record.get("rejected"))
            if pair in seen_pairs:
                fail(f"{where}: duplicate pair")
            seen_pairs.add(pair)
            if record.get("chosen", "").lower() not in legal or record.get("rejected", "").lower() not in legal:
                fail(f"{where}: non-legal action")
            if record.get("chosen") == record.get("rejected"):
                fail(f"{where}: identical actions")
            if any(word in record.get("prompt", "") for word in banned):
                fail(f"{where}: prompt contains prohibited label wording")
            if not record.get("prompt", "").rstrip().endswith("Next guess:"):
                fail(f"{where}: prompt footer")
            cand, admissible = reconstruct(bundle, history)
            candidate_words = tuple(bundle.vocab.answers[int(i)] for i in cand)
            if not len(cand):
                fail(f"{where}: unreachable empty candidate state")
                continue
            if meta.get("candidate_count") != len(cand):
                fail(f"{where}: candidate count mismatch")
            if meta.get("admissible_count") != len(admissible):
                fail(f"{where}: admissible count mismatch")
            if meta.get("candidate_set_hash") != stable_hash("candidates", *candidate_words)[:20]:
                fail(f"{where}: candidate-set digest mismatch")
            if meta.get("state_hash") != stable_hash("state", key)[:20]:
                fail(f"{where}: state digest mismatch")
            if meta.get("strict_answer_partition") and not set(candidate_words) <= split_answers[split]:
                fail(f"{where}: cross-split answer candidate")
            expected_space = "admissible" if len(admissible) <= ADAPTIVE_THRESHOLD else "legal"
            if meta.get("action_space") != expected_space:
                fail(f"{where}: decoder action-space mismatch")
            chosen, rejected = record["chosen"].lower(), record["rejected"].lower()
            if expected_space == "admissible" and (chosen not in admissible or rejected not in admissible):
                fail(f"{where}: inadmissible restricted action")
            if meta.get("score_family") == "adaptive_hardmode_tree_depth_3":
                values = tree.action_values(cand, admissible, meta.get("tree_depth"))
            elif meta.get("score_family") == "full_legal_tree_horizon_1_expected_remaining":
                values = full_pool_horizon_one(bundle, cand)
            else:
                fail(f"{where}: unknown score family")
                continue
            cscore, rscore = values[chosen], values[rejected]
            if not cscore < rscore:
                fail(f"{where}: preference direction is reversed or tied")
            if abs(cscore - meta.get("chosen_score", float("nan"))) > 2e-7:
                fail(f"{where}: chosen score mismatch")
            if abs(rscore - meta.get("rejected_score", float("nan"))) > 2e-7:
                fail(f"{where}: rejected score mismatch")
            if not (meta["competitive_gap_min"] - 2e-7 <= rscore - cscore <= meta["competitive_gap_max"] + 2e-7):
                fail(f"{where}: non-competitive gap")
            gaps.append(rscore - cscore)
            labels[(meta.get("bucket"), meta.get("score_family"), meta.get("action_space"))] += 1
        stats[split] = {
            "rows": len(records), "distinct_states": len(train_states if split == "train" else val_states),
            "bucket_score_space": {"|".join(k): v for k, v in sorted(labels.items())},
            "gap": {"min": min(gaps) if gaps else None, "median": q(gaps, 50), "p90": q(gaps, 90), "max": max(gaps) if gaps else None},
        }
    if train_states & val_states:
        fail("train/validation exact state overlap")
    return {"ok": not failures, "failures": failures, "stats": stats,
            "state_overlap": len(train_states & val_states), "adaptive_tree_nodes_checked": tree.nodes}


def audit_old(path, bundle, val_answers):
    """Report old data properties; it does not pretend legacy design is strict."""
    rows = load_jsonl(path)
    states = Counter(r["prompt"] for r in rows)
    pairs = {(r["prompt"], r["chosen"], r["rejected"]) for r in rows}
    ids = Counter(r["id"] for r in rows)
    val_idx = {bundle.vocab.answer_index[w] for w in val_answers}
    state_cache, state_val_overlap = {}, 0
    reversed_rows = 0
    gi = {w: i for i, w in enumerate(bundle.vocab.guesses)}
    for record in rows:
        prompt = record["prompt"]
        if prompt not in state_cache:
            history = history_from_prompt(prompt)
            cand = reconstruct_candidates(bundle, history)
            state_cache[prompt] = cand
        cand = state_cache[prompt]
        pair = [record["chosen"].lower(), record["rejected"].lower()]
        st = bundle.fb.partition_stats(np.asarray([gi[w] for w in pair], dtype=np.int32), cand)
        candidate_words = {bundle.vocab.answers[int(i)] for i in cand}
        cost = st["expected_remaining"] - np.asarray([0.5 if w in candidate_words else 0.0 for w in pair])
        reversed_rows += int(not cost[0] < cost[1])
    for cand in state_cache.values():
        state_val_overlap += int(bool(set(map(int, cand)) & val_idx))
    clear = [r["meta"]["cost_gap"] for r in rows if r["meta"]["pair_type"] == "clear"]
    competitive = [r["meta"]["cost_gap"] for r in rows if r["meta"]["pair_type"] == "competitive"]
    return {
        "rows": len(rows), "distinct_states": len(states), "duplicate_rows_by_state": len(rows) - len(states),
        "max_pairs_per_state": max(states.values()), "distinct_exact_pairs": len(pairs),
        "duplicate_ids": sum(v - 1 for v in ids.values() if v > 1),
        "pair_types": dict(Counter(r["meta"]["pair_type"] for r in rows)),
        "buckets": dict(Counter(r["meta"]["bucket"] for r in rows)),
        "old_cost_direction_failures": reversed_rows,
        "states_with_possible_heldout_answer": state_val_overlap,
        "clear_gap": {"median": q(clear, 50), "max": max(clear)},
        "competitive_gap": {"median": q(competitive, 50), "max": max(competitive)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_package/data/dpo_adaptive_competitive_v2")
    ap.add_argument("--old", default="sft_package/data/dpo_midgame/train.jsonl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sft-dir", default="sft_package")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    bundle = load_artifacts(args.artifacts, mmap=True)
    train_answers = [r["answer"].lower() for r in load_jsonl(os.path.join(args.sft_dir, "eval", "train_answers.jsonl"))]
    val_answers = [r["answer"].lower() for r in load_jsonl(os.path.join(args.sft_dir, "eval", "val_answers.jsonl"))]
    report = {"corrected": audit_corrected(args.data, bundle, train_answers, val_answers),
              "old": audit_old(args.old, bundle, val_answers)}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    return 0 if report["corrected"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
