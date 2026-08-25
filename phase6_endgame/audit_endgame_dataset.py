"""
audit_endgame_dataset.py - deep audit of the Phase 6 training mix.

verify_endgame_dataset.py checks policy (no leakage, legal labels). This goes
further and checks that the data will actually TRAIN correctly:

  A. Re-derive every state FROM ITS OWN PROMPT and confirm the prompt encodes
     the state the metadata claims. If the prompt and the label disagree, the
     model is being taught noise, and no leakage check would catch it.
  B. Contradictory labels: the same prompt with two different completions.
     Mixing two files makes this possible and it is silent poison.
  C. Prompt/meta turn agreement, and the "Guess N of 6 (M remaining)" line.
  D. Tokenization: lengths, truncation under MAX_SEQ_LEN, completion token
     count, EOS handling - the things that silently corrupt the loss mask.
  E. Label distribution: is any completion pathologically over-represented?
  F. Raw dumps of real records from every bucket, printed in full, so a human
     can read what the model will actually see.

    python audit_endgame_dataset.py
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = _paths.ROOT  # data lives at the repo root
sys.path.insert(0, HERE)

from wordle_solver import load_artifacts, feedback_code, code_to_pattern

NATURAL = "sft_package/data/tree_salet/train.jsonl"
ENDGAME = "sft_package/data/tree_salet_endgame/train.jsonl"
MAX_SEQ_LEN = 640
HIST_RE = re.compile(r"^\s*(\d+)\.\s+([A-Za-z]{5})\s*->\s*([GYB]{5})\s*$")
TURN_RE = re.compile(r"^Guess (\d+) of (\d+) \((\d+) remaining\)\s*$", re.M)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def load(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def parse_history(prompt):
    """Recover [(guess, pattern), ...] from the rendered prompt text."""
    out = []
    inside = False
    for ln in prompt.splitlines():
        if ln.startswith("History:"):
            inside = True
            continue
        if inside:
            if not ln.strip():
                if out:
                    break
                continue
            m = HIST_RE.match(ln)
            if m:
                out.append((m.group(2).lower(), m.group(3)))
            elif ln.startswith("Deduced"):
                break
    return out


def main():
    bundle = load_artifacts("artifacts", mmap=True)
    V = bundle.vocab
    nat, eg = load(NATURAL), load(ENDGAME)
    mixed = nat + eg
    print(f"natural {len(nat)}   endgame {len(eg)}   mixed {len(mixed)}\n")

    # ---------------------------------------------------------------- A ----
    print("A. RE-DERIVE EACH STATE FROM ITS OWN PROMPT")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(eg), size=min(4000, len(eg)), replace=False)
    bad_n, bad_k1, unparsed = [], [], 0
    for i in sample_idx:
        r = eg[int(i)]
        hist = parse_history(r["prompt"])
        if not hist:
            unparsed += 1
            continue
        cands = np.arange(V.n_answers, dtype=np.int32)
        for g, pat in hist:
            code = sum({"B": 0, "Y": 1, "G": 2}[c] * (3 ** j)
                       for j, c in enumerate(pat))
            cands = bundle.fb.filter_indices(cands, g, code)
        n = int(len(cands))
        if n != r["meta"]["n_candidates"]:
            bad_n.append((r["id"], n, r["meta"]["n_candidates"]))
        if r["meta"]["n_candidates"] == 1:
            if n != 1 or V.answers[int(cands[0])].upper() != r["completion"]:
                bad_k1.append((r["id"], r["completion"],
                               [V.answers[int(c)].upper() for c in cands[:3]]))
    check(f"prompt history parses ({len(sample_idx)} sampled)", unparsed == 0,
          f"{unparsed} unparsable")
    check("re-derived candidate count == meta.n_candidates", not bad_n,
          f"{len(bad_n)} mismatches {bad_n[:3]}")
    check("at k=1 the completion IS the unique surviving answer", not bad_k1,
          f"{len(bad_k1)} mismatches {bad_k1[:3]}")

    # ---------------------------------------------------------------- B ----
    print("\nB. CONTRADICTORY LABELS (same prompt, different completion)")
    by_prompt = defaultdict(set)
    for r in mixed:
        by_prompt[r["prompt"]].add(r["completion"])
    conflict = {p: c for p, c in by_prompt.items() if len(c) > 1}
    check("no prompt maps to two different completions in the MIX",
          not conflict, f"{len(conflict)} conflicting prompts")
    if conflict:
        p, c = next(iter(conflict.items()))
        print("      example conflict:", sorted(c))
        print("      " + p.splitlines()[-3])
    dup_pairs = len(mixed) - len(by_prompt)
    print(f"      exact-duplicate prompts across the mix: {dup_pairs} "
          f"({100*dup_pairs/len(mixed):.1f}%)")
    overlap = set(r["prompt"] for r in nat) & set(r["prompt"] for r in eg)
    print(f"      prompts appearing in BOTH natural and endgame: {len(overlap)}")

    # ---------------------------------------------------------------- C ----
    print("\nC. PROMPT / METADATA CONSISTENCY")
    bad_turn, bad_rem, bad_hist = [], [], []
    for r in eg:
        m = TURN_RE.search(r["prompt"])
        if not m:
            bad_turn.append(r["id"]); continue
        t, tot, rem = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if t != r["meta"]["turn"]:
            bad_turn.append((r["id"], t, r["meta"]["turn"]))
        if rem != tot - t + 1:
            bad_rem.append((r["id"], t, rem))
        if len(parse_history(r["prompt"])) != t - 1:
            bad_hist.append((r["id"], t, len(parse_history(r["prompt"]))))
    check("prompt turn == meta.turn", not bad_turn, f"{len(bad_turn)} bad")
    check("'(M remaining)' == 6 - turn + 1", not bad_rem, f"{len(bad_rem)} bad")
    check("history length == turn - 1", not bad_hist, f"{len(bad_hist)} bad")
    check("every turn is within 1..6",
          all(1 <= r["meta"]["turn"] <= 6 for r in eg))

    # ---------------------------------------------------------------- D ----
    print("\nD. TOKENIZATION (what the trainer will actually see)")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        eos = tok.eos_token_id
        lens, comp_lens, trunc = [], [], 0
        for r in mixed[::7]:
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(" " + r["completion"], add_special_tokens=False)["input_ids"] + [eos]
            lens.append(len(p) + len(c))
            comp_lens.append(len(c))
            if len(p) + len(c) > MAX_SEQ_LEN:
                trunc += 1
        lens, comp_lens = np.array(lens), np.array(comp_lens)
        print(f"      total length: min {lens.min()}  mean {lens.mean():.0f}  "
              f"p95 {np.percentile(lens,95):.0f}  max {lens.max()}")
        print(f"      completion tokens (incl EOS): "
              f"{dict(sorted(Counter(comp_lens.tolist()).items()))}")
        check(f"nothing truncates at MAX_SEQ_LEN={MAX_SEQ_LEN}", trunc == 0,
              f"{trunc}/{len(lens)} would truncate")
        check("supervised tokens per example <= 6 (loss mask stays tight)",
              int(comp_lens.max()) <= 6, f"max {int(comp_lens.max())}")
        check("no empty completion", int(comp_lens.min()) >= 2)
    except Exception as e:
        print(f"      tokenizer unavailable ({e}); skipped")

    # ---------------------------------------------------------------- E ----
    print("\nE. LABEL DISTRIBUTION")
    for lbl, rows in (("natural", nat), ("endgame", eg), ("mixed", mixed)):
        c = Counter(r["completion"] for r in rows)
        top = c.most_common(3)
        share = 100 * top[0][1] / len(rows)
        print(f"      {lbl:<8} {len(c):>5} distinct   top: {top}   "
              f"top share {share:.1f}%")
    c_eg = Counter(r["completion"] for r in eg)
    check("no single completion exceeds 5% of the endgame data",
          100 * c_eg.most_common(1)[0][1] / len(eg) < 5.0,
          f"{100*c_eg.most_common(1)[0][1]/len(eg):.2f}%")
    k1w = Counter(r["completion"] for r in eg if r["meta"]["n_candidates"] == 1)
    check("k=1 labels spread over >2000 distinct words", len(k1w) > 2000,
          f"{len(k1w)} words")

    # ---------------------------------------------------------------- F ----
    print("\nF. RAW RECORDS - read these")
    shown = set()
    for want in ("1", "2", "3", "4-10"):
        for r in eg:
            b = r["meta"].get("bucket")
            if b == want and b not in shown:
                shown.add(b)
                print("\n" + "=" * 72)
                print(f"BUCKET k={want}   turn={r['meta']['turn']}   "
                      f"n_candidates={r['meta']['n_candidates']}   id={r['id']}")
                print("=" * 72)
                print(r["prompt"])
                print(f"--> COMPLETION: {r['completion']!r}")
                print(f"    meta: is_candidate={r['meta']['action_is_candidate']} "
                      f"is_answer={r['meta']['action_is_possible_answer']} "
                      f"info_gain={r['meta']['information_gain_bits']}")
                break

    print("\n" + "=" * 72)
    print("ONE NATURAL RECORD, for format comparison")
    print("=" * 72)
    nk1 = next(r for r in nat if r["meta"]["n_candidates"] == 1)
    print(nk1["prompt"])
    print(f"--> COMPLETION: {nk1['completion']!r}")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
