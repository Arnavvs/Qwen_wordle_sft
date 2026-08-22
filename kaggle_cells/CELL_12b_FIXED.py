# =============================================================================
# Section 12b - terminal-state probe (CAPPED + PROGRESS)
#
# score_all() scores all 12,972 words with no pruning, so each probe state
# costs ~26 chunk-forwards regardless of model. 294 states x 4 models is ~3.4h
# with zero output until a model finishes. Hence: per-k caps, an optional model
# subset, and progress printing.
#
# k=1 is the number that matters (the state already determines the answer), so
# it keeps the largest sample.
# =============================================================================
TERMINAL_MAX_STATES = {1: 80, 2: 30, 3: 27}   # per-k caps; None = all
TERMINAL_MODELS     = ["tree_salet", "base_qwen"]   # None = all four
PROBE_LOG_EVERY     = 10

# Training targets, for the seen-vs-unseen split.
if "TRAIN_TARGET_WORDS" not in globals():
    TRAIN_TARGET_WORDS = set()
    for _ds in {EXPERIMENTS[n]["dataset"] for n in EXPERIMENTS}:
        _p = os.path.join(SFT_DIR, f"data/{_ds}/train.jsonl")
        if os.path.exists(_p):
            for _l in open(_p, encoding="utf-8"):
                TRAIN_TARGET_WORDS.add(json.loads(_l)["completion"].upper())
print(f"distinct training targets across datasets: {len(TRAIN_TARGET_WORDS)}")

# Reuse STATES if the previous run already built it.
if "STATES" not in globals():
    STATES = terminal_states(VAL_ANSWERS, TERMINAL_KS)
for _k in TERMINAL_KS:
    print(f"  k={_k}: {len(STATES[_k])} probe states available")


@torch.no_grad()
def probe_terminal(model, states, label, caps=None):
    sc = build_scorer()
    caps = caps or {}
    rows, per_k = [], {}
    plan = {k: (states[k] if caps.get(k) is None else states[k][:caps[k]])
            for k in sorted(states)}
    total = sum(len(v) for v in plan.values())
    print(f"  scoring {total} states "
          f"({', '.join(f'k={k}:{len(v)}' for k, v in plan.items())})")
    t0 = time.perf_counter()
    done = 0
    for k, use in plan.items():
        hit1 = hit_cand = 0
        ranks, ranks_in_cand = [], []
        for st in use:
            scores = sc.score_all(model, st["prompt"])
            top1 = sc.words[int(scores.argmax().item())]
            r = sc.rank_of(scores, [st["answer"]]).get(st["answer"])
            cand_scores = [(sc.words[sc.index[c]], scores[sc.index[c]].item())
                           for c in st["candidates"] if c in sc.index]
            cand_scores.sort(key=lambda x: -x[1])
            r_in = next((i + 1 for i, (w, _) in enumerate(cand_scores)
                         if w == st["answer"]), None)
            hit1 += int(top1 == st["answer"])
            hit_cand += int(top1 in set(st["candidates"]))
            if r: ranks.append(r)
            if r_in: ranks_in_cand.append(r_in)
            rows.append({"model": label, "k": k, "answer": st["answer"],
                         "top1": top1, "correct": top1 == st["answer"],
                         "top1_is_candidate": top1 in set(st["candidates"]),
                         "rank_of_answer": r, "rank_within_candidates": r_in,
                         "answer_in_train_targets":
                             st["answer"] in TRAIN_TARGET_WORDS})
            done += 1
            if PROBE_LOG_EVERY and done % PROBE_LOG_EVERY == 0:
                el = time.perf_counter() - t0
                print(f"    {done}/{total} states  {el:.0f}s elapsed, "
                      f"~{el/done*(total-done):.0f}s left", flush=True)
        n = max(len(use), 1)
        per_k[k] = {
            "n_states": len(use),
            "top1_accuracy_pct": round(100 * hit1 / n, 2),
            "top1_is_candidate_pct": round(100 * hit_cand / n, 2),
            "median_rank_of_answer": (float(np.median(ranks)) if ranks else None),
            "mean_rank_within_candidates": (round(float(np.mean(ranks_in_cand)), 3)
                                            if ranks_in_cand else None),
            "chance_top1_pct": round(100 / sc.n, 4),
            "chance_within_candidates_pct": round(100 / k, 2),
        }
    return per_k, rows


TERMINAL, TERMINAL_ROWS = {}, []
_targets = [(n, load_adapter, f"qwen_{n}") for n in EXPERIMENTS
            if n in TRAIN_CONFIGS]
if RUN_BASE_CONTROL:
    _targets.append(("base_qwen", lambda _n: load_base_control(), "base_qwen"))
if TERMINAL_MODELS is not None:
    _targets = [t for t in _targets if t[0] in TERMINAL_MODELS]
print(f"\nprobing models: {[t[2] for t in _targets]}")

for _name, _loader, _label in _targets:
    print(f"\n--- terminal probe: {_label} ---", flush=True)
    _model = _loader(_name)
    if not _SCORER_VERIFIED:
        verify_scorer(_model)
    _t0 = time.perf_counter()
    _per_k, _rows = probe_terminal(_model, STATES, _label, caps=TERMINAL_MAX_STATES)
    TERMINAL[_label] = _per_k
    TERMINAL_ROWS.extend(_rows)
    for _k, _r in sorted(_per_k.items()):
        print(f"  k={_k}  n={_r['n_states']:>3}  "
              f"top1={_r['top1_accuracy_pct']:>6.2f}%  "
              f"(chance {_r['chance_top1_pct']}%)   "
              f"top1_is_candidate={_r['top1_is_candidate_pct']:>6.2f}%  "
              f"median_rank_of_answer={_r['median_rank_of_answer']}")
    print(f"  ({time.perf_counter()-_t0:.0f}s)")
    del _model; torch.cuda.empty_cache()

print("\n" + "=" * 66)
print("TOP-1 ACCURACY AT k=1, SPLIT BY WHETHER THE ANSWER WAS EVER A "
      "TRAINING TARGET")
print("=" * 66)
_hdr = f"{'model':<18}{'seen n':>8}{'seen acc':>10}{'unseen n':>10}{'unseen acc':>12}"
print(_hdr); print("-" * len(_hdr))
TERMINAL_SPLIT = {}
for _label in sorted({r["model"] for r in TERMINAL_ROWS}):
    _sub = [r for r in TERMINAL_ROWS if r["model"] == _label and r["k"] == 1]
    _seen = [r for r in _sub if r["answer_in_train_targets"]]
    _unseen = [r for r in _sub if not r["answer_in_train_targets"]]
    _f = lambda xs: (round(100*sum(x["correct"] for x in xs)/len(xs), 2)
                     if xs else None)
    TERMINAL_SPLIT[_label] = {"seen_n": len(_seen), "seen_acc_pct": _f(_seen),
                              "unseen_n": len(_unseen), "unseen_acc_pct": _f(_unseen)}
    print(f"{_label:<18}{len(_seen):>8}{str(_f(_seen)):>10}"
          f"{len(_unseen):>10}{str(_f(_unseen)):>12}")
