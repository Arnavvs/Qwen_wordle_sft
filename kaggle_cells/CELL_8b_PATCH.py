# =============================================================================
# PATCH for the section 8b self-test.
#
# Run this cell AFTER the existing section 8b cell, then re-run the evaluation
# cell (section 9). Nothing else needs to change.
#
# WHY: the original check used one absolute tolerance (0.02 nats) calibrated on
# fp32 CPU. On a T4 the model runs fp16, and the two scoring paths use
# different matmul shapes -- one sequence of length P+L, versus a batch of
# L-token rows against an expanded P-token cache -- so fp16 reduction order
# differs and summed log-probs disagree at the 0.01-0.1 nat level. That is
# arithmetic noise, not a broken KV cache.
#
# A genuinely broken cache shim does not produce a 0.04-nat error. It produces
# scores that are wrong by many nats with the ranking destroyed. So the check
# is now two-sided: a dtype-aware absolute tolerance AND a ranking-correlation
# test that a broken shim cannot pass.
#
# Note the discrepancy is a VALIDATION artifact only. All 12,972 words are
# scored through the same fast path, so the ranking the argmax is read off is
# internally consistent.
# =============================================================================

@torch.no_grad()
def _self_test_v2(self, model, prompt, n_probe=32, atol=None):
    dtype = next(model.parameters()).dtype
    if atol is None:
        atol = 0.02 if dtype in (torch.float32, torch.float64) else 0.40

    fast = self.score_all(model, prompt)
    p_ids = self.tok(prompt, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(self.device)
    n_probe = min(n_probe, self.n)
    # spread across the whole list: a cache mutated by the first chunk would
    # corrupt only LATER chunks, and a front-loaded probe would miss it
    probe = [int(round(i * (self.n - 1) / max(n_probe - 1, 1)))
             for i in range(n_probe)]

    got, ref_all = [], []
    for i in probe:
        ids = self.tokens[i][self.mask[i] > 0].unsqueeze(0)
        full = torch.cat([p_ids, ids], dim=1)
        logits = model(input_ids=full, use_cache=False).logits[0].float()
        lp = F.log_softmax(logits, dim=-1)
        start = p_ids.shape[1] - 1
        ref = sum(lp[start + j, ids[0, j]].item() for j in range(ids.shape[1]))
        if self.length_normalise:
            ref /= ids.shape[1]
        ref_all.append(ref)
        got.append(fast[i].item())

    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref_all, dtype=np.float64)
    worst = float(np.max(np.abs(a - b)))
    spread = float(b.max() - b.min())
    corr = (float(np.corrcoef(a, b)[0, 1])
            if a.std() > 1e-9 and b.std() > 1e-9 else 1.0)

    assert corr > 0.999, (
        f"constrained scorer does not preserve the ranking of the naive "
        f"scorer (corr={corr:.6f}). The KV-cache shim in expand_cache() does "
        f"not match this transformers version -- do not trust constrained "
        f"results.")
    assert worst < atol, (
        f"constrained scorer disagrees with naive scoring by {worst:.4f} nats "
        f"(tol {atol} for dtype {dtype}), across a probe score spread of "
        f"{spread:.1f} nats. Ranking is preserved (corr={corr:.6f}), so this "
        f"looks like arithmetic noise rather than a broken cache -- but it is "
        f"larger than expected. Investigate before trusting the numbers.")
    return {"max_abs_dev": worst, "corr": corr, "spread": spread,
            "atol": atol, "dtype": str(dtype), "n_probe": n_probe}


LegalWordScorer.self_test = _self_test_v2


def verify_scorer(model):
    """Runs once per session. Failing here is fatal and it should be."""
    global _SCORER_VERIFIED
    sc = build_scorer()
    probe = GameState("CRANE", VOCAB.n_answers).prompt(1, MAX_GUESSES)
    dev = sc.self_test(model, probe)
    print(f"  cache-reuse vs naive scoring [{dev['dtype']}]: "
          f"max |delta| = {dev['max_abs_dev']:.4f} nats "
          f"(tol {dev['atol']}), over a {dev['spread']:.1f}-nat spread, "
          f"ranking corr = {dev['corr']:.6f}  OK")
    if CONSTRAINED_PRUNE:
        w, nch = sc.verify_against_full(model, probe)
        print(f"  pruned argmax == full argmax ({w}), {nch} chunks vs "
              f"{-(-sc.n // CONSTRAINED_CHUNK)} unpruned  OK")
    if BAN_REPEATS_IN_CONSTRAINED:
        full = sc.score_all(model, probe)
        want = [sc.words[i] for i in
                torch.argsort(full, descending=True)[:4].tolist()]
        got, ban = [], []
        for _ in range(4):
            w2, _, _, _ = sc.argmax(model, probe, banned=ban or None)
            assert w2 not in ban, f"banned word {w2} was returned"
            got.append(w2); ban.append(w2)
        assert got == want, f"banning broke the ranking: {got} != {want}"
        print(f"  sequential banning walks the global ranking {got}  OK")
    _SCORER_VERIFIED = True
    return dev


_SCORER_VERIFIED = False        # force re-verification with the new test
print("patched: self_test is now dtype-aware and checks ranking correlation.")
print("Now re-run the EVALUATION cell (section 9).")
