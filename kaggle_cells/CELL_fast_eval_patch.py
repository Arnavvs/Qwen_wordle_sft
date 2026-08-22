# =============================================================================
# LIKELY FIXES for the slow eval. Run the debug cell FIRST - these are ordered
# by what the debug output would point at, and applying them blind wastes a run.
#
# Each is independent. Apply only the one the diagnosis supports.
# =============================================================================

# --- FIX 1: attention implementation ----------------------------------------
# If section A printed attn impl 'eager', that alone can cost 5-10x. Reload
# forcing sdpa. (transformers 5.0 changed the default-selection logic.)
def base_model_sdpa():
    kw = dict(trust_remote_code=True, attn_implementation="sdpa")
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, **kw)
    assert next(m.parameters()).dtype == torch.float16
    m.config.use_cache = False
    return m

# then:  base_model = base_model_sdpa   (rebind before load_sft/load_dpo)


# --- FIX 2: drop the explicit attention_mask --------------------------------
# _score_rows passes a full 2D mask alongside past_key_values. Newer
# transformers may materialise a 4D causal mask per call from it, which is
# O(batch x seq^2). With a batch-1 prompt cache expanded to C identical rows,
# there is no padding, so the mask is all ones and can be omitted - the model
# then takes its fast internal path.
_orig_score_rows = LegalWordScorer._score_rows

@torch.no_grad()
def _score_rows_nomask(self, model, cache, lp0, prompt_len, rows):
    idx = rows.to(self.device)
    toks = self.tokens[idx]; msk = self.mask[idx]
    C, L = toks.shape
    total = lp0[toks[:, 0]] * msk[:, 0]
    if L > 1:
        big = expand_cache(cache, C)
        pos = torch.arange(prompt_len, prompt_len + L,
                           device=self.device).unsqueeze(0).expand(C, L)
        out = model(input_ids=toks, past_key_values=big,
                    position_ids=pos, use_cache=False)      # no attention_mask
        for i in range(1, L):
            lp = torch.log_softmax(out.logits[:, i - 1].float(), dim=-1)
            total = total + lp.gather(1, toks[:, i:i+1]).squeeze(1) * msk[:, i]
        del out, big
    if self.length_normalise:
        total = total / msk.sum(1)
    return total

# to apply:      LegalWordScorer._score_rows = _score_rows_nomask
# to revert:     LegalWordScorer._score_rows = _orig_score_rows
#
# MUST re-verify after applying - this changes the numbers path:
#     verify_scorer(load_sft())


# --- FIX 3: memo cache (exact, always safe) ---------------------------------
# The turn-1 prompt is identical across all 246 games and turn 2 has ~141
# distinct prompts. The model is deterministic and greedy, so caching by
# prompt is exact. This helps regardless of the root cause.
#   -> use play_cached() from the earlier message; remember DECISION_CACHE.clear()
#      at the start of each model, or DPO silently replays SFT's decisions.


# --- FIX 4: reuse the known SFT baseline ------------------------------------
# The SFT adapter and decoder are identical to Phase 7, which measured
# 3.7642 / 242 solved. Re-measuring it costs half the eval budget for a number
# we already have. Only the DPO row is new.
#   -> evaluate "dpo" only, and take the SFT per-game scores from the v1
#      results file for the paired test.
