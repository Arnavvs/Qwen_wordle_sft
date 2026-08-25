"""
constrained_decode.py — exact argmax over a fixed legal-word set.

The question this answers is

    "Which of these 12,972 legal Wordle words should I play?"

not

    "Generate arbitrary text and see whether it happens to be a word."

Nothing is filtered after the fact. The model never emits free text at all in
constrained mode: every legal word is *scored*, and the highest-scoring one is
played.

--------------------------------------------------------------------------
Definition of the score
--------------------------------------------------------------------------
For a prompt `p` and a legal word `w`, tokenize exactly as the SFT data did --
`" " + w` -- and append EOS. Then

    score(w) = log P(w | p) = sum_i log P(t_i | p, t_0..t_{i-1})

summed over the word's tokens **and the EOS token**.

Including EOS matters. Without it, a word whose token sequence is a prefix of a
longer word's is scored on a strictly smaller set of constraints and is
systematically over-ranked; with EOS the scores are log-probabilities of
complete strings, so they are directly comparable across different token
lengths. No length normalisation is applied: `score(w)` is exactly the
probability the model assigns to playing `w`, which is the quantity we want to
argmax. (`length_normalise=True` is available for a sensitivity check but is a
heuristic, not the default.)

--------------------------------------------------------------------------
How it is computed
--------------------------------------------------------------------------
Naively this is 12,972 forward passes per turn. Instead:

1. The prompt is run **once** with `use_cache=True`, giving a KV cache and the
   next-token distribution `lp0` over the whole vocabulary.
2. `lp0` already gives the first-token log-probability of every legal word, for
   free -- no forward pass.
3. The remaining tokens are scored by teacher forcing: the prompt's KV cache is
   expanded to a batch of `chunk` rows and the padded `[chunk, L]` word-token
   matrix is pushed through in one forward pass. Causal masking makes the
   right-hand padding inert.

Exact branch-and-bound pruning (`prune=True`, on by default and *exact*):
every per-token log-probability is <= 0, so

    score(w) <= lp0[first_token(w)]

is a valid upper bound. Words are visited in descending order of that bound;
once the best fully-scored word beats the bound of every unvisited word, no
unvisited word can win and the scan stops. The returned argmax is identical to
scoring all 12,972 -- `verify_against_full()` asserts exactly that.

--------------------------------------------------------------------------
What is NOT done here
--------------------------------------------------------------------------
- The candidate-answer list is never consulted. The scorer ranks the full legal
  guess pool; it has no idea which words are still possible.
- The hidden answer is never consulted.
- Repeats are not banned by default (`banned` is opt-in), because banning them
  would be a policy change, not a vocabulary constraint.
"""

import numpy as np
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# KV-cache compatibility
#
# transformers has moved the cache representation twice (legacy tuple ->
# DynamicCache.key_cache/value_cache -> Cache.layers). Rather than pin a
# version, read whichever layout is present and rebuild through the same one.
# `LegalWordScorer.self_test()` verifies the result numerically at runtime, so
# a layout this shim gets wrong fails loudly instead of scoring garbage.
# ---------------------------------------------------------------------------
def _cache_layers(cache):
    if hasattr(cache, "layers"):                        # transformers >= 5
        return [(lyr.keys, lyr.values) for lyr in cache.layers]
    if hasattr(cache, "key_cache"):                     # transformers 4.x
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]                   # legacy tuple-of-tuples


def _rebuild_cache(template, layers):
    """Rebuild a cache object of the same kind as `template` from `layers`."""
    if hasattr(template, "layers"):
        import copy
        new = copy.deepcopy(template)
        for lyr, (k, v) in zip(new.layers, layers):
            lyr.keys, lyr.values = k, v
        return new
    if hasattr(template, "key_cache"):
        from transformers.cache_utils import DynamicCache
        new = DynamicCache()
        new.key_cache = [k for k, _ in layers]
        new.value_cache = [v for _, v in layers]
        return new
    return tuple(layers)


def expand_cache(cache, n):
    """Repeat a batch-1 KV cache to batch `n` without recomputing the prompt."""
    out = []
    for k, v in _cache_layers(cache):
        assert k.shape[0] == 1, f"expected batch-1 cache, got {k.shape[0]}"
        out.append((k.expand(n, *k.shape[1:]).contiguous(),
                    v.expand(n, *v.shape[1:]).contiguous()))
    return _rebuild_cache(cache, out)


# ---------------------------------------------------------------------------
class LegalWordScorer:
    """Scores every word in a fixed legal set under a causal LM."""

    def __init__(self, tokenizer, words, device="cuda", chunk=512,
                 length_normalise=False):
        self.tok = tokenizer
        self.words = list(words)
        self.device = device
        self.chunk = chunk
        self.length_normalise = length_normalise
        self.n = len(self.words)

        eos = tokenizer.eos_token_id
        seqs = []
        for w in self.words:
            # EXACTLY the training-time tokenization: " " + WORD, then EOS.
            ids = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
            seqs.append(ids + [eos])
        self.max_len = max(len(s) for s in seqs)

        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        tokens = np.full((self.n, self.max_len), pad, dtype=np.int64)
        mask = np.zeros((self.n, self.max_len), dtype=np.float32)
        for i, s in enumerate(seqs):
            tokens[i, :len(s)] = s
            mask[i, :len(s)] = 1.0

        self.tokens = torch.from_numpy(tokens).to(device)      # [N, L]
        self.mask = torch.from_numpy(mask).to(device)          # [N, L]
        self.lengths = torch.from_numpy(mask.sum(1)).to(device)
        self.first_tok = self.tokens[:, 0].clone()             # [N]
        self.index = {w: i for i, w in enumerate(self.words)}

    # -- internals ----------------------------------------------------------
    @torch.no_grad()
    def _prompt_pass(self, model, prompt):
        ids = self.tok(prompt, return_tensors="pt",
                       add_special_tokens=False)["input_ids"].to(self.device)
        out = model(input_ids=ids, use_cache=True)
        lp0 = F.log_softmax(out.logits[0, -1].float(), dim=-1)   # [V]
        return out.past_key_values, lp0, ids.shape[1]

    @torch.no_grad()
    def _score_rows(self, model, cache, lp0, prompt_len, rows):
        """Exact log P(w | prompt) for the word indices in `rows`."""
        idx = rows.to(self.device)
        toks = self.tokens[idx]                                  # [C, L]
        msk = self.mask[idx]
        C, L = toks.shape

        total = lp0[toks[:, 0]] * msk[:, 0]                      # token 0, free
        if L > 1:
            big = expand_cache(cache, C)
            attn = torch.ones(C, prompt_len + L, dtype=torch.long,
                              device=self.device)
            pos = torch.arange(prompt_len, prompt_len + L,
                               device=self.device).unsqueeze(0).expand(C, L)
            out = model(input_ids=toks, past_key_values=big,
                        attention_mask=attn, position_ids=pos, use_cache=False)
            # logits[:, i] predicts token i+1, so positions 1..L-1 read 0..L-2.
            for i in range(1, L):
                lp = F.log_softmax(out.logits[:, i - 1].float(), dim=-1)
                total = total + lp.gather(1, toks[:, i:i + 1]).squeeze(1) * msk[:, i]
            del out, big
        if self.length_normalise:
            total = total / msk.sum(1)
        return total

    # -- public API ---------------------------------------------------------
    @torch.no_grad()
    def score_all(self, model, prompt):
        """Score every legal word. No pruning. Returns a [N] float tensor."""
        cache, lp0, plen = self._prompt_pass(model, prompt)
        out = torch.empty(self.n, dtype=torch.float32, device=self.device)
        for s in range(0, self.n, self.chunk):
            rows = torch.arange(s, min(s + self.chunk, self.n))
            out[rows.to(self.device)] = self._score_rows(
                model, cache, lp0, plen, rows)
        return out

    @torch.no_grad()
    def argmax(self, model, prompt, banned=None, allowed_idx=None, prune=True):
        """Highest-scoring legal word.

        With `prune=True` this is the *same* word `score_all().argmax()` gives;
        the bound is exact, not a heuristic. Returns
        `(word, score, n_chunks_scored, margin)`.

        `margin` is the gap to the runner-up **among words actually scored**.
        With pruning on, unvisited words are known to score below the winner but
        could sit above the runner-up, so `margin` is an upper bound on the true
        margin. It is a diagnostic, never an input to a decision.
        """
        cache, lp0, plen = self._prompt_pass(model, prompt)

        bound = lp0[self.first_tok].clone()                      # [N] upper bd
        # `ban_mask` marks words that must NOT be returned, for any reason.
        # It is applied to the bound (so they sort last and prune early) AND to
        # the scores (so one cannot win from the tail of a live chunk). Masking
        # only the bound is a real bug we shipped once: the excluded word still
        # received a genuine score and could come out on top.
        ban_mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)

        if allowed_idx is not None:
            keep = torch.zeros(self.n, dtype=torch.bool, device=self.device)
            keep[torch.as_tensor(np.asarray(allowed_idx), device=self.device)] = True
            ban_mask |= ~keep
            bound = bound.masked_fill(~keep, NEG_INF)

        if banned:
            hit = [self.index[b] for b in banned if b in self.index]
            if hit:
                ix = torch.tensor(hit, device=self.device)
                bound[ix] = NEG_INF     # sorts them last
                ban_mask[ix] = True     # and removes them from the scores

        # Only ever visit words that could win. Sorting the whole vocabulary and
        # relying on the -inf bound to skip the rest still pushes a full chunk
        # of masked-out words through the model: a 3-word admissible set cost
        # MORE than an unfiltered decision (measured 87s vs 32s on CPU) because
        # both scored 512 rows. Restricting the scan pool to the admissible set
        # makes a small set genuinely cheap, which is the common case once the
        # feedback filter bites.
        if allowed_idx is not None:
            pool = torch.as_tensor(np.asarray(allowed_idx), device=self.device)
            pool = pool[~ban_mask[pool]]
            if pool.numel() == 0:                    # everything banned
                pool = torch.as_tensor(np.asarray(allowed_idx), device=self.device)
        else:
            pool = torch.arange(self.n, device=self.device)
        order = pool[torch.argsort(bound[pool], descending=True)]
        n_scan = int(order.numel())

        best_i, best_s, second = -1, NEG_INF, NEG_INF
        n_chunks = 0
        for s in range(0, n_scan, self.chunk):
            rows = order[s:s + self.chunk]
            if bound[rows[0]] == NEG_INF:
                break                                    # all remaining banned
            if best_s >= bound[rows[0]].item():
                break                # no unvisited word can beat the incumbent
            sc = self._score_rows(model, cache, lp0, plen, rows.cpu())
            # A banned word can still land in the tail of an otherwise-live
            # chunk. Masking the bound alone is not enough -- the score has to
            # be masked too, or the ban is silently ignored.
            sc = sc.masked_fill(ban_mask[rows], NEG_INF)
            n_chunks += 1
            top2 = torch.topk(sc, min(2, sc.numel()))
            if top2.values[0].item() > best_s:
                second = max(second, best_s)
                if top2.values.numel() > 1:
                    second = max(second, top2.values[1].item())
                best_s = top2.values[0].item()
                best_i = rows[top2.indices[0]].item()
            elif top2.values[0].item() > second:
                second = top2.values[0].item()

        margin = None if second == NEG_INF else best_s - second
        return self.words[best_i], best_s, n_chunks, margin

    @torch.no_grad()
    def rank_of(self, scores, words):
        """1-based ranks of `words` under a full `score_all` vector."""
        order = torch.argsort(scores, descending=True)
        pos = torch.empty_like(order)
        pos[order] = torch.arange(order.numel(), device=order.device)
        return {w: int(pos[self.index[w]].item()) + 1
                for w in words if w in self.index}

    # -- feedback-consistent selection (Phase 7) ----------------------------
    @torch.no_grad()
    def select(self, model, prompt, banned=None, allowed_idx=None, prune=True):
        """Pick a word, optionally restricted to an admissible subset.

        `allowed_idx` is an index array into `self.words`. It is intended to
        carry the FEEDBACK-CONSISTENT set: the legal words that would have
        produced exactly the feedback already observed. That set is a pure
        function of the prompt (history + the public word list) -- it never
        touches the answer list -- so restricting to it is the same class of
        move as restricting to legal words.

        Returns a dict, not a tuple, so callers can record why a word was
        chosen. The distinction matters for interpreting results:

            forced       len(allowed) == 1. The filter determined the word.
                         The model contributed nothing and this must NOT be
                         counted as a model decision.
            model_chosen len(allowed) > 1. The model ranked the admissible
                         words and picked one.

        Reporting these together would let a decoder improvement masquerade as
        a model improvement.
        """
        n_allowed = self.n if allowed_idx is None else int(len(allowed_idx))

        if allowed_idx is not None and n_allowed == 0:
            # Cannot happen while the answer is legal and the feedback honest,
            # but degrade to the full pool rather than crash a 4-hour run.
            allowed_idx, n_allowed = None, self.n

        if allowed_idx is not None and n_allowed == 1:
            i = int(allowed_idx[0])
            return {"word": self.words[i], "score": None, "n_chunks": 0,
                    "margin": None, "n_allowed": 1, "forced": True,
                    "model_chosen": False}

        w, s, nch, margin = self.argmax(model, prompt, banned=banned,
                                        allowed_idx=allowed_idx, prune=prune)
        return {"word": w, "score": s, "n_chunks": nch, "margin": margin,
                "n_allowed": n_allowed, "forced": False, "model_chosen": True}

    # -- correctness --------------------------------------------------------
    @torch.no_grad()
    def self_test(self, model, prompt, n_probe=32, atol=None):
        """Check cache-reuse scoring against naive full-sequence scoring.

        This is the guard on the KV-cache shim above. A shim that mishandles a
        transformers version produces essentially random scores -- wrong by
        many nats, with the ordering destroyed. That is the failure this must
        catch, and it is enormous.

        What it must NOT flag is float16 rounding. The two paths run different
        matmul shapes (one sequence of length P+L, versus a batch of L-token
        rows against an expanded P-token cache), so fp16 reduction order
        differs and summed log-probs disagree at the 0.01-0.1 nat level. That
        is arithmetic noise, not a broken cache.

        So the test is two-sided:
          * absolute deviation under a dtype-aware tolerance, and
          * the two score vectors still rank the probe words the same way
            (correlation ~1). A broken shim cannot preserve the ranking.

        Note this discrepancy is a *validation* artifact only. Every one of the
        12,972 words is scored through the same fast path, so the ranking the
        argmax is read off is internally consistent.

        Probe indices are spread evenly across the whole word list, not taken
        from the front: if the prompt cache were mutated in place by the first
        chunk's forward pass, only words in *later* chunks would be wrong, and
        a probe drawn from the front would miss it entirely.
        """
        dtype = next(model.parameters()).dtype
        if atol is None:
            atol = 0.02 if dtype in (torch.float32, torch.float64) else 0.40

        fast = self.score_all(model, prompt)
        p_ids = self.tok(prompt, return_tensors="pt",
                         add_special_tokens=False)["input_ids"].to(self.device)
        n_probe = min(n_probe, self.n)
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
            f"scorer (corr={corr:.6f}). The KV-cache shim in expand_cache() "
            f"does not match this transformers version -- do not trust "
            f"constrained results.")
        assert worst < atol, (
            f"constrained scorer disagrees with naive scoring by {worst:.4f} "
            f"nats (tol {atol} for dtype {dtype}), across a probe score spread "
            f"of {spread:.1f} nats. Ranking is preserved (corr={corr:.6f}), so "
            f"this looks like arithmetic noise rather than a broken cache -- "
            f"but it is larger than expected. Investigate before trusting the "
            f"numbers.")
        return {"max_abs_dev": worst, "corr": corr, "spread": spread,
                "atol": atol, "dtype": str(dtype), "n_probe": n_probe}

    @torch.no_grad()
    def verify_against_full(self, model, prompt):
        """Assert the pruned argmax equals the unpruned argmax."""
        full = self.score_all(model, prompt)
        w_full = self.words[int(full.argmax().item())]
        w_prune, _, n_chunks, _ = self.argmax(model, prompt, prune=True)
        assert w_full == w_prune, (
            f"pruning changed the answer: full={w_full} pruned={w_prune}. "
            f"The branch-and-bound bound is wrong.")
        return w_full, n_chunks


# ---------------------------------------------------------------------------
class HardModeFilter:
    """The legal words consistent with every piece of feedback received.

    A word `w` is admissible iff, for every past guess `g` with observed
    pattern `p`, `feedback_code(g, w) == p`. That is precisely the set a
    hard-mode Wordle player can compute from their own board.

    WHAT THIS IS NOT: the candidate set. Two sets are easy to conflate and the
    difference is the whole justification --

        candidate set   answers consistent with feedback   pool = 2,315 answers
                        -> uses the ANSWER LIST, privileged, never used here
        hard-mode set   legal guesses consistent with it   pool = 12,972 legal
                        -> a pure function of the prompt + the public word list

    The model is never shown this set, its size, the answer, or the answer
    list. It is a decoder-side restriction on which words may be selected,
    exactly like the legal-word constraint.

    Refinement is incremental, so cost is dominated by the first turn and
    collapses immediately after (measured: 12,972 -> ~211 -> ~7 -> ~2).
    """

    def __init__(self, legal_words_lower, feedback_code_fn):
        self.words = list(legal_words_lower)
        self._fb = feedback_code_fn
        self.history = []

    def refine(self, guess, code):
        """Apply one (guess, feedback) pair. `guess` lower-case, `code` int."""
        g = guess.lower()
        self.words = [w for w in self.words if self._fb(g, w) == code]
        self.history.append((g, code))
        return self

    def indices(self, scorer):
        """Index array into `scorer.words` (which are upper-case)."""
        return np.array([scorer.index[w.upper()] for w in self.words
                         if w.upper() in scorer.index], dtype=np.int64)

    def contains(self, word):
        return word.lower() in self.words

    def __len__(self):
        return len(self.words)
