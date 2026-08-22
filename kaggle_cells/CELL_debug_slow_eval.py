# =============================================================================
# WHY IS ONE DECISION TAKING 12s?  Paste as a cell, run, send me the output.
#
# Phase 7 did the identical work at ~0.6 s/decision. dtype is fp16 and the
# device is cuda:0, so the cause is elsewhere. This times each stage of a
# single decision separately so we can see which one is wrong instead of
# guessing again.
# =============================================================================
import time
import torch

print("=" * 70)
print("A. ENVIRONMENT")
print("=" * 70)
import transformers
print(f"  transformers {transformers.__version__}   torch {torch.__version__}")
print(f"  gpus {torch.cuda.device_count()}  "
      f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
free, total = torch.cuda.mem_get_info()
print(f"  vram free {free/2**30:.1f} / {total/2**30:.1f} GiB")
print(f"  allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB  "
      f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB")

M = load_sft()
sc = build_scorer()
base = M.base_model.model if hasattr(M, "base_model") else M
cfg = base.config
print(f"  attn impl : {getattr(cfg, '_attn_implementation', '?')}"
      f"   <-- 'sdpa' or 'flash_attention_2' is fast, 'eager' is SLOW")
print(f"  dtype     : {next(M.parameters()).dtype}")
print(f"  use_cache : {cfg.use_cache}")
print(f"  scorer    : n={sc.n} chunk={sc.chunk} max_len={sc.max_len} "
      f"device={sc.device}")

print("\n" + "=" * 70)
print("B. ONE DECISION, STAGE BY STAGE")
print("=" * 70)
g = GameState(VAL_ANSWERS[0], VOCAB.n_answers)
prompt = g.prompt(1)
ntok = len(TOKENIZER(prompt, add_special_tokens=False)["input_ids"])
print(f"  prompt tokens: {ntok}")

torch.cuda.synchronize()
t = time.perf_counter()
cache, lp0, plen = sc._prompt_pass(M, prompt)
torch.cuda.synchronize()
t_prompt = time.perf_counter() - t
print(f"  prompt forward      : {t_prompt*1000:8.1f} ms")

rows = torch.arange(sc.chunk)
torch.cuda.synchronize()
t = time.perf_counter()
big = expand_cache(cache, sc.chunk)
torch.cuda.synchronize()
t_expand = time.perf_counter() - t
kb = sum(k.numel() * k.element_size() for k, _ in _cache_layers(big)) * 2
print(f"  expand_cache({sc.chunk})   : {t_expand*1000:8.1f} ms   "
      f"({kb/2**20:.0f} MiB of KV copied)")
del big

torch.cuda.synchronize()
t = time.perf_counter()
s1 = sc._score_rows(M, cache, lp0, plen, rows)
torch.cuda.synchronize()
t_chunk = time.perf_counter() - t
print(f"  score one chunk     : {t_chunk*1000:8.1f} ms")

torch.cuda.synchronize()
t = time.perf_counter()
r = sc.select(M, prompt, allowed_idx=None, prune=True)
torch.cuda.synchronize()
t_full = time.perf_counter() - t
print(f"  FULL decision       : {t_full*1000:8.1f} ms   "
      f"chunks {r['n_chunks']}/{-(-sc.n//sc.chunk)}  -> {r['word']}")
print(f"  implied per chunk   : {t_full/max(r['n_chunks'],1)*1000:8.1f} ms")
print()
print(f"  expected if healthy : ~600 ms total, 1-2 chunks")
print(f"  if chunks is 26     : pruning is not engaging")
print(f"  if per-chunk is big : the forward pass itself is slow (attn impl?)")

print("\n" + "=" * 70)
print("C. DOES CHUNK SIZE MATTER?")
print("=" * 70)
orig = sc.chunk
for c in (128, 256, 512, 1024):
    sc.chunk = c
    torch.cuda.synchronize(); t = time.perf_counter()
    r = sc.select(M, prompt, allowed_idx=None, prune=True)
    torch.cuda.synchronize()
    print(f"  chunk {c:>5}: {(time.perf_counter()-t)*1000:8.1f} ms   "
          f"{r['n_chunks']} chunks  -> {r['word']}")
sc.chunk = orig

print("\n" + "=" * 70)
print("D. IS IT THE FILTER, NOT THE MODEL?")
print("=" * 70)
# HardModeFilter.refine is pure Python over 12,972 words on the FIRST call.
f = HardModeFilter(LEGAL_LOWER, feedback_code)
t = time.perf_counter()
f.refine("salet", feedback_code("salet", VAL_ANSWERS[0].lower()))
t_refine = time.perf_counter() - t
print(f"  first refine (12,972 words): {t_refine*1000:8.1f} ms")
t = time.perf_counter()
idx = f.indices(sc)
print(f"  indices() on {len(f):>5} words     : {(time.perf_counter()-t)*1000:8.1f} ms")
print(f"  -> x246 games = {t_refine*246:.0f}s per turn if this is the bottleneck")

print("\n" + "=" * 70)
print("E. A FILTERED DECISION (should be ~100x faster than unfiltered)")
print("=" * 70)
for _ in range(2):
    f.refine("crane", feedback_code("crane", VAL_ANSWERS[0].lower()))
idx = f.indices(sc)
torch.cuda.synchronize(); t = time.perf_counter()
r = sc.select(M, prompt, allowed_idx=idx, prune=True)
torch.cuda.synchronize()
print(f"  {len(idx)} admissible: {(time.perf_counter()-t)*1000:8.1f} ms  "
      f"chunks {r['n_chunks']}  forced={r['forced']}  -> {r['word']}")

del M
torch.cuda.empty_cache()
print("\ndone - paste all of the above")
