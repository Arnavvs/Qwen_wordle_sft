"""make_grpo_notebook.py -> wordle_phase11_grpo_kaggle.ipynb

Generates the Phase 11 GRPO notebook. Build cells here; never edit the .ipynb.

WHY THIS WAS REWRITTEN
----------------------
The first version OOMed on a T4 after 15 minutes. Root cause, measured rather
than guessed: it scored actions with

    lp = torch.log_softmax(model(...).logits.float(), dim=-1)

which materialises `(batch, seq_len, 151936)` in fp32 — 3.62 GiB for a 32x200
chunk, exactly the 3.12 GiB allocation that failed. The fix is `logits_to_keep`:
the completion is 2-3 tokens at the end of the sequence, so only the last few
logit positions are ever needed, and asking for those drops the same work to
~19 MiB. See section 5.

WHAT THE NOTEBOOK DOES
----------------------
Loads the SFT adapter, adds a fresh LoRA, and for each state computes a policy
distribution over that state's action set, forms group-relative advantages from
precomputed exact rewards, and updates with a k3 KL penalty toward the frozen
SFT reference.

It does NOT score words and it does NOT play games. Rewards come from
`build_grpo_tasks.py`; gameplay is measured by the Phase 9 harness. One
measurement path for the whole project — forking it is what voided Phase 8 v3.

    python phase11_grpo/make_grpo_notebook.py
"""
import _paths  # noqa: F401

import json
import os

NB = "phase11_grpo/wordle_phase11_grpo_kaggle.ipynb"
CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


# ===========================================================================
md(r"""
# Wordle Phase 11 — GRPO

Last planned phase. Read `phase11_grpo/RUN.md` first; the outcome readings are
pre-registered there.

### How to run this

1. Attach **both** datasets: `wordle-sft-package-v2` (has
   `sft_package/data/grpo_tasks/`) and `wordle-adapters-v2` (has
   `tree_salet_endgame/`).
2. Accelerator: **GPU T4 x2**. Only one GPU is used — a 0.5B model does not
   need sharding, and splitting it would only add transfers. The second card
   sitting idle is expected, not a misconfiguration.
3. **Restart the session before re-running after any crash.** A dead run's
   allocations stay held by the Python process: the GPU shows ~14.5 GiB in use
   at 0% utilisation, and the next run OOMs immediately no matter what the code
   does. Run -> Restart & Clear Cell Outputs.
4. Leave `SMOKE = True` for the first run — it does ~40 steps in a couple of
   minutes and prints a memory report. If that is clean, set `SMOKE = False`
   and Run All.
5. Download `tree_salet_endgame_grpo.zip`, add it to `wordle-adapters-v2`
   keeping the folder name, then score it with the Phase 9 harness.

### The number this has to beat

`decision_budget.py` priced all 922 of the model's decisions across the 246
held-out games. Perfect action selection in the *entire* restricted regime buys
**0.0698 guesses** (3.7642 → 3.6944). Phase 8's DPO drift cost 0.1708. GRPO is
here because it is the only method left that reaches the other **58% of
decisions**, where the model picks freely.

### Rewards are data, not code

Every action value was computed exactly, offline. This notebook samples, looks
up, and updates. It never scores a word.
""")

# ===========================================================================
md(r"""
---
# 1. Config
""")

code(r'''
import os, sys, json, math, time, random, subprocess, importlib, zipfile
from collections import Counter, defaultdict
import numpy as np

# Must be set BEFORE torch initialises its allocator. The first OOM here failed
# on a 34 MiB request with 297 MiB reserved-but-unallocated - textbook
# fragmentation, which is what this setting addresses.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================ EDIT THIS ====================================
SMOKE        = False     # API benchmark run: full pass. Set True for a ~2 min
                         # memory smoke when running interactively.

SEED         = 20260825
BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_ADAPTER  = "tree_salet_endgame"   # the 3.7642 policy. NEVER substitute.

# ---- estimator ------------------------------------------------------------
# "exact"  : use the WHOLE action set, weighting each action by pi(a). This is
#            the G -> infinity limit of the GRPO group estimator, and it is
#            available here only because the action set is enumerable (median 7
#            actions) and every reward is precomputed. Zero sampling variance,
#            same cost. Recommended.
# "group"  : textbook GRPO -- sample GROUP_SIZE actions from pi and use the
#            sample mean as the baseline. Kept so the two can be compared.
ESTIMATOR    = "exact"
GROUP_SIZE   = 8         # only used when ESTIMATOR == "group"

# ---- advantage ------------------------------------------------------------
# Dividing by the group std is standard GRPO but has two problems here. It
# introduces the difficulty bias Dr. GRPO identifies, and with 89.5% of our
# tasks having tied optima it produces 0/0 on any group that samples only tied
# actions. False = subtract the mean only.
SCALE_REWARDS = False

# ---- KL to the frozen SFT reference ---------------------------------------
# TRL now defaults beta=0, but Phase 8's failure here was drift, so the run
# that follows a known drift failure keeps an explicit leash. k3 estimator.
GRPO_BETA    = 0.04
GRPO_LR      = 1e-6      # deliberately below the DPO run's 5e-6
GRPO_EPOCHS  = 1
TASKS_PER_STEP = 8       # states per optimizer step (gradient accumulation)
MAX_SEQ_LEN  = 640
ADV_EPS      = 1e-6

# Two chunk sizes, because the two passes cost very different amounts. The
# no-grad pass stores no activations, so it can be wide; the backward pass
# stores them for every action in the chunk, so it must be narrow. Lower
# GRAD_CHUNK first if you still OOM.
ACTION_CHUNK = 32        # actions per forward pass, no-grad passes
GRAD_CHUNK   = 8         # actions per backward chunk

ATTN_IMPL    = "sdpa"    # eager materialises (B, heads, T, T) and is 5-10x
                         # slower, silently - a known trap in this project
GRAD_CHECKPOINT = True   # trades ~30% speed for a large activation saving
CKPT_EVERY   = 150       # optimizer steps between adapter checkpoints
EVAL_EVERY   = 150

LORA_R, LORA_ALPHA = 16, 32
LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

WORK_DIR     = "/kaggle/working"
GRPO_ADAPTER = "tree_salet_endgame_grpo"
# ===========================================================================

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except Exception: pass
set_seed()
print(f"estimator={ESTIMATOR}  scale_rewards={SCALE_REWARDS}  beta={GRPO_BETA}")
print(f"lr={GRPO_LR}  epochs={GRPO_EPOCHS}  tasks/step={TASKS_PER_STEP}  SMOKE={SMOKE}")
''')

# ===========================================================================
md(r"""
---
# 2. Environment

Kaggle ships torchao 0.10 while PEFT requires >0.16, and PEFT raises that
**lazily** — inside `from_pretrained`, not at import — so an unpatched notebook
dies at the adapter merge rather than at the top. Same fix Phases 8 and 9 use.
""")

code(r'''
import torch, torch.nn.functional as F
import transformers, peft

def fix_torchao_peft_conflict():
    try: import peft.import_utils as piu
    except Exception as e: return f"unavailable ({e})"
    try: piu.is_torchao_available(); return "no conflict"
    except ImportError: pass
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                   check=False)
    importlib.invalidate_caches()
    try: piu.is_torchao_available(); return "resolved"
    except ImportError: pass
    piu.is_torchao_available = lambda *a, **k: False
    try:
        import peft.tuners.lora.torchao as t
        t.is_torchao_available = lambda *a, **k: False
    except Exception: pass
    return "patched"

print("torchao:", fix_torchao_peft_conflict())
print("transformers", transformers.__version__, "| peft", peft.__version__)
print("torch", torch.__version__)
assert torch.cuda.is_available(), "no GPU - set Accelerator to GPU T4"
print("gpu:", torch.cuda.get_device_name(0),
      f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)")

def mem(tag=""):
    a = torch.cuda.memory_allocated()/2**30
    r = torch.cuda.max_memory_allocated()/2**30
    print(f"    [mem] {tag:22s} allocated {a:5.2f} GiB   peak {r:5.2f} GiB")
''')

# ===========================================================================
md(r"""
---
# 3. Locate the data and the adapter

Recursive search for marker files, not a guess at mount depth. Phase 10's first
push failed globbing `/kaggle/input/*`; the real mount is
`/kaggle/input/datasets/<owner>/<slug>/…`, three levels down.

The adapter is matched by **directory basename**. Substituting a different
adapter already produced one wrong published result in this project, so a miss
is a hard stop rather than a fallback.
""")

code(r'''
REQ = ["sft_package/data/grpo_tasks/train.jsonl",
       "sft_package/data/grpo_tasks/validation.jsonl",
       "sft_package/data/grpo_tasks/manifest.json"]

def _has(d):
    try: return all(os.path.exists(os.path.join(d, f)) for f in REQ)
    except OSError: return False

def find_root():
    for root in ["/kaggle/input", "."]:
        if not os.path.isdir(root): continue
        if _has(root): return root
        for dp, dn, _ in os.walk(root):
            dn[:] = [d for d in dn if not d.startswith(".")]
            if _has(dp): return dp
    lines = ["could not locate the GRPO task set. REQUIRED:"] + ["    "+f for f in REQ]
    for base in ("/kaggle/input", "."):
        lines.append(f"  tree under {base}:")
        if not os.path.isdir(base): lines.append("    <missing>"); continue
        n = 0
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if not d.startswith(".")]
            if dp.rstrip("/").count("/") - base.rstrip("/").count("/") > 3:
                dn[:] = []; continue
            lines.append(f"    {dp}/  ({len(dn)} dirs, {len(fn)} files)"); n += 1
            if n > 60: lines.append("    ... truncated"); break
    raise SystemExit("\n".join(lines))

DATA_ROOT = find_root()
TASK_DIR  = os.path.join(DATA_ROOT, "sft_package/data/grpo_tasks")
print("data root:", DATA_ROOT)

def find_adapter(name):
    for root in ["/kaggle/input", WORK_DIR]:
        if not os.path.isdir(root): continue
        for dp, dn, fn in os.walk(root):
            if os.path.basename(dp) == name and "adapter_config.json" in fn:
                return dp
    raise SystemExit(
        f"adapter {name!r} not found. Attach wordle-adapters-v2. Do NOT point "
        f"this at a different adapter - that already cost this project one "
        f"wrong published result.")

SFT_PATH = find_adapter(SFT_ADAPTER)
print("SFT adapter:", SFT_PATH)
''')

# ===========================================================================
md(r"""
---
# 4. Tasks

Audited locally by `verify_grpo_tasks.py`, but the dataset attached here could
differ from the audited copy, so the load-bearing invariants are re-asserted
against the file actually present.
""")

code(r'''
def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

TRAIN = load_jsonl(os.path.join(TASK_DIR, "train.jsonl"))
VAL   = load_jsonl(os.path.join(TASK_DIR, "validation.jsonl"))
MANIFEST = json.load(open(os.path.join(TASK_DIR, "manifest.json"), encoding="utf-8"))

assert all(len(t["actions"]) == len(t["values"]) for t in TRAIN+VAL), "ragged task"
assert all(len(t["actions"]) >= 2 for t in TRAIN+VAL), "task with <2 actions"
assert all(max(t["values"]) - min(t["values"]) > 1e-12 for t in TRAIN+VAL), \
    "task with no value spread would contribute pure noise"
assert not any("Possible answers" in t["prompt"] for t in TRAIN+VAL), "count leaked"
assert not ({t["meta"]["state_key"] for t in TRAIN}
            & {t["meta"]["state_key"] for t in VAL}), "train/val state leakage"
assert all(all(a <= b + 1e-12 for a, b in zip(t["values"], t["values"][1:]))
           for t in TRAIN+VAL), "values must be sorted ascending"

if SMOKE:
    TRAIN = TRAIN[:320]; VAL = VAL[:60]
    print("*** SMOKE MODE: truncated task set, results are NOT a result ***")

na = np.array([len(t["actions"]) for t in TRAIN])
print(f"{len(TRAIN)} train / {len(VAL)} val tasks   invariants OK")
print(f"  buckets      : {MANIFEST['buckets']}")
print(f"  actions/task : median {int(np.median(na))}  mean {na.mean():.1f}  max {na.max()}")
print(f"  exact-tree   : {sum(1 for t in TRAIN if t['meta']['exact'])}/{len(TRAIN)}")
print(f"  tied optimum : {100*np.mean([t['meta']['n_optimal_actions']>1 for t in TRAIN]):.1f}% of tasks")
''')

# ===========================================================================
md(r"""
---
# 5. Action scoring — and the fix for the OOM that killed the first run

A "completion" here is one word: 2–3 tokens at the very end of the sequence. So
only the last few logit positions are ever needed. The first version ignored
that and ran

```python
lp = torch.log_softmax(model(...).logits.float(), dim=-1)   # (B, T, 151936) fp32
```

which is **3.62 GiB** for a 32×200 chunk — precisely the 3.12 GiB allocation
that failed. Two changes fix it, both exact rather than approximations:

* **`logits_to_keep`** — ask the model for only the last *k* positions. Same
  arithmetic, ~19 MiB instead of 3.62 GiB.
* **gather + logsumexp** instead of a full `log_softmax`, so the per-token
  logprob never materialises a vocabulary-sized fp32 tensor.

All actions in a task share a prompt and are padded to a common length, so
every row has the same total length and one `logits_to_keep` covers them all.
""")

code(r'''
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

TOK = AutoTokenizer.from_pretrained(BASE_MODEL)
if TOK.pad_token is None: TOK.pad_token = TOK.eos_token
PAD = TOK.pad_token_id

# transformers renamed this argument (num_logits_to_keep -> logits_to_keep).
# Resolved once on first use and cached; None means "not resolved yet".
LOGITS_KW = None

def _encode_task(task):
    """(prompt_ids, [action_token_lists], max_action_len) for one task."""
    acts = [TOK(" " + w.upper(), add_special_tokens=False)["input_ids"] + [TOK.eos_token_id]
            for w in task["actions"]]
    la = max(len(a) for a in acts)
    p = TOK(task["prompt"], add_special_tokens=False)["input_ids"]
    p = p[-(MAX_SEQ_LEN - la):]
    return p, acts, la

def _forward_logits(model, ids, attn, keep):
    """Model logits for the last `keep` positions, trying both arg spellings."""
    global LOGITS_KW
    if LOGITS_KW is not None:
        try:
            return model(input_ids=ids, attention_mask=attn,
                         **{LOGITS_KW: keep}).logits
        except TypeError:
            LOGITS_KW = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            out = model(input_ids=ids, attention_mask=attn, **{kw: keep}).logits
            LOGITS_KW = kw
            return out
        except TypeError:
            continue
    # Old transformers: no such arg. Slice after the fact - correct, just
    # heavier. The run will still fit because ACTION_CHUNK is small.
    return model(input_ids=ids, attention_mask=attn).logits[:, -keep:]

def _logprobs_for(model, p, part, la):
    """Summed log P(action tokens) for ONE chunk of actions sharing prompt `p`.

    All rows are padded to the same total length P + la, so a single
    `logits_to_keep = la + 1` window covers every row's action tokens.
    """
    P, B = len(p), len(part)
    ids  = torch.full((B, P + la), PAD, dtype=torch.long)
    attn = torch.zeros((B, P + la), dtype=torch.long)
    tgt  = torch.full((B, la), -100, dtype=torch.long)
    pt = torch.tensor(p)
    for i, a in enumerate(part):
        ids[i, :P] = pt
        ids[i, P:P+len(a)] = torch.tensor(a)
        attn[i, :P+len(a)] = 1
        tgt[i, :len(a)] = torch.tensor(a)
    ids, attn, tgt = ids.cuda(), attn.cuda(), tgt.cuda()

    # positions P-1 .. P+la-2 predict tokens at P .. P+la-1
    logits = _forward_logits(model, ids, attn, la + 1)[:, :-1, :]   # (B, la, V)
    mask = tgt != -100
    # gather + logsumexp: never a full log_softmax over the 151,936-word vocab,
    # which is what OOMed the first version (3.62 GiB for one 32x200 chunk)
    picked = torch.gather(logits, 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1).float()
    lse = torch.logsumexp(logits.float(), dim=-1)
    return ((picked - lse) * mask).sum(-1)


def action_logprobs(model, task, chunk=None):
    """log pi(action | prompt) per action, as a (n_actions,) float32 tensor."""
    chunk = chunk or ACTION_CHUNK
    p, acts, la = _encode_task(task)
    return torch.cat([_logprobs_for(model, p, acts[s:s+chunk], la)
                      for s in range(0, len(acts), chunk)])
print("action scoring ready")
''')

# ===========================================================================
md(r"""
---
# 6. Model

SFT adapter merged into the base, fresh LoRA on top; the reference is that same
model with the new adapter disabled — so the reference **is** the 3.7642
policy. Identical construction to the Phase 8 DPO cell, so the KL term means
what it meant there.
""")

code(r'''
set_seed()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print("merging SFT adapter ...")
merged = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16,
                                         attn_implementation=ATTN_IMPL),
    SFT_PATH).merge_and_unload()
policy = get_peft_model(merged, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
    target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
for _, q in policy.named_parameters():
    if q.requires_grad and q.dtype == torch.float16:
        q.data = q.data.float()
policy.print_trainable_parameters()
policy.config.use_cache = False
if GRAD_CHECKPOINT:
    # enable_input_require_grads is required or checkpointing silently produces
    # no gradients for a LoRA-only model: the inputs to the checkpointed blocks
    # are leaves with requires_grad=False, so nothing is recomputed.
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print("gradient checkpointing: on")
policy = policy.cuda()
print("attention:", getattr(policy.config, "_attn_implementation", "?"))
mem("after model load")

# sanity: a fresh LoRA is a no-op, so policy and reference must agree exactly
with torch.no_grad():
    a = action_logprobs(policy, TRAIN[0])
    with policy.disable_adapter():
        b = action_logprobs(policy, TRAIN[0])
d = (a - b).abs().max().item()
print(f"fresh-adapter identity check: max|policy - ref| = {d:.2e}  "
      f"{'OK' if d < 1e-3 else '*** NOT A NO-OP ***'}")
mem("after first scoring")
''')

# ===========================================================================
md(r"""
---
# 7. Reference log-probabilities, computed once

The reference is frozen, so its log-probabilities never change. Computing them
up front removes an `disable_adapter()` forward pass from every training step
and keeps two autograd graphs from ever being alive together.
""")

code(r'''
def precompute_ref(rows, tag):
    out = []
    t0 = time.perf_counter()
    with torch.no_grad(), policy.disable_adapter():
        for i, t in enumerate(rows):
            out.append(action_logprobs(policy, t).detach().cpu())
            if (i+1) % 500 == 0:
                print(f"    {tag} {i+1}/{len(rows)}  {time.perf_counter()-t0:.0f}s",
                      flush=True)
    print(f"  {tag}: {len(rows)} tasks in {time.perf_counter()-t0:.0f}s")
    return out

REF_TRAIN = precompute_ref(TRAIN, "train")
REF_VAL   = precompute_ref(VAL, "val")
mem("after ref precompute")
''')

# ===========================================================================
md(r"""
---
# 8. The GRPO objective

```
r_i    = -value_i                            (lower value = better = higher reward)
A_i    = r_i - mean(r)          [ / std(r) if SCALE_REWARDS ]
KL     = k3 estimator:  exp(ref - pi) - (ref - pi) - 1        (Schulman)
loss   = -E[A * log pi]  +  beta * KL
```

**Two estimators.** `"group"` is textbook GRPO: draw `G` actions from π and use
the sample mean as the baseline. `"exact"` uses the whole action set weighted by
π — the `G → ∞` limit of the same estimator, with zero sampling variance and no
extra cost, available only because the action set here is enumerable and every
reward is precomputed. In general LLM RL neither of those holds, which is why
GRPO samples; here they do.

**Why `SCALE_REWARDS = False` by default.** Dividing by the group std is
standard GRPO but Dr. GRPO shows it introduces a difficulty bias, and in this
task it is worse than that: 89.5% of states have tied optima, so a sampled
group can easily contain only tied actions, giving std = 0 and a 0/0 advantage.
Subtracting the mean alone has neither problem. `zero_adv` is logged so you can
see how often a state carried no signal.

### The backward pass is chunked, and that is what fixes the second OOM

The first rewrite still died — not on a huge allocation this time, but on a
**34 MiB** request with 14.11 GiB already live. That is activation memory: it
scored every action of a task in one graph and backpropped through all of them
at once, so a 48-action turn-2 state held 48 sequences of activations
simultaneously.

The fix comes from the shape of the loss. With `pi` detached, the derivative
with respect to each action's log-probability is a **constant**:

```
w_j         = pi_j * adv_j
d(pg)/dlp_j = -(w_j - pi_j * sum_i w_i)
d(kl)/dlp_j =  pi_j * (1 - exp(ref_j - lp_j))
```

So the whole objective has the same gradient as the linear surrogate
`sum_j g_j * lp_j` — and a sum can be chunked. The notebook therefore does one
cheap `no_grad` pass to get `g`, then backpropagates `GRAD_CHUNK` actions at a
time. **Peak memory is set by `GRAD_CHUNK`, not by the action count.** The
derivation was checked against finite differences to 1.7e-10.

Two supporting fixes: `attn_implementation="sdpa"` (eager materialises
`(B, heads, T, T)` per layer) and gradient checkpointing.
""")

code(r'''
def action_grad_weights(task, ref_lp, lp):
    """Per-action gradient weight g, and the scalar loss, both under no_grad.

    The whole point: d(loss)/d(lp_j) turns out to be a CONSTANT per action, so
    the backward pass becomes `sum_j g_j * lp_j` -- a linear function that can
    be chunked without changing the gradient. Derivation, with pi detached:

        w_j        = pi_j * adv_j                    (advantage weight)
        d(pg)/dlp_j = -(w_j - pi_j * sum_i w_i)
        d(kl)/dlp_j =  pi_j * (1 - exp(ref_j - lp_j))

    Verified against finite differences to 1.7e-10.
    """
    vals = torch.tensor(task["values"], dtype=torch.float32, device=lp.device)
    rew  = -vals
    logpi = torch.log_softmax(lp, dim=-1)
    pi    = logpi.exp()

    if ESTIMATOR == "exact":
        base = (pi * rew).sum()
        adv  = rew - base
        if SCALE_REWARDS:
            adv = adv / ((pi * (rew - base) ** 2).sum().sqrt() + ADV_EPS)
        w = pi * adv
        g_pg = -(w - pi * w.sum())
        pg_val = -(w * logpi).sum()
        spread = adv.abs().max()
    else:
        idx = torch.multinomial(pi, GROUP_SIZE, replacement=True)
        g   = rew[idx]
        adv = g - g.mean()
        if SCALE_REWARDS:
            adv = adv / (g.std() + ADV_EPS)
        c = torch.zeros_like(rew).index_add_(0, idx, adv)      # per-action sum
        g_pg = -(c - pi * adv.sum()) / GROUP_SIZE
        pg_val = -(adv * logpi[idx]).mean()
        spread = g.std()

    d  = ref_lp - lp
    kl = (pi * (d.exp() - d - 1.0)).sum()
    g_kl = pi * (1.0 - d.exp())

    g = g_pg + GRPO_BETA * g_kl
    best = vals.min()
    metrics = dict(
        pg=float(pg_val.item()), kl=float(kl.item()),
        loss=float((pg_val + GRPO_BETA * kl).item()),
        p_opt=float(pi[vals <= best + 1e-12].sum().item()),
        zero_adv=float(spread.item() < 1e-9))
    return g, metrics


def grpo_step(task, ref_lp_cpu, scale):
    """One state: compute weights with no grad, then backward in small chunks.

    Memory is bounded by GRAD_CHUNK, NOT by the number of actions. The previous
    version backpropped through every action of a task at once, so a 48-action
    turn-2 state held 48 sequences of activations simultaneously and the T4 ran
    out on a 34 MiB request with 14.11 GiB live.
    """
    with torch.no_grad():
        lp0 = action_logprobs(policy, task, ACTION_CHUNK)
        ref = ref_lp_cpu.to(lp0.device)
        g, metrics = action_grad_weights(task, ref, lp0)

    p, acts, la = _encode_task(task)
    for s in range(0, len(acts), GRAD_CHUNK):
        part = acts[s:s + GRAD_CHUNK]
        lp_c = _logprobs_for(policy, p, part, la)
        surrogate = (g[s:s + len(part)] * lp_c).sum() * scale
        surrogate.backward()
        del lp_c, surrogate
    return metrics


@torch.no_grad()
def evaluate(rows, tag):
    """Greedy top-1 rate against the optimal action set, on held-out states."""
    ok = ex_ok = ex_n = 0
    for t in rows:
        lp = action_logprobs(policy, t, ACTION_CHUNK)
        good = t["values"][int(lp.argmax())] <= min(t["values"]) + 1e-12
        ok += good
        if t["meta"]["exact"]: ex_n += 1; ex_ok += good
    r = ok / len(rows)
    print(f"    [{tag}] optimal-action rate {100*r:.1f}%   "
          f"(exact-tree only {100*ex_ok/max(ex_n,1):.1f}%, n={ex_n})")
    return r
print("objective ready")
''')

# ===========================================================================
md(r"""
---
# 9. Train

**The stop rule is pre-registered and it is not in this notebook.** The
decision is the paired 246-game rollout run afterwards by the Phase 9 harness,
not loss, not reward, not the optimal-action rate below. Phase 8's preference
accuracy saturated at step 75 of 466 and everything after was drift that only
the rollout could see — so this checkpoints regularly and lets the rollout pick
a step.
""")

code(r'''
OUT = os.path.join(WORK_DIR, GRPO_ADAPTER)
LOG = []
set_seed()
torch.cuda.reset_peak_memory_stats()
opt = torch.optim.AdamW([q for q in policy.parameters() if q.requires_grad],
                        lr=GRPO_LR)
order = list(range(len(TRAIN)))
random.Random(SEED).shuffle(order)
total = (len(order) * GRPO_EPOCHS) // TASKS_PER_STEP
print(f"{len(order)} tasks x {GRPO_EPOCHS} epoch(s) -> ~{total} optimizer steps\n")

print("before training:")
rate0 = evaluate(VAL, "val")

step = 0; run = []; t0 = time.perf_counter()
for ep in range(GRPO_EPOCHS):
    for k, ti in enumerate(order):
        # grpo_step does its own chunked backward; nothing to backward here
        run.append(grpo_step(TRAIN[ti], REF_TRAIN[ti], 1.0 / TASKS_PER_STEP))
        if (k + 1) % TASKS_PER_STEP == 0:
            torch.nn.utils.clip_grad_norm_(
                [q for q in policy.parameters() if q.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                m = {j: float(np.mean([r[j] for r in run])) for j in run[0]}
                LOG.append(dict(step=step, **{j: round(v, 5) for j, v in m.items()}))
                print(f"  step {step:>4}  pg {m['pg']:+.4f}  kl {m['kl']:.5f}  "
                      f"p_opt {m['p_opt']:.3f}  zero_adv {m['zero_adv']:.2f}  "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f}G  "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
                run = []
                torch.cuda.reset_peak_memory_stats()
            if step % CKPT_EVERY == 0:
                policy.save_pretrained(f"{OUT}_step{step}")
                print(f"    checkpoint -> {OUT}_step{step}")
            if step % EVAL_EVERY == 0:
                evaluate(VAL, f"val@{step}")

policy.save_pretrained(OUT)
mem("after training")
print("\nafter training:")
rate1 = evaluate(VAL, "val")
print(f"\noptimal-action rate on HELD-OUT states: {100*rate0:.1f}% -> {100*rate1:.1f}%")
print("This is a PROXY. The verdict is the paired 246-game rollout.")
if LOG:
    print(f"KL drift: {LOG[0]['kl']:.5f} -> {LOG[-1]['kl']:.5f}   "
          f"(rising KL with a flat rollout = drift, the Phase 8 failure mode)")
''')

# ===========================================================================
md(r"""
---
# 10. Package
""")

code(r'''
if SMOKE:
    print("SMOKE run - not packaging. Set SMOKE = False and Run All for the real run.")
else:
    res = os.path.join(WORK_DIR, "results_phase11")
    os.makedirs(res, exist_ok=True)
    json.dump({"config": {
                   "estimator": ESTIMATOR, "group_size": GROUP_SIZE,
                   "scale_rewards": SCALE_REWARDS, "beta": GRPO_BETA,
                   "lr": GRPO_LR, "epochs": GRPO_EPOCHS,
                   "tasks_per_step": TASKS_PER_STEP, "steps": step,
                   "n_train": len(TRAIN), "n_val": len(VAL),
                   "sft_adapter": SFT_ADAPTER,
                   "seconds": round(time.perf_counter()-t0, 1)},
               "task_manifest": MANIFEST, "log": LOG,
               "val_optimal_action_rate": {"before": rate0, "after": rate1}},
              open(os.path.join(res, "grpo_results.json"), "w"), indent=2)

    zp = os.path.join(WORK_DIR, f"{GRPO_ADAPTER}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fn in os.walk(OUT):
            for f in fn:
                full = os.path.join(dp, f)
                z.write(full, os.path.join(GRPO_ADAPTER,
                                           os.path.relpath(full, OUT)))
    print(f"adapter zip: {zp}  ({os.path.getsize(zp)/2**20:.1f} MiB)")
    print(f"""
NEXT STEPS
  1. Download {GRPO_ADAPTER}.zip
  2. Add it to arnavyrr/wordle-adapters-v2 as a new version, KEEPING the
     folder name {GRPO_ADAPTER}/ -- the harness matches adapters by basename
     and refuses to substitute.
  3. Score it with the Phase 9 harness:
         ARMS            = ["sft", "sft_grpo"]
         VARIANTS_TO_RUN = ["baseline"]
         N_GAMES         = 246
     The sft + baseline gate MUST reproduce 3.7642. That is the control; if it
     does not come back, nothing else is interpretable.
  4. Read against the pre-registered A/B/C/D in phase11_grpo/RUN.md. Remember
     the ceiling: any gain beyond ~0.07 guesses cannot have come from the
     restricted regime, and RUN.md commits to re-running decision_budget.py on
     the new policy to find out where it did come from.
""")
print("RUN COMPLETE")
''')


def main():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(NB), exist_ok=True)
    with open(NB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1)
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    print(f"wrote {os.path.abspath(NB)}")
    print(f"  {len(CELLS)} cells ({n_md} markdown, {len(CELLS)-n_md} code)")
    print(f"  {os.path.getsize(NB)/1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
