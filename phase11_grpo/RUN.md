# Phase 11 — GRPO, and the ceiling it has to beat

**Status: RUN, 2026-08-25. Outcome C — flat.** Sections 1-7 are unchanged
from before the run; the predictions in section 4 are as they were written.
Results are in section 9.

---

## 1. What the previous phases leave

| | |
|---|---:|
| model + adaptive decoder @20 | **3.7642** (242/246) |
| classical `entropy` | 3.4431 |
| **gap to close** | **0.3211** |

Phase 8 tried DPO and made it worse (3.9350, paired t = −3.21). The Phase 8
audit found the data was badly built — 64% trivial pairs, a `competitive` label
uncorrelated with real value, one state contributing 177 rows. A corrected
dataset exists (`phase8_dpo_v3/`, all hard checks pass) and was never trained
on, because measuring the ceiling first made it not worth running.

## 2. The ceiling, measured before choosing a method

`decision_budget.py` prices every one of the model's own 922 decisions across
the 246 games against the best action available *at that exact state*. Unlike
Phase 8's counterfactual — which substitutes the expert and replays, so the
buckets interact and do not sum — this is additive and attributable.

**Where the decisions are:**

```
|adm| 0-1    forced/solved      11.4%
|adm| 2-10   restricted         25.4%
|adm| 11-20  restricted          5.3%
|adm| 21-100 UNrestricted       11.0%
|adm| 100+   UNrestricted       47.0%   <- turn-2 probes
```

**What the restricted regime is worth (exact tree, every decision priced):**

```
restricted decisions with a real choice   153   (0.62 per game)
model already optimal                     131   (85.6%)
  ...where the optimum was a TIE          127   (83.0%)
genuine mistakes                           22   (14.4%)

expected guesses lost to those          0.0698 per game
```

**This is the number that decided the phase.** Perfect action selection in the
entire restricted regime — zero mistakes, zero drift — buys **0.0698 guesses**,
3.7642 → 3.6944, or 22% of the gap. Phase 8's DPO drift cost **0.1708**. The
downside was 2.4x the whole available upside, which is why the corrected DPO
run was cancelled rather than merely improved.

It also explains why on-policy preference mining is not viable: 0.089 mistakes
per game across 2,069 training answers is **~185 pairs**, and 83% of the
decisions the model gets right were ties it could not have got wrong.

**So GRPO is not chosen because it is more powerful. It is chosen because it is
the only method left that reaches the other 58% of decisions** — the
unrestricted ones, where the model picks freely from 12,972 words and where
nearly half its decisions happen.

## 3. The design

### Rewards are precomputed, never computed on the GPU

`build_grpo_tasks.py` emits a task per state:

```jsonc
{"prompt": "<rendered state>",           // identical to every other phase
 "actions": ["CRANE", "SLATE", ...],     // the decoder's real action space
 "values":  [3.125, 3.250, ...],         // expected further guesses, LOWER better
 "meta":    {...}}
```

Reward is `-value`. The notebook samples, looks up, and updates; it never scores
anything. Putting the scorer inside the training loop would fork the measurement
path, which is what voided Phase 8 v3.

### Two value families, and an honest label on the weaker one

| regime | value | exact? |
|---|---|---|
| `\|adm\| <= 20` | adaptive-decoder tree, full remaining depth, integer costs | **yes** |
| `\|adm\| > 20` | exhaustive one-ply `E[remaining]` over all 12,972, top-48 kept | **no** |

Lookahead was measured, not assumed, and rejected: one turn-2 state costs
~13.5s at 83 candidates and grows sharply, so pricing top-48 actions over 150
states runs to tens of hours. The one-ply proxy is weaker but not weak —
greedily minimising `E[remaining]` *is* the classical `expected` solver, 3.4812
against `entropy`'s 3.4431, both far ahead of 3.7642.

Values are in **different units** across regimes (guesses vs candidates). That
is safe here and nowhere else: GRPO normalises advantages *within a group*, a
group is one state, so the two currencies never meet in one update. Any
statistic pooling values across states would be meaningless.

### The loss

```
r_i  = -value_i                               (lower value = better)
A_i  = r_i - mean(r)        [ / std(r) only if SCALE_REWARDS ]
KL   = k3:  exp(ref - pi) - (ref - pi) - 1    (Schulman)
loss = -E[A * log pi]  +  beta * KL
```

`pi_ref` is the frozen SFT policy, as in Phase 8.

**Two estimators, and the default is not the textbook one.** Standard GRPO
samples `G` actions and uses the sample mean as the baseline. Here the action
set is *enumerable* (median 7 actions) and every reward is precomputed, so the
exact expectation is available at the same cost:

```
baseline = sum_a pi(a) r(a)          gradient weight = pi(a) * (r(a) - baseline)
```

That is the `G -> infinity` limit of the group estimator with zero sampling
variance. Verified numerically: the group estimator's gradient weights converge
to it as G grows (max error 0.21 at G=8, 0.003 at G=512). `ESTIMATOR="group"`
is kept so the two can be compared, but `"exact"` is the default because
nothing is gained by sampling a distribution we can sum.

In general LLM RL neither condition holds — the action space is the space of
all token sequences and rewards need a rollout — which is exactly why GRPO
samples. Here they do hold.

**`SCALE_REWARDS = False` by default.** Dividing by the group std is standard
GRPO, but Dr. GRPO shows it introduces a difficulty bias, and in this task it
is worse: 89.5% of states have tied optima, so a group can easily contain only
tied actions, giving std = 0 and a 0/0 advantage. Subtracting the mean alone
has neither problem. `zero_adv` is logged so a signal-free run is visible.

**The k3 KL is exact here, not an estimate.** Summed over the enumerable action
set weighted by pi it telescopes to `sum pi (log pi - log pi_ref)` = the true
KL, because both distributions are normalised over the same set. Checked
numerically against the direct formula.

Completions are one word (2-3 tokens), so this is a contextual bandit per
state, not sequence generation.

### Why ties stop being a problem

95.3% of reachable 2-10 states have two or more exactly-optimal actions. A
pairwise preference cannot represent that and must break the tie arbitrarily —
the single worst property of the DPO framing. Group-relative advantage handles
it natively: tied actions get identical rewards, so identical advantages, and
the gradient is indifferent between them. States where *every* action is equal
are dropped at build time (`all_actions_equal`), since they carry no signal.

### Held-out answers, and why the two regimes handle them differently

No training task may reinforce a held-out answer. *How* that is enforced
differs by regime, and the difference is not cosmetic — the first draft got it
wrong and the verifier caught it.

- **Restricted.** The action space *is* the admissible set. Deleting a word
  from it would train the policy against a different action set than the
  decoder presents at deployment, and could leave a suboptimal action sitting
  at rank 1. So a training state whose admissible set contains any held-out
  answer is **dropped entirely** (~28% of otherwise-eligible restricted
  states). The action space stays faithful, or the task does not exist.
- **Unrestricted.** The menu is already a top-k slice of 12,972, so dropping a
  word shortens it without changing anything structural.

### A limitation, stated rather than buried

At unrestricted states GRPO samples from a **top-k menu**, not from the whole
vocabulary. That does not match deployment, where the model may emit any legal
word. The consequence is that this run cannot teach the model to *avoid* bad
turn-2 probes — only to rank good ones. It is a deliberate trade for a tractable
reward table, and any claim about turn-2 behaviour has to be read with it in
mind.

### Held fixed

Decoder adaptive@20, chunk 512, pruning on. The same 246 paired eval answers.
The same `render_prompt`. Training states come only from the 2,069 training
answers. Train and validation states are disjoint — a state reachable from a
training answer is often reachable from a held-out one too, so the builder
carries claimed states forward and validation may only take what training did
not.

## 4. The pre-registered reading

Written before the run.

**A — It works.** Mean improves beyond ~3.70 on the paired 246.
→ Note that this *exceeds* the restricted-regime ceiling of 0.0698, so the gain
must have come from the unrestricted regime. Confirm that directly: re-run
`decision_budget.py` on the new policy and check the unrestricted cost fell.
If the mean improved but the unrestricted cost did not, the gain is noise or a
decoder interaction, not the intervention.

**B — It moves within the ceiling.** Mean lands 3.69–3.76.
→ Consistent with fixing restricted-regime mistakes and nothing more. A real
but small result, and the honest write-up is that RL recovered most of what was
locally available and the rest of the gap is not in action selection.

**C — Flat.** No significant paired change.
→ The model's action selection was already near the ceiling the budget
predicted, and the remaining 0.32 is not action selection at all. Combined with
Phase 8, that is a clean two-method negative and the project's answer is
"distillation plus a decoder gets to 3.76, and closing the last 0.32 needs a
better model, not better action selection".

**D — It regresses.** Same shape as Phase 8.
→ Check KL first: if it grew, this is drift and `beta` was too low. If KL
stayed small and the mean still fell, the reward proxy in the unrestricted
regime is actively misleading and that is worth reporting on its own.

**Not a valid reading in any branch:** an unpaired comparison, or reading the
one-ply-scored tasks as if they were exact. `meta.exact` distinguishes them and
results must be broken down by it.

## 5. How to run it

Task set and dataset are already built and published (2026-08-25). To run:

1. Open `phase11_grpo/wordle_phase11_grpo_kaggle.ipynb` on Kaggle.
2. Attach **both** `wordle-sft-package-v2` and `wordle-adapters-v2`.
   Accelerator **GPU T4**.
3. Run All with `SMOKE = True` (~2 min). Check the memory report and that the
   fresh-adapter identity check prints `OK`.
4. Set `SMOKE = False`, Run All. ~60-90 min.

To rebuild the inputs from scratch:

```bash
.conda/python.exe phase11_grpo/build_grpo_tasks.py        # ~5 min, local
.conda/python.exe phase11_grpo/verify_grpo_tasks.py       # gate, exits non-zero on failure
.conda/python.exe phase11_grpo/make_grpo_notebook.py
.conda/python.exe tools/prepare_kaggle_dataset.py --dest uploads/kaggle_upload
.conda/python.exe tools/kaggle_run.py data-push -m "grpo tasks"
```

**Gate before believing anything:** the notebook re-runs the `sft` + `baseline`
control and must reproduce **3.7642**. It is the same control that made the
Phase 10 crossover interpretable, and the same one whose absence voided
Phase 8 v3.

**Stop rule, pre-registered:** early-stop on the **paired 246-game rollout**,
not on loss or reward. Phase 8's preference accuracy saturated at step 75 of 466
and every step after was drift that only the rollout could see. Checkpoint
every N steps and evaluate.

## 6. Cost

| stage | cost |
|---|---|
| task build | ~3 min local, no GPU |
| GRPO training | ~60-90 min T4 |
| paired eval | ~15 min (2 cells through the Phase 9 harness) |

## 7. Files

| file | what |
|---|---|
| `decision_budget.py` | prices every model decision; produces the ceiling in §2 |
| `build_grpo_tasks.py` | the precomputed reward table |
| `verify_grpo_tasks.py` | independent audit of the task set |
| `make_grpo_notebook.py` | generates the training notebook |
| `RUN.md` | this file |

---

## 8. Failures already hit, and what fixed them

Recorded so the next person does not repeat them.

| symptom | cause | fix |
|---|---|---|
| `ImportError: incompatible torchao 0.10 / needs >0.16` at the adapter merge | Kaggle's torchao is stale and PEFT raises it *lazily*, inside `from_pretrained` | `fix_torchao_peft_conflict()`, already used by Phases 8 and 9 |
| `CUDA OOM: tried to allocate 3.12 GiB` ~15 min in | scoring did `log_softmax(logits.float())`, materialising `(B, T, 151936)` in fp32 — 3.62 GiB for a 32x200 chunk | `logits_to_keep` (only the last few positions are ever needed) plus gather+logsumexp; same arithmetic, ~19 MiB |
| dataset not found, `/kaggle/input/datasets` only | mount is three levels deep; fixed-depth globs miss it | recursive walk for marker files |

The scoring rewrite is covered by a numpy test that checks it against a slow
full-`log_softmax` reference, plus padding-invariance, chunking-invariance and
the position off-by-one.

---

## 9. Result (added 2026-08-26, after the run)

### Training

3,150 tasks, 393 optimizer steps, ~50 min on a T4. Peak 4.77 GiB.

```
optimal-action rate (held-out states)  67.6% -> 68.9%
  exact-tree tasks only                77.0% -> 78.2%
zero_adv                               0.00 throughout
KL drift                               0.00005 -> 0.03257
pg                                     -0.086 -> -0.050, no trend
```

`zero_adv = 0.00` on every logged step vindicates `SCALE_REWARDS = False`:
under std-normalisation the tied-optimum states would have produced 0/0 and
contributed nothing. The proxy moved the right way. `pg` never improved.

**Reproducibility.** The run was executed twice — once interactively, once
headless through the API on a different session — and the two adapters are
**byte-identical** (`sha256 6d4c469a1c1b3790725c`), with a step-for-step
identical log. The pipeline is deterministic end to end.

### Gameplay — the verdict

246 held-out answers, decoder adaptive@20, paired. The `sft` + `baseline`
control reproduced **3.7642** exactly, and the log confirms both adapter chains
loaded (`tree_salet_endgame + tree_salet_endgame_grpo`).

| arm | mean | solved | fail |
|---|---:|---:|---:|
| `sft` | **3.7642** | 242/246 | 1.63% |
| `sft_grpo` | 3.7602 | 241/246 | 2.03% |

```
diff -0.0041   paired t = -0.28   NOT significant
better 7   worse 6   unchanged 233
```

**233 of 246 games were identical.** The -0.0041 is 6% of the 0.0698 ceiling
and well inside noise. Hard-mode violations 9.11% -> 9.24%, forced 11.39% ->
11.30%: nothing moved.

**This is outcome C**, as pre-registered: *the model's action selection was
already near the ceiling the budget predicted, and the remaining 0.32 is not
action selection at all.*

### Why this is a clean negative rather than a failed run

Everything upstream worked. Rewards were exact, the gate reproduced, the
adapter chain was right, every state carried gradient signal, and the proxy
improved. And unlike Phase 8 it did **no harm** — no drift penalty, failures
+1 game. The lower learning rate, the KL leash and the tie-aware objective all
held. The Phase 8 failure mode was avoided; there was simply nothing to win.

### Two methods, opposite failure modes, one conclusion

| | what happened | cost/benefit |
|---|---|---|
| Phase 8 DPO | drifted away from the SFT policy | **-0.1708** |
| Phase 11 GRPO | stayed put | **+0.0041**, n.s. |

The decision budget predicted this before either ran: perfect restricted-regime
play is worth 0.0698 guesses, and the model was already optimal in 85.6% of
those decisions with 83% of the remainder being ties. The gap is a capability
limit — the retrieval failure measured in Phases 4-6 — not a ranking problem.

### An adapter-loading trap, caught before it produced a number

The GRPO LoRA was trained on the *merged* SFT model, so only the GRPO delta was
saved. The Phase 9 harness loaded exactly one adapter onto stock Qwen, which
would have silently dropped every SFT weight and reported a catastrophic
regression — the same class of error that voided Phase 8 v3. `ARM_ADAPTERS` now
accepts a **chain**, merging between steps, and prints what it assembled. The
Phase 8 DPO adapter has the identical structure and was mapped the same way.

### Limitations, stated

- The unrestricted regime was trained against a **one-ply proxy** over a top-48
  menu, not the full vocabulary. So "GRPO cannot help at turn 2" is *not*
  established — only "this run did not". Teaching the model to avoid bad probes
  was outside what this reward table can express.
### The pre-registered stop rule, settled

RUN.md committed to early-stopping on the paired rollout rather than on loss.
The interactive session lost its checkpoints, but the API re-run wrote them to
output, so all three were scored on the same 246 answers:

| arm | mean | solved | fail% | vs `sft` | t | games changed |
|---|---:|---:|---:|---:|---:|---:|
| `sft` (control) | **3.7642** | 242 | 1.63 | — | — | — |
| `sft_grpo_150` | 3.7602 | 242 | 1.63 | −0.0041 | −0.38 | 7 |
| `sft_grpo_300` | **3.7520** | 241 | 2.03 | −0.0122 | −0.90 | 11 |
| `sft_grpo` (393) | 3.7602 | 241 | 2.03 | −0.0041 | −0.28 | 13 |

**No checkpoint is significantly different from the SFT baseline.** Step 300
has the lowest mean, but at t = −0.90 (p ~ 0.37) across three comparisons,
picking it would be selecting on noise — and the best of three nominally-best
results is exactly the multiple-comparison trap this project has avoided
elsewhere. It is reported, not adopted.

The one real pattern is `games changed`: 7 → 11 → 13 as training proceeds,
while the mean stays flat. The policy is moving and the movement is not
purchasing anything — a smaller, harmless version of the Phase 8 drift, and
the reason the KL leash was kept.

So the stop rule returns the same answer whichever step it picks: **outcome C
holds for the whole trajectory**, not just its endpoint.

### Artifacts

- adapter: `arnavyrr/wordle-adapters-v2` -> `tree_salet_endgame_grpo/`
- results: `results/phase11/`
- kernels: `wordle-phase-11-grpo` (train), `wordle-phase-9-prompt-harness-sweep` (score)
