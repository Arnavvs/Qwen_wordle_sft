# The maths, with worked examples

Every formula used anywhere in this project, with a real data point from the
actual files beside it. Nothing here is illustrative-only — the numbers were
printed from `artifacts/` and `sft_package/`.

Contents:

1. [Feedback encoding](#1-feedback-encoding)
2. [Constraint solving](#2-constraint-solving)
3. [Information theory: which probe is best](#3-information-theory-which-probe-is-best)
4. [Tree search](#4-tree-search)
5. [SFT: completion-only cross-entropy](#5-sft-completion-only-cross-entropy)
6. [Constrained decoding](#6-constrained-decoding)
7. [The hard-mode filter](#7-the-hard-mode-filter)
8. [DPO](#8-dpo)
9. [GRPO](#9-grpo)
10. [How each dataset was generated](#10-how-each-dataset-was-generated)

---

## 1. Feedback encoding

Each tile is one of three states. Encode them as digits and pack base-3,
little-endian by position:

$$
\text{code} = \sum_{i=0}^{4} t_i \cdot 3^i,
\qquad t_i \in \{0=\text{grey},\ 1=\text{yellow},\ 2=\text{green}\}
$$

so `code` $\in [0, 243)$. Since $243 < 256$ the entire
$12{,}972 \times 2{,}315$ feedback table fits in `uint8` — **28.6 MiB** — which
is what makes exhaustive precomputation affordable.

**Real values:**

| guess | answer | pattern | code | digits (little-endian) |
|---|---|---|---:|---|
| ADDED | DREAD | `YYBYG` | 193 | `[1,1,0,1,2]` |
| SPEED | ABIDE | `BBYBY` | 90 | `[0,0,1,0,1]` |
| SALET | ABHOR | `BYBBB` | 3 | `[0,1,0,0,0]` |
| CRANE | CRANE | `GGGGG` | 242 | `[2,2,2,2,2]` |

Check the third: only position 1 is yellow, so
$1 \cdot 3^1 = 3$. And all-green is $\sum_i 2\cdot 3^i = 2 \cdot \frac{3^5-1}{2} = 242$.

### Duplicate letters — the part that is easy to get wrong

The **answer** supplies a budget of each letter. Greens consume it first, then
yellows are assigned left to right from what remains, and surplus copies come
back grey.

`ADDED` vs `DREAD` → `YYBYG`. Walk it: the answer DREAD has D×2, R, E, A.
Position 4 (`D`) is green and consumes one D. Position 0 (`A`) is yellow, 1 (`D`)
takes the last remaining D, position 2 (`D`) has no budget left → **grey**,
position 3 (`E`) is yellow.

`SPEED` vs `ABIDE` → `BBYBY`. ABIDE has one E. The first E (position 2) claims
it; the second E (position 3) gets grey.

**32.4%** of the 2,315 answers contain a repeated letter, so this is not an edge
case — it is a third of the game.

---

## 2. Constraint solving

After a guess $g$ producing code $c$, the surviving candidate set is

$$
C' = \{\, a \in C \;:\; \mathrm{fb}(g, a) = c \,\}
$$

With the precomputed matrix this is one row lookup and a boolean mask — no
string work at all. That is `FeedbackMatrix.filter_indices`.

**Real:** starting from all 2,315 answers, `SALET → BYBBB` leaves
**102 candidates**.

This layer alone — perfect elimination, random choice among survivors — already
solves 98.4% of games within six guesses (`random` baseline, mean 4.0259).

---

## 3. Information theory: which probe is best

Partition the candidate set $C$ by the feedback a guess $g$ would produce.
Bucket $k$ has size $n_k$, and $n = |C|$. Under a uniform posterior over
surviving answers — correct here, because the daily answer is drawn from the
answer list, not weighted by English frequency — the probability of landing in
bucket $k$ is $p_k = n_k / n$.

**Entropy** (expected bits gained):

$$
H(g) = -\sum_k p_k \log_2 p_k
$$

**Expected remaining** (expected size of the surviving set):

$$
\mathbb{E}[\text{rem}](g) = \sum_k n_k \cdot p_k = \frac{1}{n}\sum_k n_k^2
$$

**Worst case** (minimax): $\max_k n_k$.

**Real, at the 102-candidate state above:**

| probe | $H$ (bits) | $\mathbb{E}[\text{rem}]$ | worst |
|---|---:|---:|---:|
| BROND | **5.155** | **3.84** | **8** |
| CRANE | 4.497 | 5.49 | 9 |
| ABHOR | 4.169 | 8.04 | 21 |
| FUZZY | 1.602 | 55.10 | 74 |

FUZZY is the intuition pump: it is a legal word, but it barely splits this
particular set, leaving 55 candidates on average against BROND's 3.84.

Note **BROND is not a possible answer**. A turn-2 probe's job is to split the
space, not to win — which is exactly why the guess pool must not be restricted
to the answer list.

### The opener, computed over everything

Scoring all 12,972 guesses against all 2,315 answers:

| metric | best | value |
|---|---|---|
| entropy | SOARE | 5.8860 bits |
| expected remaining | ROATE | 60.42 |
| worst-case bucket | AESIR / ARISE / RAISE | 168 |

These reproduce published figures for this vocabulary, which is external
validation that the feedback function and the entropy computation are correct.

---

## 4. Tree search

The information-theoretic solvers are **one-step greedy**: they maximise
immediate information, not the true minimum expected guesses. Full lookahead
minimises

$$
V(C) = \min_{g} \left[ 1 + \sum_k p_k \, V(C_k) \right],
\qquad V(\{a\}) = 0
$$

evaluated to depth 6 with a top-$k$ shortlist (100 candidate guesses, 60 in the
endgame) to keep it tractable.

**Measured:** greedy `entropy` 3.4644 vs depth-6 tree 3.4300 — so greed costs
about **0.03 guesses**. `tree_salet` reaches 3.4212, matching the known optimum
for this word list.

Cost is the reason this is not used everywhere: `choose()` runs in ~0.1 ms at 1
candidate, ~18 ms at 8, ~1.4 s at 40, and a timing sweep up to 400 candidates
**timed out after 10 minutes**. That single fact shaped the Phase 8 data design.

---

## 5. SFT: completion-only cross-entropy

A record is a `(prompt, completion)` pair. The loss is standard next-token
cross-entropy, but the prompt tokens are masked to `-100` so only the guess is
supervised:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{i \in \text{completion}} \log P_\theta(t_i \mid t_{<i})
$$

Without the mask, a ~5-token target is drowned out by ~450 tokens of state
description and the model learns to reproduce the prompt.

**Real record:**

```
completion  : 'SALET'
prompt toks : 122        completion toks : 3
completion  : [16589, 20756, 151645]   ->  ' SALET<|im_end|>'
labels      : [-100] x 122  then [16589, 20756, 151645]
```

Three supervised tokens out of 125. Note the leading space (`" " + WORD`) and
the EOS — both matter, and section 6 depends on the exact same tokenization.

---

## 6. Constrained decoding

The model never emits free text. Every legal word is **scored**, and the
argmax is played. For prompt $p$ and word $w$ tokenized exactly as in training:

$$
s(w) = \log P_\theta(w \mid p) = \sum_{i} \log P_\theta\!\left(t_i \mid p, t_{<i}\right)
$$

summed over the word's tokens **and the EOS token**.

### Why EOS must be included

Without it, a word whose token sequence is a *prefix* of another's is scored
under strictly fewer constraints and is systematically over-ranked. Including
EOS makes each $s(w)$ the log-probability of a **complete string**, so scores
are comparable across different token lengths. Real tokenizations vary from 1 to
4 tokens per word:

```
CARGO -> [356, 7581, 46, 151645]      (3 tokens + EOS)
GARBO -> [96591, 4677, 151645]        (2 tokens + EOS)
```

No length normalisation is applied: $s(w)$ *is* the probability the model
assigns to playing $w$, which is the quantity to argmax.

### Exact branch-and-bound

Naively this is 12,972 forward passes per turn. Instead the prompt is run once
with a KV cache, which yields the first-token distribution $\ell_0$ for free.
Since every per-token log-probability is $\le 0$:

$$
s(w) \;\le\; \ell_0\!\left[\mathrm{tok}_0(w)\right]
$$

That is a **valid upper bound**. Visit words in descending order of the bound
and stop once the best fully-scored word beats the bound of everything
unvisited. The result is provably identical to scoring all 12,972 —
`verify_against_full()` asserts it every run.

**Measured:** trained models need ~1.4 chunks of 512 per decision; the
*untrained* base model needs all 26, because its first-token distribution is
flat and the bound never bites. That asymmetry made the base-model control ~20×
slower than the adapter runs.

---

## 7. The hard-mode filter

A word $w$ is **admissible** iff it would have produced exactly the feedback
already observed, for every guess so far:

$$
A = \{\, w \in L \;:\; \mathrm{fb}(g_j, w) = c_j \ \ \forall j \,\}
$$

where $L$ is the 12,972-word legal pool.

**This is not the candidate set**, and the distinction is the whole
justification for using it:

| set | pool | uses the answer list? |
|---|---|---|
| candidate set | 2,315 answers | **yes** — privileged |
| admissible set $A$ | 12,972 legal guesses | no — a function of the prompt |

$A$ is computable from the visible board plus the public word list. It is what
hard-mode Wordle enforces and what a human sees.

**Measured sizes** (following the expert's own games):

| turn | median $\lvert A\rvert$ | max |
|---:|---:|---:|
| 1 | 12,972 | 12,972 |
| 2 | 211 | 769 |
| 3 | 7 | 102 |
| 4 | 2 | 20 |

Two consequences that drove the design:

- Any threshold $\ge 769$ filters at every turn from turn 2 on, so thresholds
  1000 and $10^9$ are **behaviourally identical**. They produced byte-identical
  results, which is correct rather than suspicious.
- At states with one candidate, $|A| = 1$ **38.9%** of the time — the filter
  alone determines the word and the model contributes nothing. Those decisions
  are tagged `forced` and reported separately, or a decoder win would look like
  a model win.

### Why filtering everywhere is wrong

The expert's own action is feedback-**inconsistent** most of the time early on:

```
turn 2:  40.6% consistent   <- it PROBES with a word that cannot win
turn 3:  91.8%
turn 4: 100.0%
```

An always-on filter forbids the expert's turn-2 policy in ~59% of games. That is
the known reason hard mode scores worse than free mode. Hence the **adaptive**
rule: filter only once $|A| \le \tau$. Measured optimum $\tau \approx 10$–$20$
(a plateau, not a peak).

---

## 8. DPO

### What problem it solves

Under the adaptive decoder the model's job is to **rank admissible words**. SFT
on a single expert action teaches one point of that ranking, not the ranking.
Phase 6 tested "more/better SFT data" directly and moved games by exactly zero.

DPO instead learns from pairs $(p, w^+, w^-)$: at state $p$, $w^+$ is preferred
to $w^-$.

### The loss

Let $\pi_\theta$ be the policy and $\pi_{\text{ref}}$ the frozen SFT model.
Define the sequence log-probability $s_\theta(w \mid p)$ exactly as in section 6.
Then:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(
\beta \big[
\underbrace{(s_\theta(w^+) - s_{\text{ref}}(w^+))}_{\text{how much }\theta\text{ raised }w^+}
-
\underbrace{(s_\theta(w^-) - s_{\text{ref}}(w^-))}_{\text{how much }\theta\text{ raised }w^-}
\big]\Big)
$$

The bracketed quantity is the **margin**. Training pushes it positive: raise the
good word *relative to the reference*, lower the bad one.

### Why the reference model is there

The term $s_\theta - s_{\text{ref}}$ is an **implicit reward**. DPO is the
closed-form solution to a KL-regularised bandit problem:

$$
\max_\theta \ \mathbb{E}\big[r(w)\big] - \tfrac{1}{\beta}\,
\mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right)
$$

whose optimum is $\pi_\theta(w) \propto \pi_{\text{ref}}(w)\,e^{\beta r(w)}$.
Rearranged, $r(w) = \tfrac{1}{\beta}\log\frac{\pi_\theta(w)}{\pi_{\text{ref}}(w)}$
— which is the margin term. So the reference is not a technicality: it is what
stops the model drifting arbitrarily far from a policy that already works.

$\beta$ controls that leash. $\beta \to 0$ means "barely move"; large $\beta$
means "chase the preferences and forget the SFT policy". We use $\beta = 0.1$,
the standard value, and do not sweep it on a first run.

### What the diagnostics mean

- **margin** — should rise from 0 and stay positive
- **acc** — fraction of pairs where the margin is positive; should exceed 0.5

Both say the *preference* was learned. **Neither says gameplay improved** — that
is a separate claim, settled only by playing the 246 games.

### Real pairs

The ranking objective used to build pairs is

$$
\mathrm{cost}(w) = \mathbb{E}[\text{rem}](w) \;-\; 0.5 \cdot \mathbb{1}[w \in C]
$$

The bonus is a tie-break: at small candidate counts every guess has the same
$\mathbb{E}[\text{rem}]$, so without it the objective cannot separate "a word
that might win now" from "a word that cannot". A pair is emitted only when
$\mathrm{cost}(w^-) - \mathrm{cost}(w^+) \ge$ margin — otherwise there is
nothing to teach and asserting a preference would be teaching noise.

**Competitive pair** (hard, fine discrimination):

```
history   SALET->BGBBB, ARGAL->YYYBB        |A|=7  |C|=1  turn 3
chosen    CARGO   cost 0.50
rejected  GARBO   cost 1.00      gap 0.50
teacher   tree_salet     action space: admissible
```

Anagrams. Both are consistent with every clue; CARGO is a candidate and GARBO
is not, so CARGO can win this turn and GARBO cannot.

**Clear pair** (coarse):

```
history   SALET->BBGBB                       |A|=100  |C|=27  turn 2
chosen    RUMBO   cost  4.48
rejected  AVERT   cost 25.07     gap 20.59
teacher   expected_remaining    action space: legal
```

AVERT reuses letters already known absent, so it splits the space badly.

### The action-space rule, and why it caused a real bug

At deployment the decoder restricts the model to $A$ whenever $|A| \le 20$. So
in that regime **both** words of a pair must be admissible — otherwise the pair
teaches a preference about a word the model can never play, putting probability
mass on an action the filter always vetoes.

Both the tree expert and the entropy solver pick from the *full* pool, so both
can return inadmissible words. In generation this produced:

| stage | violations |
|---|---:|
| first build | ~1,240 (19% of restricted rows) |
| after fixing `chosen` | 77 (all in `rejected`) |
| after fixing `rejected` | **0** |

The audit that catches this recomputes both costs from the state itself and
asserts the ordering — because a preference pair can be silently *backwards*,
and no loss curve would ever reveal it.

---

## 9. GRPO

Not yet run. Included because it is the planned next step and the contrast with
DPO is the point.

DPO learns from **static pairs** labelled by an external judge. GRPO learns from
**its own sampled behaviour**, scored by a reward function.

For a state $p$, sample a group of $G$ actions $w_1 \ldots w_G$ from the current
policy, score each with reward $r_i$, and normalise **within the group**:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1..r_G)}{\mathrm{std}(r_1..r_G)}
$$

Then take a clipped policy-gradient step, KL-anchored to the reference exactly
as in DPO:

$$
\mathcal{L}_{\text{GRPO}} = -\,\mathbb{E}\!\left[
\min\!\big(\rho_i \hat{A}_i,\ \mathrm{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)\,\hat{A}_i\big)
\right] + \lambda\,\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}),
\qquad
\rho_i = \frac{\pi_\theta(w_i)}{\pi_{\theta_{\text{old}}}(w_i)}
$$

The group-relative baseline is what removes the need for a separate value
network — the group mean *is* the baseline.

**Why it might suit this problem.** Under the adaptive decoder the action space
is 2–20 admissible words, so $G$ can cover a large fraction of it, and the
reward is exactly computable rather than learned:

$$
r(w) = -\,\mathbb{E}[\text{rem}](w) \quad\text{or}\quad r = -(\text{guesses used})
$$

Most RLHF pain comes from a noisy learned reward model. Here there isn't one.

**Why it is deliberately last.** Every phase so far has moved *less* than
expected, and GRPO is the most expensive and least stable option. Phase 8 tests
whether preference learning moves the policy at all on this data; if DPO on
15k clean pairs does nothing, GRPO on the same signal is very unlikely to.

---

## 10. How each dataset was generated

### Phase 2 — natural expert trajectories

Roll each expert over all 2,315 answers, recording every `(state, action)`.
7,067–7,173 rows per policy. This is the **on-policy** distribution: only states
the expert itself reaches.

The defect, measured on `tree_salet`:

```
n_candidates == 1 :  1,612 rows
distinct words    :  1,612
paths per word    :  exactly 1.00, for every word
```

The winning move is demonstrated 1,612 times — as a 1,612-way lookup with one
example per class. Nothing in that teaches "read the constraints, produce the
word that satisfies them".

*(An earlier writeup cited "6 fully-determined examples". That conflated
`fully_determined` — all five greens known, genuinely 4 — with endgame coverage.
The corrected framing is the one above.)*

### Phase 6 — endgame states, many paths per word

DAgger-style noise injection. Roll out **deliberately imperfect** games — SALET
opener so states are reachable at evaluation, then randomised continuations
mixing surviving candidates and arbitrary legal words — and label the states
reached with the expert.

Descent uses only cheap policies; the expert is called **once** per accepted
state. That matters given the tree-search costs in section 4.

| | rows | k=1 rows | words | paths/word | turn-1 share |
|---|---:|---:|---:|---:|---:|
| natural | 7,067 | 1,612 | 1,612 | 1.00 | 29.3% |
| endgame synthetic | 12,145 | 6,148 | 2,068 | 2.97 | 0% |
| **mixed** | **19,212** | **7,760** | **2,068** | **3.75** | **10.8%** |

Path diversity is real, not cosmetic: mean Jaccard overlap of the guess-sets
between two paths to the same word is **0.213** — essentially just the shared
SALET opener.

```
ABHOR, three distinct routes:
  1  SALET->BYBBB, BROAD->YYYYB, COBRA->BYYYY
  2  SALET->BYBBB, ADORN->GBYYB, ARBOR->GBYGG
  3  SALET->BYBBB, PRAWN->BYYBB, VICAR->BBBYG, ANGAS->GBBBB
```

### Phase 8 — preference pairs

Same reachable-state sampler, bucketed by $|A|$ rather than by candidate count,
because $|A|$ is the decision regime the decoder actually presents. k=1 is
excluded: the filter forces 38.9% of those and collapses the rest to ~2 words.

Final: **14,923 pairs**, mid-game weighted.

```
buckets   2-3:1500  4-10:3499  11-30:3424  31-100:3500  101-300:2000  300+:1000
types     clear 9556 | competitive 5367
teacher   tree_salet 9646 | expected_remaining 5277
space     legal 8479 | admissible 6444
gap       median 3.07     competitive median 0.50
prompts   9893 distinct
```

`tree_salet` labels states with ≤12 candidates; above that it is too slow and
the exact expected-remaining argmin is used instead. The quality cost is
bounded and small: tree_salet scores 3.4212 over 2,315 games, the pure expected
solver 3.4812 — a 0.06 gap, against a model currently 0.32 from the expert.

### What the model never sees, in every phase

The prompt is produced by one shared `render_prompt` call with
`candidates=None, show_candidate_count=False`. It contains history, derived
constraints, and the turn counter. It never contains:

- the answer
- the candidate list or candidate count
- the admissible set or its size
- any cost, entropy, or expected-remaining score

Those exist only in `meta`, for analysis and weighting. Greens render
position-by-position (`p1=A p2=L`) rather than concatenated, because five
concatenated greens would *be* the answer. Every dataset ships with an audit
asserting all of it.
