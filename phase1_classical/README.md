# A Classical (Non-LLM) Wordle Solver

A symbolic + information-theoretic Wordle solver, built as a strong algorithmic baseline for a
later 0.5B LLM + SFT/GRPO experiment. **No machine learning, no reinforcement learning, no
language model, anywhere.** Dependencies: Python ≥ 3.9 and NumPy.

The research question this exists to answer:

> How much of Wordle performance is achievable with explicit constraint solving and information
> theory, with no learned model at all?

## The three layers, measured separately

| Layer | Mechanism | Knowledge used | Solvers |
|---|---|---|---|
| 1 · Symbolic constraint solving | Exact elimination of logically impossible answers | Rules of Wordle only | shared by all; `random` isolates it |
| 2 · Information-theoretic decision making | Choose the probe that best splits the hypothesis space | None — partition structure only | `entropy`, `expected`, `minimax` |
| 3 · Probability / frequency heuristics | Prefer letters typical of answers | Letter statistics of the answer list | `frequency`, hybrid's prior term |

Reading the benchmark across these groups is the point: `random` shows what perfect elimination
alone buys, the Layer-2 solvers show what information theory adds, `frequency` shows whether
cheap letter statistics can substitute for it.

## Layout

```
wordle_solver.py              standalone solver module — the deliverable
benchmark.py                  evaluation harness
build_artifacts.py            one-shot precomputation (~50 s)
run_full_benchmark.py         full 2,315-answer evaluation
make_notebook.py              generates the notebook from the modules
validate_notebook.py          executes every notebook cell on a subset
wordle_classical_solver.ipynb Kaggle-compatible notebook (68 cells)
tests/test_wordle_solver.py   61 tests
data/                         source word lists — never modified
artifacts/                    generated bundle
environment.yml               conda environment
```

## Data

Discovered by audit, not assumed. The workspace was empty, so the canonical lists were fetched
(provenance and SHA-256 in `data/PROVENANCE.json`):

| File | Words | Content |
|---|---|---|
| `data/wordle_answers.txt` | 2,315 | possible answers |
| `data/wordle_allowed_guesses.txt` | 10,657 | additional legal guesses (disjoint from answers) |
| **legal guess pool** | **12,972** | union |

All exactly 5 lowercase ASCII letters; no duplicates, accents, punctuation, or blank lines.
Exact match to the standard original Wordle vocabulary.

## Results — full benchmark, all 2,315 answers

Every solver evaluated against **every** answer, `guess_pool="full"` (all 12,972 legal guesses
scored each turn), `max_guesses=6`, `seed=20260817`. Not a sample.

| Solver | Opener | Mean | Median | Std | Min | Max | %3 | %4 | Fail | ms/game |
|---|---|---|---|---|---|---|---|---|---|---|
| `random` | — | 4.0259 | 4 | 0.967 | 2 | 6 | 25.40 | 39.83 | 1.56% | 0.5 |
| `frequency` | STARE | 3.7459 | 4 | 0.918 | 1 | 6 | 35.98 | 38.36 | 1.38% | 0.1 |
| **`entropy`** | **SOARE** | **3.4644** | **3** | 0.586 | 2 | 6 | **52.57** | 42.76 | **0.00%** | 129 |
| `expected` | ROATE | 3.4812 | 3 | 0.575 | 2 | **5** | 48.81 | 47.13 | 0.00% | 158 |
| `minimax` | ARISE | 3.5732 | 4 | 0.625 | 1 | 6 | 42.76 | 50.19 | 0.00% | 172 |
| `hybrid` | RAISE | 3.4812 | 3 | 0.598 | 1 | **5** | 49.63 | 45.05 | 0.00% | 170 |

Failure-penalized means (failure = 7): `random` 4.0721, `frequency` 3.7909; the four
zero-failure solvers are unchanged.

**Best overall: `entropy` at 3.4644 mean, 0% failures.** `expected` and `hybrid` tie at 3.4812
with a better *tail* — max 5 guesses rather than 6.

### Contribution by layer

| Mechanism added | Mean | Δ |
|---|---|---|
| Symbolic elimination alone (`random`) | 4.0721 | — |
| + frequency heuristics (Layer 3) | 3.7909 | −0.281 |
| + information theory (Layer 2, best) | 3.4644 | −0.608 |

Exact constraint filtering alone already solves 98.4% of games within six guesses. Information
theory buys a further ~0.61 guesses **and eliminates failures entirely**.

### Mean surviving candidates after each guess

Starting uncertainty is 2,315 candidates (11.18 bits).

| Solver | after g1 | g2 | g3 | g4 |
|---|---|---|---|---|
| `random` | 214.33 | 19.50 | 3.01 | 1.44 |
| `frequency` | 71.29 | 5.70 | 1.87 | 1.31 |
| `entropy` | 62.30 | 3.35 | 1.07 | 1.00 |
| `expected` | 60.42 | 3.15 | 1.03 | 1.00 |
| `hybrid` | 61.00 | 3.37 | 1.06 | 1.00 |

### Best openers (all 12,972 scored against all 2,315 answers)

| Metric | Best | Value |
|---|---|---|
| Entropy | SOARE | 5.8860 bits |
| Expected remaining | ROATE | 60.42 candidates |
| Worst-case bucket | AESIR / ARISE / RAISE | 168 candidates |
| Best that is also a possible answer | RAISE | 5.8779 bits |

Top openers are mostly **not** possible answers — a turn-1 probe's job is to split the space,
not to win. This is precisely why the guess pool must not be restricted to the answer list.

These figures reproduce published values for this vocabulary (SOARE at 5.886 bits, ROATE at
60.42, minimax worst-case 168), which is useful external validation of the feedback function
and entropy computation.

## Quick start

```bash
conda env create -f environment.yml
conda activate wordle
python build_artifacts.py
python wordle_solver.py --answer crane
```

```bash
python wordle_solver.py --interactive
```

```bash
pytest tests/ -q
python run_full_benchmark.py
```

## Using it from Python

```python
from wordle_solver import load_artifacts, play_game, solve

bundle = load_artifacts("artifacts")      # instant — matrix is memory-mapped
solve(answer="crane", verbose=True)

r = play_game(bundle.solver("entropy"), "mummy")
print(r.n_guesses, r.guesses, r.candidates_remaining)
```

## Feedback encoding

`G`=green, `Y`=yellow, `B`=grey, encoded base-3 little-endian by position:

```
code = sum(tile[i] * 3**i)      tile in {0=B, 1=Y, 2=G}      code in [0, 243)
```

243 < 256, so the whole 12,972 × 2,315 feedback table fits in `uint8` — **28.6 MiB**, which is
what makes exhaustive precomputation cheap. `GGGGG` is code 242.

### Duplicate letters

The answer supplies a **budget** per letter. Greens consume it first; then yellows are assigned
left to right from whatever remains; surplus letters come back grey. So `added` vs `dread` is
`YYBYG` — the green at index 4 consumes budget before the yellow at index 1 — and `speed` vs
`abide` is `BBYBY`, with the second `e` greyed out. 32.4% of answers contain a repeated letter,
so this is not an edge case.

## Configuration

Everything behavioural lives in `SolverConfig` / `artifacts/solver_config.json`:

```python
SolverConfig(
    seed=20260817, max_guesses=6, strategy="hybrid",
    guess_pool="full",          # "full" | "adaptive" | "candidates"
    entropy_weight=1.0,         # all four weights are on a BITS scale
    minimax_weight=0.25,
    expected_weight=0.25,
    answer_bonus_weight=1.0,
    opening_guess=None,         # set to skip recomputing turn 1
)
```

`guess_pool="full"` scores all 12,972 legal guesses every turn — strongest and slowest.
`"adaptive"` is much faster for a very small cost in mean guesses.

## Hybrid scoring

The four component quantities have incompatible units, so they are all converted to **bits**
before being combined (rather than z-scored, whose scale would drift turn to turn):

```
score(g) =  α·H(g)                        bits of expected information gained
          − λ·log2(max_bucket(g))         bits left, worst case
          − ν·log2(E[remaining](g))       bits left, in expectation
          + μ·log2(1 + p_answer(g))       credit for possibly winning this turn
```

Because every term is in bits, `λ = 0.25` genuinely means "weight worst-case at a quarter of
raw information gain". Defaults were chosen to be interpretable, then checked with a weight
sweep — not tuned to the benchmark.

## Scope boundary

All solvers are **one-step greedy**: they optimise immediate information, not the true
minimum-expected-guesses game tree. A full lookahead search does better. This is "strong
classical", not optimal — stated so the LLM comparison is not read against a false ceiling.

Note also that these solvers receive the answer list as *input*, so they never need to know
which strings are English words. That is a real advantage over an LLM and should be
acknowledged rather than treated as a fair fight.

## Artifacts

`build_artifacts.py` writes a bundle that reproduces the solver anywhere with Python + NumPy:

```
answers.txt  valid_guesses.txt  feedback_matrix.npy
metadata.json  frequency_model.json  solver_config.json
```

Plus, after benchmarking: `benchmark_results.json`, `first_guess_analysis.csv`.

Paths are relative throughout. No `/kaggle/...` path appears in `wordle_solver.py`; Kaggle
detection lives only in the notebook's setup layer. CPU only — a GPU gives no benefit at this
problem size.
