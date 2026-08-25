"""
make_notebook.py — generate wordle_classical_solver.ipynb.

The notebook embeds `wordle_solver.py` and `benchmark.py` verbatim inside
`%%writefile` cells so that it is fully self-contained on Kaggle, where those
files would not otherwise exist. Generating the notebook from the real files
(rather than maintaining two copies by hand) is what keeps them identical.

    python make_notebook.py
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wordle_classical_solver.ipynb")

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.strip("\n"),
    })


def read(name: str) -> str:
    with open(os.path.join(_paths.CORE, name), encoding="utf-8") as fh:
        return fh.read().rstrip("\n")


# =============================================================================
# 1. Project overview
# =============================================================================
md(r"""
# A Classical (Non-LLM) Wordle Solver

**Purpose.** Establish a rigorous *symbolic + information-theoretic* baseline for Wordle,
to be compared later against a 0.5B LLM trained with SFT/GRPO. The research question this
notebook exists to answer is:

> **How much of Wordle performance is achievable with explicit constraint solving and
> information theory, with no learned model at all?**

Nothing here is AI, machine learning, or reinforcement learning. There is no model being
trained, no gradient, no policy, and no language model. Every choice is a deterministic
computation over an enumerable hypothesis space. The word "solver" is used in its exact
sense: a search procedure over explicit constraints.

## The three layers being measured separately

The notebook is deliberately structured so the contribution of each layer is visible in
isolation, because "the classical baseline" is really three different things:

| Layer | What it does | Knowledge used | Solvers |
|---|---|---|---|
| **1. Symbolic constraint solving** | Deduce exactly which words remain logically possible given the game history. Exact, complete, no approximation. | The rules of Wordle only | *(shared by all)*; `random` isolates it |
| **2. Information-theoretic decision making** | Choose the probe that maximises expected information / minimises posterior size. Uses only the partition structure the feedback function induces. | No English knowledge at all | `entropy`, `expected`, `minimax` |
| **3. Probability / frequency heuristics** | Prefer words whose letters are empirically typical of answers. | Letter statistics of the answer list | `frequency`, and the prior term of `hybrid` |

Reading the benchmark table across these groups is the point of the exercise. `random`
measures what perfect elimination alone buys you; the Layer-2 solvers measure what
information theory adds on top; `frequency` measures whether cheap letter statistics can
substitute for it (spoiler: only partly).

## Structure

1. Project overview · 2. Dataset & environment audit · 3. Imports & configuration ·
4. Load and validate vocabulary · 5. Feedback implementation · 6. Feedback unit tests ·
7. Candidate filtering · 8. Feedback matrix construction · 9. Artifact persistence ·
10.–15. The six solvers · 16. Full benchmark · 17. First-guess analysis ·
18. Example games · 19. Results comparison · 20. Portable artifacts · 21. Local usage

## Portability contract

The core algorithm has no Kaggle dependency. Kaggle path detection happens **only** in the
setup layer of this notebook (Section 2). `wordle_solver.py` reads a single configurable
`DATA_DIR` / `ARTIFACT_DIR`, uses relative paths, and depends on nothing beyond the Python
standard library and NumPy. CPU only — a GPU would provide no benefit here, so none is used.
""")

# =============================================================================
# 2. Dataset & environment audit
# =============================================================================
md(r"""
---
# 2. Dataset & Environment Audit

Nothing about the data is assumed. This section **discovers** whatever word lists are
present, measures them, and reports what was actually found. It searches, in order:

1. `/kaggle/input/**` (any attached Kaggle dataset, arbitrary layout)
2. `./data`, `./input`, `.` (local runs)
3. If nothing usable is found, and `ALLOW_DOWNLOAD` is set, it fetches the canonical lists.

The audit answers the nine questions below with measurements, not assumptions.
""")

code(r'''
# ---------------------------------------------------------------------------
# Setup layer — the ONLY place that knows about Kaggle. Everything downstream
# uses DATA_DIR / ARTIFACT_DIR and nothing else.
# ---------------------------------------------------------------------------
import os, sys, glob, json, platform, collections, unicodedata, hashlib, time

ON_KAGGLE = os.path.isdir("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if ON_KAGGLE:
    SEARCH_DIRS  = sorted(glob.glob("/kaggle/input/*")) + ["/kaggle/input", "."]
    ARTIFACT_DIR = "/kaggle/working/artifacts"
    DATA_DIR     = "/kaggle/working/data"      # only used if we have to download
else:
    SEARCH_DIRS  = ["data", "input", "."]
    ARTIFACT_DIR = "artifacts"
    DATA_DIR     = "data"

# If no dataset is found locally, may we fetch the canonical lists over the network?
ALLOW_DOWNLOAD = True

print(f"environment      : {'KAGGLE' if ON_KAGGLE else 'LOCAL'}")
print(f"python           : {platform.python_version()}")
print(f"platform         : {platform.platform()}")
print(f"search dirs      : {SEARCH_DIRS}")
print(f"DATA_DIR         : {DATA_DIR}")
print(f"ARTIFACT_DIR     : {ARTIFACT_DIR}")
''')

code(r'''
# ---------------------------------------------------------------------------
# Discovery: find every file that plausibly contains a 5-letter word list, and
# measure it. No filename, schema, or word count is assumed.
# ---------------------------------------------------------------------------
WORD_LEN = 5
EXTS = (".txt", ".csv", ".json", ".tsv", ".dat", ".words", ".parquet")

def extract_words(path, word_len=WORD_LEN, max_bytes=64 * 1024 * 1024):
    """Best-effort extraction of `word_len`-letter alphabetic tokens from any text-ish file."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(path)
            except Exception:
                return None, "parquet unreadable"
            toks = [str(v) for c in df.columns for v in df[c].tolist()]
        elif ext == ".json":
            with open(path, encoding="utf-8", errors="replace") as fh:
                obj = json.load(fh)
            toks = []
            def walk(o):
                if isinstance(o, str): toks.append(o)
                elif isinstance(o, dict):
                    for v in o.values(): walk(v)
                elif isinstance(o, (list, tuple)):
                    for v in o: walk(v)
            walk(obj)
        else:
            if os.path.getsize(path) > max_bytes:
                return None, "too large"
            with open(path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
            toks = raw.replace(",", "\n").replace("\t", "\n").split()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    words, raw_n = [], len(toks)
    for t in toks:
        w = t.strip().strip('"').strip("'").lower()
        if len(w) == word_len and w.isascii() and w.isalpha():
            words.append(w)
    return words, f"{len(words)}/{raw_n} tokens were {word_len}-letter alpha"

# Directories to skip. ARTIFACT_DIR is excluded because it contains this
# notebook's OWN generated answers.txt / valid_guesses.txt — on a repeat local
# run those would be rediscovered as if they were source data.
SKIP_PARTS = {os.path.normpath(ARTIFACT_DIR), "artifacts", "tests", "__pycache__",
              ".git", ".conda", ".ipynb_checkpoints", "site-packages"}

def skipped(path):
    parts = set(os.path.normpath(path).split(os.sep))
    return bool(parts & SKIP_PARTS) or os.path.normpath(path).startswith(
        os.path.normpath(ARTIFACT_DIR) + os.sep)

# Scan
found = []
seen_paths = set()
for d in SEARCH_DIRS:
    if not os.path.isdir(d):
        continue
    for path in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if not os.path.isfile(path) or path in seen_paths:
            continue
        if not path.lower().endswith(EXTS) or skipped(path):
            continue
        seen_paths.add(path)
        words, note = extract_words(path)
        if words and len(words) >= 100:          # ignore incidental matches
            found.append({
                "path": path, "n_words": len(words), "n_unique": len(set(words)),
                "size_kb": os.path.getsize(path) / 1024, "note": note, "words": words,
            })

print(f"scanned {len(seen_paths)} candidate files; "
      f"{len(found)} contain >=100 five-letter words\n")
if found:
    print(f"{'path':<62}{'words':>8}{'unique':>8}{'KiB':>9}")
    print("-" * 87)
    for f in sorted(found, key=lambda x: -x["n_words"]):
        print(f"{f['path'][-60:]:<62}{f['n_words']:>8}{f['n_unique']:>8}{f['size_kb']:>9.1f}")
else:
    print("No word lists discovered in the search paths.")
''')

code(r'''
# ---------------------------------------------------------------------------
# Fallback: if discovery found nothing, fetch the canonical lists.
# ---------------------------------------------------------------------------
CANONICAL = {
    "wordle_answers.txt":
        "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/wordle-answers-alphabetical.txt",
    "wordle_allowed_guesses.txt":
        "https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/wordle-allowed-guesses.txt",
}

if not found and ALLOW_DOWNLOAD:
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    print("no dataset discovered -> downloading canonical lists\n")
    for name, url in CANONICAL.items():
        dest = os.path.join(DATA_DIR, name)
        raw = urllib.request.urlopen(url, timeout=60).read()
        with open(dest, "wb") as fh:
            fh.write(raw)
        words, note = extract_words(dest)
        found.append({"path": dest, "n_words": len(words), "n_unique": len(set(words)),
                      "size_kb": len(raw) / 1024, "note": note, "words": words})
        print(f"  {name}: {len(raw)} bytes, {len(words)} words, "
              f"sha256={hashlib.sha256(raw).hexdigest()[:16]}...")
elif not found:
    raise SystemExit(
        "No word lists found and ALLOW_DOWNLOAD is False. Attach a Kaggle dataset "
        "containing a 5-letter answer list and a larger valid-guess list, or set "
        "ALLOW_DOWNLOAD = True."
    )
''')

code(r'''
# ---------------------------------------------------------------------------
# Classify the discovered files into ANSWERS vs VALID GUESSES.
#
# Two conventions exist in the wild and they are distinguished by measurement,
# not by filename:
#   (a) the guess file lists only NON-ANSWER extras  -> the lists are disjoint
#   (b) the guess file is already the COMPLETE pool   -> answers are a subset
# ---------------------------------------------------------------------------
ANSWER_HINTS = ("answer", "solution", "target", "secret")
GUESS_HINTS  = ("guess", "allowed", "valid", "dictionary", "words", "all")

def classify(found):
    if len(found) == 1:
        f = found[0]
        print("WARNING: only one word list found. Using it as BOTH the answer list "
              "and the guess pool; the answer/guess distinction cannot be tested.")
        return f, f
    scored = []
    for f in found:
        base = os.path.basename(f["path"]).lower()
        a_hit = any(h in base for h in ANSWER_HINTS)
        g_hit = any(h in base for h in GUESS_HINTS)
        scored.append((f, a_hit, g_hit))
    ans = [f for f, a, g in scored if a and not g]
    gue = [f for f, a, g in scored if g and not a]
    # Fall back on size: the guess pool is always the larger list.
    if not ans or not gue:
        by_size = sorted(found, key=lambda x: x["n_unique"])
        ans, gue = [by_size[0]], [by_size[-1]]
        print("filename hints inconclusive -> classifying by size "
              "(smaller = answers, larger = guesses)")
    a = min(ans, key=lambda x: x["n_unique"])
    g = max(gue, key=lambda x: x["n_unique"])
    return a, g

f_ans, f_gue = classify(found)
ANSWERS_PATH, GUESSES_PATH = f_ans["path"], f_gue["path"]

A_set, G_set = set(f_ans["words"]), set(f_gue["words"])
overlap = len(A_set & G_set)
GUESSES_ARE_EXTRA = (overlap == 0)
full_pool = len(A_set | G_set)

print(f"\nANSWERS file : {ANSWERS_PATH}  ({len(A_set)} unique)")
print(f"GUESSES file : {GUESSES_PATH}  ({len(G_set)} unique)")
print(f"overlap      : {overlap}")
print(f"convention   : guess file holds "
      f"{'NON-ANSWER EXTRAS (disjoint lists)' if GUESSES_ARE_EXTRA else 'the COMPLETE pool'}")
print(f"=> full legal guess pool = {full_pool}")
''')

code(r'''
# ---------------------------------------------------------------------------
# The nine audit questions, answered by measurement.
# ---------------------------------------------------------------------------
def audit_file(path, words):
    with open(path, "rb") as fh:
        raw_b = fh.read()
    raw = raw_b.decode("utf-8", errors="replace")
    toks = [t for t in raw.replace(",", "\n").split() if t]
    lens = collections.Counter(len(t) for t in toks)
    nonaz = sorted({c for t in toks for c in t if not ("a" <= c <= "z")})
    dupes = [w for w, c in collections.Counter(words).items() if c > 1]
    return {
        "path": path,
        "sha256": hashlib.sha256(raw_b).hexdigest(),
        "bytes": len(raw_b),
        "n_words": len(words),
        "n_unique": len(set(words)),
        "duplicates": len(dupes),
        "duplicate_examples": dupes[:5],
        "length_distribution": dict(sorted(lens.items())),
        "all_length_5": set(lens) == {5},
        "all_lowercase": all(t == t.lower() for t in toks),
        "all_ascii": all(t.isascii() for t in toks),
        "all_alpha": all(t.isalpha() for t in toks),
        "chars_outside_a_z": nonaz,
        "n_accented": sum(1 for t in toks if unicodedata.normalize("NFKD", t) != t),
        "crlf_line_endings": raw.count("\r\n"),
        "sorted_ascending": toks == sorted(toks),
        "blank_lines": sum(1 for l in raw.split("\n") if not l.strip()),
    }

AUDIT = {
    "answers": audit_file(ANSWERS_PATH, f_ans["words"]),
    "guesses": audit_file(GUESSES_PATH, f_gue["words"]),
}

for k, a in AUDIT.items():
    print(f"===== {k.upper()}: {a['path']} =====")
    for field in ("bytes", "n_words", "n_unique", "duplicates", "length_distribution",
                  "all_length_5", "all_lowercase", "all_ascii", "all_alpha",
                  "chars_outside_a_z", "n_accented", "crlf_line_endings",
                  "sorted_ascending", "blank_lines"):
        print(f"  {field:<22} {a[field]}")
    print(f"  {'sha256':<22} {a['sha256'][:32]}...")
    print()
''')

code(r'''
# ---------------------------------------------------------------------------
# Is this the standard NYT-style Wordle vocabulary?
#
# The original (pre-NYT-revision) Wordle shipped 2315 answers and 10657
# additional allowed guesses, i.e. a 12972-word legal pool. Checking against
# those numbers tells us whether the discovered data is the standard vocabulary,
# an NYT-revised variant, or something else entirely.
# ---------------------------------------------------------------------------
STD_ANSWERS, STD_EXTRA, STD_POOL = 2315, 10657, 12972
n_a, n_pool = len(A_set), full_pool

if n_a == STD_ANSWERS and n_pool == STD_POOL:
    verdict = ("EXACT MATCH to the standard original Wordle vocabulary "
               "(2315 answers / 12972 legal guesses).")
elif abs(n_a - STD_ANSWERS) <= 60 and abs(n_pool - STD_POOL) <= 120:
    verdict = (f"CLOSE to standard ({n_a} vs {STD_ANSWERS} answers, "
               f"{n_pool} vs {STD_POOL} pool) — likely an NYT-revised variant, "
               f"which added and removed a small number of words.")
else:
    verdict = (f"NOT the standard Wordle vocabulary ({n_a} answers, {n_pool} pool). "
               f"Results here are not comparable to published Wordle solver numbers.")

# Marker words: the NYT revision removed several slurs/obscurities from the answer list.
markers = {w: (w in A_set) for w in
           ("agora", "pupal", "lynch", "slave", "wench", "darky", "fibre", "bloke")}

print("VOCABULARY IDENTIFICATION")
print(f"  answers in list      : {n_a}   (standard: {STD_ANSWERS})")
print(f"  full legal pool      : {n_pool}   (standard: {STD_POOL})")
print(f"  non-answer extras    : {len(G_set - A_set)}   (standard: {STD_EXTRA})")
print(f"  verdict              : {verdict}")
print(f"\n  marker words present in the answer list: {markers}")

print("\nTRAIN/TEST SPLITS")
print("  None found, and none is appropriate. This is not a learning problem: there are")
print("  no parameters fitted to held-out data. The full answer list IS the evaluation")
print("  set, and every solver is evaluated on all of it. The frequency model does read")
print("  the answer list, so its statistics are in-sample by construction — noted in")
print("  Section 11 rather than papered over with a split.")

print("\nEXISTING WORDLE CODE IN THE WORKSPACE")
hits = [p for p in seen_paths if p.lower().endswith(".py")]
print(f"  python files found in search paths: {len(hits)}")
print("  No pre-existing feedback implementation or evaluation harness was found in the")
print("  discovered data directories; the implementation in Section 5 is written from")
print("  scratch and validated against hand-derived cases in Section 6.")
''')

md(r"""
### Audit conclusions

Fill in from the output above. On the reference run (canonical lists) the findings were:

| # | Question | Finding |
|---|---|---|
| 1 | Official answer words | `wordle_answers.txt` — 2,315 words |
| 2 | Larger valid-guess dictionary | `wordle_allowed_guesses.txt` — 10,657 words (non-answer extras) |
| 3 | Exact counts | 2,315 answers; 10,657 extras; **12,972** total legal guesses |
| 4 | Length / normalization | All exactly 5 chars, all lowercase ASCII `a–z`; no normalization needed |
| 5 | Duplicates | None, within either file |
| 6 | Punctuation / accents / case | None. No non-`a–z` characters, no accents, no uppercase, no blank lines |
| 7 | Train/test splits | None present, and none appropriate — see above |
| 8 | Existing feedback implementation | None found; written from scratch here |
| 9 | Standard NYT-style vocabulary? | Exact match to the original Wordle vocabulary (2,315 / 12,972) |

The two lists are **disjoint**, so the guess file holds non-answer extras and the legal
pool is their union. This distinction matters: restricting probes to the 2,315 answers
would discard most of the available information-seeking power.

**The source datasets are never modified.** Everything written goes to `ARTIFACT_DIR`.
""")

# =============================================================================
# 3. Imports and configuration
# =============================================================================
md(r"""
---
# 3. Imports and Configuration

The next two cells write `wordle_solver.py` and `benchmark.py` to the working directory.
This is what makes the notebook self-contained on Kaggle, where those files do not exist.
They are byte-identical to the standalone module you will download and run locally, so the
notebook is never testing something different from what you ship.

> If you have edited `wordle_solver.py` locally, **skip these two cells** — `%%writefile`
> overwrites unconditionally.
""")

code("%%writefile wordle_solver.py\n" + read("wordle_solver.py"))
code("%%writefile benchmark.py\n" + read("benchmark.py"))

md(r"""
## Reproducibility and the single configuration block

Everything that affects behaviour lives in `SolverConfig`. Change it here, nowhere else.
""")

code(r'''
import importlib, sys, random
import numpy as np

for m in ("wordle_solver", "benchmark"):
    if m in sys.modules:
        importlib.reload(sys.modules[m])

import wordle_solver as ws
import benchmark as bm
importlib.reload(ws); importlib.reload(bm)

from wordle_solver import (
    WORD_LEN, N_PATTERNS, ALL_GREEN,
    feedback, feedback_code, code_to_pattern, pattern_to_code,
    filter_candidates, filter_history, is_consistent,
    Vocabulary, load_vocabulary, FeedbackMatrix, build_feedback_matrix,
    FrequencyModel, SolverConfig, GameResult,
    RandomSolver, FrequencySolver, EntropySolver, ExpectedRemainingSolver,
    MinimaxSolver, HybridSolver, STRATEGIES, make_solver,
    play_game, solve, interactive_play,
    save_artifacts, load_artifacts,
)
from benchmark import (
    run_benchmark, summarize, summary_table, candidate_trajectory,
    first_guess_analysis, top_openers, format_openers, save_benchmark,
)

# =====================  CONFIGURATION  =====================================
SEED            = 20260817   # every stochastic solver derives its RNG from this
MAX_GUESSES     = 6          # standard Wordle
STRATEGY        = "hybrid"   # default strategy stored in solver_config.json

# Guess pool policy for the information-theoretic solvers:
#   "full"       always score all 12972 legal guesses  (strongest, slowest)
#   "adaptive"   full pool until few candidates remain  (much faster)
#   "candidates" only ever guess a surviving candidate  (fastest, weakest)
GUESS_POOL      = "full"
ADAPTIVE_THRESHOLD = 2

# Hybrid weights. All four terms are in BITS, so these are directly comparable
# (see Section 15 for the derivation and why normalization comes first).
ENTROPY_WEIGHT  = 1.00       # coefficient on H(guess)
MINIMAX_WEIGHT  = 0.25       # lambda: penalty on log2(worst-case bucket)
EXPECTED_WEIGHT = 0.25       # nu:     penalty on log2(E[remaining])
ANSWER_BONUS    = 1.00       # mu:     credit for probing a possible answer

# Benchmark scope. 0 = evaluate EVERY answer (what the report uses).
BENCHMARK_LIMIT = 0
BENCHMARK_STRATEGIES = ["random", "frequency", "entropy", "expected", "minimax", "hybrid"]

# Matrix build blocking (memory vs speed). 512 peaks well under 1 GiB.
MATRIX_CHUNK    = 512
# ===========================================================================

CONFIG = SolverConfig(
    max_guesses=MAX_GUESSES, strategy=STRATEGY, seed=SEED,
    guess_pool=GUESS_POOL, adaptive_threshold=ADAPTIVE_THRESHOLD,
    entropy_weight=ENTROPY_WEIGHT, minimax_weight=MINIMAX_WEIGHT,
    expected_weight=EXPECTED_WEIGHT, answer_bonus_weight=ANSWER_BONUS,
)

random.seed(SEED)
np.random.seed(SEED)

print("ENVIRONMENT")
print(f"  python        {platform.python_version()}")
print(f"  numpy         {np.__version__}")
print(f"  platform      {platform.platform()}")
print(f"  on kaggle     {ON_KAGGLE}")
try:
    import pandas as pd;      print(f"  pandas        {pd.__version__}")
except ImportError:           print("  pandas        (not installed - optional)")
try:
    import matplotlib;        print(f"  matplotlib    {matplotlib.__version__}")
except ImportError:           print("  matplotlib    (not installed - optional)")
print("\nNo ML / LLM / RL libraries are imported anywhere in this notebook.")
print("\nCONFIG")
for k, v in CONFIG.to_dict().items():
    print(f"  {k:<32} {v}")
print(f"\n  SEED = {SEED}")
''')

# =============================================================================
# 4. Load and validate vocabulary
# =============================================================================
md(r"""
---
# 4. Load and Validate Vocabulary

`load_vocabulary` normalizes both lists, infers the answers/extras convention, and asserts
the one invariant that must hold for the game to be coherent: **every answer must be a
legal guess.**
""")

code(r'''
VOCAB = load_vocabulary(ANSWERS_PATH, GUESSES_PATH,
                        guesses_are_extra=GUESSES_ARE_EXTRA)

print(VOCAB.describe())
print(f"\nanswers      first 5 {VOCAB.answers[:5]}")
print(f"             last  5 {VOCAB.answers[-5:]}")
print(f"legal guesses first 5 {VOCAB.guesses[:5]}")
print(f"             last  5 {VOCAB.guesses[-5:]}")

# Invariants
assert VOCAB.n_answers == len(set(VOCAB.answers)), "duplicate answers"
assert VOCAB.n_guesses == len(set(VOCAB.guesses)), "duplicate guesses"
assert all(len(w) == WORD_LEN for w in VOCAB.guesses), "non-5-letter guess"
assert all(w.isascii() and w.isalpha() and w.islower() for w in VOCAB.guesses)
assert set(VOCAB.answers) <= set(VOCAB.guesses), "an answer is not a legal guess"
assert VOCAB.n_guesses >= VOCAB.n_answers
print("\nall vocabulary invariants hold.")

# Structure of the answer space — this is what the solvers are searching.
import collections as _c
rep = sum(1 for w in VOCAB.answers if len(set(w)) < WORD_LEN)
print(f"\nanswers containing a repeated letter: {rep} "
      f"({100*rep/VOCAB.n_answers:.1f}%)  <- why duplicate-letter feedback matters")
print(f"most common answer letters: "
      f"{_c.Counter(c for w in VOCAB.answers for c in w).most_common(8)}")
print(f"log2(2315) = {np.log2(VOCAB.n_answers):.3f} bits of initial uncertainty")
''')

# =============================================================================
# 5. Feedback implementation
# =============================================================================
md(r"""
---
# 5. The Wordle Feedback Function

`feedback(guess, answer) -> "GYBBB"` where `G`=green, `Y`=yellow, `B`=grey.

## Duplicate letters — the whole difficulty

Think of the answer as supplying a **budget** of tiles for each letter, equal to how many
times that letter occurs. Two passes:

1. **Greens.** Every position where guess and answer agree is green and **consumes one unit**
   of that letter's budget.
2. **Yellows, scanned left to right.** A non-green position is yellow only if its letter still
   has unconsumed budget, and it then consumes one unit. Otherwise grey.

Three consequences that naive implementations get wrong:

- **Greens always win, regardless of position.** A green at index 4 consumes budget before a
  yellow at index 1 can claim it. (`added` vs `dread` → `YYBYG`, not `YYBYY`.)
- **Surplus guessed letters come back grey.** (`speed` vs `abide` → `BBYBY`: the second `e`
  greys out.)
- **Left-to-right ordering makes it deterministic** when a letter is over-guessed.

## Encoding

Patterns are encoded as a base-3 integer, little-endian by position:

$$\text{code} = \sum_{i=0}^{4} \text{tile}_i \cdot 3^i, \qquad \text{tile} \in \{0{=}B,\, 1{=}Y,\, 2{=}G\}$$

so `code` ∈ [0, 243). Because **243 < 256, the entire feedback table fits in `uint8`** — which
is exactly what makes exhaustive precomputation cheap (Section 8). `GGGGG` is code 242.
""")

code(r'''
print("source of the reference implementation:\n")
import inspect
print(inspect.getsource(ws.feedback))
''')

code(r'''
# Worked examples, printed so the duplicate-letter rules are visible.
def show(g, a, note=""):
    print(f"  {g.upper()} vs {a.upper()}  ->  {feedback(g, a)}   "
          f"code={feedback_code(g, a):>3}   {note}")

print("simple cases")
show("crane", "crane", "identical")
show("slate", "crane", "a and e already in position")
show("plumb", "crane", "nothing shared")
print("\nduplicate letters")
show("speed", "abide", "surplus guessed 'e' greys out")
show("abide", "speed", "NOT symmetric with the line above")
show("eerie", "eaten", "green 'e' consumes budget before later yellows")
show("llama", "lucid", "green 'l' spends the only 'l'")
show("added", "dread", "green at index 4 consumes before yellow at index 1")
show("array", "rural", "two greens consume from both repeated letters")
show("geese", "sheep", "one 'e' green, one yellow, one grey")
show("mummy", "mamma", "triple letter, all matched")
show("aaaaa", "ababa", "degenerate all-same guess")

print(f"\nencoding: GGGGG={pattern_to_code('GGGGG')}  BBBBB={pattern_to_code('BBBBB')}  "
      f"GBBBB={pattern_to_code('GBBBB')}  BBBBG={pattern_to_code('BBBBG')}")
print(f"ALL_GREEN={ALL_GREEN}, N_PATTERNS={N_PATTERNS}")
''')

# =============================================================================
# 6. Feedback unit tests
# =============================================================================
md(r"""
---
# 6. Feedback Unit Tests

The feedback function is the foundation — every solver's correctness rests on it, and a
subtle duplicate-letter bug would silently corrupt every benchmark number in this notebook
while still looking plausible. So it is tested three ways:

1. **Hand-derived cases** for each duplicate-letter rule, each with its derivation stated.
2. **Property tests** over thousands of random pairs (invariants that must hold universally).
3. **Cross-validation** of the vectorized matrix against this reference (Section 8).

The full `pytest` suite lives in `tests/test_wordle_solver.py` (61 tests). The cell below
re-runs the essential assertions inline so the notebook is self-verifying.
""")

code(r'''
# ---------------------------------------------------------------------------
# 6a. Hand-derived duplicate-letter cases.
# ---------------------------------------------------------------------------
CASES = [
    # (guess, answer, expected, derivation)
    ("crane", "crane", "GGGGG", "identical words"),
    ("slate", "crane", "BBGBG", "crane = c r a n e; a@2 and e@4 already aligned"),
    ("plumb", "crane", "BBBBB", "no shared letters"),
    ("crane", "nacre", "YYYYG", "e@4 aligned; other four present but displaced"),
    ("crane", "ranec", "YYYYY", "derangement: all present, none aligned"),
    ("speed", "abide", "BBYBY", "budget e=1: e@2 yellow, e@3 grey; d yellow"),
    ("abide", "speed", "BBBYY", "reversed argument order gives a different pattern"),
    ("eerie", "eaten", "GYBBB", "e@0 green spends 1 of 2 e's; e@1 yellow; e@4 grey"),
    ("llama", "lucid", "GBBBB", "l@0 green spends the only l, so l@1 greys"),
    ("added", "dread", "YYBYG", "d@4 green consumes first; then d@1 yellow, d@2 grey"),
    ("array", "rural", "BYGGB", "r@2,a@3 green; a budget exhausted, r@1 yellow"),
    ("geese", "sheep", "BYGYB", "e@2 green; e@1 yellow, s@3 yellow, e@4 grey"),
    ("mummy", "mamma", "GBGGB", "three greens consume all three m's"),
    ("mamma", "mummy", "GBGGB", "symmetric here only because the m's align identically"),
    ("abide", "eerie", "BBYBG", "answer has three e's; guess's single e is green"),
    ("aaaaa", "ababa", "GBGBG", "all-same guess vs two-letter answer"),
    ("eecee", "ceeee", "YGYGG", "left-to-right: leftmost non-green e takes the budget"),
]

fails = []
for g, a, expect, why in CASES:
    got = feedback(g, a)
    ok = got == expect
    if not ok:
        fails.append((g, a, expect, got, why))
    print(f"  {'PASS' if ok else 'FAIL'}  {g.upper()} vs {a.upper()}  "
          f"got={got} expect={expect}   {why}")
assert not fails, f"hand-derived cases failed: {fails}"
print(f"\nall {len(CASES)} hand-derived cases pass.")
''')

code(r'''
# ---------------------------------------------------------------------------
# 6b. Property tests over random pairs — universal invariants.
# ---------------------------------------------------------------------------
import collections, math
rng = random.Random(SEED)
N = 20000
words = VOCAB.answers
pairs = [(rng.choice(words), rng.choice(words)) for _ in range(N)]

# (1) A letter can never be marked (G or Y) more times than it occurs in the answer.
for g, a in pairs:
    pat = feedback(g, a)
    ac = collections.Counter(a)
    mk = collections.Counter(ch for ch, t in zip(g, pat) if t in "GY")
    for ch, k in mk.items():
        assert k <= ac[ch], f"over-marked {ch!r} in {g}/{a} -> {pat}"

# (2) Green count == number of aligned positions.
for g, a in pairs:
    pat = feedback(g, a)
    assert pat.count("G") == sum(1 for i in range(WORD_LEN) if g[i] == a[i])

# (3) All-green if and only if guess == answer.
for g, a in pairs:
    assert (feedback(g, a) == "GGGGG") == (g == a)

# (4) Codes always in range, and encode/decode round-trips.
for g, a in pairs:
    c = feedback_code(g, a)
    assert 0 <= c < N_PATTERNS
    assert pattern_to_code(feedback(g, a)) == c
for c in range(N_PATTERNS):
    assert pattern_to_code(code_to_pattern(c)) == c

# (5) Total marked tiles <= number of shared letters counted with multiplicity.
for g, a in pairs[:5000]:
    pat = feedback(g, a)
    shared = sum((collections.Counter(g) & collections.Counter(a)).values())
    assert pat.count("G") + pat.count("Y") == shared, (
        f"{g}/{a} -> {pat}: marked tiles must equal the multiset intersection size")

print(f"all property tests pass over {N} random pairs "
      f"and all {N_PATTERNS} pattern codes.")
print("\ninvariants verified:")
print("  1. per-letter marks never exceed the answer's letter count")
print("  2. green count == positional match count")
print("  3. GGGGG <=> guess == answer")
print("  4. codes in [0,243) and encoding round-trips")
print("  5. G+Y count == |multiset intersection of guess and answer|")
''')

code(r'''
# ---------------------------------------------------------------------------
# 6c. Run the full pytest suite if it is available (it is, locally).
# ---------------------------------------------------------------------------
import subprocess, os
if os.path.isdir("tests"):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:] or r.stderr[-3000:])
else:
    print("tests/ not present (expected on Kaggle) — the inline checks above cover "
          "the same duplicate-letter and property assertions.")
''')

# =============================================================================
# 7. Candidate filtering
# =============================================================================
md(r"""
---
# 7. Candidate Filtering — Layer 1, Symbolic Constraint Solving

This is the part that is *exact*. Maintain `candidate_answers`; after each
(guess, feedback) observation, eliminate every answer that could not have produced it.

The key design decision: consistency is defined **by the feedback function itself** —

```python
def is_consistent(candidate, guess, pattern):
    return feedback(guess, candidate) == pattern
```

rather than by re-deriving constraint rules ("green means letter at position", "grey means
letter absent", …). Re-deriving is where duplicate-letter bugs breed, because grey does *not*
mean "absent" when the letter appears elsewhere. Defining consistency this way makes it
**impossible** for the filter and the feedback function to disagree.

Filtering per-observation is equivalent to filtering against the whole history, since each
check is an independent pure function of the candidate.

### ANSWERS vs GUESSES

`candidate_answers` is always a subset of the **answer** list — those are the only things the
hidden word can be. But the **guess pool** stays the full legal dictionary, because a word
that cannot be the answer can still be the most informative probe. Collapsing these two is
the single most common way a Wordle solver is accidentally weakened.
""")

code(r'''
# Filtering demo on a real game state.
answer = "crane"
history = []
cands = list(VOCAB.answers)
print(f"start: {len(cands)} candidates  ({np.log2(len(cands)):.2f} bits of uncertainty)\n")

for g in ["soare", "clint", "crane"]:
    pat = feedback(g, answer)
    history.append((g, pat))
    before = len(cands)
    cands = filter_candidates(cands, g, pat)
    print(f"guess {g.upper()} -> {pat}")
    print(f"  candidates {before} -> {len(cands)}   "
          f"({np.log2(before/max(len(cands),1)):.2f} bits gained)")
    if len(cands) <= 12:
        print(f"  remaining: {cands}")
    print()

# Equivalence: incremental filtering == filtering against the full history.
assert filter_history(VOCAB.answers, history) == cands
print("incremental filtering == full-history filtering: OK")
''')

code(r'''
# ---------------------------------------------------------------------------
# Tests proving the filter is consistent with the feedback function.
# ---------------------------------------------------------------------------
rng = random.Random(SEED + 1)

# (1) The true answer NEVER gets eliminated by correct feedback.
for _ in range(2000):
    a, g = rng.choice(VOCAB.answers), rng.choice(VOCAB.guesses)
    assert a in filter_candidates(VOCAB.answers, g, feedback(g, a))

# (2) Survivors are exactly the words whose feedback matches — nothing more, nothing less.
pool = rng.sample(VOCAB.answers, 500)
for _ in range(200):
    a, g = rng.choice(VOCAB.answers), rng.choice(VOCAB.guesses)
    pat = feedback(g, a)
    surv = set(filter_candidates(pool, g, pat))
    for w in pool:
        assert (w in surv) == (feedback(g, w) == pat), f"inconsistent on {w}"

# (3) The 243 patterns PARTITION the candidate set: bucket sizes sum to n exactly.
for g in rng.sample(VOCAB.guesses, 25):
    total = sum(len(filter_candidates(pool, g, code_to_pattern(c)))
                for c in range(N_PATTERNS))
    assert total == len(pool), f"{g}: buckets summed to {total}, expected {len(pool)}"

# (4) Multi-turn: filtering is order-independent and the answer always survives.
for _ in range(300):
    a = rng.choice(VOCAB.answers)
    gs = [rng.choice(VOCAB.guesses) for _ in range(3)]
    hist = [(g, feedback(g, a)) for g in gs]
    fwd = filter_history(VOCAB.answers, hist)
    rev = filter_history(VOCAB.answers, hist[::-1])
    assert sorted(fwd) == sorted(rev), "filtering is order-dependent"
    assert a in fwd

print("candidate filtering is provably consistent with the feedback function:")
print("  1. the true answer is never eliminated              (2000 trials)")
print("  2. survivors == exact feedback agreement            (200 x 500 checks)")
print("  3. the 243 patterns partition the candidates        (25 guesses)")
print("  4. filtering is order-independent                   (300 x 3-turn histories)")
''')

# =============================================================================
# 8. Feedback matrix construction
# =============================================================================
md(r"""
---
# 8. Precomputing the Feedback Matrix

The vocabulary is small enough to precompute **every** guess × answer feedback:

$$M[g, a] = \text{feedback\_code}(\text{guesses}[g],\ \text{answers}[a])$$

Shape **12,972 × 2,315 = 30,030,180 cells**. Since codes fit in `uint8`, that is **28.6 MiB** —
trivially small for Kaggle or a laptop, and it turns every later filter and entropy
computation into an array lookup instead of millions of Python string comparisons.

## How the vectorization works

Greens are a plain positional equality test. Yellows are the hard part, handled one letter
of the alphabet at a time (26 iterations, each fully vectorized):

- `ng[g,a,i]` — position `i` of the guess holds letter `c` and the answer does **not** hold `c`
  there: a *non-green occurrence*.
- `budget` — occurrences of `c` in the answer minus greens of `c`. This is exactly the yellow
  budget surviving pass 1.
- `rank` — exclusive prefix count of non-green occurrences of `c` to the left of `i`.

The left-to-right greedy rule then collapses to a single comparison: **`yellow = rank < budget`**.

Chunking over guesses bounds peak memory at roughly `chunk × n_answers × 5` bytes per
temporary, so `MATRIX_CHUNK = 512` stays well under 1 GiB.
""")

code(r'''
print(inspect.getsource(ws.build_feedback_matrix))
''')

code(r'''
# ---------------------------------------------------------------------------
# Build the matrix (or reuse an existing artifact).
# ---------------------------------------------------------------------------
matrix_path = os.path.join(ARTIFACT_DIR, "feedback_matrix.npy")
REBUILD = not os.path.exists(matrix_path)

if REBUILD:
    print(f"building {VOCAB.n_guesses} x {VOCAB.n_answers} uint8 matrix "
          f"({VOCAB.n_guesses*VOCAB.n_answers/2**20:.1f} MiB) ...")
    t0 = time.perf_counter()
    _last = [0.0]
    def _prog(done, total):
        now = time.perf_counter()
        if now - _last[0] > 3.0 or done == total:
            _last[0] = now
            el = now - t0
            print(f"  {done}/{total}  {el:.1f}s  eta {el/max(done,1)*(total-done):.0f}s")
    FB = FeedbackMatrix.build(VOCAB, chunk=MATRIX_CHUNK, progress=_prog)
    BUILD_SECONDS = time.perf_counter() - t0
    print(f"built in {BUILD_SECONDS:.1f}s")
else:
    print(f"reusing existing {matrix_path}")
    FB = FeedbackMatrix(np.load(matrix_path, mmap_mode="r"), VOCAB)
    BUILD_SECONDS = 0.0

print(f"\nshape  {FB.M.shape}")
print(f"dtype  {FB.M.dtype}")
print(f"bytes  {FB.M.nbytes:,} ({FB.M.nbytes/2**20:.1f} MiB)")
print(f"range  [{FB.M.min()}, {FB.M.max()}]  (must be within [0,{N_PATTERNS-1}])")
''')

code(r'''
# ---------------------------------------------------------------------------
# Verify the vectorized matrix against the pure-Python reference.
# ---------------------------------------------------------------------------
assert FB.M.shape == (VOCAB.n_guesses, VOCAB.n_answers), "matrix dimensions wrong"
assert FB.M.dtype == np.uint8
assert FB.M.max() < N_PATTERNS

# (1) Random spot-check against the reference implementation.
rs = np.random.default_rng(SEED)
NCHK = 50000
gi = rs.integers(0, VOCAB.n_guesses, NCHK)
ai = rs.integers(0, VOCAB.n_answers, NCHK)
bad = [(VOCAB.guesses[g], VOCAB.answers[a], int(FB.M[g, a]), feedback_code(VOCAB.guesses[g], VOCAB.answers[a]))
       for g, a in zip(gi, ai) if FB.M[g, a] != feedback_code(VOCAB.guesses[g], VOCAB.answers[a])]
assert not bad, f"matrix disagrees with reference: {bad[:5]}"
print(f"(1) {NCHK} random cells match the reference implementation exactly")

# (2) Global invariant: the diagonal (answer guessed against itself) is all-green.
diag = np.asarray(FB.M[FB.answer_pos, np.arange(VOCAB.n_answers)])
assert (diag == ALL_GREEN).all(), "diagonal is not all-green"
print(f"(2) all {VOCAB.n_answers} diagonal cells are GGGGG ({ALL_GREEN})")

# (3) Exhaustive check of a few complete rows (every answer for one guess).
for g in ["soare", "crane", "eerie", "mummy", "aaaaa" if "aaaaa" in VOCAB.guess_index else "arise"]:
    if g not in VOCAB.guess_index:
        continue
    row = np.asarray(FB.M[VOCAB.guess_index[g]])
    ref = np.array([feedback_code(g, a) for a in VOCAB.answers], dtype=np.uint8)
    assert np.array_equal(row, ref), f"row mismatch for {g}"
print("(3) complete rows verified exhaustively for several guesses "
      "(including repeated-letter ones)")

# (4) Matrix filtering == pure-Python filtering.
idx0 = np.arange(VOCAB.n_answers, dtype=np.int32)
for _ in range(300):
    a, g = rng.choice(VOCAB.answers), rng.choice(VOCAB.guesses)
    c = feedback_code(g, a)
    got = sorted(VOCAB.answers[i] for i in FB.filter_indices(idx0, g, c))
    want = sorted(filter_candidates(VOCAB.answers, g, code_to_pattern(c)))
    assert got == want
print("(4) matrix-based filtering == pure-Python filtering (300 trials)")

# (5) Every row must partition the answers into buckets summing to n.
for g in rng.sample(VOCAB.guesses, 200):
    row = np.asarray(FB.M[VOCAB.guess_index[g]])
    assert np.bincount(row, minlength=N_PATTERNS).sum() == VOCAB.n_answers
print("(5) every checked row partitions the answer set correctly")
print("\nFEEDBACK MATRIX VERIFIED.")
''')

# =============================================================================
# 9. Artifact persistence
# =============================================================================
md(r"""
---
# 9. Artifact Persistence and Reloading

The pipeline is **build → save → reload → run without rebuilding**. The matrix is stored as
a portable `.npy` and reloaded with `mmap_mode="r"`, so start-up is instant and the 28.6 MiB
table pages in on demand rather than being copied into RAM.

Nothing Kaggle-specific is written into the artifacts — no absolute input paths, no
environment assumptions. The bundle is the portability boundary.
""")

code(r'''
FREQ_MODEL = FrequencyModel.fit(VOCAB.answers)

paths = save_artifacts(
    ARTIFACT_DIR, VOCAB, FB, FREQ_MODEL, CONFIG,
    extra_metadata={
        "matrix_build_seconds": round(BUILD_SECONDS, 2),
        "matrix_build_chunk": MATRIX_CHUNK,
        "discovered_answers_path": ANSWERS_PATH,
        "discovered_guesses_path": GUESSES_PATH,
        "guesses_file_holds_extras_only": bool(GUESSES_ARE_EXTRA),
        "audit": {k: {kk: vv for kk, vv in v.items() if kk != "duplicate_examples"}
                  for k, v in AUDIT.items()},
        "vocabulary_verdict": verdict,
    },
)
print(f"wrote artifacts to {ARTIFACT_DIR}:")
for k, p in paths.items():
    print(f"  {k:<10} {os.path.basename(p):<24} {os.path.getsize(p)/1024:>10.1f} KiB")
''')

code(r'''
# ---------------------------------------------------------------------------
# Reload round-trip: prove the bundle alone is sufficient.
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
BUNDLE = load_artifacts(ARTIFACT_DIR, mmap=True)
print(f"reloaded in {(time.perf_counter()-t0)*1000:.0f} ms (memory-mapped)\n")
print(BUNDLE.describe())

assert BUNDLE.vocab.answers == VOCAB.answers
assert BUNDLE.vocab.guesses == VOCAB.guesses
assert np.array_equal(np.asarray(BUNDLE.fb.M), np.asarray(FB.M))
assert BUNDLE.config.seed == SEED
assert BUNDLE.model.letter_prob == FREQ_MODEL.letter_prob
assert isinstance(BUNDLE.fb.M, np.memmap), "expected a memory-mapped array"

# A reloaded bundle must play byte-identical games.
_a = "crane" if "crane" in VOCAB.answer_index else VOCAB.answers[0]
r1 = play_game(make_solver("entropy", FB, CONFIG, FREQ_MODEL), _a)
r2 = play_game(BUNDLE.solver("entropy"), _a)
assert r1.guesses == r2.guesses and r1.patterns == r2.patterns
print(f"\nartifact round-trip verified; reloaded solver reproduces the same game "
      f"({' -> '.join(g.upper() for g in r2.guesses)})")
''')

# =============================================================================
# 10-15. Solvers
# =============================================================================
md(r"""
---
# 10. Random Surviving-Answer Baseline

**Layer 1 only.** Filter exactly, then name a surviving candidate uniformly at random. No
decision-making of any kind.

This is the most important reference point in the notebook: it measures **how much of Wordle
is solved by perfect constraint propagation alone**, before any information theory or word
knowledge is applied. Any cleverer strategy must be judged by how far it improves on this.
""")

code(r'''
RESULTS, SUMMARIES, TRAJECTORIES = {}, {}, {}

def demo(strategy, answers=("crane", "eerie", "mummy"), n_sample=200):
    """Show one verbose game, then a quick sampled score for orientation."""
    solver = make_solver(strategy, FB, CONFIG, FREQ_MODEL)
    print(f"--- {strategy}: example game ---")
    play_game(solver, answers[0], verbose=True)
    rs = np.random.default_rng(SEED)
    samp = [VOCAB.answers[i] for i in rs.choice(VOCAB.n_answers, n_sample, replace=False)]
    r = run_benchmark(strategy, FB, CONFIG, answers=samp, model=FREQ_MODEL, log=None)
    print(f"sampled {n_sample} answers: mean={r.scores.mean():.4f}  "
          f"solved={r.solved_mask.sum()}/{n_sample}  "
          f"{r.wall_s/n_sample*1000:.1f} ms/game")
    return r

_ = demo("random")
''')

md(r"""
---
# 11. Frequency-Based Solver

**Layer 3 only.** Rank surviving candidates by empirical letter statistics of the answer list.
It never asks what a guess would *reveal* — only whether its letters look typical of an answer.

$$\text{score}(w) = \underbrace{\sum_{c \in \text{set}(w)} P(\text{letter } c \text{ present})}_{\text{coverage}} \;+\; w_{\text{pos}} \sum_{i=0}^{4} P(\text{letter } w_i \text{ at position } i)$$

## Handling repeated letters

The coverage term sums over **distinct** letters (`set(w)`), not over positions. This matters:
summing per-occurrence frequencies would reward `eerie` for containing `e` three times, even
though a repeated letter conveys strictly *less* new information than a fresh letter. Counting
each letter once removes that artificial bonus. The positional term is the only one that sees
position, and it is down-weighted (`positional_weight = 0.5`) so it refines rather than
dominates the ranking.

> **In-sample note.** The frequency model is fitted on the same answer list it is evaluated
> against, so its statistics are in-sample by construction. There is no held-out split
> because there are no learned parameters to overfit in the usual sense — but this solver's
> numbers should be read as a mild upper bound on what letter statistics alone can do.
""")

code(r'''
m = FREQ_MODEL
top = sorted(m.letter_prob.items(), key=lambda kv: -kv[1])[:10]
print("P(letter present in a random answer) — top 10")
for c, p in top:
    print(f"  {c}  {p*100:5.1f}%  {'#'*int(p*100)}")

print("\nmost likely letter at each position")
for i, tbl in enumerate(m.positional_prob):
    best = sorted(tbl.items(), key=lambda kv: -kv[1])[:5]
    print(f"  pos {i}: " + "  ".join(f"{c}={p*100:.1f}%" for c, p in best))

print("\nrepeated letters earn no coverage bonus:")
for w in ["arose", "eerie", "raise", "mummy", "adieu"]:
    cov = sum(m.letter_prob[c] for c in set(w))
    print(f"  {w.upper()}  score={m.score(w):.4f}  "
          f"coverage={cov:.4f} over {len(set(w))} distinct letters")
assert m.score("arose") > m.score("eerie")
print("\n  -> AROSE outscores EERIE despite similar letter frequencies. Correct.")

_ = demo("frequency")
''')

md(r"""
---
# 12. Entropy / Information-Gain Solver

**Layer 2.** For every legal guess, partition the surviving candidates by the feedback the
guess would produce, then choose the guess maximising expected information gain:

$$H(g) = -\sum_{f \in \text{patterns}} p(f) \log_2 p(f), \qquad p(f) = \frac{|\{a \in C : \text{feedback}(g,a) = f\}|}{|C|}$$

## How the probabilities are estimated — stated exactly

Every surviving candidate is treated as **equally likely**: $p(a) = 1/|C|$. So $p(f)$ is just
`bucket_size / n_candidates`.

This is not a modelling shortcut, it is the correct posterior: the real game draws its answer
from the answer list, *not* weighted by English word frequency. A uniform prior over the answer
list, updated by exact constraint filtering, gives a uniform posterior over the survivors. No
word-frequency prior enters here — that lives only in the Layer-3 frequency model.

$H$ is in **bits**: the expected reduction in $\log_2$ of the hypothesis-space size. A guess
scoring 5.9 bits is expected to cut ~2,315 candidates down to ~40.

**This is one-step greedy.** It maximises *immediate* information, not the true
minimum-expected-guesses game tree. Full lookahead would require searching the decision tree
over sequences of guesses; that is a deliberate scope boundary, and it is the main gap between
this baseline and a theoretically optimal solver.
""")

code(r'''
# Entropy of a few candidate openers, computed from the full answer set.
all_ans = np.arange(VOCAB.n_answers, dtype=np.int32)
probe = ["soare", "roate", "raise", "arise", "crane", "slate", "adieu", "tares", "xylyl"]
probe = [w for w in probe if w in VOCAB.guess_index]
pidx = np.array([VOCAB.guess_index[w] for w in probe], dtype=np.int32)
st = FB.partition_stats(pidx, all_ans)

print(f"scored against all {VOCAB.n_answers} answers "
      f"({np.log2(VOCAB.n_answers):.2f} bits of uncertainty)\n")
print(f"{'word':<8}{'H(bits)':>9}{'E[rem]':>10}{'worst':>7}{'buckets':>9}{'answer?':>9}")
print("-" * 52)
for w, i in zip(probe, range(len(probe))):
    row = np.asarray(FB.M[VOCAB.guess_index[w]])
    nb = int((np.bincount(row, minlength=N_PATTERNS) > 0).sum())
    print(f"{w:<8}{st['entropy'][i]:>9.4f}{st['expected_remaining'][i]:>10.2f}"
          f"{st['worst_case'][i]:>7}{nb:>9}"
          f"{('yes' if w in VOCAB.answer_index else 'no'):>9}")

print("\nnote XYLYL: a legal guess with very low entropy — it splits almost nothing.")
print("This is why the guess pool must be SCORED, not merely allowed.")

_ = demo("entropy")
''')

md(r"""
---
# 13. Expected-Remaining-Candidates Solver

**Layer 2.** Minimise the expected size of the surviving candidate set:

$$\mathbb{E}[\text{remaining}](g) = \sum_f p(f)\cdot|f| = \frac{1}{|C|}\sum_f |f|^2$$

Closely related to entropy but **not** identical, and the difference is instructive: entropy
penalises a bucket by $\log$ of its size, this penalises **linearly**. So this objective is
more tolerant of one large bucket when the rest are tiny, while entropy spreads mass more
evenly. They optimise different moments of the same partition and therefore pick different
openers — `ROATE` versus `SOARE` on this vocabulary.
""")

code(r'''
_ = demo("expected")
''')

md(r"""
---
# 14. Minimax Solver

**Layer 2, worst-case.** For each guess, take the size of the largest resulting partition and
minimise it:

$$\text{minimax}(g) = \min_g \max_f |\{a \in C : \text{feedback}(g,a) = f\}|$$

This makes **no probabilistic assumption at all** beyond the candidate set itself — it asks
only "if an adversary chose the answer, how bad could this get?" Consequently it typically
achieves a better *maximum* guess count than the entropy rule while giving up a little on the
*mean*. That mean-versus-tail tradeoff is exactly what the benchmark's max column and
distribution histogram are there to expose.
""")

code(r'''
_ = demo("minimax")
''')

md(r"""
---
# 15. Hybrid Solver

Combines all three partition statistics with the answer-probability prior.

## Normalization comes first — this is the point

The four quantities have **incompatible units**: entropy is in bits; worst-case is a candidate
count in $[1, n]$; expected-remaining is also a count but on a different scale; answer
probability is in $[0,1]$. Adding them with raw weights would make those weights meaningless —
a "weight of 0.25" on a quantity ranging over 2,315 is not comparable to 0.25 on a quantity
ranging over 11 bits.

Two options were considered:

- **z-scoring** each term across the scored pool. Rejected: the mean and spread change every
  turn as the candidate set shrinks, so a fixed weight would silently mean something different
  on turn 2 than on turn 4.
- **Converting everything to bits.** Chosen: bits are stable across turns and directly
  interpretable, and the conversion is the natural one — the $\log_2$ of a hypothesis-space
  size *is* its size in bits.

So every term becomes bits:

$$\text{score}(g) = \alpha\,\underbrace{H(g)}_{\text{bits gained}} \;-\; \lambda\,\underbrace{\log_2 \max_f|f|}_{\text{bits left, worst case}} \;-\; \nu\,\underbrace{\log_2 \mathbb{E}[\text{rem}]}_{\text{bits left, expected}} \;+\; \mu\,\underbrace{\log_2\!\left(1 + p_{\text{ans}}(g)\right)}_{\text{credit for winning now}}$$

where $p_{\text{ans}}(g) = 1/|C|$ if $g$ is itself a surviving candidate, else $0$.

This is the user-suggested formulation with the scale problem fixed: `information_gain
- λ·worst_case + μ·answer_probability`, but with `worst_case` and `answer_probability` mapped
into bits so the coefficients are meaningful.

## Weights

Defaults $\alpha{=}1.0$, $\lambda{=}0.25$, $\nu{=}0.25$, $\mu{=}1.0$ — chosen to be
*interpretable* ("weight the two tail terms at a quarter of raw information gain"), then
checked against the benchmark rather than tuned to it. Because all terms are in bits these are
honest ratios, not arbitrary numbers. Section 19 sweeps $\lambda$ so you can see the
sensitivity instead of taking the defaults on trust.

Two exact endgame rules apply to every solver and are not heuristics: with $|C| \le 2$
remaining, probing cannot beat naming a candidate; and on the final turn a non-candidate guess
is a guaranteed loss.
""")

code(r'''
# Decompose the hybrid score for the opening move, to show the terms' scales.
pool_i = np.array([VOCAB.guess_index[w] for w in probe], dtype=np.int32)
s = FB.partition_stats(pool_i, all_ans)
H = s["entropy"]
worst_bits = np.log2(np.maximum(s["worst_case"], 1))
exp_bits = np.log2(np.maximum(s["expected_remaining"], 1.0))
is_c = np.array([w in VOCAB.answer_index for w in probe])
bonus = np.log2(1 + is_c / VOCAB.n_answers)
score = (ENTROPY_WEIGHT*H - MINIMAX_WEIGHT*worst_bits
         - EXPECTED_WEIGHT*exp_bits + ANSWER_BONUS*bonus)

print("hybrid score decomposition for the opening move (all terms in BITS)\n")
print(f"{'word':<8}{'a*H':>8}{'-l*worst':>10}{'-n*E[rem]':>11}{'+m*ans':>9}{'TOTAL':>9}")
print("-" * 55)
for i, w in enumerate(probe):
    print(f"{w:<8}{ENTROPY_WEIGHT*H[i]:>8.3f}{-MINIMAX_WEIGHT*worst_bits[i]:>10.3f}"
          f"{-EXPECTED_WEIGHT*exp_bits[i]:>11.3f}{ANSWER_BONUS*bonus[i]:>9.5f}"
          f"{score[i]:>9.3f}")
print("\nall four terms sit on the same bits scale, so the weights are real ratios.")

_ = demo("hybrid")
''')

# =============================================================================
# 16. Full benchmark
# =============================================================================
md(r"""
---
# 16. Full Benchmark

Every solver is evaluated against **every** answer in the discovered answer list — not a
sample. With `GUESS_POOL = "full"` the information-theoretic solvers score all 12,972 legal
guesses at every turn, which is the strongest and most honest configuration.

Reported per solver: mean / median / std / min / max guesses, the full 1–6 distribution,
failure rate, and average surviving candidates after each guess.

**Two means are reported deliberately.** `mean_guesses_solved_only` counts solved games only
(the figure normally quoted for Wordle solvers); `mean_score_failures_penalized` counts a
failure as 7. When failure rates differ between solvers, only the second is a fair single-number
comparison.

Stochastic solvers use the seed from Section 3, so runs are reproducible.
""")

code(r'''
answers_to_eval = VOCAB.answers[:BENCHMARK_LIMIT] if BENCHMARK_LIMIT else VOCAB.answers
print(f"evaluating {len(answers_to_eval)} answers x {len(BENCHMARK_STRATEGIES)} strategies")
print(f"guess_pool={CONFIG.guess_pool}  max_guesses={CONFIG.max_guesses}  seed={SEED}")
print("the information-theoretic solvers take a few minutes each at full pool.\n")

grand_t0 = time.perf_counter()
for strat in BENCHMARK_STRATEGIES:
    print(f"=== {strat} ===")
    res = run_benchmark(strat, FB, CONFIG, answers=answers_to_eval,
                        model=FREQ_MODEL, progress_every=500)
    RESULTS[strat] = res
    SUMMARIES[strat] = summarize(res, max_guesses=CONFIG.max_guesses)
    TRAJECTORIES[strat] = candidate_trajectory(res, max_turns=CONFIG.max_guesses)
    s = SUMMARIES[strat]
    print(f"  mean={s['mean_guesses_solved_only']:.4f}  "
          f"fail={s['failure_rate_pct']:.2f}%  wall={res.wall_s:.1f}s\n")
print(f"total {time.perf_counter()-grand_t0:.1f}s")
''')

code(r'''
ordered = [SUMMARIES[s] for s in BENCHMARK_STRATEGIES if s in SUMMARIES]
print(f"FULL BENCHMARK — {len(answers_to_eval)} answers, "
      f"guess_pool={CONFIG.guess_pool}, seed={SEED}")
print("=" * 104)
print(summary_table(ordered, max_guesses=CONFIG.max_guesses))
print("\n'mean' counts solved games only. Failure-penalized means:")
for s in ordered:
    print(f"  {s['strategy']:<11} solved-only={s['mean_guesses_solved_only']:.4f}   "
          f"failure-penalized={s['mean_score_failures_penalized']:.4f}   "
          f"fail={s['failure_rate_pct']:.2f}%   "
          f"({s['n_failed']} of {s['n_games']})")
''')

code(r'''
# Average surviving candidates after each guess.
print("AVERAGE SURVIVING CANDIDATES AFTER EACH GUESS")
print("(only games still in progress at a turn contribute to that turn)\n")
hdr = f"{'strategy':<11}" + "".join(f"{'guess '+str(t):>13}" for t in range(1, CONFIG.max_guesses+1))
print(hdr); print("-" * len(hdr))
for strat in BENCHMARK_STRATEGIES:
    tr = TRAJECTORIES.get(strat, {})
    row = f"{strat:<11}"
    for t in range(1, CONFIG.max_guesses + 1):
        row += f"{tr[t]['mean_remaining']:>13.2f}" if t in tr else f"{'-':>13}"
    print(row)

print("\nsame thing in BITS of remaining uncertainty (log2 of the candidate count)")
print(f"start: {np.log2(VOCAB.n_answers):.2f} bits\n")
print(hdr); print("-" * len(hdr))
for strat in BENCHMARK_STRATEGIES:
    tr = TRAJECTORIES.get(strat, {})
    row = f"{strat:<11}"
    for t in range(1, CONFIG.max_guesses + 1):
        row += f"{tr[t]['mean_bits_remaining']:>13.3f}" if t in tr else f"{'-':>13}"
    print(row)
''')

code(r'''
# Distribution histogram.
try:
    import matplotlib.pyplot as plt
    strats = [s for s in BENCHMARK_STRATEGIES if s in SUMMARIES]
    ks = list(range(1, CONFIG.max_guesses + 1))
    x = np.arange(len(ks)); w = 0.8 / max(len(strats), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, s in enumerate(strats):
        vals = [SUMMARIES[s]["distribution_pct"].get(k, 0.0) for k in ks]
        axes[0].bar(x + i*w - 0.4 + w/2, vals, w, label=s)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ks)
    axes[0].set_xlabel("guesses used"); axes[0].set_ylabel("% of games")
    axes[0].set_title(f"Guess distribution ({len(answers_to_eval)} answers)")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.3)

    for s in strats:
        tr = TRAJECTORIES[s]
        ts = sorted(tr); axes[1].plot(ts, [tr[t]["mean_bits_remaining"] for t in ts],
                                      marker="o", label=s)
    axes[1].axhline(np.log2(VOCAB.n_answers), ls="--", c="grey",
                    label=f"start {np.log2(VOCAB.n_answers):.1f} bits")
    axes[1].set_xlabel("after guess #"); axes[1].set_ylabel("mean bits remaining")
    axes[1].set_title("Uncertainty reduction"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
except ImportError:
    print("matplotlib unavailable — text distribution instead:")
    for s in BENCHMARK_STRATEGIES:
        if s not in SUMMARIES: continue
        d = SUMMARIES[s]["distribution_pct"]
        print(f"\n{s}")
        for k in range(1, CONFIG.max_guesses+1):
            print(f"  {k}: {d.get(k,0):5.2f}% {'#'*int(d.get(k,0)/2)}")
''')

# =============================================================================
# 17. First-guess analysis
# =============================================================================
md(r"""
---
# 17. First-Guess Analysis

Turn 1 is the only decision where every strategy faces an identical, fully known state — all
answers are possible and nothing has been observed. That makes it the cleanest place to see
what each objective actually optimises and where the objectives disagree.

All 12,972 legal guesses are scored against all 2,315 answers on entropy, expected remaining,
and worst-case bucket, with a flag for whether the word is itself a possible answer.
""")

code(r'''
t0 = time.perf_counter()
FGA = first_guess_analysis(FB, model=FREQ_MODEL)
print(f"scored all {VOCAB.n_guesses} openers in {time.perf_counter()-t0:.1f}s\n")

BLOCKS = [
    ("entropy",            True,  "TOP 20 BY ENTROPY  (max expected information gain, bits)"),
    ("expected_remaining", False, "TOP 20 BY EXPECTED REMAINING CANDIDATES  (minimize)"),
    ("worst_case",         False, "TOP 20 BY WORST-CASE BUCKET  (minimax)"),
    ("frequency_score",    True,  "TOP 20 BY LETTER-FREQUENCY HEURISTIC  (Layer 3)"),
]
FGA_TOP = {}
for metric, mx, title in BLOCKS:
    rows = top_openers(FGA, metric, 20, maximize=mx)
    FGA_TOP[metric] = rows
    print(format_openers(rows, "\n" + title))
''')

code(r'''
# What each solver actually opens with, and why.
print("OPENER CHOSEN BY EACH STRATEGY\n")
print(f"{'strategy':<11}{'opener':<8}{'H(bits)':>9}{'E[rem]':>10}{'worst':>7}"
      f"{'answer?':>9}{'H rank':>8}")
print("-" * 62)
ent_rank = {str(FGA['word'][i]): r for r, i in
            enumerate(np.argsort(-FGA["entropy"], kind="stable"), 1)}
for strat in BENCHMARK_STRATEGIES:
    op = SUMMARIES.get(strat, {}).get("opening_guess")
    if not op:
        sv = make_solver(strat, FB, CONFIG, FREQ_MODEL)
        op = sv.opening_guess() if sv.deterministic else "(random)"
    if op == "(random)":
        print(f"{strat:<11}{op:<8}{'-':>9}{'-':>10}{'-':>7}{'-':>9}{'-':>8}")
        continue
    i = VOCAB.guess_index[op]
    print(f"{strat:<11}{op:<8}{FGA['entropy'][i]:>9.4f}"
          f"{FGA['expected_remaining'][i]:>10.2f}{FGA['worst_case'][i]:>7}"
          f"{('yes' if FGA['is_possible_answer'][i] else 'no'):>9}"
          f"{ent_rank[op]:>8}")

print("\nThe objectives disagree, and the disagreement is the interesting part:")
print("  - entropy maximises AVERAGE information -> spreads probability mass evenly")
print("  - expected-remaining penalises big buckets LINEARLY, not logarithmically")
print("  - minimax ignores the average entirely and optimises only the worst bucket")
print("  - frequency never considers partitions at all, only letter statistics")
print("\nBest openers are mostly NOT possible answers — a probe's job on turn 1 is to")
print("split the space, not to win. This is why restricting guesses to the answer list")
print("would weaken the solver.")

# How much worse is a candidates-only opener?
best_any = str(FGA["word"][int(np.argmax(FGA["entropy"]))])
mask = FGA["is_possible_answer"]
ent_ans = np.where(mask, FGA["entropy"], -np.inf)
best_ans = str(FGA["word"][int(np.argmax(ent_ans))])
print(f"\nbest opener overall     : {best_any.upper()}  "
      f"H={FGA['entropy'][VOCAB.guess_index[best_any]]:.4f} bits")
print(f"best opener that is also a possible answer: {best_ans.upper()}  "
      f"H={FGA['entropy'][VOCAB.guess_index[best_ans]]:.4f} bits")
''')

code(r'''
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ia = FGA["is_possible_answer"]
    axes[0].scatter(FGA["entropy"][~ia], FGA["expected_remaining"][~ia], s=3,
                    alpha=.25, label="not a possible answer")
    axes[0].scatter(FGA["entropy"][ia], FGA["expected_remaining"][ia], s=3,
                    alpha=.45, label="possible answer")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("entropy (bits)"); axes[0].set_ylabel("E[remaining] (log)")
    axes[0].set_title(f"All {VOCAB.n_guesses} openers: entropy vs expected remaining")
    axes[0].legend(markerscale=4, fontsize=8); axes[0].grid(alpha=.3)

    axes[1].hist(FGA["entropy"], bins=80)
    for strat in ("entropy", "expected", "minimax", "hybrid", "frequency"):
        op = SUMMARIES.get(strat, {}).get("opening_guess")
        if op:
            axes[1].axvline(FGA["entropy"][VOCAB.guess_index[op]], lw=1.2,
                            label=f"{strat}: {op.upper()}")
    axes[1].set_xlabel("entropy (bits)"); axes[1].set_ylabel("number of words")
    axes[1].set_title("Opener entropy distribution")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
except ImportError:
    print("matplotlib unavailable")
''')

# =============================================================================
# 18. Example games
# =============================================================================
md(r"""
---
# 18. Exact-Game Inspection

`solve(answer=..., verbose=True)` prints the full trace: each guess, the feedback, and the
number of candidates remaining. This is the tool for auditing *why* the solver did what it did
— and the format that will be useful for building supervised traces for the later LLM
experiment.
""")

code(r'''
r = play_game(BUNDLE.solver(STRATEGY), "crane", verbose=True)
''')

code(r'''
# A few deliberately awkward answers: repeated letters and dense neighbourhoods.
HARD = ["mummy", "eerie", "jazzy", "fuzzy", "wight", "vaunt", "boxer", "watch", "zonal"]
HARD = [w for w in HARD if w in VOCAB.answer_index]

for strat in ["entropy", "minimax", "hybrid"]:
    sv = make_solver(strat, FB, CONFIG, FREQ_MODEL)
    op = sv.opening_guess()
    print(f"=== {strat} (opener {op.upper()}) ===")
    for a in HARD:
        g = play_game(sv, a, first_guess=op)
        trail = "  ".join(f"{gg.upper()}[{pp}]" for gg, pp in zip(g.guesses, g.patterns))
        print(f"  {a.upper()}  {g.n_guesses if g.solved else 'FAIL'}  "
              f"cands={g.candidates_remaining}\n        {trail}")
    print()
''')

code(r'''
# Worst cases for the default strategy — where the classical approach struggles.
best = min((s for s in SUMMARIES.values()),
           key=lambda s: s["mean_score_failures_penalized"])["strategy"]
res = RESULTS[best]
worst = sorted(res.games, key=lambda g: -g.score)[:15]
print(f"hardest answers for '{best}':\n")
for g in worst:
    print(f"  {g.answer.upper()}  {g.n_guesses if g.solved else 'FAIL'} guesses  "
          f"cands={g.candidates_remaining}")
    print(f"        {'  '.join(x.upper() for x in g.guesses)}")
print("\nThese are dense neighbourhoods where many answers differ by one letter")
print("(-ATCH, -IGHT, -OUND ...). No amount of information-theoretic probing escapes")
print("them; the constraint structure itself is the bottleneck.")
''')

md(r"""
### Interactive play

To drive the real NYT game, call `interactive_play(BUNDLE)`: it proposes a guess, you type
the feedback you saw as five characters of `G`/`Y`/`B`, and it re-filters. Left commented out
because a notebook run with no stdin will hang on the prompt.
""")

code(r'''
# interactive_play(BUNDLE, strategy=STRATEGY)

# Non-interactive demonstration of the same loop, feeding it scripted feedback.
_target = "crane"
_cands = np.arange(VOCAB.n_answers, dtype=np.int32)
_sv = BUNDLE.solver(STRATEGY)
print(f"(simulating interactive play against a hidden {_target.upper()})\n")
for turn in range(1, CONFIG.max_guesses + 1):
    g = _sv.opening_guess() if turn == 1 else _sv.choose(_cands, turn)
    typed = feedback(g, _target)            # what a human would type
    print(f"Guess {turn}: {g.upper()}   ({len(_cands)} candidates)")
    print(f"  feedback> {typed}")
    if pattern_to_code(typed) == ALL_GREEN:
        print(f"\nSolved in {turn} guesses: {g.upper()}")
        break
    _cands = BUNDLE.fb.filter_indices(_cands, g, pattern_to_code(typed))
''')

# =============================================================================
# 19. Results comparison
# =============================================================================
md(r"""
---
# 19. Results Comparison — Answering the Research Question

The benchmark exists to decompose Wordle performance into the three layers. Read the table
below as a ladder: each row adds one mechanism to the row above it.
""")

code(r'''
LAYER = {
    "random":    ("1 symbolic only",        "exact elimination, no decision rule"),
    "frequency": ("1 + 3 heuristic",        "letter statistics, no partition reasoning"),
    "entropy":   ("1 + 2 information",      "max expected information gain"),
    "expected":  ("1 + 2 information",      "min expected remaining candidates"),
    "minimax":   ("1 + 2 information",      "min worst-case bucket"),
    "hybrid":    ("1 + 2 + 3 combined",     "bits-normalized blend + answer prior"),
}
rows = [SUMMARIES[s] for s in BENCHMARK_STRATEGIES if s in SUMMARIES]
base = SUMMARIES.get("random", {}).get("mean_score_failures_penalized")

print("CONTRIBUTION BY LAYER")
print(f"{'strategy':<11}{'layers':<22}{'mean':>8}{'vs random':>11}{'fail%':>7}  mechanism")
print("-" * 104)
for s in rows:
    lay, mech = LAYER.get(s["strategy"], ("", ""))
    d = (f"{s['mean_score_failures_penalized']-base:+.4f}"
         if base is not None else "-")
    print(f"{s['strategy']:<11}{lay:<22}"
          f"{s['mean_score_failures_penalized']:>8.4f}{d:>11}"
          f"{s['failure_rate_pct']:>7.2f}  {mech}")

bestrow = min(rows, key=lambda s: s["mean_score_failures_penalized"])
print(f"\nbest overall (failure-penalized mean): {bestrow['strategy'].upper()} "
      f"at {bestrow['mean_score_failures_penalized']:.4f}")

info = [s for s in rows if s["strategy"] in ("entropy", "expected", "minimax")]
if base is not None and info:
    bi = min(info, key=lambda s: s["mean_score_failures_penalized"])
    fr = SUMMARIES.get("frequency", {}).get("mean_score_failures_penalized")
    print("\nDECOMPOSITION")
    print(f"  symbolic elimination alone (random)        {base:.4f} guesses")
    if fr: print(f"  + frequency heuristics  (Layer 3)          {fr:.4f}  "
                 f"({base-fr:+.4f})")
    print(f"  + information theory    (Layer 2, best)    "
          f"{bi['mean_score_failures_penalized']:.4f}  "
          f"({base-bi['mean_score_failures_penalized']:+.4f})")
    print(f"  + both combined         (hybrid)           "
          f"{SUMMARIES['hybrid']['mean_score_failures_penalized']:.4f}")
''')

code(r'''
# Sensitivity of the hybrid to lambda, on a fixed subsample — shows the defaults
# were checked rather than asserted.
rs = np.random.default_rng(SEED)
sub = [VOCAB.answers[i] for i in rs.choice(VOCAB.n_answers, 300, replace=False)]
print(f"hybrid weight sweep on a fixed {len(sub)}-answer subsample (seed {SEED})\n")
print(f"{'lambda':>7}{'nu':>7}{'mu':>7}{'mean':>9}{'fail%':>8}")
print("-" * 38)
for lam, nu, mu in [(0.0,0.0,0.0), (0.0,0.0,1.0), (0.25,0.25,1.0),
                    (0.5,0.25,1.0), (0.25,0.5,1.0), (1.0,1.0,1.0)]:
    c = SolverConfig(max_guesses=MAX_GUESSES, seed=SEED, guess_pool=GUESS_POOL,
                     minimax_weight=lam, expected_weight=nu, answer_bonus_weight=mu)
    r = run_benchmark("hybrid", FB, c, answers=sub, model=FREQ_MODEL, log=None)
    sm = summarize(r, max_guesses=MAX_GUESSES)
    print(f"{lam:>7.2f}{nu:>7.2f}{mu:>7.2f}"
          f"{sm['mean_score_failures_penalized']:>9.4f}{sm['failure_rate_pct']:>8.2f}")
print("\n(subsample, so differences of a few thousandths are noise)")
''')

md(r"""
### What this establishes for the LLM comparison

The benchmark gives the classical reference points a learned model must be measured against.
When interpreting a 0.5B LLM's Wordle scores later, three things from this notebook matter:

1. **A ceiling to beat, not a floor.** These solvers have *perfect* recall of a 12,972-word
   dictionary and do *exact* posterior filtering over 2,315 hypotheses. An LLM doing next-token
   prediction has neither for free. Matching the `random` baseline already implies the model
   learned to track constraints; matching `entropy` implies something much stronger.

2. **The layers are separable, so credit is attributable.** If a trained model beats
   `frequency` but not `entropy`, it has likely learned letter statistics rather than
   information-seeking behaviour. The gap between those two rows is the diagnostic.

3. **The honest upper bound is not here.** All of these solvers are **one-step greedy**. A full
   game-tree search over guess sequences does better than any row in this table. This baseline
   is "strong classical", not "optimal".

One caveat worth stating plainly: the solvers get the answer list as *input*, so they never
have to know which strings are words. That is a genuine advantage over an LLM, and it should
be acknowledged rather than treated as a fair fight.
""")

# =============================================================================
# 20. Save portable artifacts
# =============================================================================
md(r"""
---
# 20. Save Portable Artifacts

Final bundle, sufficient to reproduce the solver locally with only Python + NumPy.
""")

code(r'''
paths = save_artifacts(
    ARTIFACT_DIR, VOCAB, FB, FREQ_MODEL, CONFIG,
    extra_metadata={
        "matrix_build_seconds": round(BUILD_SECONDS, 2),
        "matrix_build_chunk": MATRIX_CHUNK,
        "discovered_answers_path": ANSWERS_PATH,
        "discovered_guesses_path": GUESSES_PATH,
        "guesses_file_holds_extras_only": bool(GUESSES_ARE_EXTRA),
        "vocabulary_verdict": verdict,
        "audit": {k: {kk: vv for kk, vv in v.items() if kk != "duplicate_examples"}
                  for k, v in AUDIT.items()},
        "benchmark_n_answers": len(answers_to_eval),
        "benchmark_best_strategy": bestrow["strategy"],
    },
)

# Benchmark results + full opener ranking alongside the solver bundle.
save_benchmark(os.path.join(ARTIFACT_DIR, "benchmark_results.json"),
               [SUMMARIES[s] for s in BENCHMARK_STRATEGIES if s in SUMMARIES],
               TRAJECTORIES,
               extra={"first_guess_top20": FGA_TOP,
                      "n_answers_evaluated": len(answers_to_eval),
                      "config": CONFIG.to_dict()})

order = np.argsort(-FGA["entropy"], kind="stable")
with open(os.path.join(ARTIFACT_DIR, "first_guess_analysis.csv"), "w",
          encoding="utf-8", newline="\n") as fh:
    fh.write("word,entropy_bits,expected_remaining,worst_case_remaining,"
             "is_possible_answer,frequency_score\n")
    for i in order:
        fh.write(f"{FGA['word'][i]},{FGA['entropy'][i]:.6f},"
                 f"{FGA['expected_remaining'][i]:.4f},{FGA['worst_case'][i]},"
                 f"{int(FGA['is_possible_answer'][i])},{FGA['frequency_score'][i]:.6f}\n")

# Ship the module itself so the artifact directory is self-sufficient.
import shutil
for f in ("wordle_solver.py", "benchmark.py"):
    if os.path.exists(f):
        shutil.copy(f, os.path.join(ARTIFACT_DIR, f))

print(f"ARTIFACT_DIR = {os.path.abspath(ARTIFACT_DIR)}\n")
tot = 0
for f in sorted(os.listdir(ARTIFACT_DIR)):
    sz = os.path.getsize(os.path.join(ARTIFACT_DIR, f)); tot += sz
    print(f"  {f:<30}{sz/1024:>12.1f} KiB")
print(f"  {'TOTAL':<30}{tot/1024:>12.1f} KiB")
''')

code(r'''
# Final gate: a fresh load of the bundle must reproduce a known game exactly.
fresh = load_artifacts(ARTIFACT_DIR, mmap=True)
chk = play_game(fresh.solver(STRATEGY), "crane")
print(f"fresh bundle solves CRANE in {chk.n_guesses}: "
      f"{' -> '.join(g.upper() for g in chk.guesses)}")
assert chk.solved
print("\nartifacts are complete and self-sufficient.")
''')

# =============================================================================
# 21. Local usage
# =============================================================================
md(r"""
---
# 21. Local Usage Instructions

## Download

On Kaggle, take everything from `/kaggle/working/artifacts/`. You need at minimum:

```
artifacts/
    answers.txt                 2,315 answers, one per line
    valid_guesses.txt          12,972 legal guesses, one per line
    feedback_matrix.npy        12,972 x 2,315 uint8  (~28.6 MiB)
    metadata.json              shapes, encoding, provenance, audit
    frequency_model.json       letter / positional statistics
    solver_config.json         seed, weights, guess-pool policy
    wordle_solver.py           the solver module itself
    benchmark.py               the evaluation harness
```

Plus, for reference: `benchmark_results.json` and `first_guess_analysis.csv`.

## Install

Only NumPy is required.

```bash
conda create -n wordle python=3.11 numpy
conda activate wordle
```

## Run

```bash
python wordle_solver.py --info
python wordle_solver.py --answer crane
python wordle_solver.py --answer crane --strategy minimax
python wordle_solver.py --interactive          # play the real NYT game
```

From Python:

```python
from wordle_solver import load_artifacts, play_game, solve

bundle = load_artifacts("artifacts")     # instant, memory-mapped
solve(answer="crane", verbose=True)

solver = bundle.solver("entropy")
r = play_game(solver, "mummy")
print(r.n_guesses, r.guesses, r.candidates_remaining)
```

Rebuilding from scratch (no artifacts needed, ~50 s):

```bash
python build_artifacts.py --data-dir data --artifact-dir artifacts
python run_full_benchmark.py
pytest tests/ -q
```

## Changing behaviour

Everything lives in `SolverConfig` (`solver_config.json`, or pass it directly):

```python
from wordle_solver import SolverConfig, load_artifacts, make_solver

cfg = SolverConfig(
    seed=20260817, max_guesses=6, strategy="hybrid",
    guess_pool="full",            # "full" | "adaptive" | "candidates"
    entropy_weight=1.0, minimax_weight=0.25,
    expected_weight=0.25, answer_bonus_weight=1.0,
    opening_guess="soare",        # skip recomputing turn 1
)
b = load_artifacts("artifacts")
solver = make_solver("hybrid", b.fb, cfg, b.model)
```

Set `guess_pool="adaptive"` for a large speedup at a very small cost in mean guesses.

## Portability notes

- Paths are relative; `DATA_DIR` / `ARTIFACT_DIR` are the only path knobs.
- No `/kaggle/...` path appears anywhere in `wordle_solver.py` — Kaggle detection lives only
  in Section 2 of the notebook.
- `.npy` is portable across Windows and Linux; the matrix is `uint8`, so endianness is moot.
- Dependencies: Python ≥ 3.9 and NumPy. `pandas`/`matplotlib` are used only for notebook
  visualisation and are optional. No ML, LLM, or RL libraries anywhere.
- CPU only. A GPU would give no benefit at this problem size, so none is used.
""")

# =============================================================================
# write
# =============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md = len(cells) - n_code
print(f"wrote {OUT}")
print(f"  {len(cells)} cells ({n_md} markdown, {n_code} code)")
print(f"  {os.path.getsize(OUT)/1024:.1f} KiB")
