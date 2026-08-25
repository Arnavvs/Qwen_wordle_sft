# Running the diagnostic on Kaggle

No training. Reuses the adapters you already trained. ~1 h on a T4.

---

## Step 1 — Upload the adapters as a Kaggle Dataset

You have them locally but only inside a 740 MB zip, most of which is optimizer
state you don't need. A trimmed copy is ready at:

```
C:\Users\HP\OneDrive\Desktop\wordle\kaggle_adapters_upload\
```

101 MB, containing exactly:

```
entropy/     adapter_config.json  adapter_model.safetensors  training_config.json
tree_soare/  adapter_config.json  adapter_model.safetensors  training_config.json
tree_salet/  adapter_config.json  adapter_model.safetensors  training_config.json
dataset_hashes.json  environment.json
```

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**
2. Drag in the **`kaggle_adapters_upload` folder** (the folder itself, not a zip)
3. Title it **`wordle-sft-adapters`**
4. **Create**

Skip this step if you already saved the adapters as a dataset in an earlier
session — just note its path.

## Step 2 — Open the notebook

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook**
2. Upload `wordle_qwen_sft_kaggle.ipynb`

## Step 3 — Turn on the GPU

Right sidebar → **Session options** → **Accelerator** → **GPU T4 x2**.

Without this the notebook stops at the environment cell and tells you so.

## Step 4 — Attach both datasets

Right sidebar → **Add Input**, add **both**:

- your existing SFT package dataset (`wordle-sft-package`)
- `wordle-sft-adapters` from Step 1

## Step 5 — Find the adapter mount path

Run this in a scratch cell first. Guessing the path is the one thing that
reliably wastes a session:

```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    if "adapter_config.json" in files:
        print("adapter at:", root)
        print("PREV_RUN_DIR =", os.path.dirname(root))
```

It prints the three adapter folders and the directory to use.

**Confirmed for this account** — Kaggle preserved the uploaded folder name, so
the path is:

```
/kaggle/input/datasets/arnavyrr/wordle-sft-adapters/kaggle_adapters_upload
```

Leave `DATASET_DIR = None`: the auto-detect requires all 14 SFT-package files
to match, so the adapters dataset cannot be picked by mistake, and it searches
deep enough for the nested `datasets/<user>/<slug>/` layout.

## Step 6 — Edit the CONFIG cell (section 2)

Set exactly these. Everything else stays as it is:

Only **five** lines actually change from the defaults:

```python
DATASET_DIR    = None          # unchanged - auto-detects the SFT package

PREV_RUN_DIR   = "/kaggle/input/datasets/arnavyrr/wordle-sft-adapters/kaggle_adapters_upload"

RUN_ENTROPY    = False         # was True - nothing trains
RUN_TREE_SOARE = False         # was True
RUN_TREE_SALET = False         # was True
RUN_SMOKE_TEST = False         # was True - nothing to smoke-test

RUN_EVALUATION = True          # unchanged
RUN_BASELINES  = True          # unchanged
```

Further down the same cell these are **already correct** — do not touch them:

```python
EVAL_MODE          = "both"    # unconstrained AND constrained
RUN_BASE_CONTROL   = True      # untrained Qwen control
RUN_TERMINAL_PROBE = True      # the k = 1,2,3 diagnostic
```

The config cell should then print:

```
NO TRAINING THIS SESSION - adapters must come from PREV_RUN_DIR
  PREV_RUN_DIR = /kaggle/input/datasets/arnavyrr/wordle-sft-adapters/kaggle_adapters_upload
```

## Step 7 — Run all

**Run → Run All.** Roughly:

| Stage | Time |
|---|---|
| setup, model download | ~3 min |
| unconstrained games, 4 models | ~4 min |
| constrained games, 4 models | ~15 min |
| classical baselines | ~10 min |
| terminal probe (294 states x 4 models) | ~25 min |

## Step 8 — Check these three lines before trusting anything

Section 8b prints them the first time constrained mode runs. If any assertion
fails the notebook stops, which is the intended behaviour:

```
cache-reuse vs naive scoring: max |delta| = 1.2e-05  OK
pruned argmax == full argmax (XXXXX), N chunks vs 26 unpruned  OK
```

Locally these came out at `2.4e-05` and identical argmax. Anything above `2e-02`
aborts the run rather than reporting bad numbers.

Also confirm section 7 says, for each of the three:

```
<name>: RUN flag off, loading previous adapter from /kaggle/input/...
```

If it says `SKIPPED` instead, `PREV_RUN_DIR` is wrong — go back to Step 5.

## Step 9 — Download the results

At the end, section 13b prints the comparison table and writes:

```
/kaggle/working/wordle_diagnostic_results.zip
```

Right sidebar → **Output** → download it. It is a few MB.

`wordle_sft_results.zip` is not touched and not regenerated — the original
evaluation stays exactly as it was.

## Step 10 — Read the verdict

Section 13c prints **A / B / C / D** with the evidence it used. Send me the zip
and I will read it against the evidence rather than the threshold.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `SKIPPED (RUN flag off and no existing adapter found)` | wrong `PREV_RUN_DIR` | rerun Step 5 |
| `no adapter for <name> in ...` | same | rerun Step 5 |
| dataset auto-detect fails | two candidate inputs | set `DATASET_DIR` explicitly to the SFT package path |
| `constrained scorer disagrees with naive scoring` | transformers changed its KV-cache layout | tell me the transformers version it prints in section 1 |
| CUDA OOM in constrained mode | batch too large for the card | set `CONSTRAINED_CHUNK = 256` |
| session times out | 9 h limit | set `EVAL_MODE = "constrained"` and `RUN_TERMINAL_PROBE = False`, run those separately |

## Want it shorter?

For a first look at only the headline question, set `RUN_TERMINAL_PROBE = False`
and `EVAL_MODE = "constrained"`. That is ~20 min and still gives the constrained
rows plus the base-Qwen control. The terminal probe is the more informative
half, though — it is what separates verdict A from B.
