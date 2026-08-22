---
name: kaggle-run
description: Run this project's notebooks on Kaggle end to end - create the notebook, attach GPU and datasets, wait for it to finish, and diagnose failures. Use whenever the user asks to run, launch, submit, re-run, or check a phase on Kaggle ("run phase 9", "push the DPO notebook", "is it done yet", "why did it fail"), or after editing a notebook generator. Built for being driven remotely from a phone over Remote Control.
---

# Running Kaggle notebooks for this project

## The rule that matters

**Drive Kaggle through the API, never the browser.** `tools/kaggle_run.py`
wraps the whole lifecycle. Reach for the browser only for the four things in
[Browser fallback](#browser-fallback) below.

This is a cost decision, not a style one. A screenshot of the Kaggle UI costs
~1-2k tokens and has to be re-taken on every state change; a 90-minute run
polled visually costs tens of thousands of tokens and misreads a busy page. The
same run through `kaggle_run.py run` costs one tool call and ~500 tokens of
output, and the status string is unambiguous.

**Never poll in a loop yourself.** `run` and `wait` block inside the Python
process with backoff. One tool call covers the entire run. Polling from the
agent side turns a 90-minute wait into dozens of billed turns.

## Setup state

Check this first if anything errors:

- CLI: `kaggle` installed into `.conda` (invoke as `python -m kaggle`).
- Account: `arnavyrr`. Auth is OAuth and working; `doctor` verifies both scopes.
- Profiles are filled in and verified against a live push.

### Auth: kaggle.json silently breaks everything

`kaggle auth login` (OAuth) writes `~/.kaggle/credentials.json` and grants
**both** the datasets and kernels scopes. A legacy `~/.kaggle/kaggle.json` API
key grants **only** datasets - every kernels call, even on a public notebook,
returns `Permission 'kernels.get' was denied`.

**And `kaggle.json` shadows OAuth.** While it exists the CLI reports
`auth_method: LEGACY_API_KEY` and never reads `credentials.json`, so logging in
appears to change nothing. That file is now parked at
`~/.kaggle/kaggle.json.disabled`. **If kernels calls start failing, check that
no `kaggle.json` has reappeared.** `kaggle config view` prints the method in
use - it must say `OAUTH`.

`--mine` also needs OAuth; `--user arnavyrr` works either way.

**Never ask for, read, or echo token contents.**

### Datasets on the account

| slug | what |
|---|---|
| `arnavyrr/wordle-sft-package-v2` | current input bundle (`kaggle_upload/`: sft_package, code, artifacts, feedback matrix). **Use this**, not v1. |
| `arnavyrr/kaggle-adapters-upload` | Phase 4 adapters only: `entropy`, `tree_salet`, `tree_soare` |
| `arnavyrr/wordle-sft-package` | superseded by v2 |

Always invoke with the project interpreter:

```bash
.conda/python.exe tools/kaggle_run.py <subcommand>
```

## Commands

| Goal | Command |
|---|---|
| **What did we leave running?** (start here in a new session) | `kaggle_run.py runs` |
| What can I run? | `kaggle_run.py profiles` |
| Auth + my dataset slugs | `kaggle_run.py doctor` |
| **Run it and tell me what happened** | `kaggle_run.py run phase9 --timeout 150` |
| Fire and forget | `kaggle_run.py push phase9` |
| Is it done? | `kaggle_run.py status phase9` |
| Wait on an already-pushed run | `kaggle_run.py wait phase9` |
| Why did it fail? | `kaggle_run.py diagnose phase9` |
| Full log | `kaggle_run.py logs phase9 --full` |
| Update the input dataset | `kaggle_run.py data-push -m "phase9 eval set"` |

Exit codes: `0` success, `1` run failed or timed out, `2` misconfiguration.

## The standard workflow

When the user says "run phase N on Kaggle":

1. **Regenerate the notebook if its sources changed.** Each profile names its
   `generator`. The notebooks inline `core/` modules at generation time, so
   editing `core/constrained_decode.py` without regenerating silently runs the
   old code. When in doubt, regenerate - it takes seconds.
2. **Re-push the dataset if inputs changed** - `data-push`. Skip if only the
   notebook changed.
3. **`run <profile> --timeout <expected_minutes * 2>`.** This pushes with the
   accelerator and datasets already attached, waits, pulls the log, and matches
   it against known failure modes.
4. **Report**: status, the headline numbers from the log tail, and any matched
   failure mode with its fix. Do not paste the whole log.
5. **On failure**, apply the fix the diagnosis names, then re-run. Most of these
   notebooks resume, so a re-run is cheap.

## Long runs span sessions - do not block on them

A 90-minute sweep outlives the conversation that started it, and the user often
picks it up the next day. **Detach by default.**

**Every push is journalled** to `.kaggle_runs/journal.jsonl`. So the first thing
to run in a new session, or whenever the user asks "what's going on":

```bash
.conda/python.exe tools/kaggle_run.py runs
```

That prints every profile's last push with its **live** status and flags what is
still going. Nothing needs to be remembered across sessions - read it off disk.

Pick the pattern by expected length:

| run | approach |
|---|---|
| under ~20 min | `run <profile>` - block, report the result |
| longer | `push <profile>`, tell the user roughly when to check back, end the turn. Next session: `runs`, then `diagnose`. |

**Stage anything long.** Do not push a 90-minute sweep as the first test of a
change - a typo costs 90 minutes and one wrong number costs a re-run. Go:

1. **Smoke** - cut the knobs right down (Phase 9: `N_GAMES=20`,
   `ARMS=["sft"]`, two variants). ~5 min, proves it runs at all.
2. **Gate** - confirm the profile's `gate` passes. For phase9 that is `SALET`
   and ~3.7642. A wrong opener means the wrong adapter; stop there.
3. **Full** - only then the real sweep.

This costs almost nothing extra because **the notebooks resume**: state is
written per `(arm, variant)`, so stage 3 skips whatever stages 1-2 already did.
Staging converts one opaque 90-minute failure into three cheap checkpoints.

## Project-specific traps

These come from real incidents in `PROJECT_README.md`. Respect them.

- **Never substitute a different adapter.** If the profile's adapter dataset is
  missing, set `ARMS = ["base"]` or stop and ask. Substituting one already
  produced a wrong published result once (Phase 8 v3, voided).
- **Check the gate before believing a number.** Phase 9 cell 7: `sft`+`baseline`
  on 40 games must open `SALET` and land near 3.7642, tolerance 0.60. A wrong
  opener means the wrong adapter is attached - stop, don't interpret the sweep.
- **`with_count` is leaky.** It is flagged `*LEAKY*` in Phase 9 output and
  excluded from the spread. Never quote it as a result.
- **The runs resume.** State lives in the results JSON and each `(arm, variant)`
  is written as it completes. Interrupting costs at most one variant. Never
  restart from scratch to "be safe".
- **`t4x2` in a profile gets one T4.** The API has no two-GPU option. Every
  notebook here uses a single GPU so this is fine, but never report that two
  were attached.
- **The `tree_salet_endgame` adapter is not on Kaggle.**
  `kaggle-adapters-upload` holds only the Phase 4 adapters. `docs/RUN_PHASE9.md`
  needs the Phase 7 `tree_salet_endgame`; the local copy is
  `uploads/phase7_adapter/tree_salet_endgame/` (Phase 8 DPO is in
  `uploads/phase8_adapter/`). Until one is uploaded, **phase7b/8/9 can only run
  `ARMS=["base"]`** - and `tree_salet` in the adapters dataset is a *different*
  adapter, so pointing at it is exactly the substitution that voided Phase 8 v3.
- **Kaggle derives the kernel slug from the `title`, not the `id`.** A profile
  whose title does not slugify to its slug pushes to a different notebook, and
  every later status/output call polls a ref that does not exist. `cmd_push`
  refuses to push on mismatch; keep `slug == slugify(title)`.
- **`kernels status` prints an enum**, `KernelWorkerStatus.ERROR`, not a bare
  word. `_status()` strips the prefix. Don't "simplify" that regex.
- **An unexplained ~21s/decision slowdown** hit the Phase 8 run and was never
  diagnosed. If a run is wildly slower than `expected_minutes`, say so rather
  than just extending the timeout.

## Driving this from a phone

Over Remote Control the user is typing one short line and cannot read a wall of
output. So:

- Take "run phase 9" as authorization for the whole sequence - regenerate,
  push, wait, diagnose - without checking in at each step.
- Prefer one long `run --timeout` call over several short ones.
- Reply in a few lines: status, the number, what to do next. Put detail in the
  files and name the path.
- Kaggle GPU quota is ~30h/week. If a push fails on quota, say so plainly and
  give the reset day - do not silently retry.
- Pushing a notebook is cheap and reversible; **pushing a dataset version is
  not** (it is public-facing and versioned). Confirm before `data-push` unless
  the user asked for it in the same message.

## Not wasting GPU hours

The weekly GPU budget is ~30h and it is the scarcest resource here. What does
and does not consume it:

- **Only the kernel burns quota, and only while it runs.** An API-pushed run
  terminates itself the moment the notebook finishes - there is no session left
  open afterwards, nothing to shut down.
- **Polling costs nothing.** `wait` runs on this machine. Waiting longer never
  costs GPU time.
- **The real waste is a hung or overrunning run**, which otherwise burns its
  full 12h before Kaggle kills it. Every push therefore sets a hard cap with
  `-t`, from `max_runtime_minutes` or `expected_minutes * 2`. Set
  `max_runtime_minutes` explicitly on any profile where that default is wrong.
- **The other waste is a browser session.** An interactive notebook opened at
  kaggle.com holds its GPU until it idles out, independent of anything here. If
  the user has been clicking around the UI, remind them to close those tabs.
- **Never set an accelerator a notebook does not use.** `phase1` is CPU-only
  (`accelerator: none`) and must stay that way.

**There is no API call that stops a running kernel.** The CLI has no `cancel` -
only `delete`, which destroys the notebook and its outputs. To stop a run
early, the user must hit Stop in the web UI. Say so plainly rather than
deleting anything.

## Browser fallback

The API genuinely cannot do these. Use `mcp__Claude_Browser__*` only here:

1. Accepting competition rules for the first time.
2. Phone-verifying an account to unlock internet/GPU.
3. Reading the GPU quota meter (no API endpoint).
4. A run stuck in `queued` for an unreasonable time, to check for a Kaggle-side
   incident.

For all four: `preview_start` with `https://www.kaggle.com`, then `read_page`
(the accessibility tree) rather than `computer{action:"screenshot"}`. The tree
is text and costs a fraction of an image. Screenshot only when you must confirm
something visual.

## Adding a profile

Append to `tools/kaggle_profiles.json`:

```json
"phase10": {
  "notebook": "phase10_grpo/wordle_phase10_kaggle.ipynb",
  "generator": "phase10_grpo/make_grpo_notebook.py",
  "slug": "wordle-phase10-grpo",
  "title": "Wordle Phase 10 - GRPO",
  "accelerator": "t4x2",
  "dataset_sources": ["<user>/wordle-sft-package", "<user>/wordle-adapters"],
  "enable_internet": true,
  "expected_minutes": 300
}
```

`slug` must be unique per notebook - pushing two profiles to one slug
overwrites the first. New failure modes go in the `FAILURE_MODES` table in
`tools/kaggle_run.py`, most specific pattern first.
