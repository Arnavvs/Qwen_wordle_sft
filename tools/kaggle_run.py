"""
kaggle_run.py - drive a Kaggle notebook run end to end from the command line.

The web UI flow in docs/RUN_PHASE*.md ("New notebook -> GPU T4 x2 -> Add Data ->
upload the .ipynb -> Run all") as an API call. One command pushes the notebook
with its accelerator and datasets already attached, waits for it to finish,
pulls the log, and prints a compact report.

    python tools/kaggle_run.py run phase9 --wait

Polling happens inside this process on purpose. An agent driving this over
Remote Control pays for one tool call, not one per poll.

Subcommands
    profiles            list configured profiles
    doctor              check auth, print username, list your datasets
    push <profile>      push the notebook (does not wait)
    status <profile>    one-line status of the latest run
    wait <profile>      block until the run leaves the running state
    logs <profile>      download output, print errors and the log tail
    diagnose <profile>  logs + match against known failure modes
    run <profile>       push + wait + diagnose        <- the one to use remotely
    data-push           push a new version of the input dataset

Every subcommand exits 0 on success, 1 on failure, 2 on misconfiguration, so a
caller can branch without parsing prose.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(ROOT, "tools", "kaggle_profiles.json")
WORKDIR = os.path.join(ROOT, ".kaggle_runs")

# Accelerator ids the Kaggle API accepts. The web UI's "GPU T4 x2" has no API
# equivalent - pushing NvidiaTeslaT4 gets a single T4. Every notebook in this
# project uses one GPU, so that costs nothing here, but do not assume two.
ACCELERATORS = {
    "none": None,
    "t4": "NvidiaTeslaT4",
    "t4x2": "NvidiaTeslaT4",   # single T4; see note above
    "p100": "NvidiaTeslaP100",
    "a100": "NvidiaTeslaA100",
    "l4": "NvidiaL4",
    "h100": "NvidiaTeslaH100",
}

# Ordered: first match wins, so specific patterns sit above generic ones.
FAILURE_MODES = [
    (r"CUDA out of memory|OutOfMemoryError",
     "GPU OOM",
     "Lower batch size or N_GAMES in cell 1. On T4 (16GB) the SFT notebooks "
     "need per_device_train_batch_size<=4 with grad accumulation."),
    (r"No space left on device|Disk quota exceeded",
     "Kaggle disk full",
     "Working dir is capped at ~20GB. Stop writing a checkpoint every epoch, "
     "or set save_total_limit=1."),
    (r"exceeded.{0,40}(time limit|12 hours|maximum execution)",
     "Session timeout",
     "12h wall clock, to a 30h/week GPU budget. These notebooks resume: re-run "
     "and completed work is skipped (see the resume note in docs/RUN_PHASE9.md)."),
    (r"FileNotFoundError.{0,200}/kaggle/input",
     "Dataset not attached, or wrong slug",
     "The notebook searched /kaggle/input and found nothing. Check "
     "dataset_sources in the profile, then set DATASET_DIR explicitly in cell 1."),
    (r"(adapter|peft).{0,80}(not found|No such file|does not exist)",
     "Adapter missing",
     "The SFT adapter dataset is not attached. Either attach it or set "
     "ARMS=['base']. Do NOT substitute a different adapter - that already cost "
     "this project one wrong result (PROJECT_README, Phase 8 v3)."),
    (r"ModuleNotFoundError: No module named '(\w+)'",
     "Missing package",
     "Add a !pip install cell at the top, or attach the package as a dataset "
     "if the notebook runs without internet."),
    (r"Your Kernel cannot use.{0,60}internet|requires internet access",
     "Internet disabled",
     "Set enable_internet=true in the profile. Needs a phone-verified account."),
    (r"URLError|urlopen error|Network is unreachable|"
     r"Temporary failure in name resolution|Failed to establish a new connection",
     "Network call failed - internet is off",
     "The notebook tried to download something with the kernel offline. Set "
     "enable_internet=true in the profile, or attach the file as a dataset. "
     "Note phase1 fetches the canonical word lists at run time."),
    (r"Traceback \(most recent call last\)",
     "Python exception",
     "See the traceback above."),
]


# Trailer every Kaggle run emits after the notebook's own output ends.
NOISE = re.compile(
    r"^\s*(\[NbConvertApp\]|\[exited with code|text = re\.sub|"
    r"Support files will be in|Making directory|Writing \d+ bytes|"
    r"/usr/(local/)?lib/|cells\[i\]\[c\] =)"
    r"|SyntaxWarning: invalid escape")


def die(msg: str, code: int = 2):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def kaggle(*args: str, check: bool = True):
    """Run the kaggle CLI via this interpreter. Returns (rc, combined output)."""
    cmd = [sys.executable, "-m", "kaggle", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        if "Authentication required" in out or "401" in out:
            die("Kaggle API not authenticated. "
                "Run: python tools/kaggle_run.py doctor")
        die("kaggle " + " ".join(args) + " failed:\n" + out.strip(), 1)
    return p.returncode, out


def load_profiles() -> dict:
    if not os.path.exists(PROFILES):
        die("no profile file at " + PROFILES)
    with open(PROFILES, encoding="utf-8") as f:
        return json.load(f)


def get_profile(name: str):
    cfg = load_profiles()
    profs = cfg.get("profiles", {})
    if name not in profs:
        die("unknown profile " + repr(name) + ". Known: "
            + ", ".join(sorted(profs)))
    user = str(cfg.get("username", "")).strip()
    if not user or user.startswith("<"):
        die("username not set in tools/kaggle_profiles.json. "
            "Run: python tools/kaggle_run.py doctor")
    return cfg, dict(profs[name])


def slugify(title: str) -> str:
    """Reproduce Kaggle's title -> slug rule.

    Kaggle derives the kernel slug from the *title*, not from the id in the
    metadata. If they disagree it pushes to the title-derived slug and every
    later status/output call polls a ref that does not exist. cmd_push refuses
    to push when they disagree rather than leaving an orphan.
    """
    s = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return re.sub(r"-+", "-", s).strip("-")


def kernel_ref(cfg: dict, prof: dict) -> str:
    return cfg["username"] + "/" + prof["slug"]


def stage_dir(name: str) -> str:
    d = os.path.join(WORKDIR, name)
    os.makedirs(d, exist_ok=True)
    return d


JOURNAL = os.path.join(WORKDIR, "journal.jsonl")


def journal_append(entry: dict):
    """Record a push so a later session can find it.

    Long runs outlive the session that started them - the point is that
    tomorrow's session can answer "what did we leave running?" from disk
    instead of from memory.
    """
    os.makedirs(WORKDIR, exist_ok=True)
    entry["pushed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def journal_read() -> list:
    if not os.path.exists(JOURNAL):
        return []
    out = []
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# --------------------------------------------------------------------------- #

def cmd_profiles(_args):
    cfg = load_profiles()
    print("username: " + str(cfg.get("username", "<unset>")) + "\n")
    print("{:<12} {:<6} {:<52} {}".format("profile", "accel", "notebook",
                                          "datasets"))
    print("-" * 100)
    for name, p in sorted(cfg.get("profiles", {}).items()):
        ds = ",".join(p.get("dataset_sources", [])) or "-"
        print("{:<12} {:<6} {:<52} {}".format(
            name, p.get("accelerator", "none"), p.get("notebook", "?"), ds))
    return 0


def cmd_doctor(_args):
    """Check both auth scopes separately - they are not the same credential.

    kaggle.json (legacy API key) is enough for datasets.*, but Kaggle no longer
    grants it kernels.* - every kernels call returns "Permission 'kernels.get'
    was denied". Pushing and polling notebooks needs an OAuth token from
    `kaggle auth login`, cached at ~/.kaggle/access_token.
    """
    cfg = load_profiles()
    ok = True

    print("== credentials ==")
    rc, out = kaggle("config", "view", check=False)
    if rc != 0 or "Authentication required" in out:
        print("NO CREDENTIALS FOUND")
        print("  Run: kaggle auth login   (browser OAuth, grants both scopes)")
        return 2
    print(out.strip())

    print("\n== datasets scope ==")
    user = str(cfg.get("username", ""))
    rc, out = kaggle("datasets", "list", "--user", user, check=False) \
        if user and not user.startswith("<") else (1, "")
    if rc == 0 and "ref" in out:
        print("OK - your datasets (use these in dataset_sources):")
        print(out.strip())
    else:
        print("could not list. Note: --mine needs OAuth; --user works with a "
              "legacy key.")
        ok = False

    print("\n== kernels scope (needed to push/poll notebooks) ==")
    rc, out = kaggle("kernels", "list", "--page-size", "1", check=False)
    if rc == 0 and "Authentication required" not in out:
        print("OK")
    else:
        ok = False
        print("DENIED - a legacy kaggle.json key cannot call kernels.*")
        print("  Fix, in an interactive terminal:")
        print("      .conda\\python.exe -m kaggle auth login")
        print("  That opens a browser, then caches a token at")
        print("      " + os.path.join(os.path.expanduser("~"), ".kaggle",
                                      "access_token"))
        print("  Keep kaggle.json where it is; the two are used side by side.")

    if str(cfg.get("username", "")).startswith("<"):
        print("\nNOTE: set \"username\" in tools/kaggle_profiles.json.")
        ok = False
    return 0 if ok else 2


def cmd_push(args):
    cfg, prof = get_profile(args.profile)
    nb = os.path.join(ROOT, prof["notebook"])
    if not os.path.exists(nb):
        gen = prof.get("generator")
        hint = (" Regenerate it with: python " + gen) if gen else ""
        die("notebook not found: " + nb + "." + hint)

    d = stage_dir(args.profile)
    # code_file resolves relative to kernel-metadata.json, so stage a copy.
    shutil.copy2(nb, os.path.join(d, os.path.basename(nb)))

    accel = prof.get("accelerator", "none")
    if accel not in ACCELERATORS:
        die("unknown accelerator " + repr(accel) + ". Known: "
            + ", ".join(ACCELERATORS))

    title = prof.get("title", prof["slug"])
    expected = slugify(title)
    if expected != prof["slug"]:
        die("title/slug mismatch for profile " + repr(args.profile) + ".\n"
            "  title    " + repr(title) + "\n"
            "  slugifies to  " + expected + "\n"
            "  slug     " + prof["slug"] + "\n"
            "Kaggle would push to the title-derived slug and status/output "
            "would then poll a ref that does not exist. Set the profile's "
            "slug to " + expected + ", or change the title so it slugifies "
            "to " + prof["slug"] + ".")

    meta = {
        "id": kernel_ref(cfg, prof),
        "title": prof.get("title", prof["slug"]),
        "code_file": os.path.basename(nb),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": prof.get("is_private", True),
        "enable_gpu": ACCELERATORS[accel] is not None,
        "enable_internet": prof.get("enable_internet", False),
        "dataset_sources": prof.get("dataset_sources", []),
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    push = ["kernels", "push", "-p", d]
    if ACCELERATORS[accel]:
        push += ["--accelerator", ACCELERATORS[accel]]

    # Hard cap on the kernel's own runtime. Without this a hung run burns its
    # full 12h against the weekly GPU budget before Kaggle kills it. Default to
    # 2x the expected time so a merely-slow run still finishes.
    cap = prof.get("max_runtime_minutes")
    if cap is None:
        cap = prof.get("expected_minutes", 60) * 2
    if cap:
        push += ["-t", str(int(cap * 60))]

    if getattr(args, "dry_run", False):
        print("DRY RUN - not calling the API\n")
        print("staged:   " + os.path.relpath(d, ROOT))
        print("notebook: {:.1f} MB".format(os.path.getsize(nb) / 1e6))
        print("runtime cap: {} min".format(cap))
        print("command:  kaggle " + " ".join(push) + "\n")
        print(json.dumps(meta, indent=2))
        return 0

    _, out = kaggle(*push)
    print(out.strip())
    journal_append({
        "profile": args.profile,
        "ref": kernel_ref(cfg, prof),
        "accelerator": ACCELERATORS[accel] or "none",
        "cap_minutes": cap,
        "datasets": prof.get("dataset_sources", []),
    })
    print("\nref:      " + kernel_ref(cfg, prof))
    print("url:      https://www.kaggle.com/code/" + kernel_ref(cfg, prof))
    print("gpu:      " + (ACCELERATORS[accel] or "none"))
    print("hard cap: {} min (Kaggle stops the kernel at this point)".format(cap))
    return 0


def _status(cfg, prof):
    """Returns (status, raw). The CLI prints an enum repr, not a bare word:

        arnavyrr/wordle-phase-1-... has status "KernelWorkerStatus.ERROR"

    so take the segment after the dot and normalise it.
    """
    _, out = kaggle("kernels", "status", kernel_ref(cfg, prof), check=False)
    m = re.search(r'status\s+"([^"]+)"', out)
    if not m:
        return "unknown", out.strip()
    return m.group(1).rsplit(".", 1)[-1].lower().replace("_", ""), out.strip()


def cmd_status(args):
    cfg, prof = get_profile(args.profile)
    st, raw = _status(cfg, prof)
    print(args.profile + ": " + st)
    if st == "unknown":
        print(raw)
    return 0


TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def cmd_wait(args):
    cfg, prof = get_profile(args.profile)
    deadline = time.time() + args.timeout * 60
    delay, last, t0 = 20, None, time.time()
    while time.time() < deadline:
        st, _ = _status(cfg, prof)
        if st != last:
            print("[{:>5}s] {}".format(int(time.time() - t0), st), flush=True)
            last = st
        if st in TERMINAL:
            return 0 if st == "complete" else 1
        time.sleep(delay)
        delay = min(delay * 1.5, 120)   # back off; these runs take 60-90 min
    print("TIMEOUT after {} min - the run is still going.".format(args.timeout))
    print("Check later: python tools/kaggle_run.py status " + args.profile)
    print("url: https://www.kaggle.com/code/" + kernel_ref(cfg, prof))
    return 1


def _fetch_log(cfg, prof, profile_name) -> str:
    d = stage_dir(profile_name + "_out")
    kaggle("kernels", "output", kernel_ref(cfg, prof), "-p", d, "-o", "-q",
           check=False)
    logs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".log")]
    if not logs:
        return ""
    with open(max(logs, key=os.path.getmtime), encoding="utf-8",
              errors="replace") as f:
        raw = f.read()
    try:  # Kaggle logs are a JSON array of {stream, time, data}
        return "\n".join(e.get("data", "") for e in json.loads(raw))
    except json.JSONDecodeError:
        return raw


def cmd_logs(args):
    cfg, prof = get_profile(args.profile)
    text = _fetch_log(cfg, prof, args.profile)
    if not text:
        print("no log available yet")
        return 1
    lines = text.splitlines()
    out_rel = os.path.relpath(stage_dir(args.profile + "_out"), ROOT)
    print("== log: {} lines, output in {} ==\n".format(len(lines), out_rel))
    print(text if args.full else "\n".join(lines[-args.tail:]))
    return 0


def cmd_diagnose(args):
    cfg, prof = get_profile(args.profile)
    st, _ = _status(cfg, prof)
    text = _fetch_log(cfg, prof, args.profile)
    lines = text.splitlines()

    print("status:   " + st)
    print("log:      {} lines".format(len(lines)))
    print("url:      https://www.kaggle.com/code/" + kernel_ref(cfg, prof))
    print("output:   " + os.path.relpath(stage_dir(args.profile + "_out"), ROOT))
    print()

    if not text:
        print("No log. If the run never started, the push probably failed.")
        return 1

    hits = [(lbl, fix) for pat, lbl, fix in FAILURE_MODES
            if re.search(pat, text, re.IGNORECASE)]

    tb = [i for i, l in enumerate(lines)
          if "Traceback (most recent call last)" in l]
    if tb:
        print("== traceback ==")
        print("\n".join(lines[tb[-1]:tb[-1] + 40]))
        print()

    if hits:
        print("== matched failure modes ==")
        for lbl, fix in hits:
            print("  [" + lbl + "] " + fix + "\n")
    elif st == "error":
        print("== no known pattern matched; log tail ==")
        print("\n".join(lines[-40:]))
    else:
        print("No failure pattern found.")

    # Every Kaggle run ends with ~20 lines of nbconvert trailer. On a failure
    # the tail is the error and worth showing raw; on success it is noise
    # hiding the actual result, so drop it.
    tail = [l for l in lines if not NOISE.match(l)] if st == "complete" else lines
    print("== log tail ==")
    print("\n".join(l for l in tail[-15:] if l.strip()))
    return 0 if st == "complete" else 1


def cmd_run(args):
    print("=== push " + args.profile + " ===")
    cmd_push(args)
    time.sleep(10)   # let the run get queued before status means anything
    print("\n=== wait (timeout {} min) ===".format(args.timeout))
    rc = cmd_wait(args)
    print("\n=== diagnose ===")
    drc = cmd_diagnose(args)
    return rc or drc


def cmd_runs(args):
    """What did we leave running? The first thing to call in a new session."""
    cfg = load_profiles()
    entries = journal_read()
    if not entries:
        print("no runs recorded yet")
        return 0

    # Latest push per profile - earlier ones are superseded.
    latest = {}
    for e in entries:
        latest[e["profile"]] = e

    print("{:<9} {:<20} {:<12} {:<7} {}".format(
        "profile", "pushed", "status", "gpu", "ref"))
    print("-" * 92)
    unfinished = []
    for name, e in sorted(latest.items()):
        prof = cfg.get("profiles", {}).get(name)
        st = "?"
        if prof:
            st, _ = _status(cfg, prof)
            if st not in TERMINAL:
                unfinished.append((name, e))
        gpu = "yes" if e.get("accelerator", "none") != "none" else "no"
        print("{:<9} {:<20} {:<12} {:<7} {}".format(
            name, e.get("pushed_at", "?")[:16], st, gpu, e.get("ref", "?")))

    if unfinished:
        print("\nstill going:")
        for name, e in unfinished:
            print("  {} - capped at {} min, pushed {}".format(
                name, e.get("cap_minutes", "?"), e.get("pushed_at", "?")))
            print("      resume with: kaggle_run.py wait {}".format(name))
    else:
        print("\nnothing running.")
    return 0


def cmd_data_push(args):
    cfg = load_profiles()
    d = os.path.join(ROOT, cfg.get("dataset_dir", "uploads/kaggle_upload"))
    if not os.path.isdir(d):
        die("dataset dir not found: " + d
            + ". Build it with: python tools/prepare_kaggle_dataset.py")
    meta = os.path.join(d, "dataset-metadata.json")
    if not os.path.exists(meta):
        slug = cfg.get("dataset_slug", "")
        if not slug or slug.startswith("<"):
            die("set \"dataset_slug\" in tools/kaggle_profiles.json first")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"title": slug.split("/")[-1], "id": slug,
                       "licenses": [{"name": "CC0-1.0"}]}, f, indent=2)
        print("created " + meta)
        _, out = kaggle("datasets", "create", "-p", d, "-r", "zip")
    else:
        _, out = kaggle("datasets", "version", "-p", d, "-m", args.message,
                        "-r", "zip")
    print(out.strip())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("profiles").set_defaults(fn=cmd_profiles)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("runs").set_defaults(fn=cmd_runs)

    p = sub.add_parser("push")
    p.add_argument("profile")
    p.add_argument("--dry-run", action="store_true",
                   help="stage and print the metadata without calling the API")
    p.set_defaults(fn=cmd_push)

    for name, fn in (("status", cmd_status), ("diagnose", cmd_diagnose)):
        p = sub.add_parser(name)
        p.add_argument("profile")
        p.set_defaults(fn=fn)

    for name, fn in (("wait", cmd_wait), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("profile")
        p.add_argument("--timeout", type=int, default=150,
                       help="minutes to wait (default 150)")
        p.set_defaults(fn=fn)

    p = sub.add_parser("logs")
    p.add_argument("profile")
    p.add_argument("--tail", type=int, default=60)
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("data-push")
    p.add_argument("-m", "--message", default="update")
    p.set_defaults(fn=cmd_data_push)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
