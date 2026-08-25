# =============================================================================
# AUTOSAVE DAEMON - run this BEFORE the long evaluation cell.
#
# Pushes a snapshot to a Kaggle Dataset immediately, then every 20 minutes from
# a background thread, for as long as the kernel lives. If the session dies
# unattended, the most recent snapshot is already published.
#
# Needs KAGGLE_USERNAME and KAGGLE_KEY in Add-ons -> Secrets, both ATTACHED.
#
# Snapshots contain: results_phase8/ (tiny) + every adapter's config+weights.
# Optimizer state and checkpoints are excluded - they are only needed to resume
# training and are what make the folder too big to move.
# =============================================================================
import os
import json
import time
import shutil
import threading
import subprocess

SLUG = "wordle-phase8-autosave"
TITLE = "Wordle Phase 8 autosave (adapters + results)"
AUTOSAVE_EVERY = 20 * 60          # seconds
WORK = "/kaggle/working"
STAGE = os.path.join(WORK, "_autosave")

# ---- credentials ------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
    USER = os.environ["KAGGLE_USERNAME"]
    print(f"credentials OK for {USER}")
except Exception as e:
    raise SystemExit(
        "No KAGGLE_USERNAME / KAGGLE_KEY in Secrets.\n"
        "Add-ons -> Secrets -> add BOTH and ATTACH them, then re-run.\n"
        f"underlying error: {e}")


def snapshot():
    """Stage results + adapter weights. Returns staged byte count."""
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(STAGE, exist_ok=True)
    total = 0

    for d in ("results_phase8", "results_phase7"):
        p = os.path.join(WORK, d)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(STAGE, d), dirs_exist_ok=True)

    keep = ("adapter_config.json", "adapter_model.safetensors",
            "adapter_model.bin", "training_config.json")
    for root, _, files in os.walk(WORK):
        if "adapter_config.json" not in files:
            continue
        if "checkpoint-" in root or "_autosave" in root:
            continue
        dst = os.path.join(STAGE, "adapters", os.path.basename(root.rstrip("/")))
        os.makedirs(dst, exist_ok=True)
        for f in keep:
            src = os.path.join(root, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst, f))
                total += os.path.getsize(src)

    json.dump({"title": TITLE, "id": f"{USER}/{SLUG}",
               "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(STAGE, "dataset-metadata.json"), "w"), indent=2)
    return total


def push(msg):
    """New version if the dataset exists, create it if not."""
    n = snapshot()
    r = subprocess.run(["kaggle", "datasets", "version", "-p", STAGE,
                        "-m", msg, "-r", "zip"], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 and ("404" in out or "not found" in out.lower()
                              or "does not exist" in out.lower()):
        r = subprocess.run(["kaggle", "datasets", "create", "-p", STAGE, "-r", "zip"],
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
    ok = r.returncode == 0
    print(f"[autosave {time.strftime('%H:%M:%S')}] {msg}  "
          f"{n/2**20:.1f} MiB  {'OK' if ok else 'FAILED'}", flush=True)
    if not ok:
        print("   " + out.strip()[:400], flush=True)
    return ok


# ---- first push, synchronously so you see it work before sleeping ----------
print("pushing initial snapshot ...")
first = push("autosave: initial")
print(f"\nhttps://www.kaggle.com/datasets/{USER}/{SLUG}")
print("(a new dataset can take a minute to appear)\n")

# ---- background loop -------------------------------------------------------
_stop = threading.Event()


def _loop():
    n = 0
    while not _stop.wait(AUTOSAVE_EVERY):
        n += 1
        try:
            push(f"autosave #{n}")
        except Exception as e:                      # never kill the thread
            print(f"[autosave] error: {e}", flush=True)


_t = threading.Thread(target=_loop, daemon=True, name="autosave")
_t.start()

print("=" * 66)
print(f"AUTOSAVE RUNNING - every {AUTOSAVE_EVERY//60} min while the kernel lives")
print("=" * 66)
print("  Now run the evaluation cell. Snapshots continue in the background.")
print("  To stop early:  _stop.set()")
print("  To force one:   push('manual')")
print()
print("  It cannot survive the kernel being killed outright, so the initial")
print("  push above is the guarantee - everything already on disk is saved.")
