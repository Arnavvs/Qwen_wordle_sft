# =============================================================================
# SAVE EVERYTHING — paste as the LAST cell. Queue it now; it runs when the
# eval finishes.
#
# Publishes adapters + all results as a Kaggle Dataset, and writes a zip either
# way so there is always a download even if the API is refused.
#
# Tolerant by design: it saves whatever exists. A half-finished eval still
# produces a useful snapshot, and nothing here raises on a missing piece.
# =============================================================================
import os
import json
import time
import shutil
import zipfile
import subprocess

SLUG = "wordle-phase8-final"
TITLE = "Wordle Phase 8 — DPO v3 adapters + results"
WORK = "/kaggle/working"
STAGE = os.path.join(WORK, "_final_save")

print("=" * 70)
print("SAVING EVERYTHING")
print("=" * 70)

# ---- 1. flush in-memory state to disk first --------------------------------
# If the eval was interrupted, STATE may hold rows that never reached the file.
try:
    save("final-save")
    print("  state flushed to results.json")
except Exception as e:
    print(f"  (no save() in scope: {e})")

# ---- 2. stage ---------------------------------------------------------------
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)
staged = []

KEEP = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
        "training_config.json", "README.md")
for root, _, files in os.walk(WORK):
    if "adapter_config.json" not in files:
        continue
    if "checkpoint-" in root or "_final_save" in root or "_autosave" in root:
        continue
    name = os.path.basename(root.rstrip("/")) or "adapter"
    dst = os.path.join(STAGE, "adapters", name)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in KEEP:
        s = os.path.join(root, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, f))
            n += os.path.getsize(s)
    staged.append((f"adapters/{name}", n))

for d in ("results_phase8", "results_phase7", "results_sweep"):
    p = os.path.join(WORK, d)
    if os.path.isdir(p):
        shutil.copytree(p, os.path.join(STAGE, d), dirs_exist_ok=True)
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(p) for f in fs)
        staged.append((d + "/", sz))

# the training curve, even if results.json never got written
try:
    if DPO_LOG:
        json.dump(DPO_LOG, open(os.path.join(STAGE, "dpo_log.json"), "w"), indent=2)
        staged.append(("dpo_log.json", os.path.getsize(os.path.join(STAGE, "dpo_log.json"))))
except Exception:
    pass
try:
    if EV:
        json.dump(EV, open(os.path.join(STAGE, "eval_partial.json"), "w"),
                  indent=2, default=str)
        staged.append(("eval_partial.json",
                       os.path.getsize(os.path.join(STAGE, "eval_partial.json"))))
        print(f"  eval rows present: {sorted(EV)}")
except Exception:
    pass

total = sum(n for _, n in staged)
print("\n  staged:")
for name, n in staged:
    print(f"    {n/2**20:>8.2f} MiB  {name}")
print(f"    {total/2**20:>8.2f} MiB  TOTAL")
if not staged:
    print("  NOTHING FOUND — is this the right kernel?")

# ---- 3. zip (always) --------------------------------------------------------
ZIP = os.path.join(WORK, "wordle_phase8_final.zip")
if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(STAGE):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, STAGE))
print(f"\n  zip: {ZIP}  ({os.path.getsize(ZIP)/2**20:.1f} MiB)")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(ZIP, WORK)))
except Exception:
    pass

# ---- 4. publish -------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
    USER = os.environ["KAGGLE_USERNAME"]
except Exception as e:
    print("\n  no credentials — stopping after the zip")
    print(f"  ({e})")
    raise SystemExit

json.dump({"title": TITLE, "id": f"{USER}/{SLUG}",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(STAGE, "dataset-metadata.json"), "w"), indent=2)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")

print("\n  publishing ...")
msg = f"phase8 final {time.strftime('%Y-%m-%d %H:%M')}"
rc, out = run(["kaggle", "datasets", "version", "-p", STAGE, "-m", msg, "-r", "zip"])
if rc != 0 and ("404" in out or "not found" in out.lower()
                or "does not exist" in out.lower()):
    rc, out = run(["kaggle", "datasets", "create", "-p", STAGE, "-r", "zip"])

print("\n" + "=" * 70)
if rc == 0:
    print(f"PUBLISHED -> https://www.kaggle.com/datasets/{USER}/{SLUG}")
    print("=" * 70)
    print("  (can take a minute to finish processing)")
    print("\n  Next session:")
    print(f'      PREV_RUN_DIR = "/kaggle/input/{SLUG}/adapters"')
else:
    print("PUBLISH FAILED — use the zip download above")
    print("=" * 70)
    print(out.strip()[:600])
