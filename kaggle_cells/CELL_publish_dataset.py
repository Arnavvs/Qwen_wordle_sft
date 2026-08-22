# =============================================================================
# OPTION B - publish the adapter as a Kaggle Dataset from code.
#
# Use when the notebook UI gives you no "New Dataset" button. This calls the
# Kaggle API directly, so it works from a live interactive session.
#
# SETUP (once):
#   1. kaggle.com -> your avatar -> Settings -> API -> Create New Token
#      -> downloads kaggle.json containing {"username": ..., "key": ...}
#   2. In this notebook: Add-ons -> Secrets -> add two secrets
#         KAGGLE_USERNAME = <the username value>
#         KAGGLE_KEY      = <the key value>
#      and attach both to the notebook.
#
# Re-running this cell with the same SLUG pushes a NEW VERSION of the dataset
# rather than failing, so it is safe to run repeatedly.
# =============================================================================
import os
import json
import glob
import shutil
import subprocess

SLUG = "wordle-phase7-adapter"      # dataset name; letters/digits/dashes only
TITLE = "Wordle Phase 7 adapter (tree_salet_endgame LoRA)"

WORK = "/kaggle/working"

# --- credentials from Kaggle Secrets ----------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = sec.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = sec.get_secret("KAGGLE_KEY")
    USER = os.environ["KAGGLE_USERNAME"]
    print(f"credentials loaded for {USER}")
except Exception as e:
    raise SystemExit(
        "Could not read KAGGLE_USERNAME / KAGGLE_KEY from Secrets.\n"
        "Add-ons -> Secrets -> add both, attach them to this notebook, re-run.\n"
        f"underlying error: {e}")

# --- stage only the files needed to reload the adapter ----------------------
found = [r for r, _, f in os.walk(WORK)
         if "adapter_config.json" in f and "checkpoint-" not in r]
assert found, "no adapter found under /kaggle/working"
SRC = found[0]
NAME = os.path.basename(SRC.rstrip("/")) or "adapter"
print(f"adapter: {SRC}")

STAGE = os.path.join(WORK, "_ds_upload")
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(os.path.join(STAGE, NAME), exist_ok=True)
for f in ("adapter_config.json", "adapter_model.safetensors",
          "adapter_model.bin", "training_config.json"):
    s = os.path.join(SRC, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(STAGE, NAME, f))
        print(f"  + {f}  ({os.path.getsize(s)/2**20:.2f} MiB)")

# results too, if present - they are tiny and worth keeping together
for extra in ("results_phase7", "results_endgame"):
    p = os.path.join(WORK, extra)
    if os.path.isdir(p):
        shutil.copytree(p, os.path.join(STAGE, extra), dirs_exist_ok=True)
        print(f"  + {extra}/")

json.dump({"title": TITLE, "id": f"{USER}/{SLUG}",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(STAGE, "dataset-metadata.json"), "w"), indent=2)

# --- create, or push a new version if it already exists ----------------------
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip())
    return p.returncode, (p.stdout + p.stderr)

print("\ncreating dataset ...")
rc, out = run(["kaggle", "datasets", "create", "-p", STAGE, "-r", "zip"])
if rc != 0 and ("already exists" in out or "409" in out):
    print("\nalready exists -> pushing a new version instead")
    rc, out = run(["kaggle", "datasets", "version", "-p", STAGE,
                   "-m", "updated adapter", "-r", "zip"])

if rc == 0:
    print(f"\nDONE -> https://www.kaggle.com/datasets/{USER}/{SLUG}")
    print("\nNext session:")
    print("  Add Input -> that dataset, then set")
    print(f'  PREV_RUN_DIR = "/kaggle/input/{SLUG}"')
    print("  and section 4 will skip training.")
else:
    print("\nFAILED - fall back to the zip download in CELL_save_adapter.py")
