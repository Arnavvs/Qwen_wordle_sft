# =============================================================================
# Publish the Phase 8 adapters as a Kaggle Dataset, from code.
#
# Use when the notebook UI gives you no "New Dataset" button on Output.
#
# SETUP (once):
#   kaggle.com -> avatar -> Settings -> API -> Create New Token
#     -> downloads kaggle.json = {"username": ..., "key": ...}
#   In this notebook: Add-ons -> Secrets -> add and ATTACH both
#       KAGGLE_USERNAME   KAGGLE_KEY
#
# Safe to re-run: if the dataset exists it pushes a new VERSION instead of
# failing. Also writes a zip either way, so you have a download even if the
# API call is refused.
# =============================================================================
import os
import json
import shutil
import zipfile
import subprocess

SLUG = "wordle-phase8-dpo"
TITLE = "Wordle Phase 8 - DPO + SFT adapters (Qwen2.5-0.5B)"
WORK = "/kaggle/working"

# ---- find every adapter (DPO, and the SFT base if it was trained here) ------
adapters = []
for root, dirs, files in os.walk(WORK):
    if "adapter_config.json" in files and "checkpoint-" not in root:
        adapters.append(root)
assert adapters, f"no adapter_config.json under {WORK}"

print("=" * 70)
print("ADAPTERS TO PUBLISH")
print("=" * 70)
for a in adapters:
    w = os.path.join(a, "adapter_model.safetensors")
    sz = os.path.getsize(w) / 2**20 if os.path.exists(w) else 0.0
    print(f"  {os.path.basename(a):<24}{sz:>8.1f} MiB   {a}")

# ---- stage only what is needed to reload -----------------------------------
# optimizer/scheduler state is only needed to RESUME training, not to evaluate,
# and it is what makes /kaggle/working too big to move.
KEEP = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
        "training_config.json", "README.md")

STAGE = os.path.join(WORK, "_ds_phase8")
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

total = 0
for a in adapters:
    dst = os.path.join(STAGE, os.path.basename(a))
    os.makedirs(dst, exist_ok=True)
    for f in KEEP:
        src = os.path.join(a, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, f))
            total += os.path.getsize(src)

# results are tiny and belong with the weights
for extra in ("results_phase8", "results_phase7"):
    p = os.path.join(WORK, extra)
    if os.path.isdir(p):
        shutil.copytree(p, os.path.join(STAGE, extra), dirs_exist_ok=True)
        print(f"  + {extra}/")

print(f"\nstaged {total/2**20:.1f} MiB")

# ---- always produce a zip, whatever happens next ----------------------------
ZIP = os.path.join(WORK, "wordle_phase8_adapters.zip")
if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(STAGE):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, STAGE))
print(f"zip: {ZIP}  ({os.path.getsize(ZIP)/2**20:.1f} MiB)")
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(ZIP, WORK)))
except Exception:
    pass

# ---- publish ----------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = sec.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = sec.get_secret("KAGGLE_KEY")
    USER = os.environ["KAGGLE_USERNAME"]
except Exception as e:
    print("\n" + "=" * 70)
    print("NO CREDENTIALS - stopping after the zip")
    print("=" * 70)
    print(f"  {e}")
    print("  Add-ons -> Secrets -> add KAGGLE_USERNAME and KAGGLE_KEY,")
    print("  ATTACH both to this notebook, then re-run this cell.")
    print("  The zip above is downloadable in the meantime.")
    raise SystemExit

json.dump({"title": TITLE, "id": f"{USER}/{SLUG}",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(STAGE, "dataset-metadata.json"), "w"), indent=2)

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.strip()[:1200])
    return p.returncode, out

print("\ncreating dataset ...")
rc, out = run(["kaggle", "datasets", "create", "-p", STAGE, "-r", "zip"])
if rc != 0 and ("already exists" in out or "409" in out or "存在" in out):
    print("\nexists -> pushing a new version")
    rc, out = run(["kaggle", "datasets", "version", "-p", STAGE,
                   "-m", "phase 8 DPO adapter", "-r", "zip"])

print("\n" + "=" * 70)
if rc == 0:
    print(f"PUBLISHED -> https://www.kaggle.com/datasets/{USER}/{SLUG}")
    print("=" * 70)
    print("\nIt can take a minute to finish processing before it is attachable.")
    print("\nNext session:")
    print("  Add Input -> that dataset, then set")
    print(f'      PREV_RUN_DIR = "/kaggle/input/{SLUG}"')
    print("  Section 3 then skips SFT, and section 6 skips DPO training.")
else:
    print("PUBLISH FAILED - use the zip download above instead")
    print("=" * 70)
