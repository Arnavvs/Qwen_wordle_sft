# =============================================================================
# SAVE / EXPORT THE ADAPTER  -- paste as a cell and run
#
# Why the Output tab looked empty or undownloadable:
#   * an adapter DIRECTORY is not offered as a download, only individual files
#   * /kaggle/working also holds checkpoint-*/optimizer.pt (~200 MB each) from
#     training, which is what makes the folder too big to move
#   * "New Dataset" from output generally needs a committed Save Version; in a
#     live interactive session it is often not there at all
#
# This builds ONE small zip containing only the files needed to reload the
# adapter, and gives you three ways to get it out.
# =============================================================================
import os
import json
import glob
import shutil
import zipfile

WORK = "/kaggle/working"

# --- 1. find the adapter -----------------------------------------------------
found = []
for root, dirs, files in os.walk(WORK):
    if "adapter_config.json" in files and "checkpoint-" not in root:
        found.append(root)
if not found:
    # fall back to checkpoints if the final save never happened
    for root, dirs, files in os.walk(WORK):
        if "adapter_config.json" in files:
            found.append(root)

print("adapters found:")
for f in found:
    print("   ", f)
assert found, f"no adapter_config.json anywhere under {WORK}"

SRC = found[0]
NAME = os.path.basename(SRC.rstrip("/")) or "adapter"

# --- 2. what actually matters ------------------------------------------------
# PeftModel.from_pretrained needs only the config + weights. The tokenizer comes
# from the base model on the Hub, and optimizer/scheduler state is only needed
# to RESUME training, not to evaluate.
KEEP = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
        "training_config.json", "README.md"]

stage = os.path.join(WORK, f"_export_{NAME}")
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(os.path.join(stage, NAME), exist_ok=True)

total = 0
for f in KEEP:
    s = os.path.join(SRC, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(stage, NAME, f))
        sz = os.path.getsize(s)
        total += sz
        print(f"  + {f:<32}{sz/2**20:>8.2f} MiB")

skipped = sum(os.path.getsize(p) for p in glob.glob(os.path.join(WORK, "**", "*"),
                                                    recursive=True)
              if os.path.isfile(p)) - total
print(f"\n  packaged {total/2**20:.1f} MiB   (left behind ~{skipped/2**20:.0f} MiB "
      f"of checkpoints/optimizer state - not needed to load the adapter)")

# --- 3. one zip, small enough to actually download --------------------------
ZIP = os.path.join(WORK, f"{NAME}_adapter.zip")
if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(stage):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, stage))

print(f"\nZIP: {ZIP}  ({os.path.getsize(ZIP)/2**20:.1f} MiB)")
with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    print(f"  integrity: {'OK' if bad is None else 'CORRUPT at ' + bad}")
    for i in z.infolist():
        print(f"    {i.file_size/2**20:>8.2f} MiB  {i.filename}")

# --- 4. three ways to get it off Kaggle -------------------------------------
print("\n" + "=" * 70)
print("OPTION A - direct download (simplest)")
print("=" * 70)
try:
    from IPython.display import FileLink, display
    display(FileLink(os.path.relpath(ZIP, WORK)))
    print("  ^ click the link above")
except Exception as e:
    print(f"  FileLink unavailable ({e})")
print("  or: right sidebar -> Data -> output -> click the .zip -> download")
print("  (a single FILE is downloadable even when a folder is not)")

print("\n" + "=" * 70)
print("OPTION B - publish as a Dataset from code (no UI button needed)")
print("=" * 70)
print("""  Add-ons -> Secrets, add:   KAGGLE_USERNAME   and   KAGGLE_KEY
  (from kaggle.com -> Settings -> API -> Create New Token, values are in
   the kaggle.json it downloads)

  then run the OPTION B cell below.""")

print("\n" + "=" * 70)
print("OPTION C - Save Version (persists output without downloading)")
print("=" * 70)
print("""  Top right -> Save Version -> "Save & Run All" (or "Quick Save" if the
  run already finished). The committed output becomes attachable to a future
  notebook via Add Input -> Your Work -> Notebook Output, which is enough to
  skip retraining next session.""")
