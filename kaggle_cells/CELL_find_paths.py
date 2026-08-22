# =============================================================================
# Find PREV_RUN_DIR (the adapter) and DATASET_DIR (the SFT package).
# Paste as a cell, run, copy the two lines it prints.
# =============================================================================
import os

# --- the adapter -> PREV_RUN_DIR --------------------------------------------
# PREV_RUN_DIR must be the folder CONTAINING the adapter folder, not the
# adapter folder itself. Kaggle also unpacks uploads under an extra directory
# level, so guessing this is the usual way to waste a session.
hits = []
for root, dirs, files in os.walk("/kaggle/input"):
    if "adapter_config.json" in files and "checkpoint-" not in root:
        hits.append(root)

print("=" * 70)
print("ADAPTERS FOUND")
print("=" * 70)
if not hits:
    print("  none under /kaggle/input")
    print("  -> Add Input the adapter dataset, or run Phase 7 to train one")
for h in hits:
    print(f"  adapter folder : {h}")
    print(f"  its name       : {os.path.basename(h)}")
    print(f"  PREV_RUN_DIR   = \"{os.path.dirname(h)}\"")
    print()

# also check working dir (same session, already trained)
work = []
for root, dirs, files in os.walk("/kaggle/working"):
    if "adapter_config.json" in files and "checkpoint-" not in root:
        work.append(root)
if work:
    print("  (also in /kaggle/working, usable without PREV_RUN_DIR:)")
    for w in work:
        print(f"     {w}")
    print()

# --- the SFT package -> DATASET_DIR -----------------------------------------
NEED = "sft_package/data/tree_salet_endgame/train.jsonl"
OLD = "sft_package/data/tree_salet/train.jsonl"
print("=" * 70)
print("SFT PACKAGE")
print("=" * 70)
v2 = [dp for dp, _, _ in os.walk("/kaggle/input")
      if os.path.exists(os.path.join(dp, NEED))]
v1 = [dp for dp, _, _ in os.walk("/kaggle/input")
      if os.path.exists(os.path.join(dp, OLD))
      and not os.path.exists(os.path.join(dp, NEED))]
for d in v2:
    print(f"  v2 (has endgame data)  : {d}")
    print(f"  DATASET_DIR = None      # auto-detect finds this")
for d in v1:
    print(f"  v1 WITHOUT endgame data: {d}")
    print("     ^ Phase 6/7 will refuse this. Attach the v2 package.")
if not v2 and not v1:
    print("  none found - Add Input the SFT package dataset")

print("\n" + "=" * 70)
print("PASTE INTO THE CONFIG CELL")
print("=" * 70)
print("DATASET_DIR  = None")
print(f'PREV_RUN_DIR = "{os.path.dirname(hits[0])}"' if hits
      else "PREV_RUN_DIR = None   # no adapter found; Phase 7 will train one")
