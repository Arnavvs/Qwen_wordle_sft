"""
organize_project.py - tidy the project root without breaking anything.

WHAT STAYS PUT AND WHY

  *.py at the root
      Every script imports its siblings flat (`from wordle_solver import ...`),
      and `prepare_kaggle_dataset.py` hard-codes root-relative paths for
      wordle_solver / tree_search / benchmark / generate_trajectories. Moving
      them into a package dir would break the import graph and the Kaggle
      packaging step for no real gain on a project this size.

  *.ipynb at the root
      make_notebook.py and make_sft_notebook.py both write to
      os.path.join(HERE, "<name>.ipynb"). Moving the outputs without editing
      the generators would just recreate them at the root on the next run.

  data/ artifacts/ sft_package/ tests/ trajectories*/
      Referenced by hard-coded relative paths in the scripts and in the
      notebook's dataset layout.

WHAT MOVES: loose logs, docs, one-off Kaggle cell snippets, result folders and
zips, and upload staging dirs. Nothing here is imported or path-referenced.

Nothing is deleted except __pycache__ / .pytest_cache. Duplicated archives are
reported, not removed - that call is the user's.

    python organize_project.py --dry-run     # show the plan
    python organize_project.py               # do it
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

MOVES = {
    "logs": [
        "bench_log.txt", "hybrid_sweep_log.txt", "power_log.txt",
        "tiebreak_log.txt", "topk_log.txt", "validate_log.txt",
        "verify_salet.txt", "tree_full_d1.txt", "tree_full_d2.txt",
        "tree_full_d3.txt", "tree_full_salet.txt", "tree_full_soare.txt",
        "tree_scaling.txt", "gen_entropy.log", "gen_salet.log",
        "gen_soare.log", "training_examples.txt",
        "training_examples_v2_no_count.txt",
    ],
    "docs": ["RUN_DIAGNOSTIC.md"],
    "kaggle_cells": [
        "CELL_8b_PATCH.py", "CELL_8b_constrained_decoding.py",
        "CELL_12b_FIXED.py", "CELL_base_constrained.py",
        "CELL_RECOVER_AND_SAVE.py", "CELL_dump_results.py",
    ],
    "results": [
        "wordle_sft_results", "wordle_sft_important_results",
        "results_reconstructed", "wordle_sft_results.zip",
        "wordle_sft_important_results.zip", "wordle_diagnostic_results.zip",
    ],
    "uploads": [
        "kaggle_upload", "kaggle_adapters_upload", "kaggle_adapters_upload.zip",
    ],
}

NUKE = ["__pycache__", ".pytest_cache"]

# (extracted dir, its archive) - same content stored twice
DUPES = [
    ("results/wordle_sft_results", "results/wordle_sft_results.zip"),
    ("uploads/kaggle_adapters_upload", "uploads/kaggle_adapters_upload.zip"),
]


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    act = not a.dry_run

    print("=" * 68)
    print("PLAN" if a.dry_run else "ORGANIZING")
    print("=" * 68)

    moved = skipped = 0
    for dest, names in MOVES.items():
        d = os.path.join(HERE, dest)
        present = [n for n in names if os.path.exists(os.path.join(HERE, n))]
        if not present:
            continue
        print(f"\n{dest}/")
        if act:
            os.makedirs(d, exist_ok=True)
        for n in present:
            src, dst = os.path.join(HERE, n), os.path.join(d, n)
            if os.path.exists(dst):
                print(f"    SKIP {n}  (already in {dest}/)")
                skipped += 1
                continue
            print(f"    <- {n}")
            if act:
                shutil.move(src, dst)
            moved += 1

    print("\ndelete (regenerable):")
    freed = 0
    for n in NUKE:
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            s = size_of(p)
            freed += s
            print(f"    {n}  ({human(s)})")
            if act:
                shutil.rmtree(p, ignore_errors=True)

    print(f"\n{moved} moved, {skipped} skipped, {human(freed)} freed")

    print("\n" + "=" * 68)
    print("DUPLICATED ARCHIVES - reported only, NOT deleted")
    print("=" * 68)
    total = 0
    for folder, zipf in DUPES:
        fp, zp = os.path.join(HERE, folder), os.path.join(HERE, zipf)
        if os.path.exists(fp) and os.path.exists(zp):
            zs = size_of(zp)
            total += zs
            print(f"  {folder}/  ({human(size_of(fp))})")
            print(f"  {zipf}  ({human(zs)})  <- same content, already extracted")
    if total:
        print(f"\n  Deleting just the .zip files would free {human(total)}.")
        print("  The extracted folders keep every file, so nothing is lost.")
        print("  Left in place - say the word and I'll remove them.")

    print("\n" + "=" * 68)
    print("ROOT AFTER")
    print("=" * 68)
    for n in sorted(os.listdir(HERE)):
        if n.startswith("."):
            continue
        p = os.path.join(HERE, n)
        tag = "DIR " if os.path.isdir(p) else "    "
        print(f"  {tag}{n}")


if __name__ == "__main__":
    main()
