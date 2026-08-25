"""
prepare_kaggle_dataset.py — assemble everything the Kaggle notebook needs into
one uploadable directory.

    python prepare_kaggle_dataset.py

Creates `kaggle_upload/` containing the SFT package, the solver modules, and the
precomputed feedback matrix. Upload that whole folder as a Kaggle Dataset; the
notebook then reads it through a single DATASET_DIR variable.

The feedback matrix (28.6 MiB) is included so the notebook can run the classical
baselines and the exact Wordle environment on Kaggle without rebuilding it
(~50 s) or needing network access.
"""

from __future__ import annotations

import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import argparse
import hashlib
import json
import os
import shutil

# (source, destination-relative-path). Everything is copied, nothing moved.
REQUIRED = [
    ("sft_package/data/entropy/train.jsonl",      "sft_package/data/entropy/train.jsonl"),
    ("sft_package/data/entropy/val.jsonl",        "sft_package/data/entropy/val.jsonl"),
    ("sft_package/data/tree_soare/train.jsonl",   "sft_package/data/tree_soare/train.jsonl"),
    ("sft_package/data/tree_soare/val.jsonl",     "sft_package/data/tree_soare/val.jsonl"),
    ("sft_package/data/tree_salet/train.jsonl",   "sft_package/data/tree_salet/train.jsonl"),
    ("sft_package/data/tree_salet/val.jsonl",     "sft_package/data/tree_salet/val.jsonl"),
    # Phase 6: synthetic endgame states (build_endgame_dataset.py).
    ("sft_package/data/tree_salet_endgame/train.jsonl",
     "sft_package/data/tree_salet_endgame/train.jsonl"),
    # Phase 8: DPO preference pairs (build_dpo_dataset.py).
    ("sft_package/data/dpo_midgame/train.jsonl",
     "sft_package/data/dpo_midgame/train.jsonl"),
    # Phase 8 v3: the audited replacement (phase8_dpo_v3/).  Unlike
    # dpo_midgame this has a real validation split, and the manifest must
    # travel with the rows — it records the value function and the bands the
    # labels were produced under, which the rows alone do not pin down.
    ("sft_package/data/dpo_v3/train.jsonl",
     "sft_package/data/dpo_v3/train.jsonl"),
    ("sft_package/data/dpo_v3/validation.jsonl",
     "sft_package/data/dpo_v3/validation.jsonl"),
    ("sft_package/data/dpo_v3/manifest.json",
     "sft_package/data/dpo_v3/manifest.json"),
    # Phase 11: the precomputed GRPO reward table. The manifest records which
    # value family produced each task and must travel with the rows.
    ("sft_package/data/grpo_tasks/train.jsonl",
     "sft_package/data/grpo_tasks/train.jsonl"),
    ("sft_package/data/grpo_tasks/validation.jsonl",
     "sft_package/data/grpo_tasks/validation.jsonl"),
    ("sft_package/data/grpo_tasks/manifest.json",
     "sft_package/data/grpo_tasks/manifest.json"),
    ("sft_package/data/disagreement/contrast.jsonl",
     "sft_package/data/disagreement/contrast.jsonl"),
    ("sft_package/data/disagreement/entropy.jsonl",
     "sft_package/data/disagreement/entropy.jsonl"),
    ("sft_package/data/disagreement/tree_soare.jsonl",
     "sft_package/data/disagreement/tree_soare.jsonl"),
    ("sft_package/eval/val_answers.jsonl",        "sft_package/eval/val_answers.jsonl"),
    ("sft_package/eval/train_answers.jsonl",      "sft_package/eval/train_answers.jsonl"),
    ("sft_package/manifest.json",                 "sft_package/manifest.json"),
    ("sft_package/README.md",                     "sft_package/README.md"),
    # Solver modules — the notebook needs the exact Wordle environment.
    ("core/wordle_solver.py",                          "code/wordle_solver.py"),
    ("core/tree_search.py",                            "code/tree_search.py"),
    ("core/benchmark.py",                              "code/benchmark.py"),
    ("core/generate_trajectories.py",                  "code/generate_trajectories.py"),
    # Vocabulary + precomputed feedback matrix.
    ("artifacts/answers.txt",                     "artifacts/answers.txt"),
    ("artifacts/valid_guesses.txt",               "artifacts/valid_guesses.txt"),
    ("artifacts/feedback_matrix.npy",             "artifacts/feedback_matrix.npy"),
    ("artifacts/metadata.json",                   "artifacts/metadata.json"),
    ("artifacts/frequency_model.json",            "artifacts/frequency_model.json"),
    ("artifacts/solver_config.json",              "artifacts/solver_config.json"),
]

OPTIONAL = [
    ("artifacts/val246_baselines.json", "artifacts/val246_baselines.json"),
    ("artifacts/benchmark_results.json", "artifacts/benchmark_results.json"),
    ("sft_package/index/id_map.jsonl", "sft_package/index/id_map.jsonl"),
    # The v3 audit trail. Shipped alongside the rows so the dataset can be
    # reviewed on Kaggle without the repository.
    ("sft_package/data/dpo_v3/audit.json",
     "sft_package/data/dpo_v3/audit.json"),
    ("phase8_dpo_v3/README.md", "phase8_dpo_v3/README.md"),
    ("phase8_dpo_v3/AUDIT.md", "phase8_dpo_v3/AUDIT.md"),
    ("phase8_dpo_v3/comparison.json", "phase8_dpo_v3/comparison.json"),
    ("sft_package/data/grpo_tasks/audit.json",
     "sft_package/data/grpo_tasks/audit.json"),
    ("phase11_grpo/RUN.md", "phase11_grpo/RUN.md"),
    ("phase11_grpo/decision_budget.json", "phase11_grpo/decision_budget.json"),
    ("phase8_dpo_v3/legacy_audit.json", "phase8_dpo_v3/legacy_audit.json"),
    ("phase8_dpo_v3/decision_structure.json",
     "phase8_dpo_v3/decision_structure.json"),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="kaggle_upload")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)

    if args.clean and os.path.isdir(args.dest):
        shutil.rmtree(args.dest)

    missing = [s for s, _ in REQUIRED if not os.path.exists(s)]
    if missing:
        print("MISSING required files — build them first:")
        for m in missing:
            print(f"  {m}")
        print("\n  python build_artifacts.py")
        print("  python generate_trajectories.py --solver <name> --verify ...")
        print("  python build_sft_package.py --clean")
        return 1

    hashes = {}
    total = 0
    for src, rel in REQUIRED + [(s, d) for s, d in OPTIONAL if os.path.exists(s)]:
        dst = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        size = os.path.getsize(dst)
        total += size
        hashes[rel] = {"sha256": sha256(dst), "bytes": size}

    manifest = {
        "purpose": "Kaggle Dataset for Qwen 0.5B Wordle SFT and audited DPO experiments",
        "upload_as": "Kaggle Dataset (private is fine)",
        "expected_mount": "/kaggle/input/<your-dataset-slug>/",
        "structure": ["sft_package/", "code/", "artifacts/", "phase8_dpo_v3/"],
        "n_files": len(hashes),
        "total_bytes": total,
        "files": hashes,
        "notes": [
            "code/ holds the exact Wordle environment used for evaluation.",
            "artifacts/feedback_matrix.npy is the 12972x2315 uint8 table; it "
            "lets the notebook run the classical baselines without rebuilding.",
            "No Hugging Face token or credential is included.",
            "sft_package/data/dpo_v3/ is the audited Phase-8 preference set. It "
            "has a real validation split; never concatenate validation.jsonl "
            "with train.jsonl. See phase8_dpo_v3/README.md for the format.",
        ],
    }
    with open(os.path.join(args.dest, "DATASET_MANIFEST.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"prepared {args.dest}/  ({len(hashes)} files, "
          f"{total/2**20:.1f} MiB)\n")
    for rel in sorted(hashes):
        print(f"  {rel:<52}{hashes[rel]['bytes']/1024:>10.1f} KiB")
    print(f"\nwrote {args.dest}/DATASET_MANIFEST.json")
    print("\nNEXT: upload this folder as a Kaggle Dataset, then set")
    print("      DATASET_DIR = \"/kaggle/input/<your-slug>\"  in the notebook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
