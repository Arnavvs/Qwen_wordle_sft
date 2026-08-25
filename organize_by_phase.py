"""
organize_by_phase.py - restructure the project by experiment phase.

THE CONSTRAINT THAT SHAPES THIS

Every script imports its siblings by TOP-LEVEL name (`from wordle_solver import
...`), and it must stay that way: `prepare_kaggle_dataset.py` copies the solver
modules into `code/` inside the Kaggle dataset, and the notebooks do
`sys.path.insert(0, CODE_DIR); from wordle_solver import ...`. Renaming them to
`core.wordle_solver` would break every notebook on Kaggle.

So the library moves to `core/` but stays importable under its bare name, via a
small `_paths.py` dropped into each phase folder:

    import _paths          # puts core/ + root on sys.path, chdir to root

The chdir matters: nearly every script uses relative data paths ("artifacts",
"sft_package/..."), and chdir'ing to the root makes all of them keep working
without touching a single path string. Only entry-point scripts import the
shim; the library modules in core/ never do.

    python organize_by_phase.py --dry-run
    python organize_by_phase.py
"""
import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))

LAYOUT = {
    "core": [
        "wordle_solver.py", "tree_search.py", "benchmark.py",
        "generate_trajectories.py", "constrained_decode.py",
    ],
    "phase1_classical": [
        "build_artifacts.py", "run_full_benchmark.py", "run_tree_benchmark.py",
        "tiebreak_experiment.py", "hybrid_sweep.py", "verify_salet_tree.py",
        "baselines_val246.py", "todays_wordle.py", "make_notebook.py",
        "wordle_classical_solver.ipynb",
    ],
    "phase2_trajectories": [
        "merge_trajectories.py", "rerender_prompts.py",
        "consolidate_results.py",
    ],
    "phase3_sft_package": [
        "build_sft_package.py", "build_weighted_dataset.py",
        "verify_no_leakage.py", "inspect_examples.py", "power_analysis.py",
    ],
    "phase4_5_sft_diagnostic": [
        "make_sft_notebook.py", "wordle_qwen_sft_kaggle.ipynb",
        "rebuild_diagnostic_results.py",
    ],
    "phase6_endgame": [
        "build_endgame_dataset.py", "verify_endgame_dataset.py",
        "audit_endgame_dataset.py", "make_endgame_notebook.py",
        "wordle_endgame_sft_kaggle.ipynb",
    ],
    "tools": [
        "prepare_kaggle_dataset.py", "validate_notebook.py",
        "organize_project.py",
    ],
}

# Scripts whose HERE is used for DATA and so must become the repo root.
# The notebook generators are deliberately excluded: their HERE should stay the
# phase folder so the .ipynb lands next to the generator that makes it.
HERE_TO_ROOT = {
    "build_artifacts.py", "run_full_benchmark.py", "run_tree_benchmark.py",
    "build_endgame_dataset.py", "verify_endgame_dataset.py",
    "audit_endgame_dataset.py", "rebuild_diagnostic_results.py",
    "prepare_kaggle_dataset.py",
}

NOTEBOOK_GENERATORS = {"make_notebook.py", "make_sft_notebook.py",
                       "make_endgame_notebook.py"}

SHIM = '''"""Make the phase folders runnable: core/ importable, root as cwd.

Every script imports the solver by its bare top-level name because the Kaggle
notebooks do the same after adding the dataset's code/ dir to sys.path. Keeping
the bare names working locally is what this file is for.

chdir(ROOT) makes relative data paths ("artifacts", "sft_package/...") resolve
from any phase folder, which is why none of them needed editing.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")

for _p in (CORE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(ROOT)
'''

CONFTEST = '''"""Let pytest find the solver modules now that they live in core/."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
'''

IMPORT_RE = re.compile(
    r"^(import|from)\s+(wordle_solver|tree_search|benchmark|"
    r"generate_trajectories|constrained_decode)\b", re.M)


def needs_shim(text, name):
    return bool(IMPORT_RE.search(text)) or name in HERE_TO_ROOT \
        or name in NOTEBOOK_GENERATORS


def patch(text, name):
    """Insert the shim import and fix HERE where it points at data."""
    changed = []

    if "import _paths" not in text:
        lines = text.split("\n")
        # after the module docstring, before the first import
        i = 0
        if lines and lines[0].startswith(('"""', "'''")):
            q = lines[0][:3]
            if lines[0].count(q) < 2:
                i = 1
                while i < len(lines) and q not in lines[i]:
                    i += 1
            i += 1
        while i < len(lines) and (not lines[i].strip()
                                  or lines[i].startswith("#")):
            i += 1
        # `from __future__` must stay the first statement after the docstring,
        # so the shim goes AFTER it, never before.
        last_future = max((j for j, l in enumerate(lines)
                           if l.startswith("from __future__")), default=None)
        if last_future is not None:
            i = last_future + 1
            lines.insert(i, "")
            i += 1
        lines.insert(i, "import _paths  # noqa: F401  (core/ on sys.path, cwd=root)")
        lines.insert(i + 1, "")
        text = "\n".join(lines)
        changed.append("shim")

    if name in HERE_TO_ROOT:
        new = re.sub(r"^HERE\s*=\s*os\.path\.dirname\(os\.path\.abspath\(__file__\)\)",
                     "HERE = _paths.ROOT  # data lives at the repo root",
                     text, count=1, flags=re.M)
        if new != text:
            text, _ = new, changed.append("HERE->ROOT")

    if name in NOTEBOOK_GENERATORS:
        new = text.replace('os.path.join(HERE, "constrained_decode.py")',
                           'os.path.join(_paths.CORE, "constrained_decode.py")')
        if new != text:
            text, _ = new, changed.append("core/constrained_decode")
        # make_notebook.py reads sibling modules to inline them
        new = text.replace('with open(os.path.join(HERE, name), encoding="utf-8") as fh:',
                           'with open(os.path.join(_paths.CORE, name), encoding="utf-8") as fh:')
        if new != text:
            text, _ = new, changed.append("core/module-read")

    return text, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    act = not a.dry_run
    print("=" * 70)
    print("PLAN" if a.dry_run else "RESTRUCTURING BY PHASE")
    print("=" * 70)

    moved = 0
    for dest, names in LAYOUT.items():
        present = [n for n in names if os.path.exists(os.path.join(HERE, n))]
        if not present:
            continue
        d = os.path.join(HERE, dest)
        print(f"\n{dest}/")
        if act:
            os.makedirs(d, exist_ok=True)
        for n in present:
            print(f"    <- {n}")
            if act:
                shutil.move(os.path.join(HERE, n), os.path.join(d, n))
            moved += 1

    if act:
        # shim into every folder except core/ (libraries must not chdir)
        for dest in LAYOUT:
            if dest == "core":
                continue
            p = os.path.join(HERE, dest, "_paths.py")
            if os.path.exists(os.path.join(HERE, dest)):
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(SHIM)
        with open(os.path.join(HERE, "conftest.py"), "w", encoding="utf-8") as fh:
            fh.write(CONFTEST)
        print("\nwrote _paths.py shims + conftest.py")

        print("\npatching moved scripts:")
        for dest, names in LAYOUT.items():
            if dest == "core":
                continue
            for n in names:
                p = os.path.join(HERE, dest, n)
                if not (os.path.exists(p) and n.endswith(".py")):
                    continue
                txt = open(p, encoding="utf-8").read()
                if not needs_shim(txt, n):
                    continue
                new, ch = patch(txt, n)
                if ch:
                    open(p, "w", encoding="utf-8").write(new)
                    print(f"    {dest}/{n}: {', '.join(ch)}")

        # prepare_kaggle_dataset copies the solver modules by path
        pk = os.path.join(HERE, "tools", "prepare_kaggle_dataset.py")
        if os.path.exists(pk):
            t = open(pk, encoding="utf-8").read()
            for m in ("wordle_solver.py", "tree_search.py", "benchmark.py",
                      "generate_trajectories.py"):
                t = t.replace(f'("{m}",', f'("core/{m}",')
            open(pk, "w", encoding="utf-8").write(t)
            print("    tools/prepare_kaggle_dataset.py: core/ source paths")

        vn = os.path.join(HERE, "tools", "validate_notebook.py")
        if os.path.exists(vn):
            t = open(vn, encoding="utf-8").read()
            t = t.replace('NB = os.path.join(HERE, "wordle_classical_solver.ipynb")',
                          'NB = os.path.join(_paths.ROOT, "phase1_classical",\n'
                          '                  "wordle_classical_solver.ipynb")')
            open(vn, "w", encoding="utf-8").write(t)
            print("    tools/validate_notebook.py: notebook path")

    print(f"\n{moved} files moved")
    print("\n" + "=" * 70)
    print("ROOT AFTER")
    print("=" * 70)
    for n in sorted(os.listdir(HERE)):
        if n.startswith(".") or n == "__pycache__":
            continue
        p = os.path.join(HERE, n)
        print(f"  {'DIR ' if os.path.isdir(p) else '    '}{n}")


if __name__ == "__main__":
    main()
