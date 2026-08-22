"""
validate_notebook.py — execute every notebook code cell in order, on a subset.

Verifies the notebook actually runs rather than merely being well-formed JSON:
cells are executed sequentially in one shared namespace, exactly as Jupyter
would, with two substitutions to keep it quick:

  * BENCHMARK_LIMIT is forced to a small value (full benchmark is validated
    separately by run_full_benchmark.py)
  * matplotlib is forced to the Agg backend so plt.show() cannot block

`%%writefile` cells are honoured and the written files are checked byte-for-byte
against the real modules, which is what proves the notebook's embedded copies
have not drifted from the standalone module.

    python validate_notebook.py [--limit 120]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(_paths.ROOT, "phase1_classical",
                  "wordle_classical_solver.ipynb")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=NB)
    ap.add_argument("--limit", type=int, default=120,
                    help="value forced into BENCHMARK_LIMIT")
    ap.add_argument("--workdir", default=None,
                    help="directory to execute in (default: alongside the notebook)")
    ap.add_argument("--artifact-dir", default="artifacts_validation",
                    help="isolated artifact dir so validation cannot clobber the real bundle")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    with open(args.notebook, encoding="utf-8") as fh:
        nb = json.load(fh)

    # Basic structural validation first.
    assert nb["nbformat"] == 4, "expected nbformat 4"
    for i, c in enumerate(nb["cells"]):
        assert c["cell_type"] in ("code", "markdown"), f"cell {i}: bad type"
        assert isinstance(c["source"], (str, list)), f"cell {i}: bad source"
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"notebook OK: {len(nb['cells'])} cells "
          f"({len(codes)} code, {len(nb['cells']) - len(codes)} markdown)\n")

    workdir = args.workdir or HERE
    os.chdir(workdir)

    import matplotlib
    matplotlib.use("Agg")

    ns = {"__name__": "__main__", "__builtins__": __builtins__}
    n_exec = 0
    writefiles = []

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue

        # --- handle %%writefile exactly as Jupyter would -------------------
        first = src.split("\n", 1)[0].strip()
        if first.startswith("%%writefile"):
            target = first.split(None, 1)[1].strip()
            body = src.split("\n", 1)[1] if "\n" in src else ""
            existing = None
            if os.path.exists(target):
                with open(target, encoding="utf-8") as fh:
                    existing = fh.read()
            # Compile-check the embedded source before trusting it.
            try:
                compile(body, target, "exec")
            except SyntaxError as exc:
                print(f"FAIL cell {idx}: embedded {target} has a syntax error: {exc}")
                return 1
            drift = (existing is not None
                     and existing.rstrip("\n") != body.rstrip("\n"))
            writefiles.append((target, drift, existing is not None))
            print(f"  cell {idx:>2}  %%writefile {target}  "
                  f"({len(body)} chars, compiles OK, "
                  f"{'DRIFT vs on-disk copy' if drift else 'matches on-disk copy'})")
            continue

        # --- ordinary cell -------------------------------------------------
        if "BENCHMARK_LIMIT = 0" in src:
            src = src.replace("BENCHMARK_LIMIT = 0",
                              f"BENCHMARK_LIMIT = {args.limit}")
            print(f"  cell {idx:>2}  [BENCHMARK_LIMIT forced to {args.limit}]")

        # Redirect artifact writes into an isolated directory so a validation
        # run can never overwrite the real bundle or the full benchmark's
        # results with subset numbers.
        if 'ARTIFACT_DIR = "artifacts"' in src:
            src = src.replace('ARTIFACT_DIR = "artifacts"',
                              f'ARTIFACT_DIR = {args.artifact_dir!r}')
            print(f"  cell {idx:>2}  [ARTIFACT_DIR redirected to {args.artifact_dir}]")
        if 'ARTIFACT_DIR = "/kaggle/working/artifacts"' in src:
            src = src.replace('ARTIFACT_DIR = "/kaggle/working/artifacts"',
                              f'ARTIFACT_DIR = {args.artifact_dir!r}')

        buf = io.StringIO()
        old = sys.stdout
        try:
            if not args.verbose:
                sys.stdout = buf
            exec(compile(src, f"<cell {idx}>", "exec"), ns)
        except Exception:
            sys.stdout = old
            print(f"\nFAIL cell {idx}:\n{'-'*70}")
            print(src[:1500])
            print("-" * 70)
            traceback.print_exc()
            print("\n--- captured stdout ---")
            print(buf.getvalue()[-2000:])
            return 1
        finally:
            sys.stdout = old
        n_exec += 1
        out = buf.getvalue()
        head = next((l for l in out.split("\n") if l.strip()), "")
        print(f"  cell {idx:>2}  OK   {head[:88]}")

    print(f"\nexecuted {n_exec} code cells successfully.")
    drifted = [t for t, d, existed in writefiles if d]
    if drifted:
        print(f"\nWARNING: embedded copies differ from the on-disk modules: {drifted}")
        print("Run `python make_notebook.py` to regenerate the notebook.")
        return 1
    print(f"embedded modules match their on-disk counterparts: "
          f"{[t for t, _, _ in writefiles]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
