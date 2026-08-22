"""Make the phase folders runnable: core/ importable, root as cwd.

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
