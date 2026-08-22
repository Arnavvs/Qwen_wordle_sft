"""Make Phase-8 v3 scripts runnable from their own directory.

Mirrors every other phase folder: put `core/` and the project root on
`sys.path`, then chdir to the root so relative data paths ("artifacts",
"sft_package/...") resolve exactly as they do everywhere else.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
for _path in (CORE, ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(ROOT)
