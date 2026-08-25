"""Make Phase-11 scripts runnable from their own directory.

Same shape as every other phase folder: put `core/` and the project root on
`sys.path`, then chdir to the root so relative data paths resolve unchanged.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "core"), os.path.join(ROOT, "phase8_dpo_v3"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)
