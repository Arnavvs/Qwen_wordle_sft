"""Make corrected Phase-8 scripts runnable from their phase directory."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
for _path in (CORE, ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(ROOT)
