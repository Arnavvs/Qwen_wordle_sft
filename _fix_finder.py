p = "phase8_dpo/make_dpo_notebook.py"
L = open(p, encoding="utf-8").read().split("\n")

start = next(i for i, l in enumerate(L) if l.startswith("def find_adapter(name):"))
end = next(i for i in range(start, len(L)) if l_ := L[i].strip() == "return None")
assert L[end].strip() == "return None", L[end]

new = '''def find_adapter(name):
    """Exact-name match only.

    An earlier version fell back to ANY adapter when the requested name was
    missing. That silently loaded a DPO adapter as the SFT base: the run looked
    entirely normal while measuring the wrong model. There is no fallback now -
    a missing name is an error, not a guess.
    """
    hits = []
    for root in ([PREV_RUN_DIR] if PREV_RUN_DIR else []) + [WORK_DIR, "/kaggle/input"]:
        if not root or not os.path.isdir(root):
            continue
        for dp, _, fs in os.walk(root):
            if "adapter_config.json" in fs and "checkpoint-" not in dp:
                hits.append(dp)
                if os.path.basename(dp.rstrip("/")) == name:
                    print(f"  adapter '{name}' -> {dp}")
                    return dp
    if hits:
        print(f"  NO adapter named '{name}'. Found instead:")
        for h in sorted(set(hits)):
            print(f"      {os.path.basename(h.rstrip('/')):<24} {h}")
        print("  Refusing to fall back - that would measure the wrong model.")
    return None'''

L[start:end + 1] = new.split("\n")
open(p, "w", encoding="utf-8").write("\n".join(L))
print("patched lines", start + 1, "to", end + 1)
