# =============================================================================
# Check what survived the restart, and print the diagnostic results inline so
# they live in the notebook itself -- no download needed.
#
# wordle_diagnostic_results.zip is ~10 KB. The 656 MiB wordle_sft_results.zip
# is just WORK_DIR with copies of adapters you already have locally; you do not
# need it, and it is what makes the download button hang.
# =============================================================================
import os, json, base64, glob

ROOT = "/kaggle/working/results"
ZIP  = "/kaggle/working/wordle_diagnostic_results.zip"

print("=" * 70)
print("WHAT IS STILL ON DISK")
print("=" * 70)
if not os.path.isdir(ROOT):
    print(f"  {ROOT} is GONE -- the session was torn down, not just restarted.")
    print("  Everything is still in the notebook cell outputs above, so nothing")
    print("  is actually lost. Re-running would cost ~1.5 h and change nothing.")
else:
    for p in sorted(glob.glob(os.path.join(ROOT, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            print(f"  {os.path.getsize(p):>8,} B  {os.path.relpath(p, ROOT)}")
    if os.path.exists(ZIP):
        print(f"\n  zip: {os.path.getsize(ZIP):,} B  {ZIP}")

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    for name in ("comparison.md", "comparison.csv"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            print(f"\n--- {name} ---")
            print(open(p, encoding="utf-8").read())

    print("=" * 70)
    print("JSON RESULTS")
    print("=" * 70)
    for name in ("verdict.json", "terminal_probe.json", "eval_settings.json"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            print(f"\n--- {name} ---")
            print(json.dumps(json.load(open(p, encoding="utf-8")), indent=1))

    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*.json"))):
        print(f"\n--- {os.path.relpath(p, ROOT)} ---")
        d = json.load(open(p, encoding="utf-8"))
        print(json.dumps(d.get("metrics", d), indent=1))

    # Base64 of the small zip: copy the block below into a file and decode with
    #   python -c "import base64,sys;open('d.zip','wb').write(base64.b64decode(sys.stdin.read()))" < b64.txt
    if os.path.exists(ZIP) and os.path.getsize(ZIP) < 2_000_000:
        print("\n" + "=" * 70)
        print("BASE64 OF wordle_diagnostic_results.zip  (copy/paste to recover)")
        print("=" * 70)
        b = base64.b64encode(open(ZIP, "rb").read()).decode()
        for i in range(0, len(b), 100):
            print(b[i:i+100])
