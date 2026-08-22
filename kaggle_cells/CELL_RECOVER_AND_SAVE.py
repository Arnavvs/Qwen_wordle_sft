# =============================================================================
# Recovery cell — run this, then re-run section 13, then 13b, then 13c.
#
# WHY IT IS NEEDED: the back-compat aliases PRIMARY / GAMES / EVAL are assigned
# at the very END of the section-9 evaluation cell, after the model loop. Your
# KeyboardInterrupt during base-Qwen-constrained aborted the cell before those
# lines ran, so section 13 died on `NameError: name 'EVAL' is not defined`.
# Nothing was lost — EVAL_ALL and GAMES_ALL were populated incrementally inside
# the loop and are intact. This just rebuilds the three missing aliases.
# =============================================================================
PRIMARY = EVAL_MODES[0]
GAMES   = GAMES_ALL[PRIMARY]
EVAL    = {k: v for k, v in EVAL_ALL[PRIMARY].items() if k in EXPERIMENTS}

# Defensive: any of these would also NameError if their cell was skipped.
for _n, _d in [("TERMINAL", {}), ("TERMINAL_SPLIT", {}),
               ("TERMINAL_ROWS", []), ("EXAMPLES", {}), ("BASELINES", [])]:
    globals().setdefault(_n, _d)

print("recovered aliases:")
print(f"  PRIMARY = {PRIMARY}")
print(f"  EVAL    = {sorted(EVAL)}")
print("\nwhat will be written to the results:")
for _m in EVAL_MODES:
    print(f"  {_m:<14} {sorted(EVAL_ALL[_m])}")
print(f"  terminal probe  {sorted(TERMINAL)}")
print(f"  baselines       {[b['model'] for b in BASELINES]}")
print(f"  examples        {sorted(EXAMPLES)}")
print("\nNow run: section 13  ->  13b  ->  13c")
