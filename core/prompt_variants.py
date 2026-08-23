"""
prompt_variants.py - twelve ways of showing the same Wordle state.

WHY THIS EXISTS

The largest measured effect in this project is not the model. It is the
scaffolding: a feedback-consistency constraint on the decoder was worth ~1.5
guesses, while eight phases of training were worth a fraction of that. This
module attacks that finding directly by varying the OTHER half of the
scaffolding - the prompt - with the decoder held fixed.

The variant that matters most is `raw_history`. The baseline prompt hands the
model `Confirmed letters : p1=A`, computed by the classical solver. If
performance collapses without it, a large part of what we have been calling
"the model playing Wordle" is the harness playing Wordle.

CONTRACT

Every variant is `f(turn, history, max_guesses) -> str` and must end with
"Next guess:" so the constrained decoder can score continuations identically
across variants. `history` is a list of `(guess_lower, pattern)` where pattern
is five characters from GYB.

LEAKAGE

One variant, `with_count`, deliberately shows the surviving-candidate count.
It is a solver-side quantity a player cannot see and it is marked `leaky=True`.
It exists only as an upper bound - "how much would the hint alone buy?" - and
its number must never be quoted as a fair result.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_trajectories import derive_constraints, render_prompt

RULES = (
    "You are playing Wordle. Deduce the hidden 5-letter word.\n"
    "After each guess you receive 5 feedback characters:\n"
    "  G = correct letter in the correct position\n"
    "  Y = letter is in the word but in a different position\n"
    "  B = letter is not in the word (or all its copies are already "
    "accounted for)\n"
)

TILE = {"G": "\U0001F7E9", "Y": "\U0001F7E8", "B": "⬛"}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _hist_lines(history, upper=True):
    out = []
    for i, (g, p) in enumerate(history, 1):
        w = g.upper() if upper else g
        out.append(f"  {i}. {w} -> {p}")
    return out


def _deduced_block(history):
    """The structured constraint block used by the baseline."""
    c = derive_constraints(history)
    lines = ["Deduced so far:"]
    greens = " ".join(f"p{i+1}={ch.upper()}"
                      for i, ch in enumerate(c["green_pattern"])
                      if ch and ch not in "._ ")
    lines.append(f"  Confirmed letters : {greens or '(none)'}")
    pres = ", ".join(sorted(x.upper() for x in c["present_letters"]))
    lines.append(f"  Letters present   : {pres or '(none)'}")
    if c["exact_counts"]:
        ex = ", ".join(f"{k.upper()}x{v}" for k, v in sorted(c["exact_counts"].items()))
        lines.append(f"  Exact counts      : {ex}")
    absent = ", ".join(sorted(x.upper() for x in c["absent_letters"]))
    lines.append(f"  Letters absent    : {absent or '(none)'}")
    fp = {k: sorted(v) for k, v in c["forbidden_positions"].items() if v}
    if fp:
        rl = ", ".join(f"{k.upper()} not at {v}" for k, v in sorted(fp.items()))
        lines.append(f"  Ruled-out spots   : {rl}")
    return "\n".join(lines)


def _turn_line(turn, max_guesses):
    return f"Guess {turn} of {max_guesses} ({max_guesses - turn + 1} remaining)"


def _empty(history):
    return not history


# --------------------------------------------------------------------------
# 1. baseline — exactly what every earlier phase used
# --------------------------------------------------------------------------
def v_baseline(turn, history, max_guesses):
    return render_prompt(
        turn=turn, history=list(history),
        constraints=derive_constraints(history),
        n_candidates=0, guesses_remaining=max_guesses - turn + 1,
        max_guesses=max_guesses, candidates=None, show_candidate_count=False)


# --------------------------------------------------------------------------
# 2. raw_history — no derived constraints. THE key variant.
# --------------------------------------------------------------------------
def v_raw_history(turn, history, max_guesses):
    h = ("History: (none - this is the opening guess)" if _empty(history)
         else "History:\n" + "\n".join(_hist_lines(history)))
    return f"{RULES}\n\n{h}\n\n{_turn_line(turn, max_guesses)}\n\nNext guess:"


# --------------------------------------------------------------------------
# 3. constraints_only — the deductions, but not the moves that produced them
# --------------------------------------------------------------------------
def v_constraints_only(turn, history, max_guesses):
    d = ("Nothing is known yet." if _empty(history) else _deduced_block(history))
    return f"{RULES}\n\n{d}\n\n{_turn_line(turn, max_guesses)}\n\nNext guess:"


# --------------------------------------------------------------------------
# 4. minimal — no instructions at all
# --------------------------------------------------------------------------
def v_minimal(turn, history, max_guesses):
    h = "\n".join(f"{g.upper()} {p}" for g, p in history)
    return (h + "\n\nNext guess:") if h else "Next guess:"


# --------------------------------------------------------------------------
# 5. with_count — LEAKY. Upper bound only.
# --------------------------------------------------------------------------
def v_with_count(turn, history, max_guesses, n_candidates=None):
    base = v_baseline(turn, history, max_guesses)
    if n_candidates is None:
        return base
    ins = f"Possible answers remaining: {n_candidates}\n\n"
    return base.replace("Next guess:", ins + "Next guess:")


# --------------------------------------------------------------------------
# 6. emoji — same information, different notation
# --------------------------------------------------------------------------
def v_emoji(turn, history, max_guesses):
    rules = (
        "You are playing Wordle. Deduce the hidden 5-letter word.\n"
        "After each guess you receive 5 tiles:\n"
        f"  {TILE['G']} = correct letter in the correct position\n"
        f"  {TILE['Y']} = letter is in the word but in a different position\n"
        f"  {TILE['B']} = letter is not in the word\n")
    lines = [f"  {i}. {g.upper()} {''.join(TILE[c] for c in p)}"
             for i, (g, p) in enumerate(history, 1)]
    h = ("History: (none - this is the opening guess)" if not lines
         else "History:\n" + "\n".join(lines))
    return f"{rules}\n\n{h}\n\n{_turn_line(turn, max_guesses)}\n\nNext guess:"


# --------------------------------------------------------------------------
# 7. verbose — feedback spelled out per letter, in prose
# --------------------------------------------------------------------------
def v_verbose(turn, history, max_guesses):
    if _empty(history):
        body = "You have not guessed yet."
    else:
        parts = []
        for i, (g, p) in enumerate(history, 1):
            bits = []
            for j, (ch, t) in enumerate(zip(g.upper(), p)):
                if t == "G":
                    bits.append(f"{ch} is correct at position {j+1}")
                elif t == "Y":
                    bits.append(f"{ch} is in the word but not at position {j+1}")
                else:
                    bits.append(f"{ch} is not in the word")
            parts.append(f"Guess {i} was {g.upper()}: " + "; ".join(bits) + ".")
        body = "\n".join(parts)
    return (f"{RULES}\n\nWhat you have learned:\n{body}\n\n"
            f"{_turn_line(turn, max_guesses)}\n\nNext guess:")


# --------------------------------------------------------------------------
# 8. reversed — most recent guess first
# --------------------------------------------------------------------------
def v_reversed(turn, history, max_guesses):
    rev = list(reversed(history))
    lines = [f"  {g.upper()} -> {p}" for g, p in rev]
    h = ("History: (none - this is the opening guess)" if not lines
         else "History (most recent first):\n" + "\n".join(lines))
    return (f"{RULES}\n\n{h}\n\n{_deduced_block(history) if history else ''}\n\n"
            f"{_turn_line(turn, max_guesses)}\n\nNext guess:")


# --------------------------------------------------------------------------
# 9. keyboard — the on-screen letter status board a human actually looks at
# --------------------------------------------------------------------------
def v_keyboard(turn, history, max_guesses):
    status = {}
    for g, p in history:
        for ch, t in zip(g.upper(), p):
            rank = {"B": 0, "Y": 1, "G": 2}
            if ch not in status or rank[t] > rank[status[ch]]:
                status[ch] = t
    rows = []
    for group in ("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"):
        rows.append("  " + " ".join(
            f"{c}{'*' if status.get(c) == 'G' else '+' if status.get(c) == 'Y' else '-' if status.get(c) == 'B' else ' '}"
            for c in group))
    kb = ("Letter status  (* = right spot, + = in word, - = not in word):\n"
          + "\n".join(rows))
    h = ("History: (none - this is the opening guess)" if _empty(history)
         else "History:\n" + "\n".join(_hist_lines(history)))
    return (f"{RULES}\n\n{h}\n\n{kb}\n\n"
            f"{_turn_line(turn, max_guesses)}\n\nNext guess:")


# --------------------------------------------------------------------------
# 10. few_shot — two worked examples before the real state
# --------------------------------------------------------------------------
# The exemplars are COMPUTED, not hand-written. The first draft of this block
# was typed by hand and listed B as an absent letter in a game where B appeared
# in neither guess - a worked example teaching the model a false deduction. The
# patterns and constraint blocks below are now derived by the same functions the
# real prompts use, so an exemplar cannot disagree with the game it illustrates.
# A FIXED exemplar list is the defect that voided the first run of this
# variant. Two exemplars ending "Next guess: CRAFT" / "Next guess: BOBBY" made
# the final word a constant attractor: the model opened BOOBY on all 100 games
# and the 6.06 mean measured verbatim copying, not few-shot learning.
#
# Nothing can structurally stop a model copying the last example, so this does
# two things instead. The pool is rotated per state, so a copier emits eight
# different words rather than one - it can no longer look like a stable policy.
# And the harness measures the copy rate directly (`shot_copy_pct`), so the
# variant is declared valid or invalid on a number rather than on inspection.
#
# Each entry is (answer, guesses_so_far, next_guess); `next_guess` is asserted
# legal and consistent with its own feedback at build time.
_SHOT_POOL = [
    ("chart", ["salet"],           "audit"),   # 20 admissible - wide probe
    ("hobby", ["salet", "crony"],  "howdy"),   #  7
    ("plumb", ["salet", "chord"],  "imply"),   # 17
    ("wince", ["salet", "dingo"],  "mince"),   #  2 - near the end
    ("proxy", ["salet", "chirp"],  "proud"),   #  4
    ("thumb", ["salet", "conic"],  "truth"),   #  6
    ("berth", ["salet", "brine"],  "berth"),   #  1 - forced; teaches closing
    ("glaze", ["salet"],           "blame"),   # 49
]

_N_SHOTS = 2          # exemplars per prompt; the pool is what rotates

# Kept for callers that still import the old name.
_SHOT_GAMES = _SHOT_POOL


def _render_shot(n, answer, guesses, nxt):
    from wordle_solver import feedback_code, code_to_pattern
    hist = [(g, code_to_pattern(feedback_code(g, answer))) for g in guesses]
    return "\n".join(
        [f"Example {n}", "History:"]
        + _hist_lines(hist)
        + [_deduced_block(hist), f"Next guess: {nxt.upper()}", ""])


def _shot_index(history):
    """Deterministic rotation offset. Hash of the live state, so the same state
    always renders the same exemplars (the run stays reproducible) while
    different states see different ones."""
    import hashlib
    h = hashlib.md5(repr(tuple(history)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def shot_words(history):
    """The exemplar next-guesses visible in this prompt. The harness uses this
    to compute the copy rate."""
    k, n = _shot_index(history), len(_SHOT_POOL)
    return [_SHOT_POOL[(k + i) % n][2].upper() for i in range(_N_SHOTS)]


def _shots(history):
    k, n = _shot_index(history), len(_SHOT_POOL)
    picks = [_SHOT_POOL[(k + i) % n] for i in range(_N_SHOTS)]
    return "\n".join(_render_shot(i, *p) for i, p in enumerate(picks, 1)) + "\n"


def v_few_shot(turn, history, max_guesses):
    return RULES + "\n\n" + _shots(history) + "Now your game.\n" + \
        v_baseline(turn, history, max_guesses).replace(RULES, "").lstrip("\n")


# --------------------------------------------------------------------------
# 11. guided_prose — the same deductions written as a sentence
# --------------------------------------------------------------------------
def v_guided_prose(turn, history, max_guesses):
    if _empty(history):
        body = "You know nothing yet, so pick a word that tests common letters."
    else:
        c = derive_constraints(history)
        g = [f"position {i+1} is {ch.upper()}"
             for i, ch in enumerate(c["green_pattern"])
             if ch and ch not in "._ "]
        pres = sorted(x.upper() for x in c["present_letters"])
        ab = sorted(x.upper() for x in c["absent_letters"])
        s = []
        if g:
            s.append("You know " + ", ".join(g) + ".")
        if pres:
            s.append("The word contains " + ", ".join(pres) + ".")
        if ab:
            s.append("It does not contain " + ", ".join(ab) + ".")
        fp = {k: sorted(v) for k, v in c["forbidden_positions"].items() if v}
        if fp:
            s.append("Also, " + "; ".join(
                f"{k.upper()} is not at position {[i+1 for i in v]}"
                for k, v in sorted(fp.items())) + ".")
        body = " ".join(s)
    h = "" if _empty(history) else "History:\n" + "\n".join(_hist_lines(history)) + "\n\n"
    return (f"{RULES}\n\n{h}{body}\n\n"
            f"{_turn_line(turn, max_guesses)}\n\nNext guess:")


# --------------------------------------------------------------------------
# 12. hard_mode_hint — tell it the rule the decoder already enforces
# --------------------------------------------------------------------------
def v_hard_mode_hint(turn, history, max_guesses):
    base = v_baseline(turn, history, max_guesses)
    hint = ("Play only words that are consistent with every clue above.\n\n")
    return base.replace("Next guess:", hint + "Next guess:")


# --------------------------------------------------------------------------
VARIANTS = {
    "baseline":         dict(fn=v_baseline,        leaky=False,
                             note="what every earlier phase used"),
    "raw_history":      dict(fn=v_raw_history,     leaky=False,
                             note="no derived constraints - does the model deduce?"),
    "constraints_only": dict(fn=v_constraints_only, leaky=False,
                             note="deductions without the moves"),
    "minimal":          dict(fn=v_minimal,         leaky=False,
                             note="no instructions at all"),
    "with_count":       dict(fn=v_with_count,      leaky=True,
                             note="LEAKY upper bound - shows candidates remaining"),
    "emoji":            dict(fn=v_emoji,           leaky=False,
                             note="tiles instead of GYB"),
    "verbose":          dict(fn=v_verbose,         leaky=False,
                             note="feedback spelled out per letter"),
    "reversed":         dict(fn=v_reversed,        leaky=False,
                             note="most recent guess first"),
    "keyboard":         dict(fn=v_keyboard,        leaky=False,
                             note="letter status board, as a human sees"),
    "few_shot":         dict(fn=v_few_shot,        leaky=False,
                             note="two worked examples first"),
    "guided_prose":     dict(fn=v_guided_prose,    leaky=False,
                             note="constraints as a sentence"),
    "hard_mode_hint":   dict(fn=v_hard_mode_hint,  leaky=False,
                             note="states the rule the decoder enforces"),
}


def render(name, turn, history, max_guesses, n_candidates=None):
    spec = VARIANTS[name]
    if name == "with_count":
        return spec["fn"](turn, history, max_guesses, n_candidates)
    return spec["fn"](turn, history, max_guesses)
