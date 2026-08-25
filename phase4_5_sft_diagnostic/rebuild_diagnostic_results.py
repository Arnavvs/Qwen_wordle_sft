"""
rebuild_diagnostic_results.py - reconstruct results/ from the notebook outputs.

The Kaggle session was torn down before wordle_diagnostic_results.zip could be
downloaded, taking /kaggle/working/results with it. Every number that was
PRINTED survives in the executed notebook, so the tree is rebuilt from that.

Fidelity, stated honestly:

  * unconstrained rows - FULL. They reproduce the original run bit for bit
    (invalid 36.74% = 8.13 format + 28.61 word), so the complete metric dicts
    are taken from wordle_sft_important_results/results.json.
  * constrained rows - PRINTED FIELDS ONLY. mean / fail / solved / invalid /
    repeat / le3 / le4 / le5 / median / max / timing / pruning are exact.
    Fields never printed (per-guess distribution, std, mean_solved_only,
    avg_candidates_after_turn, hard_mode_violation, avg_decision_margin) are
    null and flagged.
  * terminal probe - aggregates FULL. The per-state rows of terminal_probe.csv
    were only ever written to disk and are genuinely lost.
  * openers / disagreement / baselines / verdict - FULL.
"""
import _paths  # noqa: F401  (core/ on sys.path, cwd=root)

import csv
import json
import os
import shutil

HERE = _paths.ROOT  # data lives at the repo root
OUT = os.path.join(HERE, "results", "results_reconstructed")
ORIG = os.path.join(HERE, "results", "wordle_sft_important_results",
                    "results.json")

LOST = "never printed to the notebook; only existed in the lost on-disk copy"

BASELINES = [
    dict(model="random",     mean=4.0203, med=4, mx=6, le3=27.64, le4=71.54, le5=94.72,  fail=0.81),
    dict(model="frequency",  mean=3.7927, med=4, mx=6, le3=39.84, le4=79.67, le5=95.53,  fail=1.63),
    dict(model="entropy",    mean=3.4431, med=3, mx=5, le3=56.10, le4=97.97, le5=100.00, fail=0.00),
    dict(model="tree_soare", mean=3.4309, med=3, mx=5, le3=58.54, le4=95.93, le5=100.00, fail=0.00),
    dict(model="tree_salet", mean=3.4512, med=3, mx=5, le3=53.66, le4=96.75, le5=100.00, fail=0.00),
]

CONSTRAINED = {
    "entropy":    dict(mean=5.9715, med=4, mx=6, le3=10.98, le4=21.54, le5=32.52,
                       fail=62.20, repeat=36.2, secs=951, chunks=1.4, n=246),
    "tree_soare": dict(mean=5.9309, med=4, mx=6, le3=14.63, le4=22.76, le5=32.11,
                       fail=63.41, repeat=37.8, secs=944, chunks=1.4, n=246),
    "tree_salet": dict(mean=5.6870, med=4, mx=6, le3=17.07, le4=30.08, le5=38.62,
                       fail=56.10, repeat=30.5, secs=912, chunks=1.4, n=246),
    "base_qwen":  dict(mean=7.0000, med=0, mx=0, le3=0.0, le4=0.0, le5=0.0,
                       fail=100.0, repeat=100.0, secs=3067, chunks=26.0, n=62),
}

BASE_UNCON = dict(mean=7.0000, med=0, mx=0, le3=0.0, le4=0.0, le5=0.0,
                  fail=100.0, repeat=100.0, invalid_format=1.49,
                  invalid_word=0.0, secs=36, n=246)

OPENERS = {
    "unconstrained": {
        "entropy":    {"expected": "SOARE", "pct_expected": 100.0, "most_common": [["SOARE", 246]], "n_distinct": 1},
        "tree_soare": {"expected": "SOARE", "pct_expected": 100.0, "most_common": [["SOARE", 246]], "n_distinct": 1},
        "tree_salet": {"expected": "SALET", "pct_expected": 100.0, "most_common": [["SALET", 246]], "n_distinct": 1},
        "base_qwen":  {"expected": None, "pct_expected": None, "most_common": [["GUESS", 246]], "n_distinct": 1},
    },
    "constrained": {
        "entropy":    {"expected": "SOARE", "pct_expected": 100.0, "most_common": [["SOARE", 246]], "n_distinct": 1},
        "tree_soare": {"expected": "SOARE", "pct_expected": 100.0, "most_common": [["SOARE", 246]], "n_distinct": 1},
        "tree_salet": {"expected": "SALET", "pct_expected": 100.0, "most_common": [["SALET", 246]], "n_distinct": 1},
        "base_qwen":  {"_note": "not printed for this mode"},
    },
}

DISAGREE = {
    "unconstrained": {
        "entropy":    dict(n=111, ent=90.09, tree=0.00,  neither=9.91),
        "tree_soare": dict(n=109, ent=0.00,  tree=89.91, neither=10.09),
        "tree_salet": dict(n=0,   ent=0.00,  tree=0.00,  neither=0.00),
        "base_qwen":  dict(n=0,   ent=0.00,  tree=0.00,  neither=0.00),
    },
    "constrained": {
        "entropy":    dict(n=112, ent=90.18, tree=0.00,  neither=9.82),
        "tree_soare": dict(n=112, ent=0.00,  tree=93.75, neither=6.25),
        "tree_salet": dict(n=0,   ent=0.00,  tree=0.00,  neither=0.00),
    },
}

TERMINAL = {
    "qwen_tree_salet": {
        "1": dict(n_states=80, top1_accuracy_pct=20.00, top1_is_candidate_pct=20.00,
                  median_rank_of_answer=7.5, chance_top1_pct=0.0077,
                  chance_within_candidates_pct=100.0),
        "2": dict(n_states=30, top1_accuracy_pct=6.67, top1_is_candidate_pct=26.67,
                  median_rank_of_answer=18.0, chance_top1_pct=0.0077,
                  chance_within_candidates_pct=50.0),
        "3": dict(n_states=27, top1_accuracy_pct=3.70, top1_is_candidate_pct=25.93,
                  median_rank_of_answer=80.0, chance_top1_pct=0.0077,
                  chance_within_candidates_pct=33.33),
    },
    "base_qwen": {
        "1": dict(n_states=80, top1_accuracy_pct=0.0, top1_is_candidate_pct=0.0,
                  median_rank_of_answer=1388.0, chance_top1_pct=0.0077,
                  chance_within_candidates_pct=100.0),
        "2": dict(n_states=30, top1_accuracy_pct=0.0, top1_is_candidate_pct=0.0,
                  median_rank_of_answer=2947.0, chance_top1_pct=0.0077,
                  chance_within_candidates_pct=50.0),
        "3": dict(n_states=27, top1_accuracy_pct=0.0, top1_is_candidate_pct=0.0,
                  median_rank_of_answer=1861.0, chance_top1_pct=0.0077,
                  chance_within_candidates_pct=33.33),
    },
}

TERMINAL_SPLIT = {
    "qwen_tree_salet": dict(seen_n=21, seen_acc_pct=33.33, unseen_n=59, unseen_acc_pct=15.25),
    "base_qwen":       dict(seen_n=21, seen_acc_pct=0.0,   unseen_n=59, unseen_acc_pct=0.0),
}

VERDICT = {
    "verdict": "D",
    "evidence": [
        "best SFT constrained mean   = 5.6870",
        "best SFT unconstrained mean = 6.1748 (constrained gains +0.4878)",
        "base Qwen constrained mean  = 7.0000",
        "classical entropy = 3.4431, classical random = 4.0203",
        "terminal k=1 top-1 [qwen_tree_salet] = 20.0% (chance 0.0077%)",
        "terminal k=1 top-1 [base_qwen] = 0.0% (chance 0.0077%)",
        "k=1 seen vs unseen [base_qwen]: 0.0% (n=21) vs 0.0% (n=59)",
        "k=1 seen vs unseen [qwen_tree_salet]: 33.33% (n=21) vs 15.25% (n=59)",
        "-> constrained SFT still loses to random elimination; the failure is "
        "not primarily vocabulary",
    ],
    "legend": {
        "A": "SFT learned the policy; vocabulary generation was the main bottleneck.",
        "B": "SFT learned some policy but still cannot reason well enough.",
        "C": "SFT barely improved over base Qwen.",
        "D": "Something else.",
    },
}

EVAL_SETTINGS = {
    "greedy": True, "max_new_tokens": 8,
    "candidate_count_shown": False, "candidate_list_shown": False,
    "answer_shown": False, "modes_run": ["unconstrained", "constrained"],
    "constrained": {
        "legal_words": 12972,
        "score": "sum log P(token | prompt, prefix) over ' '+WORD tokens and "
                 "EOS; argmax over the legal set. Not post-hoc filtering: the "
                 "model never emits free text in this mode.",
        "chunk": 512, "exact_pruning": True, "length_normalise": False,
        "repeats_banned": False,
        "candidate_list_shown": False, "answer_shown": False,
        "verification": {
            "cache_vs_naive_max_abs_dev_nats": 0.0369,
            "tolerance_fp16": 0.40,
            "probe_score_spread_nats": 30.1,
            "ranking_correlation": 0.999998,
            "pruned_argmax_equals_full_argmax": True,
        },
    },
    "output_policy": "first 5-letter token; else first later 5-letter token "
                     "(flagged); else invalid. Invalid turns are consumed with "
                     "no feedback. No silent repair.",
}


def rowify(model, mode, d, invalid_fmt=0.0, invalid_word=0.0, extra=None):
    r = {
        "model": model, "mode": mode, "n_games": d.get("n", 246),
        "mean_failures_as_7": d["mean"],
        "failure_rate_pct": d["fail"],
        "solved_pct": round(100 - d["fail"], 2),
        "invalid_pct": round(invalid_fmt + invalid_word, 2),
        "invalid_format_rate_pct": invalid_fmt,
        "invalid_word_rate_pct": invalid_word,
        "repeated_guess_game_rate_pct": d.get("repeat"),
        "pct_le3": d["le3"], "pct_le4": d["le4"], "pct_le5": d["le5"],
        "median": d["med"], "max": d["mx"],
        "eval_seconds": d.get("secs"),
    }
    if extra:
        r.update(extra)
    return r


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for sub in ("unconstrained_sft", "constrained_sft", "base_qwen"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    orig = json.load(open(ORIG, encoding="utf-8"))
    table = []

    for b in BASELINES:
        table.append(rowify("classical " + b["model"], "symbolic", b,
                            extra={"repeated_guess_game_rate_pct": 0.0}))

    for name in ("entropy", "tree_soare", "tree_salet"):
        m = dict(orig["model_eval"][name])
        m["_provenance"] = ("full fidelity: the unconstrained run is "
                            "deterministic and reproduced the original run "
                            "exactly, so metrics come from "
                            "wordle_sft_important_results/results.json")
        json.dump({"metrics": m,
                   "opening_analysis": OPENERS["unconstrained"][name],
                   "disagreement_analysis": DISAGREE["unconstrained"][name]},
                  open(os.path.join(OUT, "unconstrained_sft",
                                    name + "__unconstrained.json"), "w",
                       encoding="utf-8"), indent=2)
        table.append(rowify("qwen_" + name, "unconstrained",
                            dict(mean=m["mean_failures_as_7"],
                                 fail=m["failure_rate_pct"],
                                 le3=m["pct_le3"], le4=m["pct_le4"],
                                 le5=m["pct_le5"], med=m["median"],
                                 mx=m["max"],
                                 repeat=m["repeated_guess_game_rate_pct"],
                                 secs=m.get("eval_seconds")),
                            m["invalid_format_rate_pct"],
                            m["invalid_word_rate_pct"]))

    br = rowify("base_qwen", "unconstrained", BASE_UNCON,
                BASE_UNCON["invalid_format"], BASE_UNCON["invalid_word"])
    table.append(br)
    json.dump({"metrics": br,
               "opening_analysis": OPENERS["unconstrained"]["base_qwen"],
               "disagreement_analysis": DISAGREE["unconstrained"]["base_qwen"]},
              open(os.path.join(OUT, "base_qwen",
                                "base_qwen__unconstrained.json"), "w",
                   encoding="utf-8"), indent=2)

    unavailable = {k: None for k in
                   ("distribution", "std", "mean_solved_only",
                    "avg_candidates_after_turn", "hard_mode_violation_pct",
                    "avg_decision_margin")}
    for name, d in CONSTRAINED.items():
        extra = dict(unavailable)
        extra["_unavailable_fields"] = LOST
        extra["avg_chunks_per_decision"] = d["chunks"]
        extra["unpruned_chunks"] = 26
        if name == "base_qwen":
            extra["subset_stride"] = 4
            extra["_note"] = ("62-game evenly spaced subset: pruning cannot "
                              "help an untrained model (26.0 of 26 chunks per "
                              "decision), making 246 games ~4h")
        label = "base_qwen (n=62 subset)" if name == "base_qwen" else "qwen_" + name
        r = rowify(label, "constrained", d, 0.0, 0.0, extra)
        table.append(r)
        tgt = "base_qwen" if name == "base_qwen" else "constrained_sft"
        json.dump({"metrics": r,
                   "opening_analysis": OPENERS["constrained"].get(name),
                   "disagreement_analysis": DISAGREE["constrained"].get(name)},
                  open(os.path.join(OUT, tgt, name + "__constrained.json"),
                       "w", encoding="utf-8"), indent=2)

    for fname, obj in [
        ("verdict.json", VERDICT),
        ("terminal_probe.json", {"per_model": TERMINAL,
                                 "seen_vs_unseen": TERMINAL_SPLIT,
                                 "_note": "aggregates exact; the per-state rows "
                                          "of terminal_probe.csv were never "
                                          "printed and are lost"}),
        ("eval_settings.json", EVAL_SETTINGS),
        ("classical_baselines_same_246.json", BASELINES),
        ("opening_analysis.json", OPENERS),
        ("disagreement_analysis.json", DISAGREE),
    ]:
        json.dump(obj, open(os.path.join(OUT, fname), "w", encoding="utf-8"),
                  indent=2)

    fields = list(table[0].keys())
    for t in table:
        for f in t:
            if f not in fields:
                fields.append(f)
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in table:
            w.writerow(t)

    lines = ["# Constrained-decoding diagnostic - VERDICT D", "",
             "246 held-out answers (base Qwen constrained: 62-game subset).", "",
             "| Model | Mode | Mean | Fail | Solved | Invalid | Repeat | <=3 | <=4 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        rep = t["repeated_guess_game_rate_pct"] or 0.0
        lines.append(
            "| {} | {} | {:.4f} | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | "
            "{:.1f} | {:.1f} |".format(
                t["model"], t["mode"], t["mean_failures_as_7"],
                t["failure_rate_pct"], t["solved_pct"], t["invalid_pct"],
                rep, t["pct_le3"], t["pct_le4"]))

    lines += ["", "## Terminal-state probe (k candidates remaining)", "",
              "| Model | k | n | top-1 | top-1 is candidate | median rank of answer |",
              "|---|---:|---:|---:|---:|---:|"]
    for mdl, per_k in TERMINAL.items():
        for k, r in sorted(per_k.items()):
            lines.append("| {} | {} | {} | {:.2f}% | {:.2f}% | {} |".format(
                mdl, k, r["n_states"], r["top1_accuracy_pct"],
                r["top1_is_candidate_pct"], r["median_rank_of_answer"]))
    lines += ["", "Chance top-1 = 0.0077% (1 of 12,972).", "",
              "## k=1 accuracy, seen vs unseen as a training target", "",
              "| Model | seen n | seen acc | unseen n | unseen acc |",
              "|---|---:|---:|---:|---:|"]
    for mdl, s in TERMINAL_SPLIT.items():
        lines.append("| {} | {} | {}% | {} | {}% |".format(
            mdl, s["seen_n"], s["seen_acc_pct"], s["unseen_n"],
            s["unseen_acc_pct"]))
    lines += ["", "~1.6 sigma at these sample sizes: suggestive, not established.",
              "", "## Verdict", "", "```"] + VERDICT["evidence"] + ["```"]

    open(os.path.join(OUT, "comparison.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    open(os.path.join(OUT, "RECONSTRUCTION_NOTES.md"), "w",
         encoding="utf-8").write(__doc__.strip() + "\n")

    shutil.make_archive(os.path.join(HERE, "results",
                                     "wordle_diagnostic_results"),
                        "zip", OUT)
    print("rebuilt:")
    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            p = os.path.join(root, f)
            print("  {:>7,} B  {}".format(os.path.getsize(p),
                                          os.path.relpath(p, OUT)))
    z = os.path.join(HERE, "results", "wordle_diagnostic_results.zip")
    print("\nzip: {:,} B  {}".format(os.path.getsize(z), z))


if __name__ == "__main__":
    main()
