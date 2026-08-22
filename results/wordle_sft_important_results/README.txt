================================================================================
WORDLE SFT RESULTS - EXECUTIVE SUMMARY
================================================================================

Evaluation Set: 246 held-out answers
Model: Qwen/Qwen2.5-0.5B-Instruct
Generated: 2026-08-17 22:27:19 UTC

KEY FINDINGS:
----------------------------------------
Best SFT Model: qwen_tree_salet
  Mean Score: 6.1748 (lower is better)
  Failure Rate: 76.4%
  Solved in <=3: 13.4%

Classical Baselines (for comparison):
  random       mean: 4.0203
  frequency    mean: 3.7927
  entropy      mean: 3.4431

OPENING WORD MEMORIZATION:
  entropy: SOARE used 100.0% of games
  tree_soare: SOARE used 100.0% of games
  tree_salet: SALET used 100.0% of games

POLICY TRANSFER (Disagreement Analysis):
  entropy: entropy 90.1% | tree 0.0% | neither 9.9%
  tree_soare: entropy 0.0% | tree 89.9% | neither 10.1%

FILES INCLUDED:
  - results.json        : Complete results (all metrics)
  - model_summary.csv   : SFT model results in spreadsheet format
  - baseline_summary.csv: Baseline results in spreadsheet format
  - adapter_configs/    : Training configs for each adapter
  - README.txt          : This file
  - wordle_sft_notebook.ipynb: Full notebook (if available)
