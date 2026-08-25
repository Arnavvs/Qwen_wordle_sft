# Playable benchmark demo

A single self-contained HTML page: play twenty real games from the 246-answer
held-out set, and see how every solver in the project scored on the same words.
Any solver's game can be replayed guess by guess.

**Live:** https://claude.ai/code/artifact/256bfce4-e519-4f56-bd4a-047bc7c3a9a7

```bash
python phase12_demo/build_demo_data.py   # pick the 20 games, gather every solver's play
python phase12_demo/make_demo.py         # inline the data -> index.html
```

`index.html` is generated. Edit `template.html`, not the output.

## Why these twenty

Stratified by how hard the SFT model found them, so the set spans a game it
solved in two through one it failed outright. Twenty random games would be
almost all threes and fours and would separate nothing. Selection is
deterministic (`SEED = 20260826`).

## Where the numbers come from

Model rows are the **exact per-game records** the published paired tests were
computed from — `results/phase10/` and `results/phase11/` — so the page cannot
disagree with the reported means. Classical solvers had no per-game record on
disk, so they are replayed against the same answers through the project's own
`play_game`.

## Correctness

The page reimplements the feedback rule in JavaScript, which is the one place a
demo like this can quietly lie. It is checked against the Python engine on 408
vectors including adversarial duplicate-letter cases (`ALLEY`/`LLAMA`,
`SASSY`/`BASIS`, `ARRAY`/`RADAR`): **408/408 exact**. Every embedded game is
also re-validated — legal guesses, scores matching the guess sequence, no early
wins, nothing over six guesses.

## A caveat the page states itself

Twenty games cannot separate these solvers: the standard error on a 20-game
mean is ~0.2 guesses, wider than most gaps in the table. The `Mean, 246` column
is the real measurement. Beating the model over twenty games is fun, not a
finding.
