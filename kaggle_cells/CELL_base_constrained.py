# =============================================================================
# base Qwen, CONSTRAINED, on an evenly-spaced subset of the held-out answers.
#
# Why a subset: pruning is exact but its bound is the first-token log-prob.
# A trained model concentrates mass on a few first tokens, so one chunk of 512
# settles the argmax (you saw "avg 1.0-1.4 chunks/decision"). Untrained Qwen has
# a diffuse next-token distribution, so the bound is flat and it must score all
# 26 chunks -- ~20x slower, ~4h for 246 games. The control does not need 246
# games to do its job.
#
# The protocol is otherwise IDENTICAL to the full runs: same prompts, same
# scorer, same exact argmax over the same 12,972 legal words, no candidate list,
# no answer. Only the number of games differs, and the row records it.
#
# stride 4 -> 62 games, ~65 min. Raise the stride if you have less time:
#   stride 6 -> 41 games ~45 min      stride 8 -> 31 games ~35 min
# =============================================================================
BASE_GAME_STRIDE = 4
LOG_EVERY_DECISIONS = 20

_answers = VAL_ANSWERS[::BASE_GAME_STRIDE]
print(f"base Qwen [constrained] on {len(_answers)} of {len(VAL_ANSWERS)} "
      f"held-out answers (stride {BASE_GAME_STRIDE})")

_sc = build_scorer()
_model = load_base_control()
if not _SCORER_VERIFIED:
    verify_scorer(_model)

_games = [GameState(a, VOCAB.n_answers, mode="constrained") for a in _answers]
_chunks, _done, _t0 = [], 0, time.perf_counter()
# rough total: the base model rarely solves, so assume most games use all turns
_total_est = len(_games) * MAX_GUESSES

with torch.no_grad():
    for _turn in range(1, MAX_GUESSES + 1):
        _active = [g for g in _games if not g.done]
        if not _active:
            break
        for _g in _active:
            _p = _g.prompt(_turn, MAX_GUESSES)
            _banned = (list(dict.fromkeys(_g.guesses))
                       if BAN_REPEATS_IN_CONSTRAINED else None)
            _w, _s, _nch, _margin = _sc.argmax(_model, _p, banned=_banned,
                                               prune=CONSTRAINED_PRUNE)
            _chunks.append(_nch)
            _apply(_g, _w, "ok", None, f"<constrained:{_s:.3f}>",
                   _turn, MAX_GUESSES, margin=_margin)
            _done += 1
            if LOG_EVERY_DECISIONS and _done % LOG_EVERY_DECISIONS == 0:
                _el = time.perf_counter() - _t0
                print(f"    turn {_turn}: {_done} decisions, {_el:.0f}s "
                      f"elapsed, ~{_el/_done*max(_total_est-_done,0):.0f}s left, "
                      f"avg {np.mean(_chunks):.1f} chunks/decision", flush=True)
        _nd = sum(1 for g in _games if g.done)
        print(f"  turn {_turn}: {_nd}/{len(_games)} finished "
              f"({time.perf_counter()-_t0:.0f}s)", flush=True)

_row = score_games(_games, f"base_qwen (n={len(_games)} subset)")
_row["avg_chunks_per_decision"] = round(float(np.mean(_chunks)), 2)
_row["max_chunks_per_decision"] = int(max(_chunks))
_row["unpruned_chunks"] = -(-len(LEGAL_GUESSES) // CONSTRAINED_CHUNK)
_row["subset_stride"] = BASE_GAME_STRIDE
_row["eval_seconds"] = round(time.perf_counter() - _t0, 1)

# Store under the key the downstream cells expect.
GAMES_ALL.setdefault("constrained", {})["base_qwen"] = _games
EVAL_ALL.setdefault("constrained", {})["base_qwen"] = _row

print(f"\n  mean={_row['mean_failures_as_7']:.4f}  "
      f"fail={_row['failure_rate_pct']:.1f}%  "
      f"solved={100-_row['failure_rate_pct']:.1f}%  "
      f"invalid={_row['invalid_format_rate_pct']+_row['invalid_word_rate_pct']:.1f}%  "
      f"repeat={_row['repeated_guess_game_rate_pct']:.1f}%  "
      f"({_row['eval_seconds']:.0f}s)")
print(f"  avg {_row['avg_chunks_per_decision']} chunks/decision "
      f"(vs {_row['unpruned_chunks']} unpruned) -- confirms pruning cannot help "
      f"an untrained model")
print("\nNOTE: this row is on a subset. It is directly comparable in protocol")
print("but noisier than the 246-game rows; the n is recorded in the results.")

del _model; torch.cuda.empty_cache()
