"""
Unit tests for wordle_solver.py.

The duplicate-letter expectations below are hand-derived from the two-pass rule
(greens consume a letter's budget first, then yellows are assigned left to
right from what remains). Each is annotated with its derivation so a
disagreement points at either the code or the annotation, never at "the test
says so".

Run:  pytest -q tests/
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wordle_solver import (  # noqa: E402
    ALL_GREEN, N_PATTERNS, WORD_LEN,
    EntropySolver, ExpectedRemainingSolver, FeedbackMatrix, FrequencyModel,
    FrequencySolver, HybridSolver, MinimaxSolver, RandomSolver, SolverConfig,
    Vocabulary, build_feedback_matrix, code_to_pattern, feedback, feedback_code,
    filter_candidates, filter_history, is_consistent, load_artifacts,
    make_solver, normalize_words, pattern_to_code, play_game, save_artifacts,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# =============================================================================
# 1. Feedback: no duplicates involved (sanity)
# =============================================================================

@pytest.mark.parametrize("guess,answer,expected", [
    ("crane", "crane", "GGGGG"),          # identical
    # slate vs crane (c r a n e): a and e are both already in position.
    ("slate", "crane", "BBGBG"),
    ("plumb", "crane", "BBBBB"),          # no shared letters
    # nacre shares 'e' in position 4, so only four letters can be yellow.
    ("crane", "nacre", "YYYYG"),
    # A derangement of 'crane' (no letter in place) is all yellow. "ranec" is
    # not an English word, but `feedback` is a pure string function and does not
    # consult any vocabulary — which is exactly what this case pins down.
    ("crane", "ranec", "YYYYY"),
])
def test_feedback_basic(guess, answer, expected):
    assert feedback(guess, answer) == expected


def test_identical_words_are_all_green():
    for w in ("crane", "mamma", "eerie", "fluff"):
        assert feedback(w, w) == "GGGGG"
        assert feedback_code(w, w) == ALL_GREEN


# =============================================================================
# 2. Duplicate letters — the cases that break naive implementations
# =============================================================================

@pytest.mark.parametrize("guess,answer,expected,why", [
    # Repeated letter in the GUESS, single occurrence in the ANSWER.
    # speed vs abide: no greens. budget e=1. L->R: s,p grey; e[2] takes the
    # single e as yellow; e[3] finds the budget spent -> grey; d is yellow.
    ("speed", "abide", "BBYBY",
     "surplus guessed 'e' must come back grey after the first is used"),

    # Same pair reversed — feedback is NOT symmetric.
    # abide vs speed: no greens. budget s1 p1 e2 d1. a,b,i grey; d yellow;
    # e yellow.
    ("abide", "speed", "BBBYY",
     "feedback(a,b) and feedback(b,a) are different functions"),

    # GREEN consumes the budget before any yellow can claim it.
    # eerie vs eaten: e[0] green consumes one of the answer's two e's.
    # Remaining e budget = 1 -> e[1] yellow. e[4] then finds nothing -> grey.
    ("eerie", "eaten", "GYBBB",
     "green takes priority over later yellows for the same letter"),

    # More occurrences in guess than answer, with a green in play.
    # llama vs lucid: l[0] green spends the answer's only l, so l[1] is grey.
    ("llama", "lucid", "GBBBB",
     "second 'l' greys out because the green already consumed the budget"),

    # Green LATER in the word still consumes before an EARLIER yellow.
    # added vs dread: d[4] is green (both 'd'), spending one of dread's two d's.
    # Budget left d=1 -> d[1] yellow, d[2] grey. a[0] and e[3] yellow.
    ("added", "dread", "YYBYG",
     "a green at position 4 must consume budget before a yellow at position 1"),

    # Multiple yellows plus a green, repeated letter on both sides.
    # array vs rural: r[2] and a[3] green. Budget r=1, a=0.
    # a[0] grey (a spent by green), r[1] yellow, y grey.
    ("array", "rural", "BYGGB",
     "greens consume from both repeated letters before yellows are assigned"),

    # Repeated letter in guess AND answer, mixed outcome.
    # geese vs sheep: e[2] green. Budget e=1, s=1.
    # e[1] yellow, s[3] yellow, e[4] grey.
    ("geese", "sheep", "BYGYB",
     "one 'e' green, one yellow, one grey in a single word"),

    # Triple letter in the answer, three greens.
    # mummy vs mamma: m at 0,2,3 all green -> spends all three m's.
    ("mummy", "mamma", "GBGGB",
     "triple-letter answer with every occurrence matched"),

    # Triple in the guess against a triple in the answer, reversed.
    ("mamma", "mummy", "GBGGB",
     "three m's in guess, three in answer, all aligned"),

    # Repeated letter in ANSWER only.
    # abide vs eerie: no greens (a/e b/e i/r d/i e/e -> wait pos4 e==e green).
    # Recheck: a-e, b-e, i-r, d-i, e-e => e[4] GREEN. budget e=3-1=2.
    # a grey, b grey, i -> 'eerie' has one i -> yellow, d grey.
    ("abide", "eerie", "BBYBG",
     "answer has three e's; guess has one, matched green"),

    # All five guessed letters identical, answer has two.
    # aaaaa vs ababa: positions 0,2,4 green (a,a,a). budget a=3-3=0.
    # positions 1,3 grey.
    ("aaaaa", "ababa", "GBGBG",
     "degenerate all-same guess against a two-letter answer"),
])
def test_feedback_duplicate_letters(guess, answer, expected, why):
    assert feedback(guess, answer) == expected, why


def test_yellow_ordering_is_left_to_right():
    """When a letter is over-guessed, the LEFTMOST non-green occurrences win.

    'eecee' vs 'ceeee': greens where both hold the same letter.
      guess  e e c e e
      answer c e e e e
      pos0 e/c no, pos1 e/e GREEN, pos2 c/e no, pos3 e/e GREEN, pos4 e/e GREEN.
    Answer has four e's; three consumed by greens -> budget e=1.
    Non-green e occurrences, left to right: pos0 then (none other; pos3/4 green).
      pos0 -> yellow (rank 0 < 1).
    c at pos2: answer has one c, zero greens of c -> yellow.
    """
    assert feedback("eecee", "ceeee") == "YGYGG"


def test_letter_budget_never_exceeded():
    """For every letter, greens+yellows for it can never exceed its count in the answer."""
    from collections import Counter
    rng = random.Random(7)
    words = _load_words(DATA_DIR)["answers"]
    for _ in range(3000):
        g = rng.choice(words)
        a = rng.choice(words)
        pat = feedback(g, a)
        acount = Counter(a)
        marked = Counter()
        for ch, t in zip(g, pat):
            if t in ("G", "Y"):
                marked[ch] += 1
        for ch, k in marked.items():
            assert k <= acount[ch], f"{g}/{a} -> {pat}: over-marked {ch!r}"


def test_green_count_equals_positional_matches():
    rng = random.Random(11)
    words = _load_words(DATA_DIR)["answers"]
    for _ in range(3000):
        g, a = rng.choice(words), rng.choice(words)
        pat = feedback(g, a)
        expect = sum(1 for i in range(WORD_LEN) if g[i] == a[i])
        assert pat.count("G") == expect


def test_feedback_rejects_wrong_length():
    with pytest.raises(ValueError):
        feedback("abcd", "crane")
    with pytest.raises(ValueError):
        feedback("crane", "abcdef")


# =============================================================================
# 3. Encoding
# =============================================================================

def test_code_roundtrip_all_patterns():
    for code in range(N_PATTERNS):
        pat = code_to_pattern(code)
        assert len(pat) == WORD_LEN
        assert pattern_to_code(pat) == code


def test_all_green_code():
    assert ALL_GREEN == N_PATTERNS - 1 == 242
    assert code_to_pattern(ALL_GREEN) == "GGGGG"
    assert pattern_to_code("GGGGG") == ALL_GREEN
    assert pattern_to_code("BBBBB") == 0


def test_encoding_is_little_endian_by_position():
    # Green in position 0 only -> 2 * 3**0 == 2
    assert pattern_to_code("GBBBB") == 2
    # Green in position 4 only -> 2 * 3**4 == 162
    assert pattern_to_code("BBBBG") == 162
    # Yellow in position 1 only -> 1 * 3**1 == 3
    assert pattern_to_code("BYBBB") == 3


def test_pattern_aliases_accepted():
    assert pattern_to_code("gybb_") == pattern_to_code("GYBBB")
    assert pattern_to_code("21000") == pattern_to_code("GYBBB")


def test_pattern_rejects_junk():
    with pytest.raises(ValueError):
        pattern_to_code("GYBBX")
    with pytest.raises(ValueError):
        pattern_to_code("GYB")


def test_codes_always_in_range():
    rng = random.Random(3)
    words = _load_words(DATA_DIR)["answers"]
    for _ in range(2000):
        c = feedback_code(rng.choice(words), rng.choice(words))
        assert 0 <= c < N_PATTERNS


# =============================================================================
# 4. Candidate filtering
# =============================================================================

def test_answer_always_survives_its_own_feedback():
    """The true answer can never be eliminated by correct feedback."""
    rng = random.Random(5)
    words = _load_words(DATA_DIR)["answers"]
    for _ in range(500):
        a = rng.choice(words)
        g = rng.choice(words)
        surv = filter_candidates(words, g, feedback(g, a))
        assert a in surv


def test_filter_is_exactly_feedback_agreement():
    rng = random.Random(13)
    words = _load_words(DATA_DIR)["answers"]
    pool = rng.sample(words, 400)
    for _ in range(40):
        g, a = rng.choice(words), rng.choice(words)
        pat = feedback(g, a)
        surv = filter_candidates(pool, g, pat)
        brute = [w for w in pool if feedback(g, w) == pat]
        assert surv == brute
        assert all(is_consistent(w, g, pat) for w in surv)
        assert not any(is_consistent(w, g, pat) for w in pool if w not in set(surv))


def test_filter_history_matches_sequential_filtering():
    rng = random.Random(17)
    words = _load_words(DATA_DIR)["answers"]
    for _ in range(30):
        a = rng.choice(words)
        hist = [(rng.choice(words), "") for _ in range(3)]
        hist = [(g, feedback(g, a)) for g, _ in hist]
        folded = filter_history(words, hist)
        step = list(words)
        for g, p in hist:
            step = filter_candidates(step, g, p)
        assert folded == step
        assert a in folded


def test_filtering_partitions_the_candidate_set():
    """The 243 possible patterns partition the candidates: sizes must sum to n."""
    rng = random.Random(19)
    words = _load_words(DATA_DIR)["answers"]
    pool = rng.sample(words, 600)
    for g in rng.sample(words, 10):
        total = 0
        for code in range(N_PATTERNS):
            total += len(filter_candidates(pool, g, code_to_pattern(code)))
        assert total == len(pool)


# =============================================================================
# 5. Feedback matrix
# =============================================================================

@pytest.fixture(scope="module")
def small_vocab():
    w = _load_words(DATA_DIR)
    rng = random.Random(23)
    answers = sorted(rng.sample(w["answers"], 150))
    extra = sorted(rng.sample(w["guesses_extra"], 250))
    return Vocabulary(answers=answers, guesses=sorted(set(answers) | set(extra)))


@pytest.fixture(scope="module")
def small_fb(small_vocab):
    return FeedbackMatrix.build(small_vocab, chunk=64)


def test_matrix_shape_and_dtype(small_vocab, small_fb):
    assert small_fb.M.shape == (small_vocab.n_guesses, small_vocab.n_answers)
    assert small_fb.M.dtype == np.uint8
    assert small_fb.M.max() < N_PATTERNS


def test_matrix_matches_reference_implementation_exhaustively(small_vocab, small_fb):
    """Every single cell of the small matrix must equal the pure-Python reference."""
    for gi, g in enumerate(small_vocab.guesses):
        for ai, a in enumerate(small_vocab.answers):
            assert small_fb.M[gi, ai] == feedback_code(g, a), f"{g}/{a}"


def test_matrix_diagonal_is_all_green(small_vocab, small_fb):
    for a_i, w in enumerate(small_vocab.answers):
        g_i = small_vocab.guess_index[w]
        assert small_fb.M[g_i, a_i] == ALL_GREEN


def test_matrix_chunking_is_invariant(small_vocab):
    a = build_feedback_matrix(small_vocab.guesses, small_vocab.answers, chunk=7)
    b = build_feedback_matrix(small_vocab.guesses, small_vocab.answers, chunk=1024)
    assert np.array_equal(a, b)


def test_matrix_filter_matches_python_filter(small_vocab, small_fb):
    rng = random.Random(29)
    for _ in range(200):
        a = rng.choice(small_vocab.answers)
        g = rng.choice(small_vocab.guesses)
        code = feedback_code(g, a)
        idx = np.arange(small_vocab.n_answers, dtype=np.int32)
        got = sorted(small_vocab.answers[i] for i in small_fb.filter_indices(idx, g, code))
        want = sorted(filter_candidates(small_vocab.answers, g, code_to_pattern(code)))
        assert got == want


def test_partition_stats_match_direct_computation(small_vocab, small_fb):
    """entropy / expected / worst must agree with a straightforward Counter version."""
    from collections import Counter
    import math
    rng = random.Random(31)
    cand = np.array(sorted(rng.sample(range(small_vocab.n_answers), 80)), dtype=np.int32)
    pool = np.array(sorted(rng.sample(range(small_vocab.n_guesses), 60)), dtype=np.int32)
    stats = small_fb.partition_stats(pool, cand, chunk=13)
    n = len(cand)
    for k, gi in enumerate(pool):
        g = small_vocab.guesses[gi]
        buckets = Counter(feedback_code(g, small_vocab.answers[i]) for i in cand)
        sizes = list(buckets.values())
        assert sum(sizes) == n
        ent = -sum((s / n) * math.log2(s / n) for s in sizes)
        exp = sum(s * s for s in sizes) / n
        assert stats["entropy"][k] == pytest.approx(ent, abs=1e-9)
        assert stats["expected_remaining"][k] == pytest.approx(exp, abs=1e-9)
        assert stats["worst_case"][k] == max(sizes)


def test_entropy_bounds(small_vocab, small_fb):
    """0 <= H <= log2(n), and H == 0 exactly when the guess splits nothing."""
    import math
    cand = np.arange(small_vocab.n_answers, dtype=np.int32)
    pool = np.arange(small_vocab.n_guesses, dtype=np.int32)
    st = small_fb.partition_stats(pool, cand)
    n = len(cand)
    assert st["entropy"].min() >= -1e-12
    assert st["entropy"].max() <= math.log2(n) + 1e-9
    # expected remaining is at least 1 and at most n
    assert st["expected_remaining"].min() >= 1.0 - 1e-9
    assert st["expected_remaining"].max() <= n + 1e-9
    # worst case bucket is between ceil(n/243) and n
    assert st["worst_case"].max() <= n
    assert st["worst_case"].min() >= 1


def test_minimax_is_argmin_of_worst_case(small_vocab, small_fb):
    cand = np.arange(small_vocab.n_answers, dtype=np.int32)
    pool = np.arange(small_vocab.n_guesses, dtype=np.int32)
    st = small_fb.partition_stats(pool, cand)
    cfg = SolverConfig(max_guesses=6, guess_pool="full", opening_guess=None)
    chosen = MinimaxSolver(small_fb, cfg).choose(cand, 1)
    best = st["worst_case"].min()
    assert st["worst_case"][small_vocab.guess_index[chosen]] == best


def test_entropy_is_argmax_of_entropy(small_vocab, small_fb):
    cand = np.arange(small_vocab.n_answers, dtype=np.int32)
    pool = np.arange(small_vocab.n_guesses, dtype=np.int32)
    st = small_fb.partition_stats(pool, cand)
    cfg = SolverConfig(max_guesses=6, guess_pool="full", opening_guess=None)
    chosen = EntropySolver(small_fb, cfg).choose(cand, 1)
    assert st["entropy"][small_vocab.guess_index[chosen]] == pytest.approx(
        st["entropy"].max(), abs=1e-9
    )


# =============================================================================
# 6. Vocabulary handling
# =============================================================================

def test_normalization_cleans_and_reports():
    # 'élite' is 5 letters and must survive as 'elite'; 'café' is only 4 and
    # must be length-rejected even though its accent strips cleanly.
    raw = "Crane\n crane \nCRANE\nélite\ncafé\nabc\nslate,plumb\n\n  \ntoolongword\n"
    words, stats = normalize_words(raw)
    assert "crane" in words
    assert words.count("crane") == 1          # duplicates collapsed
    assert stats["duplicates"] >= 2
    assert stats["uppercased"] >= 2
    assert "abc" not in words                 # wrong length rejected
    assert stats["bad_length"] >= 2
    assert "slate" in words and "plumb" in words   # comma-separated handled
    assert "elite" in words                   # accent stripped, length 5 -> kept
    assert "cafe" not in words                # accent stripped but length 4 -> dropped


def test_vocabulary_requires_answers_to_be_legal_guesses():
    with pytest.raises(ValueError):
        Vocabulary(answers=["crane", "slate"], guesses=["crane"])


def test_answer_positions_are_correct():
    v = Vocabulary(answers=["crane", "slate"], guesses=["aahed", "crane", "slate"])
    pos = v.answer_positions()
    assert [v.guesses[i] for i in pos] == ["crane", "slate"]


def test_guess_pool_inference_disjoint_lists(tmp_path):
    from wordle_solver import load_vocabulary
    (tmp_path / "a.txt").write_text("crane\nslate\n", encoding="utf-8")
    (tmp_path / "g.txt").write_text("aahed\nzymic\n", encoding="utf-8")
    v = load_vocabulary(str(tmp_path / "a.txt"), str(tmp_path / "g.txt"))
    assert v.n_answers == 2
    assert v.n_guesses == 4        # inferred "extras" convention


def test_guess_pool_inference_superset_list(tmp_path):
    from wordle_solver import load_vocabulary
    (tmp_path / "a.txt").write_text("crane\nslate\n", encoding="utf-8")
    (tmp_path / "g.txt").write_text("crane\nslate\naahed\n", encoding="utf-8")
    v = load_vocabulary(str(tmp_path / "a.txt"), str(tmp_path / "g.txt"))
    assert v.n_answers == 2
    assert v.n_guesses == 3        # inferred "already complete" convention


# =============================================================================
# 7. Solvers play legal, terminating games
# =============================================================================

@pytest.mark.parametrize("strategy", ["random", "frequency", "entropy",
                                      "expected", "minimax", "hybrid"])
def test_every_solver_finishes_and_never_cheats(strategy, small_vocab, small_fb):
    """Guesses must be legal words; a solved game's last pattern must be GGGGG."""
    cfg = SolverConfig(max_guesses=8, seed=101, guess_pool="full")
    solver = make_solver(strategy, small_fb, cfg)
    rng = random.Random(37)
    for a in rng.sample(small_vocab.answers, 12):
        r = play_game(solver, a)
        assert all(g in small_vocab.guess_index for g in r.guesses)
        assert len(r.guesses) == len(r.patterns) == len(r.candidates_remaining)
        # candidate counts must be non-increasing and never zero
        assert all(c >= 1 for c in r.candidates_remaining)
        assert r.candidates_remaining == sorted(r.candidates_remaining, reverse=True)
        if r.solved:
            assert r.patterns[-1] == "GGGGG"
            assert r.guesses[-1] == a
        # every recorded pattern must be reproducible from the answer
        for g, p in zip(r.guesses, r.patterns):
            assert feedback(g, a) == p


def test_candidate_count_matches_independent_filtering(small_vocab, small_fb):
    """The driver's candidate counts must match pure-Python history filtering."""
    cfg = SolverConfig(max_guesses=8, seed=7, guess_pool="full")
    solver = make_solver("entropy", small_fb, cfg)
    rng = random.Random(41)
    for a in rng.sample(small_vocab.answers, 8):
        r = play_game(solver, a)
        for k in range(len(r.guesses)):
            hist = list(zip(r.guesses[:k + 1], r.patterns[:k + 1]))
            expect = filter_history(small_vocab.answers, hist)
            assert len(expect) == r.candidates_remaining[k]


def test_deterministic_solvers_are_reproducible(small_vocab, small_fb):
    cfg = SolverConfig(max_guesses=8, seed=1, guess_pool="full")
    for strategy in ("entropy", "minimax", "expected", "frequency", "hybrid"):
        a = "crane" if "crane" in small_vocab.answer_index else small_vocab.answers[0]
        r1 = play_game(make_solver(strategy, small_fb, cfg), a)
        r2 = play_game(make_solver(strategy, small_fb, cfg), a)
        assert r1.guesses == r2.guesses


def test_random_solver_is_seed_reproducible(small_vocab, small_fb):
    a = small_vocab.answers[3]
    cfg = SolverConfig(max_guesses=8, seed=999)
    r1 = play_game(RandomSolver(small_fb, cfg), a)
    r2 = play_game(RandomSolver(small_fb, cfg), a)
    assert r1.guesses == r2.guesses
    r3 = play_game(RandomSolver(small_fb, SolverConfig(max_guesses=8, seed=1000)), a)
    assert isinstance(r3.guesses, list)   # different seed may or may not differ


def test_last_guess_is_forced_to_be_a_candidate(small_vocab, small_fb):
    """On the final turn a non-candidate probe is a guaranteed loss."""
    cfg = SolverConfig(max_guesses=3, seed=2, guess_pool="full",
                       force_candidate_on_last_guess=True)
    solver = make_solver("entropy", small_fb, cfg)
    rng = random.Random(43)
    for a in rng.sample(small_vocab.answers, 20):
        r = play_game(solver, a)
        if len(r.guesses) == 3:
            # the 3rd guess had to be drawn from the surviving candidates
            hist = list(zip(r.guesses[:2], r.patterns[:2]))
            surv = filter_history(small_vocab.answers, hist)
            assert r.guesses[2] in surv


def test_frequency_model_does_not_reward_useless_repeats():
    """A word repeating a common letter must not outscore one with fresh letters."""
    words = _load_words(DATA_DIR)["answers"]
    m = FrequencyModel.fit(words)
    # 'eerie' repeats e twice more; 'arose' has five distinct common letters.
    assert m.score("arose") > m.score("eerie")
    # coverage term counts distinct letters only
    cov_repeat = sum(m.letter_prob[c] for c in set("eerie"))
    assert cov_repeat == pytest.approx(
        m.letter_prob["e"] + m.letter_prob["r"] + m.letter_prob["i"]
    )


def test_solver_rejects_answer_outside_answer_list(small_fb):
    solver = make_solver("entropy", small_fb, SolverConfig())
    with pytest.raises(ValueError):
        play_game(solver, "zzzzz")


# =============================================================================
# 8. Artifact save / load round-trip
# =============================================================================

def test_artifact_roundtrip(tmp_path, small_vocab, small_fb):
    model = FrequencyModel.fit(small_vocab.answers)
    cfg = SolverConfig(strategy="hybrid", seed=4242, max_guesses=6)
    d = str(tmp_path / "artifacts")
    paths = save_artifacts(d, small_vocab, small_fb, model, cfg)
    for p in paths.values():
        assert os.path.exists(p), p

    b = load_artifacts(d)
    assert b.vocab.answers == small_vocab.answers
    assert b.vocab.guesses == small_vocab.guesses
    assert np.array_equal(np.asarray(b.fb.M), small_fb.M)
    assert b.config.seed == 4242
    assert b.config.strategy == "hybrid"
    assert b.model.letter_prob == model.letter_prob
    assert b.metadata["n_answers"] == small_vocab.n_answers
    assert b.metadata["matrix_shape"] == list(small_fb.M.shape)
    assert b.metadata["all_green_code"] == ALL_GREEN


def test_reloaded_bundle_plays_identical_games(tmp_path, small_vocab, small_fb):
    model = FrequencyModel.fit(small_vocab.answers)
    cfg = SolverConfig(strategy="entropy", seed=5, max_guesses=8, guess_pool="full")
    d = str(tmp_path / "art")
    save_artifacts(d, small_vocab, small_fb, model, cfg)
    b = load_artifacts(d)

    a = small_vocab.answers[10]
    r_direct = play_game(make_solver("entropy", small_fb, cfg), a)
    r_loaded = play_game(b.solver("entropy"), a)
    assert r_direct.guesses == r_loaded.guesses
    assert r_direct.patterns == r_loaded.patterns


def test_resaving_a_memmapped_bundle_does_not_fail(tmp_path, small_vocab, small_fb):
    """Reload-then-resave must work even though the matrix file is memory-mapped.

    On Windows an open mapping locks the file, so a naive np.save over it raises
    OSError(EINVAL). save_artifacts must detect that the mapping already points
    at the destination and skip the redundant rewrite.
    """
    model = FrequencyModel.fit(small_vocab.answers)
    d = str(tmp_path / "art")
    save_artifacts(d, small_vocab, small_fb, model, SolverConfig())

    b = load_artifacts(d, mmap=True)
    assert isinstance(b.fb.M, np.memmap)

    # This is the notebook's build -> save -> reload -> save-again flow.
    save_artifacts(d, b.vocab, b.fb, b.model, SolverConfig(seed=777))

    b2 = load_artifacts(d, mmap=True)
    assert np.array_equal(np.asarray(b2.fb.M), small_fb.M), "matrix corrupted by resave"
    assert b2.config.seed == 777, "config did not update on resave"


def test_resave_while_a_separate_memmap_holds_the_file(tmp_path, small_vocab, small_fb):
    """Save an in-memory matrix while a DIFFERENT array memory-maps the destination.

    This is the notebook's actual Section 9 -> Section 20 flow: the freshly built
    matrix is saved, reloaded as a bundle (creating a memmap), and then saved
    again from the original in-memory array. On Windows the live mapping locks
    the file, so save_artifacts must recognise the contents are unchanged and
    skip the rewrite rather than raising OSError(EINVAL).
    """
    model = FrequencyModel.fit(small_vocab.answers)
    d = str(tmp_path / "art")
    save_artifacts(d, small_vocab, small_fb, model, SolverConfig())

    bundle = load_artifacts(d, mmap=True)          # holds an open mapping
    assert isinstance(bundle.fb.M, np.memmap)
    assert not isinstance(small_fb.M, np.memmap)   # the array we re-save is in RAM

    save_artifacts(d, small_vocab, small_fb, model, SolverConfig(seed=31337))

    reloaded = load_artifacts(d, mmap=False)
    assert np.array_equal(reloaded.fb.M, small_fb.M), "matrix corrupted"
    assert reloaded.config.seed == 31337, "config did not update"
    assert bundle.metadata["n_answers"] == small_vocab.n_answers


def test_load_artifacts_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(str(tmp_path / "nope"))


def test_memmap_load_is_readonly_but_usable(tmp_path, small_vocab, small_fb):
    model = FrequencyModel.fit(small_vocab.answers)
    d = str(tmp_path / "art")
    save_artifacts(d, small_vocab, small_fb, model, SolverConfig())
    b = load_artifacts(d, mmap=True)
    assert isinstance(b.fb.M, np.memmap)
    idx = np.arange(small_vocab.n_answers, dtype=np.int32)
    g = small_vocab.guesses[0]
    code = feedback_code(g, small_vocab.answers[0])
    out = b.fb.filter_indices(idx, g, code)
    assert small_vocab.answers[0] in [small_vocab.answers[i] for i in out]


# =============================================================================
# helpers
# =============================================================================

_CACHE = {}


def _load_words(data_dir):
    if "w" in _CACHE:
        return _CACHE["w"]
    ans_p = os.path.join(data_dir, "wordle_answers.txt")
    gue_p = os.path.join(data_dir, "wordle_allowed_guesses.txt")
    if not os.path.exists(ans_p):
        pytest.skip(f"dataset not found at {data_dir}")
    with open(ans_p, encoding="utf-8") as fh:
        answers = [w for w in fh.read().split() if w]
    with open(gue_p, encoding="utf-8") as fh:
        extra = [w for w in fh.read().split() if w]
    _CACHE["w"] = {"answers": answers, "guesses_extra": extra}
    return _CACHE["w"]
