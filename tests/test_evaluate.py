"""Chapter 9: resolution, slices, judge guards, and the promotion gate."""

import pytest

from rightsize.evaluate import (
    PairwiseVerdict,
    PromotionGate,
    SlicedEvaluation,
    SliceResult,
    consistent_only,
    contamination_score,
    ngrams,
    position_bias_rate,
    required_trials,
    resolution_floor,
    wilson_interval,
)


# --- Pass 1: the interval ---------------------------------------------------


def test_the_published_intervals_at_seventy_percent():
    """The exact numbers @tbl-intervals prints."""
    expected = {
        40: (0.546, 0.819),
        100: (0.604, 0.781),
        200: (0.633, 0.759),
        1000: (0.671, 0.728),
    }
    for trials, (low, high) in expected.items():
        got = wilson_interval(round(0.70 * trials), trials)
        assert got[0] == pytest.approx(low, abs=0.001)
        assert got[1] == pytest.approx(high, abs=0.001)


def test_the_opening_scenes_two_point_difference_overlaps_heavily():
    """72% and 70% on 200 examples are the same number in different
    clothes, which is the chapter's opening."""
    candidate = wilson_interval(round(0.72 * 200), 200)
    incumbent = wilson_interval(round(0.70 * 200), 200)
    assert candidate[0] < incumbent[1]
    assert incumbent[0] < candidate[1]


def test_intervals_narrow_as_the_sample_grows():
    widths = [
        wilson_interval(round(0.70 * n), n)[1] - wilson_interval(round(0.70 * n), n)[0]
        for n in (40, 100, 200, 1000, 5000)
    ]
    assert widths == sorted(widths, reverse=True)


def test_wilson_stays_inside_zero_and_one_near_the_edges():
    """Where the normal approximation puts the bound above 1.0."""
    low, high = wilson_interval(19, 20)
    assert 0.0 <= low <= high <= 1.0
    assert high < 1.0


def test_a_perfect_score_still_has_a_lower_bound_below_one():
    low, high = wilson_interval(20, 20)
    assert low < 1.0
    assert high == pytest.approx(1.0, abs=0.001)


def test_zero_trials_is_zero_rather_than_a_division_error():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# --- Pass 2: the resolution floor -------------------------------------------


def test_the_published_sample_size_requirements():
    """@tbl-floor's right-hand column."""
    assert required_trials(0.70, 0.02) == 8077
    assert required_trials(0.70, 0.05) == 1248
    assert required_trials(0.70, 0.10) == 291
    assert required_trials(0.70, 0.15) == 118


def test_the_published_floors_by_task_set_size():
    """@tbl-floor's left-hand column."""
    assert resolution_floor(100) == pytest.approx(0.161, abs=0.001)
    assert resolution_floor(200) == pytest.approx(0.119, abs=0.001)
    assert resolution_floor(500) == pytest.approx(0.078, abs=0.001)
    assert resolution_floor(1000) == pytest.approx(0.056, abs=0.001)


def test_two_hundred_examples_cannot_resolve_two_points():
    """The chapter's opening, stated as the arithmetic that caused it."""
    assert resolution_floor(200) > 0.02


def test_halving_the_effect_roughly_quadruples_the_examples():
    """The inverse-square relationship, which is why 'collect more
    examples' stops being practical advice quickly."""
    ten = required_trials(0.70, 0.10)
    five = required_trials(0.70, 0.05)
    assert five / ten == pytest.approx(4.0, abs=0.35)


def test_an_impossible_improvement_raises_rather_than_returning_nonsense():
    with pytest.raises(ValueError):
        required_trials(0.70, 0.40)
    with pytest.raises(ValueError):
        required_trials(0.0, 0.10)


def test_a_zero_size_set_resolves_nothing():
    assert resolution_floor(0) == 1.0


# --- Pass 3: slices ---------------------------------------------------------


def _sliced(*pairs) -> SlicedEvaluation:
    return SlicedEvaluation(
        [SliceResult(name, hits, n) for name, hits, n in pairs]
    )


def test_the_worst_slice_is_found_not_the_mean():
    evaluation = _sliced(("common", 900, 1000), ("rare", 10, 40))
    assert evaluation.worst.name == "rare"
    assert evaluation.overall > 0.85


def test_a_slice_carries_its_own_floor_not_the_whole_sets():
    """A 40-example slice resolves what 40 examples resolve."""
    small = SliceResult("rare", 28, 40)
    large = SliceResult("common", 700, 1000)
    assert small.floor > large.floor * 2


def test_a_slice_collapse_is_caught_while_the_aggregate_improves():
    """Chapter 5's pruned model: up overall, broken on the tail."""
    before = _sliced(("common", 850, 1000), ("rare", 30, 40))
    after = _sliced(("common", 920, 1000), ("rare", 12, 40))
    assert after.overall > before.overall
    problems = after.regressions_against(before)
    assert len(problems) == 1
    assert "rare" in problems[0]


def test_a_drop_smaller_than_the_slices_floor_is_not_reported():
    """Otherwise a small slice generates noise alerts forever."""
    before = _sliced(("rare", 29, 40))
    after = _sliced(("rare", 28, 40))
    assert after.regressions_against(before) == []


def test_a_new_slice_with_no_baseline_is_skipped_not_crashed():
    before = _sliced(("common", 850, 1000))
    after = _sliced(("common", 850, 1000), ("brand-new", 5, 40))
    assert after.regressions_against(before) == []


def test_an_empty_evaluation_is_zero_rather_than_a_division_error():
    assert SlicedEvaluation([]).overall == 0.0
    assert SlicedEvaluation([]).worst is None


# --- Pass 4: the judge guard ------------------------------------------------


def test_a_stable_judge_has_no_position_bias():
    verdicts = [PairwiseVerdict(str(i), "a", "a") for i in range(10)]
    assert position_bias_rate(verdicts) == 0.0


def test_flipping_on_order_is_counted():
    verdicts = [
        PairwiseVerdict("1", "a", "a"),
        PairwiseVerdict("2", "a", "b"),
        PairwiseVerdict("3", "b", "a"),
        PairwiseVerdict("4", "tie", "tie"),
    ]
    assert position_bias_rate(verdicts) == pytest.approx(0.5)


def test_unstable_comparisons_can_be_dropped_and_counted():
    verdicts = [
        PairwiseVerdict("1", "a", "a"),
        PairwiseVerdict("2", "a", "b"),
    ]
    kept = consistent_only(verdicts)
    assert len(kept) == 1
    assert len(verdicts) - len(kept) == 1


def test_no_verdicts_is_zero_bias_rather_than_an_error():
    assert position_bias_rate([]) == 0.0


# --- Pass 5: the promotion gate ---------------------------------------------


INCUMBENT = _sliced(("common", 700, 1000), ("rare", 28, 40))


def test_a_genuine_improvement_above_the_floor_passes():
    candidate = _sliced(("common", 850, 1000), ("rare", 34, 40))
    gate = PromotionGate(min_trials=100)
    assert gate.blockers(candidate, INCUMBENT) == []


def test_a_gain_below_the_resolution_floor_is_blocked():
    """The chapter's opening, fired before the deployment instead of
    three weeks after it."""
    candidate = _sliced(("common", 715, 1000), ("rare", 29, 40))
    gate = PromotionGate(min_trials=100)
    problems = gate.blockers(candidate, INCUMBENT)
    assert any("resolution floor" in p for p in problems)


def test_a_slice_regression_blocks_even_when_the_aggregate_rises():
    candidate = _sliced(("common", 950, 1000), ("rare", 8, 40))
    problems = PromotionGate().blockers(candidate, INCUMBENT)
    assert any("rare" in p for p in problems)


def test_an_undersized_task_set_is_blocked():
    candidate = _sliced(("common", 30, 40))
    problems = PromotionGate(min_trials=500).blockers(candidate, INCUMBENT)
    assert any("below the 500" in p for p in problems)


def test_an_unreliable_judge_blocks_the_release():
    candidate = _sliced(("common", 850, 1000), ("rare", 34, 40))
    problems = PromotionGate(min_trials=100).blockers(
        candidate, INCUMBENT, position_bias=0.18
    )
    assert any("flipped on" in p for p in problems)


def test_the_gate_does_not_block_an_aggregate_regression_by_itself():
    """Chapter 1's envelope permits trading quality for memory or latency.
    A gate forbidding any aggregate drop forbids the trades this book is
    about; what it forbids is an unmeasurable claim and a slice collapse."""
    candidate = _sliced(("common", 640, 1000), ("rare", 28, 40))
    gate = PromotionGate(min_trials=100, max_slice_regression=0.10)
    assert candidate.overall < INCUMBENT.overall
    assert gate.blockers(candidate, INCUMBENT) == []


# --- Contamination ----------------------------------------------------------


PASSAGE = (
    "the quick brown fox jumps over the lazy dog and then runs away fast today"
)


def test_a_verbatim_example_scores_fully_contaminated():
    assert contamination_score(PASSAGE, ngrams(PASSAGE)) == 1.0


def test_a_paraphrase_is_completely_invisible():
    """The asymmetry that matters: a high score is strong evidence of
    leakage, a low score is weak evidence of cleanliness."""
    paraphrase = (
        "a fast auburn fox leapt above the sleepy hound and afterwards "
        "fled quickly"
    )
    assert contamination_score(paraphrase, ngrams(PASSAGE)) == 0.0


def test_text_shorter_than_the_window_produces_no_ngrams():
    assert ngrams("too short", 13) == set()
    assert contamination_score("too short", ngrams(PASSAGE)) == 0.0


def test_a_smaller_window_matches_more_coincidentally():
    """Why the usable range starts around 8: ordinary English repeats
    short spans constantly."""
    other = "the quick brown fox was seen in a completely different document"
    assert len(ngrams(other, 4) & ngrams(PASSAGE, 4)) > 0
    assert len(ngrams(other, 13) & ngrams(PASSAGE, 13)) == 0
