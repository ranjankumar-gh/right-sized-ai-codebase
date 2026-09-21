"""Chapter 15: the size-escalation boundary and the limit check."""

import pytest

from rightsize.limits import LimitCheck, SizeResponse

NEARLY_FLAT = [(1.0, 0.700), (3.0, 0.702), (7.0, 0.705)]
SHALLOW = [(1.0, 0.700), (3.0, 0.710), (7.0, 0.720)]
MODEST = [(1.0, 0.620), (3.0, 0.665), (7.0, 0.700)]
STEEP = [(1.0, 0.450), (3.0, 0.620), (7.0, 0.780)]


# --- Pass 1: the size response ----------------------------------------------


def test_the_published_verdict_table():
    """@tbl-size-response."""
    expected = {
        ("shallow", 200): "unresolvable",
        ("shallow", 5000): "unresolvable",
        ("shallow", 50000): "capability-bound",
        ("nearly flat", 200): "unresolvable",
        ("nearly flat", 50000): "size-insensitive",
        ("steep", 200): "capability-bound",
        ("modest", 200): "unresolvable",
        ("modest", 5000): "capability-bound",
    }
    shapes = {"shallow": SHALLOW, "nearly flat": NEARLY_FLAT,
              "modest": MODEST, "steep": STEEP}
    for (name, trials), verdict in expected.items():
        assert SizeResponse(shapes[name], trials).verdict == verdict, (name, trials)


def test_a_steep_response_is_visible_even_on_a_small_task_set():
    """The useful asymmetry: when capability genuinely binds you do not
    need a large evaluation to see it."""
    assert SizeResponse(STEEP, 200).verdict == "capability-bound"
    assert SizeResponse(STEEP, 200).span > SizeResponse(STEEP, 200).floor * 2


def test_an_eight_point_gain_on_two_hundred_examples_is_noise():
    """The common case, and the one that gets larger models approved."""
    response = SizeResponse(MODEST, 200)
    assert response.span == pytest.approx(0.08, abs=0.001)
    assert response.verdict == "unresolvable"


def test_the_same_measurement_is_settled_by_a_few_hundred_more_examples():
    """547, not five thousand: an afternoon of labelling."""
    assert SizeResponse(MODEST, 200).examples_needed == 547
    assert SizeResponse(MODEST, 5000).verdict == "capability-bound"


def test_the_same_span_changes_verdict_as_the_task_set_grows():
    """'Cannot tell' and 'bounded below a point' are different answers.
    The first leaves the decision open; the second closes it."""
    assert SizeResponse(NEARLY_FLAT, 200).verdict == "unresolvable"
    assert SizeResponse(NEARLY_FLAT, 5000).verdict == "unresolvable"
    assert SizeResponse(NEARLY_FLAT, 50000).verdict == "size-insensitive"


def test_a_bounded_negligible_effect_is_an_answer_not_a_failure():
    """The floor has dropped below the span worth caring about, so the
    effect is established as too small to want."""
    bounded = SizeResponse(NEARLY_FLAT, 50000)
    assert bounded.span <= bounded.floor
    assert bounded.floor <= bounded.negligible_span
    assert bounded.verdict == "size-insensitive"


def test_fewer_than_three_sizes_is_refused_rather_than_answered():
    assert SizeResponse([(1.0, 0.5), (7.0, 0.8)], 5000).verdict.startswith(
        "insufficient"
    )


def test_a_flat_response_at_a_resolvable_scale_says_size_does_not_help():
    flat = [(1.0, 0.700), (3.0, 0.700), (7.0, 0.700)]
    assert SizeResponse(flat, 50000).verdict == "size-insensitive"


def test_the_floor_uses_the_measured_baseline_not_a_default():
    """MODEST starts at 62%, so its floor differs from a 70% baseline."""
    assert SizeResponse(MODEST, 200).floor == pytest.approx(0.129, abs=0.002)
    assert SizeResponse(SHALLOW, 200).floor == pytest.approx(0.119, abs=0.002)


def test_an_empty_response_does_not_crash():
    empty = SizeResponse([], 1000)
    assert empty.span == 0.0
    assert empty.verdict.startswith("insufficient")


# --- Pass 2: the limit check ------------------------------------------------


def _check(**overrides) -> LimitCheck:
    base = dict(
        task_contract_written=True,
        adaptation_attempted=True,
        output_constrained=True,
        failures_checked_for_slice_concentration=True,
        failure_concentrated_in_a_slice=False,
        response=SizeResponse(STEEP, 5000),
    )
    return LimitCheck(**{**base, **overrides})


def test_a_complete_check_with_a_steep_response_justifies_escalation():
    assert _check().blockers() == []
    assert _check().escalation_is_justified


def test_a_slice_concentrated_failure_blocks_escalation():
    """The check that fires most often. Chapter 10 routes that slice
    rather than moving every request to a larger model."""
    blocked = _check(failure_concentrated_in_a_slice=True)
    assert not blocked.escalation_is_justified
    assert any("routing" in b for b in blocked.blockers())


def test_skipping_adaptation_blocks_escalation():
    assert any("adaptation" in b for b in _check(adaptation_attempted=False).blockers())


def test_skipping_the_output_constraint_blocks_escalation():
    assert any("constrained" in b for b in _check(output_constrained=False).blockers())


def test_no_task_contract_blocks_escalation():
    assert any(
        "task contract" in b for b in _check(task_contract_written=False).blockers()
    )


def test_an_unresolvable_response_blocks_and_names_the_examples_needed():
    """Escalating on evidence you cannot resolve is the failure this
    gate exists to prevent."""
    blocked = _check(response=SizeResponse(MODEST, 200))
    assert not blocked.escalation_is_justified
    assert any("547 examples would settle it" in b for b in blocked.blockers())


def test_a_size_insensitive_response_blocks_with_a_plain_statement():
    flat = SizeResponse([(1.0, 0.7), (3.0, 0.7), (7.0, 0.7)], 50000)
    blocked = _check(response=flat)
    assert any("escalating buys nothing" in b for b in blocked.blockers())


def test_too_few_sizes_blocks_escalation():
    two_points = SizeResponse([(1.0, 0.45), (7.0, 0.78)], 5000)
    assert any("three sizes" in b for b in _check(response=two_points).blockers())


def test_every_blocker_is_reported_at_once():
    nothing_done = LimitCheck(
        task_contract_written=False,
        adaptation_attempted=False,
        output_constrained=False,
        failures_checked_for_slice_concentration=False,
        failure_concentrated_in_a_slice=True,
        response=SizeResponse(MODEST, 200),
    )
    assert len(nothing_done.blockers()) == 6


def test_the_gate_blocks_a_retreat_rather_than_a_change():
    """Unlike every other gate in the book: this one blocks reaching for
    a larger model until the cheaper options have been tried."""
    assert not _check(adaptation_attempted=False).escalation_is_justified
    assert _check(adaptation_attempted=True).escalation_is_justified
