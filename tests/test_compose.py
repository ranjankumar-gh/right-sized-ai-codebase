"""Chapter 10: cascade economics, the latency tax, and speculative decoding."""

import pytest

from rightsize.compose import (
    CascadeEconomics,
    CascadeLatency,
    CompositionGate,
    SpeculativeDecoding,
    best_draft_length,
    should_escalate,
    top_two_margin,
)


def _cascade(rate: float, small: float = 0.1, large: float = 1.0):
    return CascadeEconomics(small, large, rate)


# --- Pass 1: cascade economics ----------------------------------------------


def test_the_small_model_is_paid_on_every_request():
    """The structural cost that sets the break-even at 1 - r."""
    assert _cascade(0.0).cost_per_request == pytest.approx(0.1)


def test_the_published_cascade_costs():
    """@tbl-cascade-cost."""
    expected = {0.05: 0.150, 0.10: 0.200, 0.20: 0.300,
                0.50: 0.600, 0.90: 1.000, 0.95: 1.050}
    for rate, cost in expected.items():
        assert _cascade(rate).cost_per_request == pytest.approx(cost, abs=1e-9)


def test_break_even_is_one_minus_the_cost_ratio():
    assert _cascade(0.1, small=0.10).break_even_rate == pytest.approx(0.90)
    assert _cascade(0.1, small=0.25).break_even_rate == pytest.approx(0.75)
    assert _cascade(0.1, small=0.50).break_even_rate == pytest.approx(0.50)


def test_past_the_break_even_the_cascade_costs_more_than_always_large():
    """Not graceful degradation toward the large model's cost: strictly
    worse than not having built it."""
    assert _cascade(0.95).saving < 0
    assert _cascade(0.95).cost_per_request > 1.0


def test_at_the_break_even_it_saves_exactly_nothing():
    assert _cascade(0.90).saving == pytest.approx(0.0, abs=1e-9)


def test_headroom_reports_distance_to_the_break_even():
    """40% escalation is comfortable against a 90% break-even and
    negative against a 50% one."""
    assert _cascade(0.40, small=0.10).headroom == pytest.approx(0.50)
    assert _cascade(0.40, small=0.50).headroom == pytest.approx(0.10)
    assert _cascade(0.60, small=0.50).headroom < 0


def test_two_close_models_break_even_early():
    """A 3B in front of a 7B is usually two models doing the work of one."""
    assert _cascade(0.1, small=0.5).break_even_rate == pytest.approx(0.50)


def test_a_zero_cost_large_model_does_not_divide_by_zero():
    assert CascadeEconomics(0.1, 0.0, 0.5).break_even_rate == 0.0
    assert CascadeEconomics(0.1, 0.0, 0.5).saving == 0.0


# --- Pass 2: the latency tax ------------------------------------------------


OPENING = CascadeLatency(small_ms=120, large_ms=900, escalation_rate=0.15)


def test_the_opening_scenes_numbers():
    assert OPENING.mean_ms == pytest.approx(255.0)
    assert OPENING.quantile_ms(0.50) == pytest.approx(120.0)
    assert OPENING.quantile_ms(0.99) == pytest.approx(1020.0)
    assert OPENING.tail_penalty == pytest.approx(1.133, abs=0.001)


def test_the_median_improves_while_the_tail_degrades():
    """Both from the same architecture on the same day."""
    assert OPENING.quantile_ms(0.50) < 900
    assert OPENING.quantile_ms(0.99) > 900


def test_the_latency_tax_does_not_depend_on_the_escalation_rate():
    """Fixed per escalated request, so the tail is worse from the first one."""
    rare = CascadeLatency(120, 900, 0.01)
    common = CascadeLatency(120, 900, 0.50)
    assert rare.escalated_ms == common.escalated_ms == 1020
    assert rare.tail_penalty == common.tail_penalty


def test_latency_is_bimodal_with_nothing_in_between():
    """Which is why p50 and p99 can move in opposite directions."""
    seen = {OPENING.quantile_ms(q / 100) for q in range(1, 100)}
    assert seen == {120.0, 1020.0}


def test_the_quantile_boundary_sits_at_one_minus_the_escalation_rate():
    cascade = CascadeLatency(120, 900, 0.20)
    assert cascade.quantile_ms(0.79) == 120.0
    assert cascade.quantile_ms(0.81) == 1020.0


# --- Pass 3: the escalation signal ------------------------------------------


def test_a_confident_distribution_has_a_wide_margin():
    assert top_two_margin({"A": 0.9, "B": 0.05, "C": 0.05}) == pytest.approx(0.85)


def test_an_unsure_distribution_has_a_narrow_margin():
    assert top_two_margin({"A": 0.35, "B": 0.33, "C": 0.32}) == pytest.approx(0.02)


def test_a_single_candidate_is_treated_as_certain():
    assert top_two_margin({"A": 1.0}) == 1.0
    assert top_two_margin({}) == 0.0


def test_a_narrow_margin_escalates():
    assert should_escalate({"A": 0.35, "B": 0.33}, threshold=0.20)
    assert not should_escalate({"A": 0.9, "B": 0.05}, threshold=0.20)


def test_a_schema_forced_choice_always_escalates():
    """The Chapter 8 interlock. A margin computed on a masked
    distribution describes a forced choice, not a confidence, so the
    case the schema could not express goes to the larger model."""
    confident_looking = {"A": 0.95, "B": 0.05}
    assert not should_escalate(confident_looking, threshold=0.20)
    assert should_escalate(confident_looking, threshold=0.20,
                           schema_forced_choice=True)


# --- Pass 4: speculative decoding -------------------------------------------


def test_the_published_speedup_table():
    """@tbl-speculative."""
    expected = {
        (0.4, 3): 1.25, (0.4, 7): 0.98, (0.4, 10): 0.83,
        (0.6, 3): 1.67, (0.8, 5): 2.46, (0.9, 10): 3.43,
    }
    for (acceptance, length), speedup in expected.items():
        got = SpeculativeDecoding(acceptance, length).speedup
        assert got == pytest.approx(speedup, abs=0.01)


def test_a_poorly_matched_draft_model_is_a_slowdown():
    """The counterintuitive result: it does not merely fail to help."""
    slow = SpeculativeDecoding(acceptance_rate=0.4, draft_length=10)
    assert slow.is_slower
    assert slow.speedup < 1.0


def test_a_well_matched_draft_model_is_a_large_speedup():
    assert SpeculativeDecoding(0.9, 10).speedup > 3.0


def test_expected_tokens_can_exceed_the_draft_length():
    """The target model produces one token itself on rejection."""
    assert SpeculativeDecoding(0.9, 3).expected_tokens > 3.0 * 0.9


def test_perfect_acceptance_yields_every_drafted_token_plus_one():
    assert SpeculativeDecoding(1.0, 5).expected_tokens == pytest.approx(6.0)


def test_the_optimal_draft_length_rises_with_the_acceptance_rate():
    """@tbl-draft-length: it is not a constant to tune once."""
    assert best_draft_length(0.3) == 1
    assert best_draft_length(0.5) == 2
    assert best_draft_length(0.7) == 4
    assert best_draft_length(0.9) == 10


def test_a_longer_draft_is_not_monotonically_better():
    """Each extra token costs whether accepted or not, and its chance of
    surviving falls geometrically."""
    at_optimum = SpeculativeDecoding(0.6, best_draft_length(0.6)).speedup
    much_longer = SpeculativeDecoding(0.6, 20).speedup
    assert much_longer < at_optimum


def test_a_more_expensive_draft_model_shortens_the_optimum():
    assert best_draft_length(0.8, draft_cost=0.30) < best_draft_length(
        0.8, draft_cost=0.05
    )


# --- Pass 5: the composition gate -------------------------------------------


HEALTHY = _cascade(0.15)
HEALTHY_LATENCY = CascadeLatency(120, 900, 0.15)


def test_a_healthy_cascade_on_a_mean_bound_service_passes():
    gate = CompositionGate()
    assert gate.blockers(HEALTHY, HEALTHY_LATENCY, tail_bound=False) == []


def test_the_same_cascade_is_blocked_on_a_tail_bound_service():
    """The chapter's opening, caught before the architecture review."""
    problems = CompositionGate().blockers(
        HEALTHY, HEALTHY_LATENCY, tail_bound=True
    )
    assert any("tail-bound" in p for p in problems)


def test_a_cascade_past_its_break_even_is_blocked():
    problems = CompositionGate().blockers(
        _cascade(0.95), CascadeLatency(120, 900, 0.95), tail_bound=False
    )
    assert any("costs" in p and "more than always-large" in p for p in problems)


def test_thin_headroom_is_blocked_even_while_still_saving():
    """Escalation drifts upward, so saving today is not enough."""
    thin = _cascade(0.80)
    assert thin.saving > 0
    problems = CompositionGate().blockers(
        thin, CascadeLatency(120, 900, 0.80), tail_bound=False
    )
    assert any("headroom" in p for p in problems)


def test_a_bad_speculative_configuration_is_blocked_and_the_fix_named():
    """A blocker that only says 'too slow' leaves the reader to work out
    the fix. This one names the draft length that would work."""
    misconfigured = SpeculativeDecoding(0.4, 10)
    problems = CompositionGate().blockers(
        HEALTHY, HEALTHY_LATENCY, tail_bound=False, speculative=misconfigured,
    )
    optimum = best_draft_length(misconfigured.acceptance_rate,
                                misconfigured.draft_cost)
    assert any(
        "speculative" in p and f"best length here is {optimum}" in p
        for p in problems
    )
    assert optimum < misconfigured.draft_length


def test_a_good_speculative_configuration_passes():
    assert CompositionGate().blockers(
        HEALTHY, HEALTHY_LATENCY, tail_bound=False,
        speculative=SpeculativeDecoding(0.8, 6),
    ) == []
