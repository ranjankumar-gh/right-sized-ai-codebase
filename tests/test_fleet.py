"""Chapter 13: fleet economics, rollout duration, the version tail, telemetry."""

import pytest

from rightsize.fleet import (
    FleetEconomics,
    RolloutProgress,
    UpdatePlan,
    telemetry_violations,
)


def _fleet(requests_per_device: int) -> FleetEconomics:
    return FleetEconomics(
        device_count=1000,
        device_capex=500.0,
        amortization_months=36,
        requests_per_device_per_month=requests_per_device,
        rented_cost_per_request=0.002,
    )


# --- Pass 1: the utilization inversion --------------------------------------


def test_the_published_fleet_table():
    """@tbl-fleet."""
    expected = {100: 0.13889, 1000: 0.01389, 10000: 0.00139, 100000: 0.00014}
    for requests, cost in expected.items():
        assert _fleet(requests).owned_cost_per_request == pytest.approx(
            cost, abs=0.00001
        )


def test_owned_cost_per_request_falls_with_utilization():
    """The inversion: a fixed numerator over a growing denominator."""
    costs = [_fleet(r).owned_cost_per_request for r in (100, 1000, 10000, 100000)]
    assert costs == sorted(costs, reverse=True)
    assert costs[0] / costs[-1] == pytest.approx(1000.0, rel=0.01)


def test_the_monthly_capex_does_not_move_with_utilization():
    """Which is the whole reason the per-request cost falls."""
    assert _fleet(100).monthly_capex == _fleet(100000).monthly_capex


def test_the_break_even_is_about_seven_thousand_requests_per_device():
    assert _fleet(1).break_even_requests_per_device == pytest.approx(6944, abs=1)


def test_renting_wins_below_the_crossing_and_owning_above_it():
    assert _fleet(1000).advantage < 1.0
    assert _fleet(10000).advantage > 1.0


def test_the_gap_widens_on_both_sides_of_the_crossing():
    """They cross exactly once, so the advantage is monotone."""
    advantages = [_fleet(r).advantage for r in (100, 1000, 10000, 100000)]
    assert advantages == sorted(advantages)
    assert advantages[-1] == pytest.approx(14.4, abs=0.1)


def test_an_idle_fleet_has_infinite_cost_per_request():
    """The operational form: a device doing nothing generates cost and
    no value."""
    assert _fleet(0).owned_cost_per_request == float("inf")


def test_free_rented_capacity_has_no_break_even():
    never = FleetEconomics(1000, 500.0, 36, 1000, rented_cost_per_request=0.0)
    assert never.break_even_requests_per_device == float("inf")


# --- Pass 2: rollout duration -----------------------------------------------


def _plan(delta_fraction: float) -> UpdatePlan:
    return UpdatePlan(
        artifact_mb=2048, device_count=10000,
        delta_fraction=delta_fraction,
        per_device_mbps=1.0, shared_uplink_mbps=100.0,
    )


def test_the_published_rollout_table():
    """@tbl-rollout."""
    assert _plan(1.0).days_through_shared_uplink == pytest.approx(19.0, abs=0.1)
    assert _plan(0.10).days_through_shared_uplink == pytest.approx(1.9, abs=0.1)
    assert _plan(0.05).days_through_shared_uplink == pytest.approx(0.9, abs=0.1)


def test_a_full_model_rollout_is_a_different_operating_model():
    """Nineteen days means a fault found on day three cannot be undone
    before day twenty-two."""
    assert _plan(1.0).days_through_shared_uplink > 14


def test_a_five_percent_delta_finishes_inside_a_day():
    assert _plan(0.05).days_through_shared_uplink < 1.0


def test_delta_size_scales_the_rollout_linearly():
    assert _plan(0.10).days_through_shared_uplink == pytest.approx(
        _plan(1.0).days_through_shared_uplink * 0.10, rel=0.01
    )


def test_per_device_time_and_fleet_time_are_different_questions():
    """One site's experience against the fleet's. Only the second bounds
    how fast a bad release can be undone."""
    plan = _plan(1.0)
    assert plan.hours_per_device == pytest.approx(4.55, abs=0.05)
    assert plan.days_through_shared_uplink > plan.hours_per_device / 24


def test_no_shared_uplink_reports_zero_rather_than_guessing():
    unmetered = UpdatePlan(2048, 10000, 1.0, per_device_mbps=1.0)
    assert unmetered.days_through_shared_uplink == 0.0
    assert unmetered.hours_per_device > 0


def test_a_dead_link_takes_forever_rather_than_dividing_by_zero():
    assert UpdatePlan(2048, 10, 1.0, per_device_mbps=0.0).hours_per_device == float(
        "inf"
    )


# --- Pass 3: the version tail -----------------------------------------------


TAIL = RolloutProgress(reach_per_attempt=0.70, permanently_unreachable=0.01)


def test_the_published_tail_table():
    """@tbl-tail."""
    expected = {1: 0.3070, 3: 0.0367, 7: 0.0102, 30: 0.0100}
    for day, stale in expected.items():
        assert TAIL.stale_fraction_after(day) == pytest.approx(stale, abs=0.0005)


def test_the_tail_converges_to_the_permanent_fraction():
    assert TAIL.stale_fraction_after(365) == pytest.approx(0.01, abs=0.0001)


def test_a_faster_rollout_converges_sooner_to_the_same_floor():
    """Retry policy controls how fast, never where."""
    fast = RolloutProgress(0.90, permanently_unreachable=0.01)
    assert fast.stale_fraction_after(3) < TAIL.stale_fraction_after(3)
    assert fast.stale_fraction_after(30) == pytest.approx(
        TAIL.stale_fraction_after(30), abs=0.0001
    )


def test_a_target_above_the_floor_is_reachable():
    assert TAIL.attempts_to_reach(0.02) == 4


def test_a_target_below_the_floor_returns_none():
    """The answer worth having: no retry schedule reaches it, and the
    fix is a truck rather than a timer."""
    assert TAIL.attempts_to_reach(0.005) is None


def test_with_no_permanent_fraction_the_tail_really_does_vanish():
    perfect = RolloutProgress(0.70, permanently_unreachable=0.0)
    assert perfect.stale_fraction_after(30) < 1e-9
    assert perfect.attempts_to_reach(0.001) is not None


# --- Pass 4: the telemetry boundary -----------------------------------------


def test_allowed_aggregate_fields_pass():
    clean = {
        "artifact_version": "b4200",
        "tokens_per_second": 24.0,
        "escalation_rate": 0.12,
        "throttled": True,
    }
    assert telemetry_violations(clean) == []


def test_request_content_is_rejected_with_the_fix_named():
    problems = telemetry_violations({"last_prompt": "..."})
    assert len(problems) == 1
    assert "aggregate it on the device" in problems[0]


def test_every_content_shaped_field_is_caught():
    for field in ("prompt", "user_input", "model_output", "response_text",
                  "document_id", "request_payload"):
        assert telemetry_violations({field: "x"}), f"{field} slipped through"


def test_an_unlisted_field_is_flagged_rather_than_silently_allowed():
    """An allow-list, because the failure is asymmetric: a missing field
    is a slower diagnosis, an extra one is a disclosure."""
    problems = telemetry_violations({"device_serial": "abc"})
    assert any("not on the allow-list" in p for p in problems)


def test_every_violation_is_reported_at_once():
    problems = telemetry_violations(
        {"last_prompt": "x", "device_serial": "y", "tokens_per_second": 1}
    )
    assert len(problems) == 2
