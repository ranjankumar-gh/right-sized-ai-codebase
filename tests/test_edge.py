"""Chapter 12: thermal derate, the energy envelope, and the unattended device."""

import pytest

from rightsize.edge import (
    DutyCycle,
    EnergyProfile,
    ThermalProfile,
    UnattendedReadiness,
    edge_memory_headroom,
    energy_per_token_mj,
    survives_without_swap,
)

# The chapter's opening device: 40 tok/s peak, throttling to 60% after 30s.
BENCH_DEVICE = ThermalProfile(
    peak_tokens_per_sec=40.0, sustained_fraction=0.60, seconds_to_throttle=30.0
)


# --- Pass 1: the thermal derate ---------------------------------------------


def test_the_published_derate_table():
    """@tbl-derate."""
    expected = {10: 40.0, 30: 40.0, 60: 32.0, 300: 25.6}
    for window, reported in expected.items():
        assert BENCH_DEVICE.average_over(window) == pytest.approx(reported, abs=0.05)


def test_a_short_benchmark_cannot_detect_throttling_at_all():
    """The trap: 10s and 30s report identical numbers, so a shorter run
    does not look more wrong, it looks like a stable result."""
    assert BENCH_DEVICE.average_over(10) == BENCH_DEVICE.average_over(30)
    assert BENCH_DEVICE.average_over(10) == BENCH_DEVICE.peak_tokens_per_sec


def test_no_benchmark_length_reaches_the_sustained_rate():
    """Longer runs approach it from above and never arrive."""
    sustained = BENCH_DEVICE.sustained_tokens_per_sec
    for window in (60, 300, 3600, 86400):
        assert BENCH_DEVICE.average_over(window) > sustained


def test_longer_benchmarks_report_lower_throughput():
    windows = [10, 60, 300, 3600]
    reported = [BENCH_DEVICE.average_over(w) for w in windows]
    assert reported == sorted(reported, reverse=True)


def test_the_overstatement_factor_shrinks_with_benchmark_length():
    assert BENCH_DEVICE.benchmark_overstates_by(10) == pytest.approx(1 / 0.60, abs=0.01)
    assert BENCH_DEVICE.benchmark_overstates_by(3600) < 1.02


def test_a_device_that_does_not_throttle_reports_the_same_number_always():
    steady = ThermalProfile(40.0, sustained_fraction=1.0, seconds_to_throttle=30.0)
    assert steady.average_over(10) == steady.average_over(10_000)


# --- Pass 2: the energy envelope --------------------------------------------


def test_the_published_battery_table():
    """@tbl-battery, on a 20 Wh battery."""
    expected = {25: 14_400, 100: 3_600, 270: 1_333, 500: 720}
    for millijoules, responses in expected.items():
        got = EnergyProfile(millijoules).responses_from_battery(20)
        assert got == pytest.approx(responses, rel=0.001)


def test_tokens_per_joule_is_the_reciprocal_of_energy_per_token():
    assert EnergyProfile(100).tokens_per_joule == pytest.approx(10.0)
    assert EnergyProfile(25).tokens_per_joule == pytest.approx(40.0)


def test_power_alone_ranks_backends_wrongly():
    """Twice the watts at three times the speed is the more efficient
    option, and a wattage comparison says the opposite."""
    thirsty_but_fast = energy_per_token_mj(watts=20, tokens_per_sec=15)
    frugal_but_slow = energy_per_token_mj(watts=10, tokens_per_sec=5)
    assert thirsty_but_fast < frugal_but_slow


def test_a_stopped_device_has_infinite_energy_per_token():
    assert energy_per_token_mj(watts=10, tokens_per_sec=0) == float("inf")


def test_a_free_token_does_not_divide_by_zero():
    assert EnergyProfile(0).tokens_per_joule == 0.0


# --- Pass 3: the duty cycle -------------------------------------------------


def _share(idle_watts: float, responses_per_hour: int) -> float:
    return DutyCycle(
        EnergyProfile(100.0, idle_watts), responses_per_hour
    ).inference_share


def test_the_published_duty_cycle_table():
    """@tbl-duty."""
    expected = {
        (0.1, 10): 0.36, (0.1, 100): 0.85, (0.1, 1000): 0.98,
        (0.5, 10): 0.10, (0.5, 100): 0.53, (0.5, 1000): 0.92,
        (2.0, 10): 0.03, (2.0, 100): 0.22, (2.0, 1000): 0.74,
    }
    for (idle, rate), share in expected.items():
        assert _share(idle, rate) == pytest.approx(share, abs=0.01)


def test_a_low_duty_cycle_device_spends_almost_nothing_on_inference():
    """The result that undercuts Parts II and III for this deployment."""
    assert _share(idle_watts=2.0, responses_per_hour=10) < 0.05


def test_a_high_duty_cycle_device_spends_almost_everything_on_inference():
    assert _share(idle_watts=0.1, responses_per_hour=1000) > 0.95


def test_halving_model_energy_barely_moves_a_low_duty_cycle_device():
    """Halving energy per token improves field life by about 1.4%."""
    battery = 20.0
    before = DutyCycle(EnergyProfile(100.0, 2.0), 10).hours_from_battery(battery)
    after = DutyCycle(EnergyProfile(50.0, 2.0), 10).hours_from_battery(battery)
    assert (after / before - 1) < 0.02


def test_the_same_halving_matters_on_a_high_duty_cycle_device():
    battery = 20.0
    before = DutyCycle(EnergyProfile(100.0, 0.1), 1000).hours_from_battery(battery)
    after = DutyCycle(EnergyProfile(50.0, 0.1), 1000).hours_from_battery(battery)
    assert (after / before - 1) > 0.80


def test_an_idle_device_still_drains_the_battery():
    idle_only = DutyCycle(EnergyProfile(100.0, 0.5), responses_per_hour=0)
    assert idle_only.inference_share == 0.0
    assert idle_only.hours_from_battery(20) == pytest.approx(40.0, rel=0.01)


def test_a_device_drawing_nothing_at_all_never_runs_out():
    never = DutyCycle(EnergyProfile(100.0, 0.0), responses_per_hour=0)
    assert never.hours_from_battery(20) == float("inf")


# --- Pass 4: the memory cliff -----------------------------------------------


def test_headroom_subtracts_the_model_and_the_reservation():
    assert edge_memory_headroom(6.0, 16.0, reserved_gib=0.5) == pytest.approx(9.5)


def test_a_tight_fit_is_rejected_on_a_device_without_swap():
    """7 GiB plus the OS on an 8 GiB board leaves 0.5 GiB, below the
    margin, and the failure is a kill rather than a slowdown."""
    assert not survives_without_swap(7.0, 8.0)


def test_a_comfortable_fit_passes():
    assert survives_without_swap(6.0, 16.0)


def test_a_model_larger_than_the_device_is_rejected():
    assert not survives_without_swap(20.0, 8.0)
    assert edge_memory_headroom(20.0, 8.0) < 0


def test_the_margin_scales_with_the_device():
    """A 15% margin is more absolute memory on a larger board, which is
    correct: more of the board is doing other things."""
    assert survives_without_swap(6.0, 8.0, safety_fraction=0.15)
    assert not survives_without_swap(6.0, 8.0, safety_fraction=0.30)


# --- Pass 5: the unattended constraint --------------------------------------


READY = UnattendedReadiness(
    cold_boot_seconds=12.0,
    max_acceptable_boot_seconds=30.0,
    has_local_fallback=True,
    verifies_artifact_before_load=True,
    keeps_previous_artifact=True,
    restarts_on_crash=True,
)


def test_a_ready_device_has_no_gaps():
    assert READY.gaps() == []


def test_a_slow_cold_boot_is_reported_with_both_numbers():
    import dataclasses

    slow = dataclasses.replace(READY, cold_boot_seconds=90.0)
    gaps = slow.gaps()
    assert any("90s against a 30s budget" in g for g in gaps)


def test_an_unverified_artifact_is_named_as_a_boot_loop_risk():
    import dataclasses

    unverified = dataclasses.replace(READY, verifies_artifact_before_load=False)
    assert any("boot loop" in g for g in unverified.gaps())


def test_no_retained_previous_artifact_means_a_physical_visit():
    import dataclasses

    no_rollback = dataclasses.replace(READY, keeps_previous_artifact=False)
    assert any("rolled back" in g for g in no_rollback.gaps())


def test_no_supervised_restart_is_a_gap():
    import dataclasses

    unsupervised = dataclasses.replace(READY, restarts_on_crash=False)
    assert any("restart" in g for g in unsupervised.gaps())


def test_every_gap_is_reported_at_once():
    nothing_ready = UnattendedReadiness(90.0, 30.0, False, False, False, False)
    assert len(nothing_ready.gaps()) == 5
