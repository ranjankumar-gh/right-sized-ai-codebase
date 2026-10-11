"""Chapter 11: concurrency ceiling, blocks, prefix cache, and goodput."""

import pytest

from rightsize.serve import (
    PrefixCacheSaving,
    RequestOutcome,
    ServiceLevel,
    ServingCapacity,
    block_waste_fraction,
    block_waste_tokens,
    blocks_for,
    can_admit,
    goodput,
    kv_bytes_per_token,
    preemption_rate,
    serving_concerns,
)
from rightsize.sizing import GIB, ModelShape

SEVEN_B = ModelShape(
    name="7B-class dense",
    total_params_b=7.0,
    active_params_b=7.0,
    layers=32,
    kv_heads=8,
    head_dim=128,
)


def _capacity(device_gib: float, context: int) -> ServingCapacity:
    return ServingCapacity(SEVEN_B, device_gib, context)


# --- Pass 1: the concurrency ceiling ----------------------------------------


def test_the_published_ceiling_table():
    """@tbl-ceiling, exactly."""
    expected = {
        (24, 2048): 74, (24, 8192): 18, (24, 32768): 4,
        (48, 2048): 170, (48, 8192): 42, (48, 32768): 10,
        (80, 2048): 298, (80, 8192): 74, (80, 32768): 18,
    }
    for (device, context), ceiling in expected.items():
        assert _capacity(device, context).concurrency_ceiling == ceiling


def test_context_length_swings_capacity_by_more_than_a_factor_of_ten():
    """The chapter's headline: context is the concurrency dial."""
    short = _capacity(24, 2048).concurrency_ceiling
    long = _capacity(24, 32768).concurrency_ceiling
    assert short / long > 15


def test_24_and_80_gib_both_report_74_because_the_floor_hides_the_gap():
    """Not a quadrupling. 80 GiB is 3.3x 24 GiB, but the weights and the
    runtime overhead are fixed costs, so the KV budget grows 4.01x
    (74.6 GiB against 18.6 GiB). The context also grows 4x, so the raw
    quotients are 74.45 and 74.61, and both floor to 74. The match is an
    effect of flooring two nearly equal quotients, not of linear scaling."""
    small = _capacity(24, 2048)
    large = _capacity(80, 8192)
    assert large.device_gib / small.device_gib == pytest.approx(3.33, abs=0.01)
    assert large.kv_budget_gib / small.kv_budget_gib == pytest.approx(4.0, abs=0.02)
    assert small.kv_budget_gib / small.gib_per_request == pytest.approx(74.45, abs=0.01)
    assert large.kv_budget_gib / large.gib_per_request == pytest.approx(74.61, abs=0.01)
    assert small.concurrency_ceiling == large.concurrency_ceiling == 74


def test_a_small_device_at_short_context_beats_a_large_one_at_long_context():
    """74 against 18: the smaller machine wins outright when the
    workload moves further than the hardware."""
    assert _capacity(24, 2048).concurrency_ceiling > (
        _capacity(80, 32768).concurrency_ceiling * 4
    )


def test_halving_the_context_roughly_doubles_the_ceiling():
    assert _capacity(48, 4096).concurrency_ceiling == pytest.approx(
        _capacity(48, 8192).concurrency_ceiling * 2, rel=0.05
    )


def test_the_weights_are_a_fixed_cost_and_the_cache_is_per_request():
    capacity = _capacity(24, 8192)
    assert capacity.weights_gib == pytest.approx(3.4, abs=0.1)
    assert capacity.gib_per_request == pytest.approx(1.0, abs=0.05)


def test_kv_per_token_is_independent_of_the_workload():
    assert kv_bytes_per_token(SEVEN_B) / 1024 == pytest.approx(128.0, abs=0.5)


def test_grouped_query_attention_raises_the_ceiling():
    """Ch1 showed GQA shrinks the cache; here that is concurrency."""
    full_mha = ModelShape("full-mha", 7.0, 7.0, 32, kv_heads=32, head_dim=128)
    gqa = ServingCapacity(SEVEN_B, 48, 8192).concurrency_ceiling
    mha = ServingCapacity(full_mha, 48, 8192).concurrency_ceiling
    assert gqa > mha * 3


def test_a_model_that_does_not_fit_has_no_cache_budget():
    tiny_device = ServingCapacity(SEVEN_B, 4.0, 2048)
    assert tiny_device.kv_budget_gib < 0
    assert tiny_device.concurrency_ceiling == 0


def test_the_ceiling_is_floored_because_a_half_fitting_request_does_not_run():
    capacity = _capacity(24, 32768)
    assert capacity.concurrency_ceiling == int(capacity.concurrency_ceiling)
    assert capacity.concurrency_ceiling * capacity.gib_per_request <= capacity.kv_budget_gib


# --- Pass 2: blocks ---------------------------------------------------------


def test_blocks_round_up():
    assert blocks_for(1, 16) == 1
    assert blocks_for(16, 16) == 1
    assert blocks_for(17, 16) == 2
    assert blocks_for(0, 16) == 0


def test_the_published_block_waste_table():
    """@tbl-blocks."""
    expected = {
        (37, 16): 0.297, (37, 32): 0.730, (37, 64): 0.730,
        (150, 16): 0.067, (150, 64): 0.280,
        (2000, 16): 0.0, (2000, 32): 0.008, (2000, 64): 0.024,
    }
    for (length, block), fraction in expected.items():
        assert block_waste_fraction(length, block) == pytest.approx(
            fraction, abs=0.001
        )


def test_waste_is_bounded_by_one_block_regardless_of_length():
    """Which is why it is noise at 2,000 tokens and severe at 37."""
    for length in (37, 150, 2000, 100_000):
        assert block_waste_tokens(length, 64) < 64


def test_an_exact_multiple_wastes_nothing():
    assert block_waste_tokens(512, 16) == 0
    assert block_waste_tokens(512, 64) == 0


def test_a_large_block_wastes_most_of_a_short_sequence():
    assert block_waste_fraction(37, 64) > 0.7


def test_the_same_block_is_irrelevant_on_a_long_sequence():
    assert block_waste_fraction(2000, 64) < 0.03


# --- Pass 3: the prefix cache -----------------------------------------------


SHARED = PrefixCacheSaving(
    shared_prefix_tokens=2000, unique_tokens=200, concurrent_requests=32
)


def test_the_published_prefix_saving():
    assert SHARED.blocks_without_sharing == 4416
    assert SHARED.blocks_with_sharing == 541
    assert SHARED.saving == pytest.approx(0.88, abs=0.01)


def test_the_saving_grows_with_concurrency():
    """The shared portion is stored once however many reference it."""
    few = PrefixCacheSaving(2000, 200, 2).saving
    many = PrefixCacheSaving(2000, 200, 64).saving
    assert many > few


def test_only_whole_blocks_are_shared():
    """A prefix is reused to the last block boundary it fills; the
    partial block after it belongs to the request."""
    partial = PrefixCacheSaving(shared_prefix_tokens=20, unique_tokens=100,
                                concurrent_requests=8, block_size=16)
    assert partial.shared_blocks == 1


def test_no_shared_prefix_saves_nothing():
    alone = PrefixCacheSaving(0, 200, 32)
    assert alone.saving == pytest.approx(0.0, abs=0.001)


def test_a_single_request_saves_nothing():
    assert PrefixCacheSaving(2000, 200, 1).saving == pytest.approx(0.0, abs=0.01)


# --- Pass 4: goodput --------------------------------------------------------


LEVEL = ServiceLevel(max_ttft_s=1.0, max_inter_token_latency_s=0.05)


def test_a_request_inside_every_bound_counts():
    assert goodput([RequestOutcome(0.8, 0.03)], LEVEL) == 1.0


def test_a_slow_first_token_fails_even_if_decode_is_fast():
    assert goodput([RequestOutcome(3.0, 0.01)], LEVEL) == 0.0


def test_slow_decode_fails_even_if_the_first_token_was_quick():
    assert goodput([RequestOutcome(0.2, 0.40)], LEVEL) == 0.0


def test_an_incomplete_request_never_counts():
    assert goodput([RequestOutcome(0.2, 0.01, completed=False)], LEVEL) == 0.0


def test_goodput_cannot_be_improved_by_sacrificing_the_other_metric():
    """Throughput rises by batching until every request is too slow;
    goodput records that as the failure it is."""
    fast_but_few = [RequestOutcome(0.5, 0.02)] * 3 + [
        RequestOutcome(0.5, 0.02, completed=False)
    ] * 7
    many_but_slow = [RequestOutcome(4.0, 0.02)] * 10
    assert goodput(fast_but_few, LEVEL) == pytest.approx(0.3)
    assert goodput(many_but_slow, LEVEL) == 0.0


def test_preemption_rate_is_reported_separately():
    outcomes = [
        RequestOutcome(0.5, 0.02),
        RequestOutcome(2.0, 0.02, preempted=True),
        RequestOutcome(0.5, 0.02),
        RequestOutcome(0.5, 0.02),
    ]
    assert preemption_rate(outcomes) == pytest.approx(0.25)


def test_empty_outcomes_are_zero_rather_than_a_division_error():
    assert goodput([], LEVEL) == 0.0
    assert preemption_rate([]) == 0.0


# --- Pass 5: admission ------------------------------------------------------


def test_admission_stops_below_the_ceiling():
    """A safety margin, because the ceiling assumes every resident
    request reaches full context."""
    capacity = _capacity(24, 2048)
    assert can_admit(capacity, resident_requests=60)
    assert not can_admit(capacity, resident_requests=70)


def test_the_planned_and_real_ceilings_are_both_named_when_they_differ():
    """The chapter's opening, as arithmetic against production numbers."""
    problems = serving_concerns(_capacity(24, 2048), measured_context_tokens=6000)
    assert len(problems) == 1
    assert "the real ceiling is 25, not 74" in problems[0]


def test_preemption_is_reported_as_admit_less_not_tune_more():
    problems = serving_concerns(
        _capacity(24, 2048), 2048, observed_preemption_rate=0.03
    )
    assert any("admit fewer" in p for p in problems)


def test_a_model_that_does_not_fit_is_reported_before_anything_else():
    problems = serving_concerns(ServingCapacity(SEVEN_B, 4.0, 2048), 2048)
    assert "no cache budget" in problems[0]


def test_a_ceiling_of_one_says_batching_cannot_help():
    problems = serving_concerns(ServingCapacity(SEVEN_B, 24, 131072), 131072)
    assert any("batching cannot help" in p for p in problems)


def test_a_correctly_planned_deployment_has_no_concerns():
    assert serving_concerns(_capacity(48, 8192), measured_context_tokens=8192) == []


def test_a_hybrid_request_is_not_its_length_times_the_per_token_cost():
    """Ch11 asks Ch1 for whole requests, so windowed layers stop growing."""
    hybrid = ModelShape(
        "5:1 sliding:full", 4.0, 4.0, 36, kv_heads=8, head_dim=128,
        full_attention_layers=6, window_layers=30, window_tokens=512,
    )
    capacity = ServingCapacity(hybrid, 24, 32768)
    scaled = kv_bytes_per_token(hybrid) * 32768 / GIB
    assert capacity.gib_per_request < scaled / 5
