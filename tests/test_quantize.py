"""Chapter 6: quantization mechanics, granularity, and outlier damage."""

import random

import pytest

from rightsize.quantize import (
    QuantizationPlan,
    clipped_range,
    dequantize_group,
    quantize_group,
    quantize_tensor,
    relative_error,
)
from rightsize.sizing import bytes_per_weight


def gaussian_weights(count: int = 4096, seed: int = 11) -> list[float]:
    """The chapter's sample. Seeded so every number in it reproduces."""
    random.seed(seed)
    return [random.gauss(0.0, 0.02) for _ in range(count)]


WEIGHTS = gaussian_weights()


# --- Pass 1: one group ------------------------------------------------------


def test_a_round_trip_stays_near_the_original():
    group = WEIGHTS[:128]
    codes, scale, zero = quantize_group(group, bits=8)
    assert relative_error(group, dequantize_group(codes, scale, zero)) < 0.02


def test_codes_stay_inside_the_grid():
    codes, _, _ = quantize_group(WEIGHTS[:128], bits=4)
    assert all(0 <= c <= 15 for c in codes)


def test_a_constant_group_does_not_divide_by_zero():
    codes, scale, zero = quantize_group([0.5] * 32, bits=4)
    assert scale == 1.0
    assert all(c == codes[0] for c in codes)


def test_an_all_zero_group_survives_symmetric_quantization():
    """max(abs(v)) is zero here, which would make the scale zero."""
    codes, scale, zero = quantize_group([0.0] * 32, bits=4, symmetric=True)
    assert scale > 0.0
    assert all(c == zero for c in codes)
    assert dequantize_group(codes, scale, zero) == [0.0] * 32


def test_symmetric_quantization_centres_the_zero_point_on_the_grid():
    """Offset binary: 4-bit symmetric stores zero at code 8."""
    _, _, zero = quantize_group(WEIGHTS[:128], bits=4, symmetric=True)
    assert zero == 8


def test_symmetric_quantization_keeps_negative_weights_negative():
    """Regression: with zero_point 0, every negative value was clamped to
    code 0 and restored as 0. Each value must come back within half a
    grid step, and the negative ones must come back negative."""
    group = [-0.03, -0.02, -0.01, -0.002, 0.0, 0.002, 0.01, 0.02, 0.03]
    codes, scale, zero = quantize_group(group, bits=4, symmetric=True)
    restored = dequantize_group(codes, scale, zero)
    assert all(abs(a - b) <= scale / 2 + 1e-12 for a, b in zip(group, restored))
    assert all(r < 0 for a, r in zip(group, restored) if a <= -0.01)
    assert all(r > 0 for a, r in zip(group, restored) if a >= 0.01)


def test_symmetric_round_trip_on_gaussian_weights_with_negatives():
    """Gaussian weights are half negative. The restored error must be in
    line with the asymmetric 4-bit figure, not the all-zeros-for-negatives
    error the bug produced."""
    restored = quantize_tensor(WEIGHTS, bits=4, group_size=128, symmetric=True)
    assert relative_error(WEIGHTS, restored) < 0.15
    negatives = [i for i, w in enumerate(WEIGHTS) if w < 0]
    assert all(restored[i] < 0 for i in negatives if WEIGHTS[i] < -0.01)


# --- Pass 2: the bit sweep --------------------------------------------------


def test_error_roughly_doubles_per_bit_removed():
    """The shape of the chapter's first table, asserted as a shape."""
    errors = {
        bits: relative_error(WEIGHTS, quantize_tensor(WEIGHTS, bits, 128))
        for bits in (8, 6, 4, 3, 2)
    }
    assert errors[8] < errors[6] < errors[4] < errors[3] < errors[2]
    assert errors[3] / errors[4] == pytest.approx(2.0, abs=0.4)
    assert errors[2] / errors[3] == pytest.approx(2.0, abs=0.5)


def test_the_published_bit_sweep_figures():
    """The exact numbers the chapter prints, so the text cannot drift."""
    expected = {8: 0.0060, 6: 0.0238, 4: 0.1015, 3: 0.2155, 2: 0.5085}
    for bits, value in expected.items():
        actual = relative_error(WEIGHTS, quantize_tensor(WEIGHTS, bits, 128))
        assert actual == pytest.approx(value, abs=0.0005)


def test_eight_bit_is_close_to_free():
    assert relative_error(WEIGHTS, quantize_tensor(WEIGHTS, 8, 128)) < 0.01


def test_two_bit_loses_about_half_the_signal():
    assert relative_error(WEIGHTS, quantize_tensor(WEIGHTS, 2, 128)) > 0.4


# --- Pass 3: granularity ----------------------------------------------------


def test_smaller_groups_reduce_error_monotonically():
    errors = [
        relative_error(WEIGHTS, quantize_tensor(WEIGHTS, 4, group))
        for group in (4096, 256, 128, 64, 32)
    ]
    assert errors == sorted(errors, reverse=True)


def test_the_published_group_sweep_figures():
    expected = {4096: 0.1339, 256: 0.1110, 128: 0.1015, 64: 0.0929, 32: 0.0820}
    for group, value in expected.items():
        actual = relative_error(WEIGHTS, quantize_tensor(WEIGHTS, 4, group))
        assert actual == pytest.approx(value, abs=0.0005)


def test_group_32_costs_about_11_percent_more_memory_than_group_128():
    """The trade the chapter tells readers to price with Ch1's estimator."""
    ratio = bytes_per_weight(4, 32) / bytes_per_weight(4, 128) - 1
    assert ratio == pytest.approx(0.113, abs=0.005)


def test_the_accuracy_gain_from_group_32_is_smaller_than_the_memory_cost():
    """1.95 points of error against 11.3% more memory: a real decision,
    not an obvious one."""
    coarse = relative_error(WEIGHTS, quantize_tensor(WEIGHTS, 4, 128))
    fine = relative_error(WEIGHTS, quantize_tensor(WEIGHTS, 4, 32))
    assert (coarse - fine) == pytest.approx(0.0195, abs=0.002)


# --- Pass 4: the outlier ----------------------------------------------------


def _others_error(multiple: float) -> float:
    """Error on the 127 innocent values when one neighbour is inflated."""
    group = WEIGHTS[:128]
    spiked = list(group)
    spiked[64] = max(abs(v) for v in group) * multiple
    codes, scale, zero = quantize_group(spiked, bits=4)
    restored = dequantize_group(codes, scale, zero)
    return relative_error(group[:64] + group[65:], restored[:64] + restored[65:])


def test_one_outlier_wrecks_the_other_127_values():
    """The chapter's central demonstration: 10.6% becomes 65.1%, and the
    damage is to the values that were never outliers."""
    clean_group = WEIGHTS[:128]
    codes, scale, zero = quantize_group(clean_group, bits=4)
    clean = relative_error(clean_group, dequantize_group(codes, scale, zero))
    assert clean == pytest.approx(0.1058, abs=0.001)
    assert _others_error(10) == pytest.approx(0.6507, abs=0.002)


def test_a_large_enough_outlier_destroys_them_completely():
    """Relative error of exactly 1.0: the survivors carry no information,
    because they have all collapsed onto one code."""
    assert _others_error(50) == pytest.approx(1.0, abs=0.001)


def test_damage_saturates_rather_than_growing_without_bound():
    """Past total collapse there is nothing further to lose."""
    assert _others_error(50) == pytest.approx(_others_error(200), abs=0.001)


def test_a_smaller_group_contains_the_blast_radius():
    """Why per-group scaling exists: an outlier can only take down the
    values that share its scale."""
    spiked = list(WEIGHTS)
    spiked[64] = max(abs(v) for v in WEIGHTS[:128]) * 50
    wide = relative_error(WEIGHTS, quantize_tensor(spiked, 4, 512))
    narrow = relative_error(WEIGHTS, quantize_tensor(spiked, 4, 32))
    assert narrow < wide


# --- Pass 6: the plan -------------------------------------------------------


def test_a_sensible_plan_for_a_large_model_has_no_concerns():
    plan = QuantizationPlan(weight_bits=4, group_size=128, kv_bits=8)
    assert plan.concerns(params_b=13.0) == []


def test_a_small_model_at_4_bit_is_warned_about_the_penalty():
    """The chapter's opening scene, fired before the deployment."""
    plan = QuantizationPlan(weight_bits=4, group_size=128, kv_bits=8)
    concerns = plan.concerns(params_b=3.0)
    assert any("small-model penalty" in c for c in concerns)


def test_the_same_small_model_passes_at_a_smaller_group():
    plan = QuantizationPlan(weight_bits=4, group_size=64, kv_bits=8)
    assert plan.concerns(params_b=3.0) == []


def test_an_unquantized_cache_is_flagged():
    plan = QuantizationPlan(weight_bits=4, group_size=64, kv_bits=16)
    assert any("KV cache" in c for c in plan.concerns(params_b=13.0))


def test_two_bit_weights_are_flagged_as_needing_more_than_rounding():
    plan = QuantizationPlan(weight_bits=2, group_size=64, kv_bits=8)
    assert any("round-to-nearest" in c for c in plan.concerns(params_b=13.0))


def test_every_concern_is_reported_at_once():
    plan = QuantizationPlan(weight_bits=2, group_size=16, kv_bits=16)
    assert len(plan.concerns(params_b=3.0)) >= 3


# --- The clipping mitigation ------------------------------------------------


def test_clipping_recovers_almost_all_the_outlier_damage():
    """65.07% with a min-max grid, 11.20% clipped, against 10.58% clean."""
    group = WEIGHTS[:128]
    spiked = list(group)
    spiked[64] = max(abs(v) for v in group) * 10

    low, high = clipped_range(spiked, keep=0.98)
    clamped = [max(low, min(high, v)) for v in spiked]
    codes, scale, zero = quantize_group(clamped, bits=4)
    restored = dequantize_group(codes, scale, zero)
    others = relative_error(
        group[:64] + group[65:], restored[:64] + restored[65:]
    )
    assert others == pytest.approx(0.1120, abs=0.002)
    assert others < _others_error(10) / 5


def test_clipped_range_is_narrower_than_the_full_range():
    values = WEIGHTS[:128]
    low, high = clipped_range(values, keep=0.98)
    assert low > min(values)
    assert high < max(values)


def test_keeping_everything_returns_the_full_range():
    values = WEIGHTS[:128]
    low, high = clipped_range(values, keep=1.0)
    assert (low, high) == (min(values), max(values))


def test_clipping_is_bounded_by_the_values_it_keeps():
    """Every clipped value lands inside the retained range, which is what
    bounds the error on the survivors."""
    values = WEIGHTS[:128]
    low, high = clipped_range(values, keep=0.90)
    clamped = [max(low, min(high, v)) for v in values]
    assert min(clamped) >= low
    assert max(clamped) <= high
    assert sum(1 for a, b in zip(values, clamped) if a != b) > 0
