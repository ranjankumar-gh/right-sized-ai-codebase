"""Tests for the Chapter 1 footprint estimator."""

import pytest

from rightsize.sizing import (
    GIB,
    ModelShape,
    Workload,
    bytes_per_weight,
    decode_ceiling_tokens_per_sec,
    estimate_footprint,
    kv_cache_bytes,
    shape_from_config,
)

DENSE_7B = ModelShape(
    name="7B-class dense",
    total_params_b=7.0,
    active_params_b=7.0,
    layers=32,
    kv_heads=8,
    head_dim=128,
)

MOE_30B_A3B = ModelShape(
    name="30B-total / 3B-active MoE",
    total_params_b=30.0,
    active_params_b=3.0,
    layers=48,
    kv_heads=4,
    head_dim=128,
)


class TestShapeFromConfig:
    def test_reads_a_modern_gqa_config(self) -> None:
        config = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 4096,
        }
        shape = shape_from_config("candidate", config, total_params_b=7.0)
        assert shape.layers == 32
        assert shape.kv_heads == 8
        assert shape.head_dim == 128

    def test_derives_head_dim_when_absent(self) -> None:
        config = {
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "hidden_size": 2048,
        }
        shape = shape_from_config("candidate", config, total_params_b=3.0)
        assert shape.head_dim == 2048 // 16

    def test_falls_back_to_query_heads_without_gqa(self) -> None:
        """Pre-GQA models give every query head its own KV head. Assuming
        GQA here would underestimate the cache by the head ratio."""
        config = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
        }
        shape = shape_from_config("legacy", config, total_params_b=7.0)
        assert shape.kv_heads == 32

    def test_gqa_fallback_changes_the_cache_materially(self) -> None:
        base = {"num_hidden_layers": 32, "num_attention_heads": 32,
                "hidden_size": 4096}
        legacy = shape_from_config("legacy", base, 7.0)
        gqa = shape_from_config("gqa", {**base, "num_key_value_heads": 8}, 7.0)
        workload = Workload(4096, 8)
        assert kv_cache_bytes(legacy, workload) == pytest.approx(
            kv_cache_bytes(gqa, workload) * 4
        )

    def test_active_params_defaults_to_total_for_dense(self) -> None:
        config = {"num_hidden_layers": 16, "num_attention_heads": 16,
                  "num_key_value_heads": 8, "hidden_size": 1024}
        shape = shape_from_config("dense", config, total_params_b=1.0)
        assert shape.active_params_b == 1.0

    def test_moe_keeps_total_and_active_separate(self) -> None:
        config = {"num_hidden_layers": 48, "num_attention_heads": 32,
                  "num_key_value_heads": 4, "hidden_size": 4096}
        shape = shape_from_config(
            "moe", config, total_params_b=30.0, active_params_b=3.0
        )
        assert shape.total_params_b == 30.0
        assert shape.active_params_b == 3.0


class TestBytesPerWeight:
    def test_full_precision_has_no_metadata_overhead(self) -> None:
        assert bytes_per_weight(16) == 2.0
        assert bytes_per_weight(8) == 1.0

    def test_four_bit_carries_group_metadata(self) -> None:
        """The chapter's central correction: 4-bit is not 0.5 bytes."""
        assert bytes_per_weight(4, group_size=128) == pytest.approx(
            (4 + 20 / 128) / 8
        )
        assert bytes_per_weight(4, group_size=128) > 0.5

    def test_metadata_overhead_is_about_four_percent(self) -> None:
        naive = 4 / 8
        actual = bytes_per_weight(4, group_size=128)
        assert 0.03 < (actual - naive) / naive < 0.05

    def test_smaller_groups_cost_more_metadata(self) -> None:
        assert bytes_per_weight(4, group_size=32) > bytes_per_weight(
            4, group_size=128
        )

    def test_ungrouped_quantization_skips_the_correction(self) -> None:
        assert bytes_per_weight(4, group_size=None) == 0.5


class TestKVCache:
    def test_scales_linearly_with_context(self) -> None:
        small = kv_cache_bytes(DENSE_7B, Workload(1024, 1))
        large = kv_cache_bytes(DENSE_7B, Workload(4096, 1))
        assert large == pytest.approx(small * 4)

    def test_scales_linearly_with_batch(self) -> None:
        single = kv_cache_bytes(DENSE_7B, Workload(2048, 1))
        batched = kv_cache_bytes(DENSE_7B, Workload(2048, 8))
        assert batched == pytest.approx(single * 8)

    def test_matches_the_hand_calculation(self) -> None:
        # 2 (K and V) * 32 layers * 8 kv heads * 128 head dim * 2 bytes
        # * 2048 tokens * 1 sequence
        expected = 2 * 32 * 8 * 128 * 2 * 2048 * 1
        assert kv_cache_bytes(DENSE_7B, Workload(2048, 1)) == expected

    def test_quantizing_the_cache_halves_it(self) -> None:
        full = kv_cache_bytes(DENSE_7B, Workload(2048, 1), kv_bits=16)
        half = kv_cache_bytes(DENSE_7B, Workload(2048, 1), kv_bits=8)
        assert half == pytest.approx(full / 2)


class TestFootprint:
    def test_weights_dominate_at_short_context_and_batch_one(self) -> None:
        fp = estimate_footprint(DENSE_7B, Workload(512, 1), weight_bits=4)
        assert fp.weights_gib > fp.kv_cache_gib

    def test_cache_overtakes_weights_under_real_serving_load(self) -> None:
        """The chapter's warning, as an assertion."""
        fp = estimate_footprint(DENSE_7B, Workload(32768, 16), weight_bits=4)
        assert fp.kv_cache_gib > fp.weights_gib

    def test_total_range_brackets_its_components(self) -> None:
        fp = estimate_footprint(DENSE_7B, Workload(4096, 4))
        assert fp.total_low_gib < fp.total_high_gib
        assert fp.total_low_gib > fp.weights_gib + fp.kv_cache_gib

    def test_fits_in_judges_against_the_pessimistic_end(self) -> None:
        fp = estimate_footprint(DENSE_7B, Workload(2048, 1), weight_bits=4)
        assert fp.fits_in(fp.total_high_gib)
        assert not fp.fits_in(fp.total_high_gib - 0.01)

    def test_moe_is_charged_memory_on_total_parameters(self) -> None:
        moe = estimate_footprint(MOE_30B_A3B, Workload(2048, 1), weight_bits=4)
        dense = estimate_footprint(DENSE_7B, Workload(2048, 1), weight_bits=4)
        assert moe.weights_gib > dense.weights_gib * 4


class TestDecodeCeiling:
    def test_more_bandwidth_lifts_the_ceiling(self) -> None:
        slow = decode_ceiling_tokens_per_sec(DENSE_7B, 100.0)
        fast = decode_ceiling_tokens_per_sec(DENSE_7B, 900.0)
        assert fast == pytest.approx(slow * 9)

    def test_quantization_lifts_the_ceiling(self) -> None:
        full = decode_ceiling_tokens_per_sec(DENSE_7B, 900.0, weight_bits=16)
        quantized = decode_ceiling_tokens_per_sec(
            DENSE_7B, 900.0, weight_bits=4
        )
        assert quantized > full * 3

    def test_moe_decodes_at_the_speed_of_its_active_parameters(self) -> None:
        """Memory is charged on total, speed on active. That asymmetry is
        the entire reason the estimator takes both numbers."""
        moe = decode_ceiling_tokens_per_sec(MOE_30B_A3B, 900.0, weight_bits=4)
        equivalent_dense = ModelShape(
            name="3B dense",
            total_params_b=3.0,
            active_params_b=3.0,
            layers=48,
            kv_heads=4,
            head_dim=128,
        )
        assert moe == pytest.approx(
            decode_ceiling_tokens_per_sec(equivalent_dense, 900.0, 4)
        )

    def test_matches_the_hand_calculation(self) -> None:
        active_gib = 7.0e9 * bytes_per_weight(16) / GIB
        assert decode_ceiling_tokens_per_sec(
            DENSE_7B, 900.0
        ) == pytest.approx(900.0 / active_gib)
