"""Chapter 4: adapter sizing, training memory, and the two silent-failure guards."""

import pytest

from rightsize.adapt import (
    GIB,
    IGNORE_INDEX,
    LORA_TARGETS_ALL,
    LORA_TARGETS_ATTENTION,
    DeploymentGate,
    LossMaskNotApplied,
    TemplateDrift,
    adapter_parameter_count,
    check_loss_mask,
    check_template_match,
    estimate_training_memory,
    linear_shapes,
    supervised_fraction,
)

SMOLLM3_3B = {
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "tie_word_embeddings": True,
}
SMOLLM3_TOTAL_PARAMS = 3_075_096_576


# --- Pass 1: the adapter budget ---------------------------------------------


def test_adapter_is_about_one_percent_at_rank_16():
    """The chapter's headline: rank 16 over all linears is ~1% of a 3B."""
    trainable = adapter_parameter_count(SMOLLM3_3B, rank=16)
    assert trainable / 1e6 == pytest.approx(30.2, abs=0.1)
    assert trainable / SMOLLM3_TOTAL_PARAMS == pytest.approx(0.0098, abs=0.0002)


def test_adapter_size_is_linear_in_rank():
    r8 = adapter_parameter_count(SMOLLM3_3B, rank=8)
    r16 = adapter_parameter_count(SMOLLM3_3B, rank=16)
    r32 = adapter_parameter_count(SMOLLM3_3B, rank=32)
    assert r16 == 2 * r8
    assert r32 == 2 * r16


def test_attention_only_declines_to_touch_where_the_capacity_is():
    """Ch3 showed 87% of computing params are in the MLP; an attention-only
    adapter trains a quarter of a percent and skips all of it."""
    attn = adapter_parameter_count(SMOLLM3_3B, 16, LORA_TARGETS_ATTENTION)
    every = adapter_parameter_count(SMOLLM3_3B, 16, LORA_TARGETS_ALL)
    assert attn / SMOLLM3_TOTAL_PARAMS == pytest.approx(0.0025, abs=0.0002)
    assert attn < every / 3


def test_grouped_query_attention_shrinks_the_kv_adapters():
    shapes = linear_shapes(SMOLLM3_3B)
    assert shapes["k_proj"] == shapes["v_proj"]
    assert shapes["k_proj"][1] < shapes["q_proj"][1]


def test_head_dim_is_derived_when_absent():
    without = {k: v for k, v in SMOLLM3_3B.items() if k != "head_dim"}
    assert adapter_parameter_count(without) == adapter_parameter_count(SMOLLM3_3B)


# --- Pass 2: the memory estimate --------------------------------------------


def _memory(method: str) -> float:
    trainable = adapter_parameter_count(SMOLLM3_3B, rank=16)
    return estimate_training_memory(
        SMOLLM3_TOTAL_PARAMS, method,
        trainable_params=trainable, activation_gib=0.0,
    ).total_gib


def test_full_finetuning_a_3b_does_not_fit_a_24gib_card():
    assert _memory("full") == pytest.approx(45.8, abs=0.2)


def test_lora_and_qlora_both_fit_a_24gib_card():
    assert _memory("lora") == pytest.approx(6.2, abs=0.2)
    assert _memory("qlora") == pytest.approx(1.9, abs=0.2)


def test_training_costs_eight_times_inference():
    """The chapter's deep dive claims exactly 8x: 16 bytes against 2."""
    inference_gib = SMOLLM3_TOTAL_PARAMS * 2 / GIB
    assert _memory("full") / inference_gib == pytest.approx(8.0, rel=0.01)


def test_full_finetuning_fits_24gib_up_to_about_1_6b():
    """The threshold the chapter tells readers to plan around."""
    fits = estimate_training_memory(1_600_000_000, "full", activation_gib=0.0)
    does_not = estimate_training_memory(1_700_000_000, "full", activation_gib=0.0)
    assert fits.fits_in(24.0)
    assert not does_not.fits_in(24.0)


def test_unknown_method_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        estimate_training_memory(1_000_000, "peft-ish")


# --- Pass 3, guard one: loss masking ----------------------------------------


def test_supervised_fraction_matches_the_worked_example():
    """14 masked prompt positions, 5 supervised completion positions."""
    labels = [IGNORE_INDEX] * 14 + [10, 11, 12, 13, 14]
    assert supervised_fraction(labels) == pytest.approx(5 / 19)


def test_unmasked_labels_raise_rather_than_training_on_the_prompt():
    with pytest.raises(LossMaskNotApplied):
        check_loss_mask(list(range(100)))


def test_a_properly_masked_batch_passes_and_returns_its_fraction():
    labels = [IGNORE_INDEX] * 14 + [10, 11, 12, 13, 14]
    assert check_loss_mask(labels) == pytest.approx(5 / 19)


def test_threshold_sits_below_one_so_the_all_supervised_case_is_caught():
    """A hard equality check would pass exactly the case this guards."""
    almost_all = [IGNORE_INDEX] + list(range(99))
    with pytest.raises(LossMaskNotApplied):
        check_loss_mask(almost_all)


def test_empty_labels_are_zero_rather_than_a_division_error():
    assert supervised_fraction([]) == 0.0


# --- Pass 3, guard two: template drift --------------------------------------


IM_START = "<|im_start|>user\nhi<|im_end|>\n"


def test_identical_renders_pass():
    check_template_match(IM_START, IM_START)


def test_a_differing_token_raises_and_names_the_offset():
    with pytest.raises(TemplateDrift, match="diverge at character"):
        check_template_match(IM_START, IM_START.replace("user", "User"))


def test_a_missing_generation_prompt_is_caught_as_a_prefix():
    """The most common real form of this bug: one side appends the
    generation prompt and the other does not. A zip comparison alone
    reports no difference at all, so the prefix branch has to exist."""
    with pytest.raises(TemplateDrift, match="prefix"):
        check_template_match(IM_START + "<|im_start|>assistant\n", IM_START)


def test_a_trailing_newline_is_drift_and_not_noise():
    with pytest.raises(TemplateDrift):
        check_template_match(IM_START, IM_START + "\n")


# --- Pass 6: the deployment gate --------------------------------------------


GATE = DeploymentGate(
    min_quality=0.80,
    max_p95_latency_s=1.5,
    max_memory_gib=6.0,
    max_artifact_gib=2.0,
)


def test_a_candidate_inside_every_bound_has_no_failures():
    assert GATE.failures(0.84, 1.2, 5.5, 1.8) == []


def test_quality_alone_does_not_pass_the_gate():
    """The chapter's position: an adaptation can move all four terms."""
    failures = GATE.failures(0.95, 2.4, 5.5, 1.8)
    assert len(failures) == 1
    assert "p95" in failures[0]


def test_every_failure_is_reported_not_just_the_first():
    """Learning about these one release at a time is how a loop
    becomes a quarter."""
    failures = GATE.failures(0.10, 9.0, 99.0, 99.0)
    assert len(failures) == 4


def test_a_merged_adapter_that_grew_the_artifact_is_caught():
    failures = GATE.failures(0.84, 1.2, 5.5, 2.6)
    assert len(failures) == 1
    assert "artifact" in failures[0]
