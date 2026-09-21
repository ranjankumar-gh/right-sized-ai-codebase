"""Chapter 5: pruning plans, the embedding floor, and the distillation objective."""

import math

import pytest

from rightsize.capability import parameter_budget
from rightsize.compress import (
    CompressionViability,
    calibration_coverage,
    deepest_useful_cut,
    distillation_mode,
    kd_divergence,
    prune_depth,
    prune_width,
    pruning_floor_params,
    select_survivors,
    softmax,
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


# --- Pass 1: the cut --------------------------------------------------------


def test_halving_the_mlp_removes_40_percent_not_half():
    """The chapter's central arithmetic surprise: the MLP holds 87% of the
    computing parameters, but the vocabulary table is not a computing
    parameter and does not move."""
    base = parameter_budget(SMOLLM3_3B).total
    pruned = parameter_budget(prune_width(SMOLLM3_3B, mlp_keep=0.5)).total
    assert pruned / base == pytest.approx(0.60, abs=0.01)
    assert pruned / base > 0.5


def test_depth_pruning_is_linear_in_the_non_embedding_parameters():
    base = parameter_budget(SMOLLM3_3B)
    two_thirds = parameter_budget(prune_depth(SMOLLM3_3B, 24 / 36))
    assert two_thirds.non_embedding / base.non_embedding == pytest.approx(
        24 / 36, abs=0.001
    )


def test_pruning_raises_the_embedding_tax():
    """9% at the base, 16% after halving the depth. Every cut makes the
    remaining model a larger fraction lookup table."""
    assert parameter_budget(SMOLLM3_3B).embedding_fraction == pytest.approx(
        0.085, abs=0.005
    )
    deep_cut = parameter_budget(prune_depth(SMOLLM3_3B, 0.5))
    assert deep_cut.embedding_fraction == pytest.approx(0.157, abs=0.005)


def test_width_pruning_preserves_the_grouped_query_ratio():
    """Cutting the ratio would change the KV cache Ch1 sized, which is a
    deployment decision and not a capacity one."""
    pruned = prune_width(SMOLLM3_3B, head_keep=0.5)
    before = SMOLLM3_3B["num_attention_heads"] / SMOLLM3_3B["num_key_value_heads"]
    after = pruned["num_attention_heads"] / pruned["num_key_value_heads"]
    assert before == after


def test_a_cut_never_removes_the_last_layer_or_head():
    assert prune_depth(SMOLLM3_3B, 0.0)["num_hidden_layers"] == 1
    assert prune_width(SMOLLM3_3B, head_keep=0.0)["num_attention_heads"] == 1


# --- Pass 2: the floor ------------------------------------------------------


def test_the_floor_is_the_embedding_table():
    assert pruning_floor_params(SMOLLM3_3B) / 1e6 == pytest.approx(262.7, abs=0.5)


def test_no_structural_cut_reaches_below_the_floor():
    floor = pruning_floor_params(SMOLLM3_3B)
    savage = prune_width(prune_depth(SMOLLM3_3B, 0.02), mlp_keep=0.01, head_keep=0.01)
    assert parameter_budget(savage).total > floor


def test_a_500m_target_leaves_almost_no_computing_parameters():
    """The callout's claim: at 500M the vocabulary is the design problem."""
    keep = deepest_useful_cut(SMOLLM3_3B, 500_000_000)
    assert keep is not None
    assert keep < 0.1


def test_a_target_below_the_floor_returns_none_rather_than_a_bad_fraction():
    assert deepest_useful_cut(SMOLLM3_3B, 200_000_000) is None


def test_a_target_above_the_model_needs_no_cut():
    assert deepest_useful_cut(SMOLLM3_3B, 10_000_000_000) == 1.0


# --- Pass 4: the mode decision ----------------------------------------------


def test_matching_vocabularies_allow_logit_distillation():
    assert distillation_mode(SMOLLM3_3B, SMOLLM3_3B) == "logit"


def test_a_pruned_student_keeps_logit_distillation_available():
    """Minitron's comfortable case: the student is the teacher with
    pieces removed, so the vocabulary matches by construction."""
    student = prune_depth(SMOLLM3_3B, 0.5)
    assert distillation_mode(SMOLLM3_3B, student) == "logit"


def test_shrinking_the_vocabulary_removes_logit_distillation():
    """The coupling the chapter warns about: the cut that reaches the
    parameter target eliminates the recovery method."""
    smaller_vocab = {**SMOLLM3_3B, "vocab_size": 32000}
    assert distillation_mode(SMOLLM3_3B, smaller_vocab) == "sequence"


# --- Pass 5: the objective --------------------------------------------------


def test_a_student_matching_its_teacher_scores_zero():
    assert kd_divergence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_divergence_is_positive_when_they_disagree():
    assert kd_divergence([3.0, 1.0, 1.0], [1.0, 1.0, 3.0]) > 0.0


def test_forward_kl_punishes_a_student_that_drops_a_teacher_mode():
    """The mode-covering property, on the case Exercise 2 asks about: a
    bimodal teacher against a student that committed to one mode.

    Forward KL (teacher to student) is the large number, because the
    teacher holds mass where the student holds almost none. The reverse
    is far smaller. A mirror-image pair would be symmetric by accident,
    so the example has to be genuinely lopsided to show anything.
    """
    bimodal_teacher = [3.0, 3.0, 0.0]
    committed_student = [6.0, 0.0, 0.0]
    forward = kd_divergence(committed_student, bimodal_teacher)
    reverse = kd_divergence(bimodal_teacher, committed_student)
    assert forward > reverse * 1.5


def test_temperature_squared_scaling_is_applied():
    """Without the T^2 factor, raising the temperature would silently
    shrink this term against any hard-label term beside it."""
    student, teacher = [3.0, 1.0, 1.0], [1.0, 1.0, 3.0]

    def raw_kl(temperature: float) -> float:
        p = softmax(teacher, temperature)
        q = softmax(student, temperature)
        return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)

    for temperature in (1.0, 2.0, 4.0):
        assert kd_divergence(student, teacher, temperature) == pytest.approx(
            raw_kl(temperature) * temperature**2
        )

    # And the factor is load-bearing: the unscaled divergence collapses
    # as temperature rises, which is exactly what it compensates for.
    assert raw_kl(4.0) < raw_kl(1.0)


def test_softmax_is_a_distribution_and_survives_large_logits():
    assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert not math.isnan(sum(softmax([1000.0, 1001.0])))


# --- Pass 6: the viability gate ---------------------------------------------


READY = CompressionViability(
    quantization_attempted=True,
    constraint_durable_months=24,
    teacher_output_licensed=True,
    healing_tokens_available_b=5.0,
    hard_case_slice_exists=True,
)


def test_a_ready_project_has_no_blockers():
    assert READY.blockers() == []


def test_skipping_quantization_is_a_blocker():
    import dataclasses

    not_tried = dataclasses.replace(READY, quantization_attempted=False)
    assert any("quantization" in b for b in not_tried.blockers())


def test_a_short_lived_constraint_is_a_blocker():
    """The entry condition that most often fails: compressing for hardware
    that is replaced before the burden is paid off."""
    import dataclasses

    brief = dataclasses.replace(READY, constraint_durable_months=8)
    assert any("outlives" in b for b in brief.blockers())


def test_missing_hard_case_slice_is_a_blocker():
    """Without it the evaluation cannot detect the failure mode that
    opened the chapter."""
    import dataclasses

    no_slice = dataclasses.replace(READY, hard_case_slice_exists=False)
    assert any("hard-case slice" in b for b in no_slice.blockers())


def test_unlicensed_teacher_output_is_a_blocker():
    import dataclasses

    unlicensed = dataclasses.replace(READY, teacher_output_licensed=False)
    assert any("licensed" in b for b in unlicensed.blockers())


def test_every_blocker_is_reported_at_once():
    nothing_ready = CompressionViability(False, 1, False, 0.0, False)
    assert len(nothing_ready.blockers()) == 5


# --- Pass 3: the importance criterion ---------------------------------------


def test_survivors_are_the_highest_scoring_units():
    importance = {0: 0.9, 1: 0.1, 2: 0.5, 3: 0.3, 4: 0.7}
    assert select_survivors(importance, 0.6) == [0, 2, 4]


def test_survivors_come_back_in_model_order_not_rank_order():
    """The caller indexes weight tensors with these, so order matters."""
    importance = {0: 0.1, 1: 0.9, 2: 0.5}
    assert select_survivors(importance, 1.0) == [0, 1, 2]


def test_ties_break_deterministically_so_two_runs_cut_the_same_units():
    importance = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    assert select_survivors(importance, 0.5) == select_survivors(importance, 0.5)
    assert select_survivors(importance, 0.5) == [0, 1]


def test_at_least_one_unit_always_survives():
    assert len(select_survivors({0: 0.1, 1: 0.2}, 0.01)) == 1


def test_an_invalid_keep_fraction_raises_rather_than_emptying_the_layer():
    with pytest.raises(ValueError):
        select_survivors({0: 1.0}, 0.0)
    with pytest.raises(ValueError):
        select_survivors({0: 1.0}, 1.5)


def test_traffic_sampled_calibration_under_represents_nothing_by_construction():
    """Sampling traffic gives coverage near 1.0 everywhere, which is
    exactly why it feels rigorous and causes the chapter's failure."""
    coverage = calibration_coverage(
        {"common": 980, "rare": 20}, {"common": 0.98, "rare": 0.02}
    )
    assert coverage["common"] == pytest.approx(1.0, abs=0.01)
    assert coverage["rare"] == pytest.approx(1.0, abs=0.01)


def test_a_missing_slice_is_reported_as_zero_coverage():
    """The failure mode the callout warns about: capacity serving a slice
    absent from calibration scores as unimportant and gets cut."""
    coverage = calibration_coverage(
        {"common": 1000}, {"common": 0.98, "rare": 0.02}
    )
    assert coverage["rare"] == 0.0


def test_stratified_calibration_over_represents_the_rare_slice():
    coverage = calibration_coverage(
        {"common": 500, "rare": 500}, {"common": 0.98, "rare": 0.02}
    )
    assert coverage["rare"] > 1.0
    assert coverage["common"] < 1.0


def test_an_empty_calibration_set_is_zero_coverage_not_a_crash():
    assert calibration_coverage({}, {"common": 1.0}) == {"common": 0.0}
