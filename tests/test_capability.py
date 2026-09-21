"""Chapter 3: reading capability evidence off a model card and a config."""

import pytest

from rightsize.capability import (
    CHINCHILLA_TOKENS_PER_PARAM,
    TrainingProfile,
    parameter_budget,
    read_candidate,
)

# Values read from the published config.json of each model.
GEMMA_3_270M = {
    "vocab_size": 262144,
    "hidden_size": 640,
    "intermediate_size": 2048,
    "num_hidden_layers": 18,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "tie_word_embeddings": True,
}

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


def test_chinchilla_profile_is_one_times_by_construction():
    """Chinchilla is the definition of the ratio, so it must come back as 1x."""
    chinchilla = TrainingProfile("Chinchilla", total_params_b=70.0, train_tokens_t=1.4)
    assert chinchilla.tokens_per_param == pytest.approx(20.0)
    assert chinchilla.overtraining_multiple == pytest.approx(1.0)


def test_gpt3_is_far_below_compute_optimal():
    gpt3 = TrainingProfile("GPT-3", total_params_b=175.0, train_tokens_t=0.3)
    assert gpt3.overtraining_multiple < 0.1


def test_smaller_model_in_a_family_carries_the_higher_multiple():
    """The inverse relationship the chapter argues for, on one family."""
    big = TrainingProfile("Gemma 3 27B", 27.0, 14.0)
    small = TrainingProfile("Gemma 3 270M", 0.27, 6.0)
    assert small.overtraining_multiple > big.overtraining_multiple * 40


def test_budget_reproduces_the_published_gemma_270m_split():
    """Google published 170M embedding / 100M transformer. Reproduce it.

    This is the test that keeps parameter_budget honest: the formula is
    only worth printing in a book if it lands on a number someone else
    published independently.
    """
    budget = parameter_budget(GEMMA_3_270M)
    assert budget.embedding / 1e6 == pytest.approx(167.8, abs=0.5)
    assert budget.non_embedding / 1e6 == pytest.approx(100.3, abs=0.5)
    assert budget.total / 1e6 == pytest.approx(268.1, abs=0.5)


def test_budget_reproduces_the_released_smollm3_parameter_count():
    budget = parameter_budget(SMOLLM3_3B)
    assert budget.total / 1e9 == pytest.approx(3.08, abs=0.02)


def test_embedding_tax_dominates_the_smallest_model():
    """63% against 9%: the reason headline params stop comparing below 2B."""
    assert parameter_budget(GEMMA_3_270M).embedding_fraction > 0.6
    assert parameter_budget(SMOLLM3_3B).embedding_fraction < 0.1


def test_untying_embeddings_doubles_the_table():
    tied = parameter_budget(GEMMA_3_270M)
    untied = parameter_budget({**GEMMA_3_270M, "tie_word_embeddings": False})
    assert untied.embedding == 2 * tied.embedding
    assert untied.non_embedding == tied.non_embedding


def test_mlp_holds_far_more_than_the_textbook_two_thirds():
    """GQA plus a gated MLP push attention's share to ~13%, not 33%."""
    budget = parameter_budget(SMOLLM3_3B)
    assert budget.mlp / budget.non_embedding > 0.8
    assert budget.attention / budget.non_embedding < 0.2


def test_grouped_query_attention_shrinks_the_stored_projections():
    full_mha = parameter_budget({**SMOLLM3_3B, "num_key_value_heads": 16})
    gqa = parameter_budget(SMOLLM3_3B)
    assert gqa.attention < full_mha.attention


def test_head_dim_is_derived_when_the_config_omits_it():
    without = {k: v for k, v in SMOLLM3_3B.items() if k != "head_dim"}
    assert parameter_budget(without).attention == parameter_budget(SMOLLM3_3B).attention


def test_missing_kv_head_count_falls_back_to_full_attention():
    """Models predating GQA have no num_key_value_heads field."""
    without = {k: v for k, v in SMOLLM3_3B.items() if k != "num_key_value_heads"}
    full_mha = parameter_budget({**SMOLLM3_3B, "num_key_value_heads": 16})
    assert parameter_budget(without).attention == full_mha.attention


def test_ungated_mlp_is_two_thirds_of_a_gated_one():
    gated = parameter_budget(SMOLLM3_3B)
    ungated = parameter_budget(SMOLLM3_3B, gated_mlp=False)
    assert ungated.mlp == pytest.approx(gated.mlp * 2 / 3)


def test_read_candidate_carries_both_readings():
    candidate = read_candidate("SmolLM3 3B", SMOLLM3_3B, 3.08, 11.2)
    assert candidate.name == "SmolLM3 3B"
    assert candidate.profile.overtraining_multiple == pytest.approx(181.8, rel=0.01)
    assert candidate.budget.embedding_fraction < 0.1


def test_summary_row_lines_up_with_its_header():
    """The chapter prints a header literal above these rows; if the widths
    drift apart the book ships a table that does not align."""
    header = f"{'model':<16}{'multiple':>12}{'compute params':>16}{'tax':>7}"
    row = read_candidate("SmolLM3 3B", SMOLLM3_3B, 3.08, 11.2).summary_row()
    assert len(row) == len(header)


def test_chinchilla_constant_is_the_published_ratio():
    assert CHINCHILLA_TOKENS_PER_PARAM == 20.0
