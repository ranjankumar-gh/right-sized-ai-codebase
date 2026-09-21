"""Chapter 8: the masking mechanism, the constraint tax, and schema pressure."""

import pytest

from rightsize.structured import (
    NEG_INF,
    SchemaComplexity,
    StructuredOutcome,
    StructuredReport,
    allowed_continuations,
    mask_logits,
    pick,
    schema_complexity,
)

DECISIONS = ["APPROVE", "REJECT", "ESCALATE"]


# --- Pass 1: the mechanism --------------------------------------------------


def test_an_empty_prefix_allows_every_first_character():
    assert allowed_continuations("", DECISIONS) == {"A", "R", "E"}


def test_a_prefix_narrows_the_legal_set():
    assert allowed_continuations("A", DECISIONS) == {"P"}
    assert allowed_continuations("E", DECISIONS) == {"S"}


def test_a_complete_option_has_no_continuations():
    assert allowed_continuations("APPROVE", DECISIONS) == set()


def test_a_prefix_matching_nothing_has_no_continuations():
    assert allowed_continuations("Z", DECISIONS) == set()


def test_a_shared_prefix_keeps_both_branches_alive():
    assert allowed_continuations("A", ["APPROVE", "APPEND"]) == {"P"}
    assert allowed_continuations("APP", ["APPROVE", "APPEND"]) == {"R", "E"}


def test_masking_sets_illegal_tokens_to_negative_infinity():
    masked = mask_logits({"A": 0.2, "E": 0.6, "Z": 0.9}, {"A", "E"})
    assert masked["Z"] == NEG_INF
    assert masked["A"] == 0.2


def test_masking_depends_only_on_the_allowed_set():
    """The mask is computed from the grammar and the output so far, never
    from the model's preferences. That is why constrained decoding
    composes with quantization, adapters, and distillation."""
    confident = mask_logits({"A": 0.1, "E": 0.9}, allowed={"A"})
    unsure = mask_logits({"A": 0.4, "E": 0.5}, allowed={"A"})
    assert confident["E"] == NEG_INF
    assert unsure["E"] == NEG_INF
    assert confident["A"] == 0.1
    assert unsure["A"] == 0.4


def test_pick_returns_the_highest_legal_token():
    assert pick({"A": 0.2, "E": 0.6, "R": 0.19}) == "E"
    assert pick(mask_logits({"A": 0.2, "E": 0.6, "R": 0.19}, {"A", "R"})) == "A"


def test_pick_returns_none_when_the_mask_leaves_nothing():
    """A dead end in a hand-written grammar. Reporting it beats emitting a
    stop token and pretending the output finished."""
    assert pick(mask_logits({"A": 0.2, "E": 0.6}, set())) is None


# --- Pass 2: where the guarantee does damage --------------------------------


BELIEFS = {"E": 0.61, "A": 0.20, "R": 0.19}


def test_masking_changes_nothing_when_the_schema_allows_the_answer():
    allowed = allowed_continuations("", DECISIONS)
    assert pick(mask_logits(BELIEFS, allowed)) == pick(BELIEFS) == "E"


def test_removing_the_escape_hatch_forces_a_low_confidence_answer():
    """The chapter's central demonstration: a 61% belief the schema cannot
    express becomes a 20% answer, delivered with no signal that it was the
    model's third preference."""
    binary = allowed_continuations("", ["APPROVE", "REJECT"])
    forced = pick(mask_logits(BELIEFS, binary))
    assert forced == "A"
    assert BELIEFS[forced] == pytest.approx(0.20)
    assert BELIEFS[forced] < BELIEFS["E"]


def test_the_suppressed_preference_outranked_everything_legal():
    """What Exercise 3 asks the reader to alert on."""
    binary = allowed_continuations("", ["APPROVE", "REJECT"])
    masked = mask_logits(BELIEFS, binary)
    best_legal = max(v for v in masked.values() if v != NEG_INF)
    assert BELIEFS["E"] > best_legal * 3


# --- Pass 3: the four-number report -----------------------------------------


def _report(*triples) -> StructuredReport:
    return StructuredReport(
        [StructuredOutcome(v, c, e) for v, c, e in triples]
    )


def test_the_four_rates_are_computed_independently():
    report = _report(
        (True, True, True),
        (True, False, True),
        (True, False, True),
        (False, False, False),
    )
    assert report.schema_validity == pytest.approx(0.75)
    assert report.answer_accuracy == pytest.approx(0.25)
    assert report.executable_accuracy == pytest.approx(0.25)
    assert report.wrong_but_valid == pytest.approx(0.50)


def test_perfect_validity_can_coexist_with_poor_accuracy():
    """The shape of the constraint tax: every output parses, most are wrong."""
    report = _report(*([(True, False, True)] * 9 + [(True, True, True)]))
    assert report.schema_validity == 1.0
    assert report.answer_accuracy == pytest.approx(0.10)
    assert report.wrong_but_valid == pytest.approx(0.90)


def test_executable_accuracy_requires_both_execution_and_correctness():
    """A well-formed wrong tool call runs, which is the worse failure."""
    report = _report((True, True, False), (True, False, True))
    assert report.executable_accuracy == 0.0


def test_an_empty_report_is_zero_rather_than_a_division_error():
    assert StructuredReport([]).schema_validity == 0.0
    assert StructuredReport([]).wrong_but_valid == 0.0


def test_a_regression_hidden_by_rising_validity_is_reported():
    """The constraint tax is a regression every conventional dashboard
    records as an improvement."""
    before = _report((False, False, False), (True, True, True))
    after = _report((True, False, True), (True, False, True))
    problems = after.regression_against(before)
    assert after.schema_validity > before.schema_validity
    assert any("answer accuracy fell" in p for p in problems)
    assert any("wrong-but-valid rose" in p for p in problems)


def test_a_genuine_improvement_reports_no_regression():
    before = _report((True, False, True), (True, False, True))
    after = _report((True, True, True), (True, True, True))
    assert after.regression_against(before) == []


# --- Pass 4: schema pressure ------------------------------------------------


FLAT = {
    "properties": {
        "decision": {"enum": ["APPROVE", "REJECT", "ESCALATE"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
}

NESTED = {
    "properties": {
        "result": {
            "properties": {
                "decision": {"enum": ["APPROVE", "REJECT", "ESCALATE"]},
                "evidence": {
                    "properties": {
                        "source": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["source", "confidence"],
                },
            },
            "required": ["decision"],
        },
    },
    "required": ["result"],
}


def test_a_flat_schema_reports_depth_one():
    complexity = schema_complexity(FLAT)
    assert complexity.max_depth == 1
    assert complexity.field_count == 2
    assert complexity.enum_branches == 3


def test_nesting_is_counted_through_every_level():
    complexity = schema_complexity(NESTED)
    assert complexity.max_depth == 3
    assert complexity.field_count == 5


def test_the_same_information_nested_carries_more_pressure():
    """The chapter's lever: these two schemas carry the same information
    and only one asks a small model to hold three levels while deciding."""
    assert schema_complexity(NESTED).pressure_score > (
        schema_complexity(FLAT).pressure_score * 2
    )


def test_depth_is_weighted_harder_than_field_count():
    """Published degradation tracks nesting rather than field count."""
    deep = SchemaComplexity(max_depth=3, field_count=1, required_count=0,
                            enum_branches=0)
    wide = SchemaComplexity(max_depth=1, field_count=3, required_count=0,
                            enum_branches=0)
    assert deep.pressure_score > wide.pressure_score


def test_an_empty_schema_is_depth_one_and_costs_nothing_structural():
    complexity = schema_complexity({})
    assert complexity.max_depth == 1
    assert complexity.field_count == 0
    assert complexity.required_count == 0


def test_a_schema_with_no_properties_key_does_not_crash():
    assert schema_complexity({"required": ["a"]}).required_count == 1
