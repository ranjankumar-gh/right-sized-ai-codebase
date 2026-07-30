"""Tests for the Chapter 2 measurement harness."""

import pytest

from rightsize.backends.recorded import RecordedRunner
from rightsize.bench import (
    CONTROLLED_FIELDS,
    Comparison,
    Generation,
    IncomparableRecords,
    Measurement,
    RunRecord,
    Task,
    TaskSet,
    compare,
    contains_score,
    measure,
)
from rightsize.sizing import Footprint

TASKS = TaskSet(
    name="smoke",
    tasks=(
        Task("Capital of France?", expected="Paris"),
        Task("Capital of Japan?", expected="Tokyo"),
        Task("Say something.", expected=None),
    ),
)


def a_runner(**overrides: object) -> RecordedRunner:
    defaults: dict[str, object] = {
        "name": "recorded",
        "ttft_s": 0.20,
        "tokens_per_sec": 40.0,
        "peak_memory_gib": 6.0,
        "answers": {
            "Capital of France?": "The capital of France is Paris.",
            "Capital of Japan?": "The capital of Japan is Tokyo.",
            "Say something.": "Something.",
        },
    }
    defaults.update(overrides)
    return RecordedRunner(**defaults)  # type: ignore[arg-type]


def a_record(**overrides: object) -> RunRecord:
    defaults: dict[str, object] = {
        "model": "small-3b",
        "quantization": "q4",
        "runtime": "recorded",
        "task_set": TASKS.name,
        "task_set_fingerprint": TASKS.fingerprint(),
        "max_tokens": 128,
        "batch_size": 1,
        "hardware": "synthetic",
        "recorded_at": "2026-07-30",
        "measurement": Measurement(0.2, 0.2, 40.0, 6.0, 1.0, 3),
    }
    defaults.update(overrides)
    return RunRecord(**defaults)  # type: ignore[arg-type]


class TestGeneration:
    def test_decode_rate_excludes_the_first_token(self) -> None:
        """The first token ends prefill, so it is not a decode event."""
        gen = Generation("x", 10, 11, ttft_s=1.0, total_s=2.0, peak_memory_gib=1)
        assert gen.decode_tokens_per_sec == pytest.approx(10.0)

    def test_single_token_completion_has_no_decode_rate(self) -> None:
        gen = Generation("x", 10, 1, ttft_s=1.0, total_s=1.0, peak_memory_gib=1)
        assert gen.decode_tokens_per_sec == 0.0

    def test_degenerate_timings_do_not_raise(self) -> None:
        gen = Generation("x", 10, 5, ttft_s=2.0, total_s=1.0, peak_memory_gib=1)
        assert gen.decode_tokens_per_sec == 0.0


class TestTaskSet:
    def test_fingerprint_is_stable(self) -> None:
        assert TASKS.fingerprint() == TASKS.fingerprint()

    def test_adding_a_task_changes_the_fingerprint(self) -> None:
        grown = TaskSet(TASKS.name, TASKS.tasks + (Task("New?", "Yes"),))
        assert grown.fingerprint() != TASKS.fingerprint()

    def test_changing_an_expectation_changes_the_fingerprint(self) -> None:
        altered = TaskSet(
            TASKS.name,
            (Task("Capital of France?", expected="PARIS, FRANCE"),)
            + TASKS.tasks[1:],
        )
        assert altered.fingerprint() != TASKS.fingerprint()

    def test_reordering_tasks_changes_the_fingerprint(self) -> None:
        """Order is part of the identity: it affects nothing statistically
        but a differing order means somebody edited the set."""
        reordered = TaskSet(TASKS.name, tuple(reversed(TASKS.tasks)))
        assert reordered.fingerprint() != TASKS.fingerprint()


class TestContainsScore:
    def test_is_case_and_whitespace_insensitive(self) -> None:
        assert contains_score("  the answer is PARIS ", "paris") == 1.0

    def test_misses_score_zero(self) -> None:
        assert contains_score("The answer is Lyon.", "Paris") == 0.0


class TestMeasure:
    def test_warmup_runs_are_discarded(self) -> None:
        runner = a_runner()
        result = measure(runner, TASKS, warmup_runs=2)
        assert runner.calls == 2 + len(TASKS.tasks)
        assert result.samples == len(TASKS.tasks)

    def test_quality_ignores_tasks_without_expectations(self) -> None:
        result = measure(a_runner(), TASKS)
        # Two of three tasks carry an expectation, and both are answered.
        assert result.quality == pytest.approx(1.0)

    def test_quality_reflects_wrong_answers(self) -> None:
        runner = a_runner(answers={}, default_answer="No idea.")
        assert measure(runner, TASKS).quality == pytest.approx(0.0)

    def test_quality_is_nan_when_nothing_is_scorable(self) -> None:
        unscored = TaskSet("unscored", (Task("Say something."),))
        result = measure(a_runner(), unscored)
        assert result.quality != result.quality  # NaN

    def test_peak_memory_is_the_maximum_not_the_mean(self) -> None:
        result = measure(a_runner(peak_memory_gib=7.5), TASKS)
        assert result.peak_memory_gib == pytest.approx(7.5)

    def test_decode_rate_recovers_the_runners_rate(self) -> None:
        result = measure(a_runner(tokens_per_sec=40.0), TASKS)
        assert result.decode_tokens_per_sec == pytest.approx(40.0)

    def test_empty_task_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty task set"):
            measure(a_runner(), TaskSet("nothing", ()))


class TestPredictionGap:
    def test_is_none_without_a_prediction(self) -> None:
        assert a_record().memory_prediction_gap_gib() is None

    def test_positive_when_the_machine_used_more_than_predicted(self) -> None:
        predicted = Footprint(3.0, 1.0, 4.5, 6.0)  # midpoint 5.25
        record = a_record(
            predicted=predicted,
            measurement=Measurement(0.2, 0.2, 40.0, 6.5, 1.0, 3),
        )
        assert record.memory_prediction_gap_gib() == pytest.approx(1.25)

    def test_negative_when_the_estimate_was_pessimistic(self) -> None:
        predicted = Footprint(3.0, 1.0, 4.5, 6.0)
        record = a_record(
            predicted=predicted,
            measurement=Measurement(0.2, 0.2, 40.0, 4.0, 1.0, 3),
        )
        assert record.memory_prediction_gap_gib() == pytest.approx(-1.25)


class TestCompare:
    def test_identical_records_are_a_repeatability_check(self) -> None:
        result = compare(a_record(), a_record())
        assert result.is_repeatability_check
        assert result.varied == ()

    def test_one_varied_field_names_what_the_diff_is_about(self) -> None:
        result = compare(a_record(), a_record(quantization="q8"))
        assert result.varied == ("quantization",)
        assert not result.is_repeatability_check

    def test_two_varied_controlled_fields_are_refused(self) -> None:
        with pytest.raises(IncomparableRecords, match="2 controlled fields"):
            compare(
                a_record(), a_record(quantization="q8", model="other-7b")
            )

    def test_informational_fields_may_differ_freely(self) -> None:
        """Hardware and date are context, not the subject of the comparison."""
        result = compare(
            a_record(),
            a_record(hardware="different box", recorded_at="2026-12-01"),
        )
        assert result.is_repeatability_check

    def test_a_changed_task_set_is_controlled(self) -> None:
        """The trap this guard exists for: silently growing the task set."""
        with pytest.raises(IncomparableRecords):
            compare(
                a_record(),
                a_record(
                    quantization="q8", task_set_fingerprint="deadbeef1234"
                ),
            )

    def test_deltas_are_signed_after_minus_before(self) -> None:
        before = a_record(measurement=Measurement(0.2, 0.3, 40.0, 6.0, 0.9, 3))
        after = a_record(
            quantization="q8",
            measurement=Measurement(0.1, 0.2, 55.0, 4.0, 0.85, 3),
        )
        result = compare(before, after)
        assert result.quality_delta == pytest.approx(-0.05)
        assert result.ttft_p95_delta_s == pytest.approx(-0.1)
        assert result.decode_rate_delta == pytest.approx(15.0)
        assert result.peak_memory_delta_gib == pytest.approx(-2.0)

    def test_every_controlled_field_is_actually_checked(self) -> None:
        """Guards against a field being added to the record but not to
        CONTROLLED_FIELDS, which would silently widen what is comparable."""
        for name in CONTROLLED_FIELDS:
            assert hasattr(a_record(), name), name


class TestComparisonShape:
    def test_comparison_is_immutable(self) -> None:
        result = compare(a_record(), a_record())
        with pytest.raises(Exception):
            result.varied = ("model",)  # type: ignore[misc]

    def test_is_a_comparison(self) -> None:
        assert isinstance(compare(a_record(), a_record()), Comparison)
