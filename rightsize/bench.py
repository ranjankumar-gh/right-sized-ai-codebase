"""Measure a model against an envelope, and keep the result interpretable.

Introduced in Chapter 2. The organizing idea is that a measurement is not a
number: it is a number plus the conditions that produced it. A RunRecord
carries both, so a result stays meaningful after everyone has forgotten how
it was produced, and so two results can be refused as incomparable rather
than silently averaged.
"""

import hashlib
import statistics
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from rightsize.sizing import Footprint

# Controlled fields define what a measurement is *of*. Two records may differ
# in exactly one of these and still be comparable; the difference is then
# what the comparison is about. Differ in two and the comparison attributes
# an effect to the wrong cause.
CONTROLLED_FIELDS = frozenset(
    {
        "model",
        "quantization",
        "runtime",
        "task_set_fingerprint",
        "max_tokens",
        "batch_size",
    }
)


@dataclass(frozen=True)
class Generation:
    """One completion, with the timings only the runner can observe."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float
    peak_memory_gib: float

    @property
    def decode_tokens_per_sec(self) -> float:
        """Sustained decode rate, with prefill excluded.

        The first token arrives at the end of prefill, so it is not a
        decode event. Including it inflates the rate on short completions
        and makes prefill cost invisible, which is the whole reason this
        is reported separately from ttft_s.
        """
        decode_s = self.total_s - self.ttft_s
        if decode_s <= 0 or self.completion_tokens <= 1:
            return 0.0
        return (self.completion_tokens - 1) / decode_s


@runtime_checkable
class Runner(Protocol):
    """One execution path: a runtime, a model, and a quantization.

    Implementations wrap transformers, llama.cpp, Ollama, vLLM, or a
    recorded trace. The harness treats them as interchangeable, which is
    what makes a like-for-like comparison across runtimes possible.

    Marked runtime_checkable so a custom runner can be isinstance-checked.
    Note that this verifies method names only, not signatures: a runner with
    a wrong generate() signature satisfies the check and fails when called.
    """

    name: str

    def generate(self, prompt: str, max_tokens: int) -> Generation: ...


@dataclass(frozen=True)
class Task:
    """One prompt, optionally with what a correct answer must contain."""

    prompt: str
    expected: str | None = None


@dataclass(frozen=True)
class TaskSet:
    """The prompts a measurement was taken over.

    Fingerprinted rather than merely named, because a task set that gains
    three examples is a different task set, and a name does not notice.
    """

    name: str
    tasks: tuple[Task, ...]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for task in self.tasks:
            digest.update(task.prompt.encode("utf-8"))
            digest.update(b"\x00")
            digest.update((task.expected or "").encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:12]


def contains_score(generated: str, expected: str) -> float:
    """Deliberately minimal: does the completion contain the answer?

    Not a good metric. Do not use it for a release decision.
    """
    return 1.0 if expected.strip().lower() in generated.strip().lower() else 0.0


@dataclass(frozen=True)
class Measurement:
    """What was observed, aggregated over a task set."""

    ttft_p50_s: float
    ttft_p95_s: float
    decode_tokens_per_sec: float
    peak_memory_gib: float
    quality: float
    samples: int


def measure(
    runner: Runner,
    task_set: TaskSet,
    *,
    max_tokens: int = 128,
    warmup_runs: int = 2,
) -> Measurement:
    """Run a task set through a runner and aggregate the result.

    The warmup runs are discarded. Without them the first measurement
    carries weight loading, allocator growth, and kernel autotuning, which
    can dominate the numbers by an order of magnitude and has nothing to
    do with steady-state behavior.
    """
    if not task_set.tasks:
        raise ValueError("cannot measure an empty task set")

    warmup_prompt = task_set.tasks[0].prompt
    for _ in range(warmup_runs):
        runner.generate(warmup_prompt, max_tokens)

    generations = [
        runner.generate(task.prompt, max_tokens) for task in task_set.tasks
    ]

    scored = [
        contains_score(gen.text, task.expected)
        for task, gen in zip(task_set.tasks, generations)
        if task.expected is not None
    ]
    ttfts = sorted(gen.ttft_s for gen in generations)

    return Measurement(
        ttft_p50_s=statistics.median(ttfts),
        ttft_p95_s=_percentile(ttfts, 0.95),
        # Pooled across generations rather than averaged per generation, so
        # long completions carry the weight they deserve.
        decode_tokens_per_sec=_pooled_decode_rate(generations),
        peak_memory_gib=max(gen.peak_memory_gib for gen in generations),
        quality=statistics.fmean(scored) if scored else float("nan"),
        samples=len(generations),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Honest about small samples: with 8 values
    the p95 is the largest one, and pretending otherwise invents precision."""
    if not sorted_values:
        raise ValueError("no values")
    index = min(
        len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1)))
    )
    return sorted_values[index]


def _pooled_decode_rate(generations: list[Generation]) -> float:
    decode_tokens = sum(max(g.completion_tokens - 1, 0) for g in generations)
    decode_seconds = sum(max(g.total_s - g.ttft_s, 0.0) for g in generations)
    return decode_tokens / decode_seconds if decode_seconds > 0 else 0.0


@dataclass(frozen=True)
class RunRecord:
    """A measurement bound to the conditions that produced it."""

    # Controlled: what the measurement is of. compare() checks
    # task_set_fingerprint, not the task_set name.
    model: str
    quantization: str
    runtime: str
    task_set: str
    task_set_fingerprint: str
    max_tokens: int
    batch_size: int
    # Informational: context for a human reading this in six months.
    hardware: str
    recorded_at: str
    measurement: Measurement
    predicted: Footprint | None = None
    notes: str = ""

    def memory_prediction_gap_gib(self) -> float | None:
        """Observed peak memory minus the midpoint of the prediction.

        Positive means the machine used more than the arithmetic expected,
        which is the common direction and usually means runtime overhead
        above the budgeted band. Negative means the estimate was
        pessimistic, often because the cache never filled.

        Returns None when nothing was predicted, rather than pretending a
        gap of zero.
        """
        if self.predicted is None:
            return None
        midpoint = (
            self.predicted.total_low_gib + self.predicted.total_high_gib
        ) / 2
        return self.measurement.peak_memory_gib - midpoint


class IncomparableRecords(Exception):
    """Raised when a comparison would attribute an effect to the wrong cause."""


@dataclass(frozen=True)
class Comparison:
    """A diff between two records, labelled by what actually varied."""

    varied: tuple[str, ...]
    quality_delta: float
    ttft_p95_delta_s: float
    decode_rate_delta: float
    peak_memory_delta_gib: float

    @property
    def is_repeatability_check(self) -> bool:
        """No controlled field varied, so this measures the harness itself."""
        return not self.varied


def compare(before: RunRecord, after: RunRecord) -> Comparison:
    """Diff two records, permitting exactly one controlled field to vary.

    This is the one-variable rule made executable. Vary the quantization
    and the diff is about quantization. Vary quantization and the task set
    together and the diff is about nothing, so this refuses rather than
    returning a number someone will quote.
    """
    varied = tuple(
        field
        for field in sorted(CONTROLLED_FIELDS)
        if getattr(before, field) != getattr(after, field)
    )
    if len(varied) > 1:
        raise IncomparableRecords(
            "cannot attribute a difference when "
            f"{len(varied)} controlled fields vary: {', '.join(varied)}. "
            "Hold all but one fixed and measure again."
        )

    first, second = before.measurement, after.measurement
    return Comparison(
        varied=varied,
        quality_delta=second.quality - first.quality,
        ttft_p95_delta_s=second.ttft_p95_s - first.ttft_p95_s,
        decode_rate_delta=(
            second.decode_tokens_per_sec - first.decode_tokens_per_sec
        ),
        peak_memory_delta_gib=second.peak_memory_gib - first.peak_memory_gib,
    )


def with_notes(record: RunRecord, notes: str) -> RunRecord:
    """Annotate a record without touching what was measured."""
    return replace(record, notes=notes)
