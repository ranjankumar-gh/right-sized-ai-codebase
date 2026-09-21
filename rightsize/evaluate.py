import math
from dataclasses import dataclass

Z_95 = 1.959964
Z_POWER_80 = 0.841621


def wilson_interval(
    successes: int, trials: int, z: float = Z_95
) -> tuple[float, float]:
    """Confidence interval for a proportion, Wilson score method.

    Wilson rather than the normal approximation because the normal one
    misbehaves exactly where evaluation lives: small samples, and rates
    near 0 or 1. At 19/20 the normal approximation puts the upper bound
    above 1.0, which is not a thing an accuracy can be.
    """
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def required_trials(
    baseline: float, improvement: float,
    z_alpha: float = Z_95, z_power: float = Z_POWER_80,
) -> int:
    """Examples per arm needed to detect `improvement` over `baseline`.

    Two-proportion comparison at 95% confidence and 80% power. The
    inverse-square term is what makes small effects so expensive:
    halving the difference you want to see quadruples the examples.
    """
    candidate = baseline + improvement
    if not 0.0 < baseline < 1.0 or not 0.0 < candidate < 1.0:
        raise ValueError("baseline and baseline+improvement must be in (0, 1)")
    variance = baseline * (1 - baseline) + candidate * (1 - candidate)
    return math.ceil((z_alpha + z_power) ** 2 * variance / improvement**2)


def resolution_floor(trials: int, baseline: float = 0.70) -> float:
    """Smallest improvement this many examples can detect.

    Search rather than algebra because required_trials is not trivially
    invertible, and a tenth of a point of resolution is finer than any
    decision this informs.
    """
    if trials <= 0:
        return 1.0
    improvement = 0.001
    while improvement < 1.0 - baseline:
        if required_trials(baseline, improvement) <= trials:
            return improvement
        improvement += 0.001
    return 1.0 - baseline


@dataclass(frozen=True)
class SliceResult:
    """One named subset, scored on its own terms."""

    name: str
    successes: int
    trials: int

    @property
    def accuracy(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.trials)

    @property
    def floor(self) -> float:
        """This slice's own resolution floor, on its own size.

        Not the whole set's. A 40-example slice resolves what 40
        examples resolve, and reporting it against the full set's floor
        is how slice gates become theatre.
        """
        return resolution_floor(self.trials, baseline=self.accuracy or 0.70)


@dataclass(frozen=True)
class SlicedEvaluation:
    slices: list[SliceResult]

    @property
    def overall(self) -> float:
        total = sum(s.trials for s in self.slices)
        return sum(s.successes for s in self.slices) / total if total else 0.0

    @property
    def worst(self) -> SliceResult | None:
        return min(self.slices, key=lambda s: s.accuracy, default=None)

    def regressions_against(
        self, baseline: "SlicedEvaluation", tolerance: float = 0.0
    ) -> list[str]:
        """Slices whose accuracy fell by more than the larger of their own
        floor and the given tolerance.

        Each slice is judged against its own resolution floor rather than
        a fixed threshold, which stops a small slice generating noise
        alerts. The tolerance is a global override: it raises the bar for
        every slice whose floor is below it. Floors shrink as slices grow,
        so raising the tolerance silences the largest slices first and
        the smallest slices last.
        """
        before = {s.name: s for s in baseline.slices}
        problems = []
        for current in self.slices:
            previous = before.get(current.name)
            if previous is None:
                continue
            drop = previous.accuracy - current.accuracy
            if drop > max(current.floor, tolerance):
                problems.append(
                    f"{current.name}: {previous.accuracy:.1%} -> "
                    f"{current.accuracy:.1%} (down {drop:.1%}, "
                    f"beyond this slice's {current.floor:.1%} floor)"
                )
        return problems


@dataclass(frozen=True)
class PairwiseVerdict:
    """A judge's verdict on the same pair, presented both ways round."""

    item: str
    forward: str  # "a", "b", or "tie", with A shown first
    reversed_: str  # the same comparison with B shown first

    @property
    def is_consistent(self) -> bool:
        """Agreement requires the verdict to follow the answer, not
        the position. Shown B-first, a genuine preference for A must
        come back as 'a'."""
        return self.forward == self.reversed_


def position_bias_rate(verdicts: list[PairwiseVerdict]) -> float:
    """Share of comparisons where swapping the order changed the answer.

    This is a lower bound on the judge's error rate that costs one extra
    judge call per item and needs no labels. A rate above a few percent
    means the margins this judge reports are not real.
    """
    if not verdicts:
        return 0.0
    flips = sum(1 for v in verdicts if not v.is_consistent)
    return flips / len(verdicts)


def consistent_only(verdicts: list[PairwiseVerdict]) -> list[PairwiseVerdict]:
    """Discard comparisons the judge could not answer stably.

    Dropping them biases the remaining sample toward easy pairs, which
    is a real cost and is better than scoring on verdicts you know are
    unstable. Report how many were dropped.
    """
    return [v for v in verdicts if v.is_consistent]


@dataclass(frozen=True)
class PromotionGate:
    """Everything a candidate must clear to replace what is in production.

    Deliberately not a score. Chapter 1's envelope has four terms, this
    chapter added resolution and slices, and a gate that collapses them
    into one number can be passed by improving whichever term is
    cheapest.
    """

    max_slice_regression: float = 0.0
    max_position_bias: float = 0.05
    min_trials: int = 0
    require_resolvable: bool = True

    def blockers(
        self,
        candidate: SlicedEvaluation,
        incumbent: SlicedEvaluation,
        position_bias: float = 0.0,
    ) -> list[str]:
        problems = list(
            candidate.regressions_against(incumbent, self.max_slice_regression)
        )
        trials = sum(s.trials for s in candidate.slices)
        if trials < self.min_trials:
            problems.append(
                f"{trials} examples, below the {self.min_trials} this gate "
                f"requires"
            )
        if position_bias > self.max_position_bias:
            problems.append(
                f"judge flipped on {position_bias:.1%} of swapped pairs, "
                f"above the {self.max_position_bias:.1%} limit; margins "
                f"below that rate are not evidence"
            )
        if self.require_resolvable:
            delta = candidate.overall - incumbent.overall
            floor = resolution_floor(trials, baseline=incumbent.overall)
            if 0 < delta < floor:
                problems.append(
                    f"claimed gain of {delta:.1%} is below this set's "
                    f"{floor:.1%} resolution floor; it is not evidence of "
                    f"an improvement"
                )
        return problems


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Word n-grams. 13 follows the window used by large training runs.

    Text is split on whitespace, so the units are words, not model
    tokens. Below about 8 words, coincidental matches dominate: ordinary
    English repeats short spans constantly, and the check stops
    discriminating.
    """
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def contamination_score(example: str, corpus_ngrams: set, n: int = 13) -> float:
    """Share of the example's n-grams that appear in the corpus.

    A high score is strong evidence of leakage. A low score is weak
    evidence of cleanliness, which is the asymmetry that matters: a
    paraphrased test item shares almost no 13-grams with its original
    and is entirely invisible here.
    """
    grams = ngrams(example, n)
    if not grams:
        return 0.0
    return len(grams & corpus_ngrams) / len(grams)
