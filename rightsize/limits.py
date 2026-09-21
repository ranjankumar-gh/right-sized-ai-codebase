from dataclasses import dataclass

from rightsize.evaluate import resolution_floor


@dataclass(frozen=True)
class SizeResponse:
    """Quality measured at several model sizes on one task set.

    points are (parameters in billions, accuracy) pairs. trials is the
    size of the task set, and it is what decides whether any of this
    means anything.
    """

    points: list[tuple[float, float]]
    trials: int
    # A span this small is not worth having even if it is real, so a
    # task set precise enough to bound the effect below it has answered
    # the question rather than failed to.
    negligible_span: float = 0.02

    @property
    def span(self) -> float:
        """Accuracy range across the sizes measured."""
        if len(self.points) < 2:
            return 0.0
        accuracies = [accuracy for _, accuracy in self.points]
        return max(accuracies) - min(accuracies)

    @property
    def floor(self) -> float:
        """The smallest span this task set could resolve."""
        if not self.points:
            return 1.0
        baseline = min(accuracy for _, accuracy in self.points)
        return resolution_floor(self.trials, baseline=baseline)

    @property
    def verdict(self) -> str:
        """One of three answers, and the third is the common one.

        'unresolvable' is not a failure of the method. It is the
        method working: the task set cannot distinguish the hypotheses,
        and reporting a slope from it would be reporting noise. It is
        distinct from 'size-insensitive', which says the effect has
        been bounded and is too small to want.
        """
        if len(self.points) < 3:
            return "insufficient: measure at least three sizes"
        if self.span > self.floor:
            ordered = sorted(self.points)
            if ordered[-1][1] > ordered[0][1] + self.floor:
                return "capability-bound"
            return "size-insensitive"
        # The span is below the floor, so there are two cases and they
        # are different answers. A tight floor has bounded the effect
        # as too small to want; a loose one has established nothing.
        if self.floor <= self.negligible_span:
            return "size-insensitive"
        return "unresolvable"

    @property
    def examples_needed(self) -> int:
        """Task set size that would resolve the span observed.

        Returns the current size when the span is already resolvable.
        """
        from rightsize.evaluate import required_trials

        if self.span <= 0:
            return self.trials
        baseline = min(accuracy for _, accuracy in self.points)
        if baseline + self.span >= 1.0:
            return self.trials
        return max(self.trials, required_trials(baseline, self.span))


@dataclass(frozen=True)
class LimitCheck:
    """What must be true before escalating model size.

    Every field is cheaper than the escalation it gates. The gate
    exists because reaching for a larger model is the one intervention
    that always appears to be available, which makes it the default
    response to not knowing what else to do.
    """

    task_contract_written: bool
    adaptation_attempted: bool
    output_constrained: bool
    failures_checked_for_slice_concentration: bool
    failure_concentrated_in_a_slice: bool
    response: SizeResponse

    def blockers(self) -> list[str]:
        checks = [
            (self.task_contract_written,
             "no written task contract; Chapter 4 showed a fine-tune "
             "without one optimises whatever the data happened to encode, "
             "and the same is true of a model choice"),
            (self.adaptation_attempted,
             "adaptation not attempted; Chapter 4's ladder is cheaper than "
             "a larger model and reversible"),
            (self.output_constrained,
             "output not constrained; Chapter 8 removes a failure class "
             "without touching the weights"),
            (self.failures_checked_for_slice_concentration,
             "failures not examined by slice; Chapter 9's machinery "
             "distinguishes a slice problem from a capability limit"),
            (not self.failure_concentrated_in_a_slice,
             "failures concentrate in a slice, so this is a routing or "
             "adaptation problem: Chapter 10 sends that slice elsewhere "
             "without moving every request"),
        ]
        problems = [message for passed, message in checks if not passed]
        verdict = self.response.verdict
        if verdict == "unresolvable":
            problems.append(
                f"quality response across sizes is {self.response.span:.1%}, "
                f"below this task set's {self.response.floor:.1%} floor; "
                f"{self.response.examples_needed:,} examples would settle it"
            )
        elif verdict == "size-insensitive":
            problems.append(
                "quality does not rise with model size on this task; "
                "escalating buys nothing"
            )
        elif verdict.startswith("insufficient"):
            problems.append(verdict)
        return problems

    @property
    def escalation_is_justified(self) -> bool:
        return not self.blockers()
