from dataclasses import dataclass


@dataclass(frozen=True)
class CascadeEconomics:
    """Cost of a small-first cascade against always using the large model.

    Costs are per request in whatever unit you like, as long as both use
    the same one. The ratio is what matters.
    """

    small_cost: float
    large_cost: float
    escalation_rate: float

    @property
    def cost_per_request(self) -> float:
        """The small model is paid every time. That is the whole story."""
        return self.small_cost + self.escalation_rate * self.large_cost

    @property
    def break_even_rate(self) -> float:
        """Escalation rate at which the cascade stops saving anything."""
        if self.large_cost <= 0:
            return 0.0
        return max(0.0, 1.0 - self.small_cost / self.large_cost)

    @property
    def saving(self) -> float:
        """Fraction saved against always-large. Negative means it costs more."""
        if self.large_cost <= 0:
            return 0.0
        return 1.0 - self.cost_per_request / self.large_cost

    @property
    def headroom(self) -> float:
        """How far the escalation rate can drift before this stops paying.

        Monitor this rather than the escalation rate on its own. A rate
        of 40% is comfortable against a 90% break-even and alarming
        against a 50% one.
        """
        return self.break_even_rate - self.escalation_rate


@dataclass(frozen=True)
class CascadeLatency:
    """Where a cascade's latency actually lands.

    Deliberately reports the escalated path separately from the mean,
    because reporting only the mean is how an SLA written on p99 gets
    violated by an architecture that improved every number anyone looked
    at.
    """

    small_ms: float
    large_ms: float
    escalation_rate: float

    @property
    def fast_path_ms(self) -> float:
        return self.small_ms

    @property
    def escalated_ms(self) -> float:
        """Both models, in series. Independent of the escalation rate."""
        return self.small_ms + self.large_ms

    @property
    def mean_ms(self) -> float:
        return self.small_ms + self.escalation_rate * self.large_ms

    @property
    def tail_penalty(self) -> float:
        """Escalated latency against always-large. Above 1.0 means the
        tail is worse than not having built the cascade."""
        return self.escalated_ms / self.large_ms if self.large_ms else 0.0

    def quantile_ms(self, quantile: float) -> float:
        """Latency at a quantile, from the two-point distribution.

        A cascade's latency is bimodal rather than continuous, so any
        quantile above (1 - escalation_rate) sits on the slow mode and
        every quantile below it sits on the fast one. There is nothing
        in between, which is why a cascade's p50 and p99 can move in
        opposite directions.
        """
        return (
            self.escalated_ms
            if quantile > 1.0 - self.escalation_rate
            else self.fast_path_ms
        )


def top_two_margin(distribution: dict[str, float]) -> float:
    """Gap between the best and second-best candidate.

    The cheapest usable confidence signal: no second model, no extra
    forward pass, just the distribution the model already produced. A
    wide margin means the model was not torn; a narrow one means the
    next token was close to a coin flip.
    """
    if len(distribution) < 2:
        return 1.0 if distribution else 0.0
    ranked = sorted(distribution.values(), reverse=True)
    return ranked[0] - ranked[1]


def should_escalate(
    distribution: dict[str, float],
    threshold: float,
    schema_forced_choice: bool = False,
) -> bool:
    """Whether this request should go to the larger model.

    schema_forced_choice is the Chapter 8 interlock. When a grammar has
    masked away what the model actually preferred, the remaining
    distribution describes a forced choice, and its margin is not a
    confidence signal. Escalating is the safe reading: the case the
    schema could not express is exactly the case worth a second look.
    """
    if schema_forced_choice:
        return True
    return top_two_margin(distribution) < threshold


@dataclass(frozen=True)
class SpeculativeDecoding:
    """Expected speedup from drafting `draft_length` tokens at a time.

    acceptance_rate is the fraction of drafted tokens the target model
    accepts, and it is the variable everything turns on. draft_cost is
    the draft model's forward-pass cost as a fraction of the target's.
    """

    acceptance_rate: float
    draft_length: int
    draft_cost: float = 0.10

    @property
    def expected_tokens(self) -> float:
        """Accepted tokens per verification step.

        A run of accepted tokens ends at the first rejection, so the
        count is geometric and capped by the draft length. The trailing
        +1 is the token the target model produces itself on rejection,
        which is why this can exceed the draft length.
        """
        a, g = self.acceptance_rate, self.draft_length
        if a >= 1.0:
            return g + 1.0
        return (1.0 - a ** (g + 1)) / (1.0 - a)

    @property
    def speedup(self) -> float:
        """Against plain autoregressive decoding.

        The denominator is the cost of one verification plus the drafts
        that fed it. Drafting is paid for every proposed token, including
        the rejected ones, which is what makes a long draft with a poor
        acceptance rate actively harmful.
        """
        return self.expected_tokens / (1.0 + self.draft_cost * self.draft_length)

    @property
    def is_slower(self) -> bool:
        return self.speedup < 1.0


def best_draft_length(
    acceptance_rate: float, draft_cost: float = 0.10, limit: int = 32
) -> int:
    """Draft length maximising speedup at this acceptance rate.

    There is an optimum rather than a monotone improvement: each extra
    drafted token costs draft_cost whether accepted or not, and the
    chance it survives falls geometrically.
    """
    return max(
        range(1, limit + 1),
        key=lambda g: SpeculativeDecoding(acceptance_rate, g, draft_cost).speedup,
    )


@dataclass(frozen=True)
class CompositionGate:
    """Whether a composed system should be built, or kept.

    Runs before the build and again on live numbers, because every
    input here drifts. The escalation rate in particular is a property
    of current traffic rather than of the design.
    """

    min_headroom: float = 0.20
    max_tail_penalty: float = 1.0
    min_speculative_speedup: float = 1.10

    def blockers(
        self,
        economics: CascadeEconomics,
        latency: CascadeLatency,
        tail_bound: bool,
        speculative: SpeculativeDecoding | None = None,
    ) -> list[str]:
        problems = []
        if economics.saving <= 0:
            problems.append(
                f"cascade costs {-economics.saving:.0%} more than always-large "
                f"at a {economics.escalation_rate:.0%} escalation rate; "
                f"break-even is {economics.break_even_rate:.0%}"
            )
        elif economics.headroom < self.min_headroom:
            problems.append(
                f"only {economics.headroom:.0%} of headroom before the "
                f"cascade stops paying; escalation drifts upward"
            )
        if tail_bound and latency.tail_penalty > self.max_tail_penalty:
            problems.append(
                f"escalated requests take {latency.escalated_ms:.0f} ms, "
                f"{latency.tail_penalty:.2f}x always-large, and this service "
                f"is tail-bound; consider running both in parallel"
            )
        if speculative and speculative.speedup < self.min_speculative_speedup:
            better = best_draft_length(
                speculative.acceptance_rate, speculative.draft_cost
            )
            problems.append(
                f"speculative decoding at {speculative.acceptance_rate:.0%} "
                f"acceptance and draft length {speculative.draft_length} gives "
                f"{speculative.speedup:.2f}x; best length here is {better}"
            )
        return problems
