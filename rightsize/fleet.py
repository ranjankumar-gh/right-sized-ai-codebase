from dataclasses import dataclass


@dataclass(frozen=True)
class FleetEconomics:
    """Cost per request on hardware you own, against hardware you rent.

    The asymmetry is the whole point: the owned side has a fixed
    numerator and a growing denominator, so its cost per request falls
    with use. The rented side is a rate and does not move.
    """

    device_count: int
    device_capex: float
    amortization_months: int
    requests_per_device_per_month: int
    rented_cost_per_request: float

    @property
    def monthly_capex(self) -> float:
        if self.amortization_months <= 0:
            return 0.0
        return self.device_count * self.device_capex / self.amortization_months

    @property
    def monthly_requests(self) -> int:
        return self.device_count * self.requests_per_device_per_month

    @property
    def owned_cost_per_request(self) -> float:
        if self.monthly_requests <= 0:
            return float("inf")
        return self.monthly_capex / self.monthly_requests

    @property
    def break_even_requests_per_device(self) -> float:
        """Utilization at which owning becomes cheaper than renting."""
        if self.rented_cost_per_request <= 0 or self.device_count <= 0:
            return float("inf")
        return self.monthly_capex / self.rented_cost_per_request / self.device_count

    @property
    def advantage(self) -> float:
        """How many times cheaper owning is. Below 1.0, renting wins."""
        owned = self.owned_cost_per_request
        return self.rented_cost_per_request / owned if owned else 0.0


SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class UpdatePlan:
    """What it costs, in time, to move an artifact across a fleet."""

    artifact_mb: float
    device_count: int
    delta_fraction: float = 1.0
    per_device_mbps: float = 1.0
    shared_uplink_mbps: float | None = None

    @property
    def transferred_mb_per_device(self) -> float:
        return self.artifact_mb * self.delta_fraction

    @property
    def total_transferred_gb(self) -> float:
        return self.transferred_mb_per_device * self.device_count / 1024.0

    @property
    def hours_per_device(self) -> float:
        if self.per_device_mbps <= 0:
            return float("inf")
        return self.transferred_mb_per_device * 8 / self.per_device_mbps / 3600.0

    @property
    def days_through_shared_uplink(self) -> float:
        """The number that actually bounds a rollout.

        Per-device time is what one site experiences; this is what the
        fleet experiences, and it is the one that decides how fast a bad
        release can be undone.
        """
        if not self.shared_uplink_mbps:
            return 0.0
        megabits = self.total_transferred_gb * 1024 * 8
        return megabits / self.shared_uplink_mbps / SECONDS_PER_DAY


@dataclass(frozen=True)
class RolloutProgress:
    """Devices still on the old version, by day.

    reach_per_attempt is the share of live devices that successfully
    take the update on any given attempt. permanently_unreachable is
    the share that never will: decommissioned, moved, broken radio,
    or sitting in a building nobody has access to.
    """

    reach_per_attempt: float
    permanently_unreachable: float = 0.0

    def stale_fraction_after(self, attempts: int) -> float:
        reachable = 1.0 - self.permanently_unreachable
        missed = (1.0 - self.reach_per_attempt) ** attempts
        return self.permanently_unreachable + reachable * missed

    def attempts_to_reach(self, target_stale: float) -> int | None:
        """Attempts needed to get the stale fraction below a target.

        Returns None when the target is below the permanent floor,
        which is the answer worth having: no retry schedule reaches it,
        and the fix is a truck rather than a timer.
        """
        if target_stale <= self.permanently_unreachable:
            return None
        for attempt in range(1, 366):
            if self.stale_fraction_after(attempt) < target_stale:
                return attempt
        return None


ALLOWED_FIELDS = frozenset({
    "artifact_version", "model_sha", "runtime_version",
    "tokens_per_second", "energy_per_token_mj", "device_temperature_c",
    "throttled", "memory_peak_gib", "oom_kills", "crash_count",
    "requests_served", "escalation_rate", "schema_valid_rate",
    "uncertain_rate", "latency_p50_ms", "latency_p99_ms",
})

FORBIDDEN_SUBSTRINGS = ("prompt", "input", "output", "response", "text",
                        "content", "user", "document", "payload")


def telemetry_violations(fields: dict) -> list[str]:
    """Fields that must not leave a device you do not own.

    Chapter 12's devices sit in customers' buildings, so the inputs
    that caused a failure are frequently the one thing you may not
    collect. Everything in ALLOWED_FIELDS is a property of the device
    or an aggregate over requests, never a request.
    """
    problems = []
    for name in fields:
        lowered = name.lower()
        if any(bad in lowered for bad in FORBIDDEN_SUBSTRINGS):
            problems.append(
                f"{name!r} looks like request content; aggregate it on the "
                f"device and send the statistic instead"
            )
        elif name not in ALLOWED_FIELDS:
            problems.append(
                f"{name!r} is not on the allow-list; add it deliberately "
                f"or drop it"
            )
    return problems
