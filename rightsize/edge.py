from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalProfile:
    """How a device's throughput falls once its thermal budget runs out.

    A deliberately crude two-rate model: full speed until the budget is
    spent, then a reduced steady rate. Real devices ramp rather than
    step, and the two-rate version is enough to show that a benchmark
    shorter than seconds_to_throttle measures only the first rate.
    """

    peak_tokens_per_sec: float
    sustained_fraction: float
    seconds_to_throttle: float

    @property
    def sustained_tokens_per_sec(self) -> float:
        return self.peak_tokens_per_sec * self.sustained_fraction

    def average_over(self, window_seconds: float) -> float:
        """Throughput a benchmark of this length would report."""
        if window_seconds <= self.seconds_to_throttle:
            return self.peak_tokens_per_sec
        hot = self.seconds_to_throttle * self.peak_tokens_per_sec
        throttled = (
            (window_seconds - self.seconds_to_throttle)
            * self.sustained_tokens_per_sec
        )
        return (hot + throttled) / window_seconds

    def benchmark_overstates_by(self, window_seconds: float) -> float:
        """How much a benchmark of this length flatters the device."""
        if self.sustained_tokens_per_sec <= 0:
            return 0.0
        return self.average_over(window_seconds) / self.sustained_tokens_per_sec


JOULES_PER_WATT_HOUR = 3600.0


@dataclass(frozen=True)
class EnergyProfile:
    """What a device spends per token, and what that buys from a battery."""

    millijoules_per_token: float
    idle_watts: float = 0.0

    @property
    def tokens_per_joule(self) -> float:
        if self.millijoules_per_token <= 0:
            return 0.0
        return 1000.0 / self.millijoules_per_token

    def tokens_from_battery(self, watt_hours: float) -> float:
        """Total tokens before flat, ignoring idle draw.

        Optimistic by construction: Pass 3 adds idle, which on a
        low-duty-cycle device is usually the larger term.
        """
        return watt_hours * JOULES_PER_WATT_HOUR * self.tokens_per_joule

    def responses_from_battery(
        self, watt_hours: float, tokens_per_response: int = 200
    ) -> float:
        return self.tokens_from_battery(watt_hours) / tokens_per_response


def energy_per_token_mj(watts: float, tokens_per_sec: float) -> float:
    """Convert a power reading and a rate into the number that matters.

    Power alone ranks devices wrongly: a backend at twice the watts and
    three times the speed is the more efficient one, and a wattage
    comparison says the opposite.
    """
    if tokens_per_sec <= 0:
        return float("inf")
    return watts * 1000.0 / tokens_per_sec


@dataclass(frozen=True)
class DutyCycle:
    """Energy split between working and waiting, over an hour.

    Most embedded deployments are not under sustained load. They wake,
    answer, and sleep, and the sleeping is where the hours go.
    """

    profile: EnergyProfile
    responses_per_hour: int
    tokens_per_response: int = 200

    @property
    def inference_joules_per_hour(self) -> float:
        tokens = self.responses_per_hour * self.tokens_per_response
        return tokens * self.profile.millijoules_per_token / 1000.0

    @property
    def idle_joules_per_hour(self) -> float:
        return self.profile.idle_watts * 3600.0

    @property
    def total_joules_per_hour(self) -> float:
        return self.inference_joules_per_hour + self.idle_joules_per_hour

    @property
    def inference_share(self) -> float:
        """Fraction of the energy budget inference is responsible for.

        This is the number that decides whether the rest of this book
        is relevant to your device. Below roughly a fifth, model work
        is optimising a rounding error.
        """
        total = self.total_joules_per_hour
        return self.inference_joules_per_hour / total if total else 0.0

    def hours_from_battery(self, watt_hours: float) -> float:
        rate = self.total_joules_per_hour
        return watt_hours * JOULES_PER_WATT_HOUR / rate if rate else float("inf")


def edge_memory_headroom(
    footprint_high_gib: float, device_gib: float, reserved_gib: float = 0.5
) -> float:
    """Memory left after the model, the OS reservation, and nothing else.

    reserved_gib is the operating system, the application, and whatever
    else shares the board. Negative headroom on a device with swap is a
    performance problem; on a device without it, the allocator fails or
    the kernel kills the process.
    """
    return device_gib - footprint_high_gib - reserved_gib


def survives_without_swap(
    footprint_high_gib: float,
    device_gib: float,
    reserved_gib: float = 0.5,
    safety_fraction: float = 0.15,
) -> bool:
    """Whether to deploy this model to a device with no swap.

    Judged against the pessimistic end of Chapter 1's estimate, with a
    further margin, because the failure is not graceful. A server that
    is short of memory gets slow. A board without swap gets an
    out-of-memory kill, and Pass 5 is about what happens next.
    """
    headroom = edge_memory_headroom(footprint_high_gib, device_gib, reserved_gib)
    return headroom >= device_gib * safety_fraction


@dataclass(frozen=True)
class UnattendedReadiness:
    """Whether a device can survive its own failures without a visit.

    Every field is a question with a yes or no answer that somebody
    already knows. The value of the checklist is that 'no' is cheap
    here and expensive in a field enclosure.
    """

    cold_boot_seconds: float
    max_acceptable_boot_seconds: float
    has_local_fallback: bool
    verifies_artifact_before_load: bool
    keeps_previous_artifact: bool
    restarts_on_crash: bool

    def gaps(self) -> list[str]:
        checks = [
            (self.cold_boot_seconds <= self.max_acceptable_boot_seconds,
             f"cold boot takes {self.cold_boot_seconds:.0f}s against a "
             f"{self.max_acceptable_boot_seconds:.0f}s budget; after a power "
             f"cut the device is unavailable for that long"),
            (self.has_local_fallback,
             "no local fallback; when the model cannot answer, the device "
             "has nothing to do and no one to ask"),
            (self.verifies_artifact_before_load,
             "artifact is not verified before loading; a truncated download "
             "becomes a boot loop rather than a rejected update"),
            (self.keeps_previous_artifact,
             "no previous artifact retained; a bad update cannot be rolled "
             "back locally and needs a physical visit"),
            (self.restarts_on_crash,
             "no supervised restart; a single crash leaves the unit off "
             "until someone drives to it"),
        ]
        return [message for passed, message in checks if not passed]
