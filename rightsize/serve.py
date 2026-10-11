import math
from dataclasses import dataclass

from rightsize.sizing import GIB, ModelShape, Workload, bytes_per_weight, kv_cache_bytes


def kv_bytes_per_token(shape: ModelShape, kv_bits: float = 16) -> float:
    """KV cache one token claims, across all layers.

    Chapter 1 computed this for a whole workload. Serving needs it per
    token, because the question here is how many tokens fit rather than
    whether one workload does. Exact only when every layer is full
    attention; a hybrid's cache is not proportional to its length.
    """
    return kv_cache_bytes(shape, Workload(context_tokens=1, batch_size=1), kv_bits)


@dataclass(frozen=True)
class ServingCapacity:
    """How many requests a device holds at once.

    The weights are a fixed cost and the cache is a per-request one, so
    capacity is set by the second term and quoted capacity figures that
    omit the context length are meaningless.
    """

    shape: ModelShape
    device_gib: float
    context_tokens: int
    weight_bits: float = 4
    kv_bits: float = 16
    runtime_overhead_gib: float = 2.0

    @property
    def weights_gib(self) -> float:
        params = self.shape.total_params_b * 1e9
        return params * bytes_per_weight(self.weight_bits) / GIB

    @property
    def kv_budget_gib(self) -> float:
        """What is left for the cache. Negative means the model alone
        does not fit, which Chapter 1's estimator should have caught."""
        return self.device_gib - self.weights_gib - self.runtime_overhead_gib

    @property
    def gib_per_request(self) -> float:
        """Asked of Chapter 1 for one whole request rather than scaled up
        from kv_bytes_per_token, because on a hybrid model part of the
        cache stops growing with context."""
        request = Workload(context_tokens=self.context_tokens, batch_size=1)
        return kv_cache_bytes(self.shape, request, self.kv_bits) / GIB

    @property
    def concurrency_ceiling(self) -> int:
        """Requests that fit at full context. Floored, not rounded:
        a request that half fits does not run."""
        if self.kv_budget_gib <= 0 or self.gib_per_request <= 0:
            return 0
        return int(self.kv_budget_gib // self.gib_per_request)


def blocks_for(length_tokens: int, block_size: int = 16) -> int:
    """Blocks a sequence of this length occupies."""
    if length_tokens <= 0:
        return 0
    return math.ceil(length_tokens / block_size)


def block_waste_tokens(length_tokens: int, block_size: int = 16) -> int:
    """Tokens allocated and unused in the sequence's final block.

    Bounded by block_size - 1 regardless of sequence length, which is
    why this matters enormously for short sequences and not at all for
    long ones. The absolute waste is the same; the fraction is not.
    """
    return blocks_for(length_tokens, block_size) * block_size - length_tokens


def block_waste_fraction(length_tokens: int, block_size: int = 16) -> float:
    if length_tokens <= 0:
        return 0.0
    return block_waste_tokens(length_tokens, block_size) / length_tokens


@dataclass(frozen=True)
class PrefixCacheSaving:
    """What a shared prompt prefix saves across concurrent requests.

    The shared portion is stored once and referenced by every request
    that starts with it, so the saving grows with concurrency rather
    than with prompt length alone.
    """

    shared_prefix_tokens: int
    unique_tokens: int
    concurrent_requests: int
    block_size: int = 16

    @property
    def shared_blocks(self) -> int:
        """Only whole blocks can be shared: a prefix is reused up to the
        last block boundary it completely fills, and the partial block
        after it belongs to the request."""
        return self.shared_prefix_tokens // self.block_size

    @property
    def blocks_without_sharing(self) -> int:
        per_request = blocks_for(self.shared_prefix_tokens + self.unique_tokens,
                                 self.block_size)
        return per_request * self.concurrent_requests

    @property
    def blocks_with_sharing(self) -> int:
        remainder = (
            self.shared_prefix_tokens - self.shared_blocks * self.block_size
        )
        per_request = blocks_for(remainder + self.unique_tokens, self.block_size)
        return self.shared_blocks + per_request * self.concurrent_requests

    @property
    def saving(self) -> float:
        if not self.blocks_without_sharing:
            return 0.0
        return 1.0 - self.blocks_with_sharing / self.blocks_without_sharing


@dataclass(frozen=True)
class RequestOutcome:
    """One served request, as the client experienced it."""

    ttft_s: float
    inter_token_latency_s: float
    completed: bool = True
    preempted: bool = False


@dataclass(frozen=True)
class ServiceLevel:
    """The bounds a request must meet to have been served acceptably."""

    max_ttft_s: float
    max_inter_token_latency_s: float

    def met_by(self, outcome: RequestOutcome) -> bool:
        return (
            outcome.completed
            and outcome.ttft_s <= self.max_ttft_s
            and outcome.inter_token_latency_s <= self.max_inter_token_latency_s
        )


def goodput(outcomes: list[RequestOutcome], level: ServiceLevel) -> float:
    """Share of requests that met every bound.

    Throughput counts requests finished and says nothing about whether
    anyone waited too long. Mean latency says nothing about how many
    finished. Goodput is the one number that cannot be improved by
    making the other worse, which is why it is the one to alert on.
    """
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if level.met_by(o)) / len(outcomes)


def preemption_rate(outcomes: list[RequestOutcome]) -> float:
    """Share of requests evicted and rescheduled.

    A non-zero rate means the server admitted work it could not hold
    and is recomputing tokens it already produced. It is the earliest
    signal of the failure that opened this chapter, and it appears
    before the latency graph moves.
    """
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.preempted) / len(outcomes)


def can_admit(
    capacity: ServingCapacity, resident_requests: int, safety_margin: float = 0.9
) -> bool:
    """Whether another request should be accepted.

    The margin exists because the ceiling assumes every resident request
    reaches full context, and admitting exactly to the ceiling leaves
    nothing for the sequences that grow. Rejecting at the door is
    cheaper than evicting later: a rejected request costs nothing, and
    a preempted one costs everything already computed for it.
    """
    return resident_requests < int(capacity.concurrency_ceiling * safety_margin)


def serving_concerns(
    capacity: ServingCapacity,
    measured_context_tokens: int,
    observed_preemption_rate: float = 0.0,
) -> list[str]:
    """Configuration problems, before they become incidents."""
    problems = []
    if capacity.kv_budget_gib <= 0:
        problems.append(
            f"weights plus overhead exceed the {capacity.device_gib:.0f} GiB "
            f"device; there is no cache budget at all"
        )
    if capacity.concurrency_ceiling < 2:
        problems.append(
            f"ceiling is {capacity.concurrency_ceiling} request(s) at "
            f"{capacity.context_tokens} tokens; batching cannot help here"
        )
    if measured_context_tokens > capacity.context_tokens:
        planned = capacity.concurrency_ceiling
        actual = ServingCapacity(
            capacity.shape, capacity.device_gib, measured_context_tokens,
            capacity.weight_bits, capacity.kv_bits, capacity.runtime_overhead_gib,
        ).concurrency_ceiling
        problems.append(
            f"planned for {capacity.context_tokens} tokens but traffic "
            f"averages {measured_context_tokens}; the real ceiling is "
            f"{actual}, not {planned}"
        )
    if observed_preemption_rate > 0.0:
        problems.append(
            f"{observed_preemption_rate:.1%} of requests are being preempted "
            f"and recomputed; admit fewer rather than tuning anything else"
        )
    return problems
