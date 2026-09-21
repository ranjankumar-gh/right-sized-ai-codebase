import math
from dataclasses import dataclass

from rightsize.capability import parameter_budget


def prune_width(
    config: dict, mlp_keep: float = 1.0, head_keep: float = 1.0
) -> dict:
    """Narrow each layer, keeping the layer count.

    Query and key/value heads are cut together to preserve the
    grouped-query ratio. Cutting them independently changes the
    ratio, which changes the KV cache Chapter 1 sized, and is a
    separate decision that should be made deliberately.
    """
    pruned = dict(config)
    pruned["intermediate_size"] = int(config["intermediate_size"] * mlp_keep)
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    pruned["num_attention_heads"] = max(1, int(heads * head_keep))
    pruned["num_key_value_heads"] = max(1, int(kv_heads * head_keep))
    return pruned


def prune_depth(config: dict, layers_keep: float) -> dict:
    """Remove whole layers. Latency falls roughly linearly with this."""
    pruned = dict(config)
    pruned["num_hidden_layers"] = max(
        1, int(config["num_hidden_layers"] * layers_keep)
    )
    return pruned


def pruning_floor_params(config: dict) -> int:
    """Smallest parameter count reachable without touching the vocabulary.

    Every structural cut removes non-embedding parameters. None of them
    removes a row of the embedding table, so this is the asymptote.
    """
    return parameter_budget(config).embedding


def deepest_useful_cut(config: dict, target_params: int) -> float | None:
    """Fraction of non-embedding parameters that may remain, or None.

    Returns None when the target is below the floor, which means the
    target is unreachable by pruning alone at this vocabulary size and
    the next move is a tokenizer decision rather than a pruning one.
    """
    budget = parameter_budget(config)
    remaining = target_params - budget.embedding
    if remaining <= 0:
        return None
    return min(1.0, remaining / budget.non_embedding)


def select_survivors(
    importance: dict[int, float], keep_fraction: float
) -> list[int]:
    """Indices to keep, highest importance first, returned in model order.

    Ties are broken by index so that two runs over the same calibration
    data cut the same units. A pruning step that is not deterministic
    cannot be debugged, because the model you are holding is not the
    model the last run produced.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    keep_count = max(1, round(len(importance) * keep_fraction))
    ranked = sorted(importance.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted(index for index, _ in ranked[:keep_count])


def calibration_coverage(
    slice_counts: dict[str, int], traffic_share: dict[str, float]
) -> dict[str, float]:
    """Over- or under-representation of each slice in the calibration set.

    A value below 1.0 means the slice is rarer in calibration than in
    traffic, and importance scoring will under-rate whatever capacity
    serves it. This is the arithmetic behind the failure that opened
    the chapter.
    """
    total = sum(slice_counts.values())
    if total == 0:
        return {name: 0.0 for name in traffic_share}
    return {
        name: (slice_counts.get(name, 0) / total) / share
        for name, share in traffic_share.items()
        if share > 0
    }


def distillation_mode(teacher_config: dict, student_config: dict) -> str:
    """Which distillation the two models can actually support.

    Logit distillation needs position i of the output distribution to
    mean the same token to both models. A shared vocabulary size is a
    necessary condition and not a sufficient one: two tokenizers can
    agree on size and disagree on contents, so confirm the tokenizer
    is the same one before trusting a 'logit' result.
    """
    if teacher_config["vocab_size"] == student_config["vocab_size"]:
        return "logit"
    return "sequence"


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax at a temperature."""
    scaled = [value / temperature for value in logits]
    largest = max(scaled)
    exponentials = [math.exp(value - largest) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def kd_divergence(
    student_logits: list[float],
    teacher_logits: list[float],
    temperature: float = 2.0,
) -> float:
    """Distillation loss at one position: KL from teacher to student.

    The temperature softens both distributions so the teacher's
    low-probability mass carries signal. That mass is the whole point:
    a hard label says 'cat', while a softened teacher says 'cat, and
    this was somewhat lynx and not at all bicycle'.

    The T-squared factor compensates for gradients scaling as 1/T^2,
    so changing the temperature does not silently rescale the
    contribution of this term against any hard-label term beside it.
    """
    teacher = softmax(teacher_logits, temperature)
    student = softmax(student_logits, temperature)
    divergence = sum(
        p * math.log(p / q)
        for p, q in zip(teacher, student)
        if p > 0.0
    )
    return divergence * temperature**2


@dataclass(frozen=True)
class CompressionViability:
    """Preconditions for owning a derived model, checked before starting.

    Every field is a question with an answer the team already has. The
    value of writing them down is that a 'no' here costs a meeting,
    and the same 'no' discovered later costs a quarter.
    """

    quantization_attempted: bool
    constraint_durable_months: int
    teacher_output_licensed: bool
    healing_tokens_available_b: float
    hard_case_slice_exists: bool

    def blockers(self, min_healing_tokens_b: float = 1.0) -> list[str]:
        checks = [
            (self.quantization_attempted,
             "quantization not tried; it is cheaper and reversible (Ch 6)"),
            (self.constraint_durable_months >= 12,
             f"constraint lasts {self.constraint_durable_months} months; "
             f"the derived-model burden outlives it"),
            (self.teacher_output_licensed,
             "teacher outputs not licensed for training a competing model"),
            (self.healing_tokens_available_b >= min_healing_tokens_b,
             f"{self.healing_tokens_available_b}B healing tokens available, "
             f"need at least {min_healing_tokens_b}B"),
            (self.hard_case_slice_exists,
             "no hard-case slice; a pruned model regresses on the tail "
             "and a traffic-weighted eval will not show it"),
        ]
        return [message for passed, message in checks if not passed]
