from dataclasses import dataclass

GIB = 1024**3

LORA_TARGETS_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_TARGETS_MLP = ("gate_proj", "up_proj", "down_proj")
LORA_TARGETS_ALL = LORA_TARGETS_ATTENTION + LORA_TARGETS_MLP


def linear_shapes(config: dict) -> dict[str, tuple[int, int]]:
    """Input and output widths of every adaptable linear layer.

    Names follow the Llama-style convention the current families share.
    A model that names its projections differently needs its own map,
    which is the usual reason a LoRA config silently adapts nothing.
    """
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim") or hidden // heads
    return {
        "q_proj": (hidden, heads * head_dim),
        "k_proj": (hidden, kv_heads * head_dim),
        "v_proj": (hidden, kv_heads * head_dim),
        "o_proj": (heads * head_dim, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }


def adapter_parameter_count(
    config: dict,
    rank: int = 16,
    targets: tuple[str, ...] = LORA_TARGETS_ALL,
) -> int:
    """Trainable parameters a LoRA adapter adds at this rank.

    Linear in rank and linear in layer count, which is why a rank that
    looks cheap on a 4-layer prototype is not cheap on a 36-layer model.
    """
    shapes = linear_shapes(config)
    per_layer = sum(rank * (shapes[t][0] + shapes[t][1]) for t in targets)
    return per_layer * config["num_hidden_layers"]


# A trainable parameter under Adam at mixed precision costs: 2 bytes of
# bf16 weight, 2 of bf16 gradient, 4 of fp32 master weight, and 4 each
# for Adam's first and second moments. Frozen parameters carry only
# their stored weight.
BYTES_PER_TRAINABLE_PARAM = 16.0
BYTES_PER_FROZEN_PARAM_BF16 = 2.0
BYTES_PER_FROZEN_PARAM_NF4 = 0.5


@dataclass(frozen=True)
class TrainingMemory:
    method: str
    frozen_gib: float
    trainable_gib: float
    activation_gib: float

    @property
    def total_gib(self) -> float:
        return self.frozen_gib + self.trainable_gib + self.activation_gib

    def fits_in(self, device_gib: float) -> bool:
        """Conservative, in the same direction as Chapter 1's estimator."""
        return self.total_gib <= device_gib


def estimate_training_memory(
    total_params: int,
    method: str,
    trainable_params: int = 0,
    activation_gib: float = 4.0,
) -> TrainingMemory:
    """Steady-state training memory for one of three methods.

    activation_gib is a placeholder for a term that depends on batch
    size, sequence length, and whether gradient checkpointing is on. It
    is the one term here you should replace with a measurement.
    """
    if method == "full":
        frozen, trainable = 0.0, total_params * BYTES_PER_TRAINABLE_PARAM
    elif method == "lora":
        frozen = total_params * BYTES_PER_FROZEN_PARAM_BF16
        trainable = trainable_params * BYTES_PER_TRAINABLE_PARAM
    elif method == "qlora":
        frozen = total_params * BYTES_PER_FROZEN_PARAM_NF4
        trainable = trainable_params * BYTES_PER_TRAINABLE_PARAM
    else:
        raise ValueError(f"unknown method: {method!r}")
    return TrainingMemory(
        method=method,
        frozen_gib=frozen / GIB,
        trainable_gib=trainable / GIB,
        activation_gib=activation_gib,
    )


IGNORE_INDEX = -100


class LossMaskNotApplied(Exception):
    """Raised when a batch would train on its own prompts."""


def supervised_fraction(labels: list[int]) -> float:
    """Share of positions the loss is actually computed on."""
    if not labels:
        return 0.0
    return sum(1 for t in labels if t != IGNORE_INDEX) / len(labels)


def check_loss_mask(labels: list[int], max_supervised: float = 0.95) -> float:
    """Fail a run whose labels are unmasked, rather than discovering it later.

    A fraction at or near 1.0 means no mask was applied. The threshold is
    below 1.0 rather than at it because a short prompt with a long
    completion legitimately approaches it, and a hard equality check
    would pass exactly the case it is meant to catch.
    """
    fraction = supervised_fraction(labels)
    if fraction > max_supervised:
        raise LossMaskNotApplied(
            f"{fraction:.1%} of positions are supervised; the prompt is "
            f"almost certainly unmasked. Expected roughly the completion's "
            f"share of the sequence."
        )
    return fraction


class TemplateDrift(Exception):
    """Raised when training and serving disagree on the rendered prompt."""


def check_template_match(training_render: str, serving_render: str) -> None:
    """Compare the two renders of the same conversation, exactly.

    Deliberately an exact comparison. Every difference here is a
    difference the model was trained on, including whitespace, and
    normalizing before comparing would hide precisely the drift that
    causes the damage.
    """
    if training_render == serving_render:
        return
    for i, (a, b) in enumerate(zip(training_render, serving_render)):
        if a != b:
            raise TemplateDrift(
                f"renders diverge at character {i}: "
                f"training {training_render[i:i + 40]!r} vs "
                f"serving {serving_render[i:i + 40]!r}"
            )
    raise TemplateDrift(
        f"one render is a prefix of the other: training "
        f"{len(training_render)} chars, serving {len(serving_render)}"
    )


@dataclass(frozen=True)
class DeploymentGate:
    """Bounds an adapted model must clear before it may replace the base.

    Quality alone is not a gate. Chapter 1's envelope has four terms and
    an adaptation can move all of them, so all of them are checked.
    """

    min_quality: float
    max_p95_latency_s: float
    max_memory_gib: float
    max_artifact_gib: float

    def failures(self, quality: float, p95_s: float,
                 memory_gib: float, artifact_gib: float) -> list[str]:
        """Every bound the candidate missed, not just the first."""
        checks = [
            (quality >= self.min_quality,
             f"quality {quality:.3f} below {self.min_quality:.3f}"),
            (p95_s <= self.max_p95_latency_s,
             f"p95 {p95_s:.2f}s above {self.max_p95_latency_s:.2f}s"),
            (memory_gib <= self.max_memory_gib,
             f"memory {memory_gib:.1f} GiB above {self.max_memory_gib:.1f}"),
            (artifact_gib <= self.max_artifact_gib,
             f"artifact {artifact_gib:.1f} GiB above {self.max_artifact_gib:.1f}"),
        ]
        return [message for passed, message in checks if not passed]
