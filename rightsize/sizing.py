"""Predict a model's memory footprint and decode ceiling before downloading it.

Introduced in Chapter 1. Every figure this module produces is computed from
architecture and workload facts, not measured. The point is to eliminate
candidates cheaply: a two-minute calculation beats a thirty-minute
download-and-try loop, and it generalizes across machines in a way a
measurement never does.
"""

from dataclasses import dataclass

GIB = 1024**3

# Runtime workspace, allocator fragmentation, CUDA graphs, and activation
# buffers. Not derivable from architecture; this is the observed band for
# server runtimes and the reason estimate_footprint returns a range.
OVERHEAD_GIB_LOW = 0.5
OVERHEAD_GIB_HIGH = 2.0


@dataclass(frozen=True)
class ModelShape:
    """Architecture facts, all readable from a model's config.json."""

    name: str
    total_params_b: float
    active_params_b: float  # Equals total_params_b for dense models.
    layers: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Workload:
    """What you actually intend to run, not what the benchmark ran."""

    context_tokens: int
    batch_size: int


@dataclass(frozen=True)
class Footprint:
    """A memory estimate stated as a range, because that is what we know."""

    weights_gib: float
    kv_cache_gib: float
    total_low_gib: float
    total_high_gib: float

    def fits_in(self, device_gib: float) -> bool:
        """Conservative by design: judged against the high end.

        The cost of being wrong is asymmetric. Overestimating wastes some
        memory; underestimating takes down the service.
        """
        return self.total_high_gib <= device_gib


def shape_from_config(
    name: str,
    config: dict,
    total_params_b: float,
    active_params_b: float | None = None,
) -> ModelShape:
    """Build a ModelShape from a Hugging Face config.json.

    Parameter counts are not in config.json, so they stay explicit. Take
    them from the model card or the safetensors index rather than
    inferring them from the name: a "7B" is rarely exactly 7.0e9, and a 3%
    error lands exactly where memory ceilings are decided.
    """
    head_dim = config.get("head_dim")
    if head_dim is None:
        head_dim = config["hidden_size"] // config["num_attention_heads"]
    return ModelShape(
        name=name,
        total_params_b=total_params_b,
        active_params_b=active_params_b or total_params_b,
        layers=config["num_hidden_layers"],
        # Falls back to num_attention_heads for models predating GQA,
        # where every query head has its own KV head. Assuming GQA where
        # there is none underestimates the cache by the head ratio.
        kv_heads=config.get(
            "num_key_value_heads", config["num_attention_heads"]
        ),
        head_dim=head_dim,
    )


def bytes_per_weight(bits: float, group_size: int | None = 128) -> float:
    """Bytes per stored weight, including quantization scale metadata.

    Sub-8-bit formats store a scale and zero-point per group of weights.
    That metadata is small per group and large enough in aggregate to
    decide whether a model clears a memory ceiling: at group size 128 it
    makes a nominally 4-bit model closer to 4.16 bits.
    """
    if bits >= 8 or group_size is None:
        return bits / 8
    metadata_bits_per_weight = (16 + 4) / group_size
    return (bits + metadata_bits_per_weight) / 8


def kv_cache_bytes(
    shape: ModelShape, workload: Workload, kv_bits: float = 16
) -> float:
    """Bytes held by the key/value cache at full context.

    The leading 2 covers keys and values. Note that kv_heads, not the query
    head count, drives this: grouped-query attention cuts this term by the
    ratio between them, which is why nearly every current small model uses
    it.

    Unlike the weights, this term is linear in both context length and
    batch size, which is why it is the term that most often ends a
    deployment.
    """
    per_token_per_layer = 2 * shape.kv_heads * shape.head_dim * (kv_bits / 8)
    return (
        per_token_per_layer
        * shape.layers
        * workload.context_tokens
        * workload.batch_size
    )


def estimate_footprint(
    shape: ModelShape,
    workload: Workload,
    weight_bits: float = 16,
    kv_bits: float = 16,
) -> Footprint:
    """Total memory needed to serve this shape under this workload.

    Uses total_params_b: every expert of a mixture-of-experts model must be
    resident even though only some run per token. Memory is charged on the
    total.
    """
    weights = shape.total_params_b * 1e9 * bytes_per_weight(weight_bits) / GIB
    kv = kv_cache_bytes(shape, workload, kv_bits) / GIB
    return Footprint(
        weights_gib=weights,
        kv_cache_gib=kv,
        total_low_gib=weights + kv + OVERHEAD_GIB_LOW,
        total_high_gib=weights + kv + OVERHEAD_GIB_HIGH,
    )


def decode_ceiling_tokens_per_sec(
    shape: ModelShape,
    bandwidth_gib_per_sec: float,
    weight_bits: float = 16,
) -> float:
    """Upper bound on single-stream decode speed.

    Decoding reads the active weights once per token, so throughput is
    bounded by memory bandwidth divided by that read. Uses active
    parameters: an MoE model touches only its selected experts per token,
    which is exactly why it is cheaper to run than to hold.

    This is a ceiling, not a prediction. Real systems land below it. If the
    ceiling is already below your target, no runtime tuning will save you -
    quantize, pick fewer active parameters, or buy more bandwidth.
    """
    active_gib = (
        shape.active_params_b * 1e9 * bytes_per_weight(weight_bits) / GIB
    )
    return bandwidth_gib_per_sec / active_gib
