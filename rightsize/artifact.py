import re
from dataclasses import dataclass, field
from datetime import date

GGUF_TYPE = re.compile(
    r"^(IQ|Q|BF|F)(\d+)(?:_(K))?(?:_(XXS|XS|NL|S|M|L|0|1))?$"
)


@dataclass(frozen=True)
class GgufType:
    """What a GGUF type name specifies, beyond its bit width."""

    name: str
    bits: int
    family: str  # "K", "IQ", "legacy", or "float"
    variant: str | None
    mixed_precision: bool
    needs_imatrix: bool

    @property
    def is_uniform(self) -> bool:
        """True when every tensor gets the same treatment.

        Only uniform formats behave the way Chapter 6's arithmetic
        predicts, because that arithmetic assumed one bit width for
        everything.
        """
        return not self.mixed_precision


def parse_gguf_type(name: str) -> GgufType | None:
    """Parse a llama.cpp quantization type name, or None if unrecognized.

    Returning None rather than raising is deliberate: new types appear
    with llama.cpp releases, and a tool that crashes on an unfamiliar
    name is worse than one that reports it cannot classify it.
    """
    canonical = name.strip().upper()
    matched = GGUF_TYPE.match(canonical)
    if not matched:
        return None
    prefix, bits, kind, variant = matched.groups()
    # Store the canonical form rather than whatever was passed in, so a
    # parsed type compares equal however it was written and can be
    # written straight into a lineage record.
    if prefix in ("F", "BF"):
        return GgufType(canonical, int(bits), "float", None, False, False)
    if prefix == "IQ":
        return GgufType(canonical, int(bits), "IQ", variant, True, True)
    if kind == "K":
        return GgufType(canonical, int(bits), "K", variant, True, False)
    return GgufType(canonical, int(bits), "legacy", variant, False, False)


@dataclass(frozen=True)
class RuntimeSupport:
    """What one runtime loads, and when that was last confirmed.

    verified_on is the most important field here. Support changes every
    release, so a matrix without a date is a claim with no expiry and
    will be trusted long after it stopped being true.
    """

    runtime: str
    verified_on: date
    formats: frozenset[str]
    caveats: dict[str, str] = field(default_factory=dict)

    def loads(self, format_name: str) -> bool:
        return format_name.lower() in self.formats

    def is_stale(self, today: date, max_age_days: int = 180) -> bool:
        return (today - self.verified_on).days > max_age_days


# Checked 2026-10-13 against https://docs.vllm.ai/en/latest/features/quantization/
# and https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/.
# Re-verify before trusting it; this is a snapshot, not a standard.
VLLM = RuntimeSupport(
    runtime="vLLM",
    verified_on=date(2026, 10, 13),
    formats=frozenset({"gptq", "awq", "fp8", "gguf", "bitsandbytes"}),
    # Only entries that QUALIFY the support belong here. Neutral facts
    # (which kernel is the default for GPTQ or AWQ, say) are useful and
    # are not caveats, and mixing them in would train you to skim the
    # ones that actually invalidate a plan. A caveat says what the source
    # does NOT establish; it is not a claim that the source rules it out.
    caveats={
        "gguf": "the support matrix lists GGUF on NVIDIA Volta through Hopper "
                "and AMD GPUs; the page does not say which GGML types have "
                "optimized kernels, so measure throughput rather than assuming it",
        "fp8": "FP8 W8A8 (llm-compressor) is listed for Ada, Hopper and AMD "
               "GPUs only, so it does not establish support on Volta, Turing "
               "or Ampere; E4M3 and E5M2 are listed for the KV cache, and the "
               "KV-cache page does not mention INT8, so INT8 KV cache support "
               "is unconfirmed",
        "bitsandbytes": "in-flight quantization; the support matrix lists "
                        "bitsandbytes on NVIDIA Volta through Hopper with "
                        "no bit-width split, so it does not establish 8-bit support",
    },
)


def deployment_blockers(
    format_name: str, support: RuntimeSupport, today: date
) -> list[str]:
    """Why this format cannot be deployed on this runtime, if it cannot.

    Runs before the accuracy sweep, not after. A configuration the
    runtime will not execute is not a candidate and should never have
    entered the comparison.
    """
    problems = []
    if support.is_stale(today):
        age = (today - support.verified_on).days
        problems.append(
            f"{support.runtime} support last verified {age} days ago; "
            f"re-check before trusting this result"
        )
    if not support.loads(format_name):
        problems.append(
            f"{support.runtime} does not load {format_name}; supported: "
            f"{', '.join(sorted(support.formats))}"
        )
    elif format_name.lower() in support.caveats:
        problems.append(
            f"{format_name} loads with a caveat: "
            f"{support.caveats[format_name.lower()]}"
        )
    return problems


@dataclass(frozen=True)
class ArtifactLineage:
    """How one quantized file was produced.

    source_precision_bits is the field that earns this record's place:
    16 means the artifact was made from full-precision weights, and
    anything lower means it was made from something already quantized.
    """

    artifact: str
    source_model: str
    source_precision_bits: int
    format_name: str
    method: str
    tool: str
    tool_version: str
    calibration_set: str | None
    produced_on: date

    @property
    def was_requantized(self) -> bool:
        return self.source_precision_bits < 16

    def warnings(self) -> list[str]:
        """Lineage problems, worst first."""
        problems = []
        if self.was_requantized:
            problems.append(
                f"requantized: produced from {self.source_precision_bits}-bit "
                f"weights rather than 16-bit. llama.cpp's own documentation "
                f"warns this 'can severely reduce quality compared to "
                f"quantizing from 16bit or 32bit'"
            )
        parsed = parse_gguf_type(self.format_name)
        if parsed and parsed.needs_imatrix and not self.calibration_set:
            problems.append(
                f"{self.format_name} is an importance-matrix format but no "
                f"calibration set is recorded; the artifact cannot be "
                f"reproduced and its tail coverage is unknown"
            )
        if not self.tool_version:
            problems.append(
                "no tool version recorded; quantization output is not "
                "stable across releases, so this cannot be reproduced"
            )
        return problems
