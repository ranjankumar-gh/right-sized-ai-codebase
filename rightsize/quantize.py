import math
from dataclasses import dataclass


def quantize_group(
    values: list[float], bits: int = 4, symmetric: bool = False
) -> tuple[list[int], float, int]:
    """Map one group of values onto a grid of 2**bits codes.

    Returns the codes plus the two numbers needed to read them back.
    Those two are the metadata Chapter 1's bytes_per_weight charges
    for, and they are per group, which is why group size is a memory
    decision as well as an accuracy one.
    """
    highest_code = 2**bits - 1
    if symmetric:
        # Offset binary. The code at zero_point stores the value 0, codes
        # above it store positive values and codes below it store negative
        # ones. A zero point of 0 would clamp every negative value to code 0.
        bound = max(abs(v) for v in values) or 1.0
        scale = bound / (2 ** (bits - 1) - 1)
        zero_point = 2 ** (bits - 1)
    else:
        low, high = min(values), max(values)
        scale = (high - low) / highest_code if high > low else 1.0
        zero_point = round(-low / scale)
    codes = [
        max(0, min(highest_code, round(v / scale) + zero_point))
        for v in values
    ]
    return codes, scale, zero_point


def dequantize_group(
    codes: list[int], scale: float, zero_point: int
) -> list[float]:
    """Read the grid back. This is exactly what a kernel does per tile."""
    return [(code - zero_point) * scale for code in codes]


def quantize_tensor(
    values: list[float],
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> list[float]:
    """Quantize and immediately restore, to measure what was lost.

    A real runtime keeps the codes and dequantizes per tile during the
    matrix multiply. This returns the restored values because the only
    thing we want to look at here is the error.
    """
    restored: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        codes, scale, zero_point = quantize_group(group, bits, symmetric)
        restored.extend(dequantize_group(codes, scale, zero_point))
    return restored


def relative_error(original: list[float], restored: list[float]) -> float:
    """RMS error as a fraction of the signal's own RMS.

    Reported relative rather than absolute so the number means the same
    thing across tensors with different magnitudes. A relative error of
    1.0 means the restored values carry no information: you would do as
    well replacing every one of them with a constant.
    """
    squared_error = sum(
        (a - b) ** 2 for a, b in zip(original, restored)
    ) / len(original)
    signal = sum(a**2 for a in original) / len(original)
    return math.sqrt(squared_error) / math.sqrt(signal) if signal else 0.0


def clipped_range(values: list[float], keep: float = 0.999) -> tuple[float, float]:
    """Range covering the central `keep` fraction, ignoring the tails.

    Trades unbounded error on a few values for bounded error on all the
    rest, which is the right trade whenever the few are outliers and the
    wrong one whenever they are signal. Which it is depends on the
    tensor, so measure rather than assume.
    """
    ordered = sorted(values)
    drop = int(len(ordered) * (1.0 - keep) / 2)
    return ordered[drop], ordered[len(ordered) - 1 - drop]


@dataclass(frozen=True)
class QuantizationPlan:
    """A configuration, stated before it is measured.

    The fields are deliberately the ones a runtime has to support
    rather than the ones a paper reports. Chapter 7 is about what
    happens when a plan is valid here and unsupported there.
    """

    weight_bits: int
    group_size: int
    kv_bits: int
    symmetric: bool = True

    def concerns(self, params_b: float) -> list[str]:
        """Known-risky combinations, stated before the GPU time.

        Not a correctness check. These are the settings that most often
        produce a disappointing measurement, flagged so the measurement
        is expected rather than surprising.
        """
        checks = [
            (self.weight_bits >= 3,
             f"{self.weight_bits}-bit weights need more than round-to-nearest; "
             f"plain rounding loses about half the RMS signal at 2 bits"),
            (not (self.weight_bits <= 4 and params_b < 4.0)
             or self.group_size <= 64,
             f"a {params_b:g}B model at {self.weight_bits}-bit pays the "
             f"small-model penalty; group 64 or smaller is the cheap "
             f"mitigation before anything harder"),
            (self.kv_bits <= 8,
             "the KV cache is unquantized and is usually the larger term; "
             "see Chapter 1's estimator at your real context length"),
            (self.group_size >= 32,
             f"group {self.group_size} stores {2.5 / self.group_size:.3f} "
             f"bytes of scale and zero-point metadata per weight, more than "
             f"group 32 stores ({2.5 / 32:.3f}); check bytes_per_weight"),
        ]
        return [message for passed, message in checks if not passed]
