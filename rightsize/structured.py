from collections.abc import Sequence
from dataclasses import dataclass

NEG_INF = float("-inf")


def allowed_continuations(prefix: str, options: Sequence[str]) -> set[str]:
    """Characters that keep at least one permitted option reachable.

    This is a finite-state machine over a set of literals, which is the
    simplest constraint worth having and the shape an enum field takes.
    A real engine compiles a grammar, but the operation at each step is
    exactly this: given what has been emitted, what may come next?
    """
    return {
        option[len(prefix)]
        for option in options
        if option.startswith(prefix) and len(option) > len(prefix)
    }


def mask_logits(
    logits: dict[str, float], allowed: set[str]
) -> dict[str, float]:
    """Set every illegal token to negative infinity.

    Nothing here consults the model. The mask is computed from the
    grammar and the output so far, which is why constrained decoding
    composes with quantization, adapters, and distillation: it operates
    on the distribution rather than on the weights.
    """
    return {
        token: (score if token in allowed else NEG_INF)
        for token, score in logits.items()
    }


def pick(logits: dict[str, float]) -> str | None:
    """Greedy choice, or None when the mask left nothing legal."""
    live = {t: s for t, s in logits.items() if s != NEG_INF}
    return max(live, key=live.__getitem__) if live else None


@dataclass(frozen=True)
class StructuredOutcome:
    """One generation, judged on both axes independently."""

    schema_valid: bool
    answer_correct: bool
    executed: bool = False  # Tool calls only: did the call actually run?


@dataclass(frozen=True)
class StructuredReport:
    """The four rates, kept apart on purpose.

    Collapsing these into one number is how a system that got better at
    packaging is recorded as having got better at the task.
    """

    outcomes: list[StructuredOutcome]

    @property
    def schema_validity(self) -> float:
        return self._rate(lambda o: o.schema_valid)

    @property
    def answer_accuracy(self) -> float:
        return self._rate(lambda o: o.answer_correct)

    @property
    def executable_accuracy(self) -> float:
        return self._rate(lambda o: o.executed and o.answer_correct)

    @property
    def wrong_but_valid(self) -> float:
        """The number that moves when a grammar is introduced.

        Schema-valid and incorrect: output that parses, passes every
        structural check, and is wrong. These are the ones that reach
        production without touching an error path.
        """
        return self._rate(lambda o: o.schema_valid and not o.answer_correct)

    def _rate(self, predicate) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if predicate(o)) / len(self.outcomes)

    def regression_against(self, baseline: "StructuredReport") -> list[str]:
        """What got worse, stated in the direction that matters.

        Validity rising while accuracy falls is the expected shape of
        the constraint tax, and it is the shape a single aggregate
        metric is guaranteed to miss.
        """
        problems = []
        if self.answer_accuracy < baseline.answer_accuracy:
            problems.append(
                f"answer accuracy fell "
                f"{baseline.answer_accuracy:.1%} -> {self.answer_accuracy:.1%}"
            )
        if self.wrong_but_valid > baseline.wrong_but_valid:
            problems.append(
                f"wrong-but-valid rose "
                f"{baseline.wrong_but_valid:.1%} -> {self.wrong_but_valid:.1%}; "
                f"failures that used to be caught now parse cleanly"
            )
        return problems


@dataclass(frozen=True)
class SchemaComplexity:
    """Structural cost of a schema, from the schema alone."""

    max_depth: int
    field_count: int
    required_count: int
    enum_branches: int

    @property
    def pressure_score(self) -> float:
        """Rough ordering of how much capacity this schema consumes.

        Depth carries the largest weight (3.0 per level, against 0.5 per
        field, required property, and 0.25 per enum branch). Those weights
        are set in this method, not fitted to data. The score ranks two
        schemas against each other; it does not predict accuracy loss.
        """
        return (
            self.max_depth * 3.0
            + self.field_count * 0.5
            + self.required_count * 0.5
            + self.enum_branches * 0.25
        )


def schema_complexity(schema: dict, _depth: int = 1) -> SchemaComplexity:
    """Walk a JSON Schema and count what a small model has to hold."""
    fields = required = enums = 0
    deepest = _depth

    for key, value in (schema.get("properties") or {}).items():
        fields += 1
        if isinstance(value, dict):
            if "enum" in value:
                enums += len(value["enum"])
            nested = value.get("properties") or (
                (value.get("items") or {}).get("properties")
            )
            if nested:
                inner = schema_complexity(
                    value if value.get("properties") else value["items"],
                    _depth + 1,
                )
                deepest = max(deepest, inner.max_depth)
                fields += inner.field_count
                required += inner.required_count
                enums += inner.enum_branches
    required += len(schema.get("required") or [])
    return SchemaComplexity(deepest, fields, required, enums)
