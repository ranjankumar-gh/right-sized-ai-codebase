"""A runner that replays fixed behavior instead of running a model.

Exists so the test suite and the book's worked examples need no GPU, no
downloaded weights, and no network. Every number it emits is synthetic and
deterministic. It is a teaching and testing instrument, never evidence
about a real model.
"""

from dataclasses import dataclass, field

from rightsize.bench import Generation


@dataclass
class RecordedRunner:
    """Replay a runtime's characteristics deterministically.

    Given a time to first token, a decode rate, and a memory figure, it
    synthesizes timings that are internally consistent: total time is
    prefill plus however long the stated rate takes to emit the tokens. That
    consistency is what makes it useful for exercising the harness, since a
    bug in the aggregation shows up as an implausible output.
    """

    name: str
    ttft_s: float
    tokens_per_sec: float
    peak_memory_gib: float
    answers: dict[str, str] = field(default_factory=dict)
    default_answer: str = "I do not know."
    # Charged on every call, including warmups, so a test can prove the
    # warmup runs were actually discarded.
    calls: int = 0

    def generate(self, prompt: str, max_tokens: int) -> Generation:
        self.calls += 1
        text = self.answers.get(prompt, self.default_answer)
        completion_tokens = min(max_tokens, _rough_token_count(text))
        decode_s = (
            (completion_tokens - 1) / self.tokens_per_sec
            if completion_tokens > 1
            else 0.0
        )
        return Generation(
            text=text,
            prompt_tokens=_rough_token_count(prompt),
            completion_tokens=completion_tokens,
            ttft_s=self.ttft_s,
            total_s=self.ttft_s + decode_s,
            peak_memory_gib=self.peak_memory_gib,
        )


def _rough_token_count(text: str) -> int:
    """A stand-in for a tokenizer, adequate for synthetic traces only.

    Real token counts come from the model's own tokenizer, and the gap
    between families is exactly why tokens per second does not compare
    across them.
    """
    return max(1, len(text) // 4)
