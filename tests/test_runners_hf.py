"""Tests for the transformers runner.

The generation test needs real weights, so it is skip-guarded behind
RIGHTSIZE_HF_MODEL. Everything checkable without a download is checked
unconditionally, including that the module imports with no torch installed.
"""

import importlib.util
import os

import pytest

from rightsize.bench import Runner, Task, TaskSet, measure
from rightsize.runners.hf import TransformersRunner

HF_MODEL = os.environ.get("RIGHTSIZE_HF_MODEL")
requires_weights = pytest.mark.skipif(
    not HF_MODEL,
    reason="set RIGHTSIZE_HF_MODEL to a small model id to run this",
)
requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers is not installed",
)


class TestConstruction:
    """Runs with no torch and no transformers present."""

    def test_importable_without_a_gpu_stack(self) -> None:
        runner = TransformersRunner(model_id="some/model")
        assert runner.model_id == "some/model"

    def test_derives_a_name_from_the_model_id(self) -> None:
        runner = TransformersRunner(model_id="some/model")
        assert runner.name == "transformers:some/model"

    def test_explicit_name_is_kept(self) -> None:
        runner = TransformersRunner(model_id="some/model", name="baseline")
        assert runner.name == "baseline"

    def test_satisfies_the_runner_protocol(self) -> None:
        assert isinstance(TransformersRunner(model_id="some/model"), Runner)

    def test_nothing_is_loaded_on_construction(self) -> None:
        """Loading in __init__ would put weight loading inside whichever
        call happens first, which is the cost the warmup boundary exists
        to separate out."""
        runner = TransformersRunner(model_id="some/model")
        assert runner._model is None
        assert runner._tokenizer is None


@requires_transformers
@requires_weights
class TestGeneration:
    def test_produces_a_usable_generation(self) -> None:
        runner = TransformersRunner(model_id=HF_MODEL, device="cpu")
        gen = runner.generate("The capital of France is", max_tokens=8)
        assert gen.text
        assert gen.prompt_tokens > 0
        assert gen.completion_tokens > 0
        assert gen.total_s >= gen.ttft_s

    def test_peak_memory_is_nan_off_cuda(self) -> None:
        """Better an explicit non-answer than a plausible wrong number."""
        runner = TransformersRunner(model_id=HF_MODEL, device="cpu")
        gen = runner.generate("Hello", max_tokens=4)
        assert gen.peak_memory_gib != gen.peak_memory_gib  # NaN

    def test_works_through_the_harness(self) -> None:
        runner = TransformersRunner(model_id=HF_MODEL, device="cpu")
        tasks = TaskSet(
            "tiny", (Task("The capital of France is", expected="Paris"),)
        )
        result = measure(runner, tasks, max_tokens=8, warmup_runs=1)
        assert result.samples == 1
