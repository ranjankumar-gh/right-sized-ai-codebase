"""A Runner over Hugging Face transformers.

The one runner in this repository that touches a real model. Its job is to
report timings the harness cannot see from outside: when the first token
actually arrived, and how much memory the device held at its peak.

transformers and torch are imported lazily so the rest of the package, and
its test suite, work without them installed.
"""

import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from rightsize.bench import Generation


@dataclass
class TransformersRunner:
    """Generate through transformers, timing prefill and decode separately.

    Pass `device="cuda"` for real peak-memory figures. On CPU and MPS the
    peak_memory_gib field is reported as NaN rather than as a plausible
    wrong number, because torch does not expose an equivalent high-water
    mark and an RSS reading includes the whole process.
    """

    model_id: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    name: str = ""
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"transformers:{self.model_id}"

    def load(self) -> None:
        """Load weights. Called on first generate if not called explicitly.

        Worth calling by hand when you care where the load cost lands: the
        harness discards warmup runs, but it cannot discard a load that
        happens inside the first one.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=getattr(torch, self.dtype),
        ).to(self.device)
        self._model.eval()

    def generate(self, prompt: str, max_tokens: int) -> Generation:
        import torch
        from transformers import TextIteratorStreamer

        if self._model is None:
            self.load()

        # Reset before the request so the high-water mark covers prefill,
        # which is where the peak occurs. Reading without resetting reports
        # the peak of the whole session instead of this request.
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            streamer=streamer,
            # Greedy, because sampling adds variance the harness would
            # report as a property of the model.
            do_sample=False,
        )
        thread = Thread(target=self._model.generate, kwargs=kwargs)

        started = time.perf_counter()
        thread.start()

        chunks: list[str] = []
        ttft_s = float("nan")
        for chunk in streamer:
            if not chunks:
                ttft_s = time.perf_counter() - started
            chunks.append(chunk)
        thread.join()
        total_s = time.perf_counter() - started

        text = "".join(chunks)
        completion_tokens = len(
            self._tokenizer(text, add_special_tokens=False)["input_ids"]
        )

        return Generation(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft_s,
            total_s=total_s,
            peak_memory_gib=self._peak_memory_gib(),
        )

    def _peak_memory_gib(self) -> float:
        import torch

        if not self.device.startswith("cuda"):
            return float("nan")
        return torch.cuda.max_memory_allocated() / 1024**3
