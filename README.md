# Right-Sized AI - Companion Code

Companion repository for *Right-Sized AI: Small Language Models in Production*. Builds one
running project, **Rightsize**, incrementally across the book's chapters.

Rightsize is a constraint-aware measurement and deployment toolkit. It starts in Chapter 2 as a
harness that measures one model against one envelope, and by Part V the same harness is training,
compressing, serving, and shipping models to edge hardware.

## Tags

Each chapter and appendix has its own git tag, matching the book's chapter slugs
(e.g. `ch02-your-first-right-sized-model`). Check out a tag to see the codebase exactly as it
stood at the end of that chapter:

```
git checkout <slug>
```

## Running the tests

```
uv run pytest tests/ -v
```

Tests run without a GPU and without downloading model weights. Anything that needs real hardware
or a real model is skip-guarded behind an environment variable and documented in the chapter
section below; those tests skip cleanly and are not required for the rest of the suite to pass.

To run the guarded ones as well, install the `hf` extra and name a small model:

```
uv run --extra hf --with pytest python -m pytest tests/ -v
```

with `RIGHTSIZE_HF_MODEL` set. `hf-internal-testing/tiny-random-gpt2` downloads in seconds and
is enough to exercise the runner end to end:

```
RIGHTSIZE_HF_MODEL=hf-internal-testing/tiny-random-gpt2    # bash
$env:RIGHTSIZE_HF_MODEL = "hf-internal-testing/tiny-random-gpt2"   # PowerShell
```

The guarded tests are worth running at least once. A guard that has never been lifted is an
assumption, not a check.

## Chapter notes

**Chapter 1** adds `rightsize/sizing.py`. Pure arithmetic, no dependencies. Run
`uv run python examples/ch01_sizing.py` to estimate a footprint.

**Chapter 2** adds `rightsize/bench.py`, the recorded backend, and
`rightsize/runners/hf.py`. Run `uv run python examples/ch02_measure.py` for a worked
comparison against the recorded backend, which needs no GPU and no weights.

To exercise the real transformers runner you need `torch` and `transformers` installed,
which this repository deliberately does not pin as dependencies, and a model to point at:

```
uv pip install torch transformers
RIGHTSIZE_HF_MODEL=<a-small-model-id> uv run pytest tests/test_runners_hf.py -v
```

Without that variable set, those tests skip and the rest of the suite passes unaffected.

**Chapter 3** adds `rightsize/capability.py`. Pure arithmetic over a model card and a
`config.json`, with no dependencies and no network access. Run
`uv run python examples/ch03_capability.py` to compute a model's overtraining multiple and
its parameter budget. The tests verify the budget against Google's published 170M/100M
split for Gemma 3 270M and against SmolLM3 3B's released parameter count.

**Chapter 4** adds `rightsize/adapt.py`: the LoRA adapter budget, the training-memory
estimator that chooses between full fine-tuning, LoRA, and QLoRA, the loss-mask and
chat-template guards, and the deployment gate. The module itself has no ML dependencies
and its tests run without a GPU.

`examples/ch04_finetune.py` and `examples/ch04_preference.py` are real training scripts
and do need the training stack, which this repository deliberately does not pin:

```
uv pip install torch transformers peft trl datasets
```

**Chapter 5** adds `rightsize/compress.py`: pruning plans read through Chapter 3's
`parameter_budget`, the embedding floor that limits them, activation-importance selection,
the distillation-mode decision, and the KD objective implemented on one position so it can
be read and tested without a GPU. `examples/ch05_distill.py` carries the batched torch
version and needs `torch` installed.

**Chapter 6** adds `rightsize/quantize.py`: a readable quantizer, the bit and group-size
sweeps, percentile clipping, and the quantization plan. Pure arithmetic on a seeded
Gaussian sample, so every figure the chapter prints reproduces in a second without a GPU.
Run `uv run python examples/ch06_quantize.py`.

**Chapter 7** adds `rightsize/artifact.py`: a parser for llama.cpp quantization type names,
a runtime support matrix that carries its own verification date, and an artifact lineage
record that detects a model quantized twice. The `VLLM` value is a snapshot verified on one
date, not a standard - re-verify it before trusting it.

**Chapter 8** adds `rightsize/structured.py`: the token-masking mechanism constrained
decoding runs on, the four-number report that keeps a validity guarantee apart from a
correctness claim, and a schema complexity score. Pure Python, no inference engine needed.
Run `uv run python examples/ch08_constrained.py`.

**Chapter 9** adds `rightsize/evaluate.py`: Wilson intervals, the resolution floor and its
inverse (examples required to detect a given difference), per-slice evaluation with each
slice scored against its own floor, a position-bias guard for model judges, n-gram
contamination scoring, and the promotion gate. No dependencies, no inference needed.

**Chapter 10** adds `rightsize/compose.py`: cascade economics with the break-even and
headroom, the cascade latency tax and its bimodal quantiles, the top-two-margin escalation
signal with the Chapter 8 schema interlock, speculative decoding speedup and optimal draft
length, and the composition gate. Run `uv run python examples/ch10_compose.py`.

**Chapter 11** adds `rightsize/serve.py`: the concurrency ceiling (Chapter 1's KV arithmetic
inverted), block accounting and fragmentation, prefix-cache saving, goodput and preemption
rate, and context-aware admission control. The chapter prints its tables from the module
directly rather than through an example script; see `tests/test_serve.py` for worked values.

**Chapter 12** adds `rightsize/edge.py`: the thermal derate and what a benchmark of a given
length actually reports, the energy envelope in tokens per joule, the duty-cycle split that
decides whether model work matters at all on a battery device, no-swap memory headroom, and
the unattended-readiness checklist.

**Chapter 13** adds `rightsize/fleet.py`: the utilization inversion and its break-even,
rollout duration over constrained links, version-tail convergence, and the telemetry
allow-list. Pure arithmetic, no fleet required.

**Chapter 14** adds `rightsize/cases.py`: `DeploymentProfile`, the primary-constraint rule,
and `select_techniques`, which sorts the book into what binds, what merely works, and what
cannot apply to a given deployment. The inert list is the one worth reading first.

**Chapter 15** adds `rightsize/limits.py`: `SizeResponse`, which says whether a measured
quality difference across model sizes can be distinguished from noise on your task set, and
`LimitCheck`, the gate that must pass before escalating model size. It is the one gate in
the book that blocks a retreat rather than a change.

## Reproducing the book's tables

Chapters 1 to 10 each ship their own example script. Chapters 11 to 15 print their tables
from the modules directly, so one script reproduces all of Part V:

```
uv run python examples/reproduce_tables.py
```

Every figure it prints appears in the book, and every one is also asserted in `tests/`, so
a change that moves a number fails the suite rather than silently contradicting the text.

## License

MIT. See [LICENSE](LICENSE).

The code is deliberately more permissive than the book it accompanies. The book's prose is
all rights reserved; this repository is yours to use in production, commercial work included,
without asking.
