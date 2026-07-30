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
