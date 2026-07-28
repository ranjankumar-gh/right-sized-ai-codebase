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

Setup notes specific to a chapter are added here as that chapter's increment is built.
