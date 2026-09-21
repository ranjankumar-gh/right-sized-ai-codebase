"""Chapter 9: the five passes end to end, on the numbers the chapter prints.

Every input is either a published figure (named where it is used) or a
seeded synthetic set, labelled as such. The output shows what the library
decides for each input; it is not a measurement of any real model.
"""

import random

from rightsize.evaluate import (
    PairwiseVerdict,
    PromotionGate,
    SliceResult,
    SlicedEvaluation,
    consistent_only,
    position_bias_rate,
    required_trials,
    resolution_floor,
    wilson_interval,
)

# Pass 1: what a 70% accuracy claims, by task-set size.
print("Pass 1: 95% interval around 70% accuracy")
for n in (40, 200, 1000, 5000):
    low, high = wilson_interval(round(0.70 * n), n)
    print(f"  {n:>5} examples: {low:.1%} to {high:.1%}")

# Pass 2: the floor, in both directions.
print("\nPass 2: resolution floor from a 70% baseline")
for n in (200, 1000):
    print(f"  {n:>5} examples resolve {resolution_floor(n):.1%}")
for improvement in (0.02, 0.05, 0.10):
    print(f"  detecting {improvement:.0%} needs {required_trials(0.70, improvement)} per arm")

# Pass 3: a slice gate, and what the tolerance does to it.
# Synthetic: the aggregate rises while the rare slice falls, as in Chapter 5.
incumbent = SlicedEvaluation([
    SliceResult("common", 870, 1000),
    SliceResult("rare", 150, 200),
])
candidate = SlicedEvaluation([
    SliceResult("common", 915, 1000),
    SliceResult("rare", 120, 200),
])
print(f"\nPass 3: overall {incumbent.overall:.1%} -> {candidate.overall:.1%}")
for tolerance in (0.0, 0.10):
    print(f"  tolerance {tolerance:.0%}: {candidate.regressions_against(incumbent, tolerance)}")

# A large slice and a small slice, each dropping 5 points. Floors shrink as
# slices grow, so a tolerance silences the large slice first.
big_before = SlicedEvaluation([SliceResult("large", 850, 1000)])
big_after = SlicedEvaluation([SliceResult("large", 800, 1000)])
small_before = SlicedEvaluation([SliceResult("small", 30, 40)])
small_after = SlicedEvaluation([SliceResult("small", 28, 40)])
print(f"  floors: large {big_after.slices[0].floor:.1%}, small {small_after.slices[0].floor:.1%}")
for tolerance in (0.0, 0.06):
    big = big_after.regressions_against(big_before, tolerance)
    small = small_after.regressions_against(small_before, tolerance)
    print(f"  tolerance {tolerance:.0%}: large slice blocked={bool(big)}, small slice blocked={bool(small)}")

# Pass 4: a synthetic judge, seeded, that flips on 18% of swapped pairs.
rng = random.Random(9)
verdicts = []
for i in range(100):
    forward = rng.choice(["a", "b"])
    if rng.random() < 0.18:
        reverse = "b" if forward == "a" else "a"
    else:
        reverse = forward
    verdicts.append(PairwiseVerdict(f"item-{i:03d}", forward, reverse))
rate = position_bias_rate(verdicts)
kept = consistent_only(verdicts)
print(f"\nPass 4: judge flipped on {rate:.1%} of 100 swapped pairs; "
      f"{len(verdicts) - len(kept)} dropped, {len(kept)} kept")

# Pass 5: the promotion gate, on three cases.
gate = PromotionGate(min_trials=100)

# (a) The chapter's opening: 72% against 70% on 200 examples.
opening_candidate = SlicedEvaluation([SliceResult("all", 144, 200)])
opening_incumbent = SlicedEvaluation([SliceResult("all", 140, 200)])
print("\nPass 5 (a) 72% against 70% on 200 examples:")
for problem in gate.blockers(opening_candidate, opening_incumbent):
    print(f"  BLOCKED: {problem}")

# (b) Chapter 5's pruned model, with the judge rate from Pass 4 applied.
print("\nPass 5 (b) the pruned model, judge flip rate from Pass 4 applied:")
for problem in gate.blockers(candidate, incumbent, position_bias=rate):
    print(f"  BLOCKED: {problem}")

# (c) Chapter 8's published run, pooled as one slice: 15,000 generations,
# answer accuracy 19.7% freeform and 11.0% under the grammar. The counts are
# rounded from the paper's figures, which are not reproduced in this repository.
freeform = SlicedEvaluation([SliceResult("pooled", round(0.197 * 15000), 15000)])
grammar = SlicedEvaluation([SliceResult("pooled", round(0.110 * 15000), 15000)])
print("\nPass 5 (c) Chapter 8's grammar run, answer accuracy only:")
for problem in gate.blockers(grammar, freeform):
    print(f"  BLOCKED: {problem}")
