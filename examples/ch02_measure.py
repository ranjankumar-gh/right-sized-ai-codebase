"""Chapter 2: measure two configurations and compare them honestly.

Uses the recorded backend, so this runs anywhere with no GPU and no model
download. Every number below is synthetic and illustrative. It shows how to
read a comparison, not what to expect from a real model.
"""

from dataclasses import replace

from rightsize.backends.recorded import RecordedRunner
from rightsize.bench import (
    IncomparableRecords,
    RunRecord,
    Task,
    TaskSet,
    compare,
    measure,
)
from rightsize.sizing import ModelShape, Workload, estimate_footprint

tasks = TaskSet(
    name="capitals-smoke",
    tasks=(
        Task("Capital of France?", expected="Paris"),
        Task("Capital of Japan?", expected="Tokyo"),
        Task("Capital of Peru?", expected="Lima"),
        Task("Capital of Kenya?", expected="Nairobi"),
    ),
)

CORRECT = {
    "Capital of France?": "The capital of France is Paris.",
    "Capital of Japan?": "The capital of Japan is Tokyo.",
    "Capital of Peru?": "The capital of Peru is Lima.",
    "Capital of Kenya?": "The capital of Kenya is Nairobi.",
}
# The 4-bit variant loses one of the four. On a set this small that is a
# single example, which is exactly why Chapter 9 exists.
LOSSY = {**CORRECT, "Capital of Peru?": "The capital of Peru is Bogota."}

shape = ModelShape("3B-class dense", 3.0, 3.0, 28, 4, 128)
workload = Workload(context_tokens=2048, batch_size=1)


def record(runner: RecordedRunner, quantization: str, bits: int) -> RunRecord:
    return RunRecord(
        model="small-3b",
        quantization=quantization,
        runtime="recorded",
        task_set=tasks.name,
        task_set_fingerprint=tasks.fingerprint(),
        max_tokens=64,
        batch_size=workload.batch_size,
        hardware="synthetic",
        recorded_at="2026-07-30",
        measurement=measure(runner, tasks, max_tokens=64),
        predicted=estimate_footprint(shape, workload, weight_bits=bits),
    )


eight_bit = record(
    RecordedRunner("q8", 0.31, 34.0, 4.9, answers=CORRECT), "q8", 8
)
four_bit = record(
    RecordedRunner("q4", 0.22, 58.0, 2.8, answers=LOSSY), "q4", 4
)

for run in (eight_bit, four_bit):
    m = run.measurement
    gap = run.memory_prediction_gap_gib()
    print(f"[{run.quantization}] quality {m.quality:.2f}  "
          f"ttft p95 {m.ttft_p95_s:.2f}s  decode {m.decode_tokens_per_sec:.0f} tok/s")
    print(f"       peak {m.peak_memory_gib:.1f} GiB  "
          f"predicted {run.predicted.total_low_gib:.1f}-"
          f"{run.predicted.total_high_gib:.1f} GiB  gap {gap:+.1f} GiB")

delta = compare(eight_bit, four_bit)
print(f"\nvaried: {delta.varied[0]}")
print(f"  quality     {delta.quality_delta:+.2f}")
print(f"  ttft p95    {delta.ttft_p95_delta_s:+.2f} s")
print(f"  decode      {delta.decode_rate_delta:+.0f} tok/s")
print(f"  peak memory {delta.peak_memory_delta_gib:+.1f} GiB")

# What the guard is for. Somebody adds an example to the task set between
# the two runs, so the diff now reflects both the quantization change and a
# different set of questions. The harness refuses instead of returning a
# number that would get quoted in a decision meeting.
grown = TaskSet(
    tasks.name, tasks.tasks + (Task("Capital of Chile?", "Santiago"),)
)
contaminated = replace(four_bit, task_set_fingerprint=grown.fingerprint())

try:
    compare(eight_bit, contaminated)
except IncomparableRecords as exc:
    print(f"\nrefused: {exc}")
