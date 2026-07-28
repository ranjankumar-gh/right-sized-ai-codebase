"""Chapter 1: estimate a candidate's footprint before downloading it."""

from rightsize.sizing import (
    ModelShape,
    Workload,
    decode_ceiling_tokens_per_sec,
    estimate_footprint,
)

# A representative 7B-class dense shape. Read your own model's real values
# from its config.json rather than trusting any table.
candidate = ModelShape(
    name="7B-class dense",
    total_params_b=7.0,
    active_params_b=7.0,
    layers=32,
    kv_heads=8,
    head_dim=128,
)

serving = Workload(context_tokens=8192, batch_size=16)
footprint = estimate_footprint(candidate, serving, weight_bits=4)

low, high = footprint.total_low_gib, footprint.total_high_gib
print(f"weights:   {footprint.weights_gib:.1f} GiB")
print(f"kv cache:  {footprint.kv_cache_gib:.1f} GiB")
print(f"total:     {low:.1f}-{high:.1f} GiB")
print(f"fits 24GB: {footprint.fits_in(24.0)}")
print(f"ceiling:   {decode_ceiling_tokens_per_sec(candidate, 900.0, 4):.0f} tok/s")
