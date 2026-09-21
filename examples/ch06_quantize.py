import random

from rightsize.quantize import quantize_tensor, relative_error

random.seed(11)
weights = [random.gauss(0.0, 0.02) for _ in range(4096)]

print("bits  relative error")
for bits in (8, 6, 4, 3, 2):
    restored = quantize_tensor(weights, bits=bits, group_size=128)
    print(f"{bits:>4}  {relative_error(weights, restored):>13.2%}")


from rightsize.sizing import bytes_per_weight

print("group  relative error  bytes/weight  vs group 128")
baseline = bytes_per_weight(4, group_size=128)
for group in (4096, 256, 128, 64, 32):
    restored = quantize_tensor(weights, bits=4, group_size=group)
    cost = bytes_per_weight(4, group_size=group)
    print(
        f"{group:>5}  {relative_error(weights, restored):>13.2%}  "
        f"{cost:>12.4f}  {cost / baseline - 1:>+11.1%}"
    )


from rightsize.quantize import quantize_group, dequantize_group

group = weights[:128]
codes, scale, zero = quantize_group(group, bits=4)
clean = relative_error(group, dequantize_group(codes, scale, zero))
print(f"clean group of 128:            {clean:.2%}")

for multiple in (10, 50, 200):
    spiked = list(group)
    spiked[64] = max(abs(v) for v in group) * multiple
    codes, scale, zero = quantize_group(spiked, bits=4)
    restored = dequantize_group(codes, scale, zero)
    # Error on the 127 innocent values, excluding the outlier itself.
    others = relative_error(
        group[:64] + group[65:], restored[:64] + restored[65:]
    )
    print(f"one value {multiple:>3}x larger:  others at {others:.2%}")


from rightsize.quantize import clipped_range

spiked = list(group)
spiked[64] = max(abs(v) for v in group) * 10

codes, scale, zero = quantize_group(spiked, bits=4)
restored = dequantize_group(codes, scale, zero)
minmax = relative_error(
    group[:64] + group[65:], restored[:64] + restored[65:]
)
print(f"outlier present, min-max grid:   {minmax:.2%}")

low, high = clipped_range(spiked, keep=0.98)
clamped = [max(low, min(high, v)) for v in spiked]
codes, scale, zero = quantize_group(clamped, bits=4)
restored = dequantize_group(codes, scale, zero)
others = relative_error(
    group[:64] + group[65:], restored[:64] + restored[65:]
)
print(f"outlier present, 98% clipped:    {others:.2%}")


from rightsize.sizing import ModelShape, Workload, estimate_footprint

shape = ModelShape(
    name="7B-class dense",
    total_params_b=7.0,
    active_params_b=7.0,
    layers=32,
    kv_heads=8,
    head_dim=128,
)
serving = Workload(context_tokens=8192, batch_size=16)

print("kv bits  weights   cache    total")
for kv_bits in (16, 8, 4):
    footprint = estimate_footprint(shape, serving, weight_bits=4, kv_bits=kv_bits)
    print(
        f"{kv_bits:>7}  {footprint.weights_gib:>6.1f}  "
        f"{footprint.kv_cache_gib:>6.1f}  "
        f"{footprint.total_low_gib:>4.1f}-{footprint.total_high_gib:.1f} GiB"
    )
