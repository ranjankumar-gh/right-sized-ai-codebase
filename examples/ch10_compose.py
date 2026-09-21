from rightsize.compose import CascadeLatency

cascade = CascadeLatency(small_ms=120, large_ms=900, escalation_rate=0.15)
print(f"always-large p99:  {900:>6.0f} ms")
print(f"cascade mean:      {cascade.mean_ms:>6.0f} ms")
print(f"cascade p50:       {cascade.quantile_ms(0.50):>6.0f} ms")
print(f"cascade p99:       {cascade.quantile_ms(0.99):>6.0f} ms")
print(f"tail penalty:      {cascade.tail_penalty:>6.2f}x")
