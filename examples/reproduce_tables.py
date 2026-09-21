"""Reprint every computed table from Chapters 11 to 15.

Chapters 1 to 10 each ship an example script that produces their output.
Chapters 11 to 15 print their tables from the modules directly, so this
script stands in for the missing per-chapter examples and lets a reader
reproduce every published figure in Part V with one command:

    uv run python examples/reproduce_tables.py

No GPU, no weights, no network. Everything here is arithmetic over the
inputs the chapters state.
"""

from rightsize.cases import DeploymentProfile, select_techniques
from rightsize.edge import DutyCycle, EnergyProfile, ThermalProfile
from rightsize.fleet import FleetEconomics, RolloutProgress, UpdatePlan
from rightsize.limits import SizeResponse
from rightsize.serve import ServingCapacity, block_waste_fraction
from rightsize.sizing import ModelShape

SEVEN_B = ModelShape(
    name="7B-class dense",
    total_params_b=7.0,
    active_params_b=7.0,
    layers=32,
    kv_heads=8,
    head_dim=128,
)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def chapter_11() -> None:
    rule("Ch11 @tbl-ceiling: concurrency ceiling, 7B at 4-bit")
    print(f"{'device':>8}{'2k':>8}{'8k':>8}{'32k':>8}")
    for device in (24, 48, 80):
        row = "".join(
            f"{ServingCapacity(SEVEN_B, device, ctx).concurrency_ceiling:>8}"
            for ctx in (2048, 8192, 32768)
        )
        print(f"{device:>6} GiB{row}")

    rule("Ch11 @tbl-blocks: KV wasted in the final block")
    print(f"{'tokens':>8}{'block 16':>10}{'block 32':>10}{'block 64':>10}")
    for length in (37, 150, 2000):
        row = "".join(
            f"{block_waste_fraction(length, block):>9.1%} "
            for block in (16, 32, 64)
        )
        print(f"{length:>8}{row}")


def chapter_12() -> None:
    rule("Ch12 @tbl-derate: what a benchmark reports, by duration")
    device = ThermalProfile(40.0, sustained_fraction=0.60,
                            seconds_to_throttle=30.0)
    for window in (10, 30, 60, 300):
        reported = device.average_over(window)
        share = reported / device.peak_tokens_per_sec
        print(f"{window:>6} s  {reported:>6.1f} tok/s  {share:>6.0%} of peak")
    print(f"{'sustained':>8}  {device.sustained_tokens_per_sec:>6.1f} tok/s  "
          f"{device.sustained_fraction:>6.0%} of peak")

    rule("Ch12 @tbl-battery: a 20 Wh battery, by energy per token")
    for millijoules in (25, 100, 270, 500):
        profile = EnergyProfile(millijoules)
        print(f"{millijoules:>5} mJ/token  "
              f"{profile.tokens_from_battery(20):>12,.0f} tokens  "
              f"{profile.responses_from_battery(20):>9,.0f} responses")

    rule("Ch12 @tbl-duty: inference as a share of the energy budget")
    print(f"{'idle':>7}{'10/hr':>9}{'100/hr':>9}{'1000/hr':>9}")
    for idle in (0.1, 0.5, 2.0):
        row = "".join(
            f"{DutyCycle(EnergyProfile(100.0, idle), rate).inference_share:>9.0%}"
            for rate in (10, 100, 1000)
        )
        print(f"{idle:>5} W{row}")


def chapter_13() -> None:
    rule("Ch13 @tbl-fleet: owned against rented, 1,000 devices at $500")
    for rate in (100, 1000, 6944, 10000, 100000):
        economics = FleetEconomics(1000, 500.0, 36, rate, 0.002)
        if abs(economics.advantage - 1) < 0.01:
            verdict = "break-even"
        elif economics.advantage > 1:
            verdict = f"{economics.advantage:.1f}x better"
        else:
            # Report the inverse, because "0.0x worse" reads as nothing.
            verdict = f"{1 / economics.advantage:.0f}x worse"
        print(f"{rate:>8,}/device/mo  "
              f"${economics.owned_cost_per_request:>9.5f}  {verdict}")

    rule("Ch13 @tbl-rollout: 2 GB to 10,000 devices, shared 100 Mbps")
    for label, fraction in (("full model", 1.0), ("delta 10%", 0.10),
                            ("delta 5%", 0.05)):
        plan = UpdatePlan(2048, 10000, fraction, 1.0, 100.0)
        print(f"{label:>11}  {plan.transferred_mb_per_device:>7,.0f} MB/device  "
              f"{plan.total_transferred_gb / 1024:>5.1f} TB  "
              f"{plan.days_through_shared_uplink:>5.1f} days")

    rule("Ch13 @tbl-tail: version tail, 70% reach, 1% permanently offline")
    progress = RolloutProgress(0.70, permanently_unreachable=0.01)
    for day in (1, 3, 7, 30):
        print(f"  day {day:>2}  {progress.stale_fraction_after(day):>7.2%} stale")


def chapter_14() -> None:
    rule("Ch14 @tbl-selection: how five deployments select from the book")
    cases = [
        DeploymentProfile("Support assistant", "server", 40),
        DeploymentProfile("Local code assistant", "workstation", 1),
        DeploymentProfile("Industrial gateway", "gateway", 2,
                          owns_hardware=True, fleet_size=1200),
        DeploymentProfile("Handheld inspector", "battery", 1,
                          owns_hardware=True, battery_powered=True,
                          inference_energy_share=0.04, fleet_size=4000),
        DeploymentProfile("Nightly extraction", "batch", 256,
                          interactive=False),
    ]
    print(f"{'deployment':<22}{'primary constraint':<44}"
          f"{'bind':>5}{'avail':>7}{'inert':>7}")
    for case in cases:
        selection = select_techniques(case)
        print(f"{case.name:<22}{case.primary_constraint:<44}"
              f"{len(selection.binding):>5}{len(selection.available):>7}"
              f"{len(selection.inert):>7}")


def chapter_15() -> None:
    rule("Ch15 @tbl-size-response: can the response be told from noise?")
    shapes = [
        ("1B 70 / 3B 71 / 7B 72", [(1.0, 0.700), (3.0, 0.710), (7.0, 0.720)]),
        ("1B 70 / 3B 70.2 / 7B 70.5",
         [(1.0, 0.700), (3.0, 0.702), (7.0, 0.705)]),
        ("1B 45 / 3B 62 / 7B 78", [(1.0, 0.450), (3.0, 0.620), (7.0, 0.780)]),
        ("1B 62 / 3B 66.5 / 7B 70",
         [(1.0, 0.620), (3.0, 0.665), (7.0, 0.700)]),
    ]
    print(f"{'response':<28}{'span':>7}{'200':>18}{'5,000':>18}{'50,000':>18}")
    for label, points in shapes:
        row = "".join(
            f"{SizeResponse(points, trials).verdict:>18}"
            for trials in (200, 5000, 50000)
        )
        print(f"{label:<28}{SizeResponse(points, 200).span * 100:>6.1f}p{row}")


if __name__ == "__main__":
    for chapter in (chapter_11, chapter_12, chapter_13, chapter_14, chapter_15):
        chapter()
    print()
