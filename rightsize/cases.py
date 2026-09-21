from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentProfile:
    """Enough about a deployment to say which chapters apply.

    Deliberately small. Every field here changes which techniques are
    worth reaching for, and anything that does not has been left out.
    """

    name: str
    serving_class: str  # server, workstation, gateway, battery, batch
    concurrent_requests: int
    owns_hardware: bool = False
    battery_powered: bool = False
    inference_energy_share: float = 1.0
    interactive: bool = True
    fleet_size: int = 1

    @property
    def primary_constraint(self) -> str:
        """The bound that decides what you build.

        Ordered, not scored: the first condition that matches wins,
        because a deployment has one primary constraint by definition
        and the ordering encodes which dominates when several apply.
        """
        if self.battery_powered and self.inference_energy_share < 0.20:
            return "idle power"
        if self.battery_powered:
            return "energy per token"
        if self.serving_class == "gateway":
            return "sustained throughput under a thermal limit"
        if self.concurrent_requests <= 1:
            return "single-stream decode speed"
        if not self.interactive:
            return "throughput per unit cost"
        return "concurrency at the target context length"


Predicate = Callable[[DeploymentProfile], bool]

# (technique, binds when, inert when). A technique that is neither
# binding nor inert is available: it will work and will not move the
# number that decides this deployment.
TECHNIQUES: list[tuple[str, Predicate, Predicate]] = [
    ("Ch1 footprint estimate", lambda p: True, lambda p: False),
    ("Ch2 measurement harness", lambda p: True, lambda p: False),
    ("Ch4 adaptation",
     lambda p: p.inference_energy_share >= 0.20,
     lambda p: False),
    ("Ch5 pruning and distillation",
     lambda p: p.serving_class in {"gateway", "battery"}
     and p.inference_energy_share >= 0.20,
     lambda p: p.serving_class in {"server", "batch"}),
    ("Ch6 quantization",
     lambda p: p.inference_energy_share >= 0.20,
     lambda p: False),
    ("Ch7 format and runtime choice", lambda p: True, lambda p: False),
    ("Ch8 constrained output", lambda p: True, lambda p: False),
    ("Ch9 slices and gates", lambda p: True, lambda p: False),
    ("Ch10 cascade or routing",
     lambda p: p.serving_class in {"server", "batch"},
     lambda p: p.serving_class in {"gateway", "battery"}),
    ("Ch10 speculative decoding",
     lambda p: p.concurrent_requests <= 2 and not p.battery_powered,
     lambda p: p.concurrent_requests > 8),
    ("Ch11 batching, paged KV, prefix cache",
     lambda p: p.concurrent_requests > 1,
     lambda p: p.concurrent_requests <= 1),
    ("Ch12 thermal and energy budget",
     lambda p: p.serving_class in {"gateway", "battery"},
     lambda p: p.serving_class in {"server", "batch", "workstation"}),
    ("Ch13 fleet economics and rollout",
     lambda p: p.fleet_size > 1 and p.owns_hardware,
     lambda p: p.fleet_size <= 1),
]


@dataclass(frozen=True)
class TechniqueSelection:
    binding: list[str]
    available: list[str]
    inert: list[str]


def select_techniques(profile: DeploymentProfile) -> TechniqueSelection:
    """Sort the book into what binds, what works, and what cannot apply.

    The inert list is the one worth reading first. It is the work you
    would otherwise do before discovering the premise was false, which
    is what the chapter's opening cost six weeks.
    """
    binding, available, inert = [], [], []
    for name, binds, is_inert in TECHNIQUES:
        if is_inert(profile):
            inert.append(name)
        elif binds(profile):
            binding.append(name)
        else:
            available.append(name)
    return TechniqueSelection(binding, available, inert)
