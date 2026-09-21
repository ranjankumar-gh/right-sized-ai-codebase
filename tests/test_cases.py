"""Chapter 14: primary constraint and technique selection across five cases."""

from rightsize.cases import DeploymentProfile, select_techniques

SUPPORT = DeploymentProfile("Support assistant", "server", 40)
WORKSTATION = DeploymentProfile("Local code assistant", "workstation", 1)
GATEWAY = DeploymentProfile(
    "Industrial gateway", "gateway", 2, owns_hardware=True, fleet_size=1200
)
HANDHELD = DeploymentProfile(
    "Handheld inspector", "battery", 1, owns_hardware=True,
    battery_powered=True, inference_energy_share=0.04, fleet_size=4000,
)
BATCH = DeploymentProfile(
    "Nightly extraction", "batch", 256, interactive=False
)
ALL_CASES = [SUPPORT, WORKSTATION, GATEWAY, HANDHELD, BATCH]


# --- The primary constraint -------------------------------------------------


def test_five_deployments_have_five_different_constraints():
    """The chapter's premise: the same shortlist produces five answers."""
    constraints = {case.primary_constraint for case in ALL_CASES}
    assert len(constraints) == len(ALL_CASES)


def test_the_published_constraints():
    assert SUPPORT.primary_constraint == "concurrency at the target context length"
    assert WORKSTATION.primary_constraint == "single-stream decode speed"
    assert GATEWAY.primary_constraint == "sustained throughput under a thermal limit"
    assert HANDHELD.primary_constraint == "idle power"
    assert BATCH.primary_constraint == "throughput per unit cost"


def test_idle_power_outranks_energy_per_token_when_the_share_is_low():
    """Ordered, not scored: a low inference share dominates, because
    optimising energy per token cannot move a 4% term."""
    busy = DeploymentProfile("busy", "battery", 1, battery_powered=True,
                             inference_energy_share=0.80)
    assert busy.primary_constraint == "energy per token"
    assert HANDHELD.primary_constraint == "idle power"


def test_a_batch_pipeline_has_no_latency_constraint():
    """Which reverses the concurrency arithmetic: nothing is waiting."""
    assert "throughput" in BATCH.primary_constraint
    assert "latency" not in BATCH.primary_constraint


# --- Selection: the published table -----------------------------------------


def test_the_published_selection_table():
    """@tbl-selection."""
    expected = {
        "Support assistant": (9, 0, 4),
        "Local code assistant": (8, 2, 3),
        "Industrial gateway": (12, 0, 1),
        "Handheld inspector": (7, 4, 2),
        "Nightly extraction": (9, 0, 4),
    }
    for case in ALL_CASES:
        selection = select_techniques(case)
        counts = (
            len(selection.binding),
            len(selection.available),
            len(selection.inert),
        )
        assert counts == expected[case.name], case.name


def test_every_technique_lands_in_exactly_one_bucket():
    for case in ALL_CASES:
        selection = select_techniques(case)
        names = selection.binding + selection.available + selection.inert
        assert len(names) == len(set(names)) == 13


# --- Inertness, which is the chapter's contribution -------------------------


def test_all_of_chapter_11_is_inert_at_one_concurrent_request():
    """Batching with one request is not a small gain, it is none."""
    inert = select_techniques(WORKSTATION).inert
    assert any("Ch11" in name for name in inert)


def test_chapter_11_binds_the_moment_there_is_more_than_one_request():
    assert any("Ch11" in name for name in select_techniques(SUPPORT).binding)


def test_a_cascade_is_inert_where_there_is_no_larger_model_to_reach():
    """The chapter's opening, derivable from the profile before any work."""
    assert any("cascade" in name for name in select_techniques(GATEWAY).inert)
    assert any("cascade" in name for name in select_techniques(HANDHELD).inert)


def test_a_cascade_binds_on_a_server():
    assert any("cascade" in name for name in select_techniques(SUPPORT).binding)


def test_speculative_decoding_is_inert_at_high_batch_occupancy():
    """Ch10's forward reference to Ch11, as a predicate."""
    assert any("speculative" in n for n in select_techniques(BATCH).inert)
    assert any("speculative" in n for n in select_techniques(SUPPORT).inert)


def test_speculative_decoding_binds_on_a_single_stream_workstation():
    assert any(
        "speculative" in name for name in select_techniques(WORKSTATION).binding
    )


def test_fleet_techniques_are_inert_for_a_single_device():
    assert any("Ch13" in name for name in select_techniques(WORKSTATION).inert)


def test_fleet_techniques_bind_for_an_owned_fleet():
    for case in (GATEWAY, HANDHELD):
        assert any("Ch13" in name for name in select_techniques(case).binding)


# --- The available column ---------------------------------------------------


def test_the_model_techniques_are_available_but_not_binding_on_the_handheld():
    """Chapter 12's duty-cycle finding, expressed as a selection: they
    work, and they do not touch the constraint."""
    available = select_techniques(HANDHELD).available
    for chapter in ("Ch4", "Ch5", "Ch6"):
        assert any(chapter in name for name in available), chapter


def test_the_same_techniques_bind_once_the_duty_cycle_rises():
    """At a hundred responses an hour the cancelled project becomes the
    right project."""
    busy = DeploymentProfile(
        "busy handheld", "battery", 1, owns_hardware=True,
        battery_powered=True, inference_energy_share=0.60, fleet_size=4000,
    )
    binding = select_techniques(busy).binding
    for chapter in ("Ch4", "Ch5", "Ch6"):
        assert any(chapter in name for name in binding), chapter


def test_only_the_handheld_has_a_large_available_column():
    """Every other case's techniques either bind or cannot apply."""
    others = [SUPPORT, GATEWAY, BATCH]
    assert len(select_techniques(HANDHELD).available) == 4
    for case in others:
        assert len(select_techniques(case).available) == 0


# --- Cross-case invariants --------------------------------------------------


def test_the_harness_binds_in_every_single_case():
    """The least glamorous chapter, and the only one in every binding
    column."""
    for case in ALL_CASES:
        assert any("Ch2" in name for name in select_techniques(case).binding)


def test_the_footprint_estimate_binds_in_every_case():
    for case in ALL_CASES:
        assert any("Ch1" in name for name in select_techniques(case).binding)


def test_nothing_is_inert_everywhere():
    """A technique inert in all five cases would not belong in the book."""
    inert_everywhere = set.intersection(
        *(set(select_techniques(case).inert) for case in ALL_CASES)
    )
    assert inert_everywhere == set()


def test_the_gateway_engages_the_most_of_the_book():
    counts = {c.name: len(select_techniques(c).binding) for c in ALL_CASES}
    assert max(counts, key=counts.__getitem__) == "Industrial gateway"


def test_changing_one_field_moves_techniques_between_columns():
    """Exercise 2: a workstation that becomes a shared service."""
    shared = DeploymentProfile("shared service", "server", 40)
    before = select_techniques(WORKSTATION)
    after = select_techniques(shared)
    moved = set(before.inert) & set(after.binding)
    assert moved, "expected Ch11 to move from inert to binding"
