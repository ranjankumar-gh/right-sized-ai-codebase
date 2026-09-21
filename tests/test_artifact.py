"""Chapter 7: format names, runtime support, and artifact lineage."""

from datetime import date

from rightsize.artifact import (
    VLLM,
    ArtifactLineage,
    RuntimeSupport,
    deployment_blockers,
    parse_gguf_type,
)

TODAY = date(2026, 10, 13)  # the day VLLM's matrix was checked

# Every quantization type llama.cpp's tools/quantize README documents.
DOCUMENTED_TYPES = [
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL",
    "Q2_K_S", "Q2_K",
    "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_XS",
    "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0", "F16",
]


# --- Pass 1: the name parser ------------------------------------------------


def test_every_documented_type_parses():
    """If llama.cpp documents it, the parser classifies it."""
    unparsed = [name for name in DOCUMENTED_TYPES if parse_gguf_type(name) is None]
    assert unparsed == []


def test_an_unknown_type_returns_none_rather_than_raising():
    """New types arrive with llama.cpp releases. A tool that crashes on an
    unfamiliar name is worse than one that says it cannot classify it."""
    assert parse_gguf_type("Q9_ZZZ") is None
    assert parse_gguf_type("") is None
    assert parse_gguf_type("not-a-type") is None


def test_the_bit_width_is_read_from_the_name():
    assert parse_gguf_type("Q4_K_M").bits == 4
    assert parse_gguf_type("Q8_0").bits == 8
    assert parse_gguf_type("IQ2_XXS").bits == 2
    assert parse_gguf_type("F16").bits == 16


def test_k_quants_are_mixed_precision_and_legacy_formats_are_not():
    """The chapter's central point: only the legacy formats behave the way
    Chapter 6's uniform arithmetic predicts."""
    assert parse_gguf_type("Q4_K_M").mixed_precision
    assert parse_gguf_type("Q6_K").mixed_precision
    assert not parse_gguf_type("Q4_0").mixed_precision
    assert not parse_gguf_type("Q8_0").mixed_precision


def test_only_legacy_and_float_types_are_uniform():
    uniform = {t for t in DOCUMENTED_TYPES if parse_gguf_type(t).is_uniform}
    assert uniform == {"Q8_0", "F16"}


def test_iq_types_are_flagged_as_needing_calibration():
    assert parse_gguf_type("IQ4_XS").needs_imatrix
    assert parse_gguf_type("IQ2_M").needs_imatrix
    assert not parse_gguf_type("Q4_K_M").needs_imatrix
    assert not parse_gguf_type("Q4_0").needs_imatrix


def test_bit_width_alone_does_not_order_two_formats():
    """Q5_0 is nominally above Q4_K_M and is a uniform format against a
    designed mixture, so the numbers are not comparable as a ranking."""
    higher_bits = parse_gguf_type("Q5_K_M")
    lower_bits = parse_gguf_type("Q4_K_M")
    assert higher_bits.bits > lower_bits.bits
    assert higher_bits.mixed_precision and lower_bits.mixed_precision

    uniform_five = parse_gguf_type("Q5_0")
    designed_four = parse_gguf_type("Q4_K_M")
    assert uniform_five.bits > designed_four.bits
    assert uniform_five.is_uniform != designed_four.is_uniform


def test_parsing_is_case_and_whitespace_insensitive():
    assert parse_gguf_type("  q4_k_m  ") == parse_gguf_type("Q4_K_M")


# --- Pass 2 and 3: the dated support matrix ---------------------------------


def test_a_supported_format_with_no_caveat_has_no_blockers():
    assert deployment_blockers("awq", VLLM, TODAY) == []
    assert deployment_blockers("gptq", VLLM, TODAY) == []


def test_an_unsupported_format_is_blocked_and_the_alternatives_are_named():
    blockers = deployment_blockers("exl2", VLLM, TODAY)
    assert len(blockers) == 1
    assert "does not load exl2" in blockers[0]
    assert "awq" in blockers[0]


def test_a_caveat_is_surfaced_rather_than_buried():
    """'FP8 supported' alongside 'INT8 KV cache unconfirmed' is how a
    deployment gets mis-sized on the term Chapter 6 showed is larger."""
    blockers = deployment_blockers("fp8", VLLM, TODAY)
    assert len(blockers) == 1
    assert "INT8 KV cache support is unconfirmed" in blockers[0]


def test_the_gguf_caveat_warns_about_kernel_coverage():
    blockers = deployment_blockers("gguf", VLLM, TODAY)
    assert any("optimized kernels" in b for b in blockers)


def test_neutral_facts_are_not_recorded_as_caveats():
    """Kernel defaults are worth knowing and are not qualifications. Mixing
    them in would train a reader to skim the entries that matter."""
    assert "gptq" not in VLLM.caveats
    assert "awq" not in VLLM.caveats
    assert set(VLLM.caveats) <= VLLM.formats


def test_a_matrix_goes_stale_and_says_so():
    a_year_later = date(2027, 10, 13)
    blockers = deployment_blockers("awq", VLLM, a_year_later)
    assert any("last verified" in b for b in blockers)


def test_a_fresh_matrix_is_not_stale():
    """Default window is 180 days: 2027-01-01 is 80 days out and fresh,
    2027-06-01 is 231 days out and is not."""
    assert not VLLM.is_stale(TODAY)
    assert not VLLM.is_stale(date(2027, 1, 1))
    assert VLLM.is_stale(date(2027, 6, 1))


def test_staleness_and_an_unsupported_format_are_both_reported():
    blockers = deployment_blockers("exl2", VLLM, date(2027, 10, 13))
    assert len(blockers) == 2


def test_a_runtime_with_no_caveats_still_works():
    bare = RuntimeSupport("bare", TODAY, frozenset({"gguf"}))
    assert deployment_blockers("gguf", bare, TODAY) == []


# --- Pass 4: artifact lineage -----------------------------------------------


def _lineage(**overrides) -> ArtifactLineage:
    base = dict(
        artifact="model-Q4_K_M.gguf",
        source_model="HuggingFaceTB/SmolLM3-3B",
        source_precision_bits=16,
        format_name="Q4_K_M",
        method="k-quant",
        tool="llama.cpp",
        tool_version="b4200",
        calibration_set=None,
        produced_on=TODAY,
    )
    return ArtifactLineage(**{**base, **overrides})


def test_a_complete_lineage_from_full_precision_has_no_warnings():
    assert _lineage().warnings() == []


def test_a_requantized_artifact_is_detected():
    """The accident the tooling permits: quantizing a model that a hub
    already quantized, which quantizes the error rather than the weights."""
    twice = _lineage(source_precision_bits=4, format_name="Q3_K_M")
    assert twice.was_requantized
    assert any("requantized" in w for w in twice.warnings())


def test_full_precision_sources_are_not_flagged():
    assert not _lineage(source_precision_bits=16).was_requantized


def test_an_imatrix_format_without_a_calibration_set_is_flagged():
    """IQ formats bake calibration into the weights, so an unrecorded
    calibration set makes the artifact irreproducible."""
    iq = _lineage(format_name="IQ4_XS", calibration_set=None)
    assert any("calibration set" in w for w in iq.warnings())


def test_an_imatrix_format_with_a_calibration_set_is_clean():
    iq = _lineage(format_name="IQ4_XS", calibration_set="stratified-v3")
    assert iq.warnings() == []


def test_a_k_quant_without_calibration_is_not_flagged():
    """K-quants do not consume an importance matrix, so the absence of a
    calibration set is not a defect for them."""
    assert _lineage(format_name="Q4_K_M", calibration_set=None).warnings() == []


def test_a_missing_tool_version_is_flagged_as_irreproducible():
    assert any("tool version" in w for w in _lineage(tool_version="").warnings())


def test_every_lineage_problem_is_reported_at_once():
    broken = _lineage(
        source_precision_bits=4, format_name="IQ3_M",
        calibration_set=None, tool_version="",
    )
    assert len(broken.warnings()) == 3


def test_the_requantization_warning_is_reported_first():
    """Worst first: requantization is the one that cannot be undone."""
    broken = _lineage(source_precision_bits=4, tool_version="")
    assert "requantized" in broken.warnings()[0]
