"""Unit and integration tests for the criterion-evidence engine (design doc 03 §8).

Contract-level tests: the five frozen review specs and the rule lexicon load and
validate, the four-state semantics behave exactly as specified (support cue hit,
negated cue is not support/contradiction, no cue -> UNKNOWN, support+contradict
-> MIXED, LOW-axis silence -> UNKNOWN), evidence sentences/offsets map back to
the original text, aggregation follows Q = 0.50 S + 0.30 U + 0.20 X, schema
validation rejects malformed configs, the engine never reads an annotation
column, and a smoke pass over real processed records works.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.core import (  # noqa: E402
    AUTOMATION_LEVELS,
    CriterionRecordEvidence,
    ReviewSpec,
)
from biolit_relay.criterion_evidence import (  # noqa: E402
    FRAME_COLUMNS,
    CriterionEvidenceEngine,
    concat_text,
    field_of_span,
    load_engines,
    load_lexicon,
    load_review_specs,
)

CONFIGS = ROOT / "configs"
PROCESSED = ROOT / "data" / "processed"
REVIEW_IDS = (
    "Jeyaraman_2020",
    "Meijboom_2021",
    "Muthu_2021",
    "van_der_Valk_2021",
    "Leenaars_2020",
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lexicon() -> dict:
    return load_lexicon(CONFIGS / "rule_lexicon.yaml")


@pytest.fixture(scope="module")
def real_specs() -> dict[str, ReviewSpec]:
    return load_review_specs(CONFIGS / "review_specs.yaml")


@pytest.fixture(scope="module")
def real_engines() -> dict[str, CriterionEvidenceEngine]:
    return load_engines(CONFIGS)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _minimal_review(**overrides: object) -> dict:
    review: dict = {
        "id": "Synthetic_2026",
        "topic": "Synthetic review used only by unit tests",
        "source_reference": "https://example.org/synthetic-review",
        "eligibility_text": (
            "Synthetic eligibility criteria: randomized controlled trials of "
            "mesenchymal stem-cell therapy in knee osteoarthritis were included."
        ),
        "rule_version": "v1.0.0",
        "criteria": [
            {
                "id": "population",
                "name": "Knee osteoarthritis",
                "required": True,
                "automation_level": "MEDIUM",
                "support_cues": ["knee osteoarthritis"],
                "contradict_cues": ["hip osteoarthritis"],
                "source_excerpt": "knee osteoarthritis",
            },
            {
                "id": "intervention",
                "name": "Mesenchymal stem-cell therapy",
                "required": True,
                "automation_level": "MEDIUM",
                "support_cues": ["mesenchymal stem cell"],
                "contradict_cues": ["no stem cell therapy"],
                "source_excerpt": "mesenchymal stem-cell therapy",
            },
            {
                "id": "design",
                "name": "Randomized controlled trial",
                "required": True,
                "automation_level": "LOW",
                "support_cues": ["randomized controlled trial"],
                "contradict_cues": [],
                "source_excerpt": "randomized controlled trials",
            },
            {
                "id": "full_text",
                "name": "Full text obtainable",
                "required": True,
                "automation_level": "LOW",
                "support_cues": [],
                "contradict_cues": [],
                "source_excerpt": "full-text availability is not stated in the abstract",
            },
        ],
    }
    review.update(overrides)
    return review


def _specs_data(reviews: list[dict]) -> dict:
    return {"schema_version": 1, "rule_version": "v1.0.0", "reviews": reviews}


@pytest.fixture()
def synthetic_spec(tmp_path: Path) -> ReviewSpec:
    path = _write_yaml(tmp_path / "review_specs.yaml", _specs_data([_minimal_review()]))
    return load_review_specs(path)["Synthetic_2026"]


@pytest.fixture()
def engine(synthetic_spec: ReviewSpec, lexicon: dict) -> CriterionEvidenceEngine:
    return CriterionEvidenceEngine(synthetic_spec, lexicon)


def _states(evidence: CriterionRecordEvidence) -> dict[str, str]:
    return {hit.criterion_id: hit.state for hit in evidence.hits}


def _hit(evidence: CriterionRecordEvidence, criterion_id: str):
    for hit in evidence.hits:
        if hit.criterion_id == criterion_id:
            return hit
    raise AssertionError(f"no hit for {criterion_id}")


# ---------------------------------------------------------------------------
# Frozen configuration loading
# ---------------------------------------------------------------------------


def test_real_specs_load_and_validate(real_specs: dict[str, ReviewSpec]) -> None:
    assert set(real_specs) == set(REVIEW_IDS)
    for review_id, spec in real_specs.items():
        assert spec.review_id == review_id
        assert spec.rule_version == "v1.0.0"
        assert re.fullmatch(r"[0-9a-f]{64}", spec.spec_hash)
        assert len(spec.eligibility_text) >= 100
        assert 2 <= len(spec.criteria) <= 6
        assert spec.required_criteria
        criterion_ids = [criterion.criterion_id for criterion in spec.criteria]
        assert len(criterion_ids) == len(set(criterion_ids))
        for criterion in spec.criteria:
            assert criterion.name
            assert criterion.automation_level in AUTOMATION_LEVELS
            assert criterion.source_excerpt
            for cue in (*criterion.support_cues, *criterion.contradict_cues):
                assert cue == cue.lower()
                assert cue.strip() == cue


def test_specs_use_verbatim_published_eligibility_text(
    real_specs: dict[str, ReviewSpec],
) -> None:
    expected_fragments = {
        "Jeyaraman_2020": "Population: Patients with knee osteoarthritis",
        "Meijboom_2021": "the number of patients who retransitioned was reported",
        "Muthu_2021": "The study must have a dichotomous primary or secondary outcome.",
        "van_der_Valk_2021": "cross-sectional associations between HairGC and measurements of obesity",
        "Leenaars_2020": "MTX was administered to RA patients or RA animal models",
    }
    for review_id, fragment in expected_fragments.items():
        assert fragment in real_specs[review_id].eligibility_text


def test_specs_mark_low_axes_that_are_unanswerable_from_abstracts(
    real_specs: dict[str, ReviewSpec],
) -> None:
    low_axes = {
        ("Jeyaraman_2020", "comparator"),
        ("Meijboom_2021", "retransition"),
        ("Meijboom_2021", "baseline_characteristics"),
        ("Meijboom_2021", "language"),
        ("Meijboom_2021", "full_text"),
        ("Muthu_2021", "perioperative_variable"),
        ("Muthu_2021", "binary_outcome"),
        ("Muthu_2021", "statistical_significance"),
        ("van_der_Valk_2021", "timing"),
        ("van_der_Valk_2021", "age_group"),
        ("van_der_Valk_2021", "language_peer_review"),
        ("Leenaars_2020", "in_vivo"),
        ("Leenaars_2020", "efficacy"),
    }
    actual_low = {
        (spec.review_id, criterion.criterion_id)
        for spec in real_specs.values()
        for criterion in spec.criteria
        if criterion.automation_level == "LOW"
    }
    assert low_axes <= actual_low
    # deliberately unanswerable axes have no cues at all
    for review_id, criterion_id in (
        ("Meijboom_2021", "language"),
        ("Meijboom_2021", "full_text"),
        ("Muthu_2021", "statistical_significance"),
        ("van_der_Valk_2021", "language_peer_review"),
    ):
        criterion = real_specs[review_id].criterion(criterion_id)
        assert not criterion.support_cues
        assert not criterion.contradict_cues


def test_real_lexicon_validates_frozen_design(lexicon: dict) -> None:
    assert lexicon["schema_version"] == 1
    assert lexicon["negation"]["window_tokens_before"] == 5
    assert "not only" in lexicon["negation"]["exceptions"]
    assert tuple(lexicon["states"]) == ("SUPPORT", "CONTRADICT", "UNKNOWN", "MIXED")
    scoring = lexicon["scoring"]
    assert scoring["support_weight"] == 0.50
    assert scoring["unknown_weight"] == 0.30
    assert scoring["mixed_weight"] == 0.20
    assert scoring["relevance_weight"] == 0.60
    assert scoring["false_negative_risk_weight"] == 0.40
    assert scoring["unknown_is_exclusion"] is False
    assert scoring["auto_exclude_records"] is False
    assert lexicon["change_control"]["main_experiment_rules_frozen_before_results"] is True


def test_all_five_engines_build(real_engines: dict[str, CriterionEvidenceEngine]) -> None:
    assert set(real_engines) == set(REVIEW_IDS)
    for review_id, engine in real_engines.items():
        assert engine.review_spec.review_id == review_id
        evidence = engine.evaluate("empty", "", "")
        assert isinstance(evidence, CriterionRecordEvidence)
        assert len(evidence.hits) == len(engine.review_spec.required_criteria)
        assert all(hit.state == "UNKNOWN" for hit in evidence.hits)
        assert evidence.s_ratio == 0.0
        assert evidence.u_ratio == 1.0
        assert evidence.x_ratio == 0.0


def test_frozen_configs_reference_no_annotation_fields() -> None:
    spec_text = (CONFIGS / "review_specs.yaml").read_text(encoding="utf-8")
    lexicon_text = (CONFIGS / "rule_lexicon.yaml").read_text(encoding="utf-8")
    for text in (spec_text, lexicon_text):
        assert "label_included" not in text
        assert "labels.csv" not in text
        assert "oracle" not in text.lower()


# ---------------------------------------------------------------------------
# Four-state semantics
# ---------------------------------------------------------------------------


def test_support_cue_hit_emits_support_with_auditable_evidence(
    engine: CriterionEvidenceEngine,
) -> None:
    title = "Mesenchymal Stem Cell Therapy for Knee Osteoarthritis"
    abstract = "A randomized controlled trial was performed."
    evidence = engine.evaluate("R1", title, abstract)
    states = _states(evidence)
    assert states["population"] == "SUPPORT"
    assert states["intervention"] == "SUPPORT"
    assert states["design"] == "SUPPORT"
    assert states["full_text"] == "UNKNOWN"

    population = _hit(evidence, "population")
    combined = concat_text(title, abstract)
    assert population.matched_phrase == "Knee Osteoarthritis"
    assert combined[population.start_char : population.end_char] == population.matched_phrase
    assert population.evidence_sentence == title
    assert population.cue_polarity == "support"
    assert re.fullmatch(r"Synthetic_2026:population:sup:\d+", population.rule_id)
    assert population.confidence_level == "medium"


def test_confirmed_contradict_cue_emits_contradict(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate("R1", "Hip osteoarthritis patients were included", "")
    population = _hit(evidence, "population")
    assert population.state == "CONTRADICT"
    assert population.cue_polarity == "contradict"
    assert population.matched_phrase.lower() == "hip osteoarthritis"
    assert re.fullmatch(r"Synthetic_2026:population:con:\d+", population.rule_id)


def test_negated_support_cue_is_not_support_and_not_contradict(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate(
        "R1",
        "Patients without knee osteoarthritis",
        "A randomized controlled trial was performed.",
    )
    population = _hit(evidence, "population")
    assert population.state == "UNKNOWN"
    assert population.matched_phrase == ""
    assert population.start_char == -1
    assert evidence.to_row()["n_contradict"] == 0


def test_negated_contradict_cue_is_not_contradict(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate(
        "R1",
        "Knee osteoarthritis patients",
        "No evidence of hip osteoarthritis was reported.",
    )
    population = _hit(evidence, "population")
    # contradiction cue is locally negated: no contradiction may be emitted and
    # the resolved support cue keeps the criterion at SUPPORT
    assert population.state == "SUPPORT"


def test_no_cue_is_unknown_with_empty_evidence(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate("R1", "A cooking recipe", "Boil water and add salt.")
    assert all(hit.state == "UNKNOWN" for hit in evidence.hits)
    for hit in evidence.hits:
        assert hit.matched_phrase == ""
        assert hit.evidence_sentence == ""
        assert hit.start_char == -1
        assert hit.end_char == -1
        assert hit.rule_id == ""
        assert hit.cue_polarity == "none"
    assert evidence.s_ratio == 0.0
    assert evidence.u_ratio == 1.0
    assert evidence.q_score == pytest.approx(0.30)


def test_support_and_contradict_cues_produce_mixed(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate(
        "R1",
        "Knee osteoarthritis and hip osteoarthritis patients",
        "Mesenchymal stem cell therapy was evaluated.",
    )
    population = _hit(evidence, "population")
    assert population.state == "MIXED"
    assert population.cue_polarity == "mixed"
    assert population.matched_phrase
    assert population.evidence_sentence
    assert population.confidence_level == "low"


def test_ambiguous_negation_scope_produces_mixed(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate(
        "R1",
        "patients did not receive placebo, knee osteoarthritis was assessed",
        "",
    )
    assert _states(evidence)["population"] == "MIXED"


def test_low_axis_silence_stays_unknown_but_explicit_evidence_is_low_confidence(
    engine: CriterionEvidenceEngine,
) -> None:
    silent = engine.evaluate(
        "R1", "Knee osteoarthritis patients", "Mesenchymal stem cell therapy was given."
    )
    assert _states(silent)["design"] == "UNKNOWN"
    assert _states(silent)["full_text"] == "UNKNOWN"

    explicit = engine.evaluate(
        "R1",
        "Knee osteoarthritis patients",
        "A randomized controlled trial was performed.",
    )
    design = _hit(explicit, "design")
    assert design.state == "SUPPORT"
    assert design.confidence_level == "low"  # LOW automation level
    # MEDIUM automation level keeps a higher confidence
    assert _hit(explicit, "population").confidence_level == "medium"
    # cue-less criterion can never become anything but UNKNOWN
    assert _hit(explicit, "full_text").state == "UNKNOWN"


def test_title_negation_does_not_leak_into_abstract(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate(
        "R1",
        "No knee osteoarthritis patients",
        "Mesenchymal stem cell therapy was administered.",
    )
    states = _states(evidence)
    assert states["population"] == "UNKNOWN"
    assert states["intervention"] == "SUPPORT"


# ---------------------------------------------------------------------------
# Aggregation and the offset contract
# ---------------------------------------------------------------------------


def test_aggregation_ratios_and_q_formula(engine: CriterionEvidenceEngine) -> None:
    evidence = engine.evaluate(
        "R1",
        "Knee osteoarthritis patients",
        "No stem cell therapy was given. No randomized controlled trial was performed.",
    )
    states = _states(evidence)
    assert states["population"] == "SUPPORT"
    assert states["intervention"] == "CONTRADICT"
    assert states["design"] == "UNKNOWN"
    assert states["full_text"] == "UNKNOWN"
    assert evidence.s_ratio == pytest.approx(0.25)
    assert evidence.u_ratio == pytest.approx(0.50)
    assert evidence.x_ratio == pytest.approx(0.0)
    expected_q = 0.50 * 0.25 + 0.30 * 0.50 + 0.20 * 0.0
    assert evidence.q_score == pytest.approx(expected_q)
    row = evidence.to_row()
    assert row["n_support"] == 1
    assert row["n_contradict"] == 1
    assert row["n_unknown"] == 2
    assert row["n_mixed"] == 0


def test_ratios_are_computed_over_required_criteria_only(
    tmp_path: Path, lexicon: dict
) -> None:
    review = _minimal_review()
    review["criteria"].append(
        {
            "id": "optional_axis",
            "name": "Optional axis not scored",
            "required": False,
            "automation_level": "LOW",
            "support_cues": ["biomarker"],
            "contradict_cues": [],
            "source_excerpt": "optional axis",
        }
    )
    spec = load_review_specs(
        _write_yaml(tmp_path / "review_specs.yaml", _specs_data([review]))
    )["Synthetic_2026"]
    engine = CriterionEvidenceEngine(spec, lexicon)
    evidence = engine.evaluate("R1", "Knee osteoarthritis biomarker study", "")
    assert len(evidence.hits) == 4  # only required criteria are emitted
    assert all(hit.criterion_id != "optional_axis" for hit in evidence.hits)
    assert evidence.s_ratio == pytest.approx(0.25)


def test_evidence_offsets_are_in_concatenated_original_text(
    engine: CriterionEvidenceEngine,
) -> None:
    title = "Mesenchymal Stem Cell Therapy for Knee Osteoarthritis"
    abstract = "A randomized controlled trial was conducted."
    evidence = engine.evaluate("R1", title, abstract)
    combined = concat_text(title, abstract)
    populated = _hit(evidence, "population")
    design = _hit(evidence, "design")
    assert field_of_span(title, populated.start_char, populated.end_char) == "title"
    assert field_of_span(title, design.start_char, design.end_char) == "abstract"
    assert design.start_char >= len(title) + 1
    for hit in evidence.hits:
        if hit.state == "UNKNOWN":
            continue
        assert combined[hit.start_char : hit.end_char] == hit.matched_phrase
        assert hit.evidence_sentence in combined
        assert hit.evidence_sentence.strip() == hit.evidence_sentence
        assert hit.end_char > hit.start_char
    # the implicit title/abstract boundary keeps the abstract sentence separate
    assert design.evidence_sentence == "A randomized controlled trial was conducted."


def test_concat_text_and_field_of_span_helpers() -> None:
    assert concat_text("Title", "Abstract") == "Title Abstract"
    assert concat_text("Title", "") == "Title"
    assert concat_text("", "Abstract") == "Abstract"
    assert concat_text("", "") == ""
    assert field_of_span("Title", 0, 5) == "title"
    assert field_of_span("Title", 6, 14) == "abstract"
    assert field_of_span("", 0, 3) == "abstract"
    assert field_of_span("Title", 2, 8) == "cross-boundary"


def test_non_unknown_hits_always_carry_the_full_audit_trail(
    engine: CriterionEvidenceEngine,
) -> None:
    evidence = engine.evaluate(
        "R1",
        "Knee osteoarthritis and hip osteoarthritis patients",
        "No stem cell therapy was used; a randomized controlled trial was planned.",
    )
    for hit in evidence.hits:
        if hit.state == "UNKNOWN":
            assert hit.matched_phrase == ""
            assert hit.evidence_sentence == ""
            assert hit.start_char == -1
            assert hit.end_char == -1
            assert hit.rule_id == ""
        else:
            assert hit.matched_phrase
            assert hit.evidence_sentence
            assert hit.start_char >= 0
            assert hit.end_char > hit.start_char
            assert hit.rule_id
            assert hit.confidence_level in ("high", "medium", "low")
            assert hit.cue_polarity in ("support", "contradict", "mixed", "none")


# ---------------------------------------------------------------------------
# evaluate_frame
# ---------------------------------------------------------------------------


def test_evaluate_frame_contract(engine: CriterionEvidenceEngine) -> None:
    records = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3"],
            "title": [
                "Mesenchymal stem cell therapy for knee osteoarthritis",
                "Hip osteoarthritis cohort",
                None,
            ],
            "abstract": [
                "A randomized controlled trial was performed.",
                "Mesenchymal stem cell therapy was discussed.",
                float("nan"),
            ],
            "has_title": [1, 1, 0],
            "has_abstract": [1, 1, 0],
            "doi": ["", "", ""],
            "source_url": ["", "", ""],
            "publication_year": [2020, 2021, 2022],
            "type": ["journal-article", "journal-article", "journal-article"],
        }
    )
    frame, details = engine.evaluate_frame(records)
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert frame["record_id"].tolist() == ["R1", "R2", "R3"]
    assert len(details) == 3
    assert frame["q_score"].between(0.0, 1.0).all()
    # CONTRADICT carries no positive Q term: s + u + x == 1 - contradict_ratio
    totals = frame["s_ratio"] + frame["u_ratio"] + frame["x_ratio"]
    assert totals.le(1.0 + 1e-9).all()
    n_required = len(engine.review_spec.required_criteria)
    contradict_share = frame["n_contradict"] / n_required
    assert (totals + contradict_share).round(9).eq(1.0).all()
    assert frame[["n_support", "n_contradict", "n_unknown", "n_mixed"]].sum(axis=1).eq(
        n_required
    ).all()
    # row values equal the single-record API
    for record_id, raw_title, raw_abstract in zip(
        records["record_id"], records["title"], records["abstract"]
    ):
        expected = engine.evaluate(record_id, raw_title, raw_abstract)
        cached = details[record_id]
        assert cached.q_score == pytest.approx(expected.q_score)
        assert cached.hits == expected.hits
    # NaN title/abstract is treated as empty text, never as an error
    assert details["R3"].u_ratio == 1.0
    assert details["R3"].q_score == pytest.approx(0.30)


def test_evaluate_frame_validates_input(engine: CriterionEvidenceEngine) -> None:
    with pytest.raises(TypeError):
        engine.evaluate_frame([{"record_id": "R1"}])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.evaluate_frame(pd.DataFrame({"record_id": ["R1"], "title": ["t"]}))
    with pytest.raises(ValueError):
        engine.evaluate_frame(
            pd.DataFrame({"record_id": [None], "title": ["t"], "abstract": ["a"]})
        )


def test_engine_is_annotation_blind(engine: CriterionEvidenceEngine) -> None:
    base = pd.DataFrame(
        {
            "record_id": ["R1", "R2"],
            "title": ["Knee osteoarthritis trial", "Unrelated title"],
            "abstract": ["Mesenchymal stem cell therapy.", "Nothing relevant here."],
        }
    )
    with_extra = base.copy()
    with_extra["label_included"] = [1, 0]  # must be ignored, never read
    frame_a, details_a = engine.evaluate_frame(base)
    frame_b, details_b = engine.evaluate_frame(with_extra)
    pd.testing.assert_frame_equal(frame_a, frame_b)
    for record_id in details_a:
        assert details_a[record_id].hits == details_b[record_id].hits


def test_modules_never_reference_annotation_fields() -> None:
    forbidden = re.compile(
        r"label_included|labels\.csv|gold_label|\blabels?\b|openalex_id.*label",
        re.IGNORECASE,
    )
    for module_name in ("text_normalizer.py", "criterion_evidence.py"):
        source = (ROOT / "src" / "biolit_relay" / module_name).read_text(encoding="utf-8")
        assert not forbidden.search(source), module_name


# ---------------------------------------------------------------------------
# Engine construction guards
# ---------------------------------------------------------------------------


def test_engine_rejects_invalid_construction(
    synthetic_spec: ReviewSpec, lexicon: dict
) -> None:
    with pytest.raises(TypeError):
        CriterionEvidenceEngine("not-a-spec", lexicon)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CriterionEvidenceEngine(synthetic_spec, ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CriterionEvidenceEngine(synthetic_spec, lexicon, scoring_weights=(0.5, 0.3))
    with pytest.raises(ValueError):
        CriterionEvidenceEngine(synthetic_spec, lexicon, scoring_weights=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        CriterionEvidenceEngine(synthetic_spec, lexicon, max_cached_texts=-1)


def test_custom_scoring_weights_are_respected(
    synthetic_spec: ReviewSpec, lexicon: dict
) -> None:
    default_engine = CriterionEvidenceEngine(synthetic_spec, lexicon)
    custom_engine = CriterionEvidenceEngine(
        synthetic_spec, lexicon, scoring_weights=(0.40, 0.40, 0.20)
    )
    assert custom_engine.scoring_weights == (0.40, 0.40, 0.20)
    empty = custom_engine.evaluate("R1", "", "")
    assert empty.q_score == pytest.approx(0.40)  # all UNKNOWN -> unknown weight
    # the default tuple resolves to the frozen lexicon weights
    assert default_engine.scoring_weights == (0.50, 0.30, 0.20)


# ---------------------------------------------------------------------------
# Schema validation errors: review specs
# ---------------------------------------------------------------------------


def _uppercase_cue(data: dict) -> None:
    data["reviews"][0]["criteria"][0]["support_cues"] = ["Knee Osteoarthritis"]


def _too_few_criteria(data: dict) -> None:
    data["reviews"][0]["criteria"] = data["reviews"][0]["criteria"][:1]


def _too_many_criteria(data: dict) -> None:
    criterion = copy.deepcopy(data["reviews"][0]["criteria"][0])
    criterion["id"] = "extra_axis"
    data["reviews"][0]["criteria"].extend([copy.deepcopy(criterion) for _ in range(3)])


def _duplicate_criterion_id(data: dict) -> None:
    data["reviews"][0]["criteria"][1]["id"] = "population"


def _bad_automation_level(data: dict) -> None:
    data["reviews"][0]["criteria"][0]["automation_level"] = "EXTREME"


def _no_required_criteria(data: dict) -> None:
    for criterion in data["reviews"][0]["criteria"]:
        criterion["required"] = False


def _overlapping_polarities(data: dict) -> None:
    data["reviews"][0]["criteria"][0]["contradict_cues"] = ["knee osteoarthritis"]


def _bad_rule_version(data: dict) -> None:
    data["reviews"][0]["rule_version"] = "1.0"


def _short_eligibility_text(data: dict) -> None:
    data["reviews"][0]["eligibility_text"] = "too short"


def _missing_topic(data: dict) -> None:
    del data["reviews"][0]["topic"]


def _duplicate_review_ids(data: dict) -> None:
    data["reviews"].append(copy.deepcopy(data["reviews"][0]))


def _spec_hash_mismatch(data: dict) -> None:
    data["reviews"][0]["spec_hash"] = "deadbeef"


@pytest.mark.parametrize(
    "mutator",
    [
        _uppercase_cue,
        _too_few_criteria,
        _too_many_criteria,
        _duplicate_criterion_id,
        _bad_automation_level,
        _no_required_criteria,
        _overlapping_polarities,
        _bad_rule_version,
        _short_eligibility_text,
        _missing_topic,
        _duplicate_review_ids,
        _spec_hash_mismatch,
    ],
    ids=lambda mutator: mutator.__name__,
)
def test_review_spec_validation_rejects_malformed_configs(
    tmp_path: Path, mutator
) -> None:
    data = _specs_data([_minimal_review()])
    mutator(data)
    path = _write_yaml(tmp_path / "review_specs.yaml", data)
    with pytest.raises(ValueError):
        load_review_specs(path)


def test_review_spec_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_review_specs(tmp_path / "does-not-exist.yaml")


# ---------------------------------------------------------------------------
# Schema validation errors: rule lexicon
# ---------------------------------------------------------------------------


def _mutated_lexicon(tmp_path: Path, mutator) -> Path:
    data = copy.deepcopy(load_lexicon(CONFIGS / "rule_lexicon.yaml"))
    mutator(data)
    return _write_yaml(tmp_path / "rule_lexicon.yaml", data)


def _weights_do_not_sum(data: dict) -> None:
    data["scoring"]["support_weight"] = 0.60


def _unknown_is_exclusion(data: dict) -> None:
    data["scoring"]["unknown_is_exclusion"] = True


def _auto_exclude(data: dict) -> None:
    data["scoring"]["auto_exclude_records"] = True


def _empty_negation_cues(data: dict) -> None:
    data["negation"]["cues"] = []


def _uppercase_negation_cue(data: dict) -> None:
    data["negation"]["cues"] = ["NOT"]


def _disable_lowercase(data: dict) -> None:
    data["normalization"]["lowercase"] = False


def _wrong_states(data: dict) -> None:
    data["states"] = ["SUPPORT", "CONTRADICT", "UNKNOWN"]


def _no_sentence_evidence(data: dict) -> None:
    data["evidence"]["store_sentence"] = False


def _disable_change_control(data: dict) -> None:
    data["change_control"]["rule_version_required"] = False


def _missing_scoring(data: dict) -> None:
    del data["scoring"]


@pytest.mark.parametrize(
    "mutator",
    [
        _weights_do_not_sum,
        _unknown_is_exclusion,
        _auto_exclude,
        _empty_negation_cues,
        _uppercase_negation_cue,
        _disable_lowercase,
        _wrong_states,
        _no_sentence_evidence,
        _disable_change_control,
        _missing_scoring,
    ],
    ids=lambda mutator: mutator.__name__,
)
def test_lexicon_validation_rejects_malformed_configs(tmp_path: Path, mutator) -> None:
    path = _mutated_lexicon(tmp_path, mutator)
    with pytest.raises(ValueError):
        load_lexicon(path)


def test_lexicon_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lexicon(tmp_path / "does-not-exist.yaml")


# ---------------------------------------------------------------------------
# Real-data smoke tests (skip gracefully while processed data is being built)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_records_sample() -> dict[str, pd.DataFrame]:
    samples: dict[str, pd.DataFrame] = {}
    for review_id in REVIEW_IDS:
        path = PROCESSED / f"{review_id}.csv"
        if not path.exists():
            continue
        samples[review_id] = pd.read_csv(path, nrows=80)
    return samples


def test_processed_schema_matches_engine_contract(real_records_sample) -> None:
    if not real_records_sample:
        pytest.skip("processed data not built yet")
    expected = {
        "record_id",
        "title",
        "abstract",
        "has_title",
        "has_abstract",
        "doi",
        "source_url",
        "publication_year",
        "type",
    }
    for review_id, records in real_records_sample.items():
        assert expected <= set(records.columns), review_id
        assert len(records) >= 50, review_id


@pytest.mark.parametrize("review_id", REVIEW_IDS)
def test_smoke_engine_on_real_processed_records(
    real_engines, real_records_sample, review_id: str
) -> None:
    if review_id not in real_records_sample:
        pytest.skip(f"processed data for {review_id} not built yet")
    records = real_records_sample[review_id]
    engine = real_engines[review_id]
    frame, details = engine.evaluate_frame(records)
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert len(frame) == len(records) == len(details)
    assert frame["record_id"].tolist() == records["record_id"].astype(str).tolist()
    assert frame["q_score"].between(0.0, 1.0).all()
    assert (frame["s_ratio"] + frame["u_ratio"] + frame["x_ratio"]).le(1.0 + 1e-9).all()
    n_required = len(engine.review_spec.required_criteria)
    assert (
        frame[["n_support", "n_contradict", "n_unknown", "n_mixed"]]
        .sum(axis=1)
        .eq(n_required)
        .all()
    )
    for evidence in details.values():
        for hit in evidence.hits:
            if hit.state == "UNKNOWN":
                continue
            assert hit.matched_phrase
            assert hit.evidence_sentence
            assert hit.start_char >= 0
            assert hit.end_char > hit.start_char


def test_smoke_real_records_produce_some_lexical_evidence(real_engines, real_records_sample) -> None:
    if not real_records_sample:
        pytest.skip("processed data not built yet")
    total_support = 0
    total_contradict = 0
    total_mixed = 0
    for review_id, records in real_records_sample.items():
        frame, _ = real_engines[review_id].evaluate_frame(records)
        total_support += int(frame["n_support"].sum())
        total_contradict += int(frame["n_contradict"].sum())
        total_mixed += int(frame["n_mixed"].sum())
    # sanity: the frozen lexicon must fire on real abstracts, not stay silent
    assert total_support + total_contradict + total_mixed > 0


def test_evidence_store_columns_if_present() -> None:
    path = PROCESSED / "evidence_store.csv"
    if not path.exists():
        pytest.skip("evidence store not built yet")
    columns = pd.read_csv(path, nrows=1).columns
    assert "review_id" in columns
    assert set(FRAME_COLUMNS) <= set(columns)

