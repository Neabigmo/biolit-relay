"""Core data contracts shared by all BioLit-Relay modules.

These dataclasses define the *frozen* interface between agents.  They contain no
gold labels: the simulation oracle keeps labels in a separate store, and every
object that is passed to the ranker, the evidence engine or the router is
label-free by construction.

The public interfaces used by the simulation pipeline are defined here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Review specification (human-curated eligibility criteria, versioned)
# ---------------------------------------------------------------------------

AUTOMATION_LEVELS = ("HIGH", "MEDIUM", "LOW")


@dataclass(frozen=True)
class CriterionSpec:
    """One operationalised eligibility-criterion axis.

    Parameters
    ----------
    criterion_id:
        Stable identifier, unique inside a review (e.g. ``population``).
    name:
        Human-readable name.
    required:
        Whether the criterion is treated as necessary in the risk score.
    automation_level:
        HIGH / MEDIUM / LOW.  LOW means "the abstract usually cannot answer
        this"; the engine still reports explicit lexical evidence but never
        infers a contradiction from silence.
    support_cues:
        Lowercase phrases that are direct positive evidence for the criterion.
    contradict_cues:
        Lowercase phrases that are direct negative evidence for the criterion.
    notes:
        Curation note shown in the evidence card.
    source_excerpt:
        Verbatim fragment of the published eligibility criteria.
    """

    criterion_id: str
    name: str
    required: bool
    automation_level: str
    support_cues: tuple[str, ...] = ()
    contradict_cues: tuple[str, ...] = ()
    notes: str = ""
    source_excerpt: str = ""

    def __post_init__(self) -> None:
        if self.automation_level not in AUTOMATION_LEVELS:
            raise ValueError(f"bad automation_level: {self.automation_level!r}")
        if not self.criterion_id:
            raise ValueError("criterion_id must be non-empty")
        for cue in (*self.support_cues, *self.contradict_cues):
            if cue != cue.lower():
                raise ValueError(f"cues must be lowercase: {cue!r}")


@dataclass(frozen=True)
class ReviewSpec:
    """Versioned, human-curated operationalisation of one review's criteria."""

    review_id: str
    topic: str
    source_reference: str
    eligibility_text: str
    criteria: tuple[CriterionSpec, ...]
    rule_version: str
    spec_hash: str = ""

    @property
    def required_criteria(self) -> tuple[CriterionSpec, ...]:
        return tuple(c for c in self.criteria if c.required)

    def criterion(self, criterion_id: str) -> CriterionSpec:
        for c in self.criteria:
            if c.criterion_id == criterion_id:
                return c
        raise KeyError(f"{self.review_id}: unknown criterion {criterion_id!r}")


# ---------------------------------------------------------------------------
# Text normalisation contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # char offset in normalised text (inclusive)
    end: int  # char offset in normalised text (exclusive)


@dataclass(frozen=True)
class NormalizedText:
    """Normalised text plus a char-index map back to the original string.

    ``orig_index[j]`` is the index in the original text of normalised char
    ``j``.  It always has ``len(orig_index) == len(text)`` so that evidence
    spans can be mapped back for auditability.
    """

    text: str
    original: str
    orig_index: tuple[int, ...] = field(default_factory=tuple)

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        if not self.orig_index:
            return start, end
        start = max(0, min(start, len(self.orig_index) - 1))
        end = max(start + 1, min(end, len(self.orig_index)))
        return self.orig_index[start], self.orig_index[end - 1] + 1


# ---------------------------------------------------------------------------
# Criterion-level evidence contract
# ---------------------------------------------------------------------------

STATES = ("SUPPORT", "CONTRADICT", "UNKNOWN", "MIXED")


@dataclass(frozen=True)
class CriterionHit:
    """Auditable evidence for one (record, criterion) pair."""

    criterion_id: str
    state: str
    matched_phrase: str = ""
    evidence_sentence: str = ""
    start_char: int = -1
    end_char: int = -1
    rule_id: str = ""
    confidence_level: str = "low"  # high / medium / low
    cue_polarity: str = ""  # support / contradict / mixed / none

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"bad state: {self.state!r}")


@dataclass(frozen=True)
class CriterionRecordEvidence:
    """All criterion hits for one record plus the aggregate risk terms.

    ``s_ratio`` = share of *required* criteria with SUPPORT,
    ``u_ratio`` = share with UNKNOWN, ``x_ratio`` = share with MIXED;
    ``q_score`` = clip(0.50*S + 0.30*U + 0.20*X, 0, 1).
    """

    record_id: str
    hits: tuple[CriterionHit, ...]
    s_ratio: float
    u_ratio: float
    x_ratio: float
    q_score: float

    def state_of(self, criterion_id: str) -> str:
        for h in self.hits:
            if h.criterion_id == criterion_id:
                return h.state
        return "UNKNOWN"

    def to_row(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "s_ratio": self.s_ratio,
            "u_ratio": self.u_ratio,
            "x_ratio": self.x_ratio,
            "q_score": self.q_score,
            "n_support": sum(1 for h in self.hits if h.state == "SUPPORT"),
            "n_contradict": sum(1 for h in self.hits if h.state == "CONTRADICT"),
            "n_unknown": sum(1 for h in self.hits if h.state == "UNKNOWN"),
            "n_mixed": sum(1 for h in self.hits if h.state == "MIXED"),
        }


# ---------------------------------------------------------------------------
# Routed record / simulation event contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutedRecord:
    record_id: str
    p_include: float
    relevance_percentile: float
    criterion_risk: float
    false_negative_risk: float
    final_score: float
    reason_codes: tuple[str, ...] = ()
    evidence: CriterionRecordEvidence | None = None


@dataclass(frozen=True)
class SimulationEvent:
    """One revealed query batch.  ``revealed_labels`` never feed back into the
    ranking that produced ``selected_record_ids``."""

    run_id: str
    review_id: str
    method: str
    seed: int
    iteration: int
    selected_record_ids: tuple[str, ...]
    revealed_labels: tuple[int, ...]
    checked: int  # includes the initial priors
    found: int
    budget_fraction: float
    recall: float
    timestamp_utc: str = ""


@dataclass(frozen=True)
class ExperimentProtocol:
    """Frozen simulation protocol (mirrors configs/experiment.yaml)."""

    prior_included: int = 5
    prior_excluded: int = 5
    query_batch_size: int = 10
    budget_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    max_budget_fraction: float = 0.30
    main_seeds: int = 10
    random_baseline_seeds: int = 20
    relevance_weight: float = 0.60
    false_negative_risk_weight: float = 0.40
    criterion_support_weight: float = 0.50
    criterion_unknown_weight: float = 0.30
    criterion_mixed_weight: float = 0.20
    unknown_is_exclusion: bool = False
    auto_exclude_records: bool = False
    methods: tuple[str, ...] = (
        "random",
        "matched_tfidf_logreg",
        "uncertainty_only",
        "criterion_only",
        "asreview_default",
        "biolit_relay",
    )

    def __post_init__(self) -> None:
        if abs(self.relevance_weight + self.false_negative_risk_weight - 1.0) > 1e-9:
            raise ValueError("relevance + false-negative-risk weights must sum to 1")
        if abs(
            self.criterion_support_weight
            + self.criterion_unknown_weight
            + self.criterion_mixed_weight
            - 1.0
        ) > 1e-9:
            raise ValueError("criterion weights must sum to 1")
