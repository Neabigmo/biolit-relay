"""Deterministic, annotation-free criterion-evidence extraction.

For every necessary eligibility criterion of a review the engine searches the
normalised title+abstract for human-curated *support* and *contradict* cue
phrases, checks local negation around each hit and emits one auditable
:class:`~biolit_relay.core.CriterionHit` per criterion with one of four states:

``SUPPORT``
    At least one non-negated support cue and no non-negated contradict cue.
``CONTRADICT``
    At least one non-negated contradict cue and no non-negated support cue.
``MIXED``
    Both resolved polarities occur, or the only hits are negation-scope
    ambiguities that the local window rule cannot resolve.
``UNKNOWN``
    No cue matched, or only *negated* cues matched.  A negated support cue is
    never converted into ``CONTRADICT`` (design pack doc 03, sections 4-5).

The class-level risk terms follow the frozen design:

    S_i = share of required criteria with SUPPORT
    U_i = share of required criteria with UNKNOWN
    X_i = share of required criteria with MIXED
    Q_i = clip(0.50*S_i + 0.30*U_i + 0.20*X_i, 0, 1)

``UNKNOWN`` deliberately carries a non-zero risk term so that "the abstract
does not say" is never treated as "does not meet the criterion".  ``CONTRADICT``
adds no support term but is always kept on the evidence card.

Text contract
-------------
Title and abstract are concatenated as ``title + " " + abstract`` (the empty
side is omitted) and normalised as one string, exactly as specified in design
doc 03, section 3.  ``CriterionHit.start_char``/``end_char`` are therefore
0-based offsets into that *concatenated original* string and
``evidence_sentence`` is the verbatim original sentence around the hit.  A
consumer that needs the field can compare ``start_char`` with
``len(title) + 1``; :func:`field_of_span` is provided for convenience.

Because titles normally have no terminal period, the engine treats the single
separator space as an implicit sentence boundary: a title negation never leaks
into the first abstract words and title/abstract evidence sentences stay
separate.  This is a deterministic text-processing rule, not a change to the
original text or to the character offsets.

Annotation blindness: this module never receives, reads or infers any benchmark
annotation.  Its only inputs are the frozen review specification, the frozen
rule lexicon and the model-visible title/abstract text.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .core import (
    AUTOMATION_LEVELS,
    CriterionHit,
    CriterionRecordEvidence,
    CriterionSpec,
    NormalizedText,
    ReviewSpec,
    STATES,
    Token,
)
from .text_normalizer import (
    NEGATION_CUES,
    NEGATION_EXCEPTIONS,
    is_negated,
    normalize,
    sentence_at,
    split_sentences,
    tokenize,
)

__all__ = [
    "CONCAT_SEPARATOR",
    "DEFAULT_SCORING_WEIGHTS",
    "FRAME_COLUMNS",
    "CriterionEvidenceEngine",
    "concat_text",
    "field_of_span",
    "load_engines",
    "load_lexicon",
    "load_review_specs",
]

#: Separator used when title and abstract are joined for matching.
CONCAT_SEPARATOR = " "

#: Frozen criterion weights (support, unknown, mixed); must sum to 1.
DEFAULT_SCORING_WEIGHTS: tuple[float, float, float] = (0.50, 0.30, 0.20)

#: Output columns of :meth:`CriterionEvidenceEngine.evaluate_frame`.
FRAME_COLUMNS: tuple[str, ...] = (
    "record_id",
    "s_ratio",
    "u_ratio",
    "x_ratio",
    "q_score",
    "n_support",
    "n_contradict",
    "n_unknown",
    "n_mixed",
)

_CONFIDENCE_BY_AUTOMATION: dict[str, str] = {
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low",
}

_POLARITY_SHORT: dict[str, str] = {"support": "sup", "contradict": "con"}

_RULE_VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
_WHITESPACE_RUN = re.compile(r"\s{2,}")
_WEIGHT_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def concat_text(title: str, abstract: str) -> str:
    """Return the exact string the engine matches against.

    ``title + " " + abstract`` when both sides are non-empty, otherwise the
    non-empty side alone.  This function defines the offset frame used by
    :class:`~biolit_relay.core.CriterionHit`.
    """
    title = title if isinstance(title, str) else ""
    abstract = abstract if isinstance(abstract, str) else ""
    if title and abstract:
        return title + CONCAT_SEPARATOR + abstract
    return title or abstract


def field_of_span(title: str, start: int, end: int) -> str:
    """Return ``"title"``, ``"abstract"`` or ``"cross-boundary"`` for a span.

    ``start``/``end`` must be offsets in the concatenated original string
    produced by :func:`concat_text` (the frame used by the evidence hits).
    """
    title = title if isinstance(title, str) else ""
    if not title:
        return "abstract"
    if end <= len(title):
        return "title"
    if start > len(title):
        return "abstract"
    return "cross-boundary"


def _title_abstract_boundary(
    title: str, abstract: str, norm: NormalizedText
) -> int | None:
    """Normalised offset of the implicit title/abstract sentence boundary.

    Titles usually have no terminal period.  Without an explicit boundary the
    first abstract sentence would be glued to the title, which would let a
    title negation reach into the first abstract words.  The boundary is the
    normalised index of the single separator space; ``None`` when one side is
    empty or when the normalised title is not a prefix of the combined text.
    """
    if not title or not abstract:
        return None
    normalised_title = normalize(title).text
    title_length = len(normalised_title)
    if (
        title_length < len(norm.text)
        and norm.text[:title_length] == normalised_title
        and norm.text[title_length] == " "
    ):
        return title_length
    return None


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------


def _load_yaml_mapping(path: Path, what: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as error:  # pragma: no cover - defensive
        raise ValueError(f"{what} is not valid YAML ({path}): {error}") from error
    if not isinstance(data, Mapping):
        raise ValueError(f"{what} root must be a mapping: {path}")
    return dict(data)


def _require_mapping(data: Mapping[str, Any], key: str, where: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: required mapping key {key!r} is missing or not a mapping")
    return dict(value)


def _require_str(data: Mapping[str, Any], key: str, where: str, *, allow_empty: bool = False) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}: required string key {key!r} is missing or not a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{where}: required string key {key!r} must be non-empty")
    return value


def _require_bool(data: Mapping[str, Any], key: str, where: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{where}: key {key!r} must be a boolean")
    return value


def _validate_cue_list(raw: Any, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{where}: cues must be a YAML list")
    cues: list[str] = []
    seen: set[str] = set()
    for index, cue in enumerate(raw):
        if not isinstance(cue, str):
            raise ValueError(f"{where}: cue #{index} must be a string")
        if not cue:
            raise ValueError(f"{where}: cue #{index} is empty")
        if cue != cue.lower():
            raise ValueError(
                f"{where}: cue {cue!r} must be lowercase "
                "(the text normaliser lower-cases all input)"
            )
        if cue != cue.strip():
            raise ValueError(f"{where}: cue {cue!r} has leading/trailing whitespace")
        if _WHITESPACE_RUN.search(cue):
            raise ValueError(f"{where}: cue {cue!r} contains repeated whitespace")
        if cue in seen:
            raise ValueError(f"{where}: duplicate cue {cue!r}")
        seen.add(cue)
        cues.append(cue)
    return tuple(cues)


def load_lexicon(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen rule lexicon.

    Validation covers everything the evidence engine relies on: normalisation
    switches, the negation window/cues/exceptions, the four evidence states,
    the criterion weights (support/unknown/mixed) and the design's hard
    constraints (``UNKNOWN`` is not exclusion, records are never auto-deleted).

    Parameters
    ----------
    path:
        Path to ``configs/rule_lexicon.yaml``.

    Returns
    -------
    dict
        The validated lexicon as a plain dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any required section or value is missing or violates the design.
    """
    data = _load_yaml_mapping(Path(path), "rule lexicon")

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("rule lexicon: schema_version must be a positive integer")

    normalization = _require_mapping(data, "normalization", "rule lexicon")
    for switch in (
        "lowercase",
        "normalize_unicode",
        "normalize_hyphens",
        "collapse_whitespace",
        "preserve_negation",
        "preserve_original_text_for_offsets",
    ):
        if switch in normalization and not isinstance(normalization[switch], bool):
            raise ValueError(f"rule lexicon: normalization.{switch} must be a boolean")
    for required_switch in (
        "lowercase",
        "normalize_unicode",
        "normalize_hyphens",
        "collapse_whitespace",
        "preserve_negation",
        "preserve_original_text_for_offsets",
    ):
        if required_switch not in normalization:
            raise ValueError(
                f"rule lexicon: normalization.{required_switch} is required"
            )
        if not normalization[required_switch]:
            raise ValueError(
                f"rule lexicon: normalization.{required_switch} must be true "
                "(frozen design requirement)"
            )

    negation = _require_mapping(data, "negation", "rule lexicon")
    window = negation.get("window_tokens_before")
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("rule lexicon: negation.window_tokens_before must be an int >= 0")
    _validate_cue_list(negation.get("cues"), "rule lexicon: negation.cues")
    if not negation.get("cues"):
        raise ValueError("rule lexicon: negation.cues must not be empty")
    _validate_cue_list(negation.get("exceptions"), "rule lexicon: negation.exceptions")

    states = data.get("states")
    if not isinstance(states, list) or tuple(states) != STATES:
        raise ValueError(
            "rule lexicon: states must be exactly "
            f"{list(STATES)!r} in that order"
        )

    scoring = _require_mapping(data, "scoring", "rule lexicon")
    weights = tuple(
        _require_float(scoring, key, "rule lexicon: scoring")
        for key in ("support_weight", "unknown_weight", "mixed_weight")
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("rule lexicon: criterion weights must be non-negative")
    if abs(sum(weights) - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"rule lexicon: criterion weights must sum to 1, got {weights!r}"
        )
    relevance = _require_float(scoring, "relevance_weight", "rule lexicon: scoring")
    risk = _require_float(scoring, "false_negative_risk_weight", "rule lexicon: scoring")
    if abs(relevance + risk - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            "rule lexicon: relevance_weight + false_negative_risk_weight must be 1"
        )
    if _require_bool(scoring, "unknown_is_exclusion", "rule lexicon: scoring"):
        raise ValueError(
            "rule lexicon: unknown_is_exclusion must be false "
            "(UNKNOWN is never exclusion; frozen design constraint)"
        )
    if _require_bool(scoring, "auto_exclude_records", "rule lexicon: scoring"):
        raise ValueError(
            "rule lexicon: auto_exclude_records must be false "
            "(rules route, never delete; frozen design constraint)"
        )

    evidence = _require_mapping(data, "evidence", "rule lexicon")
    for flag in ("store_sentence", "store_char_offsets", "store_rule_id"):
        if not _require_bool(evidence, flag, "rule lexicon: evidence"):
            raise ValueError(
                f"rule lexicon: evidence.{flag} must be true (auditability requirement)"
            )

    change_control = _require_mapping(data, "change_control", "rule lexicon")
    for flag in (
        "rule_version_required",
        "main_experiment_rules_frozen_before_results",
        "posthoc_rule_changes_must_be_labeled_exploratory",
    ):
        if not _require_bool(change_control, flag, "rule lexicon: change_control"):
            raise ValueError(
                f"rule lexicon: change_control.{flag} must be true (change-control requirement)"
            )

    return data


def _require_float(data: Mapping[str, Any], key: str, where: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: key {key!r} must be a number")
    return float(value)


def _compute_spec_hash(
    review_id: str,
    topic: str,
    source_reference: str,
    eligibility_text: str,
    rule_version: str,
    criteria: Sequence[CriterionSpec],
) -> str:
    """Stable SHA-256 of the operationalised content of one review spec."""
    payload = {
        "review_id": review_id,
        "topic": topic,
        "source_reference": source_reference,
        "eligibility_text": eligibility_text,
        "rule_version": rule_version,
        "criteria": [
            {
                "criterion_id": criterion.criterion_id,
                "name": criterion.name,
                "required": criterion.required,
                "automation_level": criterion.automation_level,
                "support_cues": list(criterion.support_cues),
                "contradict_cues": list(criterion.contradict_cues),
                "notes": criterion.notes,
                "source_excerpt": criterion.source_excerpt,
            }
            for criterion in criteria
        ],
    }
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_review_specs(path: str | Path) -> dict[str, ReviewSpec]:
    """Load and validate the frozen review specifications.

    Each review is validated for: required metadata, a valid ``rule_version``,
    2-6 criteria with unique ids, a non-empty set of required criteria, valid
    automation levels, lowercase/unique/non-empty cue phrases, and disjoint
    support vs. contradict cue sets.  A stable ``spec_hash`` is computed over
    the operationalised content; if the YAML provides ``spec_hash`` it must
    match (change-control protection).

    Parameters
    ----------
    path:
        Path to ``configs/review_specs.yaml``.

    Returns
    -------
    dict[str, ReviewSpec]
        Mapping ``review_id -> ReviewSpec`` in file order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file or any review violates the schema.
    """
    data = _load_yaml_mapping(Path(path), "review specs")

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("review specs: schema_version must be a positive integer")

    raw_reviews = data.get("reviews")
    if not isinstance(raw_reviews, list) or not raw_reviews:
        raise ValueError("review specs: 'reviews' must be a non-empty list")

    global_rule_version = data.get("rule_version")
    if global_rule_version is not None and (
        not isinstance(global_rule_version, str)
        or not _RULE_VERSION_PATTERN.match(global_rule_version)
    ):
        raise ValueError(
            "review specs: top-level rule_version must look like 'v1.0.0'"
        )

    specs: dict[str, ReviewSpec] = {}
    for index, raw in enumerate(raw_reviews):
        where = f"review #{index}"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: must be a mapping")
        raw = dict(raw)
        review_id = raw.get("id", raw.get("review_id"))
        if not isinstance(review_id, str) or not review_id.strip():
            raise ValueError(f"{where}: 'id' must be a non-empty string")
        where = f"review {review_id!r}"
        if review_id in specs:
            raise ValueError(f"{where}: duplicate review id")
        topic = _require_str(raw, "topic", where)
        source_reference = _require_str(raw, "source_reference", where)
        eligibility_text = _require_str(raw, "eligibility_text", where)
        if len(eligibility_text) < 20:
            raise ValueError(
                f"{where}: eligibility_text is implausibly short; "
                "use the verbatim published eligibility criteria"
            )
        rule_version = raw.get("rule_version", global_rule_version)
        if not isinstance(rule_version, str) or not _RULE_VERSION_PATTERN.match(
            rule_version
        ):
            raise ValueError(
                f"{where}: rule_version must look like 'v1.0.0' "
                "(set per review or at the top level)"
            )

        raw_criteria = raw.get("criteria")
        if not isinstance(raw_criteria, list):
            raise ValueError(f"{where}: 'criteria' must be a list")
        if not 2 <= len(raw_criteria) <= 6:
            raise ValueError(
                f"{where}: expected 2-6 criterion axes, got {len(raw_criteria)}"
            )

        criteria: list[CriterionSpec] = []
        seen: set[str] = set()
        for criterion_index, raw_criterion in enumerate(raw_criteria):
            criterion_where = f"{where} criterion #{criterion_index}"
            if not isinstance(raw_criterion, Mapping):
                raise ValueError(f"{criterion_where}: must be a mapping")
            raw_criterion = dict(raw_criterion)
            criterion_id = raw_criterion.get(
                "id", raw_criterion.get("criterion_id")
            )
            if not isinstance(criterion_id, str) or not criterion_id.strip():
                raise ValueError(
                    f"{criterion_where}: 'id' must be a non-empty string"
                )
            criterion_where = f"{where} criterion {criterion_id!r}"
            if criterion_id in seen:
                raise ValueError(f"{criterion_where}: duplicate criterion id")
            seen.add(criterion_id)
            name = _require_str(raw_criterion, "name", criterion_where)
            required = raw_criterion.get("required")
            if not isinstance(required, bool):
                raise ValueError(f"{criterion_where}: 'required' must be a boolean")
            automation_level = raw_criterion.get("automation_level")
            if automation_level not in AUTOMATION_LEVELS:
                raise ValueError(
                    f"{criterion_where}: automation_level must be one of "
                    f"{list(AUTOMATION_LEVELS)!r}, got {automation_level!r}"
                )
            support_cues = _validate_cue_list(
                raw_criterion.get("support_cues"), f"{criterion_where} support_cues"
            )
            contradict_cues = _validate_cue_list(
                raw_criterion.get("contradict_cues"),
                f"{criterion_where} contradict_cues",
            )
            overlap = set(support_cues) & set(contradict_cues)
            if overlap:
                raise ValueError(
                    f"{criterion_where}: cues cannot be both support and "
                    f"contradict: {sorted(overlap)!r}"
                )
            notes = raw_criterion.get("notes", "")
            if not isinstance(notes, str):
                raise ValueError(f"{criterion_where}: 'notes' must be a string")
            source_excerpt = raw_criterion.get("source_excerpt", "")
            if not isinstance(source_excerpt, str):
                raise ValueError(
                    f"{criterion_where}: 'source_excerpt' must be a string"
                )
            criteria.append(
                CriterionSpec(
                    criterion_id=criterion_id,
                    name=name,
                    required=required,
                    automation_level=automation_level,
                    support_cues=support_cues,
                    contradict_cues=contradict_cues,
                    notes=notes,
                    source_excerpt=source_excerpt,
                )
            )

        if not any(criterion.required for criterion in criteria):
            raise ValueError(
                f"{where}: at least one criterion must be marked required"
            )

        spec_hash = _compute_spec_hash(
            review_id=review_id,
            topic=topic,
            source_reference=source_reference,
            eligibility_text=eligibility_text,
            rule_version=rule_version,
            criteria=criteria,
        )
        declared_hash = raw.get("spec_hash")
        if declared_hash is not None:
            if not isinstance(declared_hash, str) or declared_hash != spec_hash:
                raise ValueError(
                    f"{where}: declared spec_hash does not match the computed "
                    f"content hash (expected {spec_hash}, got {declared_hash!r})"
                )
        specs[review_id] = ReviewSpec(
            review_id=review_id,
            topic=topic,
            source_reference=source_reference,
            eligibility_text=eligibility_text,
            criteria=tuple(criteria),
            rule_version=rule_version,
            spec_hash=spec_hash,
        )

    return specs


# ---------------------------------------------------------------------------
# Cue compilation
# ---------------------------------------------------------------------------


def _cue_to_regex(cue: str) -> str:
    """Compile one cue phrase into a regex fragment.

    Spaces and hyphens inside the cue are interchangeable, and a regular
    English plural suffix is tolerated for cues long enough to be unambiguous
    (``"mesenchymal stem cell"`` also matches ``"mesenchymal stem cells"``).
    Short abbreviations (``"mtx"``, ``"bmi"``, ``"ra"``) never receive the plural
    suffix to avoid spurious matches.
    """
    escaped = re.escape(cue)
    escaped = escaped.replace(r"\-", "[- ]").replace(r"\ ", "[- ]")
    last_token = cue.rsplit(" ", 1)[-1]
    if (
        len(cue) >= 4
        and cue[-1].isalpha()
        and not cue.endswith("s")
        and len(last_token) >= 3
    ):
        escaped += r"(?:s|es)?"
    return escaped


def _compile_criterion_pattern(cues: Sequence[str]) -> re.Pattern[str] | None:
    """Compile all cues of one polarity into a single named-group regex."""
    if not cues:
        return None
    order = sorted(range(len(cues)), key=lambda index: (-len(cues[index]), index))
    alternatives = [
        f"(?P<cue_{index}>{_cue_to_regex(cues[index])})" for index in order
    ]
    body = "|".join(alternatives)
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


@dataclass(frozen=True)
class _CriterionPatterns:
    support_cues: tuple[str, ...]
    contradict_cues: tuple[str, ...]
    support: re.Pattern[str] | None
    contradict: re.Pattern[str] | None


@dataclass(frozen=True)
class _CueOccurrence:
    cue: str
    cue_index: int
    polarity: str
    norm_start: int
    norm_end: int
    negated: bool
    ambiguous: bool


@dataclass(frozen=True)
class _PreparedText:
    norm: NormalizedText
    tokens: tuple[Token, ...]
    sentences: tuple[tuple[int, int], ...]
    combined_original: str
    title_boundary: int | None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CriterionEvidenceEngine:
    """Extract per-criterion evidence and aggregate criterion risk for one review.

    Parameters
    ----------
    review_spec:
        Frozen, human-curated review specification.
    lexicon:
        Validated lexicon dictionary (see :func:`load_lexicon`).  Negation
        window/cues/exceptions and the default criterion weights are read from
        it.
    scoring_weights:
        ``(support, unknown, mixed)`` weights for ``q_score``.  The default
        ``(0.50, 0.30, 0.20)`` resolves to the lexicon weights; pass an
        explicit tuple only for a pre-registered sensitivity analysis.

    Notes
    -----
    The engine is stateless per record: all per-record values are derived from
    the title/abstract text only.  A small bounded cache avoids re-normalising
    identical texts within one run.
    """

    def __init__(
        self,
        review_spec: ReviewSpec,
        lexicon: Mapping[str, Any],
        scoring_weights: tuple[float, float, float] = DEFAULT_SCORING_WEIGHTS,
        max_cached_texts: int = 256,
    ) -> None:
        if not isinstance(review_spec, ReviewSpec):
            raise TypeError("review_spec must be a ReviewSpec instance")
        if not isinstance(lexicon, Mapping):
            raise TypeError("lexicon must be a mapping (see load_lexicon)")
        if not review_spec.required_criteria:
            raise ValueError(
                f"{review_spec.review_id}: at least one required criterion is needed"
            )
        if max_cached_texts < 0:
            raise ValueError("max_cached_texts must be >= 0")

        self.review_spec = review_spec
        self.lexicon = dict(lexicon)
        self.scoring_weights = self._resolve_weights(scoring_weights, self.lexicon)
        self.negation_window, self.negation_cues, self.negation_exceptions = (
            _negation_settings(self.lexicon)
        )
        self._required_criteria = review_spec.required_criteria
        self._patterns: dict[str, _CriterionPatterns] = {
            criterion.criterion_id: _CriterionPatterns(
                support_cues=criterion.support_cues,
                contradict_cues=criterion.contradict_cues,
                support=_compile_criterion_pattern(criterion.support_cues),
                contradict=_compile_criterion_pattern(criterion.contradict_cues),
            )
            for criterion in review_spec.criteria
        }
        self._max_cached_texts = max_cached_texts
        self._text_cache: OrderedDict[tuple[str, str], _PreparedText] = OrderedDict()

    # -- public API ---------------------------------------------------------

    def evaluate(
        self, record_id: str, title: str | None, abstract: str | None
    ) -> CriterionRecordEvidence:
        """Evaluate one record and return its full evidence object.

        Parameters
        ----------
        record_id:
            Stable model-visible record identifier.
        title, abstract:
            Original text fields; ``None``/NaN are treated as empty strings.

        Returns
        -------
        CriterionRecordEvidence
            One hit per *required* criterion plus the aggregate ``s_ratio``,
            ``u_ratio``, ``x_ratio`` and ``q_score``.
        """
        record_id = "" if record_id is None else str(record_id)
        title_text = _as_text(title)
        abstract_text = _as_text(abstract)
        prepared = self._prepare(title_text, abstract_text)
        hits = tuple(
            self._evaluate_criterion(criterion, prepared)
            for criterion in self._required_criteria
        )

        total = len(self._required_criteria)
        counts = {"SUPPORT": 0, "CONTRADICT": 0, "UNKNOWN": 0, "MIXED": 0}
        for hit in hits:
            counts[hit.state] += 1
        support_ratio = counts["SUPPORT"] / total
        unknown_ratio = counts["UNKNOWN"] / total
        mixed_ratio = counts["MIXED"] / total
        support_weight, unknown_weight, mixed_weight = self.scoring_weights
        q_score = (
            support_weight * support_ratio
            + unknown_weight * unknown_ratio
            + mixed_weight * mixed_ratio
        )
        q_score = min(1.0, max(0.0, q_score))

        return CriterionRecordEvidence(
            record_id=record_id,
            hits=hits,
            s_ratio=support_ratio,
            u_ratio=unknown_ratio,
            x_ratio=mixed_ratio,
            q_score=q_score,
        )

    def evaluate_frame(
        self, records_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, CriterionRecordEvidence]]:
        """Evaluate a model-visible record table.

        Parameters
        ----------
        records_df:
            DataFrame with at least ``record_id``, ``title`` and ``abstract``
            columns (the processed SYNERGY schema).  Extra columns are ignored;
            no annotation column is ever read.

        Returns
        -------
        tuple[pandas.DataFrame, dict[str, CriterionRecordEvidence]]
            * one row per input record with the :data:`FRAME_COLUMNS` schema
              (same order as ``records_df``);
            * a cache ``record_id -> CriterionRecordEvidence`` so evidence
              cards, case audits and analysis reuse the full hit objects
              without re-running the engine.
        """
        if not isinstance(records_df, pd.DataFrame):
            raise TypeError("records_df must be a pandas DataFrame")
        missing = [
            column
            for column in ("record_id", "title", "abstract")
            if column not in records_df.columns
        ]
        if missing:
            raise ValueError(
                f"records_df is missing required column(s): {missing!r}; "
                f"expected the processed schema {tuple(FRAME_COLUMNS[:1])} + text fields"
            )

        rows: list[dict[str, Any]] = []
        cache: dict[str, CriterionRecordEvidence] = {}
        for row_index, record in enumerate(records_df.itertuples(index=False)):
            raw_record_id = getattr(record, "record_id")
            if _is_missing(raw_record_id):
                raise ValueError(
                    f"records_df row {row_index}: record_id is missing"
                )
            record_id = str(raw_record_id)
            evidence = self.evaluate(
                record_id,
                getattr(record, "title"),
                getattr(record, "abstract"),
            )
            rows.append(evidence.to_row())
            cache[record_id] = evidence

        frame = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
        if not frame.empty:
            for column in ("s_ratio", "u_ratio", "x_ratio", "q_score"):
                frame[column] = frame[column].astype(float)
            for column in ("n_support", "n_contradict", "n_unknown", "n_mixed"):
                frame[column] = frame[column].astype(int)
        return frame, cache

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _resolve_weights(
        scoring_weights: Sequence[float], lexicon: Mapping[str, Any]
    ) -> tuple[float, float, float]:
        try:
            requested = tuple(float(weight) for weight in scoring_weights)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "scoring_weights must be a 3-tuple of numbers"
            ) from error
        if len(requested) != 3:
            raise ValueError("scoring_weights must contain exactly 3 values")
        if any(weight < 0.0 or weight > 1.0 for weight in requested):
            raise ValueError("scoring_weights values must be in [0, 1]")

        scoring = lexicon.get("scoring", {})
        lexicon_weights = (
            float(scoring.get("support_weight", DEFAULT_SCORING_WEIGHTS[0])),
            float(scoring.get("unknown_weight", DEFAULT_SCORING_WEIGHTS[1])),
            float(scoring.get("mixed_weight", DEFAULT_SCORING_WEIGHTS[2])),
        )
        if abs(sum(lexicon_weights) - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"lexicon criterion weights must sum to 1, got {lexicon_weights!r}"
            )
        default_rounded = tuple(round(weight, 9) for weight in DEFAULT_SCORING_WEIGHTS)
        if tuple(round(weight, 9) for weight in requested) == default_rounded:
            # Default argument: the frozen lexicon is authoritative.
            return lexicon_weights
        if abs(sum(requested) - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"scoring_weights must sum to 1, got {requested!r}"
            )
        return requested

    def _prepare(self, title: str, abstract: str) -> _PreparedText:
        key = (title, abstract)
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached
        combined = concat_text(title, abstract)
        norm = normalize(combined)
        title_boundary = _title_abstract_boundary(title, abstract, norm)
        prepared = _PreparedText(
            norm=norm,
            tokens=tuple(tokenize(norm)),
            sentences=tuple(
                split_sentences(
                    norm,
                    forced_breaks=() if title_boundary is None else (title_boundary,),
                )
            ),
            combined_original=combined,
            title_boundary=title_boundary,
        )
        if self._max_cached_texts:
            self._text_cache[key] = prepared
            if len(self._text_cache) > self._max_cached_texts:
                self._text_cache.popitem(last=False)
        return prepared

    def _evaluate_criterion(
        self, criterion: CriterionSpec, prepared: _PreparedText
    ) -> CriterionHit:
        patterns = self._patterns[criterion.criterion_id]
        occurrences: list[_CueOccurrence] = []
        for polarity, pattern, cues in (
            ("support", patterns.support, patterns.support_cues),
            ("contradict", patterns.contradict, patterns.contradict_cues),
        ):
            if pattern is None:
                continue
            for match in pattern.finditer(prepared.norm.text):
                group = match.lastgroup
                if group is None:  # pragma: no cover - defensive
                    continue
                cue_index = int(group.split("_", 1)[1])
                negated, ambiguous = is_negated(
                    prepared.norm,
                    prepared.tokens,
                    match.start(),
                    window_tokens_before=self.negation_window,
                    cues=self.negation_cues,
                    exceptions=self.negation_exceptions,
                    scope_boundaries=(
                        ()
                        if prepared.title_boundary is None
                        else (prepared.title_boundary,)
                    ),
                )
                occurrences.append(
                    _CueOccurrence(
                        cue=cues[cue_index],
                        cue_index=cue_index,
                        polarity=polarity,
                        norm_start=match.start(),
                        norm_end=match.end(),
                        negated=negated,
                        ambiguous=ambiguous,
                    )
                )

        if not occurrences:
            return _unknown_hit(criterion)

        resolved_support = sorted(
            (
                occurrence
                for occurrence in occurrences
                if occurrence.polarity == "support"
                and not occurrence.negated
                and not occurrence.ambiguous
            ),
            key=lambda occurrence: (occurrence.norm_start, occurrence.norm_end),
        )
        resolved_contradict = sorted(
            (
                occurrence
                for occurrence in occurrences
                if occurrence.polarity == "contradict"
                and not occurrence.negated
                and not occurrence.ambiguous
            ),
            key=lambda occurrence: (occurrence.norm_start, occurrence.norm_end),
        )
        ambiguous_hits = sorted(
            (occurrence for occurrence in occurrences if occurrence.ambiguous),
            key=lambda occurrence: (occurrence.norm_start, occurrence.norm_end),
        )

        if resolved_support and resolved_contradict:
            evidence = min(
                resolved_support[0], resolved_contradict[0], key=lambda item: item.norm_start
            )
            return self._build_hit(
                criterion,
                prepared,
                state="MIXED",
                occurrence=evidence,
                unresolved=True,
                conflict=True,
            )
        if resolved_support:
            return self._build_hit(
                criterion,
                prepared,
                state="SUPPORT",
                occurrence=resolved_support[0],
                unresolved=False,
                conflict=False,
            )
        if resolved_contradict:
            return self._build_hit(
                criterion,
                prepared,
                state="CONTRADICT",
                occurrence=resolved_contradict[0],
                unresolved=False,
                conflict=False,
            )
        if ambiguous_hits:
            return self._build_hit(
                criterion,
                prepared,
                state="MIXED",
                occurrence=ambiguous_hits[0],
                unresolved=True,
                conflict=False,
            )
        # Only negated cues: a negated support cue is never a contradiction and
        # a negated contradiction cue is never support, so the honest state is
        # UNKNOWN (design doc 03, section 4).
        return _unknown_hit(criterion)

    def _build_hit(
        self,
        criterion: CriterionSpec,
        prepared: _PreparedText,
        *,
        state: str,
        occurrence: _CueOccurrence,
        unresolved: bool,
        conflict: bool,
    ) -> CriterionHit:
        original_start, original_end = prepared.norm.to_original_span(
            occurrence.norm_start, occurrence.norm_end
        )
        matched_phrase = prepared.combined_original[original_start:original_end]
        evidence_sentence = ""
        sentence_span = sentence_at(prepared.sentences, occurrence.norm_start)
        if sentence_span is not None:
            sentence_start, sentence_end = prepared.norm.to_original_span(*sentence_span)
            evidence_sentence = prepared.combined_original[
                sentence_start:sentence_end
            ].strip()
        confidence = _CONFIDENCE_BY_AUTOMATION[criterion.automation_level]
        if unresolved or conflict:
            confidence = "low"
        return CriterionHit(
            criterion_id=criterion.criterion_id,
            state=state,
            matched_phrase=matched_phrase,
            evidence_sentence=evidence_sentence,
            start_char=original_start,
            end_char=original_end,
            rule_id=(
                f"{self.review_spec.review_id}:{criterion.criterion_id}:"
                f"{_POLARITY_SHORT[occurrence.polarity]}:{occurrence.cue_index}"
            ),
            confidence_level=confidence,
            cue_polarity=(
                "mixed"
                if state == "MIXED"
                else occurrence.polarity
            ),
        )


def _unknown_hit(criterion: CriterionSpec) -> CriterionHit:
    """UNKNOWN hit: no locatable evidence, therefore no offsets/sentence."""
    return CriterionHit(
        criterion_id=criterion.criterion_id,
        state="UNKNOWN",
        matched_phrase="",
        evidence_sentence="",
        start_char=-1,
        end_char=-1,
        rule_id="",
        confidence_level="low",
        cue_polarity="none",
    )


def _negation_settings(
    lexicon: Mapping[str, Any],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    negation = lexicon.get("negation")
    if not isinstance(negation, Mapping):
        return 5, NEGATION_CUES, NEGATION_EXCEPTIONS
    window = negation.get("window_tokens_before", 5)
    cues = negation.get("cues") or NEGATION_CUES
    exceptions = negation.get("exceptions") or ()
    return (
        int(window),
        tuple(str(cue) for cue in cues),
        tuple(str(exception) for exception in exceptions),
    )


def _as_text(value: Any) -> str:
    """Return text, mapping None/NaN/pandas-NA to the empty string."""
    if value is None:
        return ""
    if _is_missing(value):
        return ""
    return value if isinstance(value, str) else str(value)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - exotic/array-like objects
        return False


def load_engines(
    configs_dir: str | Path,
    scoring_weights: tuple[float, float, float] = DEFAULT_SCORING_WEIGHTS,
) -> dict[str, CriterionEvidenceEngine]:
    """Convenience loader: all five engines from a configs directory.

    Parameters
    ----------
    configs_dir:
        Directory containing ``rule_lexicon.yaml`` and ``review_specs.yaml``.
    scoring_weights:
        Same semantics as in :class:`CriterionEvidenceEngine`.

    Returns
    -------
    dict[str, CriterionEvidenceEngine]
        ``review_id -> engine``.
    """
    directory = Path(configs_dir)
    lexicon = load_lexicon(directory / "rule_lexicon.yaml")
    specs = load_review_specs(directory / "review_specs.yaml")
    return {
        review_id: CriterionEvidenceEngine(
            spec, lexicon, scoring_weights=scoring_weights
        )
        for review_id, spec in specs.items()
    }
