"""Risk router: combine relevance ranking with criterion-evidence risk.

The router only *re-orders* records for human review.  It never deletes or
auto-excludes a record, and it never reads oracle labels.

For a record ``i`` with predicted inclusion probability ``p_i`` and criterion
risk ``Q_i`` (computed by the evidence engine) the design specifies:

    P_i = percentile_rank(p_i)
    F_i = percentile_rank((1 - p_i) * Q_i)
    BioLitScore_i = 0.60 * P_i + 0.40 * F_i

``P`` and ``F`` are percentile ranks inside the *current unlabelled pool*, so
both terms live on the same [0, 1] scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERCENTILE_METHOD = "average"


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] percentile ranks (average ranks for ties)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    ranks = pd.Series(values).rank(method=PERCENTILE_METHOD).to_numpy(dtype=float)
    return (ranks - 1.0) / (n - 1.0)


def criterion_risk_no_unknown(
    s_ratio: np.ndarray, x_ratio: np.ndarray
) -> np.ndarray:
    """Ablation: drop the UNKNOWN term, Q = 0.50 S + 0.50 X."""
    return np.clip(0.50 * s_ratio + 0.50 * x_ratio, 0.0, 1.0)


@dataclass(frozen=True)
class MethodSpec:
    """Frozen definition of one method under test."""

    name: str
    family: str  # random / relevance / uncertainty / criterion / asreview_elas / biolit / hybrid
    relevance_weight: float = 0.60
    risk_weight: float = 0.40
    drop_unknown: bool = False
    prior_included: int = 5
    prior_excluded: int = 5
    relevance_quota: float | None = None  # hybrid methods: share of batch from M1 queue
    permute_q: bool = False  # placebo: permute criterion scores within the round
    description: str = ""


METHODS: dict[str, MethodSpec] = {
    "random": MethodSpec("random", "random", description="M0 random querying"),
    "matched_tfidf_logreg": MethodSpec(
        "matched_tfidf_logreg", "relevance", description="M1 TF-IDF+LR active learning"
    ),
    "uncertainty_only": MethodSpec(
        "uncertainty_only", "uncertainty", description="M2 least-confident first"
    ),
    "criterion_only": MethodSpec(
        "criterion_only", "criterion", description="M3 criterion risk only, no relevance model"
    ),
    "asreview_default": MethodSpec(
        "asreview_default",
        "asreview_elas",
        description=(
            "M4 ELAS-u4 component reimplementation (TF-IDF word bigrams + LinearSVC "
            "C=0.11, class-balancing ratio 9.8, max querier); not native ASReview"
        ),
    ),
    "biolit_relay": MethodSpec(
        "biolit_relay",
        "biolit",
        description=(
            "M5 full-fusion criterion-aware risk-biased re-ranking "
            "(0.60 relevance percentile + 0.40 tail-rescue percentile)"
        ),
    ),
    "biolit_relay_nounknown": MethodSpec(
        "biolit_relay_nounknown",
        "biolit",
        drop_unknown=True,
        description="Ablation: Q without UNKNOWN term (0.50 S + 0.50 X)",
    ),
    "biolit_relay_w50": MethodSpec(
        "biolit_relay_w50", "biolit", relevance_weight=0.50, risk_weight=0.50,
        description="Sensitivity: equal relevance/risk weights",
    ),
    "biolit_relay_w70": MethodSpec(
        "biolit_relay_w70", "biolit", relevance_weight=0.70, risk_weight=0.30,
        description="Sensitivity: relevance-dominant weights",
    ),
    "matched_tfidf_logreg_prior2": MethodSpec(
        "matched_tfidf_logreg_prior2", "relevance", prior_included=2, prior_excluded=2,
        description="Sensitivity: matched AL with 2+2 priors",
    ),
    "biolit_relay_prior2": MethodSpec(
        "biolit_relay_prior2", "biolit", prior_included=2, prior_excluded=2,
        description="Sensitivity: BioLit-Relay with 2+2 priors",
    ),
    "biolit_relay_placebo": MethodSpec(
        "biolit_relay_placebo",
        "biolit",
        permute_q=True,
        description=(
            "Placebo control: criterion scores are randomly permuted within each "
            "round, preserving the marginal distribution but destroying the "
            "record-criterion correspondence"
        ),
    ),
    # --- M6: genuine hybrid rescue scheduling (fixed quota, frozen a priori) ---
    "hybrid_relay": MethodSpec(
        "hybrid_relay",
        "hybrid",
        relevance_quota=0.80,
        description=(
            "M6 Hybrid-Relay: per 10-record batch, 8 highest-p relevance slots + "
            "2 tail-rescue slots from the remaining records (80/20, fixed a priori)"
        ),
    ),
    "hybrid_relay_90": MethodSpec(
        "hybrid_relay_90",
        "hybrid",
        relevance_quota=0.90,
        description=(
            "M6 sensitivity: 90/10 quota (9 relevance slots + 1 rescue slot per "
            "10-record batch), pre-specified before running"
        ),
    ),
}


def route_scores(
    method: str,
    p_include: np.ndarray,
    q_score: np.ndarray,
    s_ratio: np.ndarray,
    x_ratio: np.ndarray,
) -> np.ndarray:
    """Return the routing score for every unlabelled record (higher = earlier)."""
    spec = METHODS[method]
    if spec.family == "criterion":
        return np.asarray(q_score, dtype=float)
    if spec.family == "relevance":
        return percentile_rank(p_include)
    if spec.family == "asreview_elas":
        return percentile_rank(p_include)
    if spec.family == "uncertainty":
        p = np.asarray(p_include, dtype=float)
        return 1.0 - np.abs(2.0 * p - 1.0)
    if spec.family == "biolit":
        q = criterion_risk_no_unknown(s_ratio, x_ratio) if spec.drop_unknown else np.asarray(
            q_score, dtype=float
        )
        p_rank = percentile_rank(p_include)
        risk = percentile_rank((1.0 - np.asarray(p_include, dtype=float)) * q)
        return spec.relevance_weight * p_rank + spec.risk_weight * risk
    raise ValueError(f"method {method!r} has no score rule")


def select_batch(
    record_ids: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Indices of the next query batch, deterministic tie-break by record_id."""
    order = np.lexsort((record_ids.astype(str), -np.asarray(scores, dtype=float)))
    return order[: max(0, min(k, order.size))]


# ---------------------------------------------------------------------------
# reason codes for evidence cards / case audit
# ---------------------------------------------------------------------------

REASON_CODE_DESCRIPTIONS = {
    "LOW_MODEL_SCORE_WITH_CRITERION_SUPPORT": "relevance below median but explicit criterion support exists",
    "CRITERION_UNKNOWN": "at least half of required criteria cannot be assessed from title/abstract",
    "MIXED_EVIDENCE": "conflicting criterion evidence in the same record",
    "HIGH_MODEL_SCORE": "relevance in the top quintile",
    "LOW_MODEL_SCORE": "relevance in the bottom half",
    "CONTRADICT_EVIDENCE": "at least one required criterion is contradicted",
    "RISK_PROMOTED": "false-negative-risk term moved the record into the batch",
}


def reason_codes(
    p_percentile: float,
    s_ratio: float,
    u_ratio: float,
    x_ratio: float,
    contradicted: bool,
    risk_weight: float = 0.40,
) -> list[str]:
    codes: list[str] = []
    if p_percentile >= 0.8:
        codes.append("HIGH_MODEL_SCORE")
    if p_percentile < 0.5:
        codes.append("LOW_MODEL_SCORE")
    if p_percentile < 0.5 and s_ratio > 0:
        codes.append("LOW_MODEL_SCORE_WITH_CRITERION_SUPPORT")
    if u_ratio >= 0.5:
        codes.append("CRITERION_UNKNOWN")
    if x_ratio > 0:
        codes.append("MIXED_EVIDENCE")
    if contradicted:
        codes.append("CONTRADICT_EVIDENCE")
    if risk_weight > 0 and p_percentile < 0.5 and (s_ratio + u_ratio + x_ratio) > 0:
        codes.append("RISK_PROMOTED")
    return codes
