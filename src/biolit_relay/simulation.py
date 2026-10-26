"""Label-safe budgeted screening simulator.

The simulator executes one screening workflow for one (review, method, seed)
triple.  Gold labels live in an :class:`Oracle` that only reveals a label *after*
the router has selected the record; the relevance ranker and the criterion
evidence engine only ever receive label-free records.

Method behaviour is defined in :mod:`biolit_relay.risk_router`; the evidence
states are produced offline by :mod:`biolit_relay.criterion_evidence` and passed
in as a label-free table.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .core import ExperimentProtocol, SimulationEvent
from .metrics import compute_metrics, curve_frame, curve_from_events
from .relevance_ranker import (
    AsreviewElasU4Ranker,
    TfidfLogRegRanker,
    build_text,
    fit_tfidf,
)
from .risk_router import METHODS, reason_codes, route_scores, select_batch


@dataclass
class ReviewBundle:
    """Everything the simulator needs for one review (all label-free except labels)."""

    review_id: str
    records: pd.DataFrame  # columns: record_id, title, abstract, has_abstract, ...
    evidence: pd.DataFrame  # aligned: record_id, q_score, s_ratio, u_ratio, x_ratio
    labels: np.ndarray  # oracle-only
    texts: np.ndarray = field(repr=False, default=None)
    X: sparse.csr_matrix | None = field(repr=False, default=None)
    vectorizer: Any = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if len(self.records) != len(self.evidence) or len(self.records) != len(self.labels):
            raise ValueError("records/evidence/labels must be aligned")
        self.records = self.records.reset_index(drop=True)
        self.evidence = self.evidence.reset_index(drop=True)
        if not np.array_equal(
            self.records["record_id"].to_numpy(), self.evidence["record_id"].to_numpy()
        ):
            raise ValueError("evidence rows are not aligned with records")
        self.record_ids = self.records["record_id"].to_numpy()
        self.n_records = len(self.records)
        self.n_included = int(self.labels.sum())
        self.q = self.evidence["q_score"].to_numpy(dtype=float)
        self.s_ratio = self.evidence["s_ratio"].to_numpy(dtype=float)
        self.u_ratio = self.evidence["u_ratio"].to_numpy(dtype=float)
        self.x_ratio = self.evidence["x_ratio"].to_numpy(dtype=float)

    def ensure_features(self) -> None:
        if self.X is None:
            self.texts = build_text(self.records)
            self.vectorizer, self.X = fit_tfidf(self.texts)


@dataclass
class RunResult:
    run_id: str
    review_id: str
    method: str
    seed: int
    n_records: int
    n_included: int
    events: list[SimulationEvent]
    curve: pd.DataFrame
    metrics: dict[str, float | None]
    selected: pd.DataFrame
    meta: dict[str, Any]


def _initial_priors(
    labels: np.ndarray,
    protocol: ExperimentProtocol,
    seed: int,
    prior_included: int | None = None,
    prior_excluded: int | None = None,
) -> np.ndarray:
    n_inc = prior_included if prior_included is not None else protocol.prior_included
    n_exc = prior_excluded if prior_excluded is not None else protocol.prior_excluded
    rng = np.random.default_rng(seed)
    included_idx = np.flatnonzero(labels == 1)
    excluded_idx = np.flatnonzero(labels == 0)
    if included_idx.size < n_inc or excluded_idx.size < n_exc:
        raise ValueError("not enough records for the requested priors")
    chosen_inc = rng.choice(included_idx, size=n_inc, replace=False)
    chosen_exc = rng.choice(excluded_idx, size=n_exc, replace=False)
    return np.concatenate([chosen_inc, chosen_exc])


def run_simulation(
    bundle: ReviewBundle,
    method: str,
    seed: int,
    protocol: ExperimentProtocol,
    config_hash: str = "",
    spec_hash: str = "",
    asreview_version: str = "",
) -> RunResult:
    """Run one screening simulation and return the full audit trail."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}")
    method_spec = METHODS[method]
    t0 = time.perf_counter()
    bundle.ensure_features()
    X = bundle.X
    labels = bundle.labels
    ids = bundle.record_ids
    n = bundle.n_records
    method_feature = method_spec.family

    priors = _initial_priors(
        labels,
        protocol,
        seed,
        prior_included=method_spec.prior_included,
        prior_excluded=method_spec.prior_excluded,
    )
    labelled = np.zeros(n, dtype=bool)
    labelled[priors] = True
    checked = int(priors.size)
    found = int(labels[priors].sum())
    max_checked = max(
        checked, int(math.ceil(protocol.max_budget_fraction * n))
    )

    ranker = (
        AsreviewElasU4Ranker() if method_feature == "asreview_elas" else TfidfLogRegRanker()
    )
    rng_random = np.random.default_rng(seed + 1_000_003)

    events: list[SimulationEvent] = []
    selected_rows: list[dict[str, Any]] = []
    prior_labels = labels[priors]
    for pos, idx in enumerate(priors):
        selected_rows.append(
            {
                "round": 0,
                "record_id": ids[idx],
                "score": np.nan,
                "p_include": np.nan,
                "q_score": float(bundle.q[idx]),
                "s_ratio": float(bundle.s_ratio[idx]),
                "u_ratio": float(bundle.u_ratio[idx]),
                "x_ratio": float(bundle.x_ratio[idx]),
                "reason_codes": "INITIAL_PRIOR",
                "revealed_label": int(labels[idx]),
                "p_percentile": np.nan,
                "selection_role": "prior",
            }
        )
    events.append(
        SimulationEvent(
            run_id="",
            review_id=bundle.review_id,
            method=method,
            seed=seed,
            iteration=0,
            selected_record_ids=tuple(ids[priors]),
            revealed_labels=tuple(int(v) for v in prior_labels),
            checked=checked,
            found=found,
            budget_fraction=checked / n,
            recall=found / max(bundle.n_included, 1),
        )
    )

    iteration = 0
    while checked < max_checked:
        unlabeled_idx = np.flatnonzero(~labelled)
        if unlabeled_idx.size == 0:
            break
        iteration += 1
        k = min(protocol.query_batch_size, max_checked - checked, unlabeled_idx.size)
        if k <= 0:
            break
        p_include = np.full(unlabeled_idx.size, np.nan)
        p_percentile = np.full(unlabeled_idx.size, np.nan)
        role_of_local: dict[int, str] = {}
        if method_feature == "random":
            scores = rng_random.random(unlabeled_idx.size)
        else:
            if method_feature != "criterion":
                p_include = ranker.fit_predict(
                    X, np.flatnonzero(labelled), labels[labelled], unlabeled_idx, seed=seed
                )
                from .risk_router import percentile_rank

                p_percentile = percentile_rank(p_include)

            if method_feature == "hybrid":
                # M6 Hybrid-Relay: fixed-quota hybrid scheduling.
                # For a full batch of 10 with quota 0.80: 8 highest-p relevance
                # slots + 2 tail-rescue slots from the remaining records.  A
                # truncated final batch keeps the same quota proportionally
                # (ceil of the relevance share) so the batch size is exact.
                quota = method_spec.relevance_quota
                if quota is None:
                    raise ValueError(f"hybrid method {method!r} requires relevance_quota")
                # relevance share of the batch (quota is the relevance quota,
                # e.g. 0.80 -> 8 of 10 records follow the M1 queue).
                n_main = int(math.ceil(k * quota))
                n_main = max(0, min(k, n_main))
                n_rescue = k - n_main
                q_unlab = np.asarray(bundle.q[unlabeled_idx], dtype=float)
                tail_priority = (1.0 - p_include) * q_unlab
                order_main = np.lexsort((ids[unlabeled_idx].astype(str), -p_include))
                main_local = order_main[:n_main]
                used = np.zeros(unlabeled_idx.size, dtype=bool)
                used[main_local] = True
                remaining = np.flatnonzero(~used)
                rescue_local = np.empty(0, dtype=int)
                if n_rescue > 0 and remaining.size > 0:
                    rem_order = np.lexsort(
                        (ids[unlabeled_idx][remaining].astype(str),
                         -tail_priority[remaining])
                    )
                    rescue_local = remaining[rem_order[:n_rescue]]
                if main_local.size + rescue_local.size < k:
                    # Defensive fill (only reachable if the remaining pool is
                    # smaller than the rescue quota): continue the M1 order.
                    taken = set(int(x) for x in main_local) | set(int(x) for x in rescue_local)
                    extra = np.array(
                        [i for i in order_main if int(i) not in taken][: k - main_local.size - rescue_local.size],
                        dtype=int,
                    )
                    rescue_local = np.concatenate([rescue_local, extra])
                batch_local = np.concatenate([main_local, rescue_local])
                scores = np.full(unlabeled_idx.size, np.nan)
                scores[main_local] = p_include[main_local]
                scores[rescue_local] = tail_priority[rescue_local]
                for loc in main_local:
                    role_of_local[int(loc)] = "main"
                for loc in rescue_local:
                    role_of_local[int(loc)] = "rescue"
            else:
                q_input = np.asarray(bundle.q[unlabeled_idx], dtype=float)
                if method_spec.permute_q:
                    rng_q = np.random.default_rng(seed + 7_000_003 + 1009 * iteration)
                    q_input = rng_q.permutation(q_input)
                scores = route_scores(
                    method,
                    p_include if method_feature != "criterion" else np.zeros(unlabeled_idx.size),
                    q_input,
                    bundle.s_ratio[unlabeled_idx],
                    bundle.x_ratio[unlabeled_idx],
                )
        if method_feature != "hybrid":
            batch_local = select_batch(ids[unlabeled_idx], scores, k)
        chosen = unlabeled_idx[batch_local]
        y_batch = labels[chosen]
        labelled[chosen] = True
        checked += chosen.size
        found += int(y_batch.sum())

        for local in batch_local:
            idx = unlabeled_idx[local]
            role = role_of_local.get(int(local), "ranking")
            p_val = float(p_include[local]) if not np.isnan(p_include[local]) else np.nan
            pct = float(p_percentile[local]) if not np.isnan(p_percentile[local]) else np.nan
            codes = reason_codes(
                p_percentile=0.0 if np.isnan(pct) else pct,
                s_ratio=float(bundle.s_ratio[idx]),
                u_ratio=float(bundle.u_ratio[idx]),
                x_ratio=float(bundle.x_ratio[idx]),
                contradicted=bool(bundle.evidence.iloc[idx].get("n_contradict", 0) > 0),
                risk_weight=method_spec.risk_weight,
            )
            if role == "main":
                codes = ["MAIN_QUEUE_SLOT"] + codes
            elif role == "rescue":
                codes = ["RESCUE_SLOT"] + codes
            selected_rows.append(
                {
                    "round": iteration,
                    "record_id": ids[idx],
                    "score": float(scores[local]),
                    "p_include": p_val,
                    "q_score": float(bundle.q[idx]),
                    "s_ratio": float(bundle.s_ratio[idx]),
                    "u_ratio": float(bundle.u_ratio[idx]),
                    "x_ratio": float(bundle.x_ratio[idx]),
                    "reason_codes": "|".join(codes),
                    "revealed_label": int(labels[idx]),
                    "p_percentile": pct,
                    "selection_role": role,
                }
            )
        events.append(
            SimulationEvent(
                run_id="",
                review_id=bundle.review_id,
                method=method,
                seed=seed,
                iteration=iteration,
                selected_record_ids=tuple(ids[chosen]),
                revealed_labels=tuple(int(v) for v in y_batch),
                checked=checked,
                found=found,
                budget_fraction=checked / n,
                recall=found / max(bundle.n_included, 1),
            )
        )

    event_dicts = [
        {
            "iteration": e.iteration,
            "checked": e.checked,
            "found": e.found,
            "budget_fraction": e.budget_fraction,
            "recall": e.recall,
        }
        for e in events
    ]
    curve = curve_from_events(event_dicts, n, bundle.n_included)
    metrics = compute_metrics(
        curve,
        budgets=protocol.budget_fractions,
        wss_target=0.95,
    )
    selected = pd.DataFrame(selected_rows)

    run_id = f"{bundle.review_id}__{method}__seed-{seed:02d}__cfg-{config_hash[:8]}"
    events = [
        SimulationEvent(
            run_id=run_id,
            review_id=e.review_id,
            method=e.method,
            seed=e.seed,
            iteration=e.iteration,
            selected_record_ids=e.selected_record_ids,
            revealed_labels=e.revealed_labels,
            checked=e.checked,
            found=e.found,
            budget_fraction=e.budget_fraction,
            recall=e.recall,
        )
        for e in events
    ]
    meta = {
        "run_id": run_id,
        "review_id": bundle.review_id,
        "method": method,
        "method_family": method_feature,
        "method_description": method_spec.description,
        "seed": seed,
        "config_hash": config_hash,
        "spec_hash": spec_hash,
        "n_records": n,
        "n_included": bundle.n_included,
        "prior_included": method_spec.prior_included,
        "prior_excluded": method_spec.prior_excluded,
        "query_batch_size": protocol.query_batch_size,
        "max_budget_fraction": protocol.max_budget_fraction,
        "max_checked": max_checked,
        "rounds": iteration,
        "relevance_weight": method_spec.relevance_weight,
        "risk_weight": method_spec.risk_weight,
        "relevance_quota": method_spec.relevance_quota,
        "drop_unknown": method_spec.drop_unknown,
        "ranker": ranker.name,
        "ranker_fallback_rounds": int(getattr(ranker, "last_fallback", False)),
        "asreview_version": asreview_version,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "n_features": int(X.shape[1]),
        "unknown_is_exclusion": protocol.unknown_is_exclusion,
        "auto_exclude_records": protocol.auto_exclude_records,
    }
    return RunResult(
        run_id=run_id,
        review_id=bundle.review_id,
        method=method,
        seed=seed,
        n_records=n,
        n_included=bundle.n_included,
        events=events,
        curve=curve_frame(curve),
        metrics=metrics,
        selected=selected,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def write_run(result: RunResult, runs_dir: Path) -> Path:
    run_dir = Path(runs_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    with open(run_dir / "events.jsonl", "w", encoding="utf-8") as fh:
        for e in result.events:
            fh.write(
                json.dumps(
                    {
                        "iteration": e.iteration,
                        "selected_record_ids": list(e.selected_record_ids),
                        "revealed_labels": list(e.revealed_labels),
                        "checked": e.checked,
                        "found": e.found,
                        "budget_fraction": e.budget_fraction,
                        "recall": e.recall,
                    }
                )
                + "\n"
            )
    result.curve.to_csv(run_dir / "curve.csv", index=False)
    result.selected.to_csv(run_dir / "selected_records.csv", index=False)
    summary = {
        **result.meta,
        **{k: (None if v is None else v) for k, v in result.metrics.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir
