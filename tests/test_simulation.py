"""Simulation protocol tests: determinism, shared priors, budget accounting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.core import ExperimentProtocol  # noqa: E402
from biolit_relay.risk_router import METHODS, select_batch  # noqa: E402
from biolit_relay.simulation import ReviewBundle, run_simulation  # noqa: E402


def bundle(n: int = 200, n_included: int = 20, seed: int = 0) -> ReviewBundle:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_included, replace=False)] = 1
    records = pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in range(n)],
            "title": [f"title {i}" for i in range(n)],
            "abstract": ["randomized trial " + str(i) for i in range(n)],
            "has_abstract": 1,
        }
    )
    evidence = pd.DataFrame(
        {
            "record_id": records["record_id"],
            "q_score": rng.random(n),
            "s_ratio": rng.random(n),
            "u_ratio": rng.random(n),
            "x_ratio": np.zeros(n),
            "n_contradict": np.zeros(n, dtype=int),
        }
    )
    return ReviewBundle("synthetic", records, evidence, labels)


def test_same_seed_same_selection_sequence() -> None:
    b = bundle()
    protocol = ExperimentProtocol()
    r1 = run_simulation(b, "biolit_relay", seed=3, protocol=protocol)
    r2 = run_simulation(b, "biolit_relay", seed=3, protocol=protocol)
    assert list(r1.selected["record_id"]) == list(r2.selected["record_id"])
    assert r1.metrics == r2.metrics


def test_different_seed_different_priors() -> None:
    b = bundle()
    protocol = ExperimentProtocol()
    r1 = run_simulation(b, "random", seed=1, protocol=protocol)
    r2 = run_simulation(b, "random", seed=2, protocol=protocol)
    p1 = set(r1.selected[r1.selected["round"] == 0]["record_id"])
    p2 = set(r2.selected[r2.selected["round"] == 0]["record_id"])
    assert p1 != p2


def test_shared_priors_across_methods() -> None:
    b = bundle()
    protocol = ExperimentProtocol()
    priors = {}
    for method in ("random", "matched_tfidf_logreg", "criterion_only", "biolit_relay"):
        r = run_simulation(b, method=method, seed=5, protocol=protocol)
        priors[method] = list(r.selected[r.selected["round"] == 0]["record_id"])
    assert len({tuple(v) for v in priors.values()}) == 1


def test_budget_counts_priors_and_caps_at_30pct() -> None:
    b = bundle(n=200)
    protocol = ExperimentProtocol()
    for method in ("random", "matched_tfidf_logreg", "biolit_relay"):
        r = run_simulation(b, method=method, seed=0, protocol=protocol)
        max_checked = int(np.ceil(0.30 * 200))
        assert r.meta["max_checked"] == max_checked
        assert len(r.selected) == max_checked  # priors included, exact batch multiples
        # curve starts at the prior reveal and ends at the budget cap
        assert r.curve["checked"].iloc[0] == protocol.prior_included + protocol.prior_excluded
        assert r.curve["checked"].iloc[-1] == max_checked
        # monotone curve
        assert r.curve["recall"].is_monotonic_increasing


def test_ties_broken_by_record_id() -> None:
    ids = np.array(["R0003", "R0001", "R0002", "R0000"])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    chosen = select_batch(ids, scores, 3)
    assert list(ids[chosen]) == ["R0000", "R0001", "R0002"]


def test_criterion_only_needs_no_relevance_model() -> None:
    b = bundle()
    r = run_simulation(b, "criterion_only", seed=0, protocol=ExperimentProtocol())
    non_prior = r.selected[r.selected["round"] > 0]
    assert non_prior["p_include"].isna().all()
    assert non_prior["score"].notna().all()


def test_method_registry_covers_config_methods() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    declared = (
        list(config["methods"]["main"])
        + list(config["methods"].get("ablations", []))
        + list(config["methods"].get("low_prior", []))
    )
    for method in declared:
        assert method in METHODS, method
