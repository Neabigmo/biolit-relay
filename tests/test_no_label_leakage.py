"""Label-leakage guards (design doc 04 section 6, doc 05 section 8).

The gold label may only live in the oracle store.  These tests fail loudly if a
label column ever reaches a model-visible artifact or if the simulator reveals a
label before the record was selected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.data_adapter import FORBIDDEN_INPUT_COLUMNS, assert_no_labels  # noqa: E402
from biolit_relay.simulation import ReviewBundle, run_simulation  # noqa: E402
from biolit_relay.core import ExperimentProtocol  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ORACLE = ROOT / "data" / "oracle"


def _synthetic_bundle(n: int = 120, n_included: int = 12, seed: int = 0) -> ReviewBundle:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=n_included, replace=False)] = 1
    records = pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in range(n)],
            "title": [f"study title {i}" for i in range(n)],
            "abstract": ["randomized controlled trial of therapy " + str(i) for i in range(n)],
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
    return ReviewBundle(review_id="synthetic", records=records, evidence=evidence, labels=labels)


def test_processed_data_has_no_label_column() -> None:
    if not any(PROCESSED.glob("*.csv")):
        pytest.skip("processed data not built yet")
    for path in PROCESSED.glob("*.csv"):
        if path.name == "evidence_store.csv":
            continue
        frame = pd.read_csv(path, nrows=1)
        assert_no_labels(frame, str(path))


def test_evidence_store_has_no_label_column() -> None:
    path = PROCESSED / "evidence_store.csv"
    if not path.exists():
        pytest.skip("evidence store not built yet")
    frame = pd.read_csv(path, nrows=1)
    assert_no_labels(frame, str(path))


def test_oracle_is_separate() -> None:
    if not ORACLE.exists():
        pytest.skip("oracle not built yet")
    for path in ORACLE.glob("*_oracle.csv"):
        frame = pd.read_csv(path, usecols=["record_id", "label_included"])
        review = path.name.replace("_oracle.csv", "")
        processed = PROCESSED / f"{review}.csv"
        assert processed.exists()
        records = pd.read_csv(processed)
        assert "label_included" not in records.columns
        assert set(records["record_id"]) == set(frame["record_id"])


@pytest.mark.parametrize("method", ["random", "matched_tfidf_logreg", "biolit_relay"])
def test_labels_revealed_only_after_selection(method: str) -> None:
    bundle = _synthetic_bundle()
    protocol = ExperimentProtocol(main_seeds=1)
    result = run_simulation(bundle, method=method, seed=0, protocol=protocol)
    n_priors = protocol.prior_included + protocol.prior_excluded
    selected = result.selected
    priors = selected[selected["round"] == 0]
    assert len(priors) == n_priors
    # priors have no model score (they were given, not ranked)
    assert priors["score"].isna().all()
    # every revealed label equals the oracle label for that record and every
    # record appears at most once
    assert selected["record_id"].is_unique
    for _, row in selected.iterrows():
        idx = int(row["record_id"][1:])
        assert int(row["revealed_label"]) == int(bundle.labels[idx])
    # the run must never include records beyond the budget cap
    assert result.meta["max_checked"] >= selected.shape[0]


def test_run_json_has_no_oracle_content(tmp_path: Path) -> None:
    bundle = _synthetic_bundle()
    result = run_simulation(bundle, method="biolit_relay", seed=0,
                            protocol=ExperimentProtocol())
    blob = json.dumps(result.meta)
    assert "label_included" not in blob
    assert "revealed" not in blob
