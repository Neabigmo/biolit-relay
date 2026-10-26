"""Unit tests for the metric implementation (design doc 06 section 7)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.metrics import (  # noqa: E402
    Curve,
    aubrc,
    compute_metrics,
    curve_frame,
    curve_from_events,
    hierarchical_bootstrap_ci,
    paired_differences,
    paired_summary,
    wss_at_95,
)


def make_curve() -> Curve:
    # 100 records, 10 included; 5 included in the first 10 records, 10 by 30 records
    return Curve(
        checked=np.array([10, 20, 30]),
        found=np.array([5, 6, 10]),
        n_records=100,
        n_included=10,
    )


def test_recall_at_budget_interpolates() -> None:
    curve = make_curve()
    r, p = curve.at(0.10)
    assert r == pytest.approx(0.5)
    assert p == pytest.approx(0.5)
    r20, _ = curve.at(0.20)
    assert r20 == pytest.approx(0.6)  # exact batch boundary
    r15, _ = curve.at(0.15)
    assert r15 == pytest.approx(0.55)  # halfway between 0.5 (at 0.1) and 0.6 (at 0.2)
    r5, _ = curve.at(0.05)
    assert r5 == pytest.approx(0.25)  # interpolated from the implicit (0, 0) point


def test_aubrc_analytic() -> None:
    # points: (0,0) (0.1,0.5) (0.2,0.6) (0.3,1.0)
    curve = make_curve()
    xs = np.array([0.0, 0.1, 0.2, 0.3])
    ys = np.array([0.0, 0.5, 0.6, 1.0])
    expected = np.trapezoid(ys, xs) / 0.3
    assert aubrc(curve, 0.0, 0.30) == pytest.approx(expected)
    # a curve with recall(x) = x has AUBRC = 0.15 on [0, 0.3]
    linear = Curve(checked=np.array([50]), found=np.array([5]), n_records=100, n_included=10)
    assert aubrc(linear) == pytest.approx(0.15)


def test_wss_at_95_not_reached() -> None:
    curve = Curve(checked=np.array([10, 20, 30]), found=np.array([5, 6, 7]),
                  n_records=100, n_included=10)
    assert wss_at_95(curve, 0.95) is None


def test_wss_at_95_value() -> None:
    curve = Curve(checked=np.array([10, 20]), found=np.array([5, 10]), n_records=100, n_included=10)
    assert wss_at_95(curve, 0.95) == pytest.approx(0.8)


def test_compute_metrics_keys() -> None:
    metrics = compute_metrics(make_curve())
    for key in ("aubrc_0_30", "recall_05", "recall_10", "recall_20", "recall_30",
                "precision_10", "wss_95", "wss_95_reached", "max_checked_fraction"):
        assert key in metrics
    assert metrics["wss_95_reached"] == 1  # 100% recall is reached at 30% budget
    not_reached = compute_metrics(
        Curve(checked=np.array([10, 20, 30]), found=np.array([1, 2, 3]),
              n_records=100, n_included=10)
    )
    assert not_reached["wss_95_reached"] == 0
    assert not_reached["wss_95"] is None


def test_curve_from_events_rejects_non_monotone() -> None:
    with pytest.raises(ValueError):
        curve_from_events(
            [{"checked": 10, "found": 5}, {"checked": 5, "found": 6}], 100, 10
        )


def test_curve_frame_columns() -> None:
    frame = curve_frame(make_curve())
    assert list(frame.columns) == [
        "checked", "checked_fraction", "found", "recall", "precision", "budget"
    ]
    assert frame["recall"].iloc[-1] == pytest.approx(1.0)


def test_paired_differences_and_bootstrap() -> None:
    import pandas as pd

    rows = []
    for review in ("A", "B"):
        for seed in range(5):
            rows.append({"review_id": review, "seed": seed, "method": "bio",
                         "aubrc": 0.5 + 0.1 * (review == "A") + 0.01 * seed})
            rows.append({"review_id": review, "seed": seed, "method": "base",
                         "aubrc": 0.4 + 0.1 * (review == "A") + 0.01 * seed})
    frame = pd.DataFrame(rows)
    raw = paired_differences(frame, "aubrc", "bio", "base")
    assert set(raw.columns) == {"review_id", "seed", "delta"}
    summary = paired_summary(raw)
    assert set(summary["review_id"]) == {"A", "B"}
    assert np.allclose(summary["mean"], 0.1)
    ci = hierarchical_bootstrap_ci(raw, n_resamples=200)
    assert ci["point"] == pytest.approx(0.1, abs=1e-9)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
