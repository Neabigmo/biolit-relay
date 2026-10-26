"""Budgeted screening metrics (design doc 05, section 4).

All metrics are computed from an append-only screening curve:
``checked`` = number of records that have been *seen* so far (including the
initial prior records) and ``found`` = number of included records among them.

The single implementation below is used for every method and every budget point
so that arithmetic cannot differ across baselines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class Curve:
    checked: np.ndarray  # int, includes priors
    found: np.ndarray  # int
    n_records: int
    n_included: int

    @property
    def x(self) -> np.ndarray:
        return self.checked / float(self.n_records)

    @property
    def recall(self) -> np.ndarray:
        return self.found / float(max(self.n_included, 1))

    @property
    def precision(self) -> np.ndarray:
        return self.found / np.maximum(self.checked, 1)

    def at(self, budget_fraction: float) -> tuple[float, float]:
        """Linear interpolation of (recall, precision) at a budget fraction.

        The curve implicitly starts at (0, 0); recall before the first revealed
        batch is interpolated between zero and the prior reveal.
        """
        x = np.concatenate(([0.0], self.x))
        recall = np.concatenate(([0.0], self.recall))
        found = np.concatenate(([0.0], self.found))
        if budget_fraction >= x[-1]:
            return float(recall[-1]), float(self.precision[-1])
        r = float(np.interp(budget_fraction, x, recall))
        found_at = float(np.interp(budget_fraction, x, found))
        checked = budget_fraction * self.n_records
        return r, found_at / max(checked, 1.0)


def curve_from_events(events: list[dict], n_records: int, n_included: int) -> Curve:
    """Build the screening curve.  Event 0 is the initial prior reveal."""
    checked = np.array([int(e["checked"]) for e in events], dtype=int)
    found = np.array([int(e["found"]) for e in events], dtype=int)
    if len(checked) == 0:
        raise ValueError("empty event list")
    if np.any(np.diff(checked) < 0) or np.any(np.diff(found) < 0):
        raise ValueError("screening curve must be monotone")
    return Curve(checked=checked, found=found, n_records=int(n_records), n_included=int(n_included))


def _curve_points_for_integration(curve: Curve, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(([0.0], curve.x))
    y = np.concatenate(([0.0], curve.recall))
    hi = min(hi, float(x[-1]))
    if hi <= lo:
        return np.array([lo, hi]), np.array([0.0, 0.0])
    inside = (x > lo) & (x < hi)
    xs = np.concatenate(([lo], x[inside], [hi]))
    ys = np.interp(xs, x, y)
    return xs, ys


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapezoid with a fallback for older NumPy versions."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def aubrc(curve: Curve, lo: float = 0.0, hi: float = 0.30) -> float:
    """Area under the recall-budget curve on [lo, hi], normalised by (hi - lo).

    Linear interpolation between batch boundaries, trapezoidal integration.
    Returns a number in [0, 1]; 0.30 means "30% budget yields 30% of included
    records" (i.e. proportional), >0.30 means better than proportional.
    """
    xs, ys = _curve_points_for_integration(curve, lo, hi)
    if xs[-1] <= lo:
        return float("nan")
    area = _trapezoid(ys, xs)
    return area / (xs[-1] - lo)


def wss_at_95(curve: Curve, target: float = 0.95) -> float | None:
    """Work saved over sampling at ``target`` recall.

    Returns ``None`` when the target recall is not reached inside the simulated
    budget (never extrapolated, per protocol).
    """
    idx = int(np.argmax(curve.recall >= target)) if np.any(curve.recall >= target) else -1
    if idx < 0:
        return None
    x = float(curve.x[idx])
    return float(1.0 - x)


def compute_metrics(
    curve: Curve,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    wss_target: float = 0.95,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "n_records": curve.n_records,
        "n_included": curve.n_included,
        "aubrc_0_30": aubrc(curve, 0.0, 0.30),
        "aubrc_0_10": aubrc(curve, 0.0, 0.10),
    }
    for b in budgets:
        tag = f"{int(round(b * 100)):02d}"
        r, p = curve.at(b)
        out[f"recall_{tag}"] = r
        out[f"precision_{tag}"] = p
        out[f"n_found_{tag}"] = r * curve.n_included
    ws = wss_at_95(curve, wss_target)
    out["wss_95"] = ws
    out["wss_95_reached"] = 0 if ws is None else 1
    out["max_checked_fraction"] = float(curve.x[-1])
    out["recall_at_max_budget"] = float(curve.recall[-1])
    return out


def curve_frame(curve: Curve) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "checked": curve.checked,
            "checked_fraction": curve.x,
            "found": curve.found,
            "recall": curve.recall,
            "precision": curve.precision,
            "budget": curve.checked / float(curve.n_records),
        }
    )


# ---------------------------------------------------------------------------
# statistics helpers used by the analysis layer
# ---------------------------------------------------------------------------


def paired_differences(
    frame: pd.DataFrame,
    metric: str,
    method: str,
    baseline: str,
    review_col: str = "review_id",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Long frame of paired per-(review, seed) differences ``method - baseline``."""
    piv = frame.pivot_table(index=[review_col, seed_col], columns="method", values=metric)
    if method not in piv.columns or baseline not in piv.columns:
        raise KeyError(f"missing method column: {method!r} / {baseline!r}")
    delta = (piv[method] - piv[baseline]).dropna().rename("delta").reset_index()
    return delta.sort_values([review_col, seed_col]).reset_index(drop=True)


def paired_summary(delta: pd.DataFrame, review_col: str = "review_id") -> pd.DataFrame:
    """Per-review mean/median/SD of the paired differences."""
    return (
        delta.groupby(review_col)["delta"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )


def hierarchical_bootstrap_ci(
    delta: pd.DataFrame,
    review_col: str = "review_id",
    n_resamples: int = 5000,
    confidence: float = 0.95,
    rng_seed: int = 20261026,
) -> dict[str, float]:
    """Bootstrap the macro-average paired difference.

    ``delta`` must be the *long* frame from :func:`paired_differences` with
    columns ``review_id``, ``seed`` and ``delta``.  Reviews are resampled with
    replacement, then seeds within each resampled review.
    """
    if "delta" not in delta.columns:
        raise KeyError("hierarchical_bootstrap_ci expects the long paired-difference frame")
    rng = np.random.default_rng(rng_seed)
    reviews = sorted(delta[review_col].unique())
    groups = {r: delta.loc[delta[review_col] == r, "delta"].to_numpy() for r in reviews}
    point = float(np.mean([groups[r].mean() for r in reviews]))
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen = rng.choice(len(reviews), size=len(reviews), replace=True)
        vals = []
        for j in chosen:
            arr = groups[reviews[j]]
            if arr.size == 0:
                continue
            vals.append(arr[rng.integers(0, arr.size, size=arr.size)].mean())
        boot[i] = float(np.mean(vals)) if vals else np.nan
    boot = boot[~np.isnan(boot)]
    alpha = 1.0 - confidence
    return {
        "point": point,
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "se": float(np.std(boot, ddof=1)),
        "n_resamples": int(boot.size),
    }
