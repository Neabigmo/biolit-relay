"""Stage 05 - aggregate run summaries, check completeness, and build result tables.

This is the first-pass evaluation used as a sanity gate before the dedicated
analysis layer (scripts 07/08/09) produces paper numbers.

Usage:
    python scripts/05_evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.metrics import (  # noqa: E402
    hierarchical_bootstrap_ci,
    paired_differences,
    paired_summary,
)
from biolit_relay.runner import rebuild_index  # noqa: E402

METRICS = [
    "aubrc_0_30",
    "recall_05",
    "recall_10",
    "recall_20",
    "recall_30",
    "precision_10",
    "precision_20",
]


def macro_average(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average per review first, then macro-average across reviews."""
    per_review = (
        frame.groupby(["review_id", "method"])[metric].mean().reset_index()
    )
    macro = (
        per_review.groupby("method")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "macro_mean", "std": "macro_sd", "count": "n_reviews"})
    )
    return per_review, macro


def main() -> int:
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    frame = rebuild_index(ROOT)
    if frame.empty:
        print("ERROR: no runs found", file=sys.stderr)
        return 2

    ok = frame[frame["status"] == "ok"].copy()
    failed = frame[frame["status"] != "ok"].copy()
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # completeness matrix
    reviews = list(config["data"]["reviews"])
    main_methods = list(config["methods"]["main"]) + list(config["methods"].get("ablations", [])) + list(
        config["methods"].get("low_prior", [])
    )
    counts = (
        ok.groupby(["review_id", "method"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=reviews, columns=main_methods, fill_value=0)
    )
    counts.to_csv(tables / "run_counts.csv")

    per_review_rows = []
    macro_rows = []
    for metric in METRICS:
        if metric not in ok.columns:
            continue
        pr, macro = macro_average(ok, metric)
        pr["metric"] = metric
        macro["metric"] = metric
        per_review_rows.append(pr)
        macro_rows.append(macro)
    per_review = pd.concat(per_review_rows, ignore_index=True)
    macro = pd.concat(macro_rows, ignore_index=True)
    per_review.to_csv(tables / "per_review_metrics.csv", index=False)
    macro.to_csv(tables / "macro_metrics.csv", index=False)

    # paired comparisons: M5 vs M1 and M4 vs M1
    pair_rows = []
    for metric in ["aubrc_0_30", "recall_05", "recall_10", "recall_20", "recall_30"]:
        for method, baseline in [("biolit_relay", "matched_tfidf_logreg"),
                                 ("asreview_default", "matched_tfidf_logreg")]:
            try:
                raw = paired_differences(ok, metric, method, baseline)
                summary = paired_summary(raw)
                ci = hierarchical_bootstrap_ci(raw, rng_seed=20261026)
            except (KeyError, ValueError) as exc:
                print(f"skip {metric} {method}: {exc}", file=sys.stderr)
                continue
            pair_rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "baseline": baseline,
                    "macro_delta": ci["point"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n_reviews": len(summary),
                    "n_reviews_positive": int((summary["mean"] > 0).sum()),
                    "per_review": "; ".join(
                        f"{r}:{v:+.4f}"
                        for r, v in zip(summary["review_id"], summary["mean"])
                    ),
                }
            )
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(tables / "paired_deltas.csv", index=False)

    # markdown summary
    lines = ["# BioLit-Relay first-pass results (script 05)", ""]
    lines.append(f"Runs: {len(frame)} total, {len(ok)} ok, {len(failed)} failed/cached-issue.")
    if len(failed):
        lines.append("\n## Failed runs (never dropped silently)\n")
        lines.append(failed[["run_id", "review_id", "method", "seed", "error"]].to_markdown(index=False))
    lines.append("\n## Macro-average AUBRC_0-30 by method\n")
    aub = macro[macro["metric"] == "aubrc_0_30"].sort_values("macro_mean", ascending=False)
    lines.append(aub[["method", "macro_mean", "macro_sd", "n_reviews"]].to_markdown(index=False))
    lines.append("\n## Paired differences vs matched active learning\n")
    lines.append(paired.to_markdown(index=False) if len(paired) else "not available")
    (tables / "main_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
