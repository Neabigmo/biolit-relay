"""Parallel experiment runner shared by scripts 03/04.

One task = one (review, method, seed) simulation.  Every task writes an
immutable run directory under ``outputs/runs/``; the aggregated
``outputs/runs/index.csv`` is rebuilt from the per-run ``summary.json`` files so
that interrupted runs can be resumed safely and failures stay visible.
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biolit_relay.core import ExperimentProtocol  # noqa: E402
from biolit_relay.metrics import compute_metrics, curve_from_events  # noqa: E402
from biolit_relay.simulation import (  # noqa: E402
    ReviewBundle,
    run_simulation,
    write_run,
)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def protocol_from_config(config: dict) -> ExperimentProtocol:
    p = config["protocol"]
    s = config["scoring"]
    return ExperimentProtocol(
        prior_included=int(p["prior_included"]),
        prior_excluded=int(p["prior_excluded"]),
        query_batch_size=int(p["query_batch_size"]),
        budget_fractions=tuple(p["budget_fractions"]),
        max_budget_fraction=float(p.get("max_budget_fraction", max(p["budget_fractions"]))),
        main_seeds=int(p["main_seeds"]),
        random_baseline_seeds=int(p["random_baseline_seeds"]),
        relevance_weight=float(s["relevance_weight"]),
        false_negative_risk_weight=float(s["false_negative_risk_weight"]),
        criterion_support_weight=float(s["criterion_support_weight"]),
        criterion_unknown_weight=float(s["criterion_unknown_weight"]),
        criterion_mixed_weight=float(s["criterion_mixed_weight"]),
    )


def config_hash(config: dict) -> str:
    import hashlib

    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@functools.lru_cache(maxsize=4)
def load_bundle(review_id: str, root_str: str, evidence_path_str: str) -> ReviewBundle:
    root = Path(root_str)
    records = pd.read_csv(root / "data" / "processed" / f"{review_id}.csv")
    oracle = pd.read_csv(root / "data" / "oracle" / f"{review_id}_oracle.csv")
    labels = oracle.set_index("record_id").loc[records["record_id"], "label_included"].to_numpy()
    evidence = pd.read_csv(evidence_path_str)
    evidence = evidence.loc[evidence["review_id"] == review_id].reset_index(drop=True)
    return ReviewBundle(
        review_id=review_id,
        records=records,
        evidence=evidence,
        labels=np.asarray(labels, dtype=int),
    )


def _evidence_path(root: Path) -> Path:
    return root / "data" / "processed" / "evidence_store.csv"


def build_tasks(config: dict, methods: Iterable[str] | None = None,
                reviews: Iterable[str] | None = None,
                seeds: Iterable[int] | None = None) -> list[dict]:
    protocol = protocol_from_config(config)
    reviews = list(reviews or config["data"]["reviews"])
    main_methods = list(config["methods"]["main"])
    ablations = list(config["methods"].get("ablations", []))
    low_prior_methods = list(config["methods"].get("low_prior", []))
    hybrid_methods = list(config["methods"].get("hybrid", []))
    control_methods = list(config["methods"].get("control", []))
    selected = (
        list(methods)
        if methods
        else main_methods + ablations + low_prior_methods + hybrid_methods + control_methods
    )

    tasks: list[dict] = []
    for review_id in reviews:
        for method in selected:
            if method == "random":
                seed_list: Iterable[int] = range(protocol.random_baseline_seeds)
            else:
                seed_list = range(protocol.main_seeds)
            if seeds is not None:
                seed_list = [s for s in seed_list if s in set(seeds)]
            for seed in seed_list:
                tasks.append({"review_id": review_id, "method": method, "seed": int(seed)})
    return tasks


def _method_kind(method: str, main_methods, ablations, low_prior_methods) -> str:
    if method == "random":
        return "random"
    if method in low_prior_methods or method.endswith("_prior2"):
        return "low_prior"
    if method in main_methods or method in ablations or method.startswith("hybrid_relay") or method.startswith("biolit_relay_placebo"):
        return "main"
    raise KeyError(f"unknown method {method!r} in config")


def _run_one(task: dict, config: dict, chash: str, root_str: str, runs_dir: str,
             spec_hashes: dict, asreview_version: str) -> dict:
    root = Path(root_str)
    protocol = protocol_from_config(config)
    started = time.perf_counter()
    review_id = task["review_id"]
    method = task["method"]
    seed = int(task["seed"])
    run_id = f"{review_id}__{method}__seed-{seed:02d}__cfg-{chash[:8]}"
    run_dir = Path(runs_dir) / run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("status") == "ok":
                return {"run_id": run_id, "status": "cached", "review_id": review_id,
                        "method": method, "seed": seed}
        except json.JSONDecodeError:
            pass
    try:
        bundle = load_bundle(review_id, root_str, str(_evidence_path(root)))
        result = run_simulation(
            bundle,
            method=method,
            seed=seed,
            protocol=protocol,
            config_hash=chash,
            spec_hash=spec_hashes.get(review_id, ""),
            asreview_version=asreview_version,
        )
        result.meta["status"] = "ok"
        write_run(result, Path(runs_dir))
        # overwrite summary with status and runtime
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["status"] = "ok"
        summary["runner_runtime_s"] = round(time.perf_counter() - started, 3)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"run_id": run_id, "status": "ok", "review_id": review_id,
                "method": method, "seed": seed, "runtime_s": summary["runner_runtime_s"]}
    except Exception as exc:  # noqa: BLE001 - failures must be recorded, never hidden
        run_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "run_id": run_id,
            "status": "failed",
            "review_id": review_id,
            "method": method,
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (run_dir / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return failure


def run_tasks(tasks: list[dict], config: dict, n_jobs: int | None = None,
              root: Path | None = None) -> pd.DataFrame:
    root = root or ROOT
    runs_dir = root / config["outputs"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    chash = config_hash(config)
    spec_hashes = _load_spec_hashes(root)
    asreview_version = _asreview_version(config)
    n_jobs = n_jobs or max(1, min(24, (os.cpu_count() or 4) - 4))

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(
        delayed(_run_one)(task, config, chash, str(root), str(runs_dir), spec_hashes,
                          asreview_version)
        for task in tasks
    )
    frame = pd.DataFrame(results)
    failed = frame[frame["status"] == "failed"]
    if len(failed):
        print(f"WARNING: {len(failed)} failed runs recorded in outputs/runs/", file=sys.stderr)
        for _, row in failed.iterrows():
            print(f"  FAILED {row['run_id']}: {row.get('error')}", file=sys.stderr)
    rebuild_index(root, config)
    return frame


def _load_spec_hashes(root: Path) -> dict:
    path = root / "data" / "manifests" / "spec_freeze.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["review_id"]: r.get("spec_hash", "") for r in data.get("reviews", [])}


def _asreview_version(config: dict | None = None) -> str:
    if config is not None:
        cfg = config.get("asreview_default", {})
        if cfg.get("reference_version"):
            return f"{cfg['reference_version']} ({cfg.get('model_preset', 'default')} preset, re-instantiated)"
    try:
        import asreview

        return getattr(asreview, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "not-installed"


def rebuild_index(root: Path | None = None, config: dict | None = None) -> pd.DataFrame:
    """Aggregate every per-run summary.json into outputs/runs/index.csv."""
    root = root or ROOT
    runs_dir = root / "outputs" / "runs"
    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        try:
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"run_id": summary_path.parent.name, "status": "corrupt"})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    preferred = [
        "run_id", "review_id", "method", "seed", "status", "n_records", "n_included",
        "prior_included", "prior_excluded", "query_batch_size", "rounds", "runtime_s",
        "runner_runtime_s", "recall_05", "recall_10", "recall_20", "recall_30",
        "aubrc_0_30", "aubrc_0_10", "precision_05", "precision_10", "precision_20",
        "precision_30", "n_found_05", "n_found_10", "n_found_20", "n_found_30",
        "wss_95", "wss_95_reached", "max_checked_fraction", "recall_at_max_budget",
        "config_hash", "spec_hash", "ranker", "relevance_weight", "risk_weight",
        "drop_unknown", "asreview_version", "n_features", "error",
    ]
    cols = [c for c in preferred if c in frame.columns] + [
        c for c in frame.columns if c not in preferred
    ]
    frame = frame[cols].sort_values(["review_id", "method", "seed"]).reset_index(drop=True)
    out = root / "outputs" / "runs" / "index.csv"
    frame.to_csv(out, index=False)
    ok = frame[frame["status"] == "ok"]
    print(
        f"index.csv: {len(frame)} runs ({len(ok)} ok, "
        f"{len(frame) - len(ok)} failed/cached) -> {out}",
        flush=True,
    )
    return frame


def failed_run_report(root: Path | None = None) -> pd.DataFrame:
    root = root or ROOT
    index = root / "outputs" / "runs" / "index.csv"
    if not index.exists():
        return pd.DataFrame()
    frame = pd.read_csv(index)
    return frame[frame["status"] != "ok"]

