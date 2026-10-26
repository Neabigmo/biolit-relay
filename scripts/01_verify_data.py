"""Stage 01 - verify the pinned SYNERGY data and build label-free processed files.

Usage:
    python scripts/01_verify_data.py [--config configs/experiment.yaml]

Outputs:
    data/processed/<review_id>.csv           label-free model input
    data/oracle/<review_id>_oracle.csv       labels (simulation oracle only)
    data/manifests/data_quality.json         counts, missing/duplicate report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.data_adapter import REVIEW_IDS, build_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reviews = config.get("reviews") or list(REVIEW_IDS)
    if reviews != list(REVIEW_IDS):
        print(f"WARNING: config reviews {reviews} differ from adapter REVIEW_IDS {REVIEW_IDS}")

    raw_root = ROOT / "data" / "raw" / "synergy-v1.0"
    manifest_path = ROOT / "data" / "manifests" / "raw_files_manifest.json"
    if not manifest_path.exists():
        print("ERROR: raw manifest missing; run scripts/00_fetch_synergy.py first", file=sys.stderr)
        return 2

    quality = build_dataset(
        raw_root=raw_root,
        out_dir=ROOT / "data" / "processed",
        oracle_dir=ROOT / "data" / "oracle",
        manifest_path=manifest_path,
    )

    # hard gate: every review must exist with labels and a title or abstract
    problems = []
    for review_id in REVIEW_IDS:
        q = quality["reviews"][review_id]
        proc = pd.read_csv(ROOT / "data" / "processed" / f"{review_id}.csv")
        if "label_included" in proc.columns:
            problems.append(f"{review_id}: label column leaked into processed data")
        if q["n_records"] == 0:
            problems.append(f"{review_id}: empty dataset")
        if q["n_included"] < 10:
            problems.append(f"{review_id}: too few included records")
        if len(proc) != q["n_records"]:
            problems.append(f"{review_id}: processed row count mismatch")

    if problems:
        print("DATA GATE FAILED:", file=sys.stderr)
        for p in problems:
            print(" -", p, file=sys.stderr)
        return 1
    print(
        "DATA GATE PASSED: "
        f"{quality['totals']['n_records']} records, "
        f"{quality['totals']['n_included']} included across {len(REVIEW_IDS)} reviews"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
