"""Stage 02 - freeze review specifications and build the criterion evidence store.

Usage:
    python scripts/02_build_specs.py [--config configs/experiment.yaml]

Outputs:
    data/manifests/spec_freeze.json          rule/spec version hashes (frozen)
    data/processed/evidence_store.csv        per-record S/U/X/Q terms (label-free)
    data/processed/evidence_details.csv      per-(record, criterion) auditable evidence

The evidence engine is label-blind by construction: it only receives title and
abstract text plus the published eligibility criteria in configs/review_specs.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.criterion_evidence import (  # noqa: E402
    CriterionEvidenceEngine,
    load_lexicon,
    load_review_specs,
)


def _spec_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--reviews", nargs="*", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reviews = list(args.reviews or config["data"]["reviews"])
    runs_dir = ROOT / config["outputs"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    lexicon = load_lexicon(ROOT / "configs" / "rule_lexicon.yaml")
    specs = load_review_specs(ROOT / "configs" / "review_specs.yaml")
    missing = [r for r in reviews if r not in specs]
    if missing:
        print(f"ERROR: no review spec for {missing}", file=sys.stderr)
        return 2

    store_rows: list[pd.DataFrame] = []
    detail_rows: list[pd.DataFrame] = []
    freeze = {"rule_lexicon_sha256": _spec_hash(lexicon), "reviews": []}
    for review_id in reviews:
        spec = specs[review_id]
        records = pd.read_csv(ROOT / "data" / "processed" / f"{review_id}.csv")
        engine = CriterionEvidenceEngine(spec, lexicon)
        evidence_df, details = engine.evaluate_frame(records)
        evidence_df = evidence_df.copy()
        evidence_df.insert(0, "review_id", review_id)
        store_rows.append(evidence_df)

        rows = []
        for record_id, record_evidence in details.items():
            for hit in record_evidence.hits:
                rows.append(
                    {
                        "review_id": review_id,
                        "record_id": record_id,
                        "criterion_id": hit.criterion_id,
                        "criterion_name": spec.criterion(hit.criterion_id).name,
                        "automation_level": spec.criterion(hit.criterion_id).automation_level,
                        "state": hit.state,
                        "matched_phrase": hit.matched_phrase,
                        "evidence_sentence": hit.evidence_sentence,
                        "start_char": hit.start_char,
                        "end_char": hit.end_char,
                        "rule_id": hit.rule_id,
                        "confidence_level": hit.confidence_level,
                    }
                )
        detail_rows.append(pd.DataFrame(rows))

        freeze["reviews"].append(
            {
                "review_id": review_id,
                "topic": spec.topic,
                "source_reference": spec.source_reference,
                "rule_version": spec.rule_version,
                "eligibility_text_sha256": hashlib.sha256(
                    spec.eligibility_text.encode("utf-8")
                ).hexdigest(),
                "criteria": [
                    {
                        "criterion_id": c.criterion_id,
                        "name": c.name,
                        "required": c.required,
                        "automation_level": c.automation_level,
                        "support_cues": list(c.support_cues),
                        "contradict_cues": list(c.contradict_cues),
                    }
                    for c in spec.criteria
                ],
                "spec_hash": _spec_hash(
                    {
                        "review_id": spec.review_id,
                        "eligibility_text": spec.eligibility_text,
                        "criteria": [
                            {
                                "criterion_id": c.criterion_id,
                                "required": c.required,
                                "automation_level": c.automation_level,
                                "support_cues": list(c.support_cues),
                                "contradict_cues": list(c.contradict_cues),
                            }
                            for c in spec.criteria
                        ],
                        "rule_version": spec.rule_version,
                    }
                ),
            }
        )
        n_states = evidence_df[["n_support", "n_contradict", "n_unknown", "n_mixed"]].sum()
        print(
            f"{review_id}: n={len(evidence_df)} mean_q={evidence_df['q_score'].mean():.3f} "
            f"states(S/X/U/M)={int(n_states['n_support'])}/"
            f"{int(n_states['n_contradict'])}/{int(n_states['n_unknown'])}/"
            f"{int(n_states['n_mixed'])}",
            flush=True,
        )

    store = pd.concat(store_rows, ignore_index=True)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()

    # label-leak check: evidence store must contain no label columns
    bad_cols = [c for c in store.columns if "label" in c.lower()]
    if bad_cols:
        print(f"ERROR: label columns leaked into evidence store: {bad_cols}", file=sys.stderr)
        return 3
    store.to_csv(ROOT / "data" / "processed" / "evidence_store.csv", index=False)
    details.to_csv(ROOT / "data" / "processed" / "evidence_details.csv", index=False)
    freeze_path = ROOT / "data" / "manifests" / "spec_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(
        f"wrote evidence_store.csv ({len(store)} rows), "
        f"evidence_details.csv ({len(details)} rows), {freeze_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
