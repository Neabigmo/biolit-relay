"""Data adapter: SYNERGY v1.0 -> label-free processed records + separate oracle.

Design rules enforced here (see BioLit-Relay design pack, doc 02):

* records keep *every* benchmark row, including rows with missing title/abstract;
* ``label_included`` is written to a separate oracle file and never appears in
  the model-visible frame;
* ``record_id`` is a stable function of the OpenAlex work id;
* a data-quality report records counts, missing rates, duplicate statistics and
  the SHA-256 of every raw file.

The raw source is the pinned SYNERGY v1.0 release on DataverseNL
(DOI 10.34894/HE6NAQ) downloaded by ``scripts/00_fetch_synergy.py``.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

REVIEW_IDS = (
    "Jeyaraman_2020",
    "Meijboom_2021",
    "Muthu_2021",
    "van_der_Valk_2021",
    "Leenaars_2020",
)

PROCESSED_COLUMNS = (
    "record_id",
    "title",
    "abstract",
    "has_title",
    "has_abstract",
    "doi",
    "source_url",
    "publication_year",
    "type",
)

FORBIDDEN_INPUT_COLUMNS = ("label_included", "label", "y", "gold_label")


# ---------------------------------------------------------------------------
# raw parsing
# ---------------------------------------------------------------------------


def _short_openalex_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    value = value.strip().rstrip("/")
    m = re.search(r"(W\d+)$", value)
    return m.group(1) if m else value


def abstract_from_inverted_index(inverted: object) -> str:
    """Rebuild plain text from an OpenAlex ``abstract_inverted_index``."""
    if not isinstance(inverted, dict) or not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if not isinstance(idxs, (list, tuple)):
            continue
        for pos in idxs:
            if isinstance(pos, int):
                positions.append((pos, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def load_raw_review(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (model-visible records, oracle labels) for one raw review dir."""
    raw_dir = Path(raw_dir)
    labels = pd.read_csv(raw_dir / "labels.csv", dtype=str)
    labels["record_id"] = labels["openalex_id"].map(_short_openalex_id)
    if labels["record_id"].duplicated().any():
        raise ValueError(f"{raw_dir.name}: duplicated openalex ids in labels.csv")
    oracle = labels[["record_id", "label_included"]].copy()
    oracle["label_included"] = oracle["label_included"].astype(int)

    rows: list[dict] = []
    works_zip = raw_dir / "works_1.zip"
    with zipfile.ZipFile(works_zip) as zf:
        names = sorted(zf.namelist())
        for name in names:
            if not name.endswith(".json"):
                continue
            works = json.loads(zf.read(name).decode("utf-8"))
            if not isinstance(works, list):
                raise ValueError(f"{works_zip}: {name} is not a JSON list")
            for work in works:
                rows.append(
                    {
                        "record_id": _short_openalex_id(work.get("id")),
                        "title": work.get("title") or work.get("display_name") or "",
                        "abstract": abstract_from_inverted_index(
                            work.get("abstract_inverted_index")
                        ),
                        "doi": work.get("doi") or "",
                        "source_url": work.get("id") or "",
                        "publication_year": work.get("publication_year"),
                        "type": work.get("type") or "",
                    }
                )
    records = pd.DataFrame(rows)
    if records["record_id"].eq("").any():
        raise ValueError(f"{raw_dir.name}: records without OpenAlex id")
    if records["record_id"].duplicated().any():
        raise ValueError(f"{raw_dir.name}: duplicated work ids in works_1.zip")

    # attach labels by record id; keep every raw record even without a label row
    merged = records.merge(oracle, on="record_id", how="left", validate="one_to_one")
    merged["label_included"] = merged["label_included"].fillna(-1).astype(int)
    merged["title"] = merged["title"].fillna("").astype(str)
    merged["abstract"] = merged["abstract"].fillna("").astype(str)
    merged["has_title"] = merged["title"].str.strip().ne("").astype(int)
    merged["has_abstract"] = merged["abstract"].str.strip().ne("").astype(int)
    if merged["label_included"].eq(-1).any():
        n_unlabelled = int(merged["label_included"].eq(-1).sum())
        raise ValueError(f"{raw_dir.name}: {n_unlabelled} records without oracle label")

    model_view = merged[list(PROCESSED_COLUMNS)].copy()
    oracle_out = merged[["record_id", "label_included"]].copy()
    return model_view, oracle_out


# ---------------------------------------------------------------------------
# quality report
# ---------------------------------------------------------------------------


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def quality_report(
    review_id: str,
    model_view: pd.DataFrame,
    oracle: pd.DataFrame,
    raw_dir: Path,
    raw_manifest: dict,
) -> dict:
    n = len(model_view)
    merged = model_view.merge(oracle, on="record_id", how="inner")
    included = merged["label_included"].eq(1)
    doi = merged["doi"].fillna("").str.strip().str.lower()
    dup_doi = int(doi[doi.ne("")].duplicated(keep=False).sum())
    titles = merged["title"].map(_norm_title)
    dup_title = int(titles[titles.ne("")].duplicated(keep=False).sum())
    dedup_key = doi.where(doi.ne(""), titles)
    n_unique = int(dedup_key[dedup_key.ne("")].nunique())
    sha256 = {}
    for f in raw_manifest["files"]:
        if f["review_id"] == review_id:
            sha256[f["filename"]] = f["local_sha256"]
    return {
        "review_id": review_id,
        "n_records": n,
        "n_included": int(included.sum()),
        "inclusion_rate": round(float(included.mean()), 6),
        "n_missing_title": int(model_view["has_title"].eq(0).sum()),
        "n_missing_abstract": int(model_view["has_abstract"].eq(0).sum()),
        "missing_abstract_rate": round(float(model_view["has_abstract"].eq(0).mean()), 6),
        "n_included_missing_abstract": int((included & merged["has_abstract"].eq(0)).sum()),
        "n_duplicate_doi_rows": dup_doi,
        "n_duplicate_title_rows": dup_title,
        "n_unique_dedup_keys": n_unique,
        "n_would_drop_if_dedup": n - n_unique,
        "raw_files_sha256": sha256,
        "raw_files_sha1_dataverse": {
            f["filename"]: f["dataverse_sha1"]
            for f in raw_manifest["files"]
            if f["review_id"] == review_id
        },
    }


def assert_no_labels(frame: pd.DataFrame, where: str) -> None:
    lowered = {c.lower() for c in frame.columns}
    leaked = [c for c in FORBIDDEN_INPUT_COLUMNS if c in lowered]
    if leaked:
        raise AssertionError(f"label leakage into {where}: {leaked}")


def build_dataset(raw_root: Path, out_dir: Path, oracle_dir: Path, manifest_path: Path) -> dict:
    raw_root = Path(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    summaries = {}
    for review_id in REVIEW_IDS:
        model_view, oracle = load_raw_review(raw_root / review_id)
        assert_no_labels(model_view, f"{review_id} model view")
        model_view.to_csv(out_dir / f"{review_id}.csv", index=False)
        oracle.to_csv(oracle_dir / f"{review_id}_oracle.csv", index=False)
        summaries[review_id] = quality_report(
            review_id, model_view, oracle, raw_root / review_id, manifest
        )
        q = summaries[review_id]
        print(
            f"{review_id}: n={q['n_records']} included={q['n_included']} "
            f"({100 * q['inclusion_rate']:.1f}%) missing_abs={q['n_missing_abstract']} "
            f"dup_doi_rows={q['n_duplicate_doi_rows']} dup_title_rows={q['n_duplicate_title_rows']}",
            flush=True,
        )
    quality = {
        "source": "SYNERGY v1.0 (DataverseNL, DOI 10.34894/HE6NAQ)",
        "reviews": summaries,
        "totals": {
            "n_records": sum(s["n_records"] for s in summaries.values()),
            "n_included": sum(s["n_included"] for s in summaries.values()),
        },
    }
    quality_path = out_dir.parent / "manifests" / "data_quality.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(f"wrote {quality_path}", flush=True)
    return quality
