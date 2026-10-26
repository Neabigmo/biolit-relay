"""Download the five pinned SYNERGY reviews (dataset v1.0) from DataverseNL.

The SYNERGY dataset is published on DataverseNL under DOI 10.34894/HE6NAQ.
This script pins the v1.0 file ids from the Dataverse manifest and downloads
only the five reviews used in the main experiment, verifies the Dataverse
SHA-1 checksum and records a local SHA-256 manifest.

It deliberately does NOT download the full 26-review corpus.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://dataverse.nl/api"
PERSISTENT_ID = "doi:10.34894/HE6NAQ"
REVIEWS = [
    "Jeyaraman_2020",
    "Meijboom_2021",
    "Muthu_2021",
    "van_der_Valk_2021",
    "Leenaars_2020",
]
WANTED = ["CITATION.txt", "labels.csv", "metadata.json", "metadata_publication.json", "works_1.zip"]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "synergy-v1.0"
MANIFEST = ROOT / "data" / "manifests" / "raw_files_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "biolit-relay/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "biolit-relay/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dst)
            return
        except Exception as exc:  # noqa: BLE001 - network robustness
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} after {exc}", flush=True)
            time.sleep(3 * (attempt + 1))


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    meta = fetch_json(f"{API}/datasets/:persistentId/?persistentId={PERSISTENT_ID}")
    version = meta["data"]["latestVersion"]
    major = version.get("versionNumber", "?")
    minor = version.get("minorVersionNumber", 0)
    print(
        f"Dataverse dataset version: {major}.{minor} "
        f"state={version.get('versionState')}",
        flush=True,
    )
    pub = version.get("publicationDate") or ""
    entries = []
    for f in version["files"]:
        directory = f.get("directoryLabel", "")
        review = directory.split("/")[-1]
        if review not in REVIEWS:
            continue
        name = f["dataFile"]["filename"]
        if name not in WANTED:
            continue
        entries.append(
            {
                "review_id": review,
                "filename": name,
                "file_id": f["dataFile"]["id"],
                "filesize": f["dataFile"]["filesize"],
                "dataverse_sha1": f["dataFile"]["checksum"]["value"],
                "directory": directory,
            }
        )
    if len(entries) != len(REVIEWS) * len(WANTED):
        missing = {(r, w) for r in REVIEWS for w in WANTED}
        found = {(e["review_id"], e["filename"]) for e in entries}
        print(f"WARNING: missing manifest entries: {sorted(missing - found)}", flush=True)

    manifest = {
        "source": "SYNERGY dataset v1.0, DataverseNL",
        "persistent_id": PERSISTENT_ID,
        "doi": "10.34894/HE6NAQ",
        "dataverse_version": "1.0",
        "publication_date": pub,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reviews": REVIEWS,
        "files": [],
    }

    for e in sorted(entries, key=lambda x: (REVIEWS.index(x["review_id"]), x["filename"])):
        review = e["review_id"]
        dst = RAW / review / e["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{API}/access/datafile/{e['file_id']}"
        if dst.exists() and dst.stat().st_size == e["filesize"]:
            print(f"skip (cached) {review}/{e['filename']}", flush=True)
        else:
            print(f"download {review}/{e['filename']} ({e['filesize'] / 1e6:.1f} MB)", flush=True)
            download(url, dst)
        got_size = dst.stat().st_size
        got_sha1 = sha1(dst)
        ok = got_size == e["filesize"] and got_sha1.lower() == e["dataverse_sha1"].lower()
        manifest["files"].append(
            {
                **e,
                "path": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "local_sha256": sha256(dst),
                "local_sha1": got_sha1,
                "verified": ok,
            }
        )
        print(f"  size={got_size} sha1_verified={ok}", flush=True)
        if not ok:
            print(f"  ERROR checksum mismatch for {dst}", flush=True)

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bad = [f for f in manifest["files"] if not f["verified"]]
    print(f"wrote {MANIFEST} ({len(manifest['files'])} files, {len(bad)} failed)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
