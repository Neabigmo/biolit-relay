"""Stage 04 - run BioLit-Relay (M5), its ablations and prior-sensitivity variants.

Usage:
    python scripts/04_run_biolit_relay.py --n-jobs 20
    python scripts/04_run_biolit_relay.py --reviews Jeyaraman_2020 --seeds 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.runner import build_tasks, load_config, run_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--reviews", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    methods = args.methods or (
        ["biolit_relay"]
        + list(config["methods"].get("ablations", []))
        + ["matched_tfidf_logreg_prior2", "biolit_relay_prior2"]
        + list(config["methods"].get("hybrid", []))
        + list(config["methods"].get("control", []))
    )
    tasks = build_tasks(config, methods=methods, reviews=args.reviews, seeds=args.seeds)
    print(f"{len(tasks)} BioLit-Relay runs planned ({len(methods)} variants)")
    if args.dry_run:
        for t in tasks[:10]:
            print("  ", t)
        print("  ...")
        return 0

    frame = run_tasks(tasks, config, n_jobs=args.n_jobs, root=ROOT)
    failed = frame[frame["status"] == "failed"]
    print(f"BioLit-Relay done: {len(frame) - len(failed)} ok / {len(failed)} failed")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())

