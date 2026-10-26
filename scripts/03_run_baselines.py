"""Stage 03 - run the baseline screening methods (M0-M4).

M0 random, M1 matched TF-IDF+LR active learning, M2 uncertainty-only,
M3 criterion-only, M4 ASReview-default components.

Usage:
    python scripts/03_run_baselines.py --n-jobs 20
    python scripts/03_run_baselines.py --reviews Jeyaraman_2020 --seeds 0
    python scripts/03_run_baselines.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.runner import build_tasks, load_config, run_tasks  # noqa: E402

BASELINE_METHODS = [
    "random",
    "matched_tfidf_logreg",
    "uncertainty_only",
    "criterion_only",
    "asreview_default",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    parser.add_argument("--reviews", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=BASELINE_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    tasks = build_tasks(
        config, methods=args.methods, reviews=args.reviews, seeds=args.seeds
    )
    print(f"{len(tasks)} baseline runs planned")
    if args.dry_run:
        for t in tasks[:10]:
            print("  ", t)
        print("  ...")
        return 0

    frame = run_tasks(tasks, config, n_jobs=args.n_jobs, root=ROOT)
    failed = frame[frame["status"] == "failed"]
    print(f"baselines done: {len(frame) - len(failed)} ok / {len(failed)} failed")
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
