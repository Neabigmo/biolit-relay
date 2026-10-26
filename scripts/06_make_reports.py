"""Stage 06 - evidence cards (label-blind) and criterion-state distributions.

Cards are the user-facing artifact of the risk router: for a record they show
the relevance score, the criterion-level evidence states and the verbatim
sentences that triggered each rule.  Cards never show the gold label.

Usage:
    python scripts/06_make_reports.py [--seed 0] [--top-n 40]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biolit_relay.risk_router import REASON_CODE_DESCRIPTIONS  # noqa: E402


def _card_html(row: pd.Series, hits: pd.DataFrame, meta: dict) -> str:
    def esc(x: object) -> str:
        return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))

    hit_rows = []
    for _, h in hits.iterrows():
        hit_rows.append(
            "<tr>"
            f"<td>{esc(h['criterion_name'])}</td>"
            f"<td class='{esc(h['state'])}'>{esc(h['state'])}</td>"
            f"<td>{esc(h['matched_phrase'])}</td>"
            f"<td class='sent'>{esc(h['evidence_sentence'])}</td>"
            "</tr>"
        )
    codes = [c for c in str(row.get("reason_codes", "")).split("|") if c]
    code_html = "".join(
        f"<span class='code' title='{esc(REASON_CODE_DESCRIPTIONS.get(c, ''))}'>{esc(c)}</span>"
        for c in codes
    )
    return f"""
<section class="card">
  <header>
    <h3>{esc(row['record_id'])} <small>{esc(row['review_id'])}</small></h3>
    <div class="score">round {int(row['round'])} &middot; p={esc(round(float(row['p_include']), 3) if pd.notna(row['p_include']) else 'n/a')}
      &middot; q={esc(round(float(row['q_score']), 3))}
      &middot; S/U/X={esc(round(float(row['s_ratio']), 2))}/{esc(round(float(row['u_ratio']), 2))}/{esc(round(float(row['x_ratio']), 2))}</div>
    <div class="codes">{code_html}</div>
  </header>
  <p class="title">{esc(row['title'])}</p>
  <p class="abstract">{esc(str(row['abstract'])[:900])}{'…' if len(str(row['abstract'])) > 900 else ''}</p>
  <table><thead><tr><th>criterion</th><th>state</th><th>matched phrase</th><th>evidence sentence</th></tr></thead>
  <tbody>{''.join(hit_rows)}</tbody></table>
  <footer>rule version {esc(meta.get('rule_version'))} &middot; spec hash {esc(meta.get('spec_hash', ''))[:12]} &middot; card generated without gold labels</footer>
</section>
"""


CSS = """
body{font-family:Georgia,'Times New Roman',serif;margin:2rem auto;max-width:1000px;color:#111;}
h1{font-size:1.4rem}.card{border:1px solid #bbb;border-radius:6px;padding:1rem;margin:1.2rem 0;}
.card h3{margin:0;font-size:1.05rem}.card h3 small{color:#666;font-weight:normal}
.score{font-family:ui-monospace,Consolas,monospace;font-size:.82rem;color:#333;margin:.3rem 0}
.codes{margin:.2rem 0}.code{display:inline-block;background:#eef2f7;border:1px solid #ccd6e0;border-radius:4px;
padding:.05rem .35rem;margin:.1rem .2rem .1rem 0;font-family:ui-monospace,Consolas,monospace;font-size:.72rem}
.title{font-weight:bold;margin:.5rem 0}.abstract{font-size:.9rem;color:#222;margin:.3rem 0}
table{width:100%;border-collapse:collapse;font-size:.8rem}th,td{border:1px solid #ddd;padding:.25rem .4rem;text-align:left;vertical-align:top}
.SUPPORT{color:#0a6b2d;font-weight:bold}.CONTRADICT{color:#a11;font-weight:bold}.MIXED{color:#8a5a00;font-weight:bold}.UNKNOWN{color:#666}
.sent{color:#333;font-style:italic}footer{font-size:.72rem;color:#777;margin-top:.5rem}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reviews = list(config["data"]["reviews"])
    freeze_path = ROOT / "data" / "manifests" / "spec_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {"reviews": []}
    meta = {r["review_id"]: r for r in freeze.get("reviews", [])}

    details = pd.read_csv(ROOT / "data" / "processed" / "evidence_details.csv")
    out_dir = ROOT / "outputs" / "evidence_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for review_id in reviews:
        runs_dir = ROOT / "outputs" / "runs"
        selected_paths = sorted(runs_dir.glob(f"{review_id}__biolit_relay__seed-{args.seed:02d}__*/selected_records.csv"))
        if not selected_paths:
            print(f"WARN: no biolit_relay run for {review_id} seed {args.seed}", file=sys.stderr)
            continue
        run_dir = selected_paths[-1].parent
        selected = pd.read_csv(selected_paths[-1])
        m1_paths = sorted(runs_dir.glob(f"{review_id}__matched_tfidf_logreg__seed-{args.seed:02d}__*/selected_records.csv"))
        m1_ids = set(pd.read_csv(m1_paths[-1])["record_id"]) if m1_paths else set()

        records = pd.read_csv(ROOT / "data" / "processed" / f"{review_id}.csv").set_index("record_id")
        # priority: risk-promoted records first, then remaining non-prior records in queue order
        non_prior = selected[selected["round"] > 0].copy()
        non_prior["promoted"] = non_prior["reason_codes"].fillna("").str.contains("RISK_PROMOTED")
        non_prior["new"] = ~non_prior["record_id"].isin(m1_ids)
        chosen = pd.concat(
            [
                non_prior[non_prior["promoted"] & non_prior["new"]],
                non_prior[non_prior["promoted"]],
                non_prior,
            ]
        ).drop_duplicates("record_id").head(args.top_n)

        cards = []
        for _, row in chosen.iterrows():
            rec = records.loc[row["record_id"]]
            detail = {
                "review_id": review_id,
                "record_id": row["record_id"],
                "round": row["round"],
                "p_include": row["p_include"],
                "q_score": row["q_score"],
                "s_ratio": row["s_ratio"],
                "u_ratio": row["u_ratio"],
                "x_ratio": row["x_ratio"],
                "reason_codes": row["reason_codes"],
                "title": rec["title"],
                "abstract": rec["abstract"],
            }
            hits = details[(details["review_id"] == review_id) & (details["record_id"] == row["record_id"])]
            cards.append(_card_html(pd.Series(detail), hits, meta.get(review_id, {})))
            summary_rows.append(
                {
                    "review_id": review_id,
                    "record_id": row["record_id"],
                    "round": int(row["round"]),
                    "promoted": bool(row["promoted"]),
                    "new_vs_matched_al": bool(row["new"]),
                    "reason_codes": row["reason_codes"],
                    "p_include": row["p_include"],
                    "q_score": row["q_score"],
                    "n_support": int(hits["state"].eq("SUPPORT").sum()),
                    "n_contradict": int(hits["state"].eq("CONTRADICT").sum()),
                    "n_unknown": int(hits["state"].eq("UNKNOWN").sum()),
                    "n_mixed": int(hits["state"].eq("MIXED").sum()),
                }
            )
        page = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>BioLit-Relay evidence cards - {review_id}</title><style>{CSS}</style></head><body>"
            f"<h1>BioLit-Relay evidence cards &middot; {review_id} &middot; seed {args.seed}</h1>"
            "<p>Routing artifacts for human review. Gold labels are deliberately absent; "
            f"runs: {run_dir.name}.</p>" + "".join(cards) + "</body></html>"
        )
        (out_dir / f"{review_id}_seed{args.seed:02d}.html").write_text(page, encoding="utf-8")
        print(f"wrote {out_dir.name}/{review_id}_seed{args.seed:02d}.html ({len(cards)} cards)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "evidence_cards_summary.csv", index=False)

    # criterion-state distribution per review (label-free)
    dist = (
        details.groupby(["review_id", "criterion_id", "state"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for state in ("SUPPORT", "CONTRADICT", "UNKNOWN", "MIXED"):
        if state not in dist.columns:
            dist[state] = 0
    n = dist[["SUPPORT", "CONTRADICT", "UNKNOWN", "MIXED"]].sum(axis=1)
    for state in ("SUPPORT", "CONTRADICT", "UNKNOWN", "MIXED"):
        dist[f"{state.lower()}_rate"] = dist[state] / n
    tables = ROOT / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    dist.to_csv(tables / "criterion_state_distribution.csv", index=False)
    print(f"wrote {tables.name}/criterion_state_distribution.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
