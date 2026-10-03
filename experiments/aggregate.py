"""WP19 -- flatten every results/*.json into results/manifest.csv.

The manifest is the single source of truth the report quotes. Every number printed in
the report must trace to a row here, which is what stops a figure and a table
disagreeing because one was retyped.

    python -m experiments.aggregate
"""

from __future__ import annotations

import csv
import json

from src.config import RESULTS

# Nested containers live in the JSON for the figure builders; flattening them into the
# manifest would produce unreadable cells.
SKIP = {"groups", "reliability_curve", "cost_curves", "subgroup_detail",
        "shap_by_seed", "params"}

ORDER = [f"exp{i:02d}" for i in range(1, 15)]


def _flat(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in SKIP:
                continue
            _flat(f"{prefix}{k}_" if prefix else f"{k}_", v, out)
    elif isinstance(value, (list, tuple)):
        out[prefix.rstrip("_")] = json.dumps(value) if len(value) <= 8 else f"[{len(value)} items]"
    else:
        out[prefix.rstrip("_")] = value


def main():
    files = sorted(RESULTS.glob("exp*.json"),
                   key=lambda p: ORDER.index(p.stem[:5]) if p.stem[:5] in ORDER else 99)
    if not files:
        raise SystemExit(f"no result files in {RESULTS} -- run the experiments first")

    records, columns = [], []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            rec = {"experiment_id": payload["experiment_id"],
                   "title": payload["title"],
                   "run_utc": payload["run_utc"],
                   "label": row["label"]}
            for key, value in row.items():
                if key in ("label", "params"):
                    continue
                if key in ("cv", "test"):
                    if value is None:
                        continue
                    _flat(f"{key}_", value, rec)
                else:
                    _flat("", {key: value}, rec)
            rec["n_params"] = len(row.get("params") or {})
            records.append(rec)
            for k in rec:
                if k not in columns:
                    columns.append(k)

    path = RESULTS / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    print(f"saved -> {path}")
    print(f"{len(files)} experiments, {len(records)} rows, {len(columns)} columns")
    for path_ in files:
        p = json.loads(path_.read_text(encoding="utf-8"))
        n_test = sum(1 for r in p["rows"] if r.get("test") is not None)
        print(f"  {p['experiment_id']:<20} {len(p['rows']):>3} rows, "
              f"{n_test:>2} with a test block, {p['runtime_sec']:>8.1f}s")


if __name__ == "__main__":
    main()
