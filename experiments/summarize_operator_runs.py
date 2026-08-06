#!/usr/bin/env python3
"""Merge operator accuracy, physical fidelity, and efficiency artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_operator_rows(run_root: Path) -> list[dict[str, object]]:
    results_path = run_root / "operator" / "results.csv"
    if not results_path.exists():
        return []
    with results_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_physical_row(run_root: Path, model: str) -> dict[str, object]:
    path = run_root / "physical" / f"{model}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_roots", nargs="+", help="Run directories containing operator/results.csv")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for root_arg in args.run_roots:
        run_root = Path(root_arg).resolve()
        for operator_row in read_operator_rows(run_root):
            model = str(operator_row["model"])
            physical_row = read_physical_row(run_root, model)
            rows.append(
                {
                    "run_root": str(run_root),
                    **operator_row,
                    **{
                        f"physical_{key}": value
                        for key, value in physical_row.items()
                        if key not in {"model", "latent_root", "split"}
                    },
                }
            )

    if not rows:
        raise FileNotFoundError("no operator/results.csv rows found")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
