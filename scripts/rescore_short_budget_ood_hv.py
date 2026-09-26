#!/usr/bin/env python
"""Recompute short-OOD HV from persisted fronts under fixed trace envelopes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.short_ood_rescoring import rescore_event_rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rescore_results(
    results_root: Path,
    out_root: Path,
    profile: str,
) -> dict[str, Any]:
    source_root = Path(results_root) / profile / "dynamic"
    source_paths = sorted(source_root.glob("*/*/events.csv"))
    if not source_paths:
        raise FileNotFoundError(f"No short-OOD event files under {source_root}")
    rows: list[dict[str, Any]] = []
    for source_path in source_paths:
        for row in _read_rows(source_path):
            row["__source_path"] = str(source_path)
            rows.append(row)

    rescored, envelopes = rescore_event_rows(rows)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rescored:
        source_path = row.pop("__source_path")
        by_source.setdefault(source_path, []).append(row)
    destination_root = Path(out_root) / profile / "dynamic"
    for source_name, source_rows in by_source.items():
        relative = Path(source_name).relative_to(source_root)
        _write_rows(destination_root / relative, source_rows)

    payload = {
        "profile": profile,
        "source_results_root": str(source_root),
        "hv_metric": "trace_normalized_hv",
        "normalization": {
            "envelope": "componentwise min/max over all predeclared phases and trajectory seeds within one environment",
            "reference": "-0.1 in every active normalized objective dimension",
            "inactive_dimensions": "dropped only when constant across the complete environment trace",
        },
        "envelopes": {env_key: envelope.as_dict() for env_key, envelope in envelopes.items()},
    }
    output_manifest = Path(out_root) / profile / "trace_hv_rescore_manifest.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="ood_v3_site_tariff_trace")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood_short",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood_short_trace_hv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = rescore_results(args.results_root, args.out_root, args.profile)
    print(
        "[short-ood-rescore] wrote trace-normalized HV for "
        f"{len(payload['envelopes'])} environments to {args.out_root}"
    )


if __name__ == "__main__":
    main()
