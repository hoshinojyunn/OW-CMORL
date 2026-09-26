#!/usr/bin/env python
"""Validate and summarize OW-CMORL short-update OOD event records."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.short_ood_protocol import id_return_retention, relative_switch_loss


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_recovery_auc(values: list[tuple[int, float]]) -> float:
    if len(values) < 2:
        return float("nan")
    values = sorted(values)
    zero = float(values[0][1])
    scale = max(abs(zero), 1e-8)
    budgets = np.asarray([item[0] for item in values], dtype=np.float64)
    recovery = np.asarray([(item[1] - zero) / scale for item in values], dtype=np.float64)
    span = float(budgets[-1] - budgets[0])
    return float(np.trapz(recovery, budgets) / span) if span > 0 else 0.0


def _validate_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (0 if row["phase"] == "id_pre" else 1 if _int(row, "budget") == 0 else 2, _int(row, "budget")))
    id_rows = [row for row in rows if row["phase"] == "id_pre"]
    ood_rows = sorted([row for row in rows if row["phase"] == "ood"], key=lambda row: _int(row, "budget"))
    return_rows = [row for row in rows if row["phase"] == "id_return"]
    errors: list[str] = []
    if len(id_rows) != 1 or len(return_rows) != 1 or not ood_rows:
        errors.append("missing required phase")
    if ood_rows and _int(ood_rows[0], "budget") != 0:
        errors.append("missing zero-update OOD point")
    budgets = [_int(row, "budget") for row in ood_rows]
    if budgets != sorted(set(budgets)):
        errors.append("nonmonotonic or duplicate update budget")
    if id_rows and ood_rows and id_rows[0]["state_hash"] != ood_rows[0]["state_hash"]:
        errors.append("zero-shot state hash mismatch")
    if return_rows and ood_rows and return_rows[0]["state_hash"] != ood_rows[-1]["state_hash"]:
        errors.append("ID-return state hash mismatch")
    if ood_rows and any(abs(_float(row, "update_count") - _int(row, "budget")) > 1e-9 for row in ood_rows):
        errors.append("reported update count does not equal cumulative update budget")
    positive = [row for row in ood_rows if _int(row, "budget") > 0]
    if positive and any(row.get("hnsw_backend", "") != "hnsw" for row in positive):
        errors.append("non-HNSW retrieval after update")
    return {
        "env_key": rows[0].get("env_key", ""),
        "method": rows[0].get("method", ""),
        "trajectory_seed": _int(rows[0], "trajectory_seed"),
        "valid": not errors,
        "errors": "; ".join(errors),
        "final_update_budget": budgets[-1] if budgets else None,
        "final_transition_count": _int(ood_rows[-1], "transition_count") if ood_rows else None,
    }


def build_report(results_root: str | Path, profile: str, analysis_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(results_root) / profile / "dynamic"
    events = []
    validations = []
    diagnostics = []
    summaries = []
    for event_path in sorted(root.glob("*/*/events.csv")):
        rows = _read_rows(event_path)
        if not rows:
            continue
        validations.append(_validate_trajectory(rows))
        events.extend(rows)
        grouped = {phase: [row for row in rows if row["phase"] == phase] for phase in ("id_pre", "ood", "id_return")}
        id_pre = grouped["id_pre"][0]
        id_return = grouped["id_return"][0]
        ood = sorted(grouped["ood"], key=lambda row: _int(row, "budget"))
        diagnostic = {
            "env_key": id_pre["env_key"],
            "method": id_pre["method"],
            "trajectory_seed": _int(id_pre, "trajectory_seed"),
        }
        for metric in ("HV", "EU"):
            diagnostic[f"zero_shot_loss_{metric}"] = relative_switch_loss(_float(id_pre, metric), _float(ood[0], metric))
            diagnostic[f"recovery_auc_{metric}"] = _relative_recovery_auc(
                [(_int(row, "budget"), _float(row, metric)) for row in ood]
            )
            diagnostic[f"id_return_retention_{metric}"] = id_return_retention(
                _float(id_pre, metric), _float(id_return, metric)
            )
        diagnostics.append(diagnostic)

    for env_key in sorted({row["env_key"] for row in events}):
        for phase in ("id_pre", "ood", "id_return"):
            matching = [row for row in events if row["env_key"] == env_key and row["phase"] == phase]
            for budget in sorted({_int(row, "budget") for row in matching}):
                point = [row for row in matching if _int(row, "budget") == budget]
                for metric in ("HV", "EU"):
                    values = np.asarray([_float(row, metric) for row in point], dtype=np.float64)
                    summaries.append(
                        {
                            "env_key": env_key,
                            "phase": phase,
                            "update_budget": budget,
                            "metric": metric,
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            "n": int(len(values)),
                            "mean_transition_count": float(np.mean([_int(row, "transition_count") for row in point])),
                        }
                    )

    output = {"events": events, "validation": validations, "diagnostics": diagnostics, "summary": summaries}
    analysis = Path(analysis_dir)
    _write_rows(analysis / "short_ood_events_summary.csv", summaries)
    _write_rows(analysis / "short_ood_diagnostics.csv", diagnostics)
    _write_rows(analysis / "short_ood_protocol_validation.csv", validations)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results_ood_short")
    parser.add_argument("--profile", default="ood_v3_site_tariff_trace")
    parser.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT / "analysis" / "short_ood_v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_report(args.results_root, args.profile, args.analysis_dir)
    invalid = [row for row in output["validation"] if not row["valid"]]
    if invalid:
        raise SystemExit(f"Invalid short-OOD trajectories: {invalid}")
    print(f"[short-ood-report] validated {len(output['validation'])} trajectories")


if __name__ == "__main__":
    main()
