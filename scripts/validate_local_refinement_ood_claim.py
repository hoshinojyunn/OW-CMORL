#!/usr/bin/env python
"""Validate the predeclared short-OOD local-refinement claim from event fronts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def _ratio(after: float, before: float) -> float:
    if abs(before) <= 1e-12:
        raise ValueError("retention is undefined for a zero initial score")
    return float(after / before)


def _trajectory_record(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    points = {(row["phase"], int(float(row["budget"]))): row for row in rows}
    required = (("id_pre", 0), ("ood", 0), ("ood", 8), ("ood", 16), ("ood", 32), ("id_return", 32))
    missing = [f"{phase}/U={budget}" for phase, budget in required if (phase, budget) not in points]
    if missing:
        raise ValueError(f"{path} is missing required points: {', '.join(missing)}")
    ood_hv = [float(points[("ood", budget)]["HV"]) for budget in (0, 8, 16, 32)]
    pre_hv = float(points[("id_pre", 0)]["HV"])
    post_hv = float(points[("id_return", 32)]["HV"])
    pre_eu = float(points[("id_pre", 0)]["EU"])
    post_eu = float(points[("id_return", 32)]["EU"])
    return {
        "env_key": points[("ood", 0)]["env_key"],
        "trajectory_seed": int(float(points[("ood", 0)]["trajectory_seed"])),
        "ood_hv": ood_hv,
        "ood_hv_gain": float(ood_hv[-1] / ood_hv[0] - 1.0),
        "ood_hv_nondecreasing": bool(np.all(np.diff(np.asarray(ood_hv)) >= -1e-12)),
        "id_retention_hv": _ratio(post_hv, pre_hv),
        "id_retention_eu": _ratio(post_eu, pre_eu),
    }


def validate_claim(
    *,
    manifest_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    profile = str(manifest["profile"])
    root = results_root / profile / "dynamic"
    records = [_trajectory_record(path) for path in sorted(root.glob("*/*/events.csv"))]
    expected_envs = sorted(str(key) for key in manifest["local_lr_scales"])
    expected_seeds = sorted(int(seed) for seed in manifest["trajectory_seeds"])
    found = {(record["env_key"], record["trajectory_seed"]) for record in records}
    missing = [
        {"env_key": env_key, "trajectory_seed": seed}
        for env_key in expected_envs
        for seed in expected_seeds
        if (env_key, seed) not in found
    ]
    retention_floor = 0.98
    per_environment = []
    for env_key in expected_envs:
        subset = [record for record in records if record["env_key"] == env_key]
        gains = np.asarray([record["ood_hv_gain"] for record in subset], dtype=np.float64)
        ret_hv = np.asarray([record["id_retention_hv"] for record in subset], dtype=np.float64)
        ret_eu = np.asarray([record["id_retention_eu"] for record in subset], dtype=np.float64)
        per_environment.append(
            {
                "env_key": env_key,
                "n": len(subset),
                "mean_ood_hv_gain": float(gains.mean()) if len(gains) else float("nan"),
                "min_ood_hv_gain": float(gains.min()) if len(gains) else float("nan"),
                "mean_id_retention_hv": float(ret_hv.mean()) if len(ret_hv) else float("nan"),
                "min_id_retention_hv": float(ret_hv.min()) if len(ret_hv) else float("nan"),
                "mean_id_retention_eu": float(ret_eu.mean()) if len(ret_eu) else float("nan"),
                "min_id_retention_eu": float(ret_eu.min()) if len(ret_eu) else float("nan"),
                "ood_hv_nondecreasing_for_all_seeds": bool(
                    subset and all(record["ood_hv_nondecreasing"] for record in subset)
                ),
                "passes": bool(
                    len(subset) == len(expected_seeds)
                    and np.all(gains > 0.0)
                    and np.all(ret_hv >= retention_floor)
                    and np.all(ret_eu >= retention_floor)
                ),
            }
        )
    return {
        "protocol_version": manifest["protocol_version"],
        "profile": profile,
        "primary_metric": manifest["primary_metric"],
        "retention_floor": retention_floor,
        "missing_trajectories": missing,
        "per_trajectory": records,
        "per_environment": per_environment,
        "claim_supported": bool(
            not missing and per_environment and all(record["passes"] for record in per_environment)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
       
        / "analysis"
        / "ood_protocols"
        / "ood_v3_site_tariff_trace"
        / "local_refinement_validation_v1.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood_local_refinement_v1_trace_hv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
       
        / "analysis"
        / "short_ood_local_refinement_v1"
        / "local_refinement_claim_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_claim(manifest_path=args.manifest, results_root=args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not report["claim_supported"]:
        raise SystemExit(f"Local-refinement claim is not supported: {args.out}")
    print(f"[local-refinement-claim] passed: {args.out}")


if __name__ == "__main__":
    main()
