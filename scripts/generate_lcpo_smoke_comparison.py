#!/usr/bin/env python
"""Summarize short-run LCPO results against the current 20-regime report.

The formal report is intentionally not rewritten.  LCPO is a short smoke run,
so this script writes a separate Markdown report and CSV.  Its HV/EU reference
points are formed from the pre-existing formal methods only, which keeps the
published rows numerically unchanged while placing LCPO on the same scale.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
METRICS = [
    "HV",
    "EU",
    "adapt_score",
    "trace_shift_regret",
    "trace_recovery_latency",
    "trace_recovery_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lcpo-root",
        type=Path,
        default=PROJECT_ROOT / "results_lcpo_smoke_20_regime",
    )
    parser.add_argument(
        "--main-report",
        type=Path,
        default=PROJECT_ROOT / "EXPERIMENT_RESULTS_ZH_20_REGIME.md",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "LCPO_SMOKE_RESULTS_20_REGIME.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "lcpo_smoke_20_regime_summary.csv",
    )
    return parser.parse_args()


def _load_formal_report_module():
    path = PROJECT_ROOT / "scripts" / "generate_20_regime_report.py"
    spec = importlib.util.spec_from_file_location("formal_report", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_main_markdown(path: Path) -> dict[str, list[list[str]]]:
    """Keep existing table cells exactly as published in the requested source."""
    parsed: dict[str, list[list[str]]] = {env_key: [] for env_key in ENV_ORDER}
    active_env: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            candidate = line[3:].strip()
            active_env = candidate if candidate in parsed else None
            continue
        if active_env is None or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 7 or cells[1] == "HV" or set(cells[0]) == {"-"}:
            continue
        parsed[active_env].append(cells)
    missing = [env_key for env_key, rows in parsed.items() if len(rows) < 5]
    if missing:
        raise ValueError(f"Could not parse all formal main-table rows for: {missing}")
    return parsed


def _base_rows(report_module) -> list[dict[str, Any]]:
    rows = report_module._dynamic_rows() + report_module._morl_rows()
    return report_module._align_rows_by_shared_plan(rows)


def _reference_pool(base_rows: list[dict[str, Any]], report_module, env_key: str):
    env_rows = [row for row in base_rows if str(row["env_key"]) == env_key]
    references: dict[int, np.ndarray] = {}
    best: dict[int, dict[str, float]] = {}
    for regime_id in range(report_module.TARGET_REGIMES):
        fronts = []
        for row in env_rows:
            for regime in row.get("regime_front_rows", []):
                if int(regime.get("regime_id", -1)) != regime_id:
                    continue
                front = np.asarray(regime.get("front", np.zeros((0, 0))), dtype=np.float64)
                if front.ndim == 2 and len(front) > 0:
                    fronts.append(front)
        if not fronts:
            continue
        reference = report_module._derive_reference_point(fronts)
        references[regime_id] = reference
        best_values = {"hv": -float("inf"), "eu": -float("inf")}
        for front in fronts:
            hv, eu = report_module._front_metrics(front, reference)
            best_values["hv"] = max(best_values["hv"], float(hv))
            best_values["eu"] = max(best_values["eu"], float(eu))
        best[regime_id] = best_values
    if len(references) != report_module.TARGET_REGIMES:
        raise ValueError(f"{env_key}: expected {report_module.TARGET_REGIMES} formal reference regimes, got {len(references)}")
    return references, best


def _lcpo_metrics(payload: dict[str, Any], report_module, references, best):
    regime_rows = report_module._regime_front_rows(list(payload.get("regime_fronts", [])))
    if len(regime_rows) != report_module.TARGET_REGIMES:
        raise ValueError(f"LCPO has {len(regime_rows)} regime fronts, expected {report_module.TARGET_REGIMES}")
    hv_samples: list[float] = []
    eu_samples: list[float] = []
    adapt_samples: list[float] = []
    for regime in regime_rows:
        regime_id = int(regime["regime_id"])
        front = np.asarray(regime["front"], dtype=np.float64)
        hv, eu = report_module._front_metrics(front, references[regime_id])
        hv_samples.append(float(hv))
        eu_samples.append(float(eu))
        target = best[regime_id]
        hv_gap = max(0.0, float(target["hv"]) - float(hv)) / max(abs(float(target["hv"])), 1e-8)
        eu_gap = max(0.0, float(target["eu"]) - float(eu)) / max(abs(float(target["eu"])), 1e-8)
        adapt_samples.append(float(np.clip(1.0 - 0.5 * (hv_gap + eu_gap), 0.0, 1.0)))

    trace_rows = report_module._episode_trace_rows_from_regime_solution_points(
        str(payload["env_key"]), regime_rows
    ) or report_module._morl_trace_rows(payload)
    sample_map = {
        "HV": hv_samples,
        "EU": eu_samples,
        "adapt_score": adapt_samples,
        "trace_shift_regret": report_module._trace_metric_list_from_rows(trace_rows, "trace_shift_regret"),
        "trace_recovery_latency": report_module._trace_metric_list_from_rows(trace_rows, "trace_recovery_latency"),
        "trace_recovery_score": report_module._trace_metric_list_from_rows(trace_rows, "trace_recovery_score"),
    }
    output: dict[str, float | int] = {}
    for metric, samples in sample_map.items():
        mean, ci, count = report_module._ci_from_samples(list(samples))
        output[f"{metric}_mean"] = float(mean)
        output[f"{metric}_ci"] = float(ci)
        output[f"{metric}_n"] = int(count)
    return output


def _format_lcpo_cell(report_module, row: dict[str, Any], metric: str) -> str:
    return report_module._format_ci(float(row[f"{metric}_mean"]), float(row[f"{metric}_ci"]))


def _summary_diagnostics(payload: dict[str, Any]) -> dict[str, float | int]:
    training = list(payload.get("training", []))
    if not training:
        raise ValueError("LCPO summary lacks per-preference training diagnostics.")
    return {
        "preference_count": int(len(training)),
        "timesteps_per_preference": int(payload.get("total_timesteps_per_preference", 0)),
        "total_training_timesteps": int(payload.get("total_training_timesteps", 0)),
        "lcpo_updates": int(sum(int(row.get("lcpo_updates", 0)) for row in training)),
        "fallback_updates": int(sum(int(row.get("fallback_updates", 0)) for row in training)),
        "constraint_accept_rate": float(np.mean([float(row.get("constraint_accept_rate", 0.0)) for row in training])),
        "shared_regime_count": int(len(payload.get("regime_fronts", []))),
    }


def write_markdown(
    *,
    path: Path,
    formal_rows: dict[str, list[list[str]]],
    lcpo_rows: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# LCPO Short-Run Results in Four Environments (20-Regime Shared Protocol)",
        "",
        "This report checks that LCPO runs under the shared protocol; it does not replace the formal results in `EXPERIMENT_RESULTS_ZH_20_REGIME.md`. LCPO is a single-objective online RL method. We train one policy per preference on the standard grid, then evaluate their joint Pareto front.",
        "",
        "## Protocol and Comparability",
        "",
        "- Each environment uses the same 20-regime seed plan as the formal benchmark; each LCPO run exports 20 `.json` and 20 `.npz` files.",
        "- LCPO uses a categorical policy per action dimension and local/OOD KL-constrained updates. This smoke run uses three action bins, two hidden layers of 32 units, 128 steps per preference, and three conjugate-gradient iterations.",
        "- To preserve published values, LCPO `HV/EU` reference points use only the formal methods (OW-CMORL, CAPQL, PGMORL, Q-Pensieve, MORL-CA). The `adapt_score` also uses their best values within each regime.",
        "- Formal-method rows are copied from `EXPERIMENT_RESULTS_ZH_20_REGIME.md`; LCPO rows are recomputed. Both show means and 95% t confidence intervals.",
        "",
        "## Smoke-Run Diagnostics",
        "",
        "| Environment | Preference policies | Steps per preference | Total training steps | LCPO / fallback updates | Constraint acceptance | Exported regimes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for env_key in ENV_ORDER:
        row = lcpo_rows[env_key]
        lines.append(
            "| "
            + " | ".join(
                [
                    env_key,
                    str(row["preference_count"]),
                    str(row["timesteps_per_preference"]),
                    str(row["total_training_timesteps"]),
                    f"{row['lcpo_updates']} / {row['fallback_updates']}",
                    f"{100.0 * float(row['constraint_accept_rate']):.1f}%",
                    str(row["shared_regime_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Metric Comparison",
            "",
            "LCPO is a short smoke run and should not be ranked against methods trained for much longer. This table checks that training, evaluation, per-regime export, and metric computation work under the same dynamic protocol.",
            "",
        ]
    )
    for env_key in ENV_ORDER:
        lines.extend(
            [
                f"### {env_key}",
                "",
                "| Method | HV | EU | adapt_score | trace_shift_regret | trace_recovery_latency | trace_recovery_score |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cells in formal_rows[env_key]:
            lines.append("| " + " | ".join(cells) + " |")
        lcpo = lcpo_rows[env_key]
        lines.append(
            "| LCPO (128-step smoke) | "
            + " | ".join(_format_lcpo_cell(_load_formal_report_module(), lcpo, metric) for metric in METRICS)
            + " |"
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "These results show that four-environment adaptation, online OOD constraints, shared-regime evaluation, and archival exports run end to end. With only 128 steps per preference, compared with roughly 200,000 steps for formal MORL baselines, these values do not support performance rankings. A formal comparison requires matched training budgets, multiple seeds, and finer action discretization.",
            "",
            "Raw results are stored in `results_lcpo_smoke_20_regime/<environment>/seed0/`; the entry point is `scripts/run_lcpo_dynamic_baseline.py`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    report_module = _load_formal_report_module()
    formal_rows = _parse_main_markdown(args.main_report)
    existing_rows = _base_rows(report_module)
    lcpo_rows: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        summary_path = args.lcpo_root / env_key / "seed0" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing LCPO summary: {summary_path}")
        payload = json.loads(summary_path.read_text())
        if payload.get("method") != "LCPO" or payload.get("env_key") != env_key:
            raise ValueError(f"Unexpected LCPO payload at {summary_path}")
        references, best = _reference_pool(existing_rows, report_module, env_key)
        metrics = _lcpo_metrics(payload, report_module, references, best)
        diagnostics = _summary_diagnostics(payload)
        row: dict[str, Any] = {
            "env_key": env_key,
            "method": "LCPO",
            "source": str(summary_path),
            **metrics,
            **diagnostics,
            "action_bins": int(payload.get("lcpo_config", {}).get("action_bins", 0)),
            "reference_pool": "formal main methods only",
        }
        lcpo_rows[env_key] = row
        csv_rows.append(row)
    write_markdown(path=args.out_md, formal_rows=formal_rows, lcpo_rows=lcpo_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(args.out_csv, index=False)
    print(json.dumps({"out_md": str(args.out_md), "out_csv": str(args.out_csv), "rows": len(csv_rows)}, indent=2))


if __name__ == "__main__":
    main()
