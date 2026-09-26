from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_20_regime_report import (
    ENV_ORDER,
    PREFERRED_DYNAMIC_RUNS,
    PRIMARY_ROOT,
    REPAIR_ROOT,
    _find_morl_candidates,
    _valid_dynamic_result,
    _valid_morl_summary,
)
from scripts.run_quick_adaptation_validation import (
    _display_method_name,
    _format_value,
    _load_dynamic_online_metrics,
    _load_frozen_trace_metrics,
    _load_regime_rows,
    _shared_trace_metrics_from_regime_rows,
)

LCPO_ROOT = PROJECT_ROOT / "results_lcpo_smoke_20_regime"
METHODS = ["dynamic", "capql", "qpensieve", "pgmorl", "morlca", "lcpo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract existing adaptation-validation evidence from current result directories.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "existing_adaptation_validation_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "existing_adaptation_validation_summary.md",
    )
    parser.add_argument(
        "--dynamic-override-root",
        type=Path,
        default=None,
        help="Optional quick-validation run root whose dynamic/<env> directories should override the formal OW-CMORL runs.",
    )
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    return parser.parse_args()


def _dynamic_online_ready(run_dir: Path) -> bool:
    return (run_dir / "metrics_history.csv").exists() and (run_dir / "final" / "objs.txt").exists()


def _resolve_dynamic_run_dir(env_key: str, dynamic_override_root: Path | None = None) -> Path | None:
    if dynamic_override_root is not None:
        override_dir = dynamic_override_root / "dynamic" / env_key
        if _valid_dynamic_result(override_dir) or _dynamic_online_ready(override_dir):
            return override_dir
    for run_name in PREFERRED_DYNAMIC_RUNS.get(env_key, [f"sharedprot_main_{env_key}_dynamic_seed0"]):
        for root in (REPAIR_ROOT, PRIMARY_ROOT):
            run_dir = root / run_name
            if _valid_dynamic_result(run_dir):
                return run_dir
    return None


def _resolve_morl_run_dir(method: str, env_key: str) -> Path | None:
    candidate_method = "q_pensieve" if method == "qpensieve" else method
    for candidate in _find_morl_candidates(candidate_method, env_key):
        if _valid_morl_summary(candidate):
            return candidate.parent
    return None


def _resolve_lcpo_run_dir(env_key: str) -> Path | None:
    run_dir = LCPO_ROOT / env_key / "seed0"
    summary_path = run_dir / "summary.json"
    shared_dir = run_dir / "shared_regime_returns"
    if summary_path.exists() and shared_dir.exists() and len(list(shared_dir.glob("*.json"))) >= 20:
        return run_dir
    return None


def _resolve_run_dir(method: str, env_key: str, dynamic_override_root: Path | None = None) -> Path | None:
    if method == "dynamic":
        return _resolve_dynamic_run_dir(env_key, dynamic_override_root=dynamic_override_root)
    if method == "lcpo":
        return _resolve_lcpo_run_dir(env_key)
    return _resolve_morl_run_dir(method, env_key)


def _row_from_run(method: str, env_key: str, run_dir: Path, eval_delta_weight: float) -> dict[str, Any]:
    regime_rows = _load_regime_rows(method, run_dir)
    shared_metrics = _shared_trace_metrics_from_regime_rows(regime_rows, eval_delta_weight)
    frozen_metrics = _load_frozen_trace_metrics(method, run_dir)
    row: dict[str, Any] = {
        "env_key": env_key,
        "method": _display_method_name(method),
        "raw_method": method,
        "run_dir": str(run_dir),
        "shared_trace_shift_regret": shared_metrics["trace_shift_regret"],
        "shared_trace_recovery_latency": shared_metrics["trace_recovery_latency"],
        "shared_trace_recovery_score": shared_metrics["trace_recovery_score"],
        "shared_trace_shift_count": shared_metrics["trace_shift_count"],
        "shared_trace_sample_count": shared_metrics["trace_sample_count"],
        "frozen_trace_shift_regret": frozen_metrics.get("frozen_trace_shift_regret", float("nan")),
        "frozen_trace_recovery_latency": frozen_metrics.get("frozen_trace_recovery_latency", float("nan")),
        "frozen_trace_recovery_score": frozen_metrics.get("frozen_trace_recovery_score", float("nan")),
        "frozen_trace_shift_count": frozen_metrics.get("frozen_trace_shift_count", float("nan")),
        "frozen_trace_pre_post_gap": frozen_metrics.get("frozen_trace_pre_post_gap", float("nan")),
    }
    if method == "dynamic":
        online_metrics = _load_dynamic_online_metrics(run_dir)
        row["online_trace_shift_regret"] = online_metrics.get("online_trace_shift_regret", float("nan"))
        row["online_trace_recovery_latency"] = online_metrics.get("online_trace_recovery_latency", float("nan"))
        row["online_trace_recovery_score"] = online_metrics.get("online_trace_recovery_score", float("nan"))
        row["online_trace_shift_count"] = online_metrics.get("online_trace_shift_count", float("nan"))
        row["online_shift_count"] = online_metrics.get("online_online_shift_count", online_metrics.get("online_shift_count", float("nan")))
        row["online_matched_recovery"] = online_metrics.get("online_matched_recovery", float("nan"))
    return row


def write_report(df: pd.DataFrame, out_md: Path) -> None:
    lines = [
        "# Existing Adaptation Validation",
        "",
        "This table is extracted from the current authoritative result directories in the workspace.",
        "Shared `trace_*` are reconstructed from shared-regime `solution_points`.",
        "Frozen `trace_*` come from each method's own final dynamic-evaluation summary.",
        "For `OW-CMORL`, `online_*` come from the last non-`shared_final` row of `metrics_history.csv`.",
        "",
    ]
    for env_key in ENV_ORDER:
        sub = df[df["env_key"] == env_key].copy()
        if sub.empty:
            continue
        lines.append(f"## {env_key}")
        lines.append("")
        lines.append("| Method | shared_trace_shift_regret | shared_trace_recovery_latency | shared_trace_recovery_score | frozen_trace_shift_regret | frozen_trace_recovery_latency | frozen_trace_recovery_score | online_trace_shift_regret | online_trace_recovery_latency | online_trace_recovery_score |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        _format_value(float(row.get("shared_trace_shift_regret", math.nan))),
                        _format_value(float(row.get("shared_trace_recovery_latency", math.nan))),
                        _format_value(float(row.get("shared_trace_recovery_score", math.nan))),
                        _format_value(float(row.get("frozen_trace_shift_regret", math.nan))),
                        _format_value(float(row.get("frozen_trace_recovery_latency", math.nan))),
                        _format_value(float(row.get("frozen_trace_recovery_score", math.nan))),
                        _format_value(float(row.get("online_trace_shift_regret", math.nan))),
                        _format_value(float(row.get("online_trace_recovery_latency", math.nan))),
                        _format_value(float(row.get("online_trace_recovery_score", math.nan))),
                    ]
                )
                + " |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    dynamic_override_root = args.dynamic_override_root.resolve() if args.dynamic_override_root is not None else None
    for env_key in ENV_ORDER:
        for method in METHODS:
            run_dir = _resolve_run_dir(method, env_key, dynamic_override_root=dynamic_override_root)
            if run_dir is None:
                continue
            try:
                rows.append(_row_from_run(method, env_key, run_dir, float(args.eval_delta_weight)))
            except Exception as exc:
                rows.append(
                    {
                        "env_key": env_key,
                        "method": _display_method_name(method),
                        "raw_method": method,
                        "run_dir": str(run_dir),
                        "error": str(exc),
                    }
                )
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    write_report(df, args.out_md)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
