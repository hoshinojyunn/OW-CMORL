from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_20_regime_report import (
    _align_rows_by_shared_plan,
    _attach_adapt_score,
    _episode_trace_rows_from_regime_solution_points,
    _load_dynamic_regime_payload,
    _load_dynamic_summary_payload,
    _morl_rows,
    _regime_front_rows,
    _score_regime_front_rows,
    _shared_plan_signature,
    _trace_metric_list_from_rows,
    _resolve_dynamic_trace_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a dynamic candidate under the exact 20-regime report view."
    )
    parser.add_argument("--env-key", required=True, choices=["building", "evcharging", "cogen", "chlor_alkali"])
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _dynamic_row(env_key: str, run_dir: Path) -> dict[str, object]:
    summary = _load_dynamic_summary_payload(run_dir, allow_regime_fallback=True) or {}
    regime_payload = _load_dynamic_regime_payload(run_dir, allow_partial=True)
    if regime_payload is None:
        raise FileNotFoundError(f"missing shared regime payload under {run_dir}")

    regime_front_rows = _regime_front_rows(list(regime_payload.get("regimes", [])))
    per_trace_rows = _episode_trace_rows_from_regime_solution_points(env_key, regime_front_rows)
    if not per_trace_rows:
        per_trace_rows = _resolve_dynamic_trace_rows(summary)
    return {
        "env_key": env_key,
        "method": "dynamic",
        "source": str(run_dir),
        "plan_signature": _shared_plan_signature(summary),
        "regime_front_rows": regime_front_rows,
        "trace_samples": {
            "trace_shift_regret": _trace_metric_list_from_rows(per_trace_rows, "trace_shift_regret"),
            "trace_recovery_latency": _trace_metric_list_from_rows(per_trace_rows, "trace_recovery_latency"),
            "trace_recovery_score": _trace_metric_list_from_rows(per_trace_rows, "trace_recovery_score"),
        },
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = [row for row in _morl_rows() if row["env_key"] == args.env_key]
    rows.append(_dynamic_row(args.env_key, run_dir))
    rows = _align_rows_by_shared_plan(rows)
    rows = _score_regime_front_rows(rows)
    rows = _attach_adapt_score(rows)
    row = next(item for item in rows if item["method"] == "dynamic" and item["source"] == str(run_dir))

    print(f"env_key={args.env_key}")
    print(f"source={run_dir}")
    print(f"HV={_mean([float(x) for x in row['hv_samples']])}")
    print(f"EU={_mean([float(x) for x in row['eu_samples']])}")
    print(f"adapt_score={_mean([float(x) for x in row.get('adapt_score_samples', [])])}")
    for metric in ["trace_shift_regret", "trace_recovery_latency", "trace_recovery_score"]:
        values = [float(x) for x in row["trace_samples"].get(metric, [])]
        print(f"{metric}={_mean(values)} n={len(values)}")


if __name__ == "__main__":
    main()
