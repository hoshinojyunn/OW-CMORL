from __future__ import annotations

import json
from pathlib import Path

from scripts.build_dynamic_subset_shared_result import _base_rows, _score_subset


def main() -> None:
    base_rows = _base_rows("chlor_alkali")
    root = Path("results_shared_protocol")
    candidates = [
        "chlor_owcmorl_union_cdfopt_v6.20260608",
        "chlor_owcmorl_union_v5_cdfopt_hifi_v1.20260608",
        "chlor_owcmorl_union_v5_cdfopt_relax_v1.20260608",
        "chlor_owcmorl_union_v4.20260608",
        "chlor_owcmorl_union_subsetsearch_v4.20260608",
        "chlor_owcmorl_union_cdfopt_v3.20260608",
        "chlor_owcmorl_search_obj02bridge_v1",
        "chlor_owcmorl_union_v6.20260608",
        "chlor_owcmorl_union_v6_cdfopt_hifi_v1.20260608",
    ]
    for name in candidates:
        run_dir = root / name
        summary_path = run_dir / "final" / "shared_eval_summary.json"
        regime_path = run_dir / "regime_fronts" / "shared_final.json"
        if not (summary_path.exists() and regime_path.exists()):
            continue
        summary = json.loads(summary_path.read_text())
        regime = json.loads(regime_path.read_text())
        trace_rows = list(summary.get("per_sample_trace_metrics", []))
        if not trace_rows:
            continue
        rec = _score_subset(
            "chlor_alkali",
            base_rows,
            regime,
            trace_rows,
            list(range(len(trace_rows))),
            run_dir,
            grid_points=201,
            bootstrap_samples=300,
            seed=0,
        )
        print(f"\n{name}")
        print(
            {
                key: rec[key]
                for key in [
                    "HV",
                    "EU",
                    "adapt_score",
                    "trace_shift_regret",
                    "trace_recovery_latency",
                    "trace_recovery_score",
                    "cdf_gap_mean",
                    "wins",
                ]
            }
        )
        print(rec["cdf_gap_by_obj"])


if __name__ == "__main__":
    main()
