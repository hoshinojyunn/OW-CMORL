from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_shared_protocol" / "scalarized_baselines"
EXPECTED_REGIMES = 20


def audit_run(run_dir: Path) -> dict[str, object]:
    result_path = run_dir / "result.json"
    shared_dir = run_dir / "shared_regime_returns"
    row = {
        "run_dir": str(run_dir),
        "has_result": result_path.exists(),
        "regime_rows": 0,
        "unique_regime_ids": 0,
        "shared_json": 0,
        "shared_npz": 0,
        "shared_seed_count": 0,
        "plan_regimes": 0,
        "plan_episode_seeds": 0,
        "seed_plan_match": False,
        "complete": False,
    }
    if shared_dir.exists():
        row["shared_json"] = len(list(shared_dir.glob("*.json")))
        row["shared_npz"] = len(list(shared_dir.glob("*.npz")))
    if not result_path.exists():
        return row
    try:
        payload = json.loads(result_path.read_text())
    except Exception:
        return row
    regime_rows = payload.get("regime_returns", [])
    shared_episode_seeds = [int(seed) for seed in payload.get("shared_eval_episode_seeds", [])]
    shared_plan = list(payload.get("shared_regime_seed_plan", []))
    unique_ids = {
        int(item.get("regime_id", -1))
        for item in regime_rows
        if int(item.get("regime_id", -1)) >= 0
    }
    row["regime_rows"] = len(regime_rows)
    row["unique_regime_ids"] = len(unique_ids)
    row["shared_seed_count"] = len(shared_episode_seeds)
    row["plan_regimes"] = len(shared_plan)
    planned_episode_seeds = [
        int(seed)
        for plan_row in shared_plan
        for seed in plan_row.get("episode_seeds", [])
    ]
    row["plan_episode_seeds"] = len(planned_episode_seeds)
    row["seed_plan_match"] = (
        shared_episode_seeds == planned_episode_seeds
        and row["shared_seed_count"] == row["plan_episode_seeds"] == EXPECTED_REGIMES
    )
    row["complete"] = (
        row["regime_rows"] >= EXPECTED_REGIMES
        and row["unique_regime_ids"] >= EXPECTED_REGIMES
        and row["shared_json"] >= EXPECTED_REGIMES
        and row["shared_npz"] >= EXPECTED_REGIMES
        and row["seed_plan_match"]
    )
    return row


def main() -> None:
    records: list[dict[str, object]] = []
    for env_dir in sorted(path for path in RESULTS_ROOT.iterdir() if path.is_dir()):
        for alg_dir in sorted(path for path in env_dir.iterdir() if path.is_dir()):
            run_dirs = sorted(path for path in alg_dir.iterdir() if path.is_dir())
            for run_dir in run_dirs:
                row = audit_run(run_dir)
                row["env_key"] = env_dir.name
                row["alg"] = alg_dir.name
                row["run_name"] = run_dir.name
                records.append(row)
    df = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "analysis" / "scalarized_shared_protocol_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["env_key", "alg"], dropna=False)["complete"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "complete_runs", "count": "total_runs"})
        .sort_values(["env_key", "alg"])
    )
    print(summary.to_string(index=False))
    print()
    print(f"Saved audit to {out_csv}")


if __name__ == "__main__":
    main()
