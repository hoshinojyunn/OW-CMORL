from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.fine_regimes import coerce_regime_seed_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild MORL shared-protocol summary from per-regime artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shared-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    shared_dir = run_dir / "shared_regime_returns"
    plan = coerce_regime_seed_plan(json.loads(args.plan_json.read_text()))
    payload = json.loads(summary_path.read_text())

    regime_rows = []
    for json_path in sorted(shared_dir.glob("*.json")):
        row = json.loads(json_path.read_text())
        regime_rows.append(
            {
                "regime": str(row["regime"]),
                "regime_id": int(row["regime_id"]),
                "regime_meta": dict(row.get("regime_meta", {})),
                "points": list(row.get("solution_points", [])),
            }
        )

    payload["regime_fronts"] = regime_rows
    payload["shared_eval_episode_seeds"] = [int(seed) for seed in args.shared_seeds]
    payload["shared_regime_seed_plan"] = list(plan)
    summary_path.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "regimes": len(regime_rows),
                "shared_eval_episode_seeds": len(payload["shared_eval_episode_seeds"]),
                "shared_regime_seed_plan": len(payload["shared_regime_seed_plan"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
