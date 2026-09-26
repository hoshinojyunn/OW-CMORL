from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_ood_report import ENV_ORDER, build_summary


EXPECTED_METHODS = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca", "lcpo"]
DISPLAY_NAMES = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
    "lcpo": "LCPO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit OOD result completion and current OW-CMORL standing.")
    parser.add_argument("--profile", type=str, default="ood_v1")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood",
    )
    return parser.parse_args()


def _print_missing(df: pd.DataFrame) -> None:
    print("Missing combinations:")
    if df.empty:
        for env_key in ENV_ORDER:
            for method in EXPECTED_METHODS:
                print(f"  {env_key}: {DISPLAY_NAMES[method]}")
        return
    present = {(str(row.env_key), str(row.method)) for row in df.itertuples(index=False)}
    missing = []
    for env_key in ENV_ORDER:
        for method in EXPECTED_METHODS:
            if (env_key, method) not in present:
                missing.append((env_key, method))
    if not missing:
        print("  none")
        return
    for env_key, method in missing:
        print(f"  {env_key}: {DISPLAY_NAMES[method]}")


def _print_env_status(df: pd.DataFrame) -> tuple[int, int]:
    print("\nEnvironment status:")
    complete_envs = 0
    dynamic_verified_wins = 0
    for env_key in ENV_ORDER:
        sub = df[df["env_key"].astype(str) == env_key].copy()
        methods = set(sub["method"].astype(str).tolist())
        missing_methods = [m for m in EXPECTED_METHODS if m not in methods]
        if missing_methods:
            print(
                f"  {env_key}: partial ({len(methods)}/{len(EXPECTED_METHODS)}) "
                f"missing={', '.join(DISPLAY_NAMES[m] for m in missing_methods)}"
            )
            if not sub.empty:
                winner = sub.sort_values("ood_normalized_score", ascending=False).iloc[0]
                print(
                    f"    current leader={DISPLAY_NAMES[str(winner['method'])]} "
                    f"score={float(winner['ood_normalized_score']):.6f}"
                )
            continue
        complete_envs += 1
        ranked = sub.sort_values("ood_normalized_score", ascending=False).reset_index(drop=True)
        winner = ranked.iloc[0]
        winner_name = DISPLAY_NAMES[str(winner["method"])]
        if str(winner["method"]) == "dynamic":
            dynamic_verified_wins += 1
        print(
            f"  {env_key}: complete winner={winner_name} "
            f"score={float(winner['ood_normalized_score']):.6f}"
        )
    return complete_envs, dynamic_verified_wins


def _print_overall(df: pd.DataFrame) -> None:
    print("\nOverall averages on completed rows:")
    if df.empty:
        print("  no rows")
        return
    agg = (
        df.groupby("method", as_index=False)[
            ["ood_normalized_score", "ood_frontier_score", "ood_composite_score"]
        ]
        .mean()
        .sort_values("ood_normalized_score", ascending=False)
    )
    for row in agg.itertuples(index=False):
        print(
            f"  {DISPLAY_NAMES[str(row.method)]}: "
            f"normalized={float(row.ood_normalized_score):.6f}, "
            f"frontier={float(row.ood_frontier_score):.6f}, "
            f"composite={float(row.ood_composite_score):.6f}"
        )


def _print_success_check(df: pd.DataFrame, complete_envs: int, dynamic_verified_wins: int) -> None:
    print("\nSuccess check:")
    if complete_envs < len(ENV_ORDER):
        print(
            f"  incomplete audit: only {complete_envs}/{len(ENV_ORDER)} environments have all methods."
        )
        print(
            f"  verified OW-CMORL wins so far: {dynamic_verified_wins}/{complete_envs}"
        )
        return
    if dynamic_verified_wins >= 3:
        print(f"  criterion 1 satisfied: OW-CMORL wins {dynamic_verified_wins}/4 environments.")
    else:
        print(f"  criterion 1 not satisfied: OW-CMORL wins {dynamic_verified_wins}/4 environments.")

    agg = (
        df.groupby("method", as_index=False)[
            ["ood_normalized_score", "ood_frontier_score", "ood_composite_score"]
        ]
        .mean()
        .sort_values("ood_normalized_score", ascending=False)
        .reset_index(drop=True)
    )
    leader = agg.iloc[0]
    print(
        f"  primary overall leader: {DISPLAY_NAMES[str(leader['method'])]} "
        f"(normalized={float(leader['ood_normalized_score']):.6f})"
    )
    if str(leader["method"]) == "dynamic":
        print("  criterion 2 satisfied: OW-CMORL is overall best on the primary OOD score.")
    else:
        print("  criterion 2 not yet satisfied on the primary OOD score.")


def main() -> None:
    args = parse_args()
    df = build_summary(args.results_root, args.profile)
    print(f"profile={args.profile}")
    print(f"rows={len(df)}")
    _print_missing(df)
    complete_envs, dynamic_verified_wins = _print_env_status(df)
    _print_overall(df)
    _print_success_check(df, complete_envs, dynamic_verified_wins)


if __name__ == "__main__":
    main()
