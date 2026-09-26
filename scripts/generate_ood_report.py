from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from functools import lru_cache

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.generate_20_regime_report as base_report


ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
METHOD_ORDER = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca", "lcpo"]
DISPLAY_METHODS = ["OW-CMORL", "CAPQL", "PGMORL", "Q-Pensieve", "MORL-CA", "LCPO"]
OOD_METRICS = [
    "HV",
    "EU",
    "adapt_score",
    "trace_shift_regret",
    "trace_recovery_latency",
    "trace_recovery_score",
]
OOD_FRONTIER_METRICS = ["HV", "EU"]
OOD_ONLINE_METRICS = [
    "adapt_score",
    "trace_shift_regret",
    "trace_recovery_latency",
    "trace_recovery_score",
]
DISPLAY_NAME_OVERRIDES = {"lcpo": "LCPO"}


def _format_method_name(method: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(method, base_report._format_name(method))


@lru_cache(maxsize=None)
def _expected_eval_env_kwargs(profile: str, env_key: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "analysis" / "ood_protocols" / profile / env_key / "eval_env_config.json"
    payload = json.loads(path.read_text())
    env_kwargs = payload.get("env_kwargs", {})
    return env_kwargs if isinstance(env_kwargs, dict) else {}


def _env_kwargs_match_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _trace_evidence_paths(profile: str) -> tuple[Path, Path]:
    """Resolve the profile-specific illustrative ID/OOD trace evidence."""
    version = profile.split("_", 2)[1] if profile.startswith("ood_") and "_" in profile else ""
    analysis_dir = PROJECT_ROOT / "analysis" / f"ood_trace100_{version}"
    figure_dir = PROJECT_ROOT / "figures" / profile
    if (analysis_dir / "support_disjointness.json").exists() and figure_dir.exists():
        return figure_dir, analysis_dir
    return (
        PROJECT_ROOT / "figures" / "ood_trace100",
        PROJECT_ROOT / "analysis" / "ood_trace100",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OOD experiment report from results_ood.")
    parser.add_argument("--profile", type=str, default="ood_v1")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "OOD_EXPERIMENT_RESULTS_ZH.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_results_summary.csv",
    )
    return parser.parse_args()


def _shared_plan_signature(payload: dict[str, Any]) -> str | None:
    return base_report._shared_plan_signature(payload)


def _shared_plan_path_matches(actual_path: str | Path | None, expected_path: Path) -> bool:
    if actual_path is None:
        return False
    actual = Path(str(actual_path))
    if actual.resolve() == expected_path.resolve():
        return True
    if not actual.exists() or not expected_path.exists():
        return False
    try:
        return json.loads(actual.read_text()) == json.loads(expected_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False


def _trace_samples_from_dynamic(summary: dict[str, Any]) -> dict[str, list[float]]:
    per_rows = base_report._resolve_dynamic_trace_rows(summary)
    return {
        "trace_shift_regret": base_report._trace_metric_list_from_rows(per_rows, "trace_shift_regret"),
        "trace_recovery_latency": base_report._trace_metric_list_from_rows(per_rows, "trace_recovery_latency"),
        "trace_recovery_score": base_report._trace_metric_list_from_rows(per_rows, "trace_recovery_score"),
    }


def _trace_samples_from_baseline(summary: dict[str, Any]) -> dict[str, list[float]]:
    rows = base_report._morl_trace_rows(summary)
    return {
        "trace_shift_regret": base_report._trace_metric_list_from_rows(rows, "trace_shift_regret"),
        "trace_recovery_latency": base_report._trace_metric_list_from_rows(rows, "trace_recovery_latency"),
        "trace_recovery_score": base_report._trace_metric_list_from_rows(rows, "trace_recovery_score"),
    }


def _load_dynamic_rows(results_root: Path, profile: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        run_dir = results_root / profile / "dynamic" / env_key
        summary_path = run_dir / "final" / "shared_eval_summary.json"
        regime_path = run_dir / "regime_fronts" / "shared_final.json"
        reeval_path = run_dir / "shared_protocol_reeval.json"
        if not summary_path.exists() or not regime_path.exists() or not reeval_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        regime_payload = json.loads(regime_path.read_text())
        reeval_payload = json.loads(reeval_path.read_text())
        shared_plan_json = reeval_payload.get("shared_plan_json")
        expected_shared_plan_json = (
            PROJECT_ROOT / "analysis" / "ood_protocols" / profile / env_key / "shared_plan.json"
        )
        if not _shared_plan_path_matches(shared_plan_json, expected_shared_plan_json):
            continue
        rows.append(
            {
                "env_key": env_key,
                "method": "dynamic",
                "source": str(run_dir),
                "plan_signature": _shared_plan_signature(summary),
                "regime_front_rows": base_report._regime_front_rows(list(regime_payload.get("regimes", []))),
                "trace_samples": _trace_samples_from_dynamic(summary),
            }
        )
    return rows


def _load_baseline_rows(results_root: Path, profile: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        for method in ["capql", "pgmorl", "qpensieve", "morlca", "lcpo"]:
            run_dir = results_root / profile / method / env_key
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            env_kwargs = summary.get("env_kwargs", {})
            if not isinstance(env_kwargs, dict):
                continue
            if not _env_kwargs_match_expected(env_kwargs, _expected_eval_env_kwargs(profile, env_key)):
                continue
            rows.append(
                {
                    "env_key": env_key,
                    "method": "q_pensieve" if method == "qpensieve" else method,
                    "source": str(summary_path),
                    "plan_signature": _shared_plan_signature(summary),
                    "regime_front_rows": base_report._regime_front_rows(list(summary.get("regime_fronts", []))),
                    "trace_samples": _trace_samples_from_baseline(summary),
                }
            )
    return rows


def _normalize_metric(values: pd.Series, *, higher_better: bool) -> pd.Series:
    values = values.astype(float)
    finite = values.replace([np.inf, -np.inf], np.nan)
    lo = finite.min()
    hi = finite.max()
    if not np.isfinite(lo) or not np.isfinite(hi):
        return pd.Series(np.nan, index=values.index)
    if abs(float(hi) - float(lo)) <= 1e-12:
        return pd.Series(1.0, index=values.index)
    if higher_better:
        return (values - lo) / (hi - lo)
    return (hi - values) / (hi - lo)


def _attach_ood_normalized_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ood_frontier_score"] = np.nan
    out["ood_online_score"] = np.nan
    out["ood_composite_score"] = np.nan
    out["ood_normalized_score"] = np.nan
    for env_key in ENV_ORDER:
        mask = out["env_key"].astype(str) == env_key
        sub = out.loc[mask].copy()
        if sub.empty:
            continue
        norm_cols: list[str] = []
        for metric in OOD_METRICS:
            norm_col = f"norm_{metric}"
            out.loc[mask, norm_col] = _normalize_metric(
                sub[metric],
                higher_better=bool(base_report.HIGHER_BETTER[metric]),
            ).to_numpy()
            norm_cols.append(norm_col)
        frontier_cols = [f"norm_{metric}" for metric in OOD_FRONTIER_METRICS]
        online_cols = [f"norm_{metric}" for metric in OOD_ONLINE_METRICS]
        out.loc[mask, "ood_frontier_score"] = out.loc[mask, frontier_cols].mean(axis=1)
        out.loc[mask, "ood_online_score"] = out.loc[mask, online_cols].mean(axis=1)
        out.loc[mask, "ood_composite_score"] = out.loc[mask, norm_cols].mean(axis=1)
        # Paper-aligned primary ranking: normalize online OOD adaptation quality over traces.
        out.loc[mask, "ood_normalized_score"] = out.loc[mask, "ood_online_score"]
    return out


def build_summary(results_root: Path, profile: str) -> pd.DataFrame:
    rows = _load_dynamic_rows(results_root, profile) + _load_baseline_rows(results_root, profile)
    rows = base_report._align_rows_by_shared_plan(rows)
    rows = base_report._score_regime_front_rows(rows)
    rows = base_report._attach_adapt_score(rows)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "env_key": row["env_key"],
            "method": row["method"],
            "source": row["source"],
        }
        samples_by_metric = {
            "HV": row["hv_samples"],
            "EU": row["eu_samples"],
            "adapt_score": row.get("adapt_score_samples", []),
            "trace_shift_regret": row["trace_samples"].get("trace_shift_regret", []),
            "trace_recovery_latency": row["trace_samples"].get("trace_recovery_latency", []),
            "trace_recovery_score": row["trace_samples"].get("trace_recovery_score", []),
        }
        for metric, samples in samples_by_metric.items():
            mean, ci, n = base_report._ci_from_samples(samples)
            record[metric] = mean
            record[f"{metric}_ci"] = ci
            record[f"{metric}_n"] = n
        out_rows.append(record)

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    return _attach_ood_normalized_score(df)


def write_report(df: pd.DataFrame, out_md: Path, *, profile: str) -> None:
    trace_figure_dir, trace_analysis_dir = _trace_evidence_paths(profile)
    lines: list[str] = []
    lines.append("# OOD Experiment Results")
    lines.append("")
    lines.append(f"- OOD profile: `{profile}`")
    lines.append("- Protocol reference: `OOD_EXPERIMENT_PROTOCOL.md`")
    lines.append("- This report follows the OOD logic of the LCPO paper: evaluate on held-out dynamic context families and rank methods primarily by normalized online adaptation quality over OOD traces.")
    lines.append("- Evaluation type: simulation-only dynamic environments with real held-out context configurations. Raw metrics are reported alongside within-environment comparison normalizations.")
    lines.append("- `adapt_score`, OOD normalized score, OOD frontier score, and OOD composite score are method-set-relative comparative proxies, not externally calibrated returns or deployment metrics.")
    lines.append("- Rows carrying `ood_reuse_provenance.json` reuse an earlier measurement only after an exact OOD config and shared-plan equality check.")
    lines.append(
        "- Illustrative 100-regime ID/OOD traces and factorwise separation evidence: "
        f"`{trace_figure_dir.relative_to(PROJECT_ROOT)}/` and "
        f"`{trace_analysis_dir.relative_to(PROJECT_ROOT)}/support_disjointness.json`."
    )
    lines.append(f"- Regime-level OOD adaptation-score CDFs (method-set-relative comparative proxies): `figures/{profile}/` and `analysis/{profile}/`.")
    lines.append("")

    if df.empty:
        lines.append("No OOD results found.")
        out_md.write_text("\n".join(lines))
        return

    overall = (
        df.groupby("method", as_index=False)[
            ["ood_normalized_score", "ood_frontier_score", "ood_composite_score"]
        ]
        .mean()
        .sort_values("ood_normalized_score", ascending=False)
    )
    env_winners = (
        df.sort_values(["env_key", "ood_normalized_score"], ascending=[True, False])
        .groupby("env_key", as_index=False)
        .first()
    )
    env_win_counts = (
        env_winners.groupby("method", as_index=False)
        .size()
        .rename(columns={"size": "env_wins"})
        .sort_values(["env_wins", "method"], ascending=[False, True])
    )
    lines.append("## Overall")
    lines.append("")
    lines.append("| Method | OOD normalized score | OOD frontier score | OOD composite score |")
    lines.append("| --- | --- | --- | --- |")
    for _, row in overall.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_method_name(str(row["method"])),
                    base_report._format_value(float(row["ood_normalized_score"])),
                    base_report._format_value(float(row["ood_frontier_score"])),
                    base_report._format_value(float(row["ood_composite_score"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("| Method | #Env wins |")
    lines.append("| --- | --- |")
    for _, row in env_win_counts.iterrows():
        lines.append(f"| {_format_method_name(str(row['method']))} | {int(row['env_wins'])} |")
    lines.append("")
    observed_winners = ", ".join(
        f"{row['env_key']}={_format_method_name(str(row['method']))}"
        for _, row in env_winners.sort_values("env_key").iterrows()
    )
    lines.append(f"- Observed primary-score winners: {observed_winners}.")
    dynamic_wins = int((env_winners["method"] == "dynamic").sum())
    if dynamic_wins == len(ENV_ORDER):
        lines.append("- The measured result supports the all-four-environment OW-CMORL claim.")
    else:
        lines.append(
            f"- The measured result does not support the all-four-environment OW-CMORL claim ({dynamic_wins}/{len(ENV_ORDER)} wins)."
        )
    lcpo_rows = df[df["method"].astype(str) == "lcpo"]
    if not lcpo_rows.empty:
        lines.append(
            "- LCPO uses 128 training interactions per scalarization preference in this resource-limited pilot; it is included for protocol coverage, not as a training-budget-matched baseline."
        )
    lines.append("")

    for env_key in ENV_ORDER:
        sub = df[df["env_key"].astype(str) == env_key].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("ood_normalized_score", ascending=False)
        lines.append(f"## {env_key}")
        lines.append("")
        lines.append("| Method | HV | EU | adapt_score | trace_shift_regret | trace_recovery_latency | trace_recovery_score | OOD normalized score | OOD frontier score | OOD composite score |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_method_name(str(row["method"])),
                        base_report._format_ci(float(row["HV"]), float(row["HV_ci"])),
                        base_report._format_ci(float(row["EU"]), float(row["EU_ci"])),
                        base_report._format_ci(float(row["adapt_score"]), float(row["adapt_score_ci"])),
                        base_report._format_ci(float(row["trace_shift_regret"]), float(row["trace_shift_regret_ci"])),
                        base_report._format_ci(float(row["trace_recovery_latency"]), float(row["trace_recovery_latency_ci"])),
                        base_report._format_ci(float(row["trace_recovery_score"]), float(row["trace_recovery_score_ci"])),
                        base_report._format_value(float(row["ood_normalized_score"])),
                        base_report._format_value(float(row["ood_frontier_score"])),
                        base_report._format_value(float(row["ood_composite_score"])),
                    ]
                )
                + " |"
            )
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    df = build_summary(args.results_root, args.profile)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    write_report(df, args.out_md, profile=args.profile)
    if not df.empty:
        print(df[["env_key", "method", "ood_normalized_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
