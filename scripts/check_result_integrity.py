from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_WEIGHTS = {
    "building": 7,
    "evcharging": 7,
    "cogen": 11,
}
ALGS = ["a2c", "acer", "acktr", "ddpg", "deepq", "ppo1", "ppo2", "trpo_mpi"]
MORL_METHODS = ["capql", "qpensieve", "pgmorl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--dynamic-prefixes", nargs="+", required=True)
    parser.add_argument("--morl-baseline-root", type=Path)
    parser.add_argument("--morl-suite-names", nargs="+", default=["morl_online_suite"])
    parser.add_argument("--summary-csv", type=Path)
    return parser.parse_args()


def baseline_report(root: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for env_key, expected in EXPECTED_WEIGHTS.items():
        env_root = root / env_key
        env_report = {"expected_per_alg": expected, "algs": {}, "ok": True}
        for alg in ALGS:
            alg_root = env_root / alg
            result_count = len(list(alg_root.glob("seed0_w*/result.json")))
            trace_count = len(list(alg_root.glob("seed0_w*/dynamic_trace.csv")))
            ok = result_count == expected and trace_count == expected
            env_report["algs"][alg] = {
                "result_json": result_count,
                "dynamic_trace_csv": trace_count,
                "ok": ok,
            }
            env_report["ok"] &= ok
        out[env_key] = env_report
    return out


def dynamic_report(root: Path, prefixes: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    configs = ["dynamic", "static", "random", "no_dyn_ablation"]
    for env_key in EXPECTED_WEIGHTS:
        env_report = {"configs": {}, "ok": True}
        for config in configs:
            matched = None
            for prefix in prefixes:
                run_dir = root / f"{prefix}_{env_key}_{config}_seed0"
                if run_dir.exists():
                    matched = run_dir
                    break
            run_dir = matched or (root / f"{prefixes[0]}_{env_key}_{config}_seed0")
            metrics = run_dir / "metrics_history.csv"
            final = run_dir / "final" / "objs.txt"
            regime_fronts = run_dir / "regime_fronts"
            ok = metrics.exists() and final.exists() and regime_fronts.exists()
            env_report["configs"][config] = {
                "run_dir": str(run_dir),
                "metrics_history": metrics.exists(),
                "final_objs": final.exists(),
                "regime_fronts": regime_fronts.exists(),
                "ok": ok,
            }
            env_report["ok"] &= ok
        out[env_key] = env_report
    return out


def morl_baseline_report(root: Path | None, suite_names: list[str]) -> dict[str, object]:
    if root is None:
        return {"present": False}
    out: dict[str, object] = {"present": False, "suites": {}, "coverage": {}, "ok": True}
    any_present = False
    coverage: dict[str, dict[str, object]] = {
        method: {"envs": {}, "ok": True}
        for method in MORL_METHODS
    }
    for method in MORL_METHODS:
        for env_key in EXPECTED_WEIGHTS:
            coverage[method]["envs"][env_key] = {
                "summary_json": False,
                "path": None,
                "suite_name": None,
                "ok": False,
            }
    for suite_name in suite_names:
        suite_root = root / suite_name
        if not suite_root.exists():
            out["suites"][suite_name] = {"present": False, "suite_root": str(suite_root)}
            continue
        any_present = True
        suite_report: dict[str, object] = {"present": True, "suite_root": str(suite_root), "methods": {}, "ok": True}
        for method in MORL_METHODS:
            method_report = {"envs": {}, "ok": True}
            for env_key in EXPECTED_WEIGHTS:
                summary_path = suite_root / method / env_key / "seed0" / "summary.json"
                ok = summary_path.exists()
                has_front = False
                has_regime_fronts = False
                has_trace_metrics = False
                num_policies = 0
                if ok:
                    data = json.loads(summary_path.read_text())
                    front_points = data.get("front_points", [])
                    regime_fronts = data.get("regime_fronts", [])
                    trace_metrics = data.get("trace_metrics", {})
                    num_policies = int(data.get("num_policies", len(front_points)))
                    has_front = bool(front_points)
                    has_regime_fronts = bool(regime_fronts)
                    has_trace_metrics = bool(trace_metrics)
                    ok = ok and has_front and has_regime_fronts and has_trace_metrics and num_policies > 0
                method_report["envs"][env_key] = {
                    "summary_json": summary_path.exists(),
                    "path": str(summary_path),
                    "num_policies": num_policies,
                    "has_front_points": has_front,
                    "has_regime_fronts": has_regime_fronts,
                    "has_trace_metrics": has_trace_metrics,
                    "ok": ok,
                }
                if ok and not coverage[method]["envs"][env_key]["summary_json"]:
                    coverage[method]["envs"][env_key] = {
                        "summary_json": True,
                        "path": str(summary_path),
                        "suite_name": suite_name,
                        "num_policies": num_policies,
                        "has_front_points": has_front,
                        "has_regime_fronts": has_regime_fronts,
                        "has_trace_metrics": has_trace_metrics,
                        "ok": True,
                    }
                method_report["ok"] &= ok
            suite_report["methods"][method] = method_report
            method_present = any(env["summary_json"] for env in method_report["envs"].values())
            if method_present:
                suite_report["ok"] &= method_report["ok"]
        out["suites"][suite_name] = suite_report
    out["present"] = any_present
    for method in MORL_METHODS:
        coverage[method]["ok"] = all(
            bool(coverage[method]["envs"][env_key]["summary_json"])
            for env_key in EXPECTED_WEIGHTS
        )
    out["coverage"] = coverage
    out["ok"] = all(bool(coverage[method]["ok"]) for method in MORL_METHODS)
    return out


def summary_report(summary_csv: Path | None) -> dict[str, object]:
    if summary_csv is None or not summary_csv.exists():
        return {"present": False}
    df = pd.read_csv(summary_csv)
    core_cols = [
        col
        for col in [
            "HV",
            "EU",
            "SP",
            "cr_hv",
            "irs",
            "trace_shift_regret",
            "trace_recovery_latency",
            "trace_recovery_score",
            "adapt_score",
        ]
        if col in df.columns
    ]
    return {
        "present": True,
        "rows": int(len(df)),
        "has_nan_any": bool(df.isna().any().any()),
        "has_nan_core": bool(df[core_cols].isna().any().any()) if core_cols else False,
        "columns": list(df.columns),
    }


def main() -> None:
    args = parse_args()
    report = {
        "baseline": baseline_report(args.baseline_root),
        "dynamic": dynamic_report(args.dynamic_root, args.dynamic_prefixes),
        "morl_baseline": morl_baseline_report(args.morl_baseline_root, args.morl_suite_names),
        "summary": summary_report(args.summary_csv),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
