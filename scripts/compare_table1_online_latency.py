from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.generate_20_regime_report as current_report

ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
METHOD_ORDER = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca"]
TABLE1_LATENCY_MD = PROJECT_ROOT / "LCPO_SMOKE_RESULTS_20_REGIME.md"
METHOD_LABEL = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
}
BASELINE_SCRIPT = {
    "capql": PROJECT_ROOT / "scripts" / "run_capql_dynamic_baseline.py",
    "pgmorl": PROJECT_ROOT / "scripts" / "run_pgmorl_dynamic_baseline.py",
    "q_pensieve": PROJECT_ROOT / "scripts" / "run_qpensieve_dynamic_baseline.py",
    "morlca": PROJECT_ROOT / "scripts" / "run_morlca_dynamic_baseline.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate the current Table 1 methods with online-step latency.")
    parser.add_argument("--env-keys", nargs="+", choices=ENV_ORDER, default=ENV_ORDER)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "results_online_step_latency_table1",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "table1_online_step_latency.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "table1_online_step_latency.csv",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _source_rows() -> dict[tuple[str, str], Path]:
    rows: dict[tuple[str, str], Path] = {}
    for row in current_report._dynamic_rows():
        rows[(str(row["env_key"]), str(row["method"]))] = Path(str(row["source"]))
    for row in current_report._morl_rows():
        rows[(str(row["env_key"]), str(row["method"]))] = Path(str(row["source"])).parent
    return rows


def _summary_path(method: str, run_dir: Path) -> Path:
    if method == "dynamic":
        return run_dir / "final" / "shared_eval_summary.json"
    return run_dir / "summary.json"


def _normalize_method_label(label: str) -> str:
    cleaned = label.replace("**", "").strip()
    cleaned = cleaned.split("\uFF08", 1)[0].split("(", 1)[0].strip()
    mapping = {value: key for key, value in METHOD_LABEL.items()}
    if cleaned not in mapping:
        return ""
    return mapping[cleaned]


def _parse_table1_old_latency() -> dict[tuple[str, str], float]:
    text = TABLE1_LATENCY_MD.read_text()
    env_key = None
    values: dict[tuple[str, str], float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            env_key = line[len("### ") :].strip()
            continue
        if env_key is None or not line.startswith("|"):
            continue
        if line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 6:
            continue
        if cells[1] == "HV":
            continue
        method = _normalize_method_label(cells[0])
        if not method:
            continue
        match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", cells[5], flags=re.IGNORECASE)
        if match is None:
            continue
        values[(env_key, method)] = float(match.group(0))
    return values


def _write_plan_json(summary_payload: dict[str, Any], target_path: Path) -> int:
    plan = list(summary_payload.get("shared_regime_seed_plan", []))
    if not plan:
        raise ValueError(f"missing shared_regime_seed_plan for {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(plan, indent=2))
    seeds = list(summary_payload.get("shared_eval_episode_seeds", []))
    return int(len(seeds) or len(plan))


def _cli_flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _append_cli_arg(argv: list[str], key: str, value: Any) -> None:
    if key in {"save_dir", "skip_train", "train_only", "shared_plan_json"}:
        return
    if value is None or value == "None":
        return
    flag = _cli_flag(key)
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.append(flag)
    if isinstance(value, list):
        argv.extend(str(item) for item in value)
    else:
        argv.append(str(value))


def _run_cmd(cmd: list[str], *, cwd: Path) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _prepare_baseline_target(source_run_dir: Path, target_run_dir: Path) -> None:
    if target_run_dir.exists():
        shutil.rmtree(target_run_dir)
    target_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run_dir, target_run_dir)


def _baseline_summary_rows(summary_payload: dict[str, Any]) -> list[dict[str, float]]:
    nested_rows: list[dict[str, float]] = []
    for key in ("per_preference", "per_policy"):
        for item in list(summary_payload.get(key, [])):
            trace_metrics = item.get("trace_metrics", {})
            if isinstance(trace_metrics, dict):
                nested_rows.append({k: float(v) for k, v in trace_metrics.items() if v is not None})
    return nested_rows


def _dynamic_summary_rows(summary_payload: dict[str, Any]) -> list[dict[str, float]]:
    rows = []
    for item in list(summary_payload.get("per_sample_trace_metrics", [])):
        if isinstance(item, dict):
            rows.append({k: float(v) for k, v in item.items() if v is not None})
    return rows


def _latency_stats(summary_payload: dict[str, Any], method: str) -> tuple[float, float, int]:
    if method == "dynamic":
        rows = _dynamic_summary_rows(summary_payload)
    else:
        rows = _baseline_summary_rows(summary_payload)
    values = [
        float(row["trace_recovery_latency"])
        for row in rows
        if "trace_recovery_latency" in row and np.isfinite(float(row["trace_recovery_latency"]))
    ]
    if not values:
        trace_metrics = dict(summary_payload.get("trace_metrics", {}))
        value = float(trace_metrics.get("trace_recovery_latency", float("nan")))
        return value, 0.0, 0 if not np.isfinite(value) else 1
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), int(len(arr))


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4f}"


def _reevaluate_dynamic(
    source_run_dir: Path,
    target_run_dir: Path,
    plan_json: Path,
    summary_payload: dict[str, Any],
    python_bin: Path,
) -> dict[str, Any]:
    meta_path = source_run_dir / "shared_protocol_reeval.json"
    seed_offset = 0
    if meta_path.exists():
        seed_offset = int(_load_json(meta_path).get("shared_regime_seed_offset", 0))
    shared_eval_episodes = _write_plan_json(summary_payload, plan_json)
    cmd = [
        str(python_bin),
        str(PROJECT_ROOT / "scripts" / "reevaluate_dynamic_run_shared_protocol.py"),
        "--source-run-dir",
        str(source_run_dir),
        "--target-run-dir",
        str(target_run_dir),
        "--shared-regime-eval-episodes",
        str(shared_eval_episodes),
        "--shared-regime-seed-offset",
        str(seed_offset),
        "--shared-plan-json",
        str(plan_json),
        "--force",
    ]
    _run_cmd(cmd, cwd=PROJECT_ROOT)
    return _load_json(_summary_path("dynamic", target_run_dir))


def _reevaluate_baseline(
    method: str,
    source_run_dir: Path,
    target_run_dir: Path,
    plan_json: Path,
    source_summary: dict[str, Any],
    python_bin: Path,
) -> dict[str, Any]:
    _prepare_baseline_target(source_run_dir, target_run_dir)
    config = _load_json(source_run_dir / "config.json")
    shared_eval_episodes = _write_plan_json(source_summary, plan_json)
    cmd = [str(python_bin), str(BASELINE_SCRIPT[method])]
    for key, value in config.items():
        _append_cli_arg(cmd, str(key), value)
    cmd.extend(
        [
            "--save-dir",
            str(target_run_dir),
            "--skip-train",
            "--shared-plan-json",
            str(plan_json),
            "--shared-regime-eval-episodes",
            str(shared_eval_episodes),
        ]
    )
    _run_cmd(cmd, cwd=PROJECT_ROOT)
    return _load_json(_summary_path(method, target_run_dir))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "env_key",
        "method",
        "label",
        "old_trace_recovery_latency",
        "new_trace_recovery_latency",
        "new_trace_recovery_latency_std",
        "latency_rows",
        "source",
        "target",
    ]
    lines = [",".join(header)]
    for row in rows:
        values = [
            str(row["env_key"]),
            str(row["method"]),
            str(row["label"]),
            repr(float(row["old_trace_recovery_latency"])),
            repr(float(row["new_trace_recovery_latency"])),
            repr(float(row["new_trace_recovery_latency_std"])),
            str(int(row["latency_rows"])),
            str(row["source"]),
            str(row["target"]),
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n")


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Table 1 Online-Step Latency Re-evaluation",
        "",
        "The original stored `trace_recovery_latency` values were computed with the previous latency definition.",
        "The `new_trace_recovery_latency` column below is re-evaluated from raw shared-trace runs after switching latency to online-step counting.",
        "",
    ]
    for env_key in ENV_ORDER:
        lines.append(f"## {env_key}")
        lines.append("")
        lines.append("| Method | Old latency | New latency (mean) | New latency (std) | Trace rows |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in rows:
            if row["env_key"] != env_key:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["label"]),
                        _format_value(float(row["old_trace_recovery_latency"])),
                        _format_value(float(row["new_trace_recovery_latency"])),
                        _format_value(float(row["new_trace_recovery_latency_std"])),
                        str(int(row["latency_rows"])),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    source_rows = _source_rows()
    old_latency_map = _parse_table1_old_latency()
    out_root = args.out_root.resolve()
    if out_root.exists() and not args.keep_artifacts:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for env_key in args.env_keys:
        for method in args.methods:
            source_run_dir = source_rows[(env_key, method)]
            source_summary = _load_json(_summary_path(method, source_run_dir))
            old_latency = float(old_latency_map.get((env_key, method), float("nan")))
            target_run_dir = out_root / method / env_key
            plan_json = out_root / "plans" / f"{method}_{env_key}_shared_plan.json"
            if method == "dynamic":
                new_summary = _reevaluate_dynamic(
                    source_run_dir,
                    target_run_dir,
                    plan_json,
                    source_summary,
                    args.python.resolve(),
                )
            else:
                new_summary = _reevaluate_baseline(
                    method,
                    source_run_dir,
                    target_run_dir,
                    plan_json,
                    source_summary,
                    args.python.resolve(),
                )
            mean_latency, std_latency, latency_rows = _latency_stats(new_summary, method)
            records.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "label": METHOD_LABEL[method],
                    "old_trace_recovery_latency": old_latency,
                    "new_trace_recovery_latency": mean_latency,
                    "new_trace_recovery_latency_std": std_latency,
                    "latency_rows": latency_rows,
                    "source": str(source_run_dir),
                    "target": str(target_run_dir),
                }
            )

    _write_csv(args.out_csv.resolve(), records)
    _write_md(args.out_md.resolve(), records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
