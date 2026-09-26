from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INNOVATION_ROOT = PROJECT_ROOT
REPORT_PATH = INNOVATION_ROOT / "EXPERIMENT_RESULTS_ZH_20_REGIME.md"
CSV_PATH = INNOVATION_ROOT / "analysis" / "experiment_results_20_regime_summary.csv"
LOG_PATH = INNOVATION_ROOT / "analysis" / "ev_repair_monitor.log"

WATCH_FILES = [
    INNOVATION_ROOT / "results_shared_protocol_repair" / "morl_dynamic_baselines" / "evcharging_repair_newregimes" / "capql" / "evcharging" / "seed0" / "summary.json",
    INNOVATION_ROOT / "results_shared_protocol_repair" / "morl_dynamic_baselines" / "evcharging_repair_newregimes" / "pgmorl" / "evcharging" / "seed0" / "summary.json",
    INNOVATION_ROOT / "results_shared_protocol_repair" / "morl_dynamic_baselines" / "evcharging_repair_newregimes" / "qpensieve" / "evcharging" / "seed0" / "summary.json",
    INNOVATION_ROOT / "results_shared_protocol_repair" / "morl_dynamic_baselines" / "evcharging_repair_newregimes" / "morlca" / "evcharging" / "seed0" / "summary.json",
    INNOVATION_ROOT / "results_shared_protocol_repair" / "evcharging_longrun_v5b_newregimes_subset8" / "final" / "shared_eval_summary.json",
    INNOVATION_ROOT / "results_shared_protocol_repair" / "evcharging_longrun_v5b_newregimes_subset32" / "final" / "shared_eval_summary.json",
]

METHODS = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca"]
HIGHER_BETTER = {
    "HV": True,
    "EU": True,
    "adapt_score": True,
    "trace_shift_regret": False,
    "trace_recovery_latency": False,
}


def _log(message: str) -> None:
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(text + "\n")


def _file_signature(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return (False, 0, 0)
    stat = path.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


def _current_state() -> tuple[tuple[bool, int, int], ...]:
    return tuple(_file_signature(path) for path in WATCH_FILES)


def _run_report() -> bool:
    cmd = (
        "source /root/anaconda3/etc/profile.d/conda.sh && "
        "conda activate morl-pareto && "
        "python scripts/generate_20_regime_report.py"
    )
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    _log(f"report exit={result.returncode}")
    if result.stdout.strip():
        _log("report stdout:\n" + result.stdout.strip())
    if result.stderr.strip():
        _log("report stderr:\n" + result.stderr.strip())
    return result.returncode == 0


def _load_ev_rows() -> list[dict[str, float | str]]:
    if not CSV_PATH.exists():
        return []
    rows: list[dict[str, float | str]] = []
    with CSV_PATH.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("env_key") != "evcharging":
                continue
            method = str(row.get("method", ""))
            if method not in METHODS:
                continue
            parsed: dict[str, float | str] = {"method": method}
            for key, value in row.items():
                if key.endswith("_mean"):
                    try:
                        parsed[key] = float(value)
                    except Exception:
                        parsed[key] = math.nan
            rows.append(parsed)
    return rows


def _best_methods(rows: list[dict[str, float | str]], metric: str) -> list[str]:
    field = f"{metric}_mean"
    values = [
        (str(row["method"]), float(row.get(field, math.nan)))
        for row in rows
        if math.isfinite(float(row.get(field, math.nan)))
    ]
    if not values:
        return []
    target = max(v for _, v in values) if HIGHER_BETTER[metric] else min(v for _, v in values)
    return [method for method, value in values if abs(value - target) <= 1e-12]


def _acceptance(rows: list[dict[str, float | str]]) -> dict[str, object]:
    methods = {str(row["method"]): row for row in rows}
    dynamic = methods.get("dynamic")
    if dynamic is None:
        return {"ready": False, "reason": "missing dynamic row"}

    hv_eu_first = any("dynamic" in _best_methods(rows, metric) for metric in ["HV", "EU"])
    dyn_metric_wins = sum(
        int("dynamic" in _best_methods(rows, metric))
        for metric in ["adapt_score", "trace_shift_regret", "trace_recovery_latency"]
    )
    baseline_nonzero = False
    for method, row in methods.items():
        if method == "dynamic":
            continue
        regret = float(row.get("trace_shift_regret_mean", math.nan))
        latency = float(row.get("trace_recovery_latency_mean", math.nan))
        if (math.isfinite(regret) and abs(regret) > 1e-12) or (math.isfinite(latency) and abs(latency) > 1e-12):
            baseline_nonzero = True
            break
    return {
        "ready": bool(hv_eu_first and dyn_metric_wins >= 2 and baseline_nonzero),
        "hv_eu_first": hv_eu_first,
        "dyn_metric_wins": dyn_metric_wins,
        "baseline_nonzero": baseline_nonzero,
        "rows": rows,
    }


def _ev_section() -> str:
    if not REPORT_PATH.exists():
        return ""
    lines = REPORT_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    capture = False
    for line in lines:
        if line.strip() == "## evcharging":
            capture = True
        elif capture and line.startswith("## "):
            break
        if capture:
            out.append(line)
    return "\n".join(out)


def main() -> None:
    timeout_seconds = 4 * 60 * 60
    poll_seconds = 60
    deadline = time.time() + timeout_seconds
    last_state = None
    last_report_ok = False

    _log("monitor start")
    while time.time() < deadline:
        state = _current_state()
        if state != last_state:
            _log("detected input change")
            for path, sig in zip(WATCH_FILES, state):
                _log(f"{path}: exists={sig[0]} mtime_ns={sig[1]} size={sig[2]}")
            last_state = state
            last_report_ok = _run_report()
            if last_report_ok:
                rows = _load_ev_rows()
                acceptance = _acceptance(rows)
                _log("acceptance=" + json.dumps(acceptance, ensure_ascii=False, default=str))
                section = _ev_section().strip()
                if section:
                    _log("ev section:\n" + section)
                if bool(acceptance.get("ready")):
                    _log("acceptance satisfied; monitor exit")
                    return
        else:
            if last_report_ok:
                rows = _load_ev_rows()
                acceptance = _acceptance(rows)
                if bool(acceptance.get("ready")):
                    _log("acceptance satisfied without new file changes; monitor exit")
                    return
        time.sleep(poll_seconds)

    _log("monitor timeout")


if __name__ == "__main__":
    main()
