from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_shared_protocol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="optional config names to run; defaults to all configured chlor candidates",
    )
    return parser.parse_args()


def _base_cmd(run_name: str) -> list[str]:
    save_dir = RESULTS_ROOT / run_name
    return [
        sys.executable,
        "-m",
        "src.dynamic_morl.run",
        "--env-name", "chlor_alkali_dynamic",
        "--obj-num", "3",
        "--ref-point", "0", "0", "0",
        "--auto-ref-point",
        "--num-time-steps", "2048",
        "--num-init-steps", "1024",
        "--num-steps", "8",
        "--num-processes", "1",
        "--ppo-epoch", "6",
        "--num-mini-batch", "4",
        "--eval-num", "2",
        "--rl-eval-interval", "4",
        "--drift-window", "6",
        "--drift-steps", "4",
        "--strict-online-context",
        "--episodes-per-regime", "1",
        "--regime-schedule", "cyclic",
        "--trace-eval-samples", "2",
        "--regime-eval-samples", "2",
        "--shared-regime-eval-episodes", "20",
        "--shared-regime-seed-offset", "0",
        "--fine-regime-clusters", "20",
        "--fine-regime-catalog-episodes", "64",
        "--fine-regime-context-steps", "4",
        "--context-sample-mode", "salient_mix",
        "--chlor-episode-length", "96",
        "--ob-rms",
        "--obj-rms",
        "--selection-method", "online-window",
        "--final-archive-mode", "union",
        "--seed", "0",
        "--save-dir", str(save_dir),
    ]


def _replace_or_append_args(cmd: list[str], updates: list[str]) -> list[str]:
    out = list(cmd)
    i = 0
    while i < len(updates):
        key = updates[i]
        if not str(key).startswith("--"):
            raise ValueError(f"Expected flag at position {i}: {key}")
        values: list[str] = []
        j = i + 1
        while j < len(updates) and not str(updates[j]).startswith("--"):
            values.append(updates[j])
            j += 1
        while key in out:
            idx = out.index(key)
            del out[idx]
            while idx < len(out) and not str(out[idx]).startswith("--"):
                del out[idx]
        out.append(key)
        out.extend(values)
        i = j
    return out


def _recovery_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_recovery_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--entropy-coef", "0.002",
            "--num-select", "4",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--context-probe-samples", "2",
            "--context-trace-steps", "4",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "24",
            "--context-history-size", "6",
            "--context-nearest-k", "1",
            "--context-history-lambda", "0.10",
            "--context-forecast-lambda", "0.40",
            "--bank-enable", "1",
            "--bank-num-slots", "8",
            "--bank-slot-size", "3",
            "--bank-merge-threshold", "0.84",
            "--bank-query-slots", "4",
            "--bank-query-topk", "2",
            "--online-shift-threshold", "0.58",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.08",
            "--dynamic-lambda", "0.18",
            "--resilience-lambda", "1.20",
            "--diversity-lambda", "0.12",
            "--shift-gap-lambda", "0.12",
            "--repeat-topk", "1",
            "--resilience-recovery-weight", "0.44",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.24",
            "--resilience-utility-weight", "0.05",
            "--resilience-gap-weight", "0.03",
            "--resilience-bank-weight", "0.06",
            "--dynamic-affinity-weight", "0.60",
            "--dynamic-freshness-weight", "0.10",
            "--dynamic-bank-weight", "0.30",
            "--expert-bank-recovery-weight", "1.40",
            "--expert-bank-regret-weight", "0.30",
            "--expert-bank-latency-weight", "0.50",
            "--expert-bank-gap-weight", "0.05",
            "--final-archive-max-samples", "24",
            "--final-score-recovery-weight", "1.70",
            "--final-score-regret-weight", "0.24",
            "--final-score-latency-weight", "0.55",
            "--final-score-bank-weight", "0.20",
            "--final-score-affinity-weight", "0.14",
            "--final-score-train-iter-weight", "0.02",
        ]
    )
    return run_name, cmd


def _balanced_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_balanced_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--entropy-coef", "0.0",
            "--num-select", "4",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--context-probe-samples", "3",
            "--context-trace-steps", "6",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "32",
            "--context-history-size", "8",
            "--context-nearest-k", "2",
            "--context-history-lambda", "0.12",
            "--context-forecast-lambda", "0.45",
            "--bank-enable", "1",
            "--bank-num-slots", "8",
            "--bank-slot-size", "4",
            "--bank-merge-threshold", "0.82",
            "--bank-query-slots", "3",
            "--bank-query-topk", "2",
            "--online-shift-threshold", "0.56",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.14",
            "--dynamic-lambda", "0.34",
            "--resilience-lambda", "0.90",
            "--diversity-lambda", "0.14",
            "--shift-gap-lambda", "0.24",
            "--repeat-topk", "2",
            "--resilience-recovery-weight", "0.34",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.18",
            "--resilience-utility-weight", "0.08",
            "--resilience-gap-weight", "0.06",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.58",
            "--dynamic-freshness-weight", "0.14",
            "--dynamic-bank-weight", "0.28",
            "--expert-bank-recovery-weight", "1.15",
            "--expert-bank-regret-weight", "0.34",
            "--expert-bank-latency-weight", "0.36",
            "--expert-bank-gap-weight", "0.08",
            "--final-archive-max-samples", "28",
            "--final-score-recovery-weight", "1.30",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.35",
            "--final-score-bank-weight", "0.22",
            "--final-score-affinity-weight", "0.16",
            "--final-score-train-iter-weight", "0.03",
        ]
    )
    return run_name, cmd


def _deepunion_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_deepunion_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "8192",
            "--num-init-steps", "3072",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.005",
            "--num-select", "4",
            "--delta-weight", "0.25",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "10",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "2",
            "--context-trace-steps", "4",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "32",
            "--context-history-size", "6",
            "--context-nearest-k", "1",
            "--context-history-lambda", "0.10",
            "--context-forecast-lambda", "0.50",
            "--bank-enable", "1",
            "--bank-num-slots", "6",
            "--bank-slot-size", "2",
            "--bank-merge-threshold", "0.82",
            "--bank-query-slots", "1",
            "--bank-query-topk", "1",
            "--online-shift-threshold", "0.60",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.05",
            "--dynamic-lambda", "0.03",
            "--resilience-lambda", "0.08",
            "--diversity-lambda", "0.32",
            "--shift-gap-lambda", "0.02",
            "--repeat-topk", "1",
            "--final-archive-mode", "union",
            "--final-archive-max-samples", "32",
            "--resilience-recovery-weight", "0.32",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.12",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.14",
            "--dynamic-affinity-weight", "0.62",
            "--dynamic-freshness-weight", "0.12",
            "--dynamic-bank-weight", "0.26",
            "--expert-bank-recovery-weight", "1.10",
            "--expert-bank-regret-weight", "0.36",
            "--expert-bank-latency-weight", "0.28",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.35",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.32",
            "--final-score-bank-weight", "0.18",
            "--final-score-affinity-weight", "0.11",
            "--final-score-train-iter-weight", "0.05",
        ]
    )
    return run_name, cmd


def _deepblend_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_deepblend_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "6144",
            "--num-init-steps", "2048",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.003",
            "--num-select", "4",
            "--delta-weight", "0.25",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "8",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "3",
            "--context-trace-steps", "6",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "48",
            "--context-history-size", "8",
            "--context-nearest-k", "2",
            "--context-history-lambda", "0.12",
            "--context-forecast-lambda", "0.50",
            "--bank-enable", "1",
            "--bank-num-slots", "6",
            "--bank-slot-size", "3",
            "--bank-merge-threshold", "0.82",
            "--bank-query-slots", "2",
            "--bank-query-topk", "1",
            "--online-shift-threshold", "0.58",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.08",
            "--dynamic-lambda", "0.08",
            "--resilience-lambda", "0.18",
            "--diversity-lambda", "0.26",
            "--shift-gap-lambda", "0.08",
            "--repeat-topk", "1",
            "--final-archive-mode", "union",
            "--final-archive-max-samples", "28",
            "--resilience-recovery-weight", "0.34",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.18",
            "--resilience-utility-weight", "0.10",
            "--resilience-gap-weight", "0.07",
            "--resilience-bank-weight", "0.13",
            "--dynamic-affinity-weight", "0.60",
            "--dynamic-freshness-weight", "0.12",
            "--dynamic-bank-weight", "0.28",
            "--expert-bank-recovery-weight", "1.20",
            "--expert-bank-regret-weight", "0.34",
            "--expert-bank-latency-weight", "0.34",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.45",
            "--final-score-regret-weight", "0.28",
            "--final-score-latency-weight", "0.36",
            "--final-score-bank-weight", "0.18",
            "--final-score-affinity-weight", "0.12",
            "--final-score-train-iter-weight", "0.04",
        ]
    )
    return run_name, cmd


def _paretobridge_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_paretobridge_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "8192",
            "--num-init-steps", "3072",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.002",
            "--num-select", "6",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "8",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "3",
            "--context-trace-steps", "6",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "48",
            "--context-history-size", "8",
            "--context-nearest-k", "2",
            "--context-history-lambda", "0.12",
            "--context-forecast-lambda", "0.50",
            "--bank-enable", "1",
            "--bank-num-slots", "8",
            "--bank-slot-size", "4",
            "--bank-merge-threshold", "0.82",
            "--bank-query-slots", "2",
            "--bank-query-topk", "1",
            "--online-shift-threshold", "0.56",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.10",
            "--dynamic-lambda", "0.18",
            "--resilience-lambda", "0.35",
            "--diversity-lambda", "0.24",
            "--shift-gap-lambda", "0.18",
            "--repeat-topk", "1",
            "--final-archive-mode", "pareto",
            "--resilience-recovery-weight", "0.32",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.10",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.62",
            "--dynamic-freshness-weight", "0.12",
            "--dynamic-bank-weight", "0.26",
            "--expert-bank-recovery-weight", "1.05",
            "--expert-bank-regret-weight", "0.40",
            "--expert-bank-latency-weight", "0.28",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.10",
            "--final-score-regret-weight", "0.32",
            "--final-score-latency-weight", "0.24",
            "--final-score-bank-weight", "0.14",
            "--final-score-affinity-weight", "0.08",
            "--final-score-train-iter-weight", "0.06",
        ]
    )
    return run_name, cmd


def _paretodeep_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_paretodeep_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "12288",
            "--num-init-steps", "4096",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.004",
            "--num-select", "5",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "10",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "2",
            "--context-trace-steps", "4",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "32",
            "--context-history-size", "6",
            "--context-nearest-k", "1",
            "--context-history-lambda", "0.10",
            "--context-forecast-lambda", "0.50",
            "--bank-enable", "1",
            "--bank-num-slots", "6",
            "--bank-slot-size", "2",
            "--bank-merge-threshold", "0.82",
            "--bank-query-slots", "1",
            "--bank-query-topk", "1",
            "--online-shift-threshold", "0.58",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.06",
            "--dynamic-lambda", "0.08",
            "--resilience-lambda", "0.16",
            "--diversity-lambda", "0.34",
            "--shift-gap-lambda", "0.04",
            "--repeat-topk", "1",
            "--final-archive-mode", "pareto",
            "--resilience-recovery-weight", "0.30",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.12",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.64",
            "--dynamic-freshness-weight", "0.12",
            "--dynamic-bank-weight", "0.24",
            "--expert-bank-recovery-weight", "1.00",
            "--expert-bank-regret-weight", "0.40",
            "--expert-bank-latency-weight", "0.28",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.00",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.22",
            "--final-score-bank-weight", "0.12",
            "--final-score-affinity-weight", "0.08",
            "--final-score-train-iter-weight", "0.08",
        ]
    )
    return run_name, cmd


def _frontierlift_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_frontierlift_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "9216",
            "--num-init-steps", "3072",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.003",
            "--num-select", "6",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "8",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "3",
            "--context-trace-steps", "6",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "48",
            "--context-history-size", "10",
            "--context-nearest-k", "3",
            "--context-history-lambda", "0.14",
            "--context-forecast-lambda", "0.50",
            "--bank-enable", "1",
            "--bank-num-slots", "8",
            "--bank-slot-size", "4",
            "--bank-merge-threshold", "0.80",
            "--bank-query-slots", "3",
            "--bank-query-topk", "2",
            "--online-shift-threshold", "0.55",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.16",
            "--dynamic-lambda", "0.26",
            "--resilience-lambda", "0.78",
            "--diversity-lambda", "0.28",
            "--shift-gap-lambda", "0.20",
            "--repeat-topk", "2",
            "--final-archive-mode", "union",
            "--final-archive-max-samples", "36",
            "--resilience-recovery-weight", "0.32",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.10",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.60",
            "--dynamic-freshness-weight", "0.12",
            "--dynamic-bank-weight", "0.28",
            "--expert-bank-recovery-weight", "1.08",
            "--expert-bank-regret-weight", "0.36",
            "--expert-bank-latency-weight", "0.30",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.22",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.28",
            "--final-score-bank-weight", "0.18",
            "--final-score-affinity-weight", "0.10",
            "--final-score-train-iter-weight", "0.05",
        ]
    )
    return run_name, cmd


def _frontierlift_v2_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_frontierlift_v2"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "12288",
            "--num-init-steps", "4096",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.002",
            "--num-select", "6",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "8",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "3",
            "--context-trace-steps", "6",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "48",
            "--context-history-size", "12",
            "--context-nearest-k", "4",
            "--context-history-lambda", "0.14",
            "--context-forecast-lambda", "0.52",
            "--bank-enable", "1",
            "--bank-num-slots", "8",
            "--bank-slot-size", "4",
            "--bank-merge-threshold", "0.78",
            "--bank-query-slots", "4",
            "--bank-query-topk", "3",
            "--online-shift-threshold", "0.54",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.18",
            "--dynamic-lambda", "0.22",
            "--resilience-lambda", "0.72",
            "--diversity-lambda", "0.34",
            "--shift-gap-lambda", "0.16",
            "--repeat-topk", "3",
            "--final-archive-mode", "union",
            "--final-archive-max-samples", "48",
            "--resilience-recovery-weight", "0.32",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.10",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.58",
            "--dynamic-freshness-weight", "0.12",
            "--dynamic-bank-weight", "0.30",
            "--expert-bank-recovery-weight", "1.08",
            "--expert-bank-regret-weight", "0.34",
            "--expert-bank-latency-weight", "0.28",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.18",
            "--final-score-regret-weight", "0.28",
            "--final-score-latency-weight", "0.26",
            "--final-score-bank-weight", "0.18",
            "--final-score-affinity-weight", "0.10",
            "--final-score-train-iter-weight", "0.05",
        ]
    )
    return run_name, cmd


def _obj02bridge_cmd() -> tuple[str, list[str]]:
    run_name = "chlor_owcmorl_search_obj02bridge_v1"
    cmd = _replace_or_append_args(
        _base_cmd(run_name),
        [
            "--num-time-steps", "9216",
            "--num-init-steps", "3072",
            "--ppo-epoch", "8",
            "--entropy-coef", "0.001",
            "--num-select", "6",
            "--delta-weight", "0.2",
            "--eval-delta-weight", "0.5",
            "--rl-eval-interval", "8",
            "--fine-regime-catalog-episodes", "256",
            "--context-probe-samples", "3",
            "--context-trace-steps", "6",
            "--context-saliency-alpha", "0.35",
            "--context-buffer-steps", "48",
            "--context-history-size", "12",
            "--context-nearest-k", "4",
            "--context-history-lambda", "0.16",
            "--context-forecast-lambda", "0.52",
            "--bank-enable", "1",
            "--bank-num-slots", "10",
            "--bank-slot-size", "4",
            "--bank-merge-threshold", "0.78",
            "--bank-query-slots", "4",
            "--bank-query-topk", "3",
            "--online-shift-threshold", "0.54",
            "--online-shift-min-gap", "6",
            "--knee-lambda", "0.18",
            "--dynamic-lambda", "0.44",
            "--resilience-lambda", "0.98",
            "--diversity-lambda", "0.24",
            "--shift-gap-lambda", "0.42",
            "--repeat-topk", "3",
            "--final-archive-mode", "union",
            "--final-archive-max-samples", "96",
            "--resilience-recovery-weight", "0.32",
            "--resilience-regret-weight", "0.16",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.12",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.56",
            "--dynamic-freshness-weight", "0.16",
            "--dynamic-bank-weight", "0.28",
            "--expert-bank-recovery-weight", "1.18",
            "--expert-bank-regret-weight", "0.34",
            "--expert-bank-latency-weight", "0.28",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.32",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.28",
            "--final-score-bank-weight", "0.20",
            "--final-score-affinity-weight", "0.14",
            "--final-score-train-iter-weight", "0.04",
        ]
    )
    return run_name, cmd


def _clean_dir(run_name: str, force: bool) -> None:
    save_dir = RESULTS_ROOT / run_name
    if save_dir.exists():
        if not force:
            raise FileExistsError(save_dir)
        shutil.rmtree(save_dir)


def _run(cmd: list[str], run_name: str) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    print(f"=== {run_name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> None:
    args = parse_args()
    config_builders = {
        "recovery_v1": _recovery_cmd,
        "balanced_v1": _balanced_cmd,
        "deepunion_v1": _deepunion_cmd,
        "deepblend_v1": _deepblend_cmd,
        "paretobridge_v1": _paretobridge_cmd,
        "paretodeep_v1": _paretodeep_cmd,
        "frontierlift_v1": _frontierlift_cmd,
        "frontierlift_v2": _frontierlift_v2_cmd,
        "obj02bridge_v1": _obj02bridge_cmd,
    }
    selected_names = list(args.only) if args.only else list(config_builders.keys())
    unknown = [name for name in selected_names if name not in config_builders]
    if unknown:
        raise ValueError(f"Unknown config(s): {unknown}")
    configs = [config_builders[name]() for name in selected_names]
    for run_name, _cmd in configs:
        _clean_dir(run_name, args.force)
    for run_name, cmd in configs:
        _run(cmd, run_name)


if __name__ == "__main__":
    main()
