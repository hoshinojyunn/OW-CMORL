from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_shared_protocol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="chlor_owcmorl_improve_v1")
    parser.add_argument(
        "--variant",
        type=str,
        default="full20_base",
        choices=[
            "full20_base",
            "full20_stable_v1",
            "full20_consistency_v1",
        ],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def _variant_updates(variant: str) -> list[str]:
    if variant == "full20_base":
        return []
    if variant == "full20_stable_v1":
        return [
            "--num-time-steps", "4096",
            "--num-init-steps", "2048",
            "--context-history-size", "12",
            "--context-nearest-k", "4",
            "--context-history-lambda", "0.18",
            "--bank-query-slots", "4",
            "--bank-query-topk", "2",
            "--online-shift-threshold", "0.53",
            "--knee-lambda", "0.18",
            "--dynamic-lambda", "0.60",
            "--resilience-lambda", "1.20",
            "--diversity-lambda", "0.16",
            "--shift-gap-lambda", "0.60",
            "--repeat-topk", "2",
            "--resilience-recovery-weight", "0.34",
            "--resilience-regret-weight", "0.18",
            "--resilience-latency-weight", "0.18",
            "--resilience-utility-weight", "0.10",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.12",
            "--dynamic-affinity-weight", "0.60",
            "--dynamic-freshness-weight", "0.18",
            "--dynamic-bank-weight", "0.22",
            "--expert-bank-recovery-weight", "1.10",
            "--expert-bank-regret-weight", "0.38",
            "--expert-bank-latency-weight", "0.30",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.45",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.32",
            "--final-score-bank-weight", "0.18",
            "--final-score-affinity-weight", "0.12",
            "--final-score-train-iter-weight", "0.04",
            "--final-archive-max-samples", "72",
        ]
    if variant == "full20_consistency_v1":
        return [
            "--num-time-steps", "4608",
            "--num-init-steps", "2304",
            "--context-history-size", "12",
            "--context-nearest-k", "4",
            "--context-history-lambda", "0.16",
            "--bank-merge-threshold", "0.78",
            "--bank-query-slots", "4",
            "--bank-query-topk", "3",
            "--online-shift-threshold", "0.54",
            "--knee-lambda", "0.18",
            "--dynamic-lambda", "0.68",
            "--resilience-lambda", "1.12",
            "--diversity-lambda", "0.18",
            "--shift-gap-lambda", "0.70",
            "--repeat-topk", "3",
            "--resilience-recovery-weight", "0.32",
            "--resilience-regret-weight", "0.16",
            "--resilience-latency-weight", "0.16",
            "--resilience-utility-weight", "0.12",
            "--resilience-gap-weight", "0.08",
            "--resilience-bank-weight", "0.16",
            "--dynamic-affinity-weight", "0.58",
            "--dynamic-freshness-weight", "0.14",
            "--dynamic-bank-weight", "0.28",
            "--expert-bank-recovery-weight", "1.16",
            "--expert-bank-regret-weight", "0.34",
            "--expert-bank-latency-weight", "0.30",
            "--expert-bank-gap-weight", "0.08",
            "--final-score-recovery-weight", "1.35",
            "--final-score-regret-weight", "0.30",
            "--final-score-latency-weight", "0.28",
            "--final-score-bank-weight", "0.20",
            "--final-score-affinity-weight", "0.14",
            "--final-score-train-iter-weight", "0.04",
            "--final-archive-max-samples", "80",
        ]
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    args = parse_args()
    save_dir = RESULTS_ROOT / args.run_name
    if save_dir.exists():
        if not args.force:
            raise FileExistsError(save_dir)
        import shutil
        shutil.rmtree(save_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.dynamic_morl.run",
        "--env-name", "chlor_alkali_dynamic",
        "--obj-num", "3",
        "--ref-point", "0", "0", "0",
        "--auto-ref-point",
        "--num-time-steps", "3072",
        "--num-init-steps", "1536",
        "--num-steps", "8",
        "--num-processes", "1",
        "--ppo-epoch", "8",
        "--num-mini-batch", "4",
        "--entropy-coef", "0.0",
        "--num-select", "6",
        "--delta-weight", "0.2",
        "--eval-delta-weight", "0.5",
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
        "--fine-regime-catalog-episodes", "256",
        "--fine-regime-context-steps", "4",
        "--context-probe-samples", "3",
        "--context-trace-steps", "6",
        "--context-sample-mode", "salient_mix",
        "--context-saliency-alpha", "0.35",
        "--context-buffer-steps", "48",
        "--context-history-size", "10",
        "--context-nearest-k", "3",
        "--context-history-lambda", "0.15",
        "--context-forecast-lambda", "0.5",
        "--bank-enable", "1",
        "--bank-num-slots", "8",
        "--bank-slot-size", "4",
        "--bank-merge-threshold", "0.80",
        "--bank-query-slots", "3",
        "--bank-query-topk", "2",
        "--online-shift-threshold", "0.55",
        "--online-shift-min-gap", "6",
        "--chlor-episode-length", "96",
        "--ob-rms",
        "--obj-rms",
        "--selection-method", "online-window",
        "--knee-lambda", "0.15",
        "--dynamic-lambda", "0.5",
        "--resilience-lambda", "1.0",
        "--diversity-lambda", "0.15",
        "--shift-gap-lambda", "0.5",
        "--repeat-topk", "2",
        "--final-archive-mode", "union",
        "--final-archive-max-samples", "64",
        "--seed", "0",
        "--save-dir", str(save_dir),
    ]
    cmd = _replace_or_append_args(cmd, _variant_updates(args.variant))
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
