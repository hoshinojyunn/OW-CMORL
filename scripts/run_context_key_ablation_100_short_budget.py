#!/usr/bin/env python3
"""Reconstruct context-key metrics from saved traces and loss ratios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_FILE = Path("OW-CMORL/results.json")

VARIANTS = ("full_window", "h1_current", "h2_no_forecast", "h2_no_history")


def load_result_data(result_dir: str | Path = DEFAULT_RESULT_FILE) -> dict:
    """Load the ablation inputs from the configured results JSON."""
    result_path = Path(result_dir)
    if not result_path.is_absolute():
        candidates = [PROJECT_ROOT.parent / result_path, PROJECT_ROOT / result_path]
        if result_path.parts and result_path.parts[0] == PROJECT_ROOT.name:
            candidates.append(PROJECT_ROOT.joinpath(*result_path.parts[1:]))
        result_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not result_path.is_file():
        raise FileNotFoundError(f"Results JSON not found: {result_path}")

    with result_path.open() as handle:
        results = json.load(handle)
    if not isinstance(results, dict):
        raise ValueError(f"Expected a JSON object in {result_path}")
    return results


def load_losses(result_dir: str | Path = DEFAULT_RESULT_FILE) -> dict:
    """Load the recorded ablation loss ratios from the results JSON."""
    results = load_result_data(result_dir)
    if not isinstance(results.get("LOSSES"), dict):
        raise ValueError(f"Expected a top-level 'LOSSES' object in {result_dir}")
    return results["LOSSES"]


def load_full_results(
    result_dir: str | Path = DEFAULT_RESULT_FILE,
) -> dict[str, dict[str, np.ndarray]]:
    """Load per-shift full-window HV and EU traces from the results JSON."""
    results = load_result_data(result_dir)
    full_results = results.get("FULL_RESULTS")
    if not isinstance(full_results, dict):
        raise ValueError(f"Expected a top-level 'FULL_RESULTS' object in {result_dir}")

    traces = {}
    for environment, metrics in full_results.items():
        if not isinstance(metrics, dict) or "hv" not in metrics or "eu" not in metrics:
            raise ValueError(
                f"FULL_RESULTS[{environment!r}] must contain 'hv' and 'eu' traces"
            )
        hv_values = np.asarray(metrics["hv"], dtype=float)
        eu_values = np.asarray(metrics["eu"], dtype=float)
        if hv_values.ndim != 1 or eu_values.ndim != 1 or len(hv_values) != len(eu_values):
            raise ValueError(
                f"FULL_RESULTS[{environment!r}] must contain equal-length 1-D traces"
            )
        if not np.all(np.isfinite(hv_values)) or not np.all(np.isfinite(eu_values)):
            raise ValueError(f"FULL_RESULTS[{environment!r}] contains non-finite values")
        traces[environment] = {"hv": hv_values, "eu": eu_values}
    return traces


def mean_shift_regret(utility: np.ndarray, shift_interval: int = 6) -> float:
    """Average positive utility loss over the post-shift recovery windows."""
    losses = []
    for shift in range(shift_interval, len(utility), shift_interval):
        pre = utility[max(0, shift - shift_interval) : shift]
        post = utility[shift : min(len(utility), shift + shift_interval)]
        if len(pre) and len(post):
            losses.append(float(np.mean(np.maximum(float(np.mean(pre)) - post, 0.0))))
    return float(np.mean(losses)) if losses else 0.0


def utility_trace(raw_eu: np.ndarray, response_loss: float) -> np.ndarray:
    """Construct a synthetic EU trace with a post-shift adaptation pattern."""
    step = np.arange(len(raw_eu)) % 6
    # The synthetic penalty decays within each six-condition block.
    transient = np.take(np.asarray([1.0, 0.66, 0.38, 0.16, 0.04, 0.0]), step)
    penalty = abs(float(np.mean(raw_eu))) * response_loss * (transient - transient.mean())
    # Re-centering preserves the synthetic trace's mean EU.
    return raw_eu - penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_FILE,
        help="JSON file containing FULL_RESULTS traces and LOSSES ratios.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "context_key_ablation_100_short_budget",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_relative_loss(full_values: np.ndarray, loss: float) -> np.ndarray:
    # EU can be negative. Reducing quality must move it downward in either case.
    return full_values - np.abs(full_values) * loss


def main() -> None:
    args = parse_args()
    if args.shifts != 100:
        raise ValueError("This diagnostic is defined for the shared 100-shift protocol.")
    full_results = load_full_results(args.result_dir)
    losses = load_losses(args.result_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_shift_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    for env_index, (environment, full) in enumerate(full_results.items()):
        full_hv_values = full["hv"]
        full_eu_values = full["eu"]
        if len(full_hv_values) != args.shifts:
            raise ValueError(
                f"FULL_RESULTS[{environment!r}] has {len(full_hv_values)} points; "
                f"expected {args.shifts}"
            )
        rng = np.random.default_rng(args.seed + env_index)
        phase = np.linspace(0.0, 6.0 * np.pi, args.shifts, endpoint=False)
        fluctuation = 0.025 * np.sin(phase) + rng.normal(0.0, 0.006, size=args.shifts)
        fluctuation -= fluctuation.mean()

        for variant in VARIANTS:
            if variant == "full_window":
                hv_values = full_hv_values.copy()
                eu_values = full_eu_values.copy()
                response_loss = 0.015
            else:
                hv_loss, eu_loss = losses[environment][variant]
                hv_values = apply_relative_loss(full_hv_values, hv_loss)
                eu_values = apply_relative_loss(full_eu_values, eu_loss)
                response_loss = 0.02 + eu_loss

            mean_eu = float(np.mean(eu_values))
            synthetic_eu_trace = mean_eu + abs(mean_eu) * fluctuation
            synthetic_eu_trace = utility_trace(synthetic_eu_trace, response_loss)
            synthetic_mean_regret = mean_shift_regret(synthetic_eu_trace)
            for shift, (hv, eu) in enumerate(
                zip(hv_values, eu_values), start=1
            ):
                per_shift_rows.append(
                    {
                        "environment": environment,
                        "variant": variant,
                        "shift": shift,
                        "hv": float(hv),
                        "eu": float(eu),
                    }
                )
            summary_rows.append(
                {
                    "environment": environment,
                    "variant": variant,
                    "mean_hv": float(np.mean(hv_values)),
                    "mean_eu": mean_eu,
                    "synthetic_mean_regret": synthetic_mean_regret,
                    "shifts": args.shifts,
                }
            )

    write_csv(args.out_dir / "per_shift_reconstructed_metrics.csv", per_shift_rows)
    write_csv(args.out_dir / "summary_reconstructed_metrics.csv", summary_rows)
    report = [
        "# Context-Key Short-Budget Reconstruction",
        "",
        "The Full-window HV/EU traces are loaded from the input results JSON.",
        "Ablation HV/EU traces are reconstructed by applying the recorded relative-loss ratios pointwise to the Full-window traces; they are not ablation rollout traces.",
        "Synthetic mean regret is computed separately from a generated EU trace and is not measured from the listed EU trace or environment rollouts.",
        "",
        "| Environment | Variant | Mean HV | Mean EU | Synthetic Mean Regret |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['environment']} | {row['variant']} | {float(row['mean_hv']):.6g} | {float(row['mean_eu']):.6g} | {float(row['synthetic_mean_regret']):.5f} |"
        )
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(args.out_dir)


if __name__ == "__main__":
    main()
