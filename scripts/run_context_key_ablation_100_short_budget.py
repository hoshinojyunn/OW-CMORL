from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_FILE = Path("OW-CMORL/results.json")

VARIANTS = ("full_window", "h1_current", "h2_no_forecast", "h2_no_history")
METRICS = ("hv", "eu", "regret")


def load_result_data(result_dir: str | Path = DEFAULT_RESULT_FILE) -> dict:
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
    results = load_result_data(result_dir)
    if not isinstance(results.get("LOSSES"), dict):
        raise ValueError(f"Expected a top-level 'LOSSES' object in {result_dir}")
    averaged = {}
    for environment, variants in results["LOSSES"].items():
        averaged[environment] = {}
        for variant, values in variants.items():
            if isinstance(values, dict):
                metric_means = []
                for metric in METRICS:
                    if metric not in values:
                        raise ValueError(f"LOSSES[{environment!r}][{variant!r}] is missing {metric!r}")
                    ratios = np.asarray(values[metric], dtype=float)
                    if ratios.ndim > 1 or ratios.size == 0 or not np.all(np.isfinite(ratios)):
                        raise ValueError(f"Loss {environment}/{variant}/{metric} must contain finite loss ratios")
                    metric_means.append(float(ratios.mean()))
                averaged[environment][variant] = np.asarray(metric_means)
                continue
            ratios = np.asarray(values, dtype=float)
            if ratios.ndim == 1:
                ratios = ratios[None, :]
            if (ratios.ndim != 2 or ratios.shape[1] != 3 or len(ratios) == 0
                    or not np.all(np.isfinite(ratios))):
                raise ValueError(
                    f"LOSSES[{environment!r}][{variant!r}] must contain "
                    "[hv_loss, eu_loss, regret_loss] or repeated triples of finite loss ratios"
                )
            averaged[environment][variant] = ratios.mean(axis=0)
    return averaged


def load_full_results(
    result_dir: str | Path = DEFAULT_RESULT_FILE,
) -> dict[str, dict[str, np.ndarray]]:
    results = load_result_data(result_dir)
    full_results = results.get("FULL_RESULTS")
    if not isinstance(full_results, dict) or not full_results:
        raise ValueError(f"Expected a nonempty top-level 'FULL_RESULTS' object in {result_dir}")

    traces = {}
    for environment, metrics in full_results.items():
        if not isinstance(metrics, dict) or any(metric not in metrics for metric in METRICS):
            raise ValueError(
                f"FULL_RESULTS[{environment!r}] must contain 'hv', 'eu' and 'regret' traces"
            )
        arrays = {metric: np.asarray(metrics[metric], dtype=float) for metric in METRICS}
        if (any(values.ndim != 1 or values.size == 0 for values in arrays.values())
                or len({values.size for values in arrays.values()}) != 1):
            raise ValueError(
                f"FULL_RESULTS[{environment!r}] must contain equal-length 1-D traces"
            )
        if any(not np.all(np.isfinite(values)) for values in arrays.values()):
            raise ValueError(f"FULL_RESULTS[{environment!r}] contains non-finite values")
        traces[environment] = arrays
    return traces


def mean_shift_regret(utility: np.ndarray, shift_interval: int = 6) -> float:
    losses = []
    for shift in range(shift_interval, len(utility), shift_interval):
        pre = utility[max(0, shift - shift_interval) : shift]
        post = utility[shift : min(len(utility), shift + shift_interval)]
        if len(pre) and len(post):
            losses.append(float(np.mean(np.maximum(float(np.mean(pre)) - post, 0.0))))
    return float(np.mean(losses)) if losses else 0.0


def synthetic_utility_trace(baseline_eu: np.ndarray, response_loss: float) -> np.ndarray:
    step = np.arange(len(baseline_eu)) % 6
    transient = np.take(np.asarray([1.0, 0.66, 0.38, 0.16, 0.04, 0.0]), step)
    penalty = abs(float(np.mean(baseline_eu))) * response_loss * (transient - transient.mean())
    return baseline_eu - penalty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_FILE,
        help="JSON with Full HV/EU/Regret, LOSSES metric objects or triples, and optional MEASURED_RESULTS.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "context_key_ablation_100_short_budget",
    )
    parser.add_argument(
        "--max-relative-error", type=float, default=0.1,
        help="Flag deviations above this fraction of the absolute measured mean (default: 0.1).",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_relative_loss(full_values: np.ndarray, loss: float) -> np.ndarray:
    return full_values - np.abs(full_values) * loss


def compare_measured_results(
    summary_rows: list[dict], measured: dict, max_relative_error: float,
) -> list[dict]:
    if not np.isfinite(max_relative_error) or max_relative_error < 0:
        raise ValueError("max_relative_error must be finite and nonnegative")
    if not isinstance(measured, dict):
        raise ValueError("MEASURED_RESULTS must be an environment/variant/metric object")
    comparisons = []
    columns = {metric: f"mean_{metric}" for metric in METRICS}
    for row in summary_rows:
        environment, variant = row["environment"], row["variant"]
        for metric, column in columns.items():
            try:
                values = np.asarray(measured[environment][variant][metric], dtype=float)
            except (KeyError, TypeError) as error:
                raise ValueError(f"Missing measured {environment}/{variant}/{metric}") from error
            if values.ndim > 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"Measured {environment}/{variant}/{metric} must be finite scalar or 1-D values")
            observed = float(values.mean())
            reconstructed = float(row[column])
            deviation = reconstructed - observed
            relative_error = abs(deviation) / abs(observed) if observed != 0 else None
            within_threshold = relative_error <= max_relative_error if relative_error is not None else deviation == 0
            comparisons.append({
                "environment": environment, "variant": variant, "metric": metric,
                "reconstructed_value": reconstructed, "measured_mean": observed,
                "deviation": deviation, "absolute_error": abs(deviation),
                "relative_error": relative_error, "within_threshold": within_threshold,
            })
    return comparisons


def main() -> None:
    args = parse_args()
    if args.shifts != 100:
        raise ValueError("This data check requires 100 input points; the response model uses six-point blocks.")
    if not np.isfinite(args.max_relative_error) or args.max_relative_error < 0:
        raise ValueError("--max-relative-error must be finite and nonnegative")
    result_data = load_result_data(args.result_dir)
    full_results = load_full_results(args.result_dir)
    losses = load_losses(args.result_dir)
    for environment in full_results:
        for variant in VARIANTS[1:]:
            if variant not in losses.get(environment, {}):
                raise ValueError(f"Missing loss ratios for {environment}/{variant}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Use a new or empty output directory to preserve previous data checks")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_shift_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    synthetic_response_rows: list[dict] = []
    report_values: list[dict] = []
    for env_index, (environment, full) in enumerate(full_results.items()):
        full_hv_values = full["hv"]
        full_eu_values = full["eu"]
        full_regret_values = full["regret"]
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
                regret_values = full_regret_values.copy()
                response_loss = 0.015
            else:
                hv_loss, eu_loss, regret_loss = losses[environment][variant]
                hv_values = apply_relative_loss(full_hv_values, hv_loss)
                eu_values = apply_relative_loss(full_eu_values, eu_loss)
                regret_values = apply_relative_loss(full_regret_values, regret_loss)
                response_loss = 0.02 + eu_loss

            mean_eu = float(np.mean(eu_values))
            synthetic_eu_trace = mean_eu + abs(mean_eu) * fluctuation
            synthetic_eu_trace = synthetic_utility_trace(synthetic_eu_trace, response_loss)
            mean_regret = float(np.mean(regret_values))
            response_model_mean_regret = mean_shift_regret(synthetic_eu_trace)
            synthetic_response_rows.extend(
                {"environment": environment, "variant": variant, "point": point,
                 "synthetic_eu": float(value), "response_loss": response_loss,
                 "response_model_mean_regret": response_model_mean_regret}
                for point, value in enumerate(synthetic_eu_trace)
            )
            for shift, (hv, eu, regret) in enumerate(
                zip(hv_values, eu_values, regret_values), start=1
            ):
                per_shift_rows.append(
                    {
                        "environment": environment,
                        "variant": variant,
                        "shift": shift,
                        "hv": float(hv),
                        "eu": float(eu),
                        "regret": float(regret),
                    }
                )
            summary_rows.append(
                {
                    "environment": environment,
                    "variant": variant,
                    "mean_hv": float(np.mean(hv_values)),
                    "mean_eu": mean_eu,
                    "mean_regret": mean_regret,
                    "shifts": args.shifts,
                }
            )
            for metric, value in (("hv", float(np.mean(hv_values))), ("eu", mean_eu),
                                  ("regret", mean_regret)):
                report_values.append({"environment": environment, "variant": variant,
                                      "metric": metric, "reconstructed_value": value})

    comparisons = None
    if "MEASURED_RESULTS" in result_data:
        comparisons = compare_measured_results(summary_rows, result_data["MEASURED_RESULTS"], args.max_relative_error)
        write_csv(args.out_dir / "deviations.csv", comparisons)
    write_csv(args.out_dir / "per_shift_reconstructed_metrics.csv", per_shift_rows)
    write_csv(args.out_dir / "summary_reconstructed_metrics.csv", summary_rows)
    summary = {
        "status": "not_performed_missing_measured_results" if comparisons is None else (
            "within_threshold" if all(row["within_threshold"] for row in comparisons) else "deviations_flagged"
        ),
        "max_relative_error": args.max_relative_error,
        "compared_values": 0 if comparisons is None else len(comparisons),
        "flagged_values": None if comparisons is None else sum(not row["within_threshold"] for row in comparisons),
        "report_values": len(report_values),
        "scope": "model-based numerical cross-check; not proof of experimental correctness",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(args.out_dir)


if __name__ == "__main__":
    main()
