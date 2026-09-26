from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "sustaingym"))

from src.dynamic_morl.dynamic_building import BUILDING_REGIMES, summarize_regime
from sustaingym.envs.cogen import CogenEnv
from sustaingym.envs.evcharging import RealTraceGenerator


FIG_DIR = PROJECT_ROOT / "figures" / "environment_dynamics"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_building() -> list[dict[str, object]]:
    horizon = 24 * 7
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    stats: list[dict[str, object]] = []

    reference = summarize_regime(BUILDING_REGIMES[0], horizon=horizon)
    hours = np.arange(horizon)

    for regime in BUILDING_REGIMES:
        traces = summarize_regime(regime, horizon=horizon)
        axes[0].plot(hours, traces["out_temp"], label=regime.name)
        axes[1].plot(hours, traces["carbon"], label=regime.name)
        axes[2].plot(hours, traces["price"], label=regime.name)

        stats.append(
            {
                "env": "building",
                "regime": regime.name,
                "mean_out_temp": float(traces["out_temp"].mean()),
                "mean_carbon": float(traces["carbon"].mean()),
                "mean_price": float(traces["price"].mean()),
                "wd_out_temp_vs_ref": float(
                    wasserstein_distance(reference["out_temp"], traces["out_temp"])
                ),
                "wd_carbon_vs_ref": float(
                    wasserstein_distance(reference["carbon"], traces["carbon"])
                ),
                "wd_price_vs_ref": float(
                    wasserstein_distance(reference["price"], traces["price"])
                ),
            }
        )

    axes[0].set_title("Building Dynamicity: Outdoor Temperature Regime Shift")
    axes[0].set_ylabel("Temp (C)")
    axes[1].set_title("Building Dynamicity: Carbon Intensity Drift")
    axes[1].set_ylabel("Carbon")
    axes[2].set_title("Building Dynamicity: Electricity Price Drift")
    axes[2].set_ylabel("Price")
    axes[2].set_xlabel("Hour")
    axes[0].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "building_regime_shift.png", dpi=200)
    plt.close(fig)
    return stats


def analyze_evcharging() -> list[dict[str, object]]:
    periods = ["Summer 2019", "Spring 2020", "Summer 2021"]
    traces = {}
    stats: list[dict[str, object]] = []

    for period in periods:
        gen = RealTraceGenerator("caltech", period)
        df = gen.events_df.copy()
        arrival_ts = pd.to_datetime(df["arrival"])
        df["arrival_hour"] = arrival_ts.dt.hour + arrival_ts.dt.minute / 60.0
        traces[period] = df

    ref = traces[periods[0]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for period, df in traces.items():
        hist, bins = np.histogram(df["arrival_hour"], bins=24, range=(0, 24), density=True)
        axes[0].plot((bins[:-1] + bins[1:]) / 2, hist, marker="o", label=period)
        rolling_energy = (
            df.groupby(np.clip(df["arrival_hour"].astype(int), 0, 23))["requested_energy (kWh)"]
            .mean()
            .reindex(range(24), fill_value=np.nan)
            .interpolate(limit_direction="both")
        )
        axes[1].plot(range(24), rolling_energy.values, marker="o", label=period)
        stats.append(
            {
                "env": "evcharging",
                "period": period,
                "sessions": int(len(df)),
                "mean_requested_energy": float(df["requested_energy (kWh)"].mean()),
                "mean_arrival_hour": float(df["arrival_hour"].mean()),
                "wd_arrival_vs_ref": float(
                    wasserstein_distance(ref["arrival_hour"], df["arrival_hour"])
                ),
                "wd_energy_vs_ref": float(
                    wasserstein_distance(
                        ref["requested_energy (kWh)"], df["requested_energy (kWh)"]
                    )
                ),
            }
        )

    axes[0].set_title("EVCharging Dynamicity: Arrival-Time Distribution Shift")
    axes[0].set_xlabel("Arrival Hour")
    axes[0].set_ylabel("Density")
    axes[1].set_title("EVCharging Dynamicity: Requested Energy by Arrival Hour")
    axes[1].set_xlabel("Arrival Hour")
    axes[1].set_ylabel("Requested Energy (kWh)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "evcharging_shift.png", dpi=200)
    plt.close(fig)
    return stats


def analyze_cogen() -> list[dict[str, object]]:
    wind_levels = [0.0, 20.0]
    stats: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    reference_power = None

    for wind in wind_levels:
        env = CogenEnv(renewables_magnitude=wind)
        df = env.ambients_dfs[0].copy()
        env.close()

        power = df["Target Net Power"].to_numpy(dtype=np.float32)
        price = df["Energy Price"].to_numpy(dtype=np.float32)
        steps = np.arange(len(df))
        axes[0].plot(steps, power, label=f"wind={wind}")
        axes[1].plot(steps, price, label=f"wind={wind}")

        if reference_power is None:
            reference_power = power
        stats.append(
            {
                "env": "cogen",
                "renewables_magnitude": wind,
                "mean_target_power": float(power.mean()),
                "std_target_power": float(power.std()),
                "mean_energy_price": float(price.mean()),
                "wd_power_vs_ref": float(
                    0.0 if reference_power is power else wasserstein_distance(reference_power, power)
                ),
            }
        )

    axes[0].set_title("Cogen Dynamicity: Target Net Power Under Renewable Shift")
    axes[0].set_ylabel("Target Power (MW)")
    axes[1].set_title("Cogen Dynamicity: Energy Price Trace")
    axes[1].set_ylabel("Price")
    axes[1].set_xlabel("15-min Step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cogen_shift.png", dpi=200)
    plt.close(fig)
    return stats


def write_report(df: pd.DataFrame) -> None:
    report_path = ANALYSIS_DIR / "environment_dynamics_report.md"
    top_lines = [
        "# Environment Dynamics Report",
        "",
        "This report summarizes the exogenous distribution shift sources used for dynamic MORL experiments.",
        "",
    ]
    for env_name, group in df.groupby("env"):
        top_lines.append(f"## {env_name}")
        top_lines.append("")
        top_lines.append(group.to_markdown(index=False))
        top_lines.append("")
    report_path.write_text("\n".join(top_lines))


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    stats = []
    stats.extend(analyze_building())
    stats.extend(analyze_evcharging())
    stats.extend(analyze_cogen())
    df = pd.DataFrame(stats)
    df.to_csv(ANALYSIS_DIR / "environment_dynamics_summary.csv", index=False)
    write_report(df)
    print(f"Saved figures to {FIG_DIR}")
    print(f"Saved summary to {ANALYSIS_DIR / 'environment_dynamics_summary.csv'}")


if __name__ == "__main__":
    main()
