from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.chlor_alkali_price import ANHUI_CHLOR_ALKALI_35KV_PROXY


DEFAULT_INPUT = (
    PROJECT_ROOT
   
    / "results_shared_protocol"
    / "sharedprot_main_chlor_alkali_dynamic_seed0.20260604_protocolfix"
    / "regime_fronts"
    / "shared_final.json"
)
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures" / "chlor_alkali_regime_analysis"
MIN_TOTAL_CURRENT_KA = 86.0
MAX_TOTAL_CURRENT_KA = 119.0
PRICE_LEVELS = {
    "summer_peak": ANHUI_CHLOR_ALKALI_35KV_PROXY.summer_peak,
    "regular_peak": ANHUI_CHLOR_ALKALI_35KV_PROXY.regular_peak,
    "flat": ANHUI_CHLOR_ALKALI_35KV_PROXY.regular_flat,
    "valley": ANHUI_CHLOR_ALKALI_35KV_PROXY.regular_valley,
}
PRICE_LOW = min(PRICE_LEVELS.values())
PRICE_HIGH = max(PRICE_LEVELS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export canonical chlor_alkali regime parameters and plot price/current-limit distributions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def _denormalize(value: float, low: float, high: float) -> float:
    return float(((float(value) + 1.0) * 0.5) * (high - low) + low)


def _nearest_price_label(price: float) -> str:
    return min(PRICE_LEVELS.items(), key=lambda item: abs(float(item[1]) - float(price)))[0]


def _resolve_current_price(meta: dict[str, float]) -> tuple[float, str]:
    value = meta.get("current_price")
    if value is not None:
        return float(value), "stored_current_price"
    value = meta.get("price_norm")
    if value is not None:
        return _denormalize(float(value), PRICE_LOW, PRICE_HIGH), "denorm_price_norm"
    return math.nan, "missing"


def _resolve_current_limit_ratio(meta: dict[str, float]) -> tuple[float, str]:
    for key in ("current_limit_ratio_proxy", "current_limit_ratio"):
        value = meta.get(key)
        if value is not None:
            return float(value), key
    return math.nan, "missing"


def _resolve_current_limit_ka(meta: dict[str, float]) -> tuple[float, str]:
    value = meta.get("current_limit_kA_proxy")
    if value is not None:
        return float(value), "stored_current_limit_kA_proxy"
    ratio, ratio_source = _resolve_current_limit_ratio(meta)
    if math.isnan(ratio):
        return math.nan, "missing"
    return float(ratio * MAX_TOTAL_CURRENT_KA), f"ratio_to_kA:{ratio_source}"


def load_regime_table(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    rows: list[dict[str, float | int | str]] = []
    for item in payload["regimes"]:
        meta = item.get("regime_meta", {})
        current_price, price_source = _resolve_current_price(meta)
        current_limit_ratio, limit_ratio_source = _resolve_current_limit_ratio(meta)
        current_limit_ka, limit_ka_source = _resolve_current_limit_ka(meta)
        rows.append(
            {
                "regime_id": int(item["regime_id"]),
                "regime": str(item["regime"]),
                "hour_sin": float(meta.get("hour_sin", math.nan)),
                "hour_cos": float(meta.get("hour_cos", math.nan)),
                "month_sin": float(meta.get("month_sin", math.nan)),
                "month_cos": float(meta.get("month_cos", math.nan)),
                "price_norm": float(meta.get("price_norm", math.nan)),
                "current_price": current_price,
                "price_band_guess": _nearest_price_label(current_price) if not math.isnan(current_price) else "unknown",
                "price_source": price_source,
                "peak_flag": float(meta.get("peak_flag", math.nan)),
                "valley_flag": float(meta.get("valley_flag", math.nan)),
                "glo_temp_norm": float(meta.get("glo_temp_norm", meta.get("glo_temp", math.nan))),
                "temp_naoh_norm": float(meta.get("temp_naoh_norm", math.nan)),
                "active_cell_ratio": float(meta.get("active_cell_ratio", math.nan)),
                "aging_index": float(meta.get("aging_index", meta.get("aging_proxy", math.nan))),
                "current_limit_ratio": current_limit_ratio,
                "current_limit_kA": current_limit_ka,
                "limit_ratio_source": limit_ratio_source,
                "limit_kA_source": limit_ka_source,
                "flow_imbalance_norm": float(meta.get("flow_imbalance_norm", meta.get("flow_imbalance", math.nan))),
                "stress_index": float(meta.get("stress_index", meta.get("stress_proxy", math.nan))),
                "eff_gap_ema": float(meta.get("eff_gap_ema", math.nan)),
            }
        )
    df = pd.DataFrame(rows).sort_values("regime_id").reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, analysis_dir: Path) -> tuple[Path, Path]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "chlor_alkali_regime_catalog.csv"
    md_path = analysis_dir / "chlor_alkali_regime_catalog.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))
    return csv_path, md_path


def plot_distributions(df: pd.DataFrame, fig_dir: Path) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.24)
    ax_joint = fig.add_subplot(grid[0, :])
    ax_price = fig.add_subplot(grid[1, 0])
    ax_limit = fig.add_subplot(grid[1, 1])

    scatter = ax_joint.scatter(
        df["current_price"],
        df["current_limit_kA"],
        c=df["regime_id"],
        cmap="viridis",
        s=80,
        edgecolor="black",
        linewidth=0.5,
    )
    for row in df.itertuples(index=False):
        ax_joint.annotate(
            str(row.regime_id),
            (row.current_price, row.current_limit_kA),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    cbar = fig.colorbar(scatter, ax=ax_joint, pad=0.01)
    cbar.set_label("Regime ID")
    ax_joint.set_title("chlor_alkali Shared 20-Regime Combinations")
    ax_joint.set_xlabel("Electricity Price (RMB/kWh)")
    ax_joint.set_ylabel("Total Current Limit (kA)")

    sns.histplot(df["current_price"], bins=min(8, len(df)), kde=True, ax=ax_price, color="#1f77b4")
    sns.rugplot(df["current_price"], ax=ax_price, color="black", height=0.08)
    ax_price.set_title("Electricity Price Distribution")
    ax_price.set_xlabel("Electricity Price (RMB/kWh)")
    ax_price.set_ylabel("Count")

    sns.histplot(
        df["current_limit_kA"],
        bins=min(8, len(df)),
        kde=True,
        ax=ax_limit,
        color="#ff7f0e",
    )
    sns.rugplot(df["current_limit_kA"], ax=ax_limit, color="black", height=0.08)
    ax_limit.set_title("Total Current Limit Distribution")
    ax_limit.set_xlabel("Total Current Limit (kA)")
    ax_limit.set_ylabel("Count")

    fig.suptitle("chlor_alkali Shared-Protocol Regime Parameters", fontsize=15, y=0.98)
    out_path = fig_dir / "chlor_alkali_regime_price_current_limit.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    df = load_regime_table(args.input)
    csv_path, md_path = write_outputs(df, args.analysis_dir)
    fig_path = plot_distributions(df, args.fig_dir)

    print(df.to_string(index=False))
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved Figure: {fig_path}")


if __name__ == "__main__":
    main()
