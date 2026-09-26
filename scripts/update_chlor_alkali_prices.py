from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dynamic_morl.chlor_alkali_price import (
    ANHUI_CHLOR_ALKALI_35KV_PROXY,
    chlor_alkali_dynamic_price,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite chlor-alkali CSV files with dynamic TOU prices.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("chlor-alkali"),
        help="Directory containing CA_all.csv / CA_train.csv / CA_test.csv",
    )
    return parser.parse_args()


def rewrite_price_column(csv_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    timestamps = pd.to_datetime(frame["datetime"], format="%y/%m/%d %H:%M:%S")
    prices = [chlor_alkali_dynamic_price(ts, proxy=ANHUI_CHLOR_ALKALI_35KV_PROXY) for ts in timestamps]

    if "current_price" in frame.columns:
        frame["current_price"] = prices
    else:
        insert_at = frame.columns.get_loc("temp_naoh") + 1
        frame.insert(insert_at, "current_price", prices)

    frame.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    for name in ("CA_all.csv", "CA_train.csv", "CA_test.csv"):
        rewrite_price_column(args.csv_dir / name)


if __name__ == "__main__":
    main()
