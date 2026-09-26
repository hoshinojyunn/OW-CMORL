from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TOUPriceProxy:
    name: str
    voltage_level: str
    summer_peak: float
    summer_flat: float
    summer_valley: float
    regular_peak: float
    regular_flat: float
    regular_valley: float



ANHUI_CHLOR_ALKALI_35KV_PROXY = TOUPriceProxy(
    name="anhui_ion_membrane_chlor_alkali_35kv_proxy",
    voltage_level="35kv",
    summer_peak=1.0046,
    summer_flat=0.6300,
    summer_valley=0.3937,
    regular_peak=0.9470,
    regular_flat=0.6300,
    regular_valley=0.3937,
)

CHLOR_ALKALI_REGIMES = (
    "summer_peak",
    "regular_peak",
    "flat",
    "valley",
)


def chlor_alkali_price_period(timestamp: pd.Timestamp) -> str:
    hour = timestamp.hour + timestamp.minute / 60.0
    if 9.0 <= hour < 12.0 or 17.0 <= hour < 22.0:
        return "peak"
    if 23.0 <= hour or hour < 8.0:
        return "valley"
    return "flat"


def chlor_alkali_dynamic_price(
    timestamp: pd.Timestamp,
    proxy: TOUPriceProxy = ANHUI_CHLOR_ALKALI_35KV_PROXY,
) -> float:
    period = chlor_alkali_price_period(timestamp)
    summer = timestamp.month in {7, 8, 9}
    if summer:
        if period == "peak":
            return proxy.summer_peak
        if period == "valley":
            return proxy.summer_valley
        return proxy.summer_flat
    if period == "peak":
        return proxy.regular_peak
    if period == "valley":
        return proxy.regular_valley
    return proxy.regular_flat


def chlor_alkali_regime_name(timestamp: pd.Timestamp) -> str:
    period = chlor_alkali_price_period(timestamp)
    if period == "peak":
        return "summer_peak" if timestamp.month in {7, 8, 9} else "regular_peak"
    return period
