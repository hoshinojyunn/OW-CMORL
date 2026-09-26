#!/usr/bin/env python
"""Audit whether existing OOD checkpoints can support causal continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.short_ood_adapters import probe_checkpoint


METHODS = ("dynamic", "capql", "qpensieve", "pgmorl", "morlca", "lcpo")
ENVIRONMENTS = ("building", "evcharging", "cogen", "chlor_alkali")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="ood_v3_site_tariff_trace")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "short_ood_v3" / "capability_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for env_key in ENVIRONMENTS:
        for method in METHODS:
            checkpoint_dir = args.results_root / args.profile / method / env_key
            report = probe_checkpoint(method, checkpoint_dir)
            payload = {"env_key": env_key, **report.as_dict()}
            reports.append(payload)
            print(json.dumps(payload, sort_keys=True))
    output = {
        "profile": args.profile,
        "resume_policy": "No checkpoint is declared short-budget-adaptation-capable until its native optimizer/state restore path is implemented and a B=0 hash guard passes.",
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
