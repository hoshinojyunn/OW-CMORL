from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.build_dynamic_subset_shared_result import (
    _load_run_args,
    _materialize_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a known chlor union subset record.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--record-json", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    record = json.loads(args.record_json.read_text())
    regime_payload = json.loads((source_run_dir / "regime_fronts" / "shared_final.json").read_text())
    summary_payload = json.loads((source_run_dir / "final" / "shared_eval_summary.json").read_text())
    run_args = _load_run_args(source_run_dir)
    _materialize_subset(source_run_dir, args.output_run_dir.resolve(), run_args, regime_payload, summary_payload, record)
    print(args.output_run_dir.resolve())


if __name__ == "__main__":
    main()
