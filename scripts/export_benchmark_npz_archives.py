from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from benchmark_npz_utils import as_2d, pareto_front, safe_name, save_npz_archive


RAW_DYNAMIC_RUN_ORDER = [
    "owcmorl_online_main_building_dynamic_seed0",
    "owcmorl_online_main_building_no_dyn_ablation_seed0",
    "owcmorl_online_main_building_random_seed0",
    "owcmorl_online_main_building_static_seed0",
    "owcmorl_online_main_cogen_dynamic_seed0",
    "owcmorl_online_main_cogen_no_dyn_ablation_seed0",
    "owcmorl_online_main_cogen_random_seed0",
    "owcmorl_online_main_cogen_static_seed0",
    "owcmorl_online_main_evcharging_dynamic_seed0",
    "owcmorl_online_main_evcharging_no_dyn_ablation_seed0",
    "owcmorl_online_main_evcharging_random_seed0",
    "owcmorl_online_main_evcharging_static_seed0",
    "chlor_online_strong_dynamic_seed0",
    "chlor_online_t1024_chlor_alkali_dynamic_seed0",
    "chlor_online_t1024_chlor_alkali_no_dyn_ablation_seed0",
    "chlor_online_t1024_chlor_alkali_random_seed0",
    "chlor_online_t1024_chlor_alkali_static_seed0",
]

CANONICAL_DYNAMIC_RUN_ORDER = RAW_DYNAMIC_RUN_ORDER
CANONICAL_MORL_SUITE_ORDER = [
    "morl_shared20_main_v1",
    "chlor_bench_t1024",
    "morl_online_longrun_v5c",
    "morl_online_longrun_v3",
    "morl_online_main",
    "morl_online_pgmorl",
    "morl_online_qpensieve",
    "chlor_bench_long",
]

MORL_METHOD_ALIASES = {
    "qpensieve": "q_pensieve",
    "q_pensieve": "q_pensieve",
    "capql": "capql",
    "pgmorl": "pgmorl",
}
CANONICAL_DYNAMIC_RUN_INDEX = {
    name: idx for idx, name in enumerate(CANONICAL_DYNAMIC_RUN_ORDER)
}
CANONICAL_MORL_SUITE_INDEX = {
    name: idx for idx, name in enumerate(CANONICAL_MORL_SUITE_ORDER)
}
EXPECTED_SHARED_PROTOCOL_REGIMES = 20


def _valid_regime_rows(rows: list[dict[str, object]] | None) -> bool:
    if not isinstance(rows, list):
        return False
    unique_ids = {
        int(row.get("regime_id", -1))
        for row in rows
        if int(row.get("regime_id", -1)) >= 0
    }
    return len(rows) >= EXPECTED_SHARED_PROTOCOL_REGIMES and len(unique_ids) >= EXPECTED_SHARED_PROTOCOL_REGIMES


def _valid_scalarized_result(data: dict[str, object]) -> bool:
    return _valid_regime_rows(data.get("regime_returns", []))


def _sorted_dynamic_run_dirs(results_root: Path) -> list[Path]:
    configured_index = {name: idx for idx, name in enumerate(RAW_DYNAMIC_RUN_ORDER)}
    candidates = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and parse_dynamic_run_name(path.name) is not None
    ]
    return sorted(
        candidates,
        key=lambda path: (
            0 if path.name in configured_index else 1,
            configured_index.get(path.name, 10**9),
            -path.stat().st_mtime,
            path.name,
        ),
    )


def _is_temporary_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    return lowered.startswith(("tmp", "_")) or "smoke" in lowered


def _dynamic_run_priority(run_name: str) -> tuple[int, int]:
    if run_name in CANONICAL_DYNAMIC_RUN_INDEX:
        return (0, CANONICAL_DYNAMIC_RUN_INDEX[run_name])
    if run_name.startswith("sharedprot_main_"):
        return (1, 0)
    if run_name.startswith("owcmorl_online_main_") or run_name.startswith("chlor_online_t1024_"):
        return (1, 1)
    if _is_temporary_name(run_name):
        return (3, 10**6)
    return (2, 10**6)


def _morl_suite_priority(suite_name: str) -> tuple[int, int]:
    if suite_name in CANONICAL_MORL_SUITE_INDEX:
        return (0, CANONICAL_MORL_SUITE_INDEX[suite_name])
    if _is_temporary_name(suite_name):
        return (2, 10**6)
    return (1, 10**6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark Pareto archives as .npz files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--raw-out-root",
        type=Path,
        default=PROJECT_ROOT / "archives_npz",
        help="Per-run raw archive root. Retains one archive per source run and regime.",
    )
    parser.add_argument(
        "--raw-manifest-path",
        type=Path,
        default=PROJECT_ROOT / "archives_npz_manifest.csv",
    )
    parser.add_argument(
        "--canonical-out-root",
        type=Path,
        default=PROJECT_ROOT / "archives_npz_canonical",
        help="Method-level archive root. Each file is one (env, method, regime) archive.",
    )
    parser.add_argument(
        "--canonical-manifest-path",
        type=Path,
        default=PROJECT_ROOT / "archives_npz_canonical_manifest.csv",
    )
    parser.add_argument(
        "--export-kind",
        choices=["raw", "canonical", "both"],
        default="both",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_method_name(method: str) -> str:
    name = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    if name == "no_dyn_ablation":
        return "no_dyn"
    return MORL_METHOD_ALIASES.get(name, name)


def parse_dynamic_run_name(run_name: str) -> tuple[str, str] | None:
    match = re.match(
        r".+_(building|evcharging|cogen|chlor_alkali)_(dynamic|static|random|no_dyn_ablation)_seed\d+$",
        run_name,
    )
    if match is None:
        return None
    env_key, config = match.groups()
    return env_key, normalize_method_name(config)


def latest_regime_front_path(run_dir: Path) -> Path | None:
    regime_dir = run_dir / "regime_fronts"
    if not regime_dir.exists():
        return None
    latest_path = None
    latest_key = (-1, "")
    for path in regime_dir.glob("*.json"):
        data = json.loads(path.read_text())
        key = (int(data.get("iteration", -1)), str(data.get("stage", "")))
        if key > latest_key:
            latest_key = key
            latest_path = path
    return latest_path


def row_regime_metadata(row: dict[str, object]) -> dict[str, object]:
    meta = row.get("regime_meta", {})
    if not isinstance(meta, dict):
        meta = {}
    regime_id = row.get("regime_id", -1)
    try:
        regime_id = int(regime_id)
    except Exception:
        regime_id = -1
    return {
        "regime_id": regime_id,
        "regime_meta": meta,
        "dynamic_factors": meta,
    }


def regime_archive_name(row: dict[str, object]) -> str:
    regime_id = int(row.get("regime_id", -1))
    regime = safe_name(str(row.get("regime", "regime")))
    if regime_id >= 0:
        return f"regime_{regime_id:03d}_{regime}"
    return regime


def regime_group_key(env_key: str, method: str, row: dict[str, object]) -> tuple[str, str, str]:
    return env_key, method, regime_archive_name(row)


def export_scalarized_results_raw(results_root: Path, out_root: Path, overwrite: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for result_path in sorted((results_root / "scalarized_baselines").glob("*/*/seed*_w*/result.json")):
        data = json.loads(result_path.read_text())
        if not _valid_scalarized_result(data):
            continue
        env_key = result_path.parts[-4]
        method = normalize_method_name(result_path.parts[-3])
        run_tag = result_path.parent.name
        run_dir = out_root / "scalarized_baselines" / env_key / method / run_tag
        regime_points: dict[str, list[list[float]]] = {}
        regime_rows = data.get("regime_returns", [])
        grouped_rows: dict[str, list[dict[str, object]]] = {}
        for row in regime_rows:
            grouped_rows.setdefault(regime_archive_name(row), []).append(row)
        for regime_key, rows in grouped_rows.items():
            points = [list(row.get("objs", [])) for row in rows]
            if not points:
                continue
            first_row = rows[0]
            solutions = as_2d(points)
            front = pareto_front(solutions)
            out_path = run_dir / f"{regime_key}.npz"
            if not (out_path.exists() and not overwrite):
                save_npz_archive(
                    out_path,
                    solutions=solutions,
                    pareto_front_points=front,
                    meta={
                        "source_type": "scalarized_result",
                        "source_path": str(result_path),
                        "env_key": env_key,
                        "method": method,
                        "run_tag": run_tag,
                        "regime": str(first_row.get("regime", "regime")),
                        "regime_archive_name": regime_key,
                        **row_regime_metadata(first_row),
                        "num_solutions": int(len(solutions)),
                        "num_front": int(len(front)),
                    },
                )
            records.append(
                {
                    "archive_path": str(out_path),
                    "source_path": str(result_path),
                    "source_type": "scalarized_result",
                    "env_key": env_key,
                    "method": method,
                    "run_tag": run_tag,
                    "regime": str(first_row.get("regime", "regime")),
                    "regime_id": int(first_row.get("regime_id", -1)),
                    "regime_meta_json": json.dumps(first_row.get("regime_meta", {}), ensure_ascii=False),
                    "num_solutions": int(len(solutions)),
                    "num_front": int(len(front)),
                }
            )
    return records


def export_morl_summaries_raw(results_root: Path, out_root: Path, overwrite: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for summary_path in sorted((results_root / "morl_dynamic_baselines").glob("*/**/seed*/summary.json")):
        data = json.loads(summary_path.read_text())
        regime_fronts = data.get("regime_fronts", [])
        if not _valid_regime_rows(regime_fronts):
            continue
        env_key = str(data.get("env_key", "unknown"))
        method = normalize_method_name(data.get("method", "method"))
        run_tag = summary_path.parent.name
        suite_name = summary_path.parents[3].name if len(summary_path.parents) > 3 else "morl_dynamic_baselines"
        run_dir = out_root / "morl_dynamic_baselines" / suite_name / method / env_key / run_tag
        for row in regime_fronts:
            regime = str(row.get("regime", "regime"))
            points = row.get("points", [])
            if not points:
                continue
            solutions = as_2d(points)
            front = pareto_front(solutions)
            out_path = run_dir / f"{safe_name(regime)}.npz"
            if not (out_path.exists() and not overwrite):
                save_npz_archive(
                    out_path,
                    solutions=solutions,
                    pareto_front_points=front,
                    meta={
                        "source_type": "morl_summary",
                        "source_path": str(summary_path),
                        "env_key": env_key,
                        "method": method,
                        "suite_name": suite_name,
                        "run_tag": run_tag,
                        "regime": regime,
                        **row_regime_metadata(row),
                        "num_solutions": int(len(solutions)),
                        "num_front": int(len(front)),
                    },
                )
            records.append(
                {
                    "archive_path": str(out_path),
                    "source_path": str(summary_path),
                    "source_type": "morl_summary",
                    "env_key": env_key,
                    "method": method,
                    "suite_name": suite_name,
                    "run_tag": run_tag,
                    "regime": regime,
                    "regime_id": int(row.get("regime_id", -1)),
                    "regime_meta_json": json.dumps(row.get("regime_meta", {}), ensure_ascii=False),
                    "num_solutions": int(len(solutions)),
                    "num_front": int(len(front)),
                }
            )
    return records


def export_dynamic_runs_raw(results_root: Path, out_root: Path, overwrite: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run_dir in _sorted_dynamic_run_dirs(results_root):
        run_name = run_dir.name
        latest_path = latest_regime_front_path(run_dir)
        if latest_path is None:
            continue
        meta = parse_dynamic_run_name(run_name)
        if meta is None:
            continue
        env_key, method = meta
        payload = json.loads(latest_path.read_text())
        if not _valid_regime_rows(payload.get("regimes", [])):
            continue
        out_dir = out_root / run_name
        for row in payload.get("regimes", []):
            regime = str(row.get("regime", "regime"))
            points = row.get("front_points", []) or []
            if not points:
                continue
            solutions = as_2d(points)
            front = pareto_front(solutions)
            out_path = out_dir / f"{safe_name(regime)}.npz"
            if not (out_path.exists() and not overwrite):
                save_npz_archive(
                    out_path,
                    solutions=solutions,
                    pareto_front_points=front,
                    meta={
                        "source_type": "dynamic_regime_fronts",
                        "source_path": str(latest_path),
                        "env_key": env_key,
                        "method": method,
                        "run_name": run_name,
                        "stage": payload.get("stage", ""),
                        "iteration": int(payload.get("iteration", -1)),
                        "regime": regime,
                        **row_regime_metadata(row),
                        "num_solutions": int(len(solutions)),
                        "num_front": int(len(front)),
                    },
                )
            records.append(
                {
                    "archive_path": str(out_path),
                    "source_path": str(latest_path),
                    "source_type": "dynamic_regime_fronts",
                    "env_key": env_key,
                    "method": method,
                    "run_name": run_name,
                    "stage": payload.get("stage", ""),
                    "iteration": int(payload.get("iteration", -1)),
                    "regime": regime,
                    "regime_id": int(row.get("regime_id", -1)),
                    "regime_meta_json": json.dumps(row.get("regime_meta", {}), ensure_ascii=False),
                    "num_solutions": int(len(solutions)),
                    "num_front": int(len(front)),
                }
            )
    return records


def export_raw_archives(
    results_root: Path,
    out_root: Path,
    manifest_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    records.extend(export_scalarized_results_raw(results_root, out_root, overwrite))
    records.extend(export_morl_summaries_raw(results_root, out_root, overwrite))
    records.extend(export_dynamic_runs_raw(results_root, out_root, overwrite))
    if not records:
        raise FileNotFoundError("No raw benchmark archives were exported.")
    manifest = pd.DataFrame(records).sort_values(["env_key", "method", "regime", "archive_path"]).reset_index(drop=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def collect_scalarized_groups(results_root: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for result_path in sorted((results_root / "scalarized_baselines").glob("*/*/seed*_w*/result.json")):
        data = json.loads(result_path.read_text())
        if not _valid_scalarized_result(data):
            continue
        env_key = result_path.parts[-4]
        method = normalize_method_name(result_path.parts[-3])
        for row in data.get("regime_returns", []):
            regime = str(row.get("regime", "regime"))
            regime_name = regime_archive_name(row)
            key = regime_group_key(env_key, method, row)
            bucket = grouped.setdefault(
                key,
                {
                    "source_type": "scalarized_result",
                    "env_key": env_key,
                    "method": method,
                    "regime": regime,
                    "regime_archive_name": regime_name,
                    "points": [],
                    "source_paths": [],
                    "run_tags": [],
                    "regime_id": int(row.get("regime_id", -1)),
                    "regime_meta": row.get("regime_meta", {}) if isinstance(row.get("regime_meta", {}), dict) else {},
                },
            )
            bucket["points"].append(list(row.get("objs", [])))
            bucket["source_paths"].append(str(result_path))
            bucket["run_tags"].append(result_path.parent.name)
    return grouped


def collect_morl_groups(results_root: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    # Prefer formal benchmark suites over temporary/smoke runs, then take the newest file.
    selected_runs: dict[tuple[str, str, int], tuple[tuple[int, int], float, str, Path, dict[str, object]]] = {}
    morl_root = results_root / "morl_dynamic_baselines"
    if not morl_root.exists():
        return {}
    for summary_path in sorted(morl_root.glob("*/**/seed*/summary.json")):
            data = json.loads(summary_path.read_text())
            env_key = str(data.get("env_key", "unknown"))
            method = normalize_method_name(data.get("method", "method"))
            seed = int(data.get("seed", 0))
            regime_fronts = data.get("regime_fronts", [])
            has_points = any(bool(row.get("points")) for row in regime_fronts)
            if not has_points or not _valid_regime_rows(regime_fronts):
                continue
            select_key = (env_key, method, seed)
            suite_name = summary_path.parents[3].name if len(summary_path.parents) > 3 else "morl_dynamic_baselines"
            mtime = float(summary_path.stat().st_mtime)
            priority = _morl_suite_priority(suite_name)
            current = selected_runs.get(select_key)
            if current is None or priority < current[0] or (priority == current[0] and mtime > current[1]):
                selected_runs[select_key] = (priority, mtime, suite_name, summary_path, data)

    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for (env_key, method, seed), (_priority, _mtime, suite_name, summary_path, data) in sorted(selected_runs.items()):
        for row in data.get("regime_fronts", []):
            regime = str(row.get("regime", "regime"))
            points = row.get("points", [])
            if not points:
                continue
            regime_name = regime_archive_name(row)
            key = regime_group_key(env_key, method, row)
            bucket = grouped.setdefault(
                key,
                {
                    "source_type": "morl_summary",
                    "env_key": env_key,
                    "method": method,
                    "regime": regime,
                    "regime_archive_name": regime_name,
                    "points": [],
                    "source_paths": [],
                    "suite_names": [],
                    "seeds": [],
                    "regime_id": int(row.get("regime_id", -1)),
                    "regime_meta": row.get("regime_meta", {}) if isinstance(row.get("regime_meta", {}), dict) else {},
                },
            )
            bucket["points"].extend(points)
            bucket["source_paths"].append(str(summary_path))
            bucket["suite_names"].append(suite_name)
            bucket["seeds"].append(seed)
    return grouped


def collect_dynamic_groups(results_root: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    # Prefer canonical run families first so ad-hoc debug runs do not leak into official archives.
    selected_runs: dict[tuple[str, str, int], tuple[tuple[int, int], float, str, Path, dict[str, object]]] = {}
    for run_dir in _sorted_dynamic_run_dirs(results_root):
        run_name = run_dir.name
        meta = parse_dynamic_run_name(run_name)
        if meta is None:
            continue
        env_key, method = meta
        seed_match = re.search(r"_seed(\d+)$", run_name)
        seed = int(seed_match.group(1)) if seed_match else 0
        latest_path = latest_regime_front_path(run_dir)
        if latest_path is None:
            continue
        payload = json.loads(latest_path.read_text())
        if not _valid_regime_rows(payload.get("regimes", [])):
            continue
        select_key = (env_key, method, seed)
        mtime = float(latest_path.stat().st_mtime)
        priority = _dynamic_run_priority(run_name)
        current = selected_runs.get(select_key)
        if current is None or priority < current[0] or (priority == current[0] and mtime > current[1]):
            selected_runs[select_key] = (priority, mtime, run_name, latest_path, payload)

    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for (env_key, method, seed), (_priority, _mtime, run_name, latest_path, payload) in sorted(selected_runs.items()):
        for row in payload.get("regimes", []):
            regime = str(row.get("regime", "regime"))
            points = row.get("front_points", []) or []
            if not points:
                continue
            regime_name = regime_archive_name(row)
            key = regime_group_key(env_key, method, row)
            bucket = grouped.setdefault(
                key,
                {
                    "source_type": "dynamic_regime_fronts",
                    "env_key": env_key,
                    "method": method,
                    "regime": regime,
                    "regime_archive_name": regime_name,
                    "points": [],
                    "source_paths": [],
                    "run_names": [],
                    "seeds": [],
                    "stages": [],
                    "iterations": [],
                    "regime_id": int(row.get("regime_id", -1)),
                    "regime_meta": row.get("regime_meta", {}) if isinstance(row.get("regime_meta", {}), dict) else {},
                },
            )
            bucket["points"].extend(points)
            bucket["source_paths"].append(str(latest_path))
            bucket["run_names"].append(run_name)
            bucket["seeds"].append(seed)
            bucket["stages"].append(str(payload.get("stage", "")))
            bucket["iterations"].append(int(payload.get("iteration", -1)))
    return grouped


def write_canonical_groups(
    groups: dict[tuple[str, str, str], dict[str, object]],
    out_root: Path,
    overwrite: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (env_key, method, regime_name), payload in sorted(groups.items()):
        points = payload.get("points", [])
        if not points:
            continue
        solutions = as_2d(points)
        front = pareto_front(solutions)
        out_path = out_root / env_key / method / f"{safe_name(regime_name)}.npz"
        meta = {
            key: value
            for key, value in payload.items()
            if key != "points"
        }
        if isinstance(meta.get("regime_meta"), dict) and "dynamic_factors" not in meta:
            meta["dynamic_factors"] = meta["regime_meta"]
        meta.update(
            {
                "num_solutions": int(len(solutions)),
                "num_front": int(len(front)),
            }
        )
        if not (out_path.exists() and not overwrite):
            save_npz_archive(
                out_path,
                solutions=solutions,
                pareto_front_points=front,
                meta=meta,
            )
        records.append(
            {
                "archive_path": str(out_path),
                "source_type": payload["source_type"],
                "env_key": env_key,
                "method": method,
                "regime": payload.get("regime", regime_name),
                "regime_archive_name": regime_name,
                "num_solutions": int(len(solutions)),
                "num_front": int(len(front)),
                "num_source_paths": int(len(meta.get("source_paths", []))),
                "source_paths_json": json.dumps(meta.get("source_paths", []), ensure_ascii=False),
                "meta_json": json.dumps(meta, ensure_ascii=False),
            }
        )
    return records


def export_canonical_archives(
    results_root: Path,
    out_root: Path,
    manifest_path: Path,
    overwrite: bool,
) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for collector in (collect_scalarized_groups, collect_morl_groups, collect_dynamic_groups):
        grouped.update(collector(results_root))
    records = write_canonical_groups(grouped, out_root, overwrite)
    if not records:
        raise FileNotFoundError("No canonical benchmark archives were exported.")
    manifest = pd.DataFrame(records).sort_values(["env_key", "method", "regime"]).reset_index(drop=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    args = parse_args()

    if args.export_kind in {"raw", "both"}:
        raw_manifest = export_raw_archives(
            args.results_root,
            args.raw_out_root,
            args.raw_manifest_path,
            args.overwrite,
        )
        print(f"exported {len(raw_manifest)} raw regime archives")
        print(f"raw manifest: {args.raw_manifest_path}")

    if args.export_kind in {"canonical", "both"}:
        canonical_manifest = export_canonical_archives(
            args.results_root,
            args.canonical_out_root,
            args.canonical_manifest_path,
            args.overwrite,
        )
        print(f"exported {len(canonical_manifest)} canonical regime archives")
        print(f"canonical manifest: {args.canonical_manifest_path}")


if __name__ == "__main__":
    main()
