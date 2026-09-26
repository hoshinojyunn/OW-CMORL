import csv

from scripts.generate_short_budget_ood_report import build_report


def _write_events(path):
    path.parent.mkdir(parents=True)
    rows = [
        {"env_key": "building", "method": "dynamic", "trajectory_seed": 101, "phase": "id_pre", "budget": 0, "state_hash": "a", "transition_count": 0, "update_count": 0, "HV": 10.0, "EU": 5.0},
        {"env_key": "building", "method": "dynamic", "trajectory_seed": 101, "phase": "ood", "budget": 0, "state_hash": "a", "transition_count": 0, "update_count": 0, "HV": 8.0, "EU": 4.0},
        {"env_key": "building", "method": "dynamic", "trajectory_seed": 101, "phase": "ood", "budget": 8, "state_hash": "b", "transition_count": 64, "update_count": 8, "HV": 9.0, "EU": 4.5, "hnsw_backend": "hnsw"},
        {"env_key": "building", "method": "dynamic", "trajectory_seed": 101, "phase": "id_return", "budget": 8, "state_hash": "b", "transition_count": 64, "update_count": 8, "HV": 9.5, "EU": 4.8},
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def test_report_derives_update_budget_diagnostics_and_validates_hashes(tmp_path):
    results_root = tmp_path / "results"
    _write_events(results_root / "profile" / "dynamic" / "building" / "seed_101" / "events.csv")
    output = build_report(results_root, "profile", tmp_path / "analysis")

    assert output["validation"][0]["valid"] is True
    assert output["diagnostics"][0]["zero_shot_loss_HV"] == 0.2
    assert output["diagnostics"][0]["id_return_retention_HV"] == 0.95
    assert (tmp_path / "analysis" / "short_ood_events_summary.csv").exists()
