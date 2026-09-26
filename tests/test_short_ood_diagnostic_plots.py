from scripts.plot_short_budget_ood_figures import plot_environment_diagnostics


def test_environment_diagnostics_writes_one_pdf_per_environment(tmp_path):
    summary = [
        {"env_key": "building", "phase": "id_pre", "update_budget": "0", "metric": "HV", "mean": "0.8", "std": "0.1", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "0", "metric": "HV", "mean": "0.4", "std": "0.1", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "8", "metric": "HV", "mean": "0.5", "std": "0.1", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "16", "metric": "HV", "mean": "0.6", "std": "0.1", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "32", "metric": "HV", "mean": "0.7", "std": "0.1", "n": "3"},
        {"env_key": "building", "phase": "id_return", "update_budget": "32", "metric": "HV", "mean": "0.9", "std": "0.1", "n": "3"},
        {"env_key": "building", "phase": "id_pre", "update_budget": "0", "metric": "EU", "mean": "8.0", "std": "1.0", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "0", "metric": "EU", "mean": "4.0", "std": "1.0", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "8", "metric": "EU", "mean": "5.0", "std": "1.0", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "16", "metric": "EU", "mean": "6.0", "std": "1.0", "n": "3"},
        {"env_key": "building", "phase": "ood", "update_budget": "32", "metric": "EU", "mean": "7.0", "std": "1.0", "n": "3"},
        {"env_key": "building", "phase": "id_return", "update_budget": "32", "metric": "EU", "mean": "9.0", "std": "1.0", "n": "3"},
    ]

    output = plot_environment_diagnostics(summary, "building", tmp_path)

    assert output == tmp_path / "ood_short_diagnostics_building.pdf"
    assert output.exists()
    assert output.stat().st_size > 0
