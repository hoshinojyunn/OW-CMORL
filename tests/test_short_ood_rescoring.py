import numpy as np
import pytest

pytest.importorskip("pymoo")

from src.short_ood_rescoring import (
    derive_trace_hv_envelope,
    rescore_event_rows,
    trace_normalized_hv,
)


def test_trace_normalized_hv_uses_one_fixed_envelope_for_negative_objectives():
    rows = [
        {
            "front_points": [[4.0, -10.0, -0.0002], [8.0, -4.0, 0.0]],
        },
        {
            "front_points": [[6.0, -8.0, -0.0001], [10.0, -3.0, 0.0]],
        },
    ]

    envelope = derive_trace_hv_envelope(rows)
    first_hv = trace_normalized_hv(np.asarray(rows[0]["front_points"]), envelope)
    second_hv = trace_normalized_hv(np.asarray(rows[1]["front_points"]), envelope)

    assert np.allclose(envelope.lower, [4.0, -10.0, -0.0002])
    assert np.allclose(envelope.upper, [10.0, -3.0, 0.0])
    assert first_hv > 0.0
    assert second_hv > 0.0
    assert first_hv != second_hv


def test_rescore_event_rows_preserves_raw_hv_and_scores_each_environment_separately():
    rows = [
        {"env_key": "ev", "HV": "0.0", "front_points": "[[4.0, -10.0], [8.0, -4.0]]"},
        {"env_key": "ev", "HV": "0.0", "front_points": "[[6.0, -8.0], [10.0, -3.0]]"},
        {"env_key": "other", "HV": "0.0", "front_points": "[[1.0, 1.0], [2.0, 2.0]]"},
        {"env_key": "other", "HV": "0.0", "front_points": "[[1.5, 1.2], [2.2, 2.5]]"},
    ]

    rescored, envelopes = rescore_event_rows(rows)

    assert set(envelopes) == {"ev", "other"}
    assert all(float(row["raw_HV"]) == 0.0 for row in rescored)
    assert all(float(row["HV"]) > 0.0 for row in rescored)
