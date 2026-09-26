import math

import numpy as np
import pytest
import torch

from src.short_ood_protocol import (
    BudgetLedger,
    ShortOODManifest,
    id_return_retention,
    relative_switch_loss,
    stable_state_hash,
)


def test_budget_ledger_rejects_nonmonotonic_budget_recording():
    ledger = BudgetLedger((0, 64, 128, 256))
    ledger.record(0)
    ledger.record(64)
    assert ledger.remaining_to(128) == 64
    with pytest.raises(ValueError, match="monotonic"):
        ledger.record(64)


def test_relative_switch_loss_penalizes_only_a_drop():
    assert relative_switch_loss(10.0, 7.5) == pytest.approx(0.25)
    assert relative_switch_loss(-10.0, -12.0) == pytest.approx(0.2)
    assert relative_switch_loss(-10.0, -8.0) == pytest.approx(0.0)
    assert relative_switch_loss(10.0, 12.0) == pytest.approx(0.0)


def test_id_return_retention_returns_nan_for_near_zero_reference():
    assert id_return_retention(10.0, 9.0) == pytest.approx(0.9)
    assert math.isnan(id_return_retention(0.0, 1.0))


def test_manifest_requires_zero_budget_and_unique_trajectory_seeds():
    with pytest.raises(ValueError, match="include 0"):
        ShortOODManifest(budgets=(64,), trajectory_seeds=(101, 202, 303))
    with pytest.raises(ValueError, match="unique"):
        ShortOODManifest(budgets=(0, 64), trajectory_seeds=(101, 101, 303))


def test_stable_state_hash_captures_numpy_parameter_changes():
    payload = {"weights": np.asarray([1.0, 2.0], dtype=np.float32)}
    before = stable_state_hash(payload)
    payload["weights"][1] = 3.0
    assert stable_state_hash(payload) != before


def test_stable_state_hash_captures_unprinted_tensor_changes():
    payload = {"weights": torch.zeros(2048, dtype=torch.float32)}
    before = stable_state_hash(payload)
    payload["weights"][1024] = 1.0
    assert stable_state_hash(payload) != before
