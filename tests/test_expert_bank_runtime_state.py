from types import SimpleNamespace

import numpy as np

from src.dynamic_morl.expert_bank import ContextExpertBank


def _sample(uid: str, embedding: list[float]):
    return SimpleNamespace(
        objs=np.asarray([1.0, 2.0]),
        metadata={
            "sample_uid": uid,
            "context_embedding": np.asarray(embedding, dtype=np.float64),
            "trace_metrics": {"trace_recovery_score": 0.5},
        },
    )


def test_expert_bank_runtime_state_restores_slots_and_rebuilds_index():
    bank = ContextExpertBank(
        max_slots=3,
        experts_per_slot=2,
        retrieval_backend="exact",
    )
    bank.add(_sample("a", [1.0, 0.0]))
    bank.add(_sample("b", [0.0, 1.0]))

    restored = ContextExpertBank.from_state_dict(bank.state_dict())
    assert restored._hnsw_dirty is True
    retrieved = restored.query(np.asarray([1.0, 0.0]), top_slots=1, top_per_slot=1)

    assert len(restored) == 2
    assert [sample.metadata["sample_uid"] for sample in retrieved] == ["a"]
