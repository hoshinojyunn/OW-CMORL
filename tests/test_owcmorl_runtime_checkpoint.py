from types import SimpleNamespace

import numpy as np

from src.dynamic_morl.expert_bank import ContextExpertBank
from src.dynamic_morl.online_context import OnlineContextMemory, OnlineTraceWindow
from src.dynamic_morl.runtime_checkpoint import (
    load_owcmorl_runtime_checkpoint,
    save_owcmorl_runtime_checkpoint,
)


def _sample():
    return SimpleNamespace(
        objs=np.asarray([1.0, 2.0]),
        metadata={
            "sample_uid": "checkpoint-sample",
            "context_embedding": np.asarray([1.0, 0.0]),
            "trace_metrics": {"trace_recovery_score": 0.5},
        },
    )


def test_runtime_checkpoint_restores_semantic_bank_and_online_memory(tmp_path):
    bank = ContextExpertBank(retrieval_backend="exact")
    sample = _sample()
    bank.add(sample)
    memory = OnlineContextMemory(max_size=3)
    window = OnlineTraceWindow(max_steps=4)
    window.append(np.asarray([[1.0, 2.0]], dtype=np.float32), ["x", "y"])
    path = tmp_path / "short_ood_runtime.pt"

    save_owcmorl_runtime_checkpoint(
        path,
        ep={"archive": [sample]},
        expert_bank=bank,
        regime_model=None,
        context_memory=memory,
        online_trace_window=window,
        prev_target_embedding=np.asarray([0.5, 0.5]),
        current_drift_score=0.2,
        current_context_state={"drift_score": 0.2},
        selected_history=[sample],
        selected_batch=[sample],
        final_samples=[sample],
        iteration=7,
        args_signature={"env_name": "toy", "obj_num": 2},
    )
    restored = load_owcmorl_runtime_checkpoint(path)

    assert restored["schema_version"] == 1
    assert restored["iteration"] == 7
    assert len(restored["expert_bank"]) == 1
    assert restored["online_trace_window"].context_feature_names == ["x", "y"]
    assert np.allclose(restored["prev_target_embedding"], [0.5, 0.5])
