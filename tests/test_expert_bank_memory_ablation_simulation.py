import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pymoo")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_expert_bank_memory_ablation_simulation.py"
SPEC = spec_from_file_location("expert_bank_memory_ablation_simulation", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_all_memory_variants_return_same_query_fanout():
    profile = MODULE.PROFILES[0]
    rng = np.random.default_rng(9)
    warmup, events = MODULE._make_event_stream(rng, profile, 12, steps=24, warmup_rounds=2)
    query = events[0][1]
    memories = (
        MODULE.SemanticExpertBank(profile),
        MODULE.FlatKNNMemory(profile.capacity, "fifo"),
        MODULE.FlatKNNMemory(profile.capacity, "prioritized"),
    )
    expected = profile.top_slots * profile.per_slot
    for memory in memories:
        for snapshot in warmup:
            memory.add(snapshot)
        retrieved = memory.query(query, expected)
        assert len(retrieved) == expected
        assert len({snapshot.uid for snapshot in retrieved}) == expected


def test_generated_stream_contains_recurrent_rare_conditions():
    profile = MODULE.PROFILES[1]
    rng = np.random.default_rng(13)
    _warmup, events = MODULE._make_event_stream(rng, profile, 16, steps=96, warmup_rounds=1)
    rare = [mode for mode, _query, _snapshot in events if mode >= profile.easy_modes]
    assert len(rare) >= 2
