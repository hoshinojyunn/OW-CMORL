import pytest
import numpy as np
import torch

from src.short_ood_adapters import (
    AdaptationAdapter,
    LCPOAdapter,
    MORLCAAdapter,
    assert_unchanged,
    probe_checkpoint,
)
from src.short_ood_protocol import ShortOODManifest
from scripts.run_short_budget_ood_adaptation import run_single_trajectory
from src.lcpo_baseline import LCPOTrainer
from src.morl_ca_baseline import MORLCABaseline


class ToyAdapter(AdaptationAdapter):
    method = "toy"

    def __init__(self):
        self.value = 0

    def state_payload(self):
        return {"value": self.value}

    def action(self, observation, preference=None):
        del observation, preference
        return 0

    def adapt(self, env, transitions):
        del env
        self.value += int(transitions)
        return {"updates": 1.0}


class _ToyActionSpace:
    def sample(self):
        return np.asarray([0.0], dtype=np.float32)


class _ToyVectorEnv:
    action_space = _ToyActionSpace()

    def reset(self):
        return np.zeros(3, dtype=np.float32)

    def step(self, action):
        assert np.asarray(action).shape == (1,)
        return np.ones(3, dtype=np.float32), 0.0, False, {"obj_raw": np.asarray([1.0, 2.0], dtype=np.float32)}


def test_zero_shot_guard_detects_mutation():
    adapter = ToyAdapter()
    before = adapter.state_hash()
    assert_unchanged(before, adapter.state_hash())
    adapter.value = 1
    with pytest.raises(RuntimeError, match="mutated"):
        assert_unchanged(before, adapter.state_hash())


def test_capability_probe_requires_capql_policy_and_critic(tmp_path):
    report = probe_checkpoint("capql", tmp_path)
    assert report.supported is False
    assert report.missing == ("model/policy.pt", "model/critic.pt")


def test_runner_emits_id_ood_budget_and_return_rows():
    adapter = ToyAdapter()
    manifest = ShortOODManifest(budgets=(0, 2), trajectory_seeds=(101,))

    def evaluate(adapter, _env, phase, budget):
        return {"HV": 10.0 + adapter.value, "EU": float(budget), "phase_value": phase}

    rows = run_single_trajectory(
        adapter=adapter,
        manifest=manifest,
        trajectory_seed=101,
        make_id_env=lambda: object(),
        make_ood_env=lambda: object(),
        evaluate=evaluate,
    )

    assert [(row["phase"], row["budget"]) for row in rows] == [
        ("id_pre", 0),
        ("ood", 0),
        ("ood", 2),
        ("id_return", 2),
    ]
    assert rows[1]["state_hash"] == rows[0]["state_hash"]
    assert rows[2]["state_hash"] != rows[1]["state_hash"]
    assert rows[3]["state_hash"] == rows[2]["state_hash"]


def test_runner_records_update_budget_and_adapter_reported_transitions():
    class UpdateAdapter(ToyAdapter):
        def adapt(self, env, updates):
            del env
            self.value += int(updates)
            return {"updates": float(updates), "transitions": float(8 * updates)}

    rows = run_single_trajectory(
        adapter=UpdateAdapter(),
        manifest=ShortOODManifest(budgets=(0, 2), trajectory_seeds=(101,)),
        trajectory_seed=101,
        make_id_env=lambda: object(),
        make_ood_env=lambda: object(),
        evaluate=lambda *_args: {"HV": 1.0, "EU": 1.0},
    )

    assert rows[2]["budget"] == 2
    assert rows[2]["update_count"] == 2.0
    assert rows[2]["transition_count"] == 16


def test_lcpo_state_round_trip_preserves_optimizers_and_context_reservoir():
    trainer = LCPOTrainer(
        obs_dim=3,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        context_dim=3,
        action_bins=3,
        hidden_sizes=(4,),
        ood_capacity=8,
        ood_recent_window=4,
        seed=7,
    )
    trainer.ood_reservoir.add(
        np.ones((2, 3), dtype=np.float32),
        2.0 * np.ones((2, 3), dtype=np.float32),
    )
    trainer.reward_var = 3.0
    trainer.lcpo_updates = 2
    payload = trainer.state_dict()

    restored = LCPOTrainer(
        obs_dim=3,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        context_dim=3,
        action_bins=3,
        hidden_sizes=(4,),
        ood_capacity=8,
        ood_recent_window=4,
        seed=99,
    )
    restored.load_state_dict(payload)

    assert restored.reward_var == pytest.approx(3.0)
    assert restored.lcpo_updates == 2
    assert restored.ood_reservoir.size == 2
    assert np.allclose(restored.ood_reservoir.contexts[:2], 2.0)
    for expected, actual in zip(trainer.policy.parameters(), restored.policy.parameters()):
        assert torch.equal(expected, actual)


def test_morlca_state_round_trip_preserves_replay_and_action():
    source = MORLCABaseline(
        obs_dim=3,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        obj_num=2,
        hidden_dim=4,
        batch_size=2,
        replay_size=8,
        start_steps=0,
    )
    source.replay.add(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        False,
    )
    payload = source.state_dict()
    restored = MORLCABaseline(
        obs_dim=3,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        obj_num=2,
        hidden_dim=4,
        batch_size=2,
        replay_size=8,
        start_steps=0,
    )
    restored.load_state_dict(payload)

    observation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.allclose(
        source.act(observation, deterministic=True),
        restored.act(observation, deterministic=True),
    )
    assert restored.replay.size == 1
    assert np.allclose(restored.replay.obs[0], source.replay.obs[0])


def test_native_adapters_consume_the_declared_transition_budget():
    env = _ToyVectorEnv()
    morlca = MORLCABaseline(
        obs_dim=3,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        obj_num=2,
        hidden_dim=4,
        batch_size=2,
        replay_size=8,
        start_steps=5,
    )
    morlca_adapter = MORLCAAdapter(morlca)
    before = morlca_adapter.state_hash()
    stats = morlca_adapter.adapt(env, 2)
    assert stats["transitions"] == 2.0
    assert stats["updates"] == 1.0
    assert morlca.start_steps == 0
    assert morlca_adapter.state_hash() != before

    trainers = [
        LCPOTrainer(
            obs_dim=3,
            action_low=np.asarray([-1.0], dtype=np.float32),
            action_high=np.asarray([1.0], dtype=np.float32),
            context_dim=3,
            action_bins=3,
            hidden_sizes=(4,),
            ood_capacity=8,
            ood_recent_window=4,
            seed=index,
        )
        for index in range(2)
    ]
    lcpo_adapter = LCPOAdapter(trainers, preferences=[(1.0, 0.0), (0.0, 1.0)])
    stats = lcpo_adapter.adapt(env, 5)
    assert stats["transitions"] == 5.0
    assert stats["updates"] == 2.0
