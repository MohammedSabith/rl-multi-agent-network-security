"""Tests for RL training — single-agent sanity checks and self-play."""

from pathlib import Path

import numpy as np

from netsim.environment import GameEngine
from netsim.evaluation.baselines import (
    random_attacker_factory,
    random_defender_factory,
    run_matchup,
)
from netsim.training.self_play import (
    make_rl_policy,
    train_attacker,
    train_defender,
)

SCENARIO_PATH = Path(__file__).parent.parent / "configs" / "scenarios" / "small_enterprise.yaml"

# Use fewer timesteps for tests — just enough to verify training runs
TEST_TIMESTEPS = 10_000


class TestSingleAgentTraining:
    """Phase A: verify MaskablePPO can train against fixed opponents."""

    def test_attacker_trains_without_error(self):
        """MaskablePPO attacker training should complete without crashes."""
        model = train_attacker(
            SCENARIO_PATH, total_timesteps=TEST_TIMESTEPS, seed=0
        )
        assert model is not None

    def test_defender_trains_without_error(self):
        """MaskablePPO defender training should complete without crashes."""
        model = train_defender(
            SCENARIO_PATH, total_timesteps=TEST_TIMESTEPS, seed=0
        )
        assert model is not None

    def test_trained_attacker_beats_random(self):
        """RL attacker (trained vs random def) should beat random attacker baseline."""
        model = train_attacker(
            SCENARIO_PATH, total_timesteps=TEST_TIMESTEPS, seed=0
        )
        rl_policy = make_rl_policy(model)

        def rl_atk_factory(engine):
            return rl_policy

        result = run_matchup(
            SCENARIO_PATH, rl_atk_factory, random_defender_factory,
            n_episodes=200, seed=42,
        )
        # Random attacker wins ~7% vs random defender
        # RL attacker should do better than that
        assert result["attacker_win_rate"] > 0.07, (
            f"RL attacker not better than random: {result['attacker_win_rate']:.1%}"
        )

    def test_trained_defender_beats_random(self):
        """RL defender (trained vs random atk) should beat random defender baseline."""
        model = train_defender(
            SCENARIO_PATH, total_timesteps=TEST_TIMESTEPS, seed=0
        )
        rl_policy = make_rl_policy(model)

        def rl_def_factory(engine):
            return rl_policy

        result = run_matchup(
            SCENARIO_PATH, random_attacker_factory, rl_def_factory,
            n_episodes=200, seed=42,
        )
        # Random defender wins ~93% vs random attacker
        # RL defender should do at least as well
        assert result["defender_win_rate"] > 0.90, (
            f"RL defender not better than random: {result['defender_win_rate']:.1%}"
        )

    def test_attacker_model_produces_valid_actions(self):
        """Trained model should only produce valid actions via action masking."""
        model = train_attacker(
            SCENARIO_PATH, total_timesteps=TEST_TIMESTEPS, seed=0
        )
        engine = GameEngine(SCENARIO_PATH, seed=42)
        for ep in range(10):
            engine.reset(seed=ep)
            while not engine.done:
                engine.begin_turn()
                obs = engine.get_attacker_obs()
                mask = engine.get_attacker_action_mask()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                # Verify the action is valid
                assert mask[action], f"Model picked invalid action {action}"
                atk_action = engine.attacker_actions.decode(int(action))
                engine.execute_attacker_action(atk_action)
                if engine.done:
                    break
                # Defender plays random
                def_obs = engine.get_defender_obs()
                def_mask = engine.get_defender_action_mask()
                valid = np.where(def_mask)[0]
                def_action = engine.defender_actions.decode(int(engine.rng.choice(valid)))
                engine.execute_defender_action(def_action)
                engine.end_turn()
