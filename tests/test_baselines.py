"""Tests for scripted baselines and environment verification."""

from pathlib import Path

import numpy as np
import pytest

from netsim.agents.scripted import ScriptedAttacker, ScriptedDefender
from netsim.environment import (
    AttackerActionType,
    DefenderAction,
    DefenderActionType,
    GameEngine,
)
from netsim.environment.attacker_env import make_random_policy
from netsim.evaluation.baselines import (
    random_attacker_factory,
    random_defender_factory,
    run_matchup,
    scripted_attacker_factory,
    scripted_defender_factory,
)

SCENARIO_PATH = Path(__file__).parent.parent / "configs" / "scenarios" / "small_enterprise.yaml"


# ------------------------------------------------------------------
# Scripted attacker tests
# ------------------------------------------------------------------


class TestScriptedAttacker:
    @pytest.fixture
    def engine(self):
        return GameEngine(SCENARIO_PATH, seed=42)

    def test_picks_scan_first(self, engine: GameEngine):
        """On first move, should scan the router (only reachable node)."""
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        obs = engine.get_attacker_obs()
        mask = engine.get_attacker_action_mask()
        action_idx = atk(obs, mask)
        action = engine.attacker_actions.decode(action_idx)
        assert action.action_type == AttackerActionType.SCAN
        assert action.target_node == "router"

    def test_exploits_after_scan(self, engine: GameEngine):
        """After scanning router, should exploit it."""
        engine.network.set_scanned("router")
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        obs = engine.get_attacker_obs()
        mask = engine.get_attacker_action_mask()
        action_idx = atk(obs, mask)
        action = engine.attacker_actions.decode(action_idx)
        assert action.action_type == AttackerActionType.EXPLOIT
        assert action.target_node == "router"
        # Should pick highest prob vuln (ssh_bruteforce=0.7 > http_rfi=0.5)
        assert action.vulnerability == "ssh_bruteforce"

    def test_escalates_when_user_access(self, engine: GameEngine):
        """After compromising router with user access, should escalate."""
        engine.network.set_compromised("router", 1)
        engine.attacker_position = "router"
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        obs = engine.get_attacker_obs()
        mask = engine.get_attacker_action_mask()
        action_idx = atk(obs, mask)
        action = engine.attacker_actions.decode(action_idx)
        assert action.action_type == AttackerActionType.ESCALATE

    def test_exfiltrates_when_possible(self, engine: GameEngine):
        """Should exfiltrate immediately when conditions are met."""
        engine.network.set_compromised("server", 2)
        engine.attacker_position = "server"
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        obs = engine.get_attacker_obs()
        mask = engine.get_attacker_action_mask()
        action_idx = atk(obs, mask)
        action = engine.attacker_actions.decode(action_idx)
        assert action.action_type == AttackerActionType.EXFILTRATE

    def test_moves_toward_target(self, engine: GameEngine):
        """After compromising router with root, should scan server (on path to data)."""
        engine.network.set_scanned("router")
        engine.network.set_compromised("router", 2)
        engine.attacker_position = "router"
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        obs = engine.get_attacker_obs()
        mask = engine.get_attacker_action_mask()
        action_idx = atk(obs, mask)
        action = engine.attacker_actions.decode(action_idx)
        # Should scan server (next on path to data node)
        assert action.action_type == AttackerActionType.SCAN
        assert action.target_node == "server"


# ------------------------------------------------------------------
# Scripted defender tests
# ------------------------------------------------------------------


class TestScriptedDefender:
    @pytest.fixture
    def engine(self):
        return GameEngine(SCENARIO_PATH, seed=42)

    def test_monitors_priority_nodes(self, engine: GameEngine):
        """Should monitor server or its neighbors first."""
        dfn = ScriptedDefender(engine.network, engine.defender_actions)
        obs = engine.get_defender_obs()
        mask = engine.get_defender_action_mask()
        action_idx = dfn(obs, mask)
        action = engine.defender_actions.decode(action_idx)
        assert action.action_type == DefenderActionType.MONITOR
        # Should target server or router (server's neighbor)
        assert action.target_node in {"server", "router"}

    def test_isolates_on_high_estimate(self, engine: GameEngine):
        """Should isolate monitored node with high compromise estimate."""
        dfn = ScriptedDefender(engine.network, engine.defender_actions)
        # Set router as monitored with high estimate
        engine.network.set_monitored("router", 3)
        engine.network.set_compromised("router", 1)
        # Simulate attacker acting loudly on router
        engine._attacker_acted_on["router"] = 0.4
        obs = engine.get_defender_obs()
        mask = engine.get_defender_action_mask()
        action_idx = dfn(obs, mask)
        action = engine.defender_actions.decode(action_idx)
        assert action.action_type == DefenderActionType.ISOLATE
        assert action.target_node == "router"

    def test_restores_after_cooldown(self, engine: GameEngine):
        """Should restore isolated nodes only after cooldown period."""
        dfn = ScriptedDefender(engine.network, engine.defender_actions)
        engine.network.set_isolated("router", True)
        # Monitor everything so there's no urgent monitoring to do
        for node_id in engine.network.node_ids:
            if not engine.network.get_config(node_id).is_entry:
                engine.network.set_monitored(node_id, 3)

        # First call — isolation age is 1, below cooldown of 3
        obs = engine.get_defender_obs()
        mask = engine.get_defender_action_mask()
        action_idx = dfn(obs, mask)
        action = engine.defender_actions.decode(action_idx)
        assert action.action_type != DefenderActionType.RESTORE

        # Second call — age 2, still below cooldown
        action_idx = dfn(obs, mask)
        action = engine.defender_actions.decode(action_idx)
        assert action.action_type != DefenderActionType.RESTORE

        # Third call — age 3, meets cooldown → restore
        action_idx = dfn(obs, mask)
        action = engine.defender_actions.decode(action_idx)
        assert action.action_type == DefenderActionType.RESTORE
        assert action.target_node == "router"


# ------------------------------------------------------------------
# Baseline matchup tests
# ------------------------------------------------------------------


class TestMatchups:
    def test_scripted_attacker_beats_random_defender(self):
        """Scripted attacker should win significantly against random defender."""
        result = run_matchup(
            SCENARIO_PATH, scripted_attacker_factory, random_defender_factory,
            n_episodes=200, seed=0,
        )
        assert result["attacker_win_rate"] > 0.3, (
            f"Scripted attacker too weak: {result['attacker_win_rate']:.1%} win rate"
        )

    def test_scripted_defender_beats_random_attacker(self):
        """Scripted defender should win significantly against random attacker."""
        result = run_matchup(
            SCENARIO_PATH, random_attacker_factory, scripted_defender_factory,
            n_episodes=200, seed=0,
        )
        assert result["defender_win_rate"] > 0.5, (
            f"Scripted defender too weak: {result['defender_win_rate']:.1%} win rate"
        )

    def test_scripted_defender_dominates_scripted_attacker(self):
        """On a single-chokepoint network, a smart defender should dominate
        a predictable attacker. The defender monitors the chokepoint (router),
        isolates on detection, and holds isolation long enough to block the
        attacker's only path. An RL attacker would need to learn timing or
        alternative strategies to beat this."""
        result = run_matchup(
            SCENARIO_PATH, scripted_attacker_factory, scripted_defender_factory,
            n_episodes=200, seed=0,
        )
        assert result["defender_win_rate"] > 0.8, (
            f"Scripted defender should dominate: {result['defender_win_rate']:.1%}"
        )

    def test_random_vs_random_completes(self):
        """Random vs random should complete all episodes."""
        result = run_matchup(
            SCENARIO_PATH, random_attacker_factory, random_defender_factory,
            n_episodes=100, seed=0,
        )
        assert result["n_episodes"] == 100
        assert result["attacker_wins"] + result["defender_wins"] == 100


# ------------------------------------------------------------------
# Episode walkthrough — verify game logic makes sense
# ------------------------------------------------------------------


class TestEpisodeWalkthrough:
    def test_full_attacker_win_scenario(self):
        """Manually step through a scripted attacker winning against a passive defender."""
        engine = GameEngine(SCENARIO_PATH, seed=0)
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        # Defender always waits
        wait_idx = engine.defender_actions.encode(DefenderAction(DefenderActionType.WAIT))
        passive_def = lambda obs, mask: wait_idx

        actions_taken = []
        while not engine.done:
            engine.begin_turn()
            obs = engine.get_attacker_obs()
            mask = engine.get_attacker_action_mask()
            atk_idx = atk(obs, mask)
            atk_action = engine.attacker_actions.decode(atk_idx)
            info = engine.execute_attacker_action(atk_action)
            actions_taken.append((atk_action, info["success"]))

            if engine.done:
                break

            def_obs = engine.get_defender_obs()
            def_mask = engine.get_defender_action_mask()
            passive_def_idx = passive_def(def_obs, def_mask)
            def_action = engine.defender_actions.decode(passive_def_idx)
            engine.execute_defender_action(def_action)
            engine.end_turn()

        # Attacker should eventually win against a passive defender
        assert engine.winner == "attacker", (
            f"Expected attacker win, got {engine.winner}. "
            f"Actions: {[(str(a.action_type), a.target_node, s) for a, s in actions_taken]}"
        )

        # Verify the action sequence makes sense
        action_types = [a.action_type for a, _ in actions_taken]
        # Should start with SCAN
        assert action_types[0] == AttackerActionType.SCAN
        # Should end with EXFILTRATE
        assert action_types[-1] == AttackerActionType.EXFILTRATE

    def test_episode_length_reasonable(self):
        """Episodes should not take too few or too many steps."""
        engine = GameEngine(SCENARIO_PATH, seed=0)
        atk = ScriptedAttacker(engine.network, engine.attacker_actions)
        wait_idx = engine.defender_actions.encode(DefenderAction(DefenderActionType.WAIT))
        passive_def = lambda obs, mask: wait_idx

        lengths = []
        for seed in range(50):
            engine.reset(seed=seed)
            while not engine.done:
                engine.begin_turn()
                obs = engine.get_attacker_obs()
                mask = engine.get_attacker_action_mask()
                atk_idx = atk(obs, mask)
                atk_action = engine.attacker_actions.decode(atk_idx)
                engine.execute_attacker_action(atk_action)
                if engine.done:
                    break
                def_obs = engine.get_defender_obs()
                def_mask = engine.get_defender_action_mask()
                engine.execute_defender_action(
                    engine.defender_actions.decode(passive_def(def_obs, def_mask))
                )
                engine.end_turn()
            if engine.winner == "attacker":
                lengths.append(engine.step_count)

        assert len(lengths) > 0, "Scripted attacker never won in 50 episodes"
        mean_len = np.mean(lengths)
        # Minimum path: scan router, exploit, escalate, scan server, exploit,
        # escalate, move, exfiltrate = ~8 steps minimum (with RNG luck)
        # Should be reasonable, not 1 step or 49 steps
        assert 5 < mean_len < 40, f"Mean episode length looks wrong: {mean_len:.1f}"
