from pathlib import Path

import numpy as np
import pytest

from netsim.environment import (
    AttackerAction,
    AttackerActionSpace,
    AttackerActionType,
    AttackerEnv,
    DefenderAction,
    DefenderActionSpace,
    DefenderActionType,
    DefenderEnv,
    GameEngine,
)

SCENARIO_PATH = Path(__file__).parent.parent / "configs" / "scenarios" / "small_enterprise.yaml"


# ------------------------------------------------------------------
# Action space tests
# ------------------------------------------------------------------


class TestActionSpaces:
    def test_attacker_action_count(self):
        engine = GameEngine(SCENARIO_PATH)
        space = engine.attacker_actions
        # scan(5) + exploit(0+2+1+1+2=6) + escalate(5) + move(5) + exfiltrate(5) + wait(1) = 27
        assert space.n == 27

    def test_defender_action_count(self):
        engine = GameEngine(SCENARIO_PATH)
        space = engine.defender_actions
        # monitor(5) + isolate(5) + restore(5) + analyze(5) + wait(1) = 21
        assert space.n == 21

    def test_attacker_encode_decode_roundtrip(self):
        engine = GameEngine(SCENARIO_PATH)
        space = engine.attacker_actions
        for i in range(space.n):
            action = space.decode(i)
            assert space.encode(action) == i

    def test_defender_encode_decode_roundtrip(self):
        engine = GameEngine(SCENARIO_PATH)
        space = engine.defender_actions
        for i in range(space.n):
            action = space.decode(i)
            assert space.encode(action) == i


# ------------------------------------------------------------------
# Game engine tests
# ------------------------------------------------------------------


class TestGameEngine:
    @pytest.fixture
    def engine(self):
        return GameEngine(SCENARIO_PATH, seed=42)

    def test_initial_state(self, engine: GameEngine):
        assert engine.attacker_position == "internet"
        assert engine.step_count == 0
        assert engine.done is False
        assert engine.winner is None

    def test_scan_action(self, engine: GameEngine):
        action = AttackerAction(AttackerActionType.SCAN, "router")
        info = engine.execute_attacker_action(action)
        assert info["success"] is True
        assert engine.network.get_state("router").scanned is True

    def test_exploit_success(self, engine: GameEngine):
        """With seed=42 and prob=0.7, ssh_bruteforce on router should succeed."""
        engine.network.set_scanned("router")
        action = AttackerAction(AttackerActionType.EXPLOIT, "router", "ssh_bruteforce")
        # Run multiple times with fresh engines to verify stochasticity
        successes = 0
        for seed in range(100):
            e = GameEngine(SCENARIO_PATH, seed=seed)
            e.network.set_scanned("router")
            info = e.execute_attacker_action(action)
            if info["success"]:
                successes += 1
        # With prob=0.7, expect ~70 successes out of 100
        assert 50 < successes < 90

    def test_escalate_action(self, engine: GameEngine):
        engine.network.set_compromised("router", 1)
        engine.attacker_position = "router"
        action = AttackerAction(AttackerActionType.ESCALATE, "router")
        # Run multiple times
        successes = 0
        for seed in range(100):
            e = GameEngine(SCENARIO_PATH, seed=seed)
            e.network.set_compromised("router", 1)
            e.attacker_position = "router"
            info = e.execute_attacker_action(action)
            if info["success"]:
                successes += 1
        # privesc_prob = 0.6
        assert 40 < successes < 80

    def test_move_action(self, engine: GameEngine):
        engine.network.set_compromised("router", 1)
        action = AttackerAction(AttackerActionType.MOVE, "router")
        info = engine.execute_attacker_action(action)
        assert info["success"] is True
        assert engine.attacker_position == "router"

    def test_exfiltrate_ends_game(self, engine: GameEngine):
        engine.network.set_compromised("server", 2)
        engine.attacker_position = "server"
        action = AttackerAction(AttackerActionType.EXFILTRATE, "server")
        info = engine.execute_attacker_action(action)
        assert info["success"] is True
        assert engine.done is True
        assert engine.winner == "attacker"

    def test_isolate_ejects_attacker(self, engine: GameEngine):
        engine.attacker_position = "router"
        action = DefenderAction(DefenderActionType.ISOLATE, "router")
        engine.execute_defender_action(action)
        assert engine.network.get_state("router").isolated is True
        assert engine.attacker_position == "internet"

    def test_restore_cleans_node(self, engine: GameEngine):
        engine.network.set_compromised("router", 2)
        engine.network.set_isolated("router", True)
        action = DefenderAction(DefenderActionType.RESTORE, "router")
        engine.execute_defender_action(action)
        state = engine.network.get_state("router")
        assert state.compromised is False
        assert state.access_level == 0
        assert state.isolated is False

    def test_restore_ejects_attacker(self, engine: GameEngine):
        engine.network.set_compromised("router", 1)
        engine.attacker_position = "router"
        action = DefenderAction(DefenderActionType.RESTORE, "router")
        engine.execute_defender_action(action)
        assert engine.attacker_position == "internet"

    def test_timeout_defender_wins(self, engine: GameEngine):
        for _ in range(engine.network.max_steps):
            engine.end_turn()
        assert engine.done is True
        assert engine.winner == "defender"

    def test_reset_clears_state(self, engine: GameEngine):
        engine.network.set_compromised("router", 2)
        engine.attacker_position = "router"
        engine.step_count = 10
        engine.done = True
        engine.reset()
        assert engine.attacker_position == "internet"
        assert engine.step_count == 0
        assert engine.done is False


# ------------------------------------------------------------------
# Action mask tests
# ------------------------------------------------------------------


class TestActionMasks:
    @pytest.fixture
    def engine(self):
        return GameEngine(SCENARIO_PATH, seed=42)

    def test_initial_attacker_mask(self, engine: GameEngine):
        mask = engine.get_attacker_action_mask()
        # Initially only scan(router) and wait should be valid
        # (internet is only adjacent to router, and router is the only reachable node)
        valid_actions = [
            engine.attacker_actions.decode(i)
            for i in range(engine.attacker_actions.n)
            if mask[i]
        ]
        action_types = {a.action_type for a in valid_actions}
        assert action_types == {AttackerActionType.SCAN, AttackerActionType.WAIT}
        scan_targets = [a.target_node for a in valid_actions if a.action_type == AttackerActionType.SCAN]
        assert scan_targets == ["router"]

    def test_exploit_available_after_scan(self, engine: GameEngine):
        engine.network.set_scanned("router")
        mask = engine.get_attacker_action_mask()
        valid_actions = [
            engine.attacker_actions.decode(i)
            for i in range(engine.attacker_actions.n)
            if mask[i]
        ]
        exploit_actions = [
            a for a in valid_actions if a.action_type == AttackerActionType.EXPLOIT
        ]
        assert len(exploit_actions) == 2  # ssh_bruteforce and http_rfi on router
        assert all(a.target_node == "router" for a in exploit_actions)

    def test_escalate_available_when_compromised(self, engine: GameEngine):
        engine.network.set_compromised("router", 1)
        engine.attacker_position = "router"
        mask = engine.get_attacker_action_mask()
        valid_actions = [
            engine.attacker_actions.decode(i)
            for i in range(engine.attacker_actions.n)
            if mask[i]
        ]
        escalate_actions = [
            a for a in valid_actions if a.action_type == AttackerActionType.ESCALATE
        ]
        assert len(escalate_actions) == 1
        assert escalate_actions[0].target_node == "router"

    def test_exfiltrate_needs_root_and_data(self, engine: GameEngine):
        # User access only — can't exfiltrate
        engine.network.set_compromised("server", 1)
        engine.attacker_position = "server"
        mask = engine.get_attacker_action_mask()
        exfil = [
            engine.attacker_actions.decode(i)
            for i in range(engine.attacker_actions.n)
            if mask[i] and engine.attacker_actions.decode(i).action_type == AttackerActionType.EXFILTRATE
        ]
        assert len(exfil) == 0

        # Root access — can exfiltrate
        engine.network.set_compromised("server", 2)
        mask = engine.get_attacker_action_mask()
        exfil = [
            engine.attacker_actions.decode(i)
            for i in range(engine.attacker_actions.n)
            if mask[i] and engine.attacker_actions.decode(i).action_type == AttackerActionType.EXFILTRATE
        ]
        assert len(exfil) == 1

    def test_wait_always_valid(self, engine: GameEngine):
        mask = engine.get_attacker_action_mask()
        wait_idx = engine.attacker_actions.encode(
            AttackerAction(AttackerActionType.WAIT)
        )
        assert mask[wait_idx]

    def test_defender_cannot_isolate_entry(self, engine: GameEngine):
        mask = engine.get_defender_action_mask()
        isolate_internet = engine.defender_actions.encode(
            DefenderAction(DefenderActionType.ISOLATE, "internet")
        )
        assert not mask[isolate_internet]

    def test_defender_restore_requires_isolation(self, engine: GameEngine):
        restore_router = engine.defender_actions.encode(
            DefenderAction(DefenderActionType.RESTORE, "router")
        )

        # Clean and not isolated — restore invalid
        mask = engine.get_defender_action_mask()
        assert not mask[restore_router]

        # Compromised but NOT isolated — still invalid (mask must not
        # leak compromised state to the defender)
        engine.network.set_compromised("router", 1)
        mask = engine.get_defender_action_mask()
        assert not mask[restore_router]

        # Isolated — restore becomes valid
        engine.network.set_isolated("router", True)
        mask = engine.get_defender_action_mask()
        assert mask[restore_router]


# ------------------------------------------------------------------
# Observation tests
# ------------------------------------------------------------------


class TestObservations:
    @pytest.fixture
    def engine(self):
        return GameEngine(SCENARIO_PATH, seed=42)

    def test_attacker_obs_shape(self, engine: GameEngine):
        obs = engine.get_attacker_obs()
        assert obs.shape == (engine.attacker_obs_size,)
        # 5 nodes * 5 features + 5 position onehot + 1 progress = 31
        assert obs.shape == (31,)

    def test_defender_obs_shape(self, engine: GameEngine):
        obs = engine.get_defender_obs()
        assert obs.shape == (engine.defender_obs_size,)
        # 5 nodes * 4 features + 1 progress = 21
        assert obs.shape == (21,)

    def test_attacker_obs_initial_values(self, engine: GameEngine):
        obs = engine.get_attacker_obs()
        # Internet (index 0) should be: scanned=1, compromised=1, access=1.0, isolated=0, reachable=0
        assert obs[0] == 1.0  # scanned
        assert obs[1] == 1.0  # compromised
        assert obs[2] == 1.0  # access_level 2/2
        assert obs[3] == 0.0  # not isolated
        # Position onehot — internet is first node
        n = engine.network.num_nodes
        pos_start = n * engine.ATTACKER_NODE_FEATURES
        assert obs[pos_start] == 1.0  # internet position

    def test_defender_obs_range(self, engine: GameEngine):
        """All defender observations should be in [0, 1]."""
        for _ in range(20):
            obs = engine.get_defender_obs()
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
            engine.step_count += 1

    def test_monitored_node_accurate_estimate(self):
        """Monitored clean node should produce low estimates."""
        estimates_monitored = []
        for seed in range(200):
            engine = GameEngine(SCENARIO_PATH, seed=seed)
            engine.network.set_monitored("router", 3)
            estimates = engine._compute_compromise_estimates()
            estimates_monitored.append(estimates["router"])
        # Router is clean (true_val=0), monitored, no attacker action
        # Expect estimates near 0, definitely below 0.5 on average
        mean = np.mean(estimates_monitored)
        assert mean < 0.3, f"Monitored clean node mean estimate too high: {mean:.3f}"

    def test_unmonitored_node_noisy_estimate(self):
        """Unmonitored node should produce random estimates ~0.5."""
        estimates_unmonitored = []
        for seed in range(200):
            engine = GameEngine(SCENARIO_PATH, seed=seed)
            estimates = engine._compute_compromise_estimates()
            estimates_unmonitored.append(estimates["router"])
        mean = np.mean(estimates_unmonitored)
        # Should be near 0.5 (uniform noise)
        assert 0.3 < mean < 0.7, f"Unmonitored mean too skewed: {mean:.3f}"

    def test_analyzed_node_exact_estimate(self):
        engine = GameEngine(SCENARIO_PATH, seed=42)
        engine._analyzed_nodes.add("router")
        estimates = engine._compute_compromise_estimates()
        assert estimates["router"] == 0.0  # clean node, exact

        engine.network.set_compromised("router", 1)
        estimates = engine._compute_compromise_estimates()
        assert estimates["router"] == 1.0  # compromised node, exact

    def test_analysis_persists_until_next_defender_action(self):
        """Analysis results must survive begin_turn so the defender opponent
        in AttackerEnv can see them."""
        engine = GameEngine(SCENARIO_PATH, seed=42)
        engine.network.set_compromised("router", 1)

        # Defender analyzes router
        engine.execute_defender_action(
            DefenderAction(DefenderActionType.ANALYZE, "router")
        )
        assert "router" in engine._analyzed_nodes

        # begin_turn should NOT clear analysis
        engine.begin_turn()
        assert "router" in engine._analyzed_nodes

        # Next defender action SHOULD clear old analysis
        engine.execute_defender_action(
            DefenderAction(DefenderActionType.WAIT)
        )
        assert "router" not in engine._analyzed_nodes

    def test_loudness_boosts_signal(self):
        """Attacker action on monitored node should produce higher estimate."""
        estimates_quiet = []
        estimates_loud = []
        for seed in range(200):
            # Quiet: monitored, no attacker action
            engine = GameEngine(SCENARIO_PATH, seed=seed)
            engine.network.set_monitored("router", 3)
            estimates = engine._compute_compromise_estimates()
            estimates_quiet.append(estimates["router"])

            # Loud: monitored, attacker exploited (loudness=0.4)
            engine2 = GameEngine(SCENARIO_PATH, seed=seed)
            engine2.network.set_monitored("router", 3)
            engine2._attacker_acted_on["router"] = 0.4
            estimates2 = engine2._compute_compromise_estimates()
            estimates_loud.append(estimates2["router"])

        assert np.mean(estimates_loud) > np.mean(estimates_quiet) + 0.2


# ------------------------------------------------------------------
# Full environment tests
# ------------------------------------------------------------------


class TestAttackerEnv:
    def test_gymnasium_compliance(self):
        env = AttackerEnv(SCENARIO_PATH, seed=0)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_action_masks_method(self):
        env = AttackerEnv(SCENARIO_PATH, seed=0)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool
        assert mask.any()  # at least wait is valid

    def test_episode_runs_to_completion(self):
        """Random agents should complete an episode within max_steps."""
        env = AttackerEnv(SCENARIO_PATH, seed=0)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            assert steps <= 100, "Episode did not terminate"
        assert done is True

    def test_attacker_can_win(self):
        """With enough tries, a random attacker should occasionally win."""
        wins = 0
        for seed in range(200):
            env = AttackerEnv(SCENARIO_PATH, seed=seed)
            obs, _ = env.reset()
            rng = np.random.default_rng(seed)
            done = False
            while not done:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = int(rng.choice(valid))
                obs, reward, done, _, _ = env.step(action)
            if reward > 0:
                wins += 1
        # Random attacker should win sometimes but not always
        assert wins > 0, "Random attacker never won in 200 episodes"
        assert wins < 200, "Random attacker won every episode"


class TestDefenderEnv:
    def test_gymnasium_compliance(self):
        env = DefenderEnv(SCENARIO_PATH, seed=0)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_action_masks_method(self):
        env = DefenderEnv(SCENARIO_PATH, seed=0)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (env.action_space.n,)
        assert mask.dtype == bool
        assert mask.any()

    def test_episode_runs_to_completion(self):
        env = DefenderEnv(SCENARIO_PATH, seed=0)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            assert steps <= 100, "Episode did not terminate"
        assert done is True

    def test_defender_sees_signal_on_expiring_monitor(self):
        """When monitoring has 1 turn left and attacker acts loudly, the
        defender's returned obs should reflect the detection signal (not
        pure noise from expired monitoring)."""
        from netsim.environment.actions import AttackerActionType, AttackerAction
        from netsim.environment.defender_env import make_random_policy

        estimates = []
        for seed in range(100):
            engine = GameEngine(SCENARIO_PATH, seed=seed)
            # Set monitoring to 1 turn (about to expire)
            engine.network.set_monitored("router", 1)
            engine.network.set_scanned("router")

            # Build a defender env with an attacker opponent that always
            # exploits router
            def attacker_exploits_router(obs, mask):
                action = AttackerAction(AttackerActionType.EXPLOIT, "router", "ssh_bruteforce")
                idx = engine.attacker_actions.encode(action)
                if mask[idx]:
                    return idx
                # Fallback to wait if exploit not valid
                wait = AttackerAction(AttackerActionType.WAIT)
                return engine.attacker_actions.encode(wait)

            env = DefenderEnv(SCENARIO_PATH, opponent_policy=attacker_exploits_router, seed=seed)
            env.engine = engine  # inject the pre-configured engine
            # Get defender mask and pick wait
            wait_idx = engine.defender_actions.encode(
                DefenderAction(DefenderActionType.WAIT)
            )
            obs, _, _, _, _ = env.step(wait_idx)
            # Extract router's compromise estimate (node index 1, feature index 3)
            router_estimate = obs[1 * 4 + 3]
            estimates.append(router_estimate)

        mean = np.mean(estimates)
        # Should be elevated (not ~0.5 noise), because obs was computed
        # while monitoring was still active
        assert mean > 0.3, f"Expiring-monitor signal too low: {mean:.3f}"

    def test_seeded_reproducibility(self):
        """Same seed should produce identical episodes."""
        results = []
        for _ in range(2):
            env = DefenderEnv(SCENARIO_PATH, seed=42)
            obs, _ = env.reset(seed=42)
            total_reward = 0.0
            done = False
            while not done:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = int(valid[0])  # deterministic: always pick first valid
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
            results.append(total_reward)
        assert results[0] == results[1], f"Same seed produced different results: {results}"

    def test_rewards_are_opposite(self):
        """Attacker win should give defender -1, timeout should give +1."""
        seed = 0
        # Run both envs with same seed for determinism
        atk_env = AttackerEnv(SCENARIO_PATH, seed=seed)
        def_env = DefenderEnv(SCENARIO_PATH, seed=seed)

        # Just verify reward logic
        atk_env.engine.done = True
        atk_env.engine.winner = "attacker"
        assert atk_env._compute_reward() == 1.0

        def_env.engine.done = True
        def_env.engine.winner = "attacker"
        assert def_env._compute_reward() == -1.0

        def_env.engine.winner = "defender"
        assert def_env._compute_reward() == 1.0

        atk_env.engine.winner = "defender"
        assert atk_env._compute_reward() == -1.0
