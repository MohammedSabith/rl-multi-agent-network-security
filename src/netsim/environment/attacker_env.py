"""Gymnasium environment for training the attacker agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import networkx as nx
import numpy as np

from .game import GameEngine


# Type for opponent policy: (observation, action_mask) -> action_index
OpponentPolicy = Callable[[np.ndarray, np.ndarray], int]


def make_random_policy(rng: np.random.Generator) -> OpponentPolicy:
    """Create a random policy that uses the given seeded RNG."""
    def policy(obs: np.ndarray, mask: np.ndarray) -> int:
        valid = np.where(mask)[0]
        return int(rng.choice(valid))
    return policy


class AttackerEnv(gym.Env):
    """Single-agent Gymnasium env from the attacker's perspective.

    Each step():
      1. Execute attacker action
      2. Opponent (defender) responds via injected policy
      3. End turn (tick monitoring, check termination)
      4. Return new observation
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str | Path,
        opponent_policy: OpponentPolicy | None = None,
        monitor_duration: int = 3,
        seed: int | None = None,
        reward_shaping: bool = False,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.engine = GameEngine(scenario_path, monitor_duration, seed)
        self.opponent_policy = opponent_policy or make_random_policy(self.engine.rng)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.engine.attacker_obs_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.engine.attacker_actions.n)

        # PBRS (Potential-Based Reward Shaping)
        self._reward_shaping = reward_shaping
        self._gamma = gamma
        self._prev_potential = 0.0
        if reward_shaping:
            self._graph_diameter = nx.diameter(self.engine.network._graph)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        obs = self.engine.get_attacker_obs()
        if self._reward_shaping:
            self._prev_potential = self._compute_potential()
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.engine.begin_turn()

        # 1. Attacker acts
        attacker_action = self.engine.attacker_actions.decode(action)
        info = self.engine.execute_attacker_action(attacker_action)

        # Check if game ended from attacker action (exfiltration)
        if not self.engine.done:
            # 2. Defender responds
            def_obs = self.engine.get_defender_obs()
            def_mask = self.engine.get_defender_action_mask()
            def_action_idx = self.opponent_policy(def_obs, def_mask)
            def_action = self.engine.defender_actions.decode(def_action_idx)
            self.engine.execute_defender_action(def_action)

            # 3. End turn
            self.engine.end_turn()

        # 4. Compute reward and observation
        reward = self._compute_reward()
        obs = self.engine.get_attacker_obs()
        terminated = self.engine.done

        return obs, reward, terminated, False, info

    def _compute_reward(self) -> float:
        env_reward = 0.0
        if self.engine.done:
            env_reward = 1.0 if self.engine.winner == "attacker" else -1.0

        if not self._reward_shaping:
            return env_reward

        # PBRS: F(s,a,s') = gamma * phi(s') - phi(s)
        # At terminal: phi(s') = 0 by convention
        if self.engine.done:
            shaped = -self._prev_potential
            self._prev_potential = 0.0
        else:
            new_potential = self._compute_potential()
            shaped = self._gamma * new_potential - self._prev_potential
            self._prev_potential = new_potential

        return env_reward + shaped

    def _compute_potential(self) -> float:
        """PBRS potential: compromised fraction + distance progress + data access."""
        net = self.engine.network
        non_entry = [n for n in net.node_ids if not net.get_config(n).is_entry]
        n_non_entry = len(non_entry)

        # 20%: fraction of non-entry nodes compromised
        comp_frac = sum(
            1 for n in non_entry if net.get_state(n).compromised
        ) / n_non_entry

        # 50%: 1 - (min distance from any compromised to any data / diameter)
        compromised = net.get_compromised_nodes()
        data_nodes = net.data_nodes
        min_dist = float("inf")
        for c in compromised:
            for d in data_nodes:
                try:
                    dist = nx.shortest_path_length(net._graph, c, d)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    pass
        if min_dist == float("inf"):
            distance_progress = 0.0
        else:
            distance_progress = 1.0 - (min_dist / self._graph_diameter)

        # 30%: max access level on any data node (0/0.5/1.0)
        max_data_access = 0.0
        for d in data_nodes:
            state = net.get_state(d)
            if state.compromised:
                max_data_access = max(max_data_access, state.access_level / 2.0)

        return 0.2 * comp_frac + 0.5 * distance_progress + 0.3 * max_data_access

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO to get valid actions."""
        return self.engine.get_attacker_action_mask()
