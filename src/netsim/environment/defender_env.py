"""Gymnasium environment for training the defender agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
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


class DefenderEnv(gym.Env):
    """Single-agent Gymnasium env from the defender's perspective.

    Each step():
      1. Opponent (attacker) acts via injected policy
      2. Execute defender action
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
        isolation_cost: float = 0.0,
    ) -> None:
        super().__init__()
        self.engine = GameEngine(scenario_path, monitor_duration, seed)
        self.opponent_policy = opponent_policy or make_random_policy(self.engine.rng)
        self._isolation_cost = isolation_cost

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.engine.defender_obs_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.engine.defender_actions.n)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        obs = self.engine.get_defender_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.engine.begin_turn()

        # 1. Attacker acts first
        atk_obs = self.engine.get_attacker_obs()
        atk_mask = self.engine.get_attacker_action_mask()
        atk_action_idx = self.opponent_policy(atk_obs, atk_mask)
        atk_action = self.engine.attacker_actions.decode(atk_action_idx)
        info = self.engine.execute_attacker_action(atk_action)

        # Check if game ended from attacker action (exfiltration)
        if not self.engine.done:
            # 2. Defender acts
            defender_action = self.engine.defender_actions.decode(action)
            def_info = self.engine.execute_defender_action(defender_action)
            info.update({"defender_" + k: v for k, v in def_info.items()})

            # 3. Compute observation BEFORE end_turn so the defender sees
            # detection signals from monitored nodes that are about to expire
            # (end_turn ticks monitoring, which could drop signals).
            obs = self.engine.get_defender_obs()

            # 4. End turn (tick monitoring, check timeout)
            self.engine.end_turn()

            # Patch progress to reflect the post-end_turn step count,
            # since obs was computed before end_turn incremented it.
            obs[-1] = self.engine.step_count / max(
                self.engine.network.max_steps, 1
            )
        else:
            obs = self.engine.get_defender_obs()

        reward = self._compute_reward()
        terminated = self.engine.done

        return obs, reward, terminated, False, info

    def _compute_reward(self) -> float:
        if self.engine.done:
            return 1.0 if self.engine.winner == "defender" else -1.0
        # Per-step cost for each isolated node (models operational disruption)
        if self._isolation_cost > 0.0:
            n_isolated = sum(
                1 for n in self.engine.network.node_ids
                if self.engine.network.get_state(n).isolated
            )
            return -self._isolation_cost * n_isolated
        return 0.0

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO to get valid actions."""
        return self.engine.get_defender_action_mask()
