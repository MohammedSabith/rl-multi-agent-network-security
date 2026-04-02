"""Self-play training loop with MaskablePPO.

Supports:
1. Single-agent training (RL vs scripted/random opponent)
2. Alternating self-play with opponent sampling (anti-forgetting)
3. Checkpoint evaluation against all past versions + scripted baselines
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

from netsim.environment import AttackerEnv, DefenderEnv, GameEngine
from netsim.environment.attacker_env import OpponentPolicy, make_random_policy
from netsim.agents.scripted import ScriptedAttacker, ScriptedDefender
from netsim.evaluation.baselines import run_matchup, PolicyFactory


DEFAULT_PPO_KWARGS = {
    "policy_kwargs": {"net_arch": [128, 128]},
    "learning_rate": 2e-4,
    "batch_size": 128,
    "n_epochs": 15,
    "ent_coef": 0.05,
    "gamma": 0.99,
}


def make_rl_policy(model: MaskablePPO) -> OpponentPolicy:
    """Wrap a trained MaskablePPO model as an opponent policy callable."""
    def policy(obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return policy


# ------------------------------------------------------------------
# Opponent sampling wrapper
# ------------------------------------------------------------------

class OpponentSamplingWrapper(gym.Wrapper):
    """Samples a different opponent policy each episode (at reset)."""

    def __init__(
        self,
        env: AttackerEnv | DefenderEnv,
        factories: list,
        weights: list[float],
        rng: np.random.Generator,
    ) -> None:
        super().__init__(env)
        self._factories = factories
        self._weights = np.array(weights, dtype=np.float64)
        self._weights /= self._weights.sum()
        self._rng = rng

    def reset(self, **kwargs):
        idx = int(self._rng.choice(len(self._factories), p=self._weights))
        self.env.opponent_policy = self._factories[idx](self.env.engine)
        return self.env.reset(**kwargs)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


def _make_frozen_factory(frozen_policy: OpponentPolicy):
    """Factory that always returns the same frozen policy."""
    def factory(engine: GameEngine) -> OpponentPolicy:
        return frozen_policy
    return factory


# ------------------------------------------------------------------
# Single-agent training
# ------------------------------------------------------------------

def train_attacker(
    scenario_path: str | Path,
    opponent_policy: OpponentPolicy | None = None,
    total_timesteps: int = 50_000,
    seed: int = 0,
    reward_shaping: bool = False,
    **ppo_kwargs,
) -> MaskablePPO:
    """Train an attacker agent against a fixed opponent."""
    gamma = ppo_kwargs.get("gamma", 0.99)
    env = AttackerEnv(
        scenario_path, opponent_policy=opponent_policy, seed=seed,
        reward_shaping=reward_shaping, gamma=gamma,
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def train_defender(
    scenario_path: str | Path,
    opponent_policy: OpponentPolicy | None = None,
    total_timesteps: int = 50_000,
    seed: int = 0,
    isolation_cost: float = 0.0,
    **ppo_kwargs,
) -> MaskablePPO:
    """Train a defender agent against a fixed opponent."""
    env = DefenderEnv(
        scenario_path, opponent_policy=opponent_policy, seed=seed,
        isolation_cost=isolation_cost,
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


# ------------------------------------------------------------------
# Self-play loop
# ------------------------------------------------------------------

def self_play(
    scenario_path: str | Path,
    n_rounds: int = 10,
    attacker_timesteps: int = 200_000,
    defender_timesteps: int = 100_000,
    eval_episodes: int = 500,
    output_dir: str | Path = "results/self_play",
    seed: int = 0,
    reward_shaping: bool = True,
    isolation_cost: float = 0.003,
    ppo_kwargs: dict | None = None,
) -> dict:
    """Run alternating self-play with opponent sampling.

    Opponent mix per round (rounds 1+):
      10% random, 20% scripted baseline, 70% latest frozen checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = Path(scenario_path)

    if ppo_kwargs is None:
        ppo_kwargs = dict(DEFAULT_PPO_KWARGS)

    history: dict = {"rounds": [], "checkpoints": {"attacker": [], "defender": []}}

    # Scripted baselines for evaluation
    def scripted_atk_factory(engine: GameEngine) -> OpponentPolicy:
        return ScriptedAttacker(engine.network, engine.attacker_actions)

    def scripted_def_factory(engine: GameEngine) -> OpponentPolicy:
        return ScriptedDefender(engine.network, engine.defender_actions)

    def random_factory(engine: GameEngine) -> OpponentPolicy:
        return make_random_policy(engine.rng.spawn(1)[0])

    # ------------------------------------------------------------------
    # Round 0: warmup against random opponents
    # ------------------------------------------------------------------
    print(f"Round 0: Training attacker vs random ({attacker_timesteps} steps)...")
    atk_model = train_attacker(
        scenario_path, opponent_policy=None,
        total_timesteps=attacker_timesteps, seed=seed,
        reward_shaping=reward_shaping, **ppo_kwargs,
    )
    atk_path = output_dir / "attacker_r0.zip"
    atk_model.save(str(atk_path))
    history["checkpoints"]["attacker"].append(str(atk_path))

    print(f"Round 0: Training defender vs random ({defender_timesteps} steps)...")
    def_model = train_defender(
        scenario_path, opponent_policy=None,
        total_timesteps=defender_timesteps, seed=seed + 1,
        isolation_cost=isolation_cost, **ppo_kwargs,
    )
    def_path = output_dir / "defender_r0.zip"
    def_model.save(str(def_path))
    history["checkpoints"]["defender"].append(str(def_path))

    # Evaluate round 0
    round_results = _evaluate_round(
        scenario_path, atk_model, def_model,
        scripted_atk_factory, scripted_def_factory, random_factory,
        eval_episodes, seed,
    )
    round_results["round"] = 0
    history["rounds"].append(round_results)
    _print_round_results(0, round_results)

    # ------------------------------------------------------------------
    # Alternating self-play with opponent sampling
    # ------------------------------------------------------------------
    gamma = ppo_kwargs.get("gamma", 0.99)

    for r in range(1, n_rounds + 1):
        # --- Train attacker with mixed defender opponents ---
        print(f"\nRound {r}: Training attacker vs mixed defenders ({attacker_timesteps} steps)...")
        frozen_def = make_rl_policy(def_model)
        def_factories = [
            lambda e: make_random_policy(e.rng.spawn(1)[0]),
            lambda e: ScriptedDefender(e.network, e.defender_actions),
            _make_frozen_factory(frozen_def),
        ]
        def_weights = [0.1, 0.2, 0.7]

        atk_env = AttackerEnv(
            scenario_path, seed=seed + r * 2,
            reward_shaping=reward_shaping, gamma=gamma,
        )
        atk_env = OpponentSamplingWrapper(
            atk_env, def_factories, def_weights,
            np.random.default_rng(seed + r * 2 + 1000),
        )
        atk_model = MaskablePPO(
            "MlpPolicy", atk_env, seed=seed + r * 2, verbose=0, **ppo_kwargs,
        )
        atk_model.learn(total_timesteps=attacker_timesteps)

        atk_path = output_dir / f"attacker_r{r}.zip"
        atk_model.save(str(atk_path))
        history["checkpoints"]["attacker"].append(str(atk_path))

        # --- Train defender with mixed attacker opponents ---
        print(f"Round {r}: Training defender vs mixed attackers ({defender_timesteps} steps)...")
        frozen_atk = make_rl_policy(atk_model)
        atk_factories = [
            lambda e: make_random_policy(e.rng.spawn(1)[0]),
            lambda e: ScriptedAttacker(e.network, e.attacker_actions),
            _make_frozen_factory(frozen_atk),
        ]
        atk_weights = [0.1, 0.2, 0.7]

        def_env = DefenderEnv(
            scenario_path, seed=seed + r * 2 + 1,
            isolation_cost=isolation_cost,
        )
        def_env = OpponentSamplingWrapper(
            def_env, atk_factories, atk_weights,
            np.random.default_rng(seed + r * 2 + 2000),
        )
        def_model = MaskablePPO(
            "MlpPolicy", def_env, seed=seed + r * 2 + 1, verbose=0, **ppo_kwargs,
        )
        def_model.learn(total_timesteps=defender_timesteps)

        def_path = output_dir / f"defender_r{r}.zip"
        def_model.save(str(def_path))
        history["checkpoints"]["defender"].append(str(def_path))

        # Evaluate this round
        round_results = _evaluate_round(
            scenario_path, atk_model, def_model,
            scripted_atk_factory, scripted_def_factory, random_factory,
            eval_episodes, seed,
        )
        round_results["round"] = r
        history["rounds"].append(round_results)
        _print_round_results(r, round_results)

    # Save history
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")

    return history


def _evaluate_round(
    scenario_path: Path,
    atk_model: MaskablePPO,
    def_model: MaskablePPO,
    scripted_atk_factory: PolicyFactory,
    scripted_def_factory: PolicyFactory,
    random_factory: PolicyFactory,
    n_episodes: int,
    seed: int,
) -> dict:
    """Evaluate current models against each other and baselines."""
    frozen_atk = make_rl_policy(atk_model)
    frozen_def = make_rl_policy(def_model)

    def rl_atk_factory(engine: GameEngine) -> OpponentPolicy:
        return frozen_atk

    def rl_def_factory(engine: GameEngine) -> OpponentPolicy:
        return frozen_def

    results = {}

    # RL vs RL
    r = run_matchup(scenario_path, rl_atk_factory, rl_def_factory, n_episodes, seed)
    results["rl_vs_rl"] = r["attacker_win_rate"]

    # RL attacker vs scripted defender
    r = run_matchup(scenario_path, rl_atk_factory, scripted_def_factory, n_episodes, seed)
    results["rl_atk_vs_scripted_def"] = r["attacker_win_rate"]

    # RL attacker vs random defender
    r = run_matchup(scenario_path, rl_atk_factory, random_factory, n_episodes, seed)
    results["rl_atk_vs_random_def"] = r["attacker_win_rate"]

    # Scripted attacker vs RL defender
    r = run_matchup(scenario_path, scripted_atk_factory, rl_def_factory, n_episodes, seed)
    results["scripted_atk_vs_rl_def"] = r["attacker_win_rate"]

    # Random attacker vs RL defender
    r = run_matchup(scenario_path, random_factory, rl_def_factory, n_episodes, seed)
    results["random_atk_vs_rl_def"] = r["attacker_win_rate"]

    return results


def _print_round_results(round_num: int, results: dict) -> None:
    """Print evaluation results for a round."""
    print(f"  Round {round_num} evaluation:")
    for key, val in results.items():
        if key == "round":
            continue
        print(f"    {key}: {val:.1%} attacker win rate")
