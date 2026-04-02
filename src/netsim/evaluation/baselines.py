"""Baseline evaluation — run matchups and report win rates."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from netsim.environment import GameEngine
from netsim.environment.attacker_env import OpponentPolicy, make_random_policy
from netsim.agents.scripted import ScriptedAttacker, ScriptedDefender


# Policy factory type: takes a GameEngine, returns a policy callable
PolicyFactory = Callable[[GameEngine], OpponentPolicy]


def random_attacker_factory(engine: GameEngine) -> OpponentPolicy:
    return make_random_policy(engine.rng.spawn(1)[0])


def random_defender_factory(engine: GameEngine) -> OpponentPolicy:
    return make_random_policy(engine.rng.spawn(1)[0])


def scripted_attacker_factory(engine: GameEngine) -> OpponentPolicy:
    return ScriptedAttacker(engine.network, engine.attacker_actions)


def scripted_defender_factory(engine: GameEngine) -> OpponentPolicy:
    return ScriptedDefender(engine.network, engine.defender_actions)


def run_matchup(
    scenario_path: str | Path,
    make_attacker: PolicyFactory,
    make_defender: PolicyFactory,
    n_episodes: int = 500,
    seed: int = 0,
) -> dict:
    """Run n_episodes of attacker vs defender and collect statistics.

    Accepts policy factories so scripted agents always reference the live
    game engine's network state (not a stale copy).
    """
    engine = GameEngine(scenario_path, seed=seed)
    attacker_policy = make_attacker(engine)
    defender_policy = make_defender(engine)

    attacker_wins = 0
    defender_wins = 0
    episode_lengths = []

    for ep in range(n_episodes):
        engine.reset(seed=seed + ep)

        while not engine.done:
            engine.begin_turn()

            # Attacker acts
            atk_obs = engine.get_attacker_obs()
            atk_mask = engine.get_attacker_action_mask()
            atk_idx = attacker_policy(atk_obs, atk_mask)
            atk_action = engine.attacker_actions.decode(atk_idx)
            engine.execute_attacker_action(atk_action)

            if engine.done:
                break

            # Defender acts
            def_obs = engine.get_defender_obs()
            def_mask = engine.get_defender_action_mask()
            def_idx = defender_policy(def_obs, def_mask)
            def_action = engine.defender_actions.decode(def_idx)
            engine.execute_defender_action(def_action)

            engine.end_turn()

        episode_lengths.append(engine.step_count)
        if engine.winner == "attacker":
            attacker_wins += 1
        else:
            defender_wins += 1

    return {
        "n_episodes": n_episodes,
        "attacker_wins": attacker_wins,
        "defender_wins": defender_wins,
        "attacker_win_rate": attacker_wins / n_episodes,
        "defender_win_rate": defender_wins / n_episodes,
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
    }


def run_all_baselines(
    scenario_path: str | Path,
    n_episodes: int = 500,
    seed: int = 0,
) -> dict[str, dict]:
    """Run all baseline matchups and return results."""
    matchups = {
        "random_vs_random": (random_attacker_factory, random_defender_factory),
        "scripted_atk_vs_random_def": (scripted_attacker_factory, random_defender_factory),
        "random_atk_vs_scripted_def": (random_attacker_factory, scripted_defender_factory),
        "scripted_vs_scripted": (scripted_attacker_factory, scripted_defender_factory),
    }

    results = {}
    for name, (atk_factory, def_factory) in matchups.items():
        results[name] = run_matchup(
            scenario_path, atk_factory, def_factory, n_episodes, seed
        )

    return results


def print_baseline_results(results: dict[str, dict]) -> None:
    """Pretty print baseline matchup results."""
    print(f"\n{'Matchup':<35} {'Atk Win%':>8} {'Def Win%':>8} {'Avg Len':>8}")
    print("-" * 63)
    for name, r in results.items():
        print(
            f"{name:<35} {r['attacker_win_rate']:>7.1%} {r['defender_win_rate']:>7.1%} "
            f"{r['mean_episode_length']:>7.1f}"
        )
    print()
