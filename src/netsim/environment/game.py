"""Core game engine — state management, action execution, detection model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netsim.network import Network
from .actions import (
    ACTION_LOUDNESS,
    AttackerAction,
    AttackerActionSpace,
    AttackerActionType,
    DefenderAction,
    DefenderActionSpace,
    DefenderActionType,
)


class GameEngine:
    """Manages the network game state and executes actions for both agents.

    The environment wrappers (AttackerEnv / DefenderEnv) delegate all game
    logic here. This class owns:
      - The Network (topology + mutable node state)
      - Attacker position tracking
      - Action execution (with stochastic outcomes)
      - Detection model (noisy compromise estimates)
      - Action mask computation
      - Observation vector construction
    """

    # Observation sizes per node
    ATTACKER_NODE_FEATURES = 5   # scanned, compromised, access_level, isolated, reachable
    DEFENDER_NODE_FEATURES = 4   # isolated, is_monitored, monitor_remaining_norm, compromise_estimate

    def __init__(
        self,
        scenario_path: str | Path,
        monitor_duration: int = 3,
        seed: int | None = None,
    ) -> None:
        self.network = Network(scenario_path)
        self.monitor_duration = monitor_duration
        self.rng = np.random.default_rng(seed)

        self.attacker_actions = AttackerActionSpace(self.network)
        self.defender_actions = DefenderActionSpace(self.network)

        # Mutable game state (reset each episode)
        self.attacker_position: str = ""
        self.step_count: int = 0
        self.done: bool = False
        self.winner: str | None = None  # "attacker" | "defender" | None

        # Per-turn detection tracking
        self._attacker_acted_on: dict[str, float] = {}  # node_id -> loudness
        self._analyzed_nodes: set[str] = set()

        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.network.reset()
        self.attacker_position = self.network.entry_node
        self.step_count = 0
        self.done = False
        self.winner = None
        self._attacker_acted_on.clear()
        self._analyzed_nodes.clear()

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def begin_turn(self) -> None:
        """Clear per-turn attacker tracking at the start of each turn.

        Note: _analyzed_nodes is NOT cleared here. It persists until the
        defender's next action so the opponent wrapper in AttackerEnv can
        see previous-turn analysis results before the defender acts again.
        """
        self._attacker_acted_on.clear()

    def end_turn(self) -> None:
        """Tick monitoring decay, advance step counter, check termination."""
        self.network.tick_monitoring()
        self.step_count += 1
        if self.step_count >= self.network.max_steps:
            self.done = True
            self.winner = "defender"

    # ------------------------------------------------------------------
    # Attacker action execution
    # ------------------------------------------------------------------

    def execute_attacker_action(self, action: AttackerAction) -> dict:
        """Execute an attacker action. Returns info dict with outcome details."""
        info: dict = {"action": action, "success": False}
        node = action.target_node
        loudness = ACTION_LOUDNESS[action.action_type]

        if action.action_type == AttackerActionType.SCAN:
            self.network.set_scanned(node)
            info["success"] = True
            self._record_loudness(node, loudness)

        elif action.action_type == AttackerActionType.EXPLOIT:
            config = self.network.get_config(node)
            vuln = next(
                v for v in config.vulnerabilities if v.name == action.vulnerability
            )
            if self.rng.random() < vuln.prob_success:
                self.network.set_compromised(node, access_level=1)
                info["success"] = True
            self._record_loudness(node, loudness)

        elif action.action_type == AttackerActionType.ESCALATE:
            config = self.network.get_config(node)
            if self.rng.random() < config.privesc_prob:
                self.network.set_compromised(node, access_level=2)
                info["success"] = True
            self._record_loudness(node, loudness)

        elif action.action_type == AttackerActionType.MOVE:
            self.attacker_position = node
            info["success"] = True
            self._record_loudness(node, loudness)

        elif action.action_type == AttackerActionType.EXFILTRATE:
            # Exfiltration always succeeds if preconditions met (mask ensures this)
            self.done = True
            self.winner = "attacker"
            info["success"] = True
            self._record_loudness(node, loudness)

        elif action.action_type == AttackerActionType.WAIT:
            info["success"] = True

        return info

    def _record_loudness(self, node_id: str, loudness: float) -> None:
        self._attacker_acted_on[node_id] = max(
            self._attacker_acted_on.get(node_id, 0.0), loudness
        )

    # ------------------------------------------------------------------
    # Defender action execution
    # ------------------------------------------------------------------

    def execute_defender_action(self, action: DefenderAction) -> dict:
        """Execute a defender action. Returns info dict.

        Clears previous analysis results before processing the new action,
        so analysis persists for exactly one observation cycle.
        """
        self._analyzed_nodes.clear()
        info: dict = {"action": action, "success": False}
        node = action.target_node

        if action.action_type == DefenderActionType.MONITOR:
            self.network.set_monitored(node, self.monitor_duration)
            info["success"] = True

        elif action.action_type == DefenderActionType.ISOLATE:
            self.network.set_isolated(node, True)
            # If attacker is on this node, eject to entry
            if self.attacker_position == node:
                self.attacker_position = self.network.entry_node
            info["success"] = True

        elif action.action_type == DefenderActionType.RESTORE:
            state = self.network.get_state(node)
            state.compromised = False
            state.access_level = 0
            state.isolated = False
            # If attacker was on this node, eject to entry
            if self.attacker_position == node:
                self.attacker_position = self.network.entry_node
            info["success"] = True

        elif action.action_type == DefenderActionType.ANALYZE:
            self._analyzed_nodes.add(node)
            info["success"] = True

        elif action.action_type == DefenderActionType.WAIT:
            info["success"] = True

        return info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_attacker_obs(self) -> np.ndarray:
        """Build the attacker's observation vector.

        Per node (5 features): scanned, compromised, access_level/2, isolated, reachable
        Global: attacker_position one-hot, progress (step/max_steps)
        """
        n = self.network.num_nodes
        node_features = np.zeros(n * self.ATTACKER_NODE_FEATURES, dtype=np.float32)
        reachable_set = set(self.network.get_attacker_reachable_nodes())

        for i, node_id in enumerate(self.network.node_ids):
            state = self.network.get_state(node_id)
            base = i * self.ATTACKER_NODE_FEATURES
            node_features[base + 0] = float(state.scanned)
            node_features[base + 1] = float(state.compromised)
            node_features[base + 2] = state.access_level / 2.0
            node_features[base + 3] = float(state.isolated)
            node_features[base + 4] = float(node_id in reachable_set)

        # Global features
        position_onehot = np.zeros(n, dtype=np.float32)
        pos_idx = self.network.node_ids.index(self.attacker_position)
        position_onehot[pos_idx] = 1.0

        progress = np.array(
            [self.step_count / max(self.network.max_steps, 1)], dtype=np.float32
        )

        return np.concatenate([node_features, position_onehot, progress])

    @property
    def attacker_obs_size(self) -> int:
        n = self.network.num_nodes
        return n * self.ATTACKER_NODE_FEATURES + n + 1

    def get_defender_obs(self) -> np.ndarray:
        """Build the defender's observation vector.

        Per node (4 features): isolated, is_monitored, monitor_remaining_norm,
                                compromise_estimate
        Global: progress (step/max_steps)
        """
        n = self.network.num_nodes
        estimates = self._compute_compromise_estimates()
        node_features = np.zeros(n * self.DEFENDER_NODE_FEATURES, dtype=np.float32)

        for i, node_id in enumerate(self.network.node_ids):
            state = self.network.get_state(node_id)
            base = i * self.DEFENDER_NODE_FEATURES
            node_features[base + 0] = float(state.isolated)
            node_features[base + 1] = float(state.is_monitored)
            node_features[base + 2] = state.monitor_remaining / max(
                self.monitor_duration, 1
            )
            node_features[base + 3] = estimates[node_id]

        progress = np.array(
            [self.step_count / max(self.network.max_steps, 1)], dtype=np.float32
        )

        return np.concatenate([node_features, progress])

    @property
    def defender_obs_size(self) -> int:
        n = self.network.num_nodes
        return n * self.DEFENDER_NODE_FEATURES + 1

    def _compute_compromise_estimates(self) -> dict[str, float]:
        """Noisy compromise estimates for each node.

        - Analyzed this turn: exact truth
        - Monitored: truth + action loudness + small noise
        - Unmonitored: pure noise (useless)
        """
        estimates: dict[str, float] = {}
        for node_id in self.network.node_ids:
            state = self.network.get_state(node_id)
            true_val = 1.0 if state.compromised else 0.0

            if node_id in self._analyzed_nodes:
                estimates[node_id] = true_val
            elif state.is_monitored:
                noise = float(self.rng.normal(0, 0.15))
                loudness = self._attacker_acted_on.get(node_id, 0.0)
                raw = true_val + loudness + noise
                estimates[node_id] = float(np.clip(raw, 0.0, 1.0))
            else:
                estimates[node_id] = float(self.rng.uniform(0.0, 1.0))
        return estimates

    # ------------------------------------------------------------------
    # Action masks
    # ------------------------------------------------------------------

    def get_attacker_action_mask(self) -> np.ndarray:
        """Boolean mask — True for valid attacker actions."""
        mask = np.zeros(self.attacker_actions.n, dtype=bool)
        net = self.network

        reachable = set(net.get_attacker_reachable_nodes())
        compromised = set(net.get_compromised_nodes())

        for i, action in enumerate(self.attacker_actions.actions):
            node = action.target_node
            atype = action.action_type

            if atype == AttackerActionType.SCAN:
                # Reachable, not yet scanned, not already compromised
                mask[i] = (
                    node in reachable
                    and not net.get_state(node).scanned
                    and node not in compromised
                )

            elif atype == AttackerActionType.EXPLOIT:
                # Scanned, reachable, not yet compromised
                mask[i] = (
                    node in reachable
                    and net.get_state(node).scanned
                    and node not in compromised
                )

            elif atype == AttackerActionType.ESCALATE:
                # On this node, compromised with user access (< root)
                state = net.get_state(node)
                mask[i] = (
                    node == self.attacker_position
                    and state.compromised
                    and state.access_level == 1
                )

            elif atype == AttackerActionType.MOVE:
                # Target is compromised, not isolated, not current position
                state = net.get_state(node)
                mask[i] = (
                    state.compromised
                    and not state.isolated
                    and node != self.attacker_position
                )

            elif atype == AttackerActionType.EXFILTRATE:
                # On this node, root access, has data, not isolated
                state = net.get_state(node)
                config = net.get_config(node)
                mask[i] = (
                    node == self.attacker_position
                    and state.access_level == 2
                    and config.has_data
                    and not state.isolated
                )

            elif atype == AttackerActionType.WAIT:
                mask[i] = True

        return mask

    def get_defender_action_mask(self) -> np.ndarray:
        """Boolean mask — True for valid defender actions."""
        mask = np.zeros(self.defender_actions.n, dtype=bool)
        net = self.network

        for i, action in enumerate(self.defender_actions.actions):
            node = action.target_node
            atype = action.action_type

            if atype == DefenderActionType.MONITOR:
                # Can monitor any non-isolated node that isn't the entry
                config = net.get_config(node)
                state = net.get_state(node)
                mask[i] = not config.is_entry and not state.isolated

            elif atype == DefenderActionType.ISOLATE:
                # Can isolate any non-entry, non-isolated node
                config = net.get_config(node)
                state = net.get_state(node)
                mask[i] = not config.is_entry and not state.isolated

            elif atype == DefenderActionType.RESTORE:
                # Only valid on isolated nodes (defender knows these — it
                # isolated them). Checking state.compromised here would leak
                # ground-truth attacker state into the mask, bypassing the
                # detection model.
                config = net.get_config(node)
                state = net.get_state(node)
                mask[i] = not config.is_entry and state.isolated

            elif atype == DefenderActionType.ANALYZE:
                # Can analyze any non-entry node
                config = net.get_config(node)
                mask[i] = not config.is_entry

            elif atype == DefenderActionType.WAIT:
                mask[i] = True

        return mask
