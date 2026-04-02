"""Scripted baseline policies for attacker and defender.

These serve as:
1. Sanity checks — scripted attacker should beat random defender
2. Training baselines — RL must beat scripted to be meaningful
3. Opponent anchors — evaluate RL against scripted to detect co-evolution collapse
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from netsim.environment.actions import (
    AttackerAction,
    AttackerActionSpace,
    AttackerActionType,
    DefenderAction,
    DefenderActionSpace,
    DefenderActionType,
)
from netsim.network import Network


class ScriptedAttacker:
    """Greedy shortest-path attacker.

    Strategy: find the shortest path from current position to a data node,
    then execute scan → exploit → escalate → move along that path. Pick the
    highest prob_success vulnerability when exploiting.

    Falls back to: retry failed exploit, then wait if truly stuck.
    """

    def __init__(self, network: Network, action_space: AttackerActionSpace) -> None:
        self.network = network
        self.action_space = action_space

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> int:
        net = self.network

        # Build valid action lookup
        valid = {
            i: self.action_space.decode(i)
            for i in range(self.action_space.n)
            if mask[i]
        }
        by_type: dict[AttackerActionType, list[tuple[int, AttackerAction]]] = {}
        for idx, action in valid.items():
            by_type.setdefault(action.action_type, []).append((idx, action))

        # Priority 1: Exfiltrate if possible
        if AttackerActionType.EXFILTRATE in by_type:
            return by_type[AttackerActionType.EXFILTRATE][0][0]

        # Priority 2: Escalate if on a compromised node with user access
        if AttackerActionType.ESCALATE in by_type:
            return by_type[AttackerActionType.ESCALATE][0][0]

        # Priority 3: Move toward the data node if not already on the path
        # Find the target data node and shortest path
        target = self._pick_target(net)
        if target is not None:
            path = self._shortest_path_to(net, target)
            if path is not None:
                # Find the next node on the path we need to reach
                next_node = self._next_target_on_path(net, path)
                if next_node is not None:
                    # Can we move to it? (already compromised)
                    if AttackerActionType.MOVE in by_type:
                        for idx, action in by_type[AttackerActionType.MOVE]:
                            if action.target_node == next_node:
                                return idx

                    # Can we exploit it? (scanned, reachable, not compromised)
                    if AttackerActionType.EXPLOIT in by_type:
                        # Pick highest prob vulnerability for this node
                        exploits = [
                            (idx, a) for idx, a in by_type[AttackerActionType.EXPLOIT]
                            if a.target_node == next_node
                        ]
                        if exploits:
                            best = max(
                                exploits,
                                key=lambda x: self._vuln_prob(net, x[1]),
                            )
                            return best[0]

                    # Can we scan it?
                    if AttackerActionType.SCAN in by_type:
                        for idx, action in by_type[AttackerActionType.SCAN]:
                            if action.target_node == next_node:
                                return idx

        # Fallback: exploit any available target (best prob)
        if AttackerActionType.EXPLOIT in by_type:
            best = max(
                by_type[AttackerActionType.EXPLOIT],
                key=lambda x: self._vuln_prob(net, x[1]),
            )
            return best[0]

        # Fallback: scan any available target
        if AttackerActionType.SCAN in by_type:
            return by_type[AttackerActionType.SCAN][0][0]

        # Fallback: move anywhere
        if AttackerActionType.MOVE in by_type:
            return by_type[AttackerActionType.MOVE][0][0]

        # Last resort: wait
        return self.action_space.encode(AttackerAction(AttackerActionType.WAIT))

    def _pick_target(self, net: Network) -> str | None:
        """Pick the data node with the shortest feasible path."""
        data_nodes = net.data_nodes
        if not data_nodes:
            return None
        best_target = None
        best_len = float("inf")
        for dn in data_nodes:
            path = self._shortest_path_to(net, dn)
            if path is not None and len(path) < best_len:
                best_target = dn
                best_len = len(path)
        return best_target if best_target else data_nodes[0]

    def _shortest_path_to(self, net: Network, target: str) -> list[str] | None:
        """Shortest feasible path, avoiding isolated nodes."""
        active = [n for n in net.node_ids if not net.get_state(n).isolated]
        subgraph = net._graph.subgraph(active)
        try:
            return nx.shortest_path(subgraph, net.entry_node, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _next_target_on_path(self, net: Network, path: list[str]) -> str | None:
        """Find the next node on the path that needs attention.

        Walk the path and find the first node that is either:
        - Not compromised (need to scan/exploit it)
        - Compromised with user access and we're on it (need to escalate — handled above)
        - Compromised but we need to move through it
        """
        for node_id in path:
            state = net.get_state(node_id)
            if not state.compromised:
                return node_id
            # If compromised with user access and it's the data node, we need
            # to be there to escalate (handled by escalate priority above)
        # All nodes on path are compromised — move to the end
        # (the data node for escalation/exfiltration)
        data_node = path[-1]
        return data_node

    def _vuln_prob(self, net: Network, action: AttackerAction) -> float:
        """Get the success probability for an exploit action."""
        config = net.get_config(action.target_node)
        for vuln in config.vulnerabilities:
            if vuln.name == action.vulnerability:
                return vuln.prob_success
        return 0.0


class ScriptedDefender:
    """Monitor-and-respond defender.

    Strategy:
    1. Keep the data node (server) and its neighbors monitored
    2. If a monitored node shows high compromise estimate, isolate it
    3. Restore isolated nodes after a cooldown
    4. Fall back to analyzing suspicious unmonitored nodes

    Uses the observation vector directly (no cheating with ground truth).
    """

    ISOLATE_THRESHOLD = 0.65   # compromise estimate above this → isolate
    ANALYZE_THRESHOLD = 0.55   # estimate above this but below isolate → analyze
    RESTORE_COOLDOWN = 3       # wait this many turns after isolating before restoring

    def __init__(
        self,
        network: Network,
        action_space: DefenderActionSpace,
        monitor_duration: int = 3,
    ) -> None:
        self.network = network
        self.action_space = action_space
        self.monitor_duration = monitor_duration
        self._node_ids = network.node_ids
        # Priority monitoring targets: data nodes + neighbors + path nodes
        data_nodes = network.data_nodes
        priority = set(data_nodes)
        for dn in data_nodes:
            priority.update(network.get_neighbors(dn))
        # Add all nodes on shortest paths from entry to data
        for dn in data_nodes:
            try:
                for path in nx.all_shortest_paths(network._graph, network.entry_node, dn):
                    priority.update(path)
            except nx.NetworkXNoPath:
                pass
        priority.discard(network.entry_node)
        self._monitor_priority = list(priority)
        # Track how long each node has been isolated (for restore cooldown)
        self._isolation_age: dict[str, int] = {}
        self._monitor_idx: int = 0  # round-robin index for monitoring rotation

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> int:
        # Parse observation: per-node features are [isolated, is_monitored,
        # monitor_remaining_norm, compromise_estimate] × num_nodes, then progress
        n = len(self._node_ids)
        node_data = {}
        for i, node_id in enumerate(self._node_ids):
            base = i * 4
            node_data[node_id] = {
                "isolated": obs[base + 0],
                "is_monitored": obs[base + 1],
                "monitor_remaining": obs[base + 2],
                "estimate": obs[base + 3],
            }

        # Track isolation age: increment for isolated nodes, remove for non-isolated
        for node_id in self._node_ids:
            if node_data[node_id]["isolated"] > 0.5:
                self._isolation_age[node_id] = self._isolation_age.get(node_id, 0) + 1
            else:
                self._isolation_age.pop(node_id, None)

        valid = {
            i: self.action_space.decode(i)
            for i in range(self.action_space.n)
            if mask[i]
        }
        by_type: dict[DefenderActionType, list[tuple[int, DefenderAction]]] = {}
        for idx, action in valid.items():
            by_type.setdefault(action.action_type, []).append((idx, action))

        # Priority 1: Isolate any monitored node with high estimate
        if DefenderActionType.ISOLATE in by_type:
            for idx, action in by_type[DefenderActionType.ISOLATE]:
                nd = node_data[action.target_node]
                if nd["is_monitored"] > 0.5 and nd["estimate"] > self.ISOLATE_THRESHOLD:
                    return idx

        # Priority 2: Monitor expired priority nodes (round-robin)
        if DefenderActionType.MONITOR in by_type:
            unmonitored = [
                (idx, action) for idx, action in by_type[DefenderActionType.MONITOR]
                if action.target_node in self._monitor_priority
                and node_data[action.target_node]["is_monitored"] < 0.5
                and node_data[action.target_node]["isolated"] < 0.5
            ]
            if unmonitored:
                pick = unmonitored[self._monitor_idx % len(unmonitored)]
                self._monitor_idx += 1
                return pick[0]

        # Priority 3: Restore isolated nodes AFTER cooldown
        if DefenderActionType.RESTORE in by_type:
            for idx, action in by_type[DefenderActionType.RESTORE]:
                age = self._isolation_age.get(action.target_node, 0)
                if age >= self.RESTORE_COOLDOWN:
                    return idx

        # Priority 4: Analyze suspicious unmonitored nodes
        if DefenderActionType.ANALYZE in by_type:
            suspicious = [
                (idx, action) for idx, action in by_type[DefenderActionType.ANALYZE]
                if node_data[action.target_node]["is_monitored"] < 0.5
                and node_data[action.target_node]["estimate"] > self.ANALYZE_THRESHOLD
                and node_data[action.target_node]["isolated"] < 0.5
            ]
            if suspicious:
                suspicious.sort(key=lambda x: node_data[x[1].target_node]["estimate"], reverse=True)
                return suspicious[0][0]

        # Priority 5: Monitor any unmonitored non-entry node
        if DefenderActionType.MONITOR in by_type:
            for idx, action in by_type[DefenderActionType.MONITOR]:
                nd = node_data[action.target_node]
                if nd["is_monitored"] < 0.5:
                    return idx

        # Fallback: wait
        return self.action_space.encode(DefenderAction(DefenderActionType.WAIT))
