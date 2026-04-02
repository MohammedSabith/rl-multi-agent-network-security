from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from .models import NodeConfig, NodeState, Vulnerability


class Network:
    """Network topology and mutable node state.

    Owns the graph structure (immutable after load) and per-node state
    (mutable during episodes, reset between them). Game logic lives in
    the environment, not here.
    """

    def __init__(self, scenario_path: str | Path) -> None:
        scenario_path = Path(scenario_path)
        with open(scenario_path) as f:
            raw = yaml.safe_load(f)

        self.name: str = raw["name"]
        self.description: str = raw.get("description", "")
        self.max_steps: int = raw["max_steps"]

        # Build graph and node configs
        self._graph = nx.Graph()
        self._configs: dict[str, NodeConfig] = {}
        self._states: dict[str, NodeState] = {}
        # Ordered list of node IDs for consistent observation vectors
        self._node_ids: list[str] = list(raw["nodes"].keys())

        for node_id, node_raw in raw["nodes"].items():
            vulns = [
                Vulnerability(
                    name=v["name"],
                    service=v["service"],
                    prob_success=v["prob_success"],
                )
                for v in node_raw.get("vulnerabilities", [])
            ]
            config = NodeConfig(
                node_id=node_id,
                node_type=node_raw["type"],
                services=node_raw.get("services", []),
                vulnerabilities=vulns,
                privesc_prob=node_raw.get("privesc_prob", 0.0),
                value=node_raw.get("value", 0.0),
                has_data=node_raw.get("has_data", False),
                is_entry=node_raw.get("entry", False),
            )
            self._configs[node_id] = config
            self._states[node_id] = NodeState()
            self._graph.add_node(node_id)

        for edge in raw["edges"]:
            a, b = edge[0], edge[1]
            if a not in self._configs:
                raise ValueError(f"Edge references unknown node: {a}")
            if b not in self._configs:
                raise ValueError(f"Edge references unknown node: {b}")
            self._graph.add_edge(a, b)

        self._validate()
        self.reset()

    def _validate(self) -> None:
        """Sanity checks on the loaded scenario."""
        entry_nodes = [n for n, c in self._configs.items() if c.is_entry]
        if len(entry_nodes) != 1:
            raise ValueError(
                f"Scenario must have exactly 1 entry node, found {len(entry_nodes)}: {entry_nodes}"
            )

        data_nodes = [n for n, c in self._configs.items() if c.has_data]
        if len(data_nodes) == 0:
            raise ValueError("Scenario must have at least 1 node with has_data=true")

        # Verify all vulnerability services exist on the node
        for node_id, config in self._configs.items():
            for vuln in config.vulnerabilities:
                if vuln.service not in config.services:
                    raise ValueError(
                        f"Node '{node_id}' vulnerability '{vuln.name}' references "
                        f"service '{vuln.service}' not in node services: {config.services}"
                    )

        # Verify graph is connected
        if not nx.is_connected(self._graph):
            raise ValueError("Network graph must be connected")

    def reset(self) -> None:
        """Reset all mutable node state for a new episode."""
        for node_id, state in self._states.items():
            state.reset(is_entry=self._configs[node_id].is_entry)

    # --- Topology queries ---

    @property
    def node_ids(self) -> list[str]:
        """Ordered list of node IDs (stable across episodes)."""
        return self._node_ids

    @property
    def num_nodes(self) -> int:
        return len(self._node_ids)

    @property
    def entry_node(self) -> str:
        for node_id, config in self._configs.items():
            if config.is_entry:
                return node_id
        raise RuntimeError("No entry node found")  # pragma: no cover

    @property
    def data_nodes(self) -> list[str]:
        return [n for n, c in self._configs.items() if c.has_data]

    def get_config(self, node_id: str) -> NodeConfig:
        return self._configs[node_id]

    def get_state(self, node_id: str) -> NodeState:
        return self._states[node_id]

    def get_neighbors(self, node_id: str) -> list[str]:
        """All topological neighbors (regardless of isolation)."""
        return list(self._graph.neighbors(node_id))

    def get_reachable_neighbors(self, node_id: str) -> list[str]:
        """Neighbors reachable from node_id (excludes isolated nodes and
        returns nothing if node_id itself is isolated)."""
        if self._states[node_id].isolated:
            return []
        return [
            n for n in self._graph.neighbors(node_id)
            if not self._states[n].isolated
        ]

    def is_adjacent(self, a: str, b: str) -> bool:
        return self._graph.has_edge(a, b)

    # --- State mutations (called by environment) ---

    def set_compromised(self, node_id: str, access_level: int) -> None:
        state = self._states[node_id]
        state.compromised = True
        state.access_level = max(state.access_level, access_level)

    def set_scanned(self, node_id: str) -> None:
        self._states[node_id].scanned = True

    def set_isolated(self, node_id: str, isolated: bool) -> None:
        self._states[node_id].isolated = isolated

    def set_monitored(self, node_id: str, duration: int) -> None:
        self._states[node_id].monitor_remaining = duration

    def tick_monitoring(self) -> None:
        """Decrement all monitoring counters by 1. Called once per turn."""
        for state in self._states.values():
            if state.monitor_remaining > 0:
                state.monitor_remaining -= 1

    # --- Derived queries ---

    def get_compromised_nodes(self) -> list[str]:
        return [n for n, s in self._states.items() if s.compromised]

    def get_attacker_reachable_nodes(self) -> list[str]:
        """Nodes the attacker can reach from any compromised, non-isolated node."""
        reachable = set()
        for node_id in self.get_compromised_nodes():
            reachable.update(self.get_reachable_neighbors(node_id))
        # Exclude already-compromised nodes
        return [n for n in reachable if not self._states[n].compromised]

    def all_compromised_isolated(self) -> bool:
        """True if every compromised node (except entry) is isolated."""
        for node_id, state in self._states.items():
            if self._configs[node_id].is_entry:
                continue
            if state.compromised and not state.isolated:
                return False
        return True
