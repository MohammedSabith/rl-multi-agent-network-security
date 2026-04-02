from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vulnerability:
    """An exploitable vulnerability on a network node."""

    name: str
    service: str
    prob_success: float


@dataclass(frozen=True)
class NodeConfig:
    """Immutable scenario configuration for a node. Loaded from YAML."""

    node_id: str
    node_type: str  # entry, router, workstation, server
    services: list[str]
    vulnerabilities: list[Vulnerability]
    privesc_prob: float
    value: float
    has_data: bool
    is_entry: bool = False


@dataclass
class NodeState:
    """Mutable state of a node during an episode. Reset between episodes."""

    compromised: bool = False
    access_level: int = 0  # 0=none, 1=user, 2=root
    scanned: bool = False
    isolated: bool = False
    monitor_remaining: int = 0

    @property
    def is_monitored(self) -> bool:
        return self.monitor_remaining > 0

    def reset(self, is_entry: bool = False) -> None:
        """Reset to start-of-episode state."""
        if is_entry:
            self.compromised = True
            self.access_level = 2  # attacker owns the entry node
            self.scanned = True
        else:
            self.compromised = False
            self.access_level = 0
            self.scanned = False
        self.isolated = False
        self.monitor_remaining = 0
