"""Action definitions, encoding/decoding, MITRE ATT&CK labels, and loudness."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from netsim.network import Network


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class AttackerActionType(Enum):
    SCAN = auto()
    EXPLOIT = auto()
    ESCALATE = auto()
    MOVE = auto()
    EXFILTRATE = auto()
    WAIT = auto()


class DefenderActionType(Enum):
    MONITOR = auto()
    ISOLATE = auto()
    RESTORE = auto()
    ANALYZE = auto()
    WAIT = auto()


# ---------------------------------------------------------------------------
# Structured actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackerAction:
    action_type: AttackerActionType
    target_node: str | None = None
    vulnerability: str | None = None  # only for EXPLOIT


@dataclass(frozen=True)
class DefenderAction:
    action_type: DefenderActionType
    target_node: str | None = None


# ---------------------------------------------------------------------------
# MITRE ATT&CK labels (metadata for evaluation, not gameplay)
# ---------------------------------------------------------------------------

ATTACKER_MITRE: dict[AttackerActionType, dict[str, str] | None] = {
    AttackerActionType.SCAN: {
        "id": "T1046",
        "name": "Network Service Discovery",
        "tactic": "Discovery",
    },
    AttackerActionType.EXPLOIT: {
        "id": "T1210",
        "name": "Exploitation of Remote Services",
        "tactic": "Lateral Movement",
    },
    AttackerActionType.ESCALATE: {
        "id": "T1068",
        "name": "Exploitation for Privilege Escalation",
        "tactic": "Privilege Escalation",
    },
    AttackerActionType.MOVE: {
        "id": "T1021",
        "name": "Remote Services",
        "tactic": "Lateral Movement",
    },
    AttackerActionType.EXFILTRATE: {
        "id": "T1041",
        "name": "Exfiltration Over C2 Channel",
        "tactic": "Exfiltration",
    },
    AttackerActionType.WAIT: None,
}

# ---------------------------------------------------------------------------
# Action loudness (for detection model)
# ---------------------------------------------------------------------------

ACTION_LOUDNESS: dict[AttackerActionType, float] = {
    AttackerActionType.SCAN: 0.1,
    AttackerActionType.EXPLOIT: 0.4,
    AttackerActionType.ESCALATE: 0.3,
    AttackerActionType.MOVE: 0.05,
    AttackerActionType.EXFILTRATE: 0.7,
    AttackerActionType.WAIT: 0.0,
}


# ---------------------------------------------------------------------------
# Action space builders — map flat Discrete index <-> structured action
# ---------------------------------------------------------------------------

class AttackerActionSpace:
    """Builds and encodes the attacker's flat Discrete action space.

    Layout: [scan(n0)..scan(nN), exploit(n0,v0)..exploit(nN,vM),
             escalate(n0)..escalate(nN), move(n0)..move(nN),
             exfiltrate(n0)..exfiltrate(nN), wait]
    """

    def __init__(self, network: Network) -> None:
        self._actions: list[AttackerAction] = []
        for node_id in network.node_ids:
            self._actions.append(
                AttackerAction(AttackerActionType.SCAN, node_id)
            )
        for node_id in network.node_ids:
            for vuln in network.get_config(node_id).vulnerabilities:
                self._actions.append(
                    AttackerAction(AttackerActionType.EXPLOIT, node_id, vuln.name)
                )
        for node_id in network.node_ids:
            self._actions.append(
                AttackerAction(AttackerActionType.ESCALATE, node_id)
            )
        for node_id in network.node_ids:
            self._actions.append(
                AttackerAction(AttackerActionType.MOVE, node_id)
            )
        for node_id in network.node_ids:
            self._actions.append(
                AttackerAction(AttackerActionType.EXFILTRATE, node_id)
            )
        self._actions.append(AttackerAction(AttackerActionType.WAIT))

    @property
    def n(self) -> int:
        return len(self._actions)

    def decode(self, index: int) -> AttackerAction:
        return self._actions[index]

    def encode(self, action: AttackerAction) -> int:
        return self._actions.index(action)

    @property
    def actions(self) -> list[AttackerAction]:
        return list(self._actions)


class DefenderActionSpace:
    """Builds and encodes the defender's flat Discrete action space.

    Layout: [monitor(n0)..monitor(nN), isolate(n0)..isolate(nN),
             restore(n0)..restore(nN), analyze(n0)..analyze(nN), wait]
    """

    def __init__(self, network: Network) -> None:
        self._actions: list[DefenderAction] = []
        for node_id in network.node_ids:
            self._actions.append(
                DefenderAction(DefenderActionType.MONITOR, node_id)
            )
        for node_id in network.node_ids:
            self._actions.append(
                DefenderAction(DefenderActionType.ISOLATE, node_id)
            )
        for node_id in network.node_ids:
            self._actions.append(
                DefenderAction(DefenderActionType.RESTORE, node_id)
            )
        for node_id in network.node_ids:
            self._actions.append(
                DefenderAction(DefenderActionType.ANALYZE, node_id)
            )
        self._actions.append(DefenderAction(DefenderActionType.WAIT))

    @property
    def n(self) -> int:
        return len(self._actions)

    def decode(self, index: int) -> DefenderAction:
        return self._actions[index]

    def encode(self, action: DefenderAction) -> int:
        return self._actions.index(action)

    @property
    def actions(self) -> list[DefenderAction]:
        return list(self._actions)
