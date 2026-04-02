from pathlib import Path

import pytest

from netsim.network import Network

SCENARIO_PATH = Path(__file__).parent.parent / "configs" / "scenarios" / "small_enterprise.yaml"


@pytest.fixture
def net():
    return Network(SCENARIO_PATH)


class TestScenarioLoading:
    def test_loads_all_nodes(self, net: Network):
        assert net.num_nodes == 5
        assert set(net.node_ids) == {
            "internet", "router", "workstation_1", "workstation_2", "server"
        }

    def test_node_order_is_stable(self, net: Network):
        """Node order must be deterministic for observation vectors."""
        net2 = Network(SCENARIO_PATH)
        assert net.node_ids == net2.node_ids

    def test_entry_node(self, net: Network):
        assert net.entry_node == "internet"
        config = net.get_config("internet")
        assert config.is_entry is True
        assert config.node_type == "entry"

    def test_data_nodes(self, net: Network):
        assert net.data_nodes == ["server"]
        config = net.get_config("server")
        assert config.has_data is True
        assert config.value == 1.0

    def test_services_and_vulns(self, net: Network):
        router_cfg = net.get_config("router")
        assert "ssh" in router_cfg.services
        assert "http" in router_cfg.services
        assert len(router_cfg.vulnerabilities) == 2
        assert router_cfg.vulnerabilities[0].name == "ssh_bruteforce"
        assert router_cfg.vulnerabilities[0].service == "ssh"
        assert router_cfg.vulnerabilities[0].prob_success == 0.7

    def test_scenario_metadata(self, net: Network):
        assert net.name == "small_enterprise"
        assert net.max_steps == 50


class TestTopology:
    def test_adjacency(self, net: Network):
        assert net.is_adjacent("internet", "router")
        assert net.is_adjacent("router", "internet")
        assert not net.is_adjacent("internet", "server")
        assert not net.is_adjacent("workstation_1", "workstation_2")

    def test_neighbors(self, net: Network):
        assert set(net.get_neighbors("router")) == {
            "internet", "workstation_1", "workstation_2", "server"
        }
        assert net.get_neighbors("internet") == ["router"]

    def test_reachable_excludes_isolated(self, net: Network):
        reachable = net.get_reachable_neighbors("router")
        assert "workstation_1" in reachable

        net.set_isolated("workstation_1", True)
        reachable = net.get_reachable_neighbors("router")
        assert "workstation_1" not in reachable

    def test_isolated_node_has_no_reachable(self, net: Network):
        net.set_isolated("router", True)
        assert net.get_reachable_neighbors("router") == []


class TestNodeState:
    def test_initial_state_entry_node(self, net: Network):
        state = net.get_state("internet")
        assert state.compromised is True
        assert state.access_level == 2
        assert state.scanned is True

    def test_initial_state_normal_node(self, net: Network):
        state = net.get_state("server")
        assert state.compromised is False
        assert state.access_level == 0
        assert state.scanned is False
        assert state.isolated is False
        assert state.is_monitored is False

    def test_set_compromised(self, net: Network):
        net.set_compromised("router", 1)
        state = net.get_state("router")
        assert state.compromised is True
        assert state.access_level == 1

        # Escalation should only increase, not decrease
        net.set_compromised("router", 2)
        assert state.access_level == 2
        net.set_compromised("router", 1)
        assert state.access_level == 2

    def test_set_scanned(self, net: Network):
        net.set_scanned("router")
        assert net.get_state("router").scanned is True

    def test_isolation(self, net: Network):
        net.set_isolated("workstation_1", True)
        assert net.get_state("workstation_1").isolated is True
        net.set_isolated("workstation_1", False)
        assert net.get_state("workstation_1").isolated is False


class TestMonitoring:
    def test_monitor_and_decay(self, net: Network):
        net.set_monitored("server", 3)
        assert net.get_state("server").is_monitored is True
        assert net.get_state("server").monitor_remaining == 3

        net.tick_monitoring()
        assert net.get_state("server").monitor_remaining == 2

        net.tick_monitoring()
        assert net.get_state("server").monitor_remaining == 1

        net.tick_monitoring()
        assert net.get_state("server").monitor_remaining == 0
        assert net.get_state("server").is_monitored is False

    def test_tick_does_not_go_negative(self, net: Network):
        net.tick_monitoring()
        assert net.get_state("server").monitor_remaining == 0


class TestReset:
    def test_reset_clears_mutable_state(self, net: Network):
        net.set_compromised("router", 2)
        net.set_scanned("router")
        net.set_isolated("workstation_1", True)
        net.set_monitored("server", 3)

        net.reset()

        assert net.get_state("router").compromised is False
        assert net.get_state("router").scanned is False
        assert net.get_state("workstation_1").isolated is False
        assert net.get_state("server").is_monitored is False

    def test_reset_preserves_entry_node(self, net: Network):
        net.reset()
        state = net.get_state("internet")
        assert state.compromised is True
        assert state.access_level == 2


class TestDerivedQueries:
    def test_compromised_nodes(self, net: Network):
        # Only entry node starts compromised
        assert net.get_compromised_nodes() == ["internet"]

        net.set_compromised("router", 1)
        assert set(net.get_compromised_nodes()) == {"internet", "router"}

    def test_attacker_reachable(self, net: Network):
        # From internet (compromised), can reach router
        reachable = net.get_attacker_reachable_nodes()
        assert reachable == ["router"]

        # Compromise router, now can reach ws1, ws2, server
        net.set_compromised("router", 1)
        reachable = set(net.get_attacker_reachable_nodes())
        assert reachable == {"workstation_1", "workstation_2", "server"}

    def test_attacker_reachable_respects_isolation(self, net: Network):
        net.set_compromised("router", 1)
        net.set_isolated("server", True)
        reachable = set(net.get_attacker_reachable_nodes())
        assert "server" not in reachable

    def test_all_compromised_isolated(self, net: Network):
        # Only entry is compromised, and we ignore entry
        assert net.all_compromised_isolated() is True

        net.set_compromised("router", 1)
        assert net.all_compromised_isolated() is False

        net.set_isolated("router", True)
        assert net.all_compromised_isolated() is True
