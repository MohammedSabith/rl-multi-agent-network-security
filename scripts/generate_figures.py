"""Generate figures that work on a fresh clone (no trained model required).

Produces:
  results/figures/network_topology.png  -- network graph with attack paths
  results/figures/round_evolution.png   -- win rates across training rounds
  results/figures/defender_comparison.png -- v2 vs v3 defender action distribution
  results/figures/results_card.png      -- headline results summary

Requires: results/self_play_v3/history.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from netsim.environment import GameEngine

SCENARIO = Path("configs/scenarios/enterprise_v2.yaml")
RESULTS = Path("results/self_play_v3")
FIGURES = Path("results/figures")

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
})

C_RED = "#f85149"
C_GREEN = "#3fb950"
C_BLUE = "#58a6ff"
C_ORANGE = "#d29922"
C_PURPLE = "#bc8cff"
C_GRAY = "#8b949e"
C_DIMGRAY = "#484f58"
C_WHITE = "#c9d1d9"


def load_history():
    with open(RESULTS / "history.json") as f:
        return json.load(f)


# ---------------------------------------------------------------
# Figure 1: Network topology with attack paths
# ---------------------------------------------------------------
def fig_network_topology():
    engine = GameEngine(SCENARIO, seed=0)
    G = engine.network._graph

    pos = {
        "internet":          (0, 0),
        "firewall":          (-2, -1.5),
        "vpn_concentrator":  (0, -1.5),
        "wifi_ap":           (2, -1.5),
        "dmz_switch":        (-3, -3),
        "web_server":        (-4, -4.5),
        "mail_server":       (-2, -4.5),
        "internal_router":   (0, -3.5),
        "workstation_1":     (-1.2, -5),
        "workstation_2":     (0, -5),
        "workstation_3":     (1.5, -3),
        "server_switch":     (-1, -6.5),
        "app_server":        (-3.5, -6),
        "db_server":         (-2.5, -8),
        "backup_server":     (0.5, -8),
        "file_server":       (-1, -8),
        "admin_workstation": (1.5, -5),
        "monitoring_server": (2.5, -6.5),
    }

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Enterprise Network Topology — RL Attacker's Winning Path",
                 fontsize=16, fontweight="bold", pad=20, color=C_WHITE)

    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=C_DIMGRAY, linewidth=1.2, zorder=1)

    scripted_path = ["internet", "firewall", "internal_router", "server_switch", "backup_server"]
    for i in range(len(scripted_path) - 1):
        u, v = scripted_path[i], scripted_path[i + 1]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=C_GRAY, linewidth=3, linestyle="--", zorder=2, alpha=0.7)

    rl_path = ["internet", "wifi_ap", "workstation_3", "internal_router",
               "workstation_1", "backup_server"]
    for i in range(len(rl_path) - 1):
        u, v = rl_path[i], rl_path[i + 1]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=C_RED, linewidth=3.5, zorder=3)

    vpn_path = ["internet", "vpn_concentrator"]
    for i in range(len(vpn_path) - 1):
        u, v = vpn_path[i], vpn_path[i + 1]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=C_ORANGE, linewidth=3, linestyle=":", zorder=3)

    bx = [pos["workstation_1"][0], pos["backup_server"][0]]
    by = [pos["workstation_1"][1], pos["backup_server"][1]]
    ax.plot(bx, by, color=C_RED, linewidth=4, zorder=4)
    ax.annotate("BYPASS\nEDGE", ((bx[0] + bx[1]) / 2 - 0.15, (by[0] + by[1]) / 2),
                fontsize=8, fontweight="bold", color=C_RED, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=C_RED, alpha=0.9))

    entry_nodes = {"internet"}
    data_nodes = {"db_server", "backup_server"}
    rl_path_nodes = set(rl_path)

    for node in G.nodes():
        x, y = pos[node]
        if node in entry_nodes:
            color, size = C_GREEN, 700
        elif node in data_nodes:
            color, size = C_ORANGE, 800
        elif node in rl_path_nodes:
            color, size = C_RED, 600
        else:
            color, size = C_BLUE, 400
        ax.scatter(x, y, s=size, c=color, zorder=5, edgecolors="#0d1117", linewidths=1.5)
        offset_y = 0.4 if node in ("server_switch", "internal_router") else 0.35
        ax.text(x, y + offset_y, node.replace("_", "\n"), fontsize=7, ha="center",
                va="bottom", color=C_WHITE,
                fontweight="bold" if node in rl_path_nodes else "normal")

    legend_items = [
        mpatches.Patch(color=C_RED, label="RL attacker path (98.5% of wins)"),
        mpatches.Patch(color=C_GRAY, label="Scripted attacker path (2.5% win rate)"),
        mpatches.Patch(color=C_ORANGE, label="VPN probe (detected, then pivot)"),
        mpatches.Patch(color=C_GREEN, label="Entry point"),
        mpatches.Patch(color=C_BLUE, label="Network node"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9,
              facecolor="#161b22", edgecolor="#30363d", labelcolor=C_WHITE)

    for label, xy, fs in [
        ("INTERNET", (0, 0.6), 9), ("DMZ", (-3, -2.3), 8),
        ("INTERNAL", (0.5, -4.2), 8), ("SERVERS", (-1.5, -7.2), 8), ("MGMT", (2, -4.2), 8),
    ]:
        ax.text(xy[0], xy[1], label, fontsize=fs, color=C_DIMGRAY,
                ha="center", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(FIGURES / "network_topology.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved network_topology.png")


# ---------------------------------------------------------------
# Figure 2: Round-by-round win rate evolution
# ---------------------------------------------------------------
def fig_round_evolution():
    history = load_history()
    rounds = [r["round"] for r in history["rounds"]]
    rl_vs_scripted = [r["rl_atk_vs_scripted_def"] * 100 for r in history["rounds"]]
    rl_vs_random = [r["rl_atk_vs_random_def"] * 100 for r in history["rounds"]]
    scripted_vs_rl = [r["scripted_atk_vs_rl_def"] * 100 for r in history["rounds"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rounds, rl_vs_scripted, "o-", color=C_RED, linewidth=2.5,
            markersize=8, label="RL attacker vs scripted defender", zorder=3)
    ax.plot(rounds, rl_vs_random, "s--", color=C_BLUE, linewidth=1.8,
            markersize=6, label="RL attacker vs random defender", alpha=0.7)
    ax.plot(rounds, scripted_vs_rl, "^--", color=C_GREEN, linewidth=1.8,
            markersize=6, label="Scripted attacker vs RL defender", alpha=0.7)

    ax.axhline(y=2.5, color=C_ORANGE, linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(10.3, 2.5, "Scripted\nbaseline\n2.5%", fontsize=8, color=C_ORANGE, va="center")

    ax.set_xlabel("Self-Play Round", fontsize=12)
    ax.set_ylabel("Attacker Win Rate (%)", fontsize=12)
    ax.set_title("Self-Play Training: Win Rate Across Rounds", fontsize=14, fontweight="bold")
    ax.set_xticks(rounds)
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper right", fontsize=9, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    ax.annotate(f"{rl_vs_scripted[-1]:.1f}%", (rounds[-1], rl_vs_scripted[-1]),
                textcoords="offset points", xytext=(0, 15),
                fontsize=11, fontweight="bold", color=C_RED, ha="center")

    fig.tight_layout()
    fig.savefig(FIGURES / "round_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved round_evolution.png")


# ---------------------------------------------------------------
# Figure 3: Defender action distribution — v2 vs v3
# ---------------------------------------------------------------
def fig_defender_comparison():
    labels = ["ISOLATE", "ANALYZE", "MONITOR", "WAIT", "RESTORE"]
    v2_vals = [80, 5, 12, 3, 0]
    v3_vals = [4.2, 69.3, 24.3, 2.2, 0.0]
    colors = [C_RED, C_BLUE, C_GREEN, C_GRAY, C_PURPLE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def make_pie(ax, vals, title):
        filtered = [(l, v, c) for l, v, c in zip(labels, vals, colors) if v > 0]
        wedges, texts = ax.pie(
            [v for _, v, _ in filtered],
            labels=[f"{l}\n{v:.1f}%" for l, v, _ in filtered],
            colors=[c for _, _, c in filtered],
            startangle=90,
            textprops={"color": C_WHITE, "fontsize": 10},
            wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    make_pie(ax1, v2_vals, "Without Isolation Cost (v2)\n\"Disconnect Everything\"")
    make_pie(ax2, v3_vals, "With Isolation Cost (v3)\n\"Protect Crown Jewels\"")

    fig.suptitle("RL Defender Action Distribution", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "defender_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved defender_comparison.png")


# ---------------------------------------------------------------
# Figure 4: Results summary card
# ---------------------------------------------------------------
def fig_results_card():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    ax.text(0.5, 0.95, "RL Attacker vs Scripted Defender", fontsize=18,
            fontweight="bold", ha="center", va="top", color=C_WHITE,
            transform=ax.transAxes)

    ax.text(0.25, 0.72, "2.5%", fontsize=48, fontweight="bold", ha="center",
            color=C_GRAY, transform=ax.transAxes)
    ax.text(0.25, 0.60, "Scripted Attacker", fontsize=12, ha="center",
            color=C_GRAY, transform=ax.transAxes)
    ax.text(0.5, 0.72, "vs", fontsize=20, ha="center",
            color=C_DIMGRAY, transform=ax.transAxes)
    ax.text(0.75, 0.72, "26.6%", fontsize=48, fontweight="bold", ha="center",
            color=C_RED, transform=ax.transAxes)
    ax.text(0.75, 0.60, "RL Attacker", fontsize=12, ha="center",
            color=C_RED, transform=ax.transAxes)
    ax.text(0.5, 0.45, "10.6x improvement", fontsize=16, ha="center",
            color=C_ORANGE, fontweight="bold", transform=ax.transAxes)

    stats = [
        ("98.5%", "wins use bypass edge"),
        ("89.5%", "wins use probe-and-pivot"),
        ("28 turns", "average winning attack chain"),
        ("5 vulns", "exploited across 6.3 nodes per win"),
    ]
    for i, (num, desc) in enumerate(stats):
        y = 0.30 - i * 0.07
        ax.text(0.3, y, num, fontsize=12, fontweight="bold", ha="right",
                color=C_BLUE, transform=ax.transAxes)
        ax.text(0.33, y, desc, fontsize=11, ha="left", color=C_GRAY,
                transform=ax.transAxes)

    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False,
                          edgecolor="#30363d", linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(FIGURES / "results_card.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved results_card.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating figures (no model required)...")
    fig_network_topology()
    fig_round_evolution()
    fig_defender_comparison()
    fig_results_card()
    print(f"\nDone. Figures saved to {FIGURES}/")
    print("To regenerate kill_chain.png and node_heatmap.png, run: python scripts/generate_figures_from_model.py")
