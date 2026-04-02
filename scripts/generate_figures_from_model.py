"""Generate figures that require trained model checkpoints.

Produces:
  results/figures/kill_chain.png   -- MITRE ATT&CK kill chain for a winning episode
  results/figures/node_heatmap.png -- attacker exploit vs defender monitor counts

Requires: results/self_play_v3/attacker_r10.zip
Run self_play() first to generate model checkpoints:

    from netsim.training.self_play import self_play
    self_play("configs/scenarios/enterprise_v2.yaml", output_dir="results/self_play_v3")
"""

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from netsim.environment import GameEngine, ATTACKER_MITRE
from netsim.agents.scripted import ScriptedDefender
from netsim.training.self_play import make_rl_policy

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
C_WHITE = "#c9d1d9"


def _load_model():
    from sb3_contrib import MaskablePPO
    model_path = RESULTS / "attacker_r10.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run self_play() first to generate model checkpoints."
        )
    return MaskablePPO.load(str(model_path))


# ---------------------------------------------------------------
# Figure 1: MITRE ATT&CK kill chain timeline
# ---------------------------------------------------------------
def fig_kill_chain():
    atk_model = _load_model()
    atk_policy = make_rl_policy(atk_model)

    engine = GameEngine(SCENARIO, seed=2)
    defender = ScriptedDefender(engine.network, engine.defender_actions)
    trace = []

    while not engine.done:
        engine.begin_turn()
        obs = engine.get_attacker_obs()
        mask = engine.get_attacker_action_mask()
        atk_idx = atk_policy(obs, mask)
        atk_action = engine.attacker_actions.decode(atk_idx)
        info = engine.execute_attacker_action(atk_action)

        mitre = ATTACKER_MITRE.get(atk_action.action_type)
        if mitre and atk_action.action_type.name != "WAIT":
            tactic = mitre["tactic"]
            tid = mitre["id"]
            if atk_action.target_node in ("vpn_concentrator", "wifi_ap"):
                if atk_action.action_type.name == "EXPLOIT":
                    tactic = "Initial Access"
                    tid = "T1133"
            trace.append({
                "turn": engine.step_count,
                "tactic": tactic,
                "technique": tid,
                "action": atk_action.action_type.name,
                "target": atk_action.target_node or "",
                "success": info["success"],
            })

        if engine.done:
            break
        def_obs = engine.get_defender_obs()
        def_mask = engine.get_defender_action_mask()
        def_idx = defender(def_obs, def_mask)
        engine.execute_defender_action(engine.defender_actions.decode(def_idx))
        engine.end_turn()

    tactic_colors = {
        "Discovery": C_BLUE,
        "Initial Access": C_ORANGE,
        "Lateral Movement": C_RED,
        "Privilege Escalation": C_PURPLE,
        "Exfiltration": C_GREEN,
    }

    fig, ax = plt.subplots(figsize=(16, 5))

    for i, t in enumerate(trace):
        color = tactic_colors.get(t["tactic"], C_GRAY)
        alpha = 1.0 if t["success"] else 0.3
        ax.barh(0, 0.8, left=i, height=0.6, color=color, alpha=alpha,
                edgecolor="#0d1117", linewidth=0.5)
        short = (t["target"].replace("_concentrator", "").replace("_server", "_srv")
                 .replace("workstation_", "ws").replace("internal_", "int_")
                 .replace("backup", "bkup"))
        if i % 2 == 0:
            ax.text(i + 0.4, -0.55, short, fontsize=6, rotation=45, ha="right",
                    color=C_GRAY, va="top")

    for i, t in enumerate(trace):
        if not t["success"]:
            ax.text(i + 0.4, 0, "X", fontsize=8, ha="center", va="center",
                    color=C_WHITE, fontweight="bold")

    ax.set_xlim(-0.5, len(trace) + 0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Action Sequence", fontsize=12)
    ax.set_yticks([])
    ax.set_title(f"MITRE ATT&CK Kill Chain — Winning Episode ({len(trace)} actions)",
                 fontsize=14, fontweight="bold")

    legend_items = [mpatches.Patch(color=c, label=t) for t, c in tactic_colors.items()]
    legend_items.append(mpatches.Patch(facecolor=C_GRAY, alpha=0.3, label="Failed attempt"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, ncol=3,
              facecolor="#161b22", edgecolor="#30363d")

    ax.annotate("VPN probe\n(detected)", xy=(5, 0.5), fontsize=8, color=C_ORANGE,
                ha="center", fontstyle="italic")
    ax.annotate("WiFi pivot", xy=(10, 0.5), fontsize=8, color=C_RED,
                ha="center", fontstyle="italic")
    ax.annotate("Bypass\nedge", xy=(21, 0.5), fontsize=8, color=C_RED,
                ha="center", fontweight="bold")
    ax.annotate("Exfil", xy=(len(trace) - 1, 0.5), fontsize=8, color=C_GREEN,
                ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES / "kill_chain.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved kill_chain.png")


# ---------------------------------------------------------------
# Figure 2: Node exploitation heatmap
# ---------------------------------------------------------------
def fig_node_heatmap():
    atk_model = _load_model()
    atk_policy = make_rl_policy(atk_model)

    engine = GameEngine(SCENARIO, seed=0)
    exploit_counts = Counter()
    monitor_counts = Counter()

    for ep in range(200):
        engine.reset(seed=ep)
        defender = ScriptedDefender(engine.network, engine.defender_actions)

        while not engine.done:
            engine.begin_turn()
            obs = engine.get_attacker_obs()
            mask = engine.get_attacker_action_mask()
            atk_idx = atk_policy(obs, mask)
            atk_action = engine.attacker_actions.decode(atk_idx)
            info = engine.execute_attacker_action(atk_action)

            if atk_action.action_type.name == "EXPLOIT" and atk_action.target_node:
                exploit_counts[atk_action.target_node] += 1

            if engine.done:
                break

            def_obs = engine.get_defender_obs()
            def_mask = engine.get_defender_action_mask()
            def_idx = defender(def_obs, def_mask)
            def_action = engine.defender_actions.decode(def_idx)
            if def_action.action_type.name == "MONITOR" and def_action.target_node:
                monitor_counts[def_action.target_node] += 1

            engine.execute_defender_action(def_action)
            engine.end_turn()

    nodes = [n for n in engine.network.node_ids if n != "internet"]
    exploit_vals = [exploit_counts.get(n, 0) for n in nodes]
    monitor_vals = [monitor_counts.get(n, 0) for n in nodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bars1 = ax1.barh(range(len(nodes)), exploit_vals, color=C_RED, alpha=0.8,
                     edgecolor="#0d1117")
    ax1.set_yticks(range(len(nodes)))
    ax1.set_yticklabels([n.replace("_", " ") for n in nodes], fontsize=9)
    ax1.set_title("RL Attacker: Exploit Attempts by Node (200 episodes)",
                  fontsize=13, fontweight="bold")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, exploit_vals):
        if val > 0:
            ax1.text(val + 5, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=8, color=C_RED)

    bars2 = ax2.barh(range(len(nodes)), monitor_vals, color=C_GREEN, alpha=0.8,
                     edgecolor="#0d1117")
    ax2.set_yticks(range(len(nodes)))
    ax2.set_yticklabels([n.replace("_", " ") for n in nodes], fontsize=9)
    ax2.set_title("Scripted Defender: Monitor Actions by Node (200 episodes)",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Action Count", fontsize=11)
    ax2.invert_yaxis()
    for bar, val in zip(bars2, monitor_vals):
        if val > 0:
            ax2.text(val + 5, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=8, color=C_GREEN)

    fig.suptitle("Where the Attacker Attacks vs Where the Defender Watches",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "node_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved node_heatmap.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating figures from trained model...")
    fig_kill_chain()
    fig_node_heatmap()
    print(f"\nDone. Figures saved to {FIGURES}/")
