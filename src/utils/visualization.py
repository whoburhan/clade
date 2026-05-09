import os
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .metrics import MetricsTracker

plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})
COLORS = {"FedAvg": "#4C72B0", "FedProx": "#DD8452", "SCAFFOLD": "#55A868",
          "IFCA": "#C44E52", "FedPer": "#8172B3", "CLADE": "#DA5025"}


def plot_convergence(results, save_path, title="Global Accuracy"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, tracker in results.items():
        rounds = [r["round"] for r in tracker.rounds]
        vals = [r["global_accuracy"] for r in tracker.rounds]
        lw = 2.5 if name == "CLADE" else 1.5
        ax.plot(rounds, vals, label=name, color=COLORS.get(name, "#333"), linewidth=lw)
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Accuracy"); ax.set_title(title)
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)


def plot_per_client_accuracy(results, save_path, title="Per-Client Accuracy"):
    all_cids = sorted(set(c for t in results.values() for c in t.client_history.keys()))
    nc, nm = len(all_cids), len(results)
    fig, ax = plt.subplots(figsize=(max(8, nc * 2), 5))
    bw = 0.8 / nm; x = np.arange(nc)
    for i, (name, tracker) in enumerate(results.items()):
        final = tracker.get_final_results()
        accs = [final["per_client"].get(c, {}).get("accuracy", 0) for c in all_cids]
        ax.bar(x + (i - nm / 2 + 0.5) * bw, accs, bw * 0.9, label=name, color=COLORS.get(name, "#333"), alpha=0.85)
    ax.set_xlabel("Client"); ax.set_ylabel("Accuracy"); ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in all_cids])
    ax.legend(); ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, 1.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path); plt.close(fig)


def generate_all_figures(results, save_dir, dataset_name=""):
    p = f"{dataset_name}_" if dataset_name else ""
    plot_convergence(results, os.path.join(save_dir, f"{p}convergence.png"), f"Global Accuracy - {dataset_name}")
    plot_per_client_accuracy(results, os.path.join(save_dir, f"{p}per_client.png"), f"Per-Client Accuracy - {dataset_name}")
