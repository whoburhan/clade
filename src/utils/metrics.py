import json, os
from typing import Dict, List
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.rounds, self.client_history, self.extra = [], {}, {}

    def log_client(self, round_num, client_id, accuracy, loss, num_samples):
        self.client_history.setdefault(client_id, []).append(
            {"round": round_num, "accuracy": accuracy, "loss": loss, "num_samples": num_samples})

    def log_round(self, round_num, global_accuracy, worst_accuracy):
        self.rounds.append({"round": round_num, "global_accuracy": global_accuracy, "worst_accuracy": worst_accuracy})

    def get_final_results(self):
        if not self.rounds:
            return {}
        return {"global_accuracy": self.rounds[-1]["global_accuracy"],
                "worst_accuracy": self.rounds[-1]["worst_accuracy"],
                "per_client": {cid: h[-1] for cid, h in self.client_history.items() if h},
                "extra": self.extra}

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"rounds": self.rounds, "client_history": {str(k): v for k, v in self.client_history.items()},
                       "extra": self.extra}, f, indent=2)

    @staticmethod
    def load(filepath):
        with open(filepath) as f:
            data = json.load(f)
        t = MetricsTracker()
        t.rounds = data["rounds"]
        t.client_history = {int(k): v for k, v in data["client_history"].items()}
        t.extra = data.get("extra", {})
        return t


def print_comparison_table(results):
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)
    all_cids = set()
    for t in results.values():
        all_cids.update(t.client_history.keys())
    cids = sorted(all_cids)
    header = f"{'Method':<15} | {'Global':>8} | {'Worst':>8}"
    for c in cids:
        header += f" | {'C'+str(c):>6}"
    print(header)
    print("-" * len(header))
    for name, tracker in results.items():
        final = tracker.get_final_results()
        row = f"{name:<15} | {final['global_accuracy']:>8.4f} | {final['worst_accuracy']:>8.4f}"
        for c in cids:
            if c in final["per_client"]:
                row += f" | {final['per_client'][c]['accuracy']:>6.4f}"
            else:
                row += f" | {'N/A':>6}"
        print(row)
    print("=" * 80)
