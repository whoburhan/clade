import copy, time
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.resnet import SplitModel, create_model
from ..client.clients import StandardClient, ScaffoldClient, FedCPAClient, FedPerClient
from ..clustering.clustering import (flatten_state_dict, compute_update_vector, compute_similarity_matrix,
                                      cluster_clients, adaptive_threshold, DriftDetector)
from ..utils.metrics import MetricsTracker


def weighted_average(state_dicts, weights):
    total = sum(weights)
    return {k: sum(sd[k].float() * (w / total) for sd, w in zip(state_dicts, weights)) for k in state_dicts[0]}


class BaseServer:
    def __init__(self, model, train_loaders, test_loaders, device, num_rounds=50, lr=0.001, local_epochs=5):
        self.global_model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.device = device
        self.num_rounds = num_rounds
        self.lr = lr
        self.local_epochs = local_epochs
        self.client_ids = sorted(train_loaders.keys())
        self.metrics = MetricsTracker()

    def _evaluate_all_clients(self, clients, round_num):
        results = {}
        for cid, client in clients.items():
            acc, loss, n = client.evaluate()
            results[cid] = {"accuracy": acc, "loss": loss, "num_samples": n}
            self.metrics.log_client(round_num, cid, acc, loss, n)
        total_samples = sum(r["num_samples"] for r in results.values())
        global_acc = sum(r["accuracy"] * r["num_samples"] for r in results.values()) / total_samples if total_samples > 0 else 0.0
        worst_acc = min(r["accuracy"] for r in results.values())
        self.metrics.log_round(round_num, global_acc, worst_acc)
        return results


class FedAvgServer(BaseServer):
    def run(self):
        print(f"\n{'='*60}\nRUNNING: FedAvg\n{'='*60}")
        clients = {cid: StandardClient(cid, self.global_model, self.train_loaders[cid], self.test_loaders[cid],
                                       self.device, self.lr, self.local_epochs, mu=0.0) for cid in self.client_ids}
        gp = copy.deepcopy(self.global_model.state_dict())
        for t in range(1, self.num_rounds + 1):
            start = time.time()
            cp = {cid: clients[cid].train(gp) for cid in self.client_ids}
            gp = weighted_average([cp[c] for c in self.client_ids], [clients[c].num_samples for c in self.client_ids])
            self.global_model.load_state_dict(gp)
            self._evaluate_all_clients(clients, t)
            r = self.metrics.rounds[-1]
            print(f"Round {t:3d}/{self.num_rounds} | Global: {r['global_accuracy']:.4f} | Worst: {r['worst_accuracy']:.4f} | {time.time()-start:.1f}s")
        return self.metrics


class FedProxServer(BaseServer):
    def __init__(self, *args, mu=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu

    def run(self):
        print(f"\n{'='*60}\nRUNNING: FedProx (mu={self.mu})\n{'='*60}")
        clients = {cid: StandardClient(cid, self.global_model, self.train_loaders[cid], self.test_loaders[cid],
                                       self.device, self.lr, self.local_epochs, mu=self.mu) for cid in self.client_ids}
        gp = copy.deepcopy(self.global_model.state_dict())
        for t in range(1, self.num_rounds + 1):
            start = time.time()
            cp = {cid: clients[cid].train(gp) for cid in self.client_ids}
            gp = weighted_average([cp[c] for c in self.client_ids], [clients[c].num_samples for c in self.client_ids])
            self.global_model.load_state_dict(gp)
            self._evaluate_all_clients(clients, t)
            r = self.metrics.rounds[-1]
            print(f"Round {t:3d}/{self.num_rounds} | Global: {r['global_accuracy']:.4f} | Worst: {r['worst_accuracy']:.4f} | {time.time()-start:.1f}s")
        return self.metrics


class ScaffoldServer(BaseServer):
    def __init__(self, *args, global_lr=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_lr = global_lr

    def run(self):
        print(f"\n{'='*60}\nRUNNING: SCAFFOLD\n{'='*60}")
        clients = {cid: ScaffoldClient(cid, self.global_model, self.train_loaders[cid], self.test_loaders[cid],
                                       self.device, self.lr, self.local_epochs) for cid in self.client_ids}
        gp = copy.deepcopy(self.global_model.state_dict())
        c_global = {n: torch.zeros_like(p) for n, p in self.global_model.named_parameters()}
        K = len(self.client_ids)
        for t in range(1, self.num_rounds + 1):
            start = time.time()
            cp, cd = {}, {}
            for cid in self.client_ids:
                cp[cid], cd[cid] = clients[cid].train(gp, c_global)
            gp = weighted_average([cp[c] for c in self.client_ids], [clients[c].num_samples for c in self.client_ids])
            for n in c_global:
                c_global[n] += sum(cd[c][n] for c in self.client_ids) / K
            self.global_model.load_state_dict(gp)
            self._evaluate_all_clients(clients, t)
            r = self.metrics.rounds[-1]
            print(f"Round {t:3d}/{self.num_rounds} | Global: {r['global_accuracy']:.4f} | Worst: {r['worst_accuracy']:.4f} | {time.time()-start:.1f}s")
        return self.metrics


class IFCAServer(BaseServer):
    def __init__(self, *args, num_clusters=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters

    def run(self):
        print(f"\n{'='*60}\nRUNNING: IFCA (K={self.num_clusters})\n{'='*60}")
        cluster_models = {c: copy.deepcopy(self.global_model.state_dict()) for c in range(self.num_clusters)}
        clients = {cid: StandardClient(cid, self.global_model, self.train_loaders[cid], self.test_loaders[cid],
                                       self.device, self.lr, self.local_epochs) for cid in self.client_ids}
        for t in range(1, self.num_rounds + 1):
            start = time.time()
            cc = {}
            for cid in self.client_ids:
                best_c, best_l = 0, float("inf")
                for c in range(self.num_clusters):
                    clients[cid].model.load_state_dict(cluster_models[c])
                    _, loss, _ = clients[cid].evaluate()
                    if loss < best_l:
                        best_l, best_c = loss, c
                cc[cid] = best_c
            cp = {cid: clients[cid].train(cluster_models[cc[cid]]) for cid in self.client_ids}
            for c in range(self.num_clusters):
                members = [cid for cid in self.client_ids if cc[cid] == c]
                if members:
                    cluster_models[c] = weighted_average([cp[m] for m in members], [clients[m].num_samples for m in members])
            for cid in self.client_ids:
                clients[cid].model.load_state_dict(cluster_models[cc[cid]])
            self._evaluate_all_clients(clients, t)
            r = self.metrics.rounds[-1]
            print(f"Round {t:3d}/{self.num_rounds} | Global: {r['global_accuracy']:.4f} | Worst: {r['worst_accuracy']:.4f} | {time.time()-start:.1f}s")
        return self.metrics


class FedPerServer(BaseServer):
    def __init__(self, model, train_loaders, test_loaders, device, num_rounds=50, lr=0.001,
                 head_lr=0.0005, local_epochs=5, head_finetune_epochs=2):
        super().__init__(model, train_loaders, test_loaders, device, num_rounds, lr, local_epochs)
        self.head_lr = head_lr
        self.head_finetune_epochs = head_finetune_epochs

    def run(self):
        print(f"\n{'='*60}\nRUNNING: FedPer\n{'='*60}")
        clients = {cid: FedPerClient(cid, self.global_model, self.train_loaders[cid], self.test_loaders[cid],
                                     self.device, self.lr, self.head_lr, self.local_epochs, self.head_finetune_epochs)
                   for cid in self.client_ids}
        ge = copy.deepcopy(self.global_model.get_extractor_params())
        for t in range(1, self.num_rounds + 1):
            start = time.time()
            ce = {cid: clients[cid].train(ge) for cid in self.client_ids}
            ge = weighted_average([ce[c] for c in self.client_ids], [clients[c].num_samples for c in self.client_ids])
            self._evaluate_all_clients(clients, t)
            r = self.metrics.rounds[-1]
            print(f"Round {t:3d}/{self.num_rounds} | Global: {r['global_accuracy']:.4f} | Worst: {r['worst_accuracy']:.4f} | {time.time()-start:.1f}s")
        return self.metrics


class FedCPAServer(BaseServer):
    def __init__(self, model, train_loaders, test_loaders, device, num_rounds=50, lr=0.001,
                 head_lr=0.0005, local_epochs=5, head_finetune_epochs=2, recluster_interval=5,
                 ema_decay=0.9, alpha_base=0.3, alpha_max=0.8, gamma=0.05, sensitivity_margin=0.15,
                 min_cluster_size=1):
        super().__init__(model, train_loaders, test_loaders, device, num_rounds, lr, local_epochs)
        self.head_lr, self.head_finetune_epochs = head_lr, head_finetune_epochs
        self.recluster_interval = recluster_interval
        self.alpha_base, self.alpha_max, self.gamma = alpha_base, alpha_max, gamma
        self.min_cluster_size = min_cluster_size
        self.drift_detector = DriftDetector(ema_decay, sensitivity_margin)
        self.cluster_assignments, self.cluster_extractors, self.client_to_cluster = {}, {}, {}

    def run(self):
        print(f"\n{'='*60}\nRUNNING: CLADE\n{'='*60}")
        clients = {cid: FedCPAClient(cid, self.global_model, self.train_loaders[cid], self.test_loaders[cid],
                                     self.device, self.lr, self.head_lr, self.local_epochs, self.head_finetune_epochs)
                   for cid in self.client_ids}
        self.cluster_assignments = {0: list(self.client_ids)}
        self.client_to_cluster = {cid: 0 for cid in self.client_ids}
        ge = copy.deepcopy(self.global_model.get_extractor_params())
        self.cluster_extractors = {0: copy.deepcopy(ge)}
        recluster_events = 0
        for t in range(1, self.num_rounds + 1):
            start = time.time()
            ce, cw, pe = {}, {}, {}
            for cid in self.client_ids:
                cl = self.client_to_cluster[cid]
                pe[cid] = copy.deepcopy(self.cluster_extractors[cl])
                ce[cid] = clients[cid].train(pe[cid])
                cw[cid] = clients[cid].num_samples
            uv = {cid: compute_update_vector(ce[cid], pe[cid]) for cid in self.client_ids}
            alpha_t = adaptive_threshold(t, self.alpha_base, self.alpha_max, self.gamma)
            should_recluster = t == 1
            if not should_recluster and t % self.recluster_interval == 0:
                drift, _ = self.drift_detector.update_and_check(self.cluster_assignments, uv, alpha_t)
                should_recluster = drift
            if should_recluster:
                sm, oids = compute_similarity_matrix(uv)
                nc = cluster_clients(sm, oids, alpha_t, self.min_cluster_size)
                self.cluster_assignments = nc
                self.client_to_cluster = {cid: cl for cl, members in nc.items() for cid in members}
                self.drift_detector.reset(nc)
                recluster_events += 1
                cs = " | ".join(f"C{c}: {m}" for c, m in nc.items())
                print(f"  [Re-clustered] {cs}")
            nce = {}
            for cl, members in self.cluster_assignments.items():
                nce[cl] = weighted_average([ce[c] for c in members], [cw[c] for c in members])
            self.cluster_extractors = nce
            self._evaluate_all_clients(clients, t)
            r = self.metrics.rounds[-1]
            ncl = len(self.cluster_assignments)
            print(f"Round {t:3d}/{self.num_rounds} | Global: {r['global_accuracy']:.4f} | Worst: {r['worst_accuracy']:.4f} | Clusters: {ncl} | a_t: {alpha_t:.3f} | {time.time()-start:.1f}s")
        self.metrics.extra["recluster_events"] = recluster_events
        self.metrics.extra["final_clusters"] = {str(k): v for k, v in self.cluster_assignments.items()}
        print(f"\nTotal re-clustering events: {recluster_events}")
        return self.metrics
