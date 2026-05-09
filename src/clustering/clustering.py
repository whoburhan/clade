import numpy as np
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def flatten_state_dict(state_dict):
    return np.concatenate([state_dict[k].cpu().numpy().flatten() for k in sorted(state_dict.keys())])


def compute_update_vector(current_params, previous_params):
    return flatten_state_dict(current_params) - flatten_state_dict(previous_params)


def cosine_similarity(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-10 or nv < 1e-10:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def compute_similarity_matrix(update_vectors):
    client_ids = sorted(update_vectors.keys())
    K = len(client_ids)
    sim_matrix = np.ones((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            sim = cosine_similarity(update_vectors[client_ids[i]], update_vectors[client_ids[j]])
            sim_matrix[i, j] = sim_matrix[j, i] = sim
    return sim_matrix, client_ids


def cluster_clients(sim_matrix, client_ids, threshold, min_cluster_size=1):
    K = len(client_ids)
    if K <= 1:
        return {0: list(client_ids)}
    dist_matrix = np.maximum(1.0 - sim_matrix, 0.0)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    clusters = {}
    for idx, cl in enumerate(labels):
        cid = int(cl) - 1
        clusters.setdefault(cid, []).append(client_ids[idx])
    return {i: clusters[old] for i, old in enumerate(sorted(clusters.keys()))}


def adaptive_threshold(round_num, alpha_base=0.3, alpha_max=0.8, gamma=0.05):
    return alpha_base + (alpha_max - alpha_base) * (1.0 - np.exp(-gamma * round_num))


class DriftDetector:
    def __init__(self, ema_decay=0.9, sensitivity_margin=0.15):
        self.beta = ema_decay
        self.delta = sensitivity_margin
        self.cluster_emas = {}

    def reset(self, cluster_assignments):
        self.cluster_emas = {cid: 1.0 for cid in cluster_assignments}

    def update_and_check(self, cluster_assignments, update_vectors, current_threshold):
        should_recluster = False
        cluster_similarities = {}
        for cluster_id, client_ids in cluster_assignments.items():
            if len(client_ids) < 2:
                self.cluster_emas[cluster_id] = 1.0
                cluster_similarities[cluster_id] = 1.0
                continue
            sims = []
            for i in range(len(client_ids)):
                for j in range(i + 1, len(client_ids)):
                    if client_ids[i] in update_vectors and client_ids[j] in update_vectors:
                        sims.append(cosine_similarity(update_vectors[client_ids[i]], update_vectors[client_ids[j]]))
            if not sims:
                continue
            mean_sim = float(np.mean(sims))
            prev = self.cluster_emas.get(cluster_id, 1.0)
            new_ema = self.beta * prev + (1.0 - self.beta) * mean_sim
            self.cluster_emas[cluster_id] = new_ema
            cluster_similarities[cluster_id] = new_ema
            if new_ema < current_threshold - self.delta:
                should_recluster = True
        return should_recluster, cluster_similarities
