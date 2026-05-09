import numpy as np
from collections import defaultdict
from typing import Dict, List


def pathological_partition(labels, cluster_class_map, clients_per_cluster=2, seed=42):
    rng = np.random.RandomState(seed)
    client_indices = {}
    client_id = 0

    class_to_clusters = defaultdict(list)
    for cluster_name, class_ids in cluster_class_map.items():
        for c in class_ids:
            class_to_clusters[c].append(cluster_name)

    class_cluster_indices = {}
    for class_id, clusters in class_to_clusters.items():
        all_idx = np.where(labels == class_id)[0]
        rng.shuffle(all_idx)
        if len(clusters) == 1:
            class_cluster_indices[(class_id, clusters[0])] = all_idx
        else:
            n = len(all_idx)
            split_size = n // len(clusters)
            for i, cn in enumerate(clusters):
                start = i * split_size
                end = start + split_size if i < len(clusters) - 1 else n
                class_cluster_indices[(class_id, cn)] = all_idx[start:end]

    for cluster_name, class_ids in cluster_class_map.items():
        pool = np.concatenate([class_cluster_indices[(c, cluster_name)] for c in class_ids])
        rng.shuffle(pool)
        for split in np.array_split(pool, clients_per_cluster):
            client_indices[client_id] = split
            client_id += 1
    return client_indices


def dirichlet_partition(labels, num_clients, alpha=0.5, seed=42, min_samples=10):
    rng = np.random.RandomState(seed)
    num_classes = len(np.unique(labels))
    client_indices = defaultdict(list)

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()
        splits = (proportions * len(idx)).astype(int)
        remainder = len(idx) - splits.sum()
        for i in range(remainder):
            splits[i % num_clients] += 1
        current = 0
        for k in range(num_clients):
            client_indices[k].extend(idx[current:current + splits[k]])
            current += splits[k]

    result = {}
    for k in range(num_clients):
        arr = np.array(client_indices[k])
        rng.shuffle(arr)
        result[k] = arr
    return result


def print_partition_summary(labels, client_indices, class_names=None):
    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    print("\n" + "=" * 60)
    print("DATA PARTITION SUMMARY")
    print("=" * 60)
    for cid in sorted(client_indices.keys()):
        cl = labels[client_indices[cid]]
        counts = np.bincount(cl, minlength=num_classes)
        present = int(np.sum(counts > 0))
        print(f"\nClient {cid}: {len(client_indices[cid])} samples, {present}/{num_classes} classes")
        for c, count in enumerate(counts):
            if count > 0:
                print(f"  {class_names[c]}: {count}")
    print("=" * 60)
