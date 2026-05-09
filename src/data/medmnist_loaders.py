import numpy as np
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import medmnist
from .partition import pathological_partition, dirichlet_partition, print_partition_summary

BLOODMNIST_CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocyte",
                      "lymphocyte", "monocyte", "neutrophil", "platelet"]
BLOODMNIST_NUM_CLASSES = 8
BLOODMNIST_CLUSTER_MAP = {
    "hematologic_malignancy": [2, 3, 0],
    "general_medicine": [6, 4, 5, 1, 7],
}

ORGANAMNIST_CLASSES = ["bladder", "femur_left", "femur_right", "heart", "kidney_left",
                       "kidney_right", "liver", "lung_left", "lung_right", "spleen", "uterus"]
ORGANAMNIST_NUM_CLASSES = 11
ORGANAMNIST_CLUSTER_MAP = {
    "thoracic": [3, 7, 8],
    "upper_abdominal": [6, 4, 5, 9],
    "pelvic": [0, 10, 1, 2],
}


def _get_transforms(split):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


class ThreeChannelWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.labels = base_dataset.labels.squeeze()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, int(label)


def get_medmnist_federated(dataset_name, data_dir="./data", num_clients=6,
                           partition_strategy="pathological", clients_per_cluster=2,
                           dirichlet_alpha=0.5, batch_size=32, num_workers=0, seed=42):
    if dataset_name == "bloodmnist":
        DataClass = medmnist.BloodMNIST
        class_names, num_classes, cluster_map = BLOODMNIST_CLASSES, BLOODMNIST_NUM_CLASSES, BLOODMNIST_CLUSTER_MAP
    elif dataset_name == "organamnist":
        DataClass = medmnist.OrganAMNIST
        class_names, num_classes, cluster_map = ORGANAMNIST_CLASSES, ORGANAMNIST_NUM_CLASSES, ORGANAMNIST_CLUSTER_MAP
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    raw_train = DataClass(split="train", transform=_get_transforms("train"), download=True, root=data_dir, size=28)
    raw_test = DataClass(split="test", transform=_get_transforms("test"), download=True, root=data_dir, size=28)
    train_dataset = ThreeChannelWrapper(raw_train)
    test_dataset = ThreeChannelWrapper(raw_test)

    if partition_strategy == "pathological":
        client_train_idx = pathological_partition(train_dataset.labels, cluster_map, clients_per_cluster, seed)
        client_test_idx = pathological_partition(test_dataset.labels, cluster_map, clients_per_cluster, seed)
    else:
        client_train_idx = dirichlet_partition(train_dataset.labels, num_clients, dirichlet_alpha, seed)
        client_test_idx = dirichlet_partition(test_dataset.labels, num_clients, dirichlet_alpha, seed)

    pin = torch.cuda.is_available()
    train_loaders, test_loaders = {}, {}
    for cid in client_train_idx:
        train_loaders[cid] = DataLoader(Subset(train_dataset, client_train_idx[cid]),
                                        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
        test_loaders[cid] = DataLoader(Subset(test_dataset, client_test_idx[cid]),
                                       batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print_partition_summary(train_dataset.labels, client_train_idx, class_names)
    info = {"num_classes": num_classes, "class_names": class_names,
            "num_clients": len(client_train_idx),
            "samples_per_client": {k: len(v) for k, v in client_train_idx.items()}}
    return train_loaders, test_loaders, info
