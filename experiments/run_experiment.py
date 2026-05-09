#!/usr/bin/env python3
import argparse, copy, os, sys, time, json
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.resnet import create_model
from src.data.medmnist_loaders import get_medmnist_federated
from src.server.servers import (FedAvgServer, FedProxServer, ScaffoldServer, IFCAServer, FedPerServer, FedCPAServer)
from src.utils.metrics import MetricsTracker, print_comparison_table
from src.utils.visualization import generate_all_figures


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda"); print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps"); print("Using Apple Silicon GPU (MPS)")
    else:
        d = torch.device("cpu"); print("Using CPU")
    return d


def load_data(dataset_name, config):
    return get_medmnist_federated(dataset_name=dataset_name, data_dir=config["data"]["data_dir"],
        num_clients=config["num_clients"], partition_strategy=config["partition"]["strategy"],
        batch_size=config["training"]["batch_size"], num_workers=config["data"]["num_workers"],
        seed=config["seed"], dirichlet_alpha=config["partition"]["dirichlet_alpha"])


def run_method(method_name, model, train_loaders, test_loaders, device, config):
    nr, lr, le = config["num_rounds"], config["training"]["learning_rate"], config["training"]["local_epochs"]
    m = copy.deepcopy(model)
    if method_name == "FedAvg":
        s = FedAvgServer(m, train_loaders, test_loaders, device, nr, lr, le)
    elif method_name == "FedProx":
        s = FedProxServer(m, train_loaders, test_loaders, device, nr, lr, le, mu=config["fedprox"]["mu"])
    elif method_name == "SCAFFOLD":
        s = ScaffoldServer(m, train_loaders, test_loaders, device, nr, lr, le, global_lr=config["scaffold"]["global_lr"])
    elif method_name == "IFCA":
        s = IFCAServer(m, train_loaders, test_loaders, device, nr, lr, le, num_clusters=config["ifca"]["num_clusters"])
    elif method_name == "FedPer":
        s = FedPerServer(m, train_loaders, test_loaders, device, nr, lr,
                         config["training"]["head_learning_rate"], le, config["training"]["head_finetune_epochs"])
    elif method_name == "CLADE":
        fc = config["fedcpa"]
        s = FedCPAServer(m, train_loaders, test_loaders, device, nr, lr,
                         config["training"]["head_learning_rate"], le, config["training"]["head_finetune_epochs"],
                         fc["recluster_interval"], fc["ema_decay"], fc["alpha_base"], fc["alpha_max"],
                         fc["gamma"], fc["sensitivity_margin"], fc["min_cluster_size"])
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return s.run()


def run_experiment(dataset_name, methods, config, seed, save_dir):
    print(f"\n{'#'*60}\nDATASET: {dataset_name.upper()} | SEED: {seed}\n{'#'*60}")
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    config["seed"] = seed
    device = get_device()
    train_loaders, test_loaders, data_info = load_data(dataset_name, config)
    model = create_model(num_classes=data_info["num_classes"], pretrained=config["model"]["pretrained"],
                         head_hidden_dim=config["model"]["head_hidden_dim"], head_dropout=config["model"]["head_dropout"])
    results = {}
    for method_name in methods:
        print(f"\n>>> Starting {method_name}...")
        start = time.time()
        metrics = run_method(method_name, model, train_loaders, test_loaders, device, config)
        print(f">>> {method_name} completed in {time.time()-start:.1f}s")
        results[method_name] = metrics
        method_dir = os.path.join(save_dir, dataset_name, f"seed_{seed}")
        os.makedirs(method_dir, exist_ok=True)
        metrics.save(os.path.join(method_dir, f"{method_name}.json"))
    print_comparison_table(results)
    fig_dir = os.path.join(save_dir, dataset_name, f"seed_{seed}", "figures")
    generate_all_figures(results, fig_dir, dataset_name)
    return results


def aggregate_seeds(save_dir, dataset_name, methods, seeds):
    print(f"\n{'='*60}\nAGGREGATED: {dataset_name.upper()} ({len(seeds)} seeds)\n{'='*60}")
    method_results = {m: [] for m in methods}
    for seed in seeds:
        for method in methods:
            fp = os.path.join(save_dir, dataset_name, f"seed_{seed}", f"{method}.json")
            if os.path.exists(fp):
                method_results[method].append(MetricsTracker.load(fp).get_final_results())
    print(f"{'Method':<15} | {'Global Acc':>14} | {'Worst Acc':>14}")
    print("-" * 50)
    summary = {}
    for method in methods:
        if not method_results[method]: continue
        ga = [r["global_accuracy"] for r in method_results[method]]
        wa = [r["worst_accuracy"] for r in method_results[method]]
        mg, sg, mw, sw = np.mean(ga), np.std(ga), np.mean(wa), np.std(wa)
        print(f"{method:<15} | {mg:.4f} +/- {sg:.4f} | {mw:.4f} +/- {sw:.4f}")
        summary[method] = {"global_mean": float(mg), "global_std": float(sg), "worst_mean": float(mw), "worst_std": float(sw)}
    sp = os.path.join(save_dir, dataset_name, "summary.json")
    with open(sp, "w") as f: json.dump(summary, f, indent=2)
    print(f"\nSaved: {sp}")


def main():
    parser = argparse.ArgumentParser(description="CLADE Experiments")
    parser.add_argument("--dataset", type=str, default="bloodmnist", choices=["bloodmnist", "organamnist", "all"])
    parser.add_argument("--method", type=str, default="all", choices=["FedAvg", "FedProx", "SCAFFOLD", "IFCA", "FedPer", "CLADE", "all"])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    if args.rounds: config["num_rounds"] = args.rounds
    if args.test:
        config["num_rounds"] = 3; config["training"]["local_epochs"] = 1; config["data"]["num_workers"] = 0
    all_methods = ["FedAvg", "FedProx", "SCAFFOLD", "IFCA", "FedPer", "CLADE"]
    methods = all_methods if args.method == "all" else [args.method]
    datasets = ["bloodmnist", "organamnist"] if args.dataset == "all" else [args.dataset]
    seeds = [args.seed] if args.seed else config["eval"]["seeds"]
    for ds in datasets:
        for seed in seeds:
            run_experiment(ds, methods, config, seed, args.save_dir)
        if len(seeds) > 1:
            aggregate_seeds(args.save_dir, ds, methods, seeds)
    print(f"\n{'='*60}\nALL EXPERIMENTS COMPLETE\n{'='*60}")


if __name__ == "__main__":
    main()
