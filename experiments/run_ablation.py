#!/usr/bin/env python3
import argparse, copy, os, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.resnet import create_model
from src.data.medmnist_loaders import get_medmnist_federated
from src.server.servers import FedCPAServer, FedPerServer
from src.utils.metrics import print_comparison_table
from src.utils.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bloodmnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config["seed"] = args.seed
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_loaders, test_loaders, info = get_medmnist_federated(
        args.dataset, config["data"]["data_dir"], batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"], seed=args.seed)
    model = create_model(num_classes=info["num_classes"], pretrained=config["model"]["pretrained"],
                         head_hidden_dim=config["model"]["head_hidden_dim"], head_dropout=config["model"]["head_dropout"])
    fc = config["fedcpa"]
    base = dict(train_loaders=train_loaders, test_loaders=test_loaders, device=device,
                num_rounds=config["num_rounds"], lr=config["training"]["learning_rate"],
                head_lr=config["training"]["head_learning_rate"], local_epochs=config["training"]["local_epochs"],
                head_finetune_epochs=config["training"]["head_finetune_epochs"])
    ablations = {
        "Full CLADE": dict(recluster_interval=fc["recluster_interval"], ema_decay=fc["ema_decay"],
                           alpha_base=fc["alpha_base"], alpha_max=fc["alpha_max"], gamma=fc["gamma"],
                           sensitivity_margin=fc["sensitivity_margin"]),
        "No Drift": dict(recluster_interval=99999, ema_decay=fc["ema_decay"], alpha_base=fc["alpha_base"],
                         alpha_max=fc["alpha_max"], gamma=fc["gamma"], sensitivity_margin=99.0),
        "Fixed Threshold": dict(recluster_interval=fc["recluster_interval"], ema_decay=fc["ema_decay"],
                                alpha_base=0.5, alpha_max=0.5, gamma=0.0, sensitivity_margin=fc["sensitivity_margin"]),
    }
    results = {}
    for name, kwargs in ablations.items():
        print(f"\n>>> Ablation: {name}")
        s = FedCPAServer(model=copy.deepcopy(model), **base, **kwargs)
        results[name] = s.run()
    print(f"\n>>> Ablation: No Clustering (FedPer)")
    s = FedPerServer(model=copy.deepcopy(model), **base)
    results["No Clustering"] = s.run()
    print_comparison_table(results)
    d = os.path.join(args.save_dir, args.dataset, f"ablation_seed_{args.seed}")
    os.makedirs(d, exist_ok=True)
    for n, t in results.items(): t.save(os.path.join(d, f"{n}.json"))
    generate_all_figures(results, os.path.join(d, "figures"), f"{args.dataset}_ablation")

if __name__ == "__main__":
    main()
