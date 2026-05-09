#!/bin/bash
# VM1 (Friend's T4): All 6 methods on BloodMNIST
cd "$(dirname "$0")"
python -m experiments.run_experiment --dataset bloodmnist --method FedAvg --seed 42
python -m experiments.run_experiment --dataset bloodmnist --method FedProx --seed 42
python -m experiments.run_experiment --dataset bloodmnist --method SCAFFOLD --seed 42
python -m experiments.run_experiment --dataset bloodmnist --method IFCA --seed 42
python -m experiments.run_experiment --dataset bloodmnist --method FedPer --seed 42
python -m experiments.run_experiment --dataset bloodmnist --method CLADE --seed 42
echo "VM1 COMPLETE"
