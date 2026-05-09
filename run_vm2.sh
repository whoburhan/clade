#!/bin/bash
# VM2 (Your T4): All 6 methods on OrganAMNIST
cd "$(dirname "$0")"
python -m experiments.run_experiment --dataset organamnist --method FedAvg --seed 42
python -m experiments.run_experiment --dataset organamnist --method FedProx --seed 42
python -m experiments.run_experiment --dataset organamnist --method SCAFFOLD --seed 42
python -m experiments.run_experiment --dataset organamnist --method IFCA --seed 42
python -m experiments.run_experiment --dataset organamnist --method FedPer --seed 42
python -m experiments.run_experiment --dataset organamnist --method CLADE --seed 42
echo "VM2 COMPLETE"
