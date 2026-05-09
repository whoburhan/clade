#!/bin/bash
# M1 MacBook Pro: Ablation study on BloodMNIST
cd "$(dirname "$0")"
python -m experiments.run_ablation --dataset bloodmnist --seed 42
echo "MAC M1 COMPLETE"
