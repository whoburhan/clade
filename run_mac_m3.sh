#!/bin/bash
# Mac M3 Air: CLADE extra seeds on both datasets
cd "$(dirname "$0")"
python -m experiments.run_experiment --dataset bloodmnist --method CLADE --seed 123
python -m experiments.run_experiment --dataset bloodmnist --method CLADE --seed 456
python -m experiments.run_experiment --dataset organamnist --method CLADE --seed 123
python -m experiments.run_experiment --dataset organamnist --method CLADE --seed 456
echo "MAC M3 COMPLETE"
