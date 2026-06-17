#!/usr/bin/env bash


set -e  # Exit immediately if any command fails

python train_model.py -d training_data -m model -v

python run_model.py -d holdout_data -m model -o holdout_outputs -v


python evaluate_model.py -d holdout_data/demographics.csv -o holdout_outputs/demographics.csv -p training_data/demographics.csv -t table.csv



