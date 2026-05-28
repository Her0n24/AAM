#!/bin/bash
#SBATCH --job-name=era5_full_AAM
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=leader_epesc
#SBATCH --time=05:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --qos=high

# Load required modules
module load jaspy

# Activate virtual environment if you have one

~/miniforge3/envs/snapsi/bin/python compute_AAM_full_field.py
