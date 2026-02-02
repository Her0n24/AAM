#!/bin/bash
#SBATCH --job-name=era5_u_ps_zm_lotus
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=leader_epesc
#SBATCH --time=01:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --qos=standard

# Load required modules
module load jaspy

# Activate virtual environment if you have one

~/miniforge3/envs/snapsi/bin/python create_zonal_mean.py --year 1991  --month 09
