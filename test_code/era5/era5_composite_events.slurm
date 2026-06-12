#!/bin/bash
#SBATCH --job-name=era5_AAM_event
#SBATCH --output=logs/compo%j.out
#SBATCH --error=logs/compo%j.err
#SBATCH --account=leader_epesc
#SBATCH --time=10:00:00
#SBATCH --mem=900G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=high

# Load required modules
module load jaspy

# Activate virtual environment if you have one

~/miniforge3/envs/snapsi/bin/python -u event_composite_all_plot.py \
    --onset-season all --composite-start onset --region all --p-min 150.0 --p-max 700 2>&1 | \
    grep --line-buffered -v -E "H5[A-Z]|major:|minor:|Object not found|Error detected|failed|unable to|can't locate|can't open|#0[0-9]"
