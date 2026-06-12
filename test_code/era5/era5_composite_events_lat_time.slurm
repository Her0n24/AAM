#!/bin/bash
#SBATCH --job-name=era5_AAM_event
#SBATCH --output=logs/compo%j.out
#SBATCH --error=logs/compo%j.err
#SBATCH --account=leader_epesc
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=high

# Load required modules
module load jaspy

# Activate virtual environment if you have one

~/miniforge3/envs/snapsi/bin/python -u event_composite_lat_time.py \
    --onset-season all --composite-start december_onset_year --p-min 150.0 --p-max 700 --region all 2>&1 | \
    grep --line-buffered -v -E "H5[A-Z]|major:|minor:|Object not found|Error detected|failed|unable to|can't locate|can't open|#0[0-9]"
