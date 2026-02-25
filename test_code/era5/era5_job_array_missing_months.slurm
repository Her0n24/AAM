#!/bin/bash
#SBATCH --job-name=era5_repro_missing
#SBATCH --output=logs/repo_%A_%a.out
#SBATCH --error=logs/repo_%A_%a.err
#SBATCH --account=leader_epesc
#SBATCH --time=01:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=1-130%5

# Activate your conda environment
module load jaspy

# Read the missing files list
MISSING_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" missing_files_scratch.txt)

if [ -n "$MISSING_FILE" ]; then
    # Parse year, month, and variable
    read -r YEAR MONTH VAR <<< "$MISSING_FILE"

    echo "Processing Year: $YEAR, Month: $MONTH"

    # Run your create_zonal_mean script for this specific year/month/variable
    cd /home/users/hhhn2/AAM/test_code/era5
    ~/miniforge3/envs/snapsi/bin/python create_zonal_mean.py --year $YEAR --month $MONTH 
else
    echo "No more files to process for task ID $SLURM_ARRAY_TASK_ID"
fi
