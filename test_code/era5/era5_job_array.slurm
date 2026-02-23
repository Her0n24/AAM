#!/bin/bash
#SBATCH --job-name=era5_u_ps_lotus
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --account=leader_epesc
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=1-100%5

# Load required modules
module load jaspy

# Activate virtual environment if you have one

# Run your command
START_YEAR=1990
END_YEAR=1996

MONTH_INDEX=$SLURM_ARRAY_TASK_ID

YEAR=$(( START_YEAR + (MONTH_INDEX -1) / 12 ))
MONTH=$(( (MONTH_INDEX - 1) % 12 + 1 ))

if [ "$YEAR" -gt "$END_YEAR" ]; then
  echo "Out of range: $YEAR-$MONTH"
  exit 0
fi

printf "Processing %04d-%02d\n" $YEAR $MONTH

~/miniforge3/envs/snapsi/bin/python create_monthly_mean.py --year $YEAR --month $MONTH
