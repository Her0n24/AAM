# ERA5 Atmospheric Angular Momentum (AAM) Analysis

This directory contains scripts to compute and analyze Atmospheric Angular Momentum (AAM) from ERA5 reanalysis data.

## Overview

The workflow processes ERA5 monthly mean data (surface pressure and zonal winds) to compute vertically-integrated AAM fields, compute climatologies, and generate diagnostic plots following methods from Scaife et al. (2022, Nature).

## Prerequisites

- Python 3.x with conda environment activated (`conda activate snapsi`)
- Required packages: `xarray`, `numpy`, `pandas`, `tqdm`, `matplotlib`
- Access to ERA5 data on LOTUS/scratch filesystem
- SLURM batch system access for large-scale processing

## Workflow

### 1. Create Monthly Mean Files

Generate monthly mean files for surface pressure (sp) and zonal wind (u) using LOTUS batch processing.

**1.1** Configure date range and paths in preprocessing scripts:
```bash
# For full field data
nano create_monthly_mean.py

# For zonal mean data
nano create_zonal_mean.py
```
Edit `start_year`, `end_year`, and file paths as needed.

**1.2** Configure SLURM batch script:
```bash
nano your_job.slurm
```
Set memory, time limits, and array job parameters.

**1.3** Submit batch job to LOTUS:
```bash
sbatch your_job.slurm
```

To monitor the job

```bash
squeue -u your_username
```

**1.4** Check for missing/failed files:
```bash
# For full field files
python find_missing_files_list_scratch.py

# For zonal mean files
python find_missing_files_list_scratch.py --zonal-mean
```
This generates `missing_files_scratch.txt` listing any corrupt or missing files.

**1.5** Reprocess missing files:
```bash
sbatch era5_job_array_missing_months.slurm
```

To look at the logs of each month

```bash
cat repo_JOB_ID_[0,n].out # n in an integer
cat repo_JOB_ID_[0,n].err
```

### 2. Compute AAM and Climatology

**2.1** Compute AAM from monthly mean fields:
```bash
python compute_AAM_full_field.py
```
- **Input**: Monthly mean `ERA5_u_YYYY-MM.nc` and `ERA5_sp_YYYY-MM.nc` files
- **Output**: 4D AAM fields `AAM_ERA5_YYYY-MM_3D.nc` with dimensions (time, level, latitude, longitude)
- **Format**: NetCDF4 with zlib compression (complevel=4)
- Uses sigma-level coefficients from `l137_a_b.csv`

**2.2** Compute climatological means:
```bash
python make_full_climatology.py
```
- Computes monthly climatologies from AAM fields
- Outputs climatology files for anomaly calculations

### 3. Visualization

**3.1** Generate summary plots (Scaife et al. 2022 style):
```bash
python plot_AAM.py
```
Creates summary diagnostics of AAM variability and trends.

**3.2** Create detailed monthly snapshots:
```bash
python plot_momentum_anomalies_lat_lon3d.py
```
Generates 3D spatial maps of AAM anomalies for detailed monthly analysis.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `create_monthly_mean.py` | Generate monthly means from ERA5 data |
| `create_zonal_mean.py` | Generate zonal mean fields |
| `find_missing_files_list_scratch.py` | Validate data files and identify corrupted/missing files |
| `compute_AAM_full_field.py` | Calculate 3D AAM from u and sp fields |
| `make_full_climatology.py` | Compute monthly climatologies |
| `plot_AAM.py` | Summary diagnostic plots |
| `plot_momentum_anomalies_lat_lon3d.py` | Detailed spatial anomaly maps |
| `l137_a_b.csv` | Sigma-level coefficients for ERA5 L137 |

## File Structure

```
monthly_mean/
  ├── variables/           # Monthly mean input files
  │   ├── ERA5_u_YYYY-MM.nc
  │   └── ERA5_sp_YYYY-MM.nc
  └── AAM/                 # Computed AAM output files
      └── AAM_ERA5_YYYY-MM_3D.nc
```

## Data Locations

- **Scratch space**: `/work/scratch-nopw2/hhhn2/ERA5/monthly_mean/variables`
- **Local workspace**: `~/AAM/test_code/era5/monthly_mean/`

## Notes

- AAM computation uses 137 model levels from ERA5
- Output files use compression to reduce storage by ~10-15x
- Vertical integration can be done post-processing if needed
- Check LOTUS queue limits before submitting large job arrays

## References

Scaife et al. (2022). "Atmospheric Angular Momentum", Nature.
