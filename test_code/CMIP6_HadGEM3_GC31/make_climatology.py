"""
This script computes the climatological mean of some zonal meaned variable at each model level and latitude from ERA5 reanalysis data and stores it as a netCDF file. 
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import os
import glob
import argparse

base_dir = os.getcwd()
Variable_data_path = f"{base_dir}/monthly_mean/variables/"
output_dir = f"{base_dir}/climatology/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

start_yr = 1980
end_yr = 2000

def make_zm_climatology(start_year, end_year, variable):
    # Load all data for the specified period
    all_files = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            pattern = f"{Variable_data_path}zonal_mean_ERA5_{variable}_{year}-{month:02d}.nc"
            month_files = glob.glob(pattern)
            if not month_files:
                print(f"Warning: No files found for {year}-{month:02d}")
                continue
            all_files.extend(month_files)
    if not all_files:
        raise FileNotFoundError(f"No files found for period {start_year}-{end_year}")
    
    # Load all data; handle datasets with problematic time encodings
    try:
        ds = xr.open_mfdataset(all_files, combine='by_coords')
    except ValueError as e:
        print("Warning: xarray failed to decode time coordinates; retrying with decode_times=False")
        # Use combine='nested' with explicit concat_dim when times aren't decoded
        ds = xr.open_mfdataset(all_files, combine='nested', concat_dim='time', decode_times=False)
        # Try to decode times using cftime if available
        try:
            import cftime  # noqa: F401
            ds = xr.decode_cf(ds)
            print("Info: successfully decoded times using cftime via xr.decode_cf")
        except Exception:
            raise ValueError(
                "Failed to decode time coordinates automatically. "
                "Install the 'cftime' package or open the files with a manual time decoding strategy. "
                "As a quick workaround, you can install cftime: `conda install -c conda-forge cftime` "
            ) from e

    print(ds.variables)

    # Resolve requested variable name to an actual data variable in the dataset.
    def _resolve_var(ds, var):
        # Common candidate names (prefer explicit zonal_mean variants)
        alt_map = {
            'u': ['u_zonal_mean', 'u', 'eastward_wind'],
            'v': ['v_zonal_mean', 'v', 'northward_wind'],
            'sp': ['surface_pressure_zonal_mean', 'sp', 'surface_air_pressure'],
            'ps': ['surface_pressure_zonal_mean', 'sp', 'surface_air_pressure'],
        }
        candidates = []
        if var in alt_map:
            candidates.extend(alt_map[var])
        candidates.extend([f"{var}_zonal_mean", var])
        # Try candidates in order
        for c in candidates:
            if c in ds.data_vars:
                return ds[c]
        # Fallback: if only one data var present, return it
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars.keys())[0]]
        raise KeyError(f"Variable '{var}' not found in dataset; tried: {candidates}")

    da = _resolve_var(ds, variable)

    # ensure dims include time and latitude
    if 'time' not in da.dims or 'latitude' not in da.dims:
        raise ValueError('dataarray must have dims (time, latitude)')
    
    # Group by month and calculate mean across all years
    climatology = da.groupby('time.month').mean('time')
    
    # Add a 'month' coordinate if not present
    if 'month' not in climatology.coords:
        climatology = climatology.rename({'month': 'month'})

    # Save climatology to netCDF
    output_file = f"{output_dir}zonal_mean_ERA5_{variable}_climatology_{start_year}-{end_year}.nc"
    climatology.to_netcdf(output_file)
    print(f"Climatology saved to: {output_file}")
    print(f"Climatology has 'month' dimension with shape: {climatology.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make zonal-mean climatology for a variable over a year range')
    parser.add_argument('variable', help='variable name to process (e.g. u, v, sp)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year (default: {end_yr})')
    args = parser.parse_args()
    make_zm_climatology(args.start, args.end, args.variable)
