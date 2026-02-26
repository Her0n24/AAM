"""
The code obtain ERA5/ ERA-Interim modelled level output files from share folder and computes monthly mean of some physical properties.
This script computes the climatological mean of a variable at each model level, latitude, and longitude from ERA5 reanalysis data and stores it as a netCDF file. 
This creates a full climatology (lat, lon, [level]) for computing temporal anomalies.
Can handle both 3D data (with level) and vertically integrated 2D data (without level).
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
scratch_path = "/work/scratch-nopw2/hhhn2"
Variable_data_path = f"{scratch_path}/ERA5/monthly_mean/AAM/full" # Full data path (not zonal mean)
output_dir = f"{base_dir}/climatology/full/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

start_yr = 1980
end_yr = 2000

def make_full_climatology(start_year, end_year, variable, vertically_integrated=False):
    """
    Compute full climatology (lat, lon, [level]) for temporal anomaly calculations.
    
    Parameters:
    -----------
    start_year : int
        First year of climatology period
    end_year : int
        Last year of climatology period (exclusive)
    variable : str
        Variable name to process
    vertically_integrated : bool
        If True, data is already vertically integrated (no level dimension)
    """
    # Load all data for the specified period
    suf = '_vertint' if vertically_integrated else ''
    all_files = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            pattern = f"{Variable_data_path}{variable}_ERA5_{year}-{month:02d}{suf}.nc"
            month_files = glob.glob(pattern)
            if not month_files:
                print(f"Warning: No files found for {year}-{month:02d}")
                continue
            all_files.extend(month_files)
    
    if not all_files:
        raise FileNotFoundError(f"No files found for period {start_year}-{end_year}")
    
    print(f"Found {len(all_files)} files")
    
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

    print("Available variables:", list(ds.variables))

    # Resolve requested variable name to an actual data variable in the dataset.
    def _resolve_var(ds, var):
        # Common candidate names
        alt_map = {
            'u': ['u', 'eastward_wind', 'u_component_of_wind'],
            'v': ['v', 'northward_wind', 'v_component_of_wind'],
            'sp': ['sp', 'surface_pressure', 'surface_air_pressure'],
            'ps': ['sp', 'surface_pressure', 'surface_air_pressure'],
            'AAM': ['AAM', 'atmospheric_angular_momentum', 'aam'],
            'u_momentum': ['u_momentum', 'u_aam', 'zonal_momentum'],
            'v_momentum': ['v_momentum', 'v_aam', 'meridional_momentum'],
        }
        candidates = []
        if var in alt_map:
            candidates.extend(alt_map[var])
        candidates.append(var)
        
        # Try candidates in order
        for c in candidates:
            if c in ds.data_vars:
                return ds[c]
        
        # Fallback: if only one data var present, return it
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars.keys())[0]]
        
        raise KeyError(f"Variable '{var}' not found in dataset; tried: {candidates}")

    da = _resolve_var(ds, variable)

    # Ensure required dimensions are present
    required_dims = {'time', 'latitude'}
    if not required_dims.issubset(set(da.dims)):
        # Check for alternative names
        dim_map = {
            'lat': 'latitude',
            'lon': 'longitude',
            'lev': 'level',
            'level': 'level',
        }
        for old_name, new_name in dim_map.items():
            if old_name in da.dims:
                da = da.rename({old_name: new_name})
    
    if 'time' not in da.dims or 'latitude' not in da.dims:
        raise ValueError(f'DataArray must have time and latitude dimensions. Found: {da.dims}')
    
    print(f"Data shape: {da.shape}")
    print(f"Dimensions: {da.dims}")
    
    # Check if data has level dimension
    has_level = 'level' in da.dims
    
    if vertically_integrated and has_level:
        print("Warning: --vertically-integrated flag set but data has level dimension")
        # Squeeze out if singleton
        if len(da.level) == 1:
            da = da.squeeze('level', drop=True)
            print("Squeezed singleton level dimension")
    elif not vertically_integrated and not has_level:
        print("Info: No level dimension found - treating as vertically integrated data")
    
    # Group by month and calculate mean across all years
    # This creates monthly climatology preserving lat, lon, and level structure (if present)
    climatology = da.groupby('time.month').mean('time')
    
    # Add metadata
    data_type = 'vertically integrated' if vertically_integrated or 'level' not in climatology.dims else '3D'
    climatology.attrs['description'] = f'Monthly climatology of {variable} ({data_type})'
    climatology.attrs['climatology_period'] = f'{start_year}-{end_year-1}'
    climatology.attrs['created_by'] = 'make_full_climatology.py'
    if vertically_integrated:
        climatology.attrs['data_type'] = 'vertically_integrated'
    
    print(f"Climatology shape: {climatology.shape}")
    print(f"Climatology dimensions: {climatology.dims}")

    # Save climatology to netCDF
    suffix = '_vi' if vertically_integrated else ''
    output_file = f"{output_dir}ERA5_{variable}_full_climatology{suffix}_{start_year}-{end_year}.nc"
    climatology.to_netcdf(output_file)
    print(f"Climatology saved to: {output_file}")
    print(f"Climatology dimension shape: {climatology.sizes}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make full climatology for a variable over a year range (3D with level or 2D vertically integrated)'
    )
    parser.add_argument('variable', help='variable name to process (e.g. u, v, AAM, u_momentum)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year (default: {end_yr})')
    parser.add_argument('--vertically-integrated', '-vi', action='store_true', 
                        help='Process vertically integrated AAM data (without level dimension)')
    args = parser.parse_args()
    make_full_climatology(args.start, args.end, args.variable, args.vertically_integrated)
