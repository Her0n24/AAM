"""
This script plots the evolution structure of the zonal meaned variable anomalies as a trial.
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import matplotlib.dates as mdates
import pandas as pd


base_dir = os.getcwd()
Variable_data_path_base = f"{base_dir}/monthly_mean/variables/"
climatology_path_base = f"{base_dir}/climatology/"
output_dir = f"{base_dir}/figures/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Default period 
start_yr, end_yr = 1980, 2000
clim_start_yr, clim_end_yr = 1980, 2000

def plot_anomalies_structure(start_year, end_year, variable, clim_start_yr=1980, clim_end_yr=2000):
    """
    Plot height×time and latitude×time cross-sections of anomalies
    """
    # Load climatology
    clim_file = f"{climatology_path_base}zonal_mean_ERA5_{variable}_climatology_{clim_start_yr}-{clim_end_yr}.nc"
    if not os.path.exists(clim_file):
        raise FileNotFoundError(f"Climatology file not found: {clim_file}")
    
    ds_climatology = xr.open_dataset(clim_file)
    print(f"Loaded climatology from: {clim_file}")
    
    # Load time series data
    all_files = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            pattern = f"{Variable_data_path_base}zonal_mean_ERA5_{variable}_{year}-{month:02d}.nc"
            month_files = glob.glob(pattern)
            if month_files:
                all_files.extend(month_files)
    
    if not all_files:
        raise FileNotFoundError(f"No time series files found for {variable} in {start_year}-{end_year}")
    
    # Sort files to ensure chronological order
    all_files.sort()
    
    # Load time series with same decoding strategy as climatology script
    try:
        ds_timeseries = xr.open_mfdataset(all_files, combine='by_coords')
    except ValueError:
        print("Warning: xarray failed to decode time coordinates; retrying with decode_times=False")
        # Load each file individually and manually combine
        datasets = []
        for f in all_files:
            ds = xr.open_dataset(f, decode_times=False)
            datasets.append(ds)
        
        # Concatenate manually
        ds_timeseries = xr.concat(datasets, dim='time', coords='minimal', compat='override')
        
        try:
            import cftime  # noqa: F401
            ds_timeseries = xr.decode_cf(ds_timeseries)
            print("Info: successfully decoded times using cftime")
        except Exception as e:
            print(f"Warning: Could not decode times ({e}); proceeding with numeric time values")
    
        # Resolve variable name
    def _resolve_var(ds, var):
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
        
        print(f"Available variables in dataset: {list(ds.data_vars.keys())}")
        
        for c in candidates:
            if c in ds.data_vars:
                return ds[c]
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars.keys())[0]]
        raise KeyError(f"Variable '{var}' not found; tried: {candidates}")
    
    da_timeseries = _resolve_var(ds_timeseries, variable)
    da_climatology = _resolve_var(ds_climatology, variable)
    
    print(f"Time series shape: {da_timeseries.shape}")
    print(f"Climatology shape: {da_climatology.shape}")
    print(f"Climatology dimensions: {da_climatology.dims}")
    print(f"Climatology coords: {list(da_climatology.coords.keys())}")
    print(f"Time series time range: {da_timeseries.time.values[0]} to {da_timeseries.time.values[-1]}")
    
    # Reconstruct proper time coordinates from filenames
    import re
    proper_times = []
    for fname in all_files:
        # Extract YYYY-MM from filename pattern: zonal_mean_ERA5_{variable}_{YYYY}-{MM}.nc
        match = re.search(r'(\d{4})-(\d{2})\.nc$', fname)
        if match:
            year, month = match.groups()
            proper_times.append(pd.Timestamp(f"{year}-{month}-01"))
    
    if len(proper_times) != len(da_timeseries.time):
        raise ValueError(f"Mismatch: {len(proper_times)} filenames vs {len(da_timeseries.time)} time steps")
    
    print(f"Reconstructed time range: {proper_times[0]} to {proper_times[-1]}")
    
        # Replace corrupted time coordinate with proper times
    da_timeseries['time'] = proper_times
    
    # Inflate climatology to match time series length
    # Create a mapping from each timestep to its corresponding month
    months = da_timeseries.time.dt.month.values
    
    print(f"Unique months in time series: {np.unique(months)}")
    print(f"Months in climatology: {da_climatology.month.values if 'month' in da_climatology.dims else 'N/A'}")
    
    # Check if climatology has all 12 months
    if 'month' in da_climatology.dims and len(da_climatology.month) == 12:
        # Expand climatology to full time series by selecting the right month for each timestep
        climatology_expanded = da_climatology.sel(month=xr.DataArray(months, dims='time'))
        climatology_expanded['time'] = da_timeseries.time
    else:
        # Climatology is incomplete - use simple broadcast/tile approach
        print(f"Warning: Climatology has only {len(da_climatology.month) if 'month' in da_climatology.dims else 0} month(s), using broadcast method")
        
        # Squeeze out the month dimension and tile along time
        clim_single = da_climatology.squeeze(drop=True)
        
        # Repeat the single climatology for each timestep
        climatology_expanded = xr.concat([clim_single] * len(da_timeseries.time), dim='time')
        climatology_expanded['time'] = da_timeseries.time
    
    print(f"Expanded climatology shape: {climatology_expanded.shape}")
    
    # Compute anomalies: simple subtraction now that dimensions match
    anomalies = da_timeseries - climatology_expanded
    
    print(f"Anomalies computed, shape: {anomalies.shape}")
    print(f"Anomalies time values (first 5): {anomalies.time.values[:5]}")
    print(f"Anomalies time values (last 5): {anomalies.time.values[-5:]}")
    
    
    # Prepare for plotting
    has_level = 'level' in anomalies.dims
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Height (level) vs Time (averaged over latitude)
    if has_level:
        anom_height_time = anomalies.mean(dim='latitude')  # average over latitude
        
        # Compute data to avoid dask chunking issues with quantile
        anom_height_time = anom_height_time.compute()
        
        # Use numeric time index instead of problematic datetime
        time_indices = np.arange(len(anom_height_time.time))
        level_vals = anom_height_time.level.values
        
        print(f"Time indices: 0 to {len(time_indices)-1}")
        print(f"Level range: {level_vals.min()} to {level_vals.max()}")
        print(f"Data range: {anom_height_time.min().values} to {anom_height_time.max().values}")
        print(f"Data shape: {anom_height_time.shape}")
        print(f"Any NaN?: {np.isnan(anom_height_time.values).any()}")
        
        vmax = np.abs(anom_height_time).quantile(0.98).values
        vmin = -vmax
        print(f"Color limits: {vmin} to {vmax}")
        
        # Use imshow with extent based on indices
        extent = [0, len(time_indices)-1, level_vals[-1], level_vals[0]]
        
        im1 = ax1.imshow(anom_height_time.T, aspect='auto', origin='upper',
                        extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        ax1.set_ylabel('Model Level', fontsize=12)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_title(f'{variable.upper()} Anomaly: Height × Time (latitude-averaged)', fontsize=14)
        
        # Create custom x-axis labels showing YYYY-MM
        # Sample every N months for readability
        n_ticks = 10
        tick_indices = np.linspace(0, len(time_indices)-1, n_ticks, dtype=int)
        tick_labels = []
        for idx in tick_indices:
            dt = pd.to_datetime(anom_height_time.time.values[idx])
            tick_labels.append(dt.strftime('%Y-%m'))
        
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels, ha='right')
        
        plt.colorbar(im1, ax=ax1, label=f'{variable} anomaly', pad=0.02)
    else:
        ax1.text(0.5, 0.5, 'No vertical levels in dataset', 
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'{variable.upper()} Anomaly: Height × Time (N/A)', fontsize=14)
    
    # Plot 2: Latitude vs Time (averaged over level if present)
    if has_level:
        anom_lat_time = anomalies.mean(dim='level')  # average over levels
    else:
        anom_lat_time = anomalies
    
    # Compute data to avoid dask chunking issues
    anom_lat_time = anom_lat_time.compute()
    
    # Use numeric time index
    time_indices = np.arange(len(anom_lat_time.time))
    lat_vals = anom_lat_time.latitude.values
    
    print(f"\nLatitude plot:")
    print(f"Time indices: 0 to {len(time_indices)-1}")
    print(f"Latitude range: {lat_vals.min()} to {lat_vals.max()}")
    print(f"Data range: {anom_lat_time.min().values} to {anom_lat_time.max().values}")
    print(f"Data shape: {anom_lat_time.shape}")
    print(f"Any NaN?: {np.isnan(anom_lat_time.values).any()}")
    
    vmax = np.abs(anom_lat_time).quantile(0.98).values
    vmin = -vmax
    print(f"Color limits: {vmin} to {vmax}")
    
    # Use imshow with extent based on indices
    extent = [0, len(time_indices)-1, lat_vals[-1], lat_vals[0]]
    
    im2 = ax2.imshow(anom_lat_time.T, aspect='auto', origin='upper',
                    extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    ax2.set_ylabel('Latitude (°N)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_title(f'{variable.upper()} Anomaly: Latitude × Time (level-averaged)', fontsize=14)
    
    # Create custom x-axis labels showing YYYY-MM
    n_ticks = 10
    tick_indices = np.linspace(0, len(time_indices)-1, n_ticks, dtype=int)
    tick_labels = []
    for idx in tick_indices:
        dt = pd.to_datetime(anom_lat_time.time.values[idx])
        tick_labels.append(dt.strftime('%Y-%m'))
    
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, ha='right')
    
    plt.colorbar(im2, ax=ax2, label=f'{variable} anomaly', pad=0.02)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}{variable}_anomalies_{start_year}-{end_year}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot zonal-mean anomaly structure for a variable over a year range')
    parser.add_argument('variable', help='variable name to process (e.g. u, v, sp)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year for data (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year for data (default: {end_yr})')
    parser.add_argument('--clim_start', type=int, default=clim_start_yr, help=f'climatology start year (default: {clim_start_yr})')
    parser.add_argument('--clim_end', type=int, default=clim_end_yr, help=f'climatology end year (default: {clim_end_yr})')
    args = parser.parse_args()
    plot_anomalies_structure(args.start, args.end, args.variable, args.clim_start, args.clim_end)