"""
This script plots the 3D structure of AAM anomalies.
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


base_dir = os.getcwd()
Variable_data_path_base = f"{base_dir}/monthly_mean/AAM/"
climatology_path_base = f"{base_dir}/climatology/"
output_dir = f"{base_dir}/figures/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Default period 
start_yr, end_yr = 1980, 2000
clim_start_yr, clim_end_yr = 1980, 2000


def calculate_climatology(variable, clim_start_yr, clim_end_yr):
    """
    Calculate and save climatology for a variable.
    Returns the climatology DataArray.
    """
    clim_file = f"{climatology_path_base}ERA5_{variable}_climatology_{clim_start_yr}-{clim_end_yr}_full_level.nc"
    
    if os.path.exists(clim_file):
        print(f"Loading existing climatology from: {clim_file}")
        ds_climatology = xr.open_dataset(clim_file)
        return ds_climatology[variable]
    
    print(f"Calculating climatology for {variable} from {clim_start_yr} to {clim_end_yr}")
    
    # Load all files for climatology period
    all_files = []
    for year in range(clim_start_yr, clim_end_yr):
        for month in range(1, 13):
            pattern = f"{Variable_data_path_base}{variable}_ERA5_{year}-{month:02d}_full_level.nc"
            month_files = glob.glob(pattern)
            if month_files:
                all_files.extend(month_files)
    
    if not all_files:
        raise FileNotFoundError(f"No files found for climatology calculation")
    
    all_files.sort()
    
    # Load time series
    try:
        ds_timeseries = xr.open_mfdataset(all_files, combine='by_coords')
    except ValueError:
        print("Warning: xarray failed to decode time coordinates; loading manually")
        datasets = []
        for f in all_files:
            ds = xr.open_dataset(f, decode_times=False)
            datasets.append(ds)
        ds_timeseries = xr.concat(datasets, dim='time', coords='minimal', compat='override')
        try:
            import cftime
            ds_timeseries = xr.decode_cf(ds_timeseries)
        except Exception as e:
            print(f"Warning: Could not decode times ({e})")
    
    # Calculate monthly climatology
    da_climatology = ds_timeseries[variable].groupby('time.month').mean(dim='time')
    
    # Save climatology
    da_climatology.to_netcdf(clim_file)
    print(f"Climatology saved to: {clim_file}")
    
    return da_climatology


def plot_anomalies_3d(start_year, end_year, variable, clim_start_yr=1980, clim_end_yr=2000):
    """
    Plot 3D structure of anomalies: multiple latitude×level slices at different times
    """
    # Calculate or load climatology
    da_climatology = calculate_climatology(variable, clim_start_yr, clim_end_yr)
    
    # Load time series data
    all_files = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            # AAM files have format: AAM_ERA5_YYYY-MM_full_level.nc
            pattern = f"{Variable_data_path_base}{variable}_ERA5_{year}-{month:02d}_full_level.nc"
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
    
    da_timeseries = ds_timeseries[variable]
    
    print(f"Time series shape: {da_timeseries.shape}")
    print(f"Climatology shape: {da_climatology.shape}")
    print(f"Climatology dimensions: {da_climatology.dims}")
    
    # Reconstruct proper time coordinates from filenames
    import re
    proper_times = []
    for fname in all_files:
        # Extract YYYY-MM from filename pattern: AAM_ERA5_YYYY-MM_full_level.nc
        match = re.search(r'(\d{4})-(\d{2})_full_level\.nc$', fname)
        if match:
            year, month = match.groups()
            proper_times.append(pd.Timestamp(f"{year}-{month}-01"))
    
    if len(proper_times) == len(da_timeseries.time):
        da_timeseries['time'] = proper_times
        print(f"Reconstructed time range: {proper_times[0]} to {proper_times[-1]}")
    
    # Inflate climatology to match time series length
    months = da_timeseries.time.dt.month.values
    
    if 'month' in da_climatology.dims and len(da_climatology.month) == 12:
        # Expand climatology to full time series by selecting the right month for each timestep
        climatology_expanded = da_climatology.sel(month=xr.DataArray(months, dims='time'))
        climatology_expanded['time'] = da_timeseries.time
    else:
        # Climatology is incomplete - use simple broadcast/tile approach
        print(f"Warning: Climatology has only {len(da_climatology.month) if 'month' in da_climatology.dims else 0} month(s), using broadcast method")
        clim_single = da_climatology.squeeze(drop=True)
        climatology_expanded = xr.concat([clim_single] * len(da_timeseries.time), dim='time')
        climatology_expanded['time'] = da_timeseries.time
    
    # Compute anomalies
    anomalies = da_timeseries - climatology_expanded
    anomalies = anomalies.compute()
    
    print(f"Anomalies shape: {anomalies.shape}")
    
    # Check for both 'level' and 'mid_level' dimension names
    has_level = 'level' in anomalies.dims or 'mid_level' in anomalies.dims
    level_dim = 'mid_level' if 'mid_level' in anomalies.dims else 'level'
    
    if not has_level:
        raise ValueError("No vertical level dimension found in dataset")
    
    # Create 3D visualization: multiple time slices
    # Select subset of times to plot (every 2 months)
    time_indices = np.arange(0, len(anomalies.time), 2)  # Every 2 months
    n_slices = len(time_indices)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Method 1: Multiple 2D slices arranged in 3D space
    ax = fig.add_subplot(111, projection='3d')
    
    lat_vals = anomalies.latitude.values
    level_vals = anomalies[level_dim].values
    
    # Compute color limits from data
    vmax = np.abs(anomalies).quantile(0.98).values
    vmin = -vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r
    
    for i, t_idx in enumerate(time_indices):
        time_val = anomalies.time.values[t_idx]
        data_slice = anomalies.isel(time=t_idx).values  # shape: (level, latitude)
        
        # Create meshgrid for this slice
        LAT, LEV = np.meshgrid(lat_vals, level_vals)
        
        # Plot as surface at time position i (time on x-axis, level on z-axis)
        surf = ax.plot_surface(np.ones_like(data_slice)*i, LAT, LEV,
                                facecolors=cmap(norm(data_slice)),
                                shade=False, alpha=0.8)
    
    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_zlabel('Model Level', fontsize=12)
    ax.set_title(f'{variable.upper()} Anomaly 3D Structure: Time × Latitude × Level using climatology {clim_start_yr}-{clim_end_yr}', fontsize=14)
    ax.invert_zaxis()  # Level 1 at top
    
    # Set time labels
    ax.set_xticks(range(n_slices))
    time_labels = [pd.to_datetime(anomalies.time.values[idx]).strftime('%Y-%m') 
                   for idx in time_indices]
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(f'{variable} anomaly', fontsize=12)
    
    plt.tight_layout()
    
    output_file = f"{output_dir}{variable}_anomalies_3d_{start_year}-{end_year}.png"
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"3D figure saved to: {output_file}")
    plt.close()
    
    # Method 2: Volumetric slice plot (latitude-level at multiple times)
    # Show more snapshots now (every 2 months, display up to 12)
    n_snapshots = min(12, len(time_indices))
    snapshot_indices = time_indices[:n_snapshots]
    
    n_cols = 4
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_snapshots > 1 else [axes]
    
    for i, t_idx in enumerate(snapshot_indices):
        if i >= len(axes):
            break
        
        time_val = pd.to_datetime(anomalies.time.values[t_idx])
        data_slice = anomalies.isel(time=t_idx)
        
        # Use explicit levels array to ensure consistent colorbar
        levels = np.linspace(vmin, vmax, 21)
        im = axes[i].contourf(lat_vals, level_vals, data_slice.values,
                             levels=levels, cmap='RdBu_r', extend='both')
        axes[i].set_xlabel('Latitude (°N)', fontsize=10)
        axes[i].set_ylabel('Model Level', fontsize=10)
        axes[i].set_title(f'{time_val.strftime("%Y-%m")}', fontsize=11)
        axes[i].invert_yaxis()
        
        cbar = plt.colorbar(im, ax=axes[i], label=f'{variable} anomaly', pad=0.02)
        cbar.set_ticks(np.linspace(vmin, vmax, 7))  # Show 7 tick marks
    
    # Hide unused subplots
    for j in range(len(snapshot_indices), len(axes)):
        axes[j].axis('off')
    
    fig.suptitle(f'{variable.upper()} Anomaly: Latitude × Level Snapshots using climatology {clim_start_yr}-{clim_end_yr}', fontsize=16)
    plt.tight_layout()
    
    output_file2 = f"{output_dir}{variable}_anomalies_snapshots_{start_year}-{end_year}.png"
    plt.savefig(output_file2, dpi=400, bbox_inches='tight')
    print(f"Snapshot figure saved to: {output_file2}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 3D structure of AAM anomalies')
    parser.add_argument('variable', help='variable name to process (e.g., AAM)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year for data (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year for data (default: {end_yr})')
    parser.add_argument('--clim_start', type=int, default=clim_start_yr, help=f'climatology start year (default: {clim_start_yr})')
    parser.add_argument('--clim_end', type=int, default=clim_end_yr, help=f'climatology end year (default: {clim_end_yr})')
    args = parser.parse_args()
    plot_anomalies_3d(args.start, args.end, args.variable, args.clim_start, args.clim_end)