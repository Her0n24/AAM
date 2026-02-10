"""
This script plots the 3D structure of zonal meaned variable anomalies.
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
Variable_data_path_base = f"{base_dir}/monthly_mean/variables/"
climatology_path_base = f"{base_dir}/climatology/"
output_dir = f"{base_dir}/figures/"

os.makedirs(output_dir, exist_ok=True)

start_yr, end_yr = 1980, 2000
clim_start_yr, clim_end_yr = 1980, 2000


def plot_anomalies_3d(start_year, end_year, variable, clim_start_yr=1980, clim_end_yr=2000):
    """
    Plot 3D structure of anomalies: multiple latitude×level slices at different times
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
    
    all_files.sort()
    
    # Load time series
    try:
        ds_timeseries = xr.open_mfdataset(all_files, combine='by_coords')
    except ValueError:
        print("Warning: xarray failed to decode time coordinates; loading manually")
        datasets = []
        for f in all_files:
            ds = xr.open_dataset(f, decode_times=False)
            if 'time' in ds.dims and 'time' not in ds.coords:
                ds = ds.set_coords('time')
            datasets.append(ds)
        ds_timeseries = xr.concat(datasets, dim='time', coords='minimal', compat='override')
        try:
            import cftime
            ds_timeseries = xr.decode_cf(ds_timeseries)
            print("Info: successfully decoded times using cftime")
        except Exception as e:
            print(f"Warning: Could not decode times ({e})")
    
    # Resolve variable
    def _resolve_var(ds, var):
        alt_map = {
            'u': ['u_zonal_mean', 'u', 'eastward_wind'],
            'v': ['v_zonal_mean', 'v', 'northward_wind'],
            'sp': ['surface_pressure_zonal_mean', 'sp', 'surface_air_pressure'],
        }
        candidates = alt_map.get(var, []) + [f"{var}_zonal_mean", var]
        for c in candidates:
            if c in ds.data_vars:
                return ds[c]
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars.keys())[0]]
        raise KeyError(f"Variable '{var}' not found; tried: {candidates}")
    
    da_timeseries = _resolve_var(ds_timeseries, variable)
    da_climatology = _resolve_var(ds_climatology, variable)
    
    # Reconstruct time from filenames
    import re
    proper_times = []
    for fname in all_files:
        match = re.search(r'(\d{4})-(\d{2})\.nc$', fname)
        if match:
            year, month = match.groups()
            proper_times.append(pd.Timestamp(f"{year}-{month}-01"))
    
    if len(proper_times) == len(da_timeseries.time):
        da_timeseries['time'] = proper_times
    
    # Inflate climatology
    months = da_timeseries.time.dt.month.values
    
    if 'month' in da_climatology.dims and len(da_climatology.month) == 12:
        climatology_expanded = da_climatology.sel(month=xr.DataArray(months, dims='time'))
        climatology_expanded['time'] = da_timeseries.time
    else:
        clim_single = da_climatology.squeeze(drop=True)
        climatology_expanded = xr.concat([clim_single] * len(da_timeseries.time), dim='time')
        climatology_expanded['time'] = da_timeseries.time
    
    # Compute anomalies
    anomalies = da_timeseries - climatology_expanded
    anomalies = anomalies.compute()
    
    print(f"Anomalies shape: {anomalies.shape}")
    
    # Create 3D visualization: multiple time slices
    # Select subset of times to plot (every 2 months)
    time_indices = np.arange(0, len(anomalies.time), 2)  # Every 2 months
    n_slices = len(time_indices)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Method 1: Multiple 2D slices arranged in 3D space
    ax = fig.add_subplot(111, projection='3d')
    
    lat_vals = anomalies.latitude.values
    level_vals = anomalies.level.values
    
    # Fixed color limits
    vmin = -60
    vmax = 60
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
    ax.set_zlabel('Level', fontsize=12)
    ax.set_title(f'{variable.upper()} Anomaly 3D Structure: Time × Latitude × Level', fontsize=14)
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
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
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
        axes[i].set_ylabel('Level', fontsize=10)
        axes[i].set_title(f'{time_val.strftime("%Y-%m")}', fontsize=11)
        axes[i].invert_yaxis()
        
        cbar = plt.colorbar(im, ax=axes[i], label=f'{variable} anomaly', pad=0.02)
        cbar.set_ticks(np.linspace(vmin, vmax, 7))  # Show 7 tick marks
    
    # Hide unused subplots
    for j in range(len(snapshot_indices), len(axes)):
        axes[j].axis('off')
    
    fig.suptitle(f'{variable.upper()} Anomaly: Latitude × Level Snapshots', fontsize=16)
    plt.tight_layout()
    
    output_file2 = f"{output_dir}{variable}_anomalies_snapshots_{start_year}-{end_year}.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Snapshot figure saved to: {output_file2}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 3D structure of zonal-mean anomalies')
    parser.add_argument('variable', help='variable name (e.g. u, v, sp)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year (default: {end_yr})')
    parser.add_argument('--clim_start', type=int, default=clim_start_yr, help=f'climatology start (default: {clim_start_yr})')
    parser.add_argument('--clim_end', type=int, default=clim_end_yr, help=f'climatology end (default: {clim_end_yr})')
    args = parser.parse_args()
    plot_anomalies_3d(args.start, args.end, args.variable, args.clim_start, args.clim_end)
