"""
This script plots latitude × longitude (north × east) maps of variable anomalies.
References plot_variable_anomalies_3d.py structure.
Intended for a single event only
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

base_dir = os.getcwd()
Variable_data_path_base = f"{base_dir}/monthly_mean/AAM/"
climatology_path_base = f"{base_dir}/climatology/full/"
output_dir = f"{base_dir}/figures/"
os.makedirs(output_dir, exist_ok=True)

# Event
start_yr, end_yr = 1997, 1998

# Climatology period
clim_start_yr, clim_end_yr = 1980, 2000


def plot_lat_lon_anomalies(start_year, end_year, variable, clim_start_yr=1980, clim_end_yr=2000, find_extremum='max'):
    """
    Plot latitude × longitude maps of anomalies for vertically integrated AAM variables.
    
    Parameters:
    -----------
    start_year : int
        Start year for time series
    end_year : int
        End year for time series
    variable : str
        Variable name (e.g., 'u', 'v', 'momentum')
    clim_start_yr : int
        Climatology start year
    clim_end_yr : int
        Climatology end year
    find_extremum : str
        Either 'max' or 'min' to find maximum or minimum AAM anomaly
    """
    # Load climatology
    clim_file = f"{climatology_path_base}ERA5_{variable}_full_climatology_vi_{clim_start_yr}-{clim_end_yr}.nc"
    print(f"Loading climatology from: {clim_file}")
    if not os.path.exists(clim_file):
        raise FileNotFoundError(f"Climatology file not found: {clim_file}")
    
    ds_climatology = xr.open_dataset(clim_file)
    print(f"Loaded climatology from: {clim_file}")
    
    # Load time series data
    all_files = []
    for year in range(start_year, end_year):
        for month in range(1, 13):
            pattern = f"{Variable_data_path_base}{variable}_ERA5_{year}-{month:02d}_vertint.nc"
            month_files = glob.glob(pattern)
            if month_files:
                all_files.extend(month_files)
                print(f"Found file: {month_files[0]}")  # Print the first file found for this month
    
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
            'u': ['u', 'eastward_wind', 'u_component_of_wind', 'u_momentum'],
            'v': ['v', 'northward_wind', 'v_component_of_wind', 'v_momentum'],
            'momentum': ['momentum', 'angular_momentum', 'aam'],
        }
        candidates = alt_map.get(var, []) + [var]
        for c in candidates:
            if c in ds.data_vars:
                return ds[c]
        if len(ds.data_vars) == 1:
            return ds[list(ds.data_vars.keys())[0]]
        raise KeyError(f"Variable '{var}' not found; tried: {candidates}")
    
    da_timeseries = _resolve_var(ds_timeseries, variable)
    da_climatology = _resolve_var(ds_climatology, variable)
    
    # Standardize dimension names
    dim_map_lat = {'latitude': 'latitude', 'lat': 'latitude', 'y': 'latitude'}
    dim_map_lon = {'longitude': 'longitude', 'lon': 'longitude', 'x': 'longitude'}
    dim_map_lev = {'level': 'level', 'lev': 'level', 'pressure': 'level', 'plev': 'level'}
    
    for old, new in dim_map_lat.items():
        if old in da_timeseries.dims and old != new:
            da_timeseries = da_timeseries.rename({old: new})
        if old in da_climatology.dims and old != new:
            da_climatology = da_climatology.rename({old: new})
    
    for old, new in dim_map_lon.items():
        if old in da_timeseries.dims and old != new:
            da_timeseries = da_timeseries.rename({old: new})
        if old in da_climatology.dims and old != new:
            da_climatology = da_climatology.rename({old: new})
    
    for old, new in dim_map_lev.items():
        if old in da_timeseries.dims and old != new:
            da_timeseries = da_timeseries.rename({old: new})
        if old in da_climatology.dims and old != new:
            da_climatology = da_climatology.rename({old: new})
    
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
    
    # AAM variables are already vertically integrated, no level selection needed
    # Squeeze out any singleton level dimension if present
    if 'level' in da_timeseries.dims:
        if len(da_timeseries.level) == 1:
            da_timeseries = da_timeseries.squeeze('level', drop=True)
            print("Removed singleton level dimension")
    if 'level' in da_climatology.dims:
        if len(da_climatology.level) == 1:
            da_climatology = da_climatology.squeeze('level', drop=True)
    
    # Inflate climatology to match time dimension
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
    print(f"Dimensions: {anomalies.dims}")
    
    # Get lat/lon values
    lat_vals = anomalies.latitude.values
    lon_vals = anomalies.longitude.values
    
    # Create snapshot plots: multiple lat×lon maps at different times
    time_indices = np.arange(0, len(anomalies.time), 1)  # Every months
    n_snapshots = min(12, len(time_indices))
    snapshot_indices = time_indices[:n_snapshots]
    
    n_cols = 3
    n_rows = 4
    
    # Determine color limits
    vmin = np.nanpercentile(anomalies.values, 1)
    vmax = np.nanpercentile(anomalies.values, 99)
    # Make symmetric
    vlim = max(abs(vmin), abs(vmax))
    vmin, vmax = -vlim, vlim
    
    print(f"Color limits: [{vmin:.2e}, {vmax:.2e}]")
    
    # Create figure with GridSpec for paired subplots (profile + map)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(35, 4*n_rows))
    gs = GridSpec(n_rows, n_cols * 2, figure=fig, width_ratios=[0.6, 6] * n_cols, hspace=0.25, wspace=0.15)
    
    map_axes = []
    profile_axes = []
    for row in range(n_rows):
        for col in range(n_cols):
            profile_axes.append(fig.add_subplot(gs[row, col*2]))
            map_axes.append(fig.add_subplot(gs[row, col*2 + 1], projection=ccrs.PlateCarree(central_longitude=180)))
    
    for i, t_idx in enumerate(snapshot_indices):
        if i >= n_snapshots:
            break
        
        time_val = pd.to_datetime(anomalies.time.values[t_idx])
        data_slice = anomalies.isel(time=t_idx)
        
        # Plot data on map
        levels = np.linspace(vmin, vmax, 21)
        im = map_axes[i].contourf(lon_vals, lat_vals, data_slice.values,
                        levels=levels, cmap='RdBu_r', extend='both',
                        transform=ccrs.PlateCarree())
        
        # Add coastlines and features
        map_axes[i].coastlines(resolution='110m', linewidth=0.5)
        map_axes[i].add_feature(cfeature.BORDERS, linewidth=0.3)
        
        # Add gridlines
        gl = map_axes[i].gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set latitude extent to -60 to 60
        map_axes[i].set_ylim(-60, 60)
        
        map_axes[i].set_xlabel('Longitude (°E)', fontsize=11)
        map_axes[i].set_ylabel('Latitude (°N)', fontsize=11)
        map_axes[i].set_title(f'{time_val.strftime("%Y-%m")}', fontsize=21)
        map_axes[i].tick_params(labelsize=24)
        
        # Find location of maximum or minimum AAM on the 2D map (in northern hemisphere)
        nh_mask_2d = lat_vals > 0  # Northern hemisphere only
        data_nh = data_slice.values[nh_mask_2d, :]  # Select NH latitudes
        lat_nh = lat_vals[nh_mask_2d]
        
        # Only find extremum if there are valid (non-NaN) values
        if np.any(np.isfinite(data_nh)):
            if find_extremum == 'min':
                extreme_2d_idx = np.unravel_index(np.nanargmin(data_nh), data_nh.shape)
            else:  # default to 'max'
                extreme_2d_idx = np.unravel_index(np.nanargmax(data_nh), data_nh.shape)
            
            extreme_lat_2d = lat_nh[extreme_2d_idx[0]]
            extreme_lon_2d = lon_vals[extreme_2d_idx[1]]
            
            # Add cross marker at the extremum location
            map_axes[i].plot(extreme_lon_2d, extreme_lat_2d, color='C1', marker='x', markersize=15, markeredgewidth=3,
                            transform=ccrs.PlateCarree(), zorder=10)
        
        # Create zonal mean profile (average over longitude, ignoring NaN)
        zonal_mean = np.nanmean(data_slice.values, axis=1)
        
        # Find latitude of maximum or minimum in northern hemisphere
        nh_mask = lat_vals > 0  # Northern hemisphere only
        nh_zonal_mean = zonal_mean[nh_mask]
        nh_lats = lat_vals[nh_mask]
        
        if np.any(np.isfinite(nh_zonal_mean)):
            if find_extremum == 'min':
                extreme_idx = np.nanargmin(nh_zonal_mean)
            else:  # default to 'max'
                extreme_idx = np.nanargmax(nh_zonal_mean)
            
            extreme_lat = nh_lats[extreme_idx]
        else:
            extreme_lat = None  # No valid data to find extremum
        
        profile_axes[i].plot(zonal_mean, lat_vals, 'C0-', linewidth=1.5)
        profile_axes[i].axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        profile_axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Add horizontal line at the latitude of extremum AAM (if found)
        if extreme_lat is not None:
            profile_axes[i].axhline(extreme_lat, color='C1', linewidth=2, linestyle='-', alpha=0.8)
        
        profile_axes[i].set_ylim(-60, 60)
        profile_axes[i].set_xlim(-5 * 1e24, 5 * 1e24)  # Adjust as needed based on variable units
        profile_axes[i].set_xlabel('Zonal Mean', fontsize=11)
        profile_axes[i].set_ylabel('Latitude (°N)', fontsize=11)
        profile_axes[i].grid(True, alpha=0.3)
        profile_axes[i].tick_params(labelsize=11)
    
    # Hide unused subplots
    for j in range(len(snapshot_indices), len(map_axes)):
        map_axes[j].axis('off')
        profile_axes[j].axis('off')
    
    # Add a single colorbar at the bottom for all subplots
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.01])  # [left, bottom, width, height]
    
    # Create discrete levels matching the contour levels
    # Ensure vmin and vmax are valid (not NaN or Inf)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        print(f"Warning: Invalid color limits detected (vmin={vmin}, vmax={vmax}), using default range")
        vmin, vmax = -1e24, 1e24
    
    levels_cbar = np.linspace(vmin, vmax, 11)
    norm = mcolors.BoundaryNorm(levels_cbar, ncolors=256)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both', spacing='proportional')
    cbar.set_label(f'{variable} anomaly', fontsize=22)
    cbar.set_ticks(levels_cbar)
    cbar.ax.tick_params(labelsize=16)
    
    fig.suptitle(f'{variable.upper()} Anomaly (Vertically Integrated): Latitude × Longitude Maps with Zonal Mean with Climatology ({clim_start_yr}-{clim_end_yr}) INCOMPLETE', fontsize=30, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])  # Leave space for colorbar at bottom and title at top
    
    output_file = f"{output_dir}{variable}_anomalies_lat_lon_{start_year}-{end_year}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close()
    
    # Create time-averaged anomaly map
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    
    # Compute time mean
    anomalies_mean = anomalies.mean(dim='time')
    
    # Determine color limits for mean
    vmin_mean = np.nanpercentile(anomalies_mean.values, 1)
    vmax_mean = np.nanpercentile(anomalies_mean.values, 99)
    vlim_mean = max(abs(vmin_mean), abs(vmax_mean))
    vmin_mean, vmax_mean = -vlim_mean, vlim_mean
    
    levels = np.linspace(vmin_mean, vmax_mean, 21)
    im = ax.contourf(lon_vals, lat_vals, anomalies_mean.values,
                    levels=levels, cmap='RdBu_r', extend='both',
                    transform=ccrs.PlateCarree())
    
    ax.coastlines(resolution='110m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_xlabel('Longitude (°E)', fontsize=22)
    ax.set_ylabel('Latitude (°N)', fontsize=22)
    ax.set_title(f'{variable.upper()} Mean Anomaly (Vertically Integrated, {start_year}-{end_year})', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=40)
    cbar.set_label(f'{variable} anomaly', fontsize=12)
    
    plt.tight_layout()
    
    output_file_mean = f"{output_dir}{variable}_anomalies_lat_lon_mean_{start_year}-{end_year}.png"
    plt.savefig(output_file_mean, dpi=400, bbox_inches='tight')
    print(f"Mean anomaly figure saved to: {output_file_mean}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot latitude × longitude maps of vertically integrated AAM variable anomalies')
    parser.add_argument('variable', help='variable name (e.g. u, v, momentum)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year (default: {end_yr})')
    parser.add_argument('--clim_start', type=int, default=clim_start_yr, help=f'climatology start (default: {clim_start_yr})')
    parser.add_argument('--clim_end', type=int, default=clim_end_yr, help=f'climatology end (default: {clim_end_yr})')
    parser.add_argument('--find_min', action='store_true', help='Find minimum AAM instead of maximum')
    args = parser.parse_args()
    extremum = 'min' if args.find_min else 'max'
    plot_lat_lon_anomalies(args.start, args.end, args.variable, args.clim_start, args.clim_end, extremum)
