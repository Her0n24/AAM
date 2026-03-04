"""
This script plots latitude × longitude (north × east) maps of variable anomalies.
References plot_variable_anomalies_3d.py structure.
Intended for a single event only
"""
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import argparse
import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

base_dir = os.getcwd()
scratch_path = "/work/scratch-nopw2/hhhn2"
Variable_data_path_base = f"{scratch_path}/ERA5/monthly_mean/AAM/full/"
climatology_path_base = f"{scratch_path}/ERA5/climatology/"
output_dir = f"{base_dir}/figures/"
sigma_coeff_path = f"{base_dir}/l137_a_b.csv"
sp_path_base = f"{scratch_path}/ERA5/monthly_mean/variables/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(climatology_path_base, exist_ok=True)

# Event
start_yr, end_yr = 1997, 1998

# Climatology period
clim_start_yr, clim_end_yr = 1980, 2000


def plot_lat_lon_anomalies(start_year, end_year, variable, clim_start_yr=1980, clim_end_yr=2000, 
                           find_extremum='max', p_min=100, p_max=1000):
    """
    Plot latitude × longitude maps of anomalies for AAM variables integrated over specified pressure range.
    
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
    p_min : float
        Minimum pressure level in hPa (top of integration range)
    p_max : float
        Maximum pressure level in hPa (bottom of integration range)
    """
    print(f"\nIntegrating from {p_min} hPa to {p_max} hPa")
    
    # Load sigma coefficients
    print(f"Loading sigma coefficients from: {sigma_coeff_path}")
    sigma_df = pd.read_csv(sigma_coeff_path)
    a_mid = sigma_df['a [Pa]'].values
    b_mid = sigma_df['b'].values
    
    # Omit level 0 and level 137 to match data with 136 levels (levels 1-136)
    a_mid = a_mid[1:137]
    b_mid = b_mid[1:137]
    
    print(f"Using {len(a_mid)} sigma coefficient levels")
    
    # Load one surface pressure file to calculate pressure levels
    sp_pattern = f"{sp_path_base}ERA5_sp_{start_year}-*.nc"
    sp_files = sorted(glob.glob(sp_pattern))
    if not sp_files:
        raise FileNotFoundError(f"No surface pressure files found: {sp_pattern}")
    
    print(f"Loading surface pressure from: {sp_files[0]}")
    ds_sp = xr.open_dataset(sp_files[0])
    sp_data = ds_sp['sp'] if 'sp' in ds_sp else ds_sp['surface_pressure']
    
    # Calculate pressure at each level (in Pa)
    # p(level, lat, lon) = a(level) + b(level) * sp(lat, lon)
    pressure_levels = a_mid[:, np.newaxis, np.newaxis] + b_mid[:, np.newaxis, np.newaxis] * sp_data.values[np.newaxis, :, :]
    pressure_levels_hpa = pressure_levels / 100.0  # Convert Pa to hPa
    
    # Find levels within the specified pressure range
    # For each lat/lon, find which levels fall within [p_min, p_max]
    # Use the mean pressure across all lat/lon to determine which levels to keep
    mean_pressure_per_level = np.mean(pressure_levels_hpa, axis=(1, 2))
    levels_in_range = np.where((mean_pressure_per_level >= p_min) & (mean_pressure_per_level <= p_max))[0]
    
    if len(levels_in_range) == 0:
        raise ValueError(f"No levels found in pressure range {p_min}-{p_max} hPa")
    
    print(f"Selected {len(levels_in_range)} levels in pressure range {p_min}-{p_max} hPa")
    print(f"Level indices: {levels_in_range[0]} to {levels_in_range[-1]}")
    # Load climatology - try 3D first, then fall back to pre-integrated
    clim_file_3d = f"{climatology_path_base}ERA5_AAM_full_climatology_{clim_start_yr}-{clim_end_yr}.nc"
    clim_file_vi = f"{climatology_path_base}ERA5_{variable}_full_climatology_vi_{clim_start_yr}-{clim_end_yr}.nc"
    
    if os.path.exists(clim_file_3d):
        print(f"Loading 3D climatology from: {clim_file_3d}")
        ds_climatology = xr.open_dataset(clim_file_3d, chunks={'level': 50})
        clim_is_3d = True
    elif os.path.exists(clim_file_vi):
        print(f"Loading pre-integrated climatology from: {clim_file_vi}")
        ds_climatology = xr.open_dataset(clim_file_vi)
        clim_is_3d = False
    else:
        raise FileNotFoundError(f"No climatology file found. Tried:\n  {clim_file_3d}\n  {clim_file_vi}")
    
    # Load time series data - use full 3D files
    all_files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            pattern = f"{Variable_data_path_base}AAM_ERA5_{year}-{month:02d}_full.nc"
            month_files = glob.glob(pattern)
            if month_files:
                all_files.extend(month_files)
                print(f"Found file: {month_files[0]}")  # Print the first file found for this month
    
    if not all_files:
        raise FileNotFoundError(f"No time series files found for {variable} in {start_year}-{end_year}")
    
    all_files.sort()
    
    # Load time series with chunking for 3D data
    try:
        ds_timeseries = xr.open_mfdataset(all_files, combine='by_coords', 
                                          chunks={'time': 1, 'level': 50, 'latitude': 200, 'longitude': 400})
    except ValueError:
        print("Warning: xarray failed to decode time coordinates; loading manually")
        datasets = []
        for f in all_files:
            ds = xr.open_dataset(f, decode_times=False, 
                                chunks={'time': 1, 'level': 50, 'latitude': 200, 'longitude': 400})
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
            'momentum': ['momentum', 'angular_momentum', 'aam', 'AAM'],
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
    
    print(f"\\nTimeseries variable shape: {da_timeseries.shape}")
    print(f"Timeseries dimensions: {da_timeseries.dims}")
    print(f"Climatology variable shape: {da_climatology.shape}")
    print(f"Climatology dimensions: {da_climatology.dims}")
    
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
    
    # Integrate over pressure levels if data has level dimension
    if 'level' in da_timeseries.dims:
        print(f"\n=== INTEGRATING OVER PRESSURE LEVELS ===")
        print(f"Before integration - shape: {da_timeseries.shape}, dims: {da_timeseries.dims}")
        
        # Select levels in the specified pressure range
        da_timeseries_selected = da_timeseries.isel(level=levels_in_range)
        
        # Calculate dp for each level (pressure thickness)
        # dp is positive downward (from lower pressure to higher pressure)
        pressure_selected = pressure_levels_hpa[levels_in_range, :, :]
        
        # Calculate pressure differences between level interfaces
        # For simplicity, approximate dp as difference between adjacent midpoints
        dp = np.zeros_like(pressure_selected)
        if len(levels_in_range) == 1:
            # Single level - use full range as dp
            dp[0] = p_max - p_min
        else:
            # Multiple levels - calculate differences
            for i in range(len(levels_in_range)):
                if i == 0:
                    # Top level
                    dp[i] = (pressure_selected[i] + pressure_selected[i+1]) / 2.0 - p_min
                elif i == len(levels_in_range) - 1:
                    # Bottom level
                    dp[i] = p_max - (pressure_selected[i-1] + pressure_selected[i]) / 2.0
                else:
                    # Middle levels
                    dp[i] = (pressure_selected[i] + pressure_selected[i+1]) / 2.0 - \
                            (pressure_selected[i-1] + pressure_selected[i]) / 2.0
        
        # Convert dp to Pa for integration (100 Pa = 1 hPa)
        dp_pa = dp * 100.0
        
        # Create xarray DataArray for dp with proper dimensions
        dp_da = xr.DataArray(dp_pa, dims=['level', 'latitude', 'longitude'],
                             coords={'level': da_timeseries_selected.level,
                                   'latitude': da_timeseries_selected.latitude,
                                   'longitude': da_timeseries_selected.longitude})
        
        # Integrate: sum(AAM * dp / g) over levels
        g = 9.80665  # m/s^2
        
        print(f"Integrating over {len(levels_in_range)} levels...")
        da_timeseries_integrated = (da_timeseries_selected * dp_da / g).sum(dim='level')
        da_timeseries_integrated = da_timeseries_integrated.compute()
        
        print(f"After integration - shape: {da_timeseries_integrated.shape}, dims: {da_timeseries_integrated.dims}")
        da_timeseries = da_timeseries_integrated
    
    # Integrate climatology the same way if it's 3D
    if clim_is_3d and 'level' in da_climatology.dims:
        print(f"\nIntegrating climatology over same pressure levels...")
        da_climatology_selected = da_climatology.isel(level=levels_in_range)
        
        # Reuse the same dp calculation
        dp_da_clim = xr.DataArray(dp_pa, dims=['level', 'latitude', 'longitude'],
                                  coords={'level': da_climatology_selected.level,
                                        'latitude': da_climatology_selected.latitude,
                                        'longitude': da_climatology_selected.longitude})
        
        da_climatology_integrated = (da_climatology_selected * dp_da_clim / g).sum(dim='level')
        da_climatology_integrated = da_climatology_integrated.compute()
        da_climatology = da_climatology_integrated
        print(f"Climatology after integration - shape: {da_climatology.shape}")
    
    # Remove any remaining singleton level dimensions
    if 'level' in da_timeseries.dims:
        if len(da_timeseries.level) == 1:
            da_timeseries = da_timeseries.squeeze('level', drop=True)
            print("Removed singleton level dimension from timeseries")
    if 'level' in da_climatology.dims:
        if len(da_climatology.level) == 1:
            da_climatology = da_climatology.squeeze('level', drop=True)
            print("Removed singleton level dimension from climatology")
    
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
    vmin = np.nanpercentile(anomalies.values, 4)
    vmax = np.nanpercentile(anomalies.values, 96)
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
        profile_axes[i].set_xlim(-1 * 1e27, 1 * 1e27)  # Adjust as needed based on variable units
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
    sm = cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both', spacing='proportional')
    cbar.set_label(f'{variable} anomaly', fontsize=22)
    cbar.set_ticks(levels_cbar.tolist())
    cbar.ax.tick_params(labelsize=16)
    
    fig.suptitle(f'{variable.upper()} Anomaly ({p_min}-{p_max} hPa): Latitude × Longitude Maps with Zonal Mean\nClimatology ({clim_start_yr}-{clim_end_yr})', fontsize=30, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])  # Leave space for colorbar at bottom and title at top
    
    output_file = f"{output_dir}{variable}_anomalies_lat_lon_{start_year}-{end_year}_{p_min}-{p_max}hPa.png"
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
    ax.set_title(f'{variable.upper()} Mean Anomaly ({p_min}-{p_max} hPa, {start_year}-{end_year})', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=40)
    cbar.set_label(f'{variable} anomaly', fontsize=12)
    
    plt.tight_layout()
    
    output_file_mean = f"{output_dir}{variable}_anomalies_lat_lon_mean_{start_year}-{end_year}_{p_min}-{p_max}hPa.png"
    plt.savefig(output_file_mean, dpi=400, bbox_inches='tight')
    print(f"Mean anomaly figure saved to: {output_file_mean}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot latitude × longitude maps of AAM variable anomalies integrated over specified pressure range')
    parser.add_argument('variable', help='variable name (e.g. u, v, momentum)')
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year (default: {end_yr})')
    parser.add_argument('--clim_start', type=int, default=clim_start_yr, help=f'climatology start (default: {clim_start_yr})')
    parser.add_argument('--clim_end', type=int, default=clim_end_yr, help=f'climatology end (default: {clim_end_yr})')
    parser.add_argument('--p-min', type=float, default=100, help='minimum pressure level in hPa (top of range, default: 100)')
    parser.add_argument('--p-max', type=float, default=1000, help='maximum pressure level in hPa (bottom of range, default: 1000)')
    parser.add_argument('--find_min', action='store_true', help='Find minimum AAM instead of maximum')
    args = parser.parse_args()
    extremum = 'min' if args.find_min else 'max'
    plot_lat_lon_anomalies(args.start, args.end, args.variable, args.clim_start, args.clim_end, extremum,
                          getattr(args, 'p_min'), getattr(args, 'p_max'))
