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
from dask.distributed import Client, LocalCluster
import argparse
import glob
import gc
import re
from typing import Optional, Tuple
from pathlib import Path
import pandas as pd
import sys
import dask
sys.path.append(str(Path(__file__).resolve().parents[1]))
from event_composite_lat_time import _infer_latitude_band_width_deg

import event_composite_all as composite_core
from plotting_utils import compute_monthly_climatology, plot_lat_lon_snapshots, plot_latitude_level_snapshots_HadGEN3, compute_active_month_percent, add_active_month_percent_labels
from scipy import stats as _stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

base_dir = os.getcwd()
scratch_path = "/work/scratch-nopw2/hhhn2"
Variable_data_path_base = f"{scratch_path}/ERA5/monthly_mean/AAM/full/"
climatology_path_base = f"{scratch_path}/ERA5/climatology/"
output_dir = f"{base_dir}/composite_non_tracking/"
composite_output_dir = f"{base_dir}/composite_non_tracking/"
sigma_coeff_path = f"{base_dir}/l137_a_b.csv"
sp_path_base = f"{scratch_path}/ERA5/monthly_mean/variables/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(climatology_path_base, exist_ok=True)
os.makedirs(composite_output_dir, exist_ok=True)
sys.path.append(str(Path(__file__).resolve().parents[1] / "CMIP6_HadGEM3_GC31"))
from CMIP6_HadGEM3_GC31.utilities import (
    _to_per_latitude_band,
    vertical_sum_over_pressure_range, REGION_BOUNDS
)

# Ensure REGION_BOUNDS is defined even if import timing/order differs when
# this module is imported from other scripts.
if 'REGION_BOUNDS' not in globals():
    try:
        from CMIP6_HadGEM3_GC31.utilities import REGION_BOUNDS as _REGION_BOUNDS_FALLBACK
        REGION_BOUNDS = _REGION_BOUNDS_FALLBACK
    except Exception:
        try:
            from CMIP6_HadGEM3_GC31.utilities import REGION_BOUNDS as _REGION_BOUNDS_FALLBACK2
            REGION_BOUNDS = _REGION_BOUNDS_FALLBACK2
        except Exception:
            # Minimal safe fallback covering full globe to avoid hard failures;
            # downstream code will still work but region selection becomes no-op.
            REGION_BOUNDS = {"all": (-180.0, 180.0)}
            raise ImportError("Could not import REGION_BOUNDS from expected modules. Using fallback covering full globe, but this may cause issues if region selection is used.")

# Climatology period
clim_start_yr, clim_end_yr = 1981, 2010

def _precompute_reduced_fields(
    aam_anomalies: xr.DataArray,
    u_anomalies: xr.DataArray,
    args: argparse.Namespace,
    level_pressure_hpa: xr.DataArray,
):
    """
    Compute all expensive reductions once for the full time series
    instead of once per ENSO event.
    """

    n_mid_levels = int(level_pressure_hpa.sizes["level"])

    if int(aam_anomalies.sizes.get("level", 0)) > n_mid_levels:
        aam_anomalies = aam_anomalies.isel(level=slice(0, n_mid_levels))

    if int(u_anomalies.sizes.get("level", 0)) > n_mid_levels:
        u_anomalies = u_anomalies.isel(level=slice(0, n_mid_levels))

    level_pressure_vals = level_pressure_hpa.values[:n_mid_levels]

    aam_anomalies = aam_anomalies.assign_coords(
        level=(("level",), level_pressure_vals)
    )

    u_anomalies = u_anomalies.assign_coords(
        level=(("level",), level_pressure_vals)
    )

    overlay_pressure_hpa = 250.0

    u_overlay_idx = int(
        np.nanargmin(
            np.abs(level_pressure_vals - overlay_pressure_hpa)
        )
    )

    print("Precomputing full-period reduced products...")

    aam_lat_lon_all = vertical_sum_over_pressure_range(
        aam_anomalies,
        p_min_hpa=args.p_min,
        p_max_hpa=args.p_max,
        level_dim="level",
    )
    
    lon_spacing_deg = np.abs(aam_anomalies['longitude'].values[1] - aam_anomalies['longitude'].values[0])
    
    # 2. Convert that single width to radians (dλ)
    dlon_rad = np.radians(lon_spacing_deg)
    
    aam_height_lat_all = aam_anomalies.sum("longitude", skipna=True) * dlon_rad

    u_height_lat_all = u_anomalies.mean(
        "longitude",
        skipna=True,
    )

    u_lat_lon_all = u_anomalies.isel(
        level=u_overlay_idx
    )

    return (
        aam_lat_lon_all,
        aam_height_lat_all,
        u_lat_lon_all,
        u_height_lat_all,
    )

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
        ds_climatology = xr.open_dataset(clim_file_3d)
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
    
    # Load time series eagerly so the script can use available RAM instead of dask chunks.
    ds_timeseries = xr.open_mfdataset(
    all_files,
    combine="nested",
    concat_dim="time",
    decode_times=False,
    coords="minimal",
    compat="override",
    chunks={
        "time": -1,         # -1 means: do not split time within an individual file
        "level": 5,         # Process 5 vertical pressure levels at a time
        "latitude": -1,     # Keep the full horizontal map intact
        "longitude": -1    # Keep the full horizontal map intact
    }
    )
    if 'time' in ds_timeseries.dims and 'time' not in ds_timeseries.coords:
        ds_timeseries = ds_timeseries.set_coords('time')
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
    plt.savefig(output_file, dpi=500, bbox_inches='tight')
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


def parse_composite_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ENSO event composites for lat-lon and height-lat AAM anomaly plots."
    )
    parser.add_argument("--p-min", type=float, default=150.0, help="Minimum pressure level in hPa.")
    parser.add_argument("--p-max", type=float, default=700, help="Maximum pressure level in hPa.")
    parser.add_argument("--start-year", type=int, default=1979, help="First onset year to consider.")
    parser.add_argument("--end-year", type=int, default=2019, help="Last onset year to consider.")
    parser.add_argument("--clim-start-year", type=int, default=1981, help="Climatology start year.")
    parser.add_argument("--clim-end-year", type=int, default=2010, help="Climatology end year.")
    parser.add_argument(
        "--enso-state",
        choices=("el_nino", "la_nina"),
        default="el_nino",
        help="ENSO state to detect.",
    )
    parser.add_argument(
        "--nino-threshold",
        type=float,
        default=None,
        help="Nino3.4 threshold. Defaults to +0.5 for El Nino and -0.5 for La Nina.",
    )
    parser.add_argument(
        "--min-enso-months",
        "--min-elnino-months",
        dest="min_enso_months",
        type=int,
        default=3,
        help="Minimum consecutive threshold months defining an event onset.",
    )
    parser.add_argument(
        "--allow-reinitiation",
        action="store_true",
        help="Allow a later threshold crossing in the 12 months after an El Nino run ends.",
    )
    parser.add_argument(
        "--reinitiation-check-months",
        type=int,
        default=12,
        help="Months after an El Nino run to check for reinitiation when reinitiation is disallowed.",
    )
    parser.add_argument(
        "--onset-season",
        choices=("all", "ndjfm"),
        default="all",
        help="Restrict event onsets to all months or NDJFM.",
    )
    parser.add_argument(
        "--composite-months",
        type=int,
        default=24,
        help="Number of relative months in each event window.",
    )
    parser.add_argument(
        "--composite-start",
        choices=("onset", "december_onset_year"),
        default="onset",
        help="Start composite at onset month or December of onset year.",
    )
    parser.add_argument(
        "--rolling-period",
        type=int,
        default=1,
        help="Circular rolling mean window along relative month before compositing.",
    )
    parser.add_argument(
        "--region",
        choices=tuple(REGION_BOUNDS.keys()),
        default="all",
        help="Longitude sector used before zonal integration.",
    )
    parser.add_argument(
        "--save-event-stack",
        action="store_true",
        help="Also save the full event stack used in the composite.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Load existing composite NetCDF and re-generate plots (no recomputation).",
    )
    return parser.parse_args()


def _load_era5_full_field_lazy(args: argparse.Namespace) -> xr.DataArray:
    """Load full-field ERA5 AAM lazily to avoid eager in-memory spikes."""
    return _load_era5_full_field_lazy_for_years(
        args,
        start_year=args.start_year,
        end_year=args.end_year + int(np.ceil(args.composite_months / 12.0)) + 1,
    )


def _load_era5_full_field_lazy_for_years(
    args: argparse.Namespace,
    *,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """Load full-field ERA5 AAM lazily for an explicit year span."""
    files = []
    for year in range(start_year, end_year + 1):
        files.extend(sorted(glob.glob(str(composite_core.AAM_DATA_DIR / f"AAM_ERA5_{year}-*_full.nc"))))
    if not files:
        raise FileNotFoundError(f"No ERA5 AAM files found in fixed directory: {composite_core.AAM_DATA_DIR}")

    ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="time",
    coords="minimal",
    compat="override",
    chunks={
        "time": -1,         # -1 means: do not split time within an individual file
        "level": 5,         # Process 5 vertical pressure levels at a time
        "latitude": -1,     # Keep the full horizontal map intact
        "longitude": -1    # Keep the full horizontal map intact
    }
    ).sortby("time")
    da = composite_core._first_data_var(ds, ("AAM", "aam", "angular_momentum", "momentum", "AAMA"))
    da = composite_core._standardize_dims(da)
    if "time" not in da.dims:
        raise ValueError("Full-field ERA5 data must include a time dimension.")

    da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    return da


def _load_era5_variable_full_field_lazy(variable: str, args: argparse.Namespace) -> xr.DataArray:
    """Load a full-field ERA5 monthly variable lazily from monthly_mean/variables."""
    return _load_era5_variable_full_field_lazy_for_years(
        variable,
        args,
        start_year=args.start_year,
        end_year=args.end_year + int(np.ceil(args.composite_months / 12.0)) + 1,
    )


def _load_era5_variable_full_field_lazy_for_years(
    variable: str,
    args: argparse.Namespace,
    *,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """Load a full-field ERA5 monthly variable lazily for an explicit year span."""
    files = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            files.extend(sorted(glob.glob(f"{sp_path_base}ERA5_{variable}_{year}-{month:02d}.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 {variable} monthly files found in {sp_path_base}")

    ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="time",
    coords="minimal",
    compat="override",
    parallel=True,
    chunks="auto",
)
    da = composite_core._first_data_var(ds, (variable, "u", "eastward_wind", "u_component_of_wind", "AAM", "aam"))
    da = composite_core._standardize_dims(da)
    if "time" not in da.dims:
        raise ValueError(f"ERA5 {variable} data must include a time dimension.")

    da = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    
    return da

def _load_era5_full_field_climatology(args: argparse.Namespace) -> xr.DataArray:
    """Compute the ERA5 AAM monthly climatology on the fly from the full field."""
    clim_da = _load_era5_full_field_lazy_for_years(
        args,
        start_year=args.clim_start_year,
        end_year=args.clim_end_year,
    )
    clim = compute_monthly_climatology(clim_da, args.clim_start_year, args.clim_end_year)
    return clim


def _load_era5_variable_climatology(variable: str, args: argparse.Namespace) -> xr.DataArray:
    """Compute a variable-specific ERA5 monthly climatology on the fly."""
    clim_da = _load_era5_variable_full_field_lazy_for_years(
        variable,
        args,
        start_year=args.clim_start_year,
        end_year=args.clim_end_year,
    )
    return compute_monthly_climatology(clim_da, args.clim_start_year, args.clim_end_year)


def _load_mean_pressure_profile_hpa(args: argparse.Namespace) -> xr.DataArray:
    """Return a 1D mean pressure profile in hPa for the ERA5 hybrid levels."""
    sigma_df = pd.read_csv(sigma_coeff_path)
    a_mid = sigma_df["a [Pa]"].values[1:137]
    b_mid = sigma_df["b"].values[1:137]

    sp_pattern = f"{sp_path_base}ERA5_sp_{args.start_year}-*.nc"
    sp_files = sorted(glob.glob(sp_pattern))
    if not sp_files:
        raise FileNotFoundError(f"No surface pressure files found: {sp_pattern}")

    with xr.open_dataset(sp_files[0]) as ds_sp:
        sp_data = ds_sp["sp"] if "sp" in ds_sp else ds_sp["surface_pressure"]
        pressure_levels = a_mid[:, np.newaxis, np.newaxis] + b_mid[:, np.newaxis, np.newaxis] * sp_data.values[np.newaxis, :, :]

    pressure_levels_hpa = pressure_levels / 100.0
    mean_pressure_per_level = np.mean(pressure_levels_hpa, axis=(1, 2))
    return xr.DataArray(
        mean_pressure_per_level,
        dims=("level",),
        coords={"level": mean_pressure_per_level},
        name="level_pressure_hpa",
        attrs={"units": "hPa"},
    )


def _build_full_field_event_stack(
    aam_da: xr.DataArray,
    clim_da: xr.DataArray,
    date_list: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[xr.DataArray, list[str]]:
    aam = composite_core._standardize_dims(aam_da)
    if clim_da is None:
        anomalies = aam
        raise ValueError("Climatology DataArray is required to compute anomalies for compositing.")
    else:
        clim = composite_core._standardize_dims(clim_da)
        if "month" not in clim.dims and "month" not in clim.coords:
            raise ValueError("Climatology must have a monthly 'month' dimension or coord.")
        clim_months = clim["month"].values
        if np.size(clim_months) and np.nanmin(clim_months) == 0 and np.nanmax(clim_months) == 11:
            clim = clim.assign_coords(month=clim["month"] + 1)
        if args.region != "all" and "longitude" in aam.dims:
            aam = composite_core._select_region(aam, args.region)
            clim = composite_core._select_region(clim, args.region)
        anomalies = aam.groupby("time.month") - clim

    if "time" not in anomalies.dims:
        raise ValueError("AAM anomalies must have a time dimension.")

    stacked = []
    event_labels = []
    seen_onsets = set()
    for onset_str, _ in date_list:
        onset_ym = onset_str[:7]
        if onset_ym in seen_onsets:
            continue
        seen_onsets.add(onset_ym)

        window_start, window_end = composite_core._compute_composite_window_from_onset(
            onset_str,
            composite_months=args.composite_months,
            composite_start=args.composite_start,
        )
        event = anomalies.sel(time=slice(window_start, window_end))
        n_avail = int(event.sizes.get("time", 0))
        if n_avail < args.composite_months:
            print(f"Skipping onset {onset_ym}: only {n_avail} months available in composite window.")
            continue

        event = event.isel(time=slice(0, args.composite_months))
        event = event.assign_coords(time=np.arange(1, args.composite_months + 1, dtype=int))
        if "month" in event.coords:
            event = event.drop_vars("month")
        event = event.rename({"time": "month"})
        stacked.append(event)
        event_labels.append(onset_ym)

    if not stacked:
        raise RuntimeError("No complete ERA5 event windows were available for compositing.")

    event_stack = xr.concat(stacked, dim=pd.Index(event_labels, name="event"))
    event_stack = composite_core._circular_rolling_mean(event_stack, dim="month", window=args.rolling_period)
    event_stack.name = "AAM_event_stack"
    return event_stack, event_labels


def _attach_month_time_coord(da: xr.DataArray) -> xr.DataArray:
    if "month" in da.dims:
        da = da.rename({"month": "time"})
    if "time" in da.dims:
        da = da.assign_coords(time=np.arange(1, int(da.sizes["time"]) + 1, dtype=int))
    return da


def _event_significance_mask(
    stack: xr.DataArray,
    spatial_dims: tuple[str, ...],
) -> xr.DataArray:
    # Compute month-by-month to avoid materializing the full (event, month, ...) array in RAM.
    n_month = int(stack.sizes["month"])
    out_shape = tuple(int(stack.sizes[d]) for d in spatial_dims)
    sig = np.zeros((n_month, *out_shape), dtype=bool)

    for m in range(n_month):
        slab = stack.isel(month=m).transpose("event", *spatial_dims).astype(np.float64)
        slab_np = slab.values
        _, p_vals = _stats.ttest_1samp(slab_np, 0.0, axis=0, nan_policy="omit")
        p_vals = np.asarray(p_vals, dtype=np.float64)
        sig[m, ...] = p_vals < 0.05

    dims = ("time",) + spatial_dims
    coords = {"time": np.arange(1, n_month + 1, dtype=int)}
    for dim in spatial_dims:
        coords[dim] = np.asarray(stack[dim].values)
    return xr.DataArray(sig, dims=dims, coords=coords, name="significant_mask")


def _prepare_monthly_anomalies(
    da: xr.DataArray,
    clim_da: xr.DataArray,
    args: argparse.Namespace,
    variable_name: str,
) -> xr.DataArray:
    data = composite_core._standardize_dims(da)
    clim = composite_core._standardize_dims(clim_da)
    if "month" not in clim.dims and "month" not in clim.coords:
        raise ValueError(f"{variable_name} climatology must have a monthly 'month' dimension or coord.")
    clim_months = clim["month"].values
    if np.size(clim_months) and np.nanmin(clim_months) == 0 and np.nanmax(clim_months) == 11:
        clim = clim.assign_coords(month=clim["month"] + 1)
    if args.region != "all" and "longitude" in data.dims:
        data = composite_core._select_region(data, args.region)
        clim = composite_core._select_region(clim, args.region)

    anomalies = data.groupby("time.month") - clim
    if "time" not in anomalies.dims:
        raise ValueError(f"{variable_name} anomalies must have a time dimension.")
    return anomalies


def _event_window_from_anomalies(
    anomalies: xr.DataArray,
    onset_str: str,
    args: argparse.Namespace,
) -> Optional[xr.DataArray]:
    window_start, window_end = composite_core._compute_composite_window_from_onset(
        onset_str,
        composite_months=args.composite_months,
        composite_start=args.composite_start,
    )
    event = anomalies.sel(time=slice(window_start, window_end))
    n_avail = int(event.sizes.get("time", 0))
    if n_avail < args.composite_months:
        return None
    event = event.isel(time=slice(0, args.composite_months))
    event = event.assign_coords(time=np.arange(1, args.composite_months + 1, dtype=int))
    if "month" in event.coords:
        event = event.drop_vars("month")
    return event.rename({"time": "month"})


def _accumulate_mean(target_sum: Optional[np.ndarray], target_count: int, values: np.ndarray) -> tuple[np.ndarray, int]:
    if target_sum is None:
        target_sum = np.zeros_like(values, dtype=np.float64)
    target_sum += np.asarray(values, dtype=np.float64)
    return target_sum, target_count + 1


def _finalize_mean(total: np.ndarray, count: int, template: xr.DataArray) -> xr.DataArray:
    if count <= 0:
        raise RuntimeError("Cannot finalize an empty composite.")
    mean_values = (total / float(count)).astype(np.float64)
    return xr.DataArray(mean_values, dims=template.dims, coords=template.coords, attrs=template.attrs, name=template.name)


def _plotting_utils_file_suffix(filename_suffix: str, dec_onset_month: str, onset_season_ndjfm: str) -> str:
    file_suffix = ""
    if filename_suffix:
        file_suffix = f"_{str(filename_suffix).strip('_')}"
    if dec_onset_month == "december_onset_year":
        file_suffix += "_december_onset_year"
    if onset_season_ndjfm == "ndjfm":
        file_suffix += "_onset_season_NDJFM"
    else:
        file_suffix += "_onset_season_all"
    return file_suffix


def _composite_result_path(args: argparse.Namespace, output_dir: str) -> str:
    tag = (
        f"ERA5_event_composite_{args.enso_state}_{args.start_year}-{args.end_year}"
        f"_{args.p_min}-{args.p_max}hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}"
    )
    return os.path.join(output_dir, f"{tag}.nc")


def _composite_cache_is_valid(result_path: str) -> bool:
    if not os.path.exists(result_path):
        return False

    try:
        with xr.open_dataset(result_path) as ds:
            required_vars = {"lat_lon_composite", "height_lat_composite", "u_lat_lon_composite", "u_height_lat_composite"}
            return ds.attrs.get("composite_format") == "pressure_hpa_v2" and required_vars.issubset(ds.data_vars)
    except Exception:
        return False


def _build_streamed_composites(
    date_list: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[
    xr.DataArray,
    Optional[xr.DataArray],
    xr.DataArray,
    Optional[xr.DataArray],
    xr.DataArray,
    xr.DataArray,
    list[str],
]:
    level_pressure_hpa = _load_mean_pressure_profile_hpa(args)

    aam_da = _load_era5_full_field_lazy(args)
    aam_clim = _load_era5_full_field_climatology(args)

    u_da = _load_era5_variable_full_field_lazy("u", args)
    u_clim = _load_era5_variable_climatology("u", args)

    aam_anomalies = _prepare_monthly_anomalies(
        aam_da,
        aam_clim,
        args,
        "AAM",
    )

    u_anomalies = _prepare_monthly_anomalies(
        u_da,
        u_clim,
        args,
        "u",
    )

    (
        aam_lat_lon_all,
        aam_height_lat_all,
        u_lat_lon_all,
        u_height_lat_all,
    ) = _precompute_reduced_fields(
        aam_anomalies,
        u_anomalies,
        args,
        level_pressure_hpa,
    )
    
    # 🚀 FIX 1: COMPUTE INTO RAM ONCE
    # This prevents the script from re-reading and re-calculating the 
    # 40-year dataset for every single event in the loop below.
    print("Computing full time-series into RAM (this will take a few minutes)...")
    aam_lat_lon_all = aam_lat_lon_all.compute()
    aam_height_lat_all = aam_height_lat_all.compute()
    u_lat_lon_all = u_lat_lon_all.compute()
    u_height_lat_all = u_height_lat_all.compute()
    print("Done computing! Moving to event processing...")

    lat_lon_sum = None
    lat_lon_sum = None
    height_lat_sum = None
    u_lat_lon_sum = None
    u_height_lat_sum = None
    event_labels: list[str] = []
    event_count = 0

    lat_lon_samples: list[list[np.ndarray]] = []
    height_lat_samples: list[list[np.ndarray]] = []
    u_lat_lon_samples: list[list[np.ndarray]] = []
    u_height_lat_samples: list[list[np.ndarray]] = []

    for onset_str, _ in date_list:
        onset_ym = onset_str[:7]
        if onset_ym in event_labels:
            continue

        aam_event = _event_window_from_anomalies(aam_anomalies, onset_str, args)
        u_event = _event_window_from_anomalies(u_anomalies, onset_str, args)
        if aam_event is None:
            print(f"Skipping onset {onset_ym}: incomplete AAM window.")
            continue
        if u_event is None:
            print(f"Skipping onset {onset_ym}: incomplete u window.")
            continue

        aam_event = composite_core._circular_rolling_mean(aam_event, dim="month", window=args.rolling_period)
        u_event = composite_core._circular_rolling_mean(u_event, dim="month", window=args.rolling_period)

        aam_lat_lon_event = _event_window_from_anomalies(
            aam_lat_lon_all,
            onset_str,
            args,
        )

        aam_height_lat_event = _event_window_from_anomalies(
            aam_height_lat_all,
            onset_str,
            args,
        )

        u_lat_lon_event = _event_window_from_anomalies(
            u_lat_lon_all,
            onset_str,
            args,
        )

        u_height_lat_event = _event_window_from_anomalies(
            u_height_lat_all,
            onset_str,
            args,
        )

        if aam_lat_lon_event is None:
            continue

        if aam_height_lat_event is None:
            continue

        if u_lat_lon_event is None:
            continue

        if u_height_lat_event is None:
            continue

        aam_lat_lon_event = aam_lat_lon_event.transpose(
            "month",
            "latitude",
            "longitude",
        )

        aam_height_lat_event = aam_height_lat_event.transpose(
            "month",
            "level",
            "latitude",
        )

        u_lat_lon_event = u_lat_lon_event.transpose(
            "month",
            "latitude",
            "longitude",
        )

        u_height_lat_event = u_height_lat_event.transpose(
            "month",
            "level",
            "latitude",
        )
        
        if lat_lon_sum is None:
            lat_lon_sum = np.zeros_like(np.asarray(aam_lat_lon_event.values, dtype=np.float64))
            height_lat_sum = np.zeros_like(np.asarray(aam_height_lat_event.values, dtype=np.float64))
            u_lat_lon_sum = np.zeros_like(np.asarray(u_lat_lon_event.values, dtype=np.float64))
            u_height_lat_sum = np.zeros_like(np.asarray(u_height_lat_event.values, dtype=np.float64))
            if args.compute_significance:
                lat_lon_samples = [[] for _ in range(int(aam_lat_lon_event.sizes["month"]))]
                height_lat_samples = [[] for _ in range(int(aam_height_lat_event.sizes["month"]))]
                u_lat_lon_samples = [[] for _ in range(int(u_lat_lon_event.sizes["month"]))]
                u_height_lat_samples = [[] for _ in range(int(u_height_lat_event.sizes["month"]))]

        lat_lon_sum, event_count = _accumulate_mean(lat_lon_sum, event_count, aam_lat_lon_event.values)
        height_lat_sum, _ = _accumulate_mean(height_lat_sum, event_count - 1, aam_height_lat_event.values)
        u_lat_lon_sum, _ = _accumulate_mean(u_lat_lon_sum, event_count - 1, u_lat_lon_event.values)
        u_height_lat_sum, _ = _accumulate_mean(u_height_lat_sum, event_count - 1, u_height_lat_event.values)

        if args.compute_significance:
            for idx in range(aam_lat_lon_event.sizes["month"]):
                lat_lon_samples[idx].append(aam_lat_lon_event.isel(month=idx).values)
                height_lat_samples[idx].append(aam_height_lat_event.isel(month=idx).values)
                u_lat_lon_samples[idx].append(u_lat_lon_event.isel(month=idx).values)
                u_height_lat_samples[idx].append(u_height_lat_event.isel(month=idx).values)

        event_labels.append(onset_ym)

    if event_count == 0:
        raise RuntimeError("No complete ERA5 event windows were available for compositing.")

    assert lat_lon_sum is not None
    assert height_lat_sum is not None
    assert u_lat_lon_sum is not None
    assert u_height_lat_sum is not None

    lat_lon_composite = _finalize_mean(lat_lon_sum, event_count, aam_lat_lon_event)
    height_lat_composite = _finalize_mean(height_lat_sum, event_count, aam_height_lat_event)
    u_lat_lon_composite = _finalize_mean(u_lat_lon_sum, event_count, u_lat_lon_event)
    u_height_lat_composite = _finalize_mean(u_height_lat_sum, event_count, u_height_lat_event)

    lat_lon_composite = lat_lon_composite.rename({"month": "time"})
    height_lat_composite = height_lat_composite.rename({"month": "time"})
    u_lat_lon_composite = u_lat_lon_composite.rename({"month": "time"})
    u_height_lat_composite = u_height_lat_composite.rename({"month": "time"})

    lat_lon_composite = _attach_month_time_coord(lat_lon_composite)
    height_lat_composite = _attach_month_time_coord(height_lat_composite)
    u_lat_lon_composite = _attach_month_time_coord(u_lat_lon_composite)
    u_height_lat_composite = _attach_month_time_coord(u_height_lat_composite)

    lat_lon_sig = None
    height_lat_sig = None
    u_lat_lon_sig = None
    u_height_lat_sig = None
    if args.compute_significance:
        lat_lon_sample_array = np.stack([np.stack(samples, axis=0) for samples in lat_lon_samples], axis=0)
        height_lat_sample_array = np.stack([np.stack(samples, axis=0) for samples in height_lat_samples], axis=0)
        u_lat_lon_sample_array = np.stack([np.stack(samples, axis=0) for samples in u_lat_lon_samples], axis=0)
        u_height_lat_sample_array = np.stack([np.stack(samples, axis=0) for samples in u_height_lat_samples], axis=0)

        lat_lon_sig = _event_significance_mask(
            xr.DataArray(
                lat_lon_sample_array,
                dims=("month", "event", "latitude", "longitude"),
                coords={
                    "month": np.arange(1, len(lat_lon_samples) + 1),
                    "latitude": lat_lon_composite.latitude,
                    "longitude": lat_lon_composite.longitude,
                },
            ),
            ("latitude", "longitude"),
        ).transpose("time", "latitude", "longitude")
        
        height_lat_sig = _event_significance_mask(
            xr.DataArray(
                height_lat_sample_array,
                dims=("month", "event", "level", "latitude"),
                coords={
                    "month": np.arange(1, len(height_lat_samples) + 1),
                    "level": height_lat_composite.level,
                    "latitude": height_lat_composite.latitude,
                },
            ),
            ("level", "latitude"),
        ).transpose("time", "level", "latitude")
        
        u_lat_lon_sig = _event_significance_mask(
            xr.DataArray(
                u_lat_lon_sample_array,
                dims=("month", "event", "latitude", "longitude"),
                coords={
                    "month": np.arange(1, len(u_lat_lon_samples) + 1),
                    "latitude": u_lat_lon_composite.latitude,
                    "longitude": u_lat_lon_composite.longitude,
                },
            ),
            ("latitude", "longitude"),
        ).transpose("time", "latitude", "longitude")
        
        u_height_lat_sig = _event_significance_mask(
            xr.DataArray(
                u_height_lat_sample_array,
                dims=("month", "event", "level", "latitude"),
                coords={
                    "month": np.arange(1, len(u_height_lat_samples) + 1),
                    "level": u_height_lat_composite.level,
                    "latitude": u_height_lat_composite.latitude,
                },
            ),
            ("level", "latitude"),
        ).transpose("time", "level", "latitude")

    return (
        lat_lon_composite,
        lat_lon_sig,
        height_lat_composite,
        height_lat_sig,
        u_lat_lon_composite,
        u_height_lat_composite,
        event_labels,
    )


def save_composite_results(
    event_count: int,
    lat_lon_composite: xr.DataArray,
    lat_lon_significance_mask: Optional[xr.DataArray],
    height_lat_composite: xr.DataArray,
    height_lat_significance_mask: Optional[xr.DataArray],
    u_lat_lon_composite: Optional[xr.DataArray],
    u_height_lat_composite: Optional[xr.DataArray],
    event_labels: list[str],
    args: argparse.Namespace,
    output_dir: str,
) -> tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    tag = (
        f"ERA5_event_composite_{args.enso_state}_{args.start_year}-{args.end_year}"
        f"_{args.p_min}-{args.p_max}hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}"
    )
    result_path = os.path.join(output_dir, f"{tag}.nc")

    data_vars = {
        "lat_lon_composite": lat_lon_composite,
        "height_lat_composite": height_lat_composite,
    }
    if u_lat_lon_composite is not None:
        data_vars["u_lat_lon_composite"] = u_lat_lon_composite
    if u_height_lat_composite is not None:
        data_vars["u_height_lat_composite"] = u_height_lat_composite
    if lat_lon_significance_mask is not None:
        data_vars["lat_lon_significance_mask"] = lat_lon_significance_mask.astype(bool)
    if height_lat_significance_mask is not None:
        data_vars["height_lat_significance_mask"] = height_lat_significance_mask.astype(bool)
    ds = xr.Dataset(data_vars)
    ds.attrs.update(
        {
            "event_onsets": ",".join(event_labels),
            "n_events": int(event_count),
            "enso_state": args.enso_state,
            "onset_season": args.onset_season,
            "composite_start": args.composite_start,
            "rolling_period": int(args.rolling_period),
            "region": args.region,
            "pressure_range_hpa": f"{args.p_min}-{args.p_max}",
            "u_overlay_pressure_hpa": 250.0,
            "composite_format": "pressure_hpa_v2",
        }
    )
    ds.to_netcdf(result_path)

    stack_path = None
    if args.save_event_stack:
        print("Warning: --save-event-stack is not supported in streamed mode; skipping stack save.")

    return result_path, stack_path


def _plot_composite_outputs(
    lat_lon_composite: xr.DataArray,
    lat_lon_mask: Optional[xr.DataArray],
    lat_time_plot: str,
    height_lat_composite: xr.DataArray,
    height_lat_mask: Optional[xr.DataArray],
    u_lat_lon_composite: Optional[xr.DataArray],
    u_height_lat_composite: Optional[xr.DataArray],
    args: argparse.Namespace,
    event_count: int,
    composite_output_dir: str,
    lat_lon_plot: str,
    height_lat_plot: str,
    active_pct: np.ndarray,
) -> None:
    
    try:
        _plot_lat_time_composite(
            lat_lon_composite,
            lat_lon_mask,
            active_pct,
            args,
            lat_time_plot,
            event_count,
        )
    except Exception as exc:
        print(f"Warning: lat-time plotting failed after saving composites: {exc}")

    try:
        plot_lat_lon_snapshots(
            lat_lon_composite,
            zonal_wind_da=u_lat_lon_composite,
            significance_mask=None if lat_lon_mask is None else lat_lon_mask.values,
            ensemble_member="ERA5_EVENT_COMPOSITE",
            start_year=args.start_year,
            end_year=args.end_year,
            clim_start_yr=args.clim_start_year,
            clim_end_yr=args.clim_end_year,
            output_dir=composite_output_dir,
            title_suffix=f"Event-mean {args.enso_state} composite | {event_count} events",
            rolling_period=args.rolling_period,
            filename_suffix=f"_event_mean_{args.enso_state}_{args.region}",
            dec_onset_month=args.composite_start,
            onset_season_ndjfm=args.onset_season,
            pmin=args.p_min,
            pmax=args.p_max,
            nino_threshold=args.nino_threshold,
            region=args.region,
        )
    except Exception as exc:
        print(f"Warning: lat-lon plotting failed after saving composites: {exc}")

    if args.region != "all":
        try:
            subregion_mean_plot = os.path.join(
                composite_output_dir,
                f"AAM_anomalies_lat_lon_snapshots_ERA5_EVENT_MEAN_SUBREGION_{args.start_year}-{args.end_year}_{args.p_min:.1f}-{args.p_max:.1f}hPa_{args.region}{_plotting_utils_file_suffix(f'_event_mean_{args.enso_state}_{args.region}', args.composite_start, args.onset_season)}.png",
            )
            plot_lat_lon_snapshots(
                lat_lon_composite,
                zonal_wind_da=u_lat_lon_composite,
                significance_mask=None if lat_lon_mask is None else lat_lon_mask.values,
                ensemble_member="ERA5_EVENT_COMPOSITE",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=args.clim_start_year,
                clim_end_yr=args.clim_end_year,
                output_dir=composite_output_dir,
                title_suffix=f"Subregion event-mean {args.enso_state} composite | {event_count} events",
                rolling_period=args.rolling_period,
                filename_suffix=f"_event_mean_subregion_{args.enso_state}_{args.region}",
                dec_onset_month=args.composite_start,
                onset_season_ndjfm=args.onset_season,
                pmin=args.p_min,
                pmax=args.p_max,
                nino_threshold=args.nino_threshold,
                region=args.region,
            )
            print(f"Saved subregion event-mean figure to {subregion_mean_plot}")
        except Exception as exc:
            print(f"Warning: subregion event-mean plotting failed after saving composites: {exc}")

    try:
        plot_latitude_level_snapshots_HadGEN3(
            height_lat_composite,
            zonal_wind_da=u_height_lat_composite,
            significance_mask=None if height_lat_mask is None else height_lat_mask.values,
            ensemble_member="ERA5_EVENT_COMPOSITE",
            start_year=args.start_year,
            end_year=args.end_year,
            clim_start_yr=args.clim_start_year,
            clim_end_yr=args.clim_end_year,
            output_dir=composite_output_dir,
            title_suffix=f"Event-mean {args.enso_state} composite | {event_count} events",
            rolling_period=args.rolling_period,
            filename_suffix=f"_event_mean_{args.enso_state}_{args.region}",
            dec_onset_month=args.composite_start,
            onset_season_ndjfm=args.onset_season,
            nino_threshold=args.nino_threshold,
        )
    except Exception as exc:
        print(f"Warning: height-lat plotting failed after saving composites: {exc}")

    print(f"Saved lat-lon figure to {lat_lon_plot}")
    print(f"Saved lat-time figure to {lat_time_plot}")
    print(f"Saved height-lat figure to {height_lat_plot}")


def _plot_lat_time_composite(
    lat_lon_composite: xr.DataArray,
    lat_lon_mask: Optional[xr.DataArray],
    active_pct: np.ndarray,
    args: argparse.Namespace,
    output_path: str,
    event_count: int,
) -> str:
    if "longitude" not in lat_lon_composite.dims:
        raise ValueError("lat_lon_composite must contain a longitude dimension for lat-time plotting")

    # 1. Convert longitude to radians and integrate (calculates total momentum around the earth)
    lon_rad = np.deg2rad(lat_lon_composite['longitude'].astype(float))
    comp = lat_lon_composite.assign_coords(longitude=lon_rad).sortby('longitude')
    comp = comp.integrate('longitude')

    # 2. Get the latitude band width in radians
    dphi_deg = _infer_latitude_band_width_deg(comp)
    dphi_rad = np.radians(dphi_deg)

    # 3. Multiply to get the absolute total AAM mass per latitude band
    comp = comp * dphi_rad

    # 4. Transpose for the plot
    comp = comp.transpose("latitude", "time")
    
    vals = comp.values
    lat_vals = comp["latitude"].values
    time_vals = comp["time"].values
    
    dphi_deg = _infer_latitude_band_width_deg(comp)

    vmax = float(np.nanpercentile(np.abs(vals), 99))
    vmin = -vmax
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    levels = np.linspace(-vmax, vmax, 13)

    _abs = max(abs(vmin), abs(vmax))
    order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
    factor = 10 ** order
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.30)
    cf = ax.contourf(time_vals, lat_vals, vals, levels=levels, cmap="RdBu_r", extend="both")

    sig_mask = None
    if lat_lon_mask is not None:
        sig_mask = lat_lon_mask.any("longitude") if "longitude" in lat_lon_mask.dims else lat_lon_mask
        sig_mask = sig_mask.transpose("latitude", "time")
        insig = np.where(sig_mask.values, 0, 1)
        if np.any(insig == 1):
            hatches = ax.contourf(
                time_vals,
                lat_vals,
                insig,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["//"],
                zorder=10,
            )
            for collection in getattr(hatches, "collections", []):
                collection.set_facecolor("none")
                collection.set_edgecolor((0.4, 0.4, 0.4, 0.35))
                collection.set_linewidth(0.0)

    cax = fig.add_axes([0.125, 0.08, 0.775, 0.02])
    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    try:
        cbar.formatter.set_useOffset(False)
        #cbar.formatter.set_scientific(False)
        cbar.update_ticks()
    except Exception:
        pass
    cbar.ax.xaxis.get_offset_text().set_visible(False)
    _sup = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
    _order_sup = str(order).translate(_sup)
    if np.isfinite(dphi_deg):
        cbar.set_label(f"AAM anomaly (×10{_order_sup} kg m² s⁻¹ per {dphi_deg:g}° latitude band)", size=14)
    else:
        cbar.set_label(f"AAM anomaly (×10{_order_sup} kg m² s⁻¹ per 0.25° latitude band)", size=14)
    cbar.ax.tick_params(labelsize=11)

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8)
    for lat in (-40, -20, 20, 40):
        ax.axhline(lat, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=2)

    add_active_month_percent_labels(ax, time_vals, active_pct)
    
    ax.set_xlim(1, args.composite_months)
    ax.set_ylim(-60, 60)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("Month since onset", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        f"ERA5 Reanalysis Event Mean AAM anomaly\n"
        f"({args.p_min:g}-{args.p_max:g} hPa) {event_count} events {args.start_year}-{args.end_year}"
    )

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_event_composite_all_plot() -> None:
    args = parse_composite_args()
    
    active_pct = None
    
    nino34 = composite_core.load_nino34(args)
    date_list = composite_core.detect_enso_state_windows(args, nino34)
    
    # 🚀 FIX 3: EXPLICITLY TELL DASK TO USE YOUR 8 SLURM CORES
    cluster = LocalCluster(n_workers=8, threads_per_worker=1, memory_limit='100GB')
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    if args.min_enso_months < 1:
        raise ValueError("--min-enso-months must be >= 1")
    if args.composite_months < 1:
        raise ValueError("--composite-months must be >= 1")
    if args.rolling_period < 1:
        raise ValueError("--rolling-period must be >= 1")

    # Always compute significance by default
    args.compute_significance = True

    result_path = _composite_result_path(args, composite_output_dir)
    rolling_tag = f"_rolling{int(args.rolling_period)}" if int(args.rolling_period) > 1 else ""
    nino_thres_tag = f"nino_thres{float(args.nino_threshold)}" if args.nino_threshold is not None else ""
    file_suffix = _plotting_utils_file_suffix(
        f"_event_mean_{args.enso_state}_{args.region}",
        args.composite_start,
        args.onset_season,
    )
    lat_lon_plot = os.path.join(
        composite_output_dir,
        f"AAM_anomalies_lat_lon_snapshots_ERA5_EVENT_MEAN_{args.start_year}-{args.end_year}_{args.p_min:.1f}-{args.p_max:.1f}hPa_{args.region}{rolling_tag}_{nino_thres_tag}{file_suffix}.png",
    )
    lat_time_plot = os.path.join(
        composite_output_dir,
        f"AAM_anomalies_lat_time_ERA5_EVENT_MEAN_{args.start_year}-{args.end_year}_{args.p_min:.1f}-{args.p_max:.1f}hPa_{args.region}{rolling_tag}_{nino_thres_tag}{file_suffix}.svg",
    )
    height_lat_plot = os.path.join(
        composite_output_dir,
        f"AAM_anomalies_lat_level_snapshots_ERA5_EVENT_MEAN_{args.start_year}-{args.end_year}_{args.region}{rolling_tag}_{nino_thres_tag}{file_suffix}.png",
    )
    
    (
        lat_lon_composite,
        lat_lon_mask,
        height_lat_composite,
        height_lat_mask,
        u_lat_lon_composite,
        u_height_lat_composite,
        event_labels,
    ) = _build_streamed_composites(
        date_list,
        args,
    )
    
    active_pct = compute_active_month_percent(
        nino34,
        event_labels,
        composite_months=args.composite_months,
        composite_start=args.composite_start,
        enso_state=args.enso_state,
    )
    
    if args.replot:
        if os.path.exists(result_path):
            print(f"Replot requested. Loading composite NetCDF from {result_path} and regenerating plots.")
            with xr.open_dataset(result_path) as ds:
                ds = ds.load()
                _plot_composite_outputs(
                    ds["lat_lon_composite"],
                    ds["lat_lon_significance_mask"] if "lat_lon_significance_mask" in ds.data_vars else None,
                    lat_time_plot,
                    ds["height_lat_composite"],
                    ds["height_lat_significance_mask"] if "height_lat_significance_mask" in ds.data_vars else None,
                    ds["u_lat_lon_composite"] if "u_lat_lon_composite" in ds.data_vars else None,
                    ds["u_height_lat_composite"] if "u_height_lat_composite" in ds.data_vars else None,
                    args,
                    int(ds.attrs.get("n_events", 0)),
                    composite_output_dir,
                    lat_lon_plot,
                    height_lat_plot,
                    active_pct
                )
            print(f"Replotted composite results from {result_path}")
            return
        else:
            raise RuntimeError(f"Replot requested but no valid composite NetCDF found at {result_path}")

    if _composite_cache_is_valid(result_path):
        print(f"Composite NetCDF already exists at {result_path}; reusing it for plotting.")
        with xr.open_dataset(result_path) as ds:
            ds = ds.load()
            _plot_composite_outputs(
                ds["lat_lon_composite"],
                ds["lat_lon_significance_mask"] if "lat_lon_significance_mask" in ds.data_vars else None,
                lat_time_plot,
                ds["height_lat_composite"],
                ds["height_lat_significance_mask"] if "height_lat_significance_mask" in ds.data_vars else None,
                ds["u_lat_lon_composite"] if "u_lat_lon_composite" in ds.data_vars else None,
                ds["u_height_lat_composite"] if "u_height_lat_composite" in ds.data_vars else None,
                args,
                int(ds.attrs.get("n_events", 0)),
                composite_output_dir,
                lat_lon_plot,
                height_lat_plot,
                active_pct
            )
        print(f"Saved composite results to {result_path}")
        return
    elif os.path.exists(result_path):
        print(f"Existing composite NetCDF at {result_path} is stale; recomputing with pressure coordinates.")


    if not date_list:
        raise RuntimeError("No ENSO events matched the requested criteria.")
    print(f"Detected {len(date_list)} ENSO event onset(s): {[onset[:7] for onset, _ in date_list]}")
    

    
    active_pct = compute_active_month_percent(
    nino34,
    event_labels,
    composite_months=args.composite_months,
    composite_start=args.composite_start,
    enso_state=args.enso_state,
)
    
    event_count = len(event_labels)
    print(f"Built streamed composites with {event_count} complete event(s): {event_labels}")
    gc.collect()
    
        # ---> ADD THIS CALCULATION <---
    active_pct = compute_active_month_percent(
        nino34,
        event_labels,
        composite_months=args.composite_months,
        composite_start=args.composite_start,
        enso_state=args.enso_state,
    )

    result_path, stack_path = save_composite_results(
        event_count,
        lat_lon_composite,
        lat_lon_mask,
        height_lat_composite,
        height_lat_mask,
        u_lat_lon_composite,
        u_height_lat_composite,
        event_labels,
        args,
        composite_output_dir,
    )

    _plot_composite_outputs(
        lat_lon_composite,
        lat_lon_mask,
        lat_time_plot,
        height_lat_composite,
        height_lat_mask,
        u_lat_lon_composite,
        u_height_lat_composite,
        args,
        len(event_labels),
        composite_output_dir,
        lat_lon_plot,
        height_lat_plot,
        active_pct
    )

    print(f"Saved composite results to {result_path}")
    if stack_path:
        print(f"Saved event stack to {stack_path}")


if __name__ == '__main__':
    run_event_composite_all_plot()
