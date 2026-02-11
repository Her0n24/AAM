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
zonal_wind_path_base = f"{base_dir}/monthly_mean/variables/"
climatology_path_base = f"{base_dir}/climatology/"
output_dir = f"{base_dir}/figures/"
pressure_lvl_dir = f"{base_dir}/l137_a_b.csv"

# wind fields data 
# zonal_mean_ERA5_u_1988-01.nc

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


def plot_anomalies_3d(start_year, end_year, variable, clim_start_yr=1980, clim_end_yr=2000, use_pressure_levels=True, find_extremum='max'):
    """
    Plot 3D structure of anomalies: multiple latitude×level slices at different times
    
    Parameters:
    -----------
    find_extremum : str
        Either 'max' or 'min' to find maximum or minimum AAM anomaly
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
    
    # Load pressure levels (always needed for proper integration)
    pressure_lvl = None
    try:
        pressure_lvl_pd = pd.read_csv(pressure_lvl_dir, header=0)
        pressure_lvl_full = pressure_lvl_pd['ph [hPa]'].values
        print(f"Loaded {len(pressure_lvl_full)} full pressure levels from {pressure_lvl_dir}")
        
        # Discard first value and compute half levels (average of consecutive values)
        pressure_lvl_full = pressure_lvl_full[1:]  # Remove first value
        pressure_lvl = (pressure_lvl_full[:-1] + pressure_lvl_full[1:]) / 2  # Half levels
        print(f"Computed {len(pressure_lvl)} half-levels from full levels")
    except Exception as e:
        print(f"Warning: Could not load pressure levels: {e}")
        print("Will use model level indices for integration (not physically accurate)")
    
    if use_pressure_levels:
        print("Using pressure levels for plotting")
    else:
        print("Using model levels for plotting")
        use_pressure_levels = False
    
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
    # Select subset of times to plot (every month)
    time_indices = np.arange(0, len(anomalies.time), 1)  # Every month
    n_slices = len(time_indices)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Method 1: Multiple 2D slices arranged in 3D space
    ax = fig.add_subplot(111, projection='3d')
    
    lat_vals = anomalies.latitude.values
    level_vals = anomalies[level_dim].values
    
    print(f"Data has {len(level_vals)} vertical levels")
    
    # Convert to pressure levels if requested
    if use_pressure_levels and pressure_lvl is not None:
        if len(pressure_lvl) != len(level_vals):
            print(f"Warning: pressure level array length ({len(pressure_lvl)}) doesn't match data levels ({len(level_vals)})")
            print("Using model levels instead")
            use_pressure_levels = False
            pressure_vals = level_vals
            vertical_label = 'Model Level'
        else:
            print(f"Successfully matched {len(pressure_lvl)} pressure levels to data")
            pressure_vals = pressure_lvl
            vertical_label = 'Pressure (hPa)'
    else:
        if use_pressure_levels:
            print("Warning: pressure_lvl is None, using model levels")
        pressure_vals = level_vals
        vertical_label = 'Model Level'
    
    print(f"Plotting with vertical axis: {vertical_label}")
    
    # Compute color limits from data
    vmax = 5e21
    vmin = -vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r
    
    for i, t_idx in enumerate(time_indices):
        time_val = anomalies.time.values[t_idx]
        data_slice = anomalies.isel(time=t_idx).values  # shape: (level, latitude)
        
        # Create meshgrid for this slice
        LAT, PRESS = np.meshgrid(lat_vals, pressure_vals)
        
        # Plot as surface at time position i (time on x-axis, pressure on z-axis)
        surf = ax.plot_surface(np.ones_like(data_slice)*i, LAT, PRESS,
                                facecolors=cmap(norm(data_slice)),
                                shade=False, alpha=0.8)
    
    ax.set_xlabel('Time Index', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_zlabel(vertical_label, fontsize=12)
    ax.set_title(f'{variable.upper()} Anomaly 3D Structure: Time × Latitude × Level using climatology {clim_start_yr}-{clim_end_yr}', fontsize=14)
    ax.invert_zaxis()  # Invert so surface is at bottom
    
    # Add vertical line at equator (latitude = 0)
    time_range = np.arange(n_slices)
    equator_lat = np.zeros_like(time_range)
    pressure_range = np.linspace(pressure_vals.min(), pressure_vals.max(), 100)
    for t in time_range:
        ax.plot([t, t], [0, 0], [pressure_vals.min(), pressure_vals.max()], 
                color='black', linewidth=2.5, alpha=0.8)
    
    # Set time labels
    ax.set_xticks(range(n_slices))
    time_labels = [pd.to_datetime(anomalies.time.values[idx]).strftime('%Y-%m') 
                   for idx in time_indices]
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label(f'{variable} anomaly', fontsize=12)
    
    plt.tight_layout()
    
    if use_pressure_levels:
        output_file = f"{output_dir}{variable}_anomalies_3d_{start_year}-{end_year}_pl.png"
    else:
        output_file = f"{output_dir}{variable}_anomalies_3d_{start_year}-{end_year}.png"
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"3D figure saved to: {output_file}")
    plt.close()
    
    # Method 2: Volumetric slice plot (latitude-level at multiple times)
    # Show more snapshots now (every month, display up to 24)
    n_snapshots = min(24, len(time_indices))
    snapshot_indices = time_indices[:n_snapshots]

    # Load zonal wind data for overlay
    print("Loading zonal wind data for overlay...")
    zonal_wind_files = []
    for t_idx in snapshot_indices:
        time_val = pd.to_datetime(anomalies.time.values[t_idx])
        year, month = time_val.year, time_val.month
        wind_pattern = f"{zonal_wind_path_base}zonal_mean_ERA5_u_{year}-{month:02d}.nc"
        wind_files = glob.glob(wind_pattern)
        if wind_files:
            zonal_wind_files.append(wind_files[0])
        else:
            zonal_wind_files.append(None)
            print(f"Warning: No zonal wind file found for {year}-{month:02d}")
    
    
    n_cols = 4
    n_rows = (n_snapshots + n_cols - 1) // n_cols
    
    # Create figure with GridSpec for paired subplots (contour + profile)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(22, 6*n_rows))
    gs = GridSpec(n_rows * 2, n_cols, figure=fig, height_ratios=[6, 1] * n_rows, hspace=0.3, wspace=0.2)
    
    contour_axes = []
    profile_axes = []
    for row in range(n_rows):
        for col in range(n_cols):
            contour_axes.append(fig.add_subplot(gs[row*2, col]))
            profile_axes.append(fig.add_subplot(gs[row*2 + 1, col]))
    
    for i, t_idx in enumerate(snapshot_indices):
        if i >= len(contour_axes):
            break
        
        time_val = pd.to_datetime(anomalies.time.values[t_idx])
        data_slice = anomalies.isel(time=t_idx)
        
        # Use explicit levels array to ensure consistent colorbar
        levels = np.linspace(vmin, vmax, 21)
        im = contour_axes[i].contourf(lat_vals, pressure_vals, data_slice.values,
                             levels=levels, cmap='RdBu_r', extend='both')
        
        # Overlay zonal wind contours
        if zonal_wind_files[i] is not None:
            try:
                ds_wind = xr.open_dataset(zonal_wind_files[i])
                # Assuming variable name is 'u' for zonal wind
                wind_var = 'u' if 'u' in ds_wind else list(ds_wind.data_vars)[0]
                wind_data = ds_wind[wind_var]
                
                # Get wind data coordinates
                wind_lat = wind_data.latitude.values if 'latitude' in wind_data.dims else wind_data.lat.values
                
                # Wind data is on full levels (137), need to convert to half levels (136) 
                # to match the AAM data
                wind_values = wind_data.values
                if len(wind_values) == len(pressure_vals) + 1:
                    # Convert wind from full levels to half levels
                    # Average consecutive levels to get half-level values
                    wind_values = (wind_values[:-1, :] + wind_values[1:, :]) / 2

                # Use the same pressure_vals array that was used for the anomaly plot
                # This ensures wind contours align with the same vertical coordinate system
                wind_contour_levels = np.arange(-60, 61, 10)  # Contours every 10 m/s
                wind_contour_levels = wind_contour_levels[np.abs(wind_contour_levels) >= 10]  # Only show |u| >= 15 m/s
                cs = contour_axes[i].contour(wind_lat, pressure_vals, wind_values,
                                    levels=wind_contour_levels, colors='black', 
                                    linewidths=0.8, alpha=0.6)
                contour_axes[i].clabel(cs, inline=True, fontsize=7, fmt='%d')
                ds_wind.close()
            except Exception as e:
                print(f"Warning: Could not overlay wind data for snapshot {i}: {e}")
        
        contour_axes[i].set_xlabel('Latitude (°N)', fontsize=10)
        contour_axes[i].set_xlim(-60, 60)
        contour_axes[i].set_ylabel(vertical_label, fontsize=10)
        contour_axes[i].set_title(f'{time_val.strftime("%Y-%m")}', fontsize=11, pad=3)
        contour_axes[i].invert_yaxis()  # Invert so surface is at bottom
        
        # Find latitude of maximum or minimum AAM in northern hemisphere
        nh_mask = lat_vals > 0  # Northern hemisphere only
        lvl_mask = pressure_vals > 100  # Pressure level constraint
        
        # Apply masks using proper 2D indexing
        nh_data = data_slice.values[np.ix_(lvl_mask, nh_mask)]
        nh_lats = lat_vals[nh_mask]
        
        # Find the extremum value and its location
        if find_extremum == 'min':
            extreme_value = np.min(nh_data)
            extreme_idx = np.unravel_index(np.argmin(nh_data), nh_data.shape)
        else:  # default to 'max'
            extreme_value = np.max(nh_data)
            extreme_idx = np.unravel_index(np.argmax(nh_data), nh_data.shape)
        
        extreme_lat = nh_lats[extreme_idx[1]]
        
        # Add vertical line at the latitude of extremum AAM
        contour_axes[i].axvline(extreme_lat, color='C1', linewidth=2, linestyle='-', alpha=0.8, zorder=10)
        
        # Add vertical line at equator
        # contour_axes[i].axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.9)
        
        # Create vertical profile plot below
        # Vertically integrate AAM anomaly at each latitude
        # Always use pressure levels for integration (even if plotting with model levels)
        if pressure_lvl is not None and len(pressure_lvl) == len(level_vals):
            vertical_integral = np.trapz(data_slice.values, x=pressure_lvl, axis=0)
        else:
            # Fallback: use whatever is on the y-axis
            vertical_integral = np.trapz(data_slice.values, x=pressure_vals, axis=0)
        
        profile_axes[i].plot(lat_vals, vertical_integral, 'C0-', linewidth=1.5)
        profile_axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        profile_axes[i].axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        profile_axes[i].set_xlim(-60, 60)
        profile_axes[i].set_ylim(-2.5e24, 2.4e24)
        profile_axes[i].set_ylabel('Total', fontsize=9)
        profile_axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(snapshot_indices), len(contour_axes)):
        contour_axes[j].axis('off')
        profile_axes[j].axis('off')
    
    # Add a single colorbar at the bottom for all subplots
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.01])  # [left, bottom, width, height]
    
    # Create discrete levels matching the contour levels
    levels = np.linspace(vmin, vmax, 11)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both', spacing='proportional')
    cbar.set_label(f'{variable} anomaly', fontsize=12)
    # Set ticks at every other level boundary for clarity
    tick_indices = np.arange(0, 11, 1)
    cbar.set_ticks(levels[tick_indices])
    
    fig.suptitle(f'{variable.upper()} Anomaly: Latitude × Level Snapshots using climatology {clim_start_yr}-{clim_end_yr} with zonal mean zonal wind', fontsize=16, y=0.905)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])  # Leave space for colorbar at bottom and reduce top gap
    
    if use_pressure_levels:
        output_file2 = f"{output_dir}{variable}_anomalies_snapshots_{start_year}-{end_year}_pl.png"
    else:
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
    parser.add_argument('--use_model', action='store_true', help='Use model levels instead of pressure levels for plotting')
    parser.add_argument('--find_min', action='store_true', help='Find minimum AAM instead of maximum')
    args = parser.parse_args()
    extremum = 'min' if args.find_min else 'max'
    plot_anomalies_3d(args.start, args.end, args.variable, args.clim_start, args.clim_end, not args.use_model, extremum)