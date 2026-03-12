import matplotlib
# Use non-interactive backend BEFORE importing pyplot
matplotlib.use('Agg')

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import pandas as pd
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.cm as cm
import matplotlib as mpl
from typing import Optional, Tuple

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=100, help='Minimum pressure level (hPa) to include (default: 100 hPa)')
parser.add_argument('--p-max', type=float, default=1000, help='Maximum pressure level (hPa) to include (default: 1000 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year to plot (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year to plot (default: 2000)')
parser.add_argument('--member', type=str, default='1', help='Ensemble member to plot (default: 1, control)')
parser.add_argument('--events-json', type=str, default=None, help='Path to events JSON file to overlay trajectories')
parser.add_argument('--no-enso', action='store_true', help='Skip loading/plotting ENSO panel (useful if remote path is slow)')
args = parser.parse_args()

base_dir = os.getcwd()
AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"
output_dir = f"{base_dir}/figures/"

CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul"
nino34_directory = f"{CMIP6_path_base}/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/"
output_dir = f"{base_dir}/figures/"

ensemble_member = f"r{args.member}i1p1f3"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Default period 
start_yr, end_yr = args.start_year, args.end_year
clim_start_yr, clim_end_yr = 1980, 2000

# Pressure selection (user inputs in hPa)
p_min_hpa = args.p_min
p_max_hpa = args.p_max

def _latitude_band_width_radians(lat_deg: np.ndarray) -> np.ndarray:
    """dphi (radians) for latitude bands centered on lat_deg."""
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=float))
    n = lat_rad.size
    edges = np.empty(n + 1, dtype=float)
    edges[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    edges[0] = -0.5 * np.pi
    edges[-1] = 0.5 * np.pi
    return np.abs(np.diff(edges))


def _pressure_range_in_coord_units(level_coord: xr.DataArray, p_min_hpa: float, p_max_hpa: float) -> tuple[float, float]:
    """Return (p_min, p_max) in the same units as `level_coord` (Pa vs hPa).

    Heuristic:
    - if max(level) > 2000 => coordinate is in Pa
    - else => coordinate is in hPa
    """
    lev_vals = np.asarray(level_coord.values, dtype=float)
    lev_max = float(np.nanmax(lev_vals))
    if lev_max > 2000.0:
        return p_min_hpa * 100.0, p_max_hpa * 100.0
    return p_min_hpa, p_max_hpa

def get_ENSO_index(start_year, end_year) -> Tuple[Optional[pd.DatetimeIndex], Optional[np.ndarray]]:
    """Get the Nino3.4 index from CMIP6 data for the specified period."""

    file_pattern = os.path.join(nino34_directory, f"nino34_ssta_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found for Nino3.4 index with pattern: {file_pattern}")
        return None, None
    
    ds = xr.open_dataset(files[0])
    if 'tos' not in ds.data_vars:
        print(f"'tos' variable not found in dataset: {files[0]}")
        return None, None
    
    # Many of these processed index files are stored as (time, 1, 1).
    # Squeeze to a true 1D time series for plotting.
    nino34 = ds['tos'].squeeze(drop=True)

    # Time can be a CFTime calendar (e.g., 360_day) which pandas can't convert.
    # Filter using Python datetime-like attributes on the cftime objects.
    time_vals = nino34['time'].values
    mask = np.array([(t.year >= start_year) and (t.year <= end_year) for t in time_vals], dtype=bool)
    nino34_filtered = nino34.isel(time=np.where(mask)[0])

    # Build month-start timestamps for plotting (aligns with AAM times).
    times_filtered = pd.DatetimeIndex(
        [pd.Timestamp(f"{t.year:04d}-{t.month:02d}-01") for t in nino34_filtered['time'].values]
    )
    nino34_filtered = np.asarray(nino34_filtered.values, dtype=float).reshape(-1)
    
    ds.close()
    
    return times_filtered, nino34_filtered

def _load_event_trajectories(json_path: str) -> tuple[list[dict], dict]:
    """Load event trajectories from the metrics JSON.
    
    Returns:
        trajectories: list of dicts with {event_id, times:DatetimeIndex, lats:ndarray}.
                     Gap months (null latitudes) are preserved as NaN.
        metadata: dict with constraints from the JSON.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('config', data.get('metadata', {}))
    events = data.get('events', [])
    
    trajectories = []
    for evt in events:
        eid = evt.get('event_id')
        monthly_data = evt.get('monthly_com_latitude', {})
        months = monthly_data.get('months', [])
        lats = monthly_data.get('com_latitude_deg', [])
        
        if not months or not lats:
            continue
        
        times_list = []
        lats_list = []
        for m, lat in zip(months, lats):
            try:
                t = pd.Timestamp(m[:7] + "-01")  # YYYY-MM-DD -> YYYY-MM-01
            except Exception:
                continue
            
            # Convert null/None to NaN
            if lat is None:
                fv = np.nan
            else:
                try:
                    fv = float(lat)
                except (TypeError, ValueError):
                    fv = np.nan
            
            times_list.append(t)
            lats_list.append(fv)
        
        if not times_list:
            continue
        if not np.any(np.isfinite(lats_list)):
            continue
        
        trajectories.append({
            'event_id': eid,
            'times': pd.DatetimeIndex(times_list),
            'lats': np.asarray(lats_list, dtype=float),
        })
    
    return trajectories, metadata

def plot_AAM_anomalies(start_year, end_year, component='AAM', *, nlevels=11, cmap_name='RdBu_r', 
                       vmin=None, vmax=None, savefile=None, show=True, events_json=None, plot_enso=True):
    """Plot AAM anomalies (variation from climatological mean) for a given period.

    Loads files matching `AAM_ERA5_{year}*.nc` from `AAM_data/monthly_mean/` for all years
    in the range [start_year, end_year], calculates the climatological mean, and plots
    the anomalies as latitude vs time using a discrete colormap.

    Args:
        start_year (int): Start year for climatology period.
        end_year (int): End year for climatology period.
        component (str): Variable name inside the netCDF files (default 'AAM').
        nlevels (int): Number of discrete color levels.
        cmap_name (str): Matplotlib colormap name.
        vmin, vmax (float): Color limits (optional).
        savefile (str): If provided, save figure to this path.
        show (bool): If True, call `plt.show()`.

    Returns:
        (fig, ax): Matplotlib figure and axis.
    """
    

    file_format = f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc" 

    # Load all data
    ds = xr.open_dataset(file_format)
    
    if component not in ds.data_vars:
        raise KeyError(f"variable '{component}' not found in dataset")

    da = ds[component]

    # Normalize common CMIP6 dimension names
    rename = {}
    if 'lat' in da.dims and 'latitude' not in da.dims:
        rename['lat'] = 'latitude'
    if 'lon' in da.dims and 'longitude' not in da.dims:
        rename['lon'] = 'longitude'
    if 'plev' in da.dims and 'level' not in da.dims:
        rename['plev'] = 'level'
    if rename:
        da = da.rename(rename)
    
    # ---- Make missing values NaN explicitly (prevents huge artefacts) ----
    fv = (
        da.encoding.get("_FillValue", None)
        or da.attrs.get("_FillValue", None)
        or da.attrs.get("missing_value", None)
    )
    print(f"_FillValue / missing_value detected: {fv}")
    
    da = da.where(np.isfinite(da))  # drop inf/-inf if present
    
    if fv is not None:
        da = da.where(da != fv)
    
    print(f"Loaded data shape: {da.shape}")
    print(f"Dimensions: {da.dims}")
    
    # Check time coordinate encoding
    print(f"\n--- Time coordinate verification ---")
    time_var = ds['time']
    print(f"Time encoding: {time_var.encoding if hasattr(time_var, 'encoding') else 'N/A'}")
    print(f"Time attributes: {dict(time_var.attrs)}")
    print(f"Time dtype: {time_var.dtype}")
    print(f"First 5 time values (raw): {time_var.values[:5]}")
    print(f"First 5 time values (decoded): {da.time.values[:5]}")
    print(f"Last 3 time values (decoded): {da.time.values[-3:]}")
    
    # Verify the time range
    first_time = da.time.values[0]
    last_time = da.time.values[-1]
    print(f"Time range: {first_time} to {last_time}")
    print(f"  First: year={first_time.year}, month={first_time.month}, day={first_time.day}")
    print(f"  Last: year={last_time.year}, month={last_time.month}, day={last_time.day}")
    print(f"Total time steps: {len(da.time)}")
    
    print(f"\n  Raw data stats: min={np.nanmin(da.values):.3e}, max={np.nanmax(da.values):.3e}, mean={np.nanmean(da.values):.3e}")
    print(f"  Raw data std dev: {np.nanstd(da.values):.3e}\n")
    
    # Process 4D data (time, level, latitude, longitude) into 2D (time, latitude)
    # Step 1: Compute zonal integral if longitude dimension exists
    if 'longitude' in da.dims:
        print("Computing zonal integral over longitude...")
        lon_rad = np.deg2rad(da['longitude'].astype(float))
        da = da.assign_coords(longitude=lon_rad).sortby('longitude')
        da = da.integrate('longitude')
        print(f"After zonal integral shape: {da.shape}")
        print(f"  Stats: min={np.nanmin(da.values):.3e}, max={np.nanmax(da.values):.3e}, mean={np.nanmean(da.values):.3e}")
    
    # Step 2: Vertical sum if level dimension exists
    if 'level' in da.dims:
        print("Summing over vertical layers (dp already applied in compute step)...")

        # Select pressure range before summing.
        da = da.sortby('level')
        pmin_u, pmax_u = _pressure_range_in_coord_units(da['level'], p_min_hpa, p_max_hpa)
        print(f"Selecting pressure range: {p_min_hpa:.0f}-{p_max_hpa:.0f} hPa (coord units: {pmin_u:.3g}-{pmax_u:.3g})")
        # slice expects increasing coords when sorted
        da = da.sel(level=slice(min(pmin_u, pmax_u), max(pmin_u, pmax_u)))

        # Those units are consistent with having already applied dp (and r^3/g), i.e. it is not “per Pa” anymore.
        
        # in compute_AAM_full_field.py you already multiplied by dp, 
        # so in plot_AAM.py you must not multiply by dp again. 
        # Doing so will inflate the magnitude by roughly a column pressure 
        # scale (~10⁴–10⁵ Pa), i.e. “a few orders of magnitude”.
        # Calculate pressure thickness (dp) between levels
        vertical_sum = da.sum(dim='level')
        
        # Create new DataArray with integrated data
        da = xr.DataArray(
            vertical_sum,
            coords={'time': da.time, 'latitude': da.latitude},
            dims=['time', 'latitude']
        )
        
        print(f"After vertical summation shape: {da.shape}")
        print(f"  Stats: min={np.nanmin(da.values):.3e}, max={np.nanmax(da.values):.3e}, mean={np.nanmean(da.values):.3e}")
    
    # Step 3: convert to "per latitude band" by multiplying by band width dφ
    if 'latitude' in da.dims:
        print("Applying latitude band width (dphi) to get per-latitude-band totals...")
        da = da.sortby('latitude')
        dphi = _latitude_band_width_radians(da['latitude'].values)
        dphi_deg = np.rad2deg(dphi)
        dphi_da = xr.DataArray(dphi, coords={'latitude': da['latitude']}, dims=('latitude',))
        da = da * dphi_da
        da.attrs['units'] = 'kg m^2 s^-1'  # now explicitly "per latitude band"
        da.attrs['long_name'] = 'AAM per latitude band (zonal & vertical integrated)'
    
    # ensure dims include time and latitude
    if 'time' not in da.dims or 'latitude' not in da.dims:
        raise ValueError(f'dataarray must have dims (time, latitude), got {da.dims}')
    
    # Calculate climatological mean
    # Group by month and calculate mean across all years
    # Use year filtering instead of string slicing for cftime compatibility
    years = np.array([t.year for t in da.time.values])
    time_mask = (years >= clim_start_yr) & (years <= clim_end_yr)
    
    print(f"\n--- Climatology period selection ---")
    print(f"Total time steps: {len(da.time)}")
    print(f"Years range: {years.min()} to {years.max()}")
    print(f"Selected for climatology ({clim_start_yr}-{clim_end_yr}): {time_mask.sum()} months")
    if time_mask.sum() > 0:
        selected_years = years[time_mask]
        print(f"  Year range in selection: {selected_years.min()} to {selected_years.max()}")
        print(f"  Expected months: {(clim_end_yr - clim_start_yr + 1) * 12}")
    
    if time_mask.sum() > 0:
        climatology = da.isel(time=time_mask).groupby('time.month').mean('time')
        print(f"Calculated climatology from {clim_start_yr} to {clim_end_yr} ({time_mask.sum()} months)")
    else:
        print(f"Warning: No data in climatology period, using full dataset")
        climatology = da.groupby('time.month').mean('time')
    
    print(f"Climatology shape: {climatology.shape}")
    print(f"  Climatology dimensions: {climatology.dims}")
    print(f"  Climatology coords: {list(climatology.coords.keys())}")
    if 'month' in climatology.coords:
        print(f"  Month values: {climatology.month.values}")
    print(f"  Stats: min={np.nanmin(climatology.values):.3e}, max={np.nanmax(climatology.values):.3e}, mean={np.nanmean(climatology.values):.3e}")
    
    # Test alignment: print a few examples
    print("\n--- Testing month alignment ---")
    test_times = da.time.values[:5]  # First 5 time points
    for t in test_times:
        month_num = t.month
        print(f"Time: {t.year}-{t.month:02d}-{t.day:02d} → month={month_num}, calendar={t.calendar if hasattr(t, 'calendar') else 'N/A'}")
    
    # Calculate anomalies by subtracting climatology from each month
    anomalies = da.groupby('time.month') - climatology
    
    print(f"\nAnomalies shape: {anomalies.shape}")
    print(f"  Stats: min={np.nanmin(anomalies.values):.3e}, max={np.nanmax(anomalies.values):.3e}, mean={np.nanmean(anomalies.values):.3e}")
    print(f"  Std dev: {np.nanstd(anomalies.values):.3e}")
    
    # Verify the subtraction is correct for a few examples
    print("\n--- Verifying anomaly calculation ---")
    for i in range(min(3, len(da.time))):
        time_val = da.time.values[i]
        month_num = time_val.month
        
        # Get raw value at this time (at mid-latitude index)
        lat_idx = len(da.latitude) // 2
        raw_val = da.isel(time=i, latitude=lat_idx).values
        
        # Get climatology for this month
        if 'month' in climatology.coords:
            clim_val = climatology.sel(month=month_num).isel(latitude=lat_idx).values
        else:
            clim_val = climatology.isel(month=month_num-1, latitude=lat_idx).values  # 0-indexed
        
        # Get anomaly
        anom_val = anomalies.isel(time=i, latitude=lat_idx).values
        
        # Check if they match
        expected_anom = raw_val - clim_val
        print(f"  Time {time_val.year}-{time_val.month:02d} (month={month_num}): raw={raw_val:.3e}, clim={clim_val:.3e}, anom={anom_val:.3e}, expected={expected_anom:.3e}, diff={abs(anom_val-expected_anom):.3e}")
    
    # Filter to display period (start_year to end_year)
    display_years = np.array([t.year for t in anomalies.time.values])
    display_mask = (display_years >= start_year) & (display_years <= end_year)
    
    if display_mask.sum() > 0:
        anomalies = anomalies.isel(time=display_mask)
        print(f"Plotting {display_mask.sum()} months from {start_year} to {end_year}")
    else:
        print(f"Warning: No data found for {start_year}-{end_year}, plotting all available data")
    
    # Convert cftime to pandas timestamps for plotting
    proper_times = []
    for time_val in anomalies.time.values:
        year = time_val.year
        month = time_val.month
        proper_times.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    times = pd.DatetimeIndex(proper_times)
    lats = anomalies['latitude'].values
    data = anomalies.values  # shape (time, lat)
    
    print(f"\nFinal plotting data shape: {data.shape}")
    print(f"  Stats: min={np.nanmin(data):.3e}, max={np.nanmax(data):.3e}, mean={np.nanmean(data):.3e}")
    print(f"  Std dev: {np.nanstd(data):.3e}")
    print(f"  NaN count: {np.isnan(data).sum()} / {data.size} ({100*np.isnan(data).sum()/data.size:.1f}%)")
    print(f"  Percentiles: 2%={np.nanpercentile(data, 2):.3e}, 50%={np.nanpercentile(data, 50):.3e}, 98%={np.nanpercentile(data, 98):.3e}")
    
    # Check if latitudes are decreasing (90 to -90) and flip if needed
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[:, ::-1]

    if vmin is None:
        vmin = np.nanpercentile(data, 2)  # Use percentiles for better scaling
    if vmax is None:
        vmax = np.nanpercentile(data, 98)
    
    # Make colorbar symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    print(f"\nColor scale: vmin={vmin:.3e}, vmax={vmax:.3e}")

    levels = np.linspace(vmin, vmax, nlevels)
    colormaps = getattr(mpl, 'colormaps', None)
    if colormaps is not None:
        base_cmap = colormaps.get_cmap(cmap_name)
    else:
        base_cmap = cm.get_cmap(cmap_name)
    cmap_disc = ListedColormap(list(base_cmap(np.linspace(0, 1, nlevels - 1))))
    norm = BoundaryNorm(levels, ncolors=nlevels - 1, clip=True)

    # Layout requirement:
    # - Keep the original AAM panel size (previously a 16x6 figure)
    # - Put Niño3.4 directly below the AAM panel
    # - Put the colorbar below Niño3.4 (and keep its original width)
    # Achieve this by increasing total figure height while allocating ~6" to the top panel.
    fig = plt.figure(figsize=(16, 6), constrained_layout=False)
    # Make the colorbar row deliberately short so the bar is thin.
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[6.0, 1.0, 0.18], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_enso = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[2, 0])

    # imshow: transpose so y is latitude (data shape: time x lat)
    times_num = np.asarray(mdates.date2num(times.to_pydatetime()), dtype=float)
    im = ax.imshow(
        data.T,
        origin='lower',
        aspect='auto',
        cmap=cmap_disc,
        norm=norm,
        extent=[times_num[0], times_num[-1], lats[0], lats[-1]],
        interpolation='bilinear'  # Smooth interpolation for higher DPI
    )

    ax.xaxis_date()
    # Smart tick spacing: estimate label width vs axis width and reduce to every
    # 2 years when yearly labels would overlap.
    years = np.unique(times.year)
    n_years = len(years)
    font_size = 14
    # Approximate average character width in points (roughly 0.6 * fontsize)
    chars = 4  # 'YYYY'
    char_width_pt = font_size * 0.6
    label_width_in = (char_width_pt / 72.0) * chars
    # Axis width in inches (figure width * axis fraction)
    axis_width_in = fig.get_size_inches()[0] * ax.get_position().width
    required_width_in = n_years * label_width_in * 1.05
    if required_width_in > axis_width_in:
        major_locator = mdates.YearLocator(2)
    else:
        major_locator = mdates.YearLocator(2)
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), ha='center', size=font_size)
    plt.setp(ax.get_yticklabels(), size=14)

    ax.set_ylabel('Latitude (°)', size=16)
    ax.set_xlabel('', size=16)
    
    # Build title with constraint information if available
    title_parts = [
        f"CMIP6 HadGEM3-GC3.1 {ensemble_member} historic zonally-integrated {component} fluctuations from Climatology ({start_year}-{end_year})",
        f"(Summed: {p_min_hpa:.0f}-{p_max_hpa:.0f} hPa)"
    ]
    
    # Load event trajectories if provided
    trajectories = None
    constraint_str = ""
    constraint_tag = ""
    if events_json:
        try:
            trajectories, metadata = _load_event_trajectories(events_json)
            # Build constraint labels from the three boolean flags
            constraint_parts = []
            if metadata.get('winter_constraint'):
                constraint_parts.append('NDJFM onset')
            if metadata.get('el_nino_constraint'):
                constraint_parts.append('El Niño')
            if metadata.get('sym_constraint'):
                constraint_parts.append('SH sym')
            # Also show key numerical parameters
            if metadata.get('onset_lat_max_deg') is not None:
                constraint_parts.append(f"onset≤{metadata['onset_lat_max_deg']}°N")
            if metadata.get('intermittency_constraint') is not None:
                constraint_parts.append(f"stall≤{metadata['intermittency_constraint']}mo")
            if metadata.get('max_southward_jump_deg') is not None:
                constraint_parts.append(f"S-jump<{metadata['max_southward_jump_deg']}°")

            constraint_str = ", ".join(constraint_parts)
            # Short tag for filename: underscore-separated
            _tag_parts = []
            if metadata.get('winter_constraint'):  _tag_parts.append('winter')
            if metadata.get('el_nino_constraint'): _tag_parts.append('elnino')
            if metadata.get('sym_constraint'):     _tag_parts.append('sym')
            constraint_tag = ("_" + "_".join(_tag_parts)) if _tag_parts else ""

            if constraint_str:
                title_parts.append(f"Constraints: {constraint_str}")
        except Exception as e:
            print(f"Warning: Failed to load events JSON: {e}")
            trajectories = None
    
    ax.set_title("\n".join(title_parts), size=18)
    ax.grid(True, alpha=0.3)
    
    # Add black solid line at equator (latitude = 0)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=10)
    
    # Overlay event trajectories if loaded
    if trajectories:
        tmin = times.min()
        tmax = times.max()
        for trj in trajectories:
            tt = trj['times']
            ll = trj['lats']
            # Clip to plotting time range
            mask = (tt >= tmin) & (tt <= tmax)
            if mask.sum() < 1:
                continue
            tt = tt[mask]
            ll = ll[mask]
            x = np.asarray(mdates.date2num(tt.to_pydatetime()), dtype=float)
            # Matplotlib will break the line at NaN gaps
            ax.plot(x, ll, color='C2', linewidth=3, alpha=0.9, zorder=20)
            # Label at first finite point
            finite = np.isfinite(ll)
            if finite.any():
                i0 = int(np.where(finite)[0][0])
                ax.text(
                    x[i0], float(ll[i0]), str(trj['event_id']),
                    fontsize=10, color='white', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='none', alpha=0.7),
                    zorder=21,
                )

    # Colorbar with better formatting
    # Place colorbar horizontally at the bottom to span the width
    # Use `shrink` close to 1.0 so the colorbar occupies most of the axis width
    cbar = fig.colorbar(
        im,
        cax=cax,
        boundaries=levels,
        extend='both',
        orientation='horizontal',
        spacing='proportional',
    )
    
    dphi_med = round(float(np.nanmedian(dphi_deg)), 2)
    
    # Determine if we need to factor out scientific notation
    max_abs_value = max(abs(vmin), abs(vmax))
    if max_abs_value >= 1e3 or max_abs_value <= 1e-3:
        # Calculate the order of magnitude
        order = int(np.floor(np.log10(max_abs_value)))
        factor = 10**order
        print(f"\nUsing scientific notation: order of magnitude = 10^{order} (factor = {factor:.2e})")
        
        # Convert order to superscript notation
        def format_exponent(exp):
            """Convert integer exponent to superscript string"""
            exp_str = str(exp)
            superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
                              '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
            return ''.join(superscript_map.get(c, c) for c in exp_str)
        
        exponent_str = format_exponent(order)
        
        # Format units with superscripts and factor
        units = da.attrs.get('units', '')
        if units:
            units_formatted = units.replace('**', '^')  # Convert ** to ^ first
            units_formatted = units_formatted.replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
            units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
            units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
            units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
            label_text = f"{component} Anomalies 10{exponent_str} {units_formatted} per {dphi_med}° latitude band"
        else:
            label_text = f"{component} Anomalies 10{exponent_str} per {dphi_med}° latitude band "
        
        # Scale tick labels by the factor
        tick_spacing = max(1, len(levels)//8)
        tick_locs = levels[::tick_spacing]
        tick_labels = [f'{val/factor:.1f}' for val in tick_locs]
        cbar.set_ticks(tick_locs.tolist())
        cbar.set_ticklabels(tick_labels)  # type: ignore[arg-type]
    else:
        print(f"\nUsing normal formatting (no scientific notation)")
        # Normal formatting without factoring
        units = da.attrs.get('units', '')
        if units:
            units_formatted = units.replace('**', '^')  # Convert ** to ^ first
            units_formatted = units_formatted.replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
            units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
            units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
            units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
            label_text = f"{component} Anomalies {units_formatted} per {dphi_med}° latitude band"
        else:
            label_text = f"{component} Anomalies kg m^2 s^-1 per {dphi_med}° latitude band"
        
        # Normal tick formatting
        tick_spacing = max(1, len(levels)//8)
        cbar.set_ticks(levels[::tick_spacing].tolist())
    
    # Label placement depends on colorbar orientation
    if getattr(cbar, 'orientation', 'vertical') == 'horizontal':
        cbar.set_label(label_text, rotation=0, labelpad=8, size=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        cbar.set_label(label_text, rotation=270, labelpad=20, size=16)

    # ---- ENSO (Niño3.4) subplot ----
    print("Entering ENSO panel section...")
    if plot_enso:
        enso_times, enso_vals = get_ENSO_index(start_year, end_year)
    else:
        enso_times, enso_vals = None, None

    if enso_times is not None and enso_vals is not None and len(enso_vals) > 0:
        enso_times_num = np.asarray(
            mdates.date2num(pd.DatetimeIndex(enso_times).to_pydatetime()),
            dtype=float,
        )
        ax_enso.plot(enso_times_num, enso_vals, color='black', linewidth=1.5)

        enso_vals = np.asarray(enso_vals, dtype=float).reshape(-1)
        enso_vals_plot = enso_vals.tolist()
        pos = enso_vals >= 0
        neg = enso_vals < 0
        ax_enso.fill_between(  # type: ignore[arg-type]
            enso_times_num,
            0.0,
            enso_vals_plot,
            where=pos,
            color='red',
            alpha=0.35,
            interpolate=True,
        )
        ax_enso.fill_between(  # type: ignore[arg-type]
            enso_times_num,
            0.0,
            enso_vals_plot,
            where=neg,
            color='blue',
            alpha=0.35,
            interpolate=True,
        )

        ax_enso.axhline(0.0, color='black', linewidth=1.0, alpha=0.8)
        ax_enso.set_ylabel('Niño3.4', size=14)
        ax_enso.set_ylim(-2.5,2.5)
        ax_enso.grid(True, alpha=0.3)
    else:
        if plot_enso:
            print("ENSO data unavailable or empty; rendering placeholder panel.")
        else:
            print("ENSO plotting disabled via --no-enso.")
        ax_enso.text(0.5, 0.5, 'No Niño3.4 data available', ha='center', va='center', transform=ax_enso.transAxes)
        ax_enso.set_ylabel('Niño3.4', size=14)

    # Shared x formatting: show years on Niño3.4 axis only
    plt.setp(ax.get_xticklabels(), visible=False)
    ax_enso.xaxis_date()
    ax_enso.xaxis.set_major_locator(major_locator)
    ax_enso.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax_enso.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_enso.set_xlabel('Year', size=16)
    plt.setp(ax_enso.get_xticklabels(), ha='center', size=font_size)

    # Final spacing
    fig.align_ylabels([ax, ax_enso])
    fig.subplots_adjust(top=0.92, bottom=0.08)

    # IMPORTANT: nudge/resize the colorbar AFTER subplots_adjust, otherwise it gets overwritten.
    # Figure coordinates are in [0, 1]; small values like 0.01–0.03 are typical.
    cax_pos = cax.get_position()
    y_nudge = -0.1       # move colorbar down
    thin_factor = 0.75    # <1 makes it thinner
    cax.set_position([
        cax_pos.x0,
        cax_pos.y0 + y_nudge,
        cax_pos.width,
        cax_pos.height * thin_factor,
    ])
    
    if savefile:
        # Save to output directory
        save_path = os.path.join(output_dir, savefile)
        fig.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()

    ds.close()
    return fig, ax

# Plot AAM anomalies for the specified climatology period
# Build filename suffix based on events JSON
filename_suffix = "_wENSO"
if args.events_json:
    filename_suffix += "_wEvents"
    # Append constraint tag read from the JSON (populated inside plot_AAM_anomalies)
    # We do a quick re-read here just to build the filename before calling the plot.
    try:
        with open(args.events_json) as _f:
            _cfg = json.load(_f).get('config', {})
        _ftags = []
        if _cfg.get('winter_constraint'):  _ftags.append('winter')
        if _cfg.get('el_nino_constraint'): _ftags.append('elnino')
        if _cfg.get('sym_constraint'):     _ftags.append('sym')
        if _ftags:
            filename_suffix += "_" + "_".join(_ftags)
    except Exception:
        pass

plot_AAM_anomalies(
    start_yr, end_yr, component='AAM',
    savefile=f'AAM_anomalies_{ensemble_member}_{start_yr}-{end_yr}_{args.p_min}-{args.p_max}hPa{filename_suffix}.png',
    show=False,  # Don't block on display; file is saved
    events_json=args.events_json,
    plot_enso=not args.no_enso,
)
print("Plot complete. File saved successfully.")
