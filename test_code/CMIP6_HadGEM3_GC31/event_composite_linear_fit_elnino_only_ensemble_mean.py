"""
Usage
- Run from AAM/test_code/ with:
python event_composite_linear_fit_elnino_only.py --start-year 1850 --end-year 2010 --rolling-period 3

To detect La Niña events instead of El Niño, use:
python event_composite_linear_fit_elnino_only.py --start-year 1850 --end-year 2010 --rolling-period 1 --enso-state la_nina --nino-threshold -0.5

To restrict composite to events that onset in NDJFM only:
python event_composite_linear_fit_elnino_only.py --start-year 1850 --end-year 2010 --rolling-period 3 --onset-season ndjfm

To composite 24 months starting from December of each onset year:
python event_composite_linear_fit_elnino_only.py --start-year 1850 --end-year 2010 --composite-months 24 --composite-start december_onset_year

To detect La Niña events that onset in NDJFM and composite 24 months from onset month:
python event_composite_linear_fit_elnino_only.py --start-year 1850 --end-year 2010 --member 1 --composite-start onset --onset-season ndjfm --enso-state la_nina --nino-threshold -0.5
Reference 
Hardiman et al., 2025
https://doi.org/10.1038/s41612-025-01283-7
"""
import xarray as xr
import numpy as np
import pandas as pd
import json
import traceback
# Allow importing shared utilities from AAM/test_code
import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utilities import _to_per_latitude_band, _reindex_to_climatology_dims, vertical_sum_over_pressure_range, get_ENSO_index, pressure_range_in_coord_units
from plotting_utils import plot_latitude_level_snapshots_HadGEN3, plot_lat_lon_snapshots
import tqdm
from scipy import stats as _stats
from matplotlib import pyplot as plt

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Note: joblib not installed. For CPU-bound parallelization, install: pip install joblib")
    sys.exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=150.0, help='Minimum pressure level (hPa) to include (default: 0 hPa)')
parser.add_argument('--p-max', type=float, default=700, help='Maximum pressure level (hPa) to include (default: 1020 hPa)')
parser.add_argument('--start-year', type=int, default=1850, help='Start year to plot (default: 1850)')
parser.add_argument('--end-year', type=int, default=2010, help='End year to plot (default: 2010)')
parser.add_argument('--enso-state', type=str, default='el_nino', choices=['el_nino', 'la_nina'], help='ENSO state to detect (default: el_nino)')
parser.add_argument('--nino-threshold', type=float, default=0.5, help='Nino3.4 threshold for El Nino state detection (default: 0.5)')
parser.add_argument('--min-elnino-months', type=int, default=3, help='Minimum consecutive months above threshold to define El Nino state (default: 3)')
parser.add_argument('--rolling-period', type=int, default=1, help='Rolling-month window for composite analysis (default: 1 = no rolling; 3 gives DJF/JFM/FMA labels)')
parser.add_argument('--composite-months', type=int, default=24,
                    help='Number of months to composite for each event window (default: 24)')
parser.add_argument('--composite-start', type=str, default='onset', choices=['onset', 'december_onset_year'],
                    help='Composite window start mode: onset month, or December of onset year (default: onset)')
parser.add_argument('--onset-season', type=str, default='all', choices=['all', 'ndjfm'],
                    help='Only composite events whose onset month falls in the given season (default: all; ndjfm = Nov–Mar only)')
parser.add_argument(
    '--region',
    type=str,
    default='all',
    choices=['all', 'pacific', 'indian', 'atlantic'],
    help='Geographic region to analyze (default: all). Pacific: 125–(-110)°, Indian: 50–100°, Atlantic: -60–10°',
)
args = parser.parse_args()

n_cpus_to_use = 20
# Define region longitude bounds (degrees East; negative = West)
REGION_BOUNDS = {
    'all': None,  # no filtering
    'pacific': (125, -110),      # 125°E to 110°W
    'indian': (50, 100),         # 50°E to 100°E
    'atlantic': (-60, 10),       # 60°W to 10°E
}

def _select_region(da, region):
    """Select a geographic region from a DataArray by longitude bounds.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data with a longitude dimension
    region : str
        Region name ('all', 'pacific', 'indian', 'atlantic')
    
    Returns
    -------
    xr.DataArray
        Subset of da containing only the selected region's longitudes
    """
    if region == 'all':
        return da
    
    # Normalize dimension names
    da_renamed = da.copy()
    if 'longitude' in da_renamed.dims and 'lon' not in da_renamed.dims:
        da_renamed = da_renamed.rename({'longitude': 'lon'})
    if 'latitude' in da_renamed.dims and 'lat' not in da_renamed.dims:
        da_renamed = da_renamed.rename({'latitude': 'lat'})
    
    lon_min, lon_max = REGION_BOUNDS[region]
    
    if 'lon' not in da_renamed.dims:
        # No longitude dimension, return as-is
        return da_renamed
    
    lon_vals = da_renamed['lon'].values
    orig_lon_range = (float(lon_vals.min()), float(lon_vals.max()))
    
    # Handle wrapping: if lon_min > lon_max and crosses dateline (Pacific case)
    if lon_min > lon_max:
        # Select lon >= lon_min OR lon <= lon_max (crosses dateline)
        mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)
    else:
        # Standard case: lon_min < lon_max
        mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    
    result = da_renamed.isel({'lon': mask})
    new_lon_range = (float(result['lon'].values.min()), float(result['lon'].values.max())) if result.sizes['lon'] > 0 else (None, None)
    
    # Debug output
    if result.sizes['lon'] > 0:
        print(f"[SELECT_REGION] {region}: original lon {orig_lon_range}, target [{lon_min}, {lon_max}], selected {result.sizes['lon']} lons in range {new_lon_range}")
    
    return result

base_dir = os.getcwd()
# AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"

#climatology_path_base = f"{base_dir}/climatology/"

# CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul" 
# nino34_directory = f"{CMIP6_path_base}/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/"
# output_dir = f"{base_dir}/figures/composites/non_tracking_algorithm/"

# Use scratch space and new directory structure due to workspace migration
CMIP6_path_base = "/work/scratch-nopw2/hhhn2"
nino34_directory = f"{CMIP6_path_base}/HadGEM3-GC31-LL/ProcessedFlds/Omon/sst_indices/nino34/historical/"
output_dir = f"{base_dir}/figures/composites/non_tracking_algorithm/with_wind_overlay/strong_events/"
climatology_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/climatology/"
AAM_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/full/"
u_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/Amon/ua/historical"
uv_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/Amon/uv/historical"
ensemble_mean_output_path = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/ensemble_mean_composite/" 
u_level_to_plot = 250.0  # hPa
save_ensemble_mean_netcdf = True  # Save region-specific netCDF files for each run
replot = False  # If True, skip composite calculation and just replot from saved ensemble mean NetCDF

if replot:
    save_ensemble_mean_netcdf = False

def detect_enso_state_windows(
    *,
    start_yr: int,
    end_yr: int,
    ensemble_member: str,
    enso_state: str = "el_nino",
    threshold: float = 0.5,
    min_consecutive_months: int = 3,
    onset_months: "set[int] | None" = None,
) -> list[tuple[str, str]]:
    """Return (onset, end) windows from first month of sustained ENSO state.

    For `enso_state='el_nino'`, onset is the first month in a run where
    Nino3.4 >= threshold for at least `min_consecutive_months`.
    For `enso_state='la_nina'`, onset is the first month in a run where
    Nino3.4 <= threshold for at least `min_consecutive_months`.

    The returned window spans onset month through onset+11.

    If `onset_months` is provided (e.g. {11, 12, 1, 2, 3} for NDJFM), only
    events whose onset falls in one of those calendar months are returned.
    """
    import pandas as pd

    if min_consecutive_months < 1:
        raise ValueError("min_consecutive_months must be >= 1")
    if enso_state not in ("el_nino", "la_nina"):
        raise ValueError("enso_state must be 'el_nino' or 'la_nina'")
    if enso_state == "la_nina" and float(threshold) > -0.5:
        raise ValueError("For la_nina, threshold must be lower than -0.5")

    enso_times, enso_vals = get_ENSO_index(start_yr, end_yr - 1, ensemble_member=ensemble_member)
    if enso_times is None or enso_vals is None:
        raise RuntimeError(f"No Nino3.4 file found for member {ensemble_member}")

    enso_da = xr.DataArray(enso_vals, coords={"time": enso_times}, dims=("time",)).sortby("time")
    enso_vals_arr = np.asarray(enso_da.values, dtype=float)
    if enso_state == "el_nino":
        in_state = enso_vals_arr >= float(threshold)
    else:
        in_state = enso_vals_arr <= float(threshold)

    windows: list[tuple[str, str]] = []
    i = 0
    n = in_state.size
    while i < n:
        if not in_state[i]:
            i += 1
            continue
        j = i
        while j < n and in_state[j]:
            j += 1
        run_len = j - i
        if run_len >= int(min_consecutive_months):
            onset_ts = pd.Timestamp(enso_da["time"].values[i]).to_period("M").to_timestamp()
            end_ts = onset_ts + pd.DateOffset(months=11)
            windows.append((onset_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))
        i = j

    if onset_months is not None:
        windows = [
            (onset, end) for onset, end in windows
            if int(onset[5:7]) in onset_months
        ]

    return windows


def _time_value_to_ymd_string(t) -> str:
    """Convert numpy/cftime-like time scalar to YYYY-MM-DD string."""
    if isinstance(t, np.datetime64):
        return str(np.datetime_as_string(t, unit="D"))
    if hasattr(t, "year") and hasattr(t, "month"):
        day = getattr(t, "day", 1)
        return f"{int(t.year):04d}-{int(t.month):02d}-{int(day):02d}"
    return str(t)


def _compute_composite_window_from_onset(
    onset_str: str,
    *,
    composite_months: int,
    composite_start: str,
) -> tuple[str, str]:
    """Return (window_start_ym, window_end_ym) for a detected onset string."""
    import pandas as pd

    if int(composite_months) < 1:
        raise ValueError("composite_months must be >= 1")
    if composite_start not in ("onset", "december_onset_year"):
        raise ValueError("composite_start must be 'onset' or 'december_onset_year'")

    onset_year = int(onset_str[:4])
    onset_month = int(onset_str[5:7])

    if composite_start == "december_onset_year":
        t_start = pd.Timestamp(f"{onset_year}-12-01")
    else:
        t_start = pd.Timestamp(f"{onset_year}-{onset_month:02d}-01")

    t_end = t_start + pd.DateOffset(months=int(composite_months) - 1)
    return t_start.strftime("%Y-%m"), t_end.strftime("%Y-%m")

def _circular_rolling_mean(da: xr.DataArray, *, dim: str, window: int) -> xr.DataArray:
    """Circular rolling mean over `dim` to preserve Jan/Dec continuity."""
    if window <= 1:
        return da
    n = int(da.sizes[dim])
    if window > n:
        raise ValueError(f"rolling_period ({window}) cannot exceed number of {dim} bins ({n})")
    left = window // 2
    right = window - left - 1
    rolled = [da.roll({dim: -offset}, roll_coords=False) for offset in range(-left, right + 1)]
    return xr.concat(rolled, dim="_roll").mean("_roll", skipna=True)

def _rolling_tick_labels(window: int, n_bins: int = 12) -> list[str]:
    """Return month-window labels for axis ticks (e.g., DJF, JFM, FMA for window=3)."""
    month_initials = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    if window <= 1:
        return [f"M{m:02d}" for m in range(1, n_bins + 1)]
    left = window // 2
    right = window - left - 1
    labels = []
    for center in range(n_bins):
        parts = [month_initials[(center + off) % 12] for off in range(-left, right + 1)]
        labels.append("".join(parts))
    return labels

def _prepare_field(da, p_min_hpa, p_max_hpa):
    # zonal mean ONLY if lon exists
    for lon_name in ("longitude", "lon"):
        if lon_name in da.dims:
            da = da.mean(dim=lon_name, skipna=True)

    # vertical integration if level exists
    if "level" in da.dims:
        da = vertical_sum_over_pressure_range(
            da, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )
    return da

from scipy.ndimage import gaussian_filter

def compute_composite_field(field, date_list, args) -> xr.DataArray:
    """Compute composite field for some variables excluding AAM anomaly 
    for given date_list and args, without plotting."""
    stacked = []
    seen = set()

    for onset_str, _ in date_list:
        if not isinstance(onset_str, str):
            onset_str = _time_value_to_ymd_string(onset_str)

        ym = onset_str[:7]
        if ym in seen:
            continue
        seen.add(ym)

        window_start, window_end = _compute_composite_window_from_onset(
            onset_str,
            composite_months=int(args.composite_months),
            composite_start=str(args.composite_start),
        )

        evt = field.sel(time=slice(window_start, window_end))
        if int(evt.sizes["time"]) < int(args.composite_months):
            continue

        evt = evt.isel(time=slice(0, int(args.composite_months)))
        evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1))
        evt = evt.rename({"time": "month"})

        stacked.append(evt)

    if not stacked:
        return None

    stack = xr.concat(stacked, dim="event")

    # rolling
    if args.rolling_period > 1:
        stack = _circular_rolling_mean(stack, dim="month", window=args.rolling_period)

    return stack.mean("event", skipna=True)

def _process_single_ensemble_member(
    ensemble_member,
    clim_start_yr,
    clim_end_yr,
    args,
    AAM_data_path_base,
    climatology_path_base,
    u_data_path_base,
    uv_data_path_base,
    u_level_to_plot,
):
    """Process a single ensemble member and return results as a dictionary.
    
    Returns dict with keys: 'composite', 'lat_lev_composite', 'lat_lon_composite',
    'u_latlon', 'u_latlev', 'uv_vi', 'onset_dates', 'onset_members', 'peak_amplitudes'
    """
    results = {
        'composite': None,
        'lat_lev_composite': None,
        'lat_lon_composite': None,
        'u_latlon': None,
        'u_latlev': None,
        'uv_vi': None,
        'onset_dates': [],
        'onset_members': [],
        'peak_amplitudes': [],
    }
    
    try:
        # Load AAM
        AAM_da = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")['AAM']
        
        # Normalize dimension names immediately after loading
        if 'latitude' in AAM_da.dims and 'lat' not in AAM_da.dims:
            AAM_da = AAM_da.rename({'latitude': 'lat'})
        if 'longitude' in AAM_da.dims and 'lon' not in AAM_da.dims:
            AAM_da = AAM_da.rename({'longitude': 'lon'})
        
        # DEBUG: Print what region is requested
        if ensemble_member == "r1i1p1f3":
            print(f"\n[ENSEMBLE MEMBER DEBUG] Processing {ensemble_member}")
            print(f"  Region requested: {args.region}")
            print(f"  Full AAM data shape before selection: {AAM_da.shape}")
            lon_coord = 'lon' if 'lon' in AAM_da.dims else 'longitude'
            print(f"  Full AAM lon range: [{float(AAM_da[lon_coord].values.min()):.1f}, {float(AAM_da[lon_coord].values.max()):.1f}]")
        
        # Apply region selection BEFORE zonal integration so ensemble mean is also region-specific
        AAM_da = _select_region(AAM_da, args.region)
        
        if "longitude" in AAM_da.dims or "lon" in AAM_da.dims:
            AAM_da = _to_per_latitude_band(AAM_da)
            
        u_da = xr.open_dataset(f"{u_data_path_base}/ua_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")['ua']
        # Normalize dimension names
        if 'latitude' in u_da.dims and 'lat' not in u_da.dims:
            u_da = u_da.rename({'latitude': 'lat'})
        if 'longitude' in u_da.dims and 'lon' not in u_da.dims:
            u_da = u_da.rename({'longitude': 'lon'})
        
        uv_da = xr.open_dataset(f"{uv_data_path_base}/uv_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")['uv']
        # Normalize dimension names
        if 'latitude' in uv_da.dims and 'lat' not in uv_da.dims:
            uv_da = uv_da.rename({'latitude': 'lat'})
        if 'longitude' in uv_da.dims and 'lon' not in uv_da.dims:
            uv_da = uv_da.rename({'longitude': 'lon'})

        # Climatology
        clim_da = xr.open_dataset(
            f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
        
        # Normalize dimension names immediately after loading
        if 'latitude' in clim_da.dims and 'lat' not in clim_da.dims:
            clim_da = clim_da.rename({'latitude': 'lat'})
        if 'longitude' in clim_da.dims and 'lon' not in clim_da.dims:
            clim_da = clim_da.rename({'longitude': 'lon'})

        # Select region before zonal mean so that zonal mean is computed over regional lons only
        u_da_regional = _select_region(u_da, args.region)
        u_da_zonal_band = u_da_regional.mean(dim="longitude", skipna=True) if "longitude" in u_da_regional.dims else u_da_regional
        
        # Use the per-latitude-band + zonal-integral climatology
        # CRITICAL: Apply region selection to climatology BEFORE zonal integration to match main AAM data
        clim_aam_data = clim_da['AAM']
        clim_aam_data = _select_region(clim_aam_data, args.region)
        if 'longitude' in clim_aam_data.dims or 'lon' in clim_aam_data.dims:
            clim_da = _to_per_latitude_band(clim_aam_data)
        else:
            clim_da = clim_aam_data
        
        # Detect ENSO state windows from Nino3.4
        _onset_months_map = {"all": None, "ndjfm": {11, 12, 1, 2, 3}}
        onset_months_filter = _onset_months_map[args.onset_season]
        date_list = detect_enso_state_windows(
            start_yr=int(args.start_year),
            end_yr=int(args.end_year),
            ensemble_member=ensemble_member,
            enso_state=str(args.enso_state),
            threshold=float(args.nino_threshold),
            min_consecutive_months=int(args.min_elnino_months),
            onset_months=onset_months_filter,
        )
        
        if not date_list:
            return results
            
        # Collect onset dates for histogram
        for _onset, _end in date_list:
            onset_str = _onset if isinstance(_onset, str) else _time_value_to_ymd_string(_onset)
            results['onset_dates'].append(onset_str)
            results['onset_members'].append(ensemble_member)
        
        # Record event peak Nino3.4 amplitudes
        try:
            enso_times_all, enso_vals_all = get_ENSO_index(int(args.start_year), int(args.end_year) - 1, ensemble_member=ensemble_member)
            if enso_times_all is not None and enso_vals_all is not None:
                enso_index = pd.to_datetime(enso_times_all)
                enso_series = pd.Series(np.asarray(enso_vals_all, dtype=float), index=enso_index.to_period('M'))
                for _onset, _end in date_list:
                    onset_str = _onset if isinstance(_onset, str) else _time_value_to_ymd_string(_onset)
                    window_start, window_end = _compute_composite_window_from_onset(
                        onset_str,
                        composite_months=int(args.composite_months),
                        composite_start=str(args.composite_start),
                    )
                    start_period = pd.Period(window_start, freq='M')
                    end_period = pd.Period(window_end, freq='M')
                    ev_vals = enso_series.loc[start_period:end_period].values
                    if ev_vals.size == 0:
                        continue
                    if args.enso_state == 'el_nino':
                        peak = float(np.nanmax(ev_vals))
                    else:
                        peak = float(abs(np.nanmin(ev_vals)))
                    results['peak_amplitudes'].append(peak)
        except Exception:
            pass
        
        # Process fields
        p_selected, _ = pressure_range_in_coord_units(u_da.plev, p_min_hpa=u_level_to_plot, p_max_hpa=float(args.p_max))
        u_level = u_da.sel(plev=p_selected, method="nearest")
        u_zm = u_da_zonal_band
        
        aam_latlon = vertical_sum_over_pressure_range(AAM_da, p_min_hpa=float(args.p_min), p_max_hpa=float(args.p_max), level_dim="level")
        aam_latlev = AAM_da.sum(dim="longitude", skipna=True) if "longitude" in AAM_da.dims else AAM_da
        uv_vi = vertical_sum_over_pressure_range(uv_da, p_min_hpa=float(args.p_min), p_max_hpa=float(args.p_max), level_dim="plev")
        
        u_latlon_comp = compute_composite_field(u_level, date_list=date_list, args=args)
        u_latlvl_zm_comp = compute_composite_field(u_zm, date_list=date_list, args=args)
        uv_latlev_comp = compute_composite_field(uv_vi, date_list, args)
        
        if u_latlon_comp is not None:
            results['u_latlon'] = u_latlon_comp.expand_dims({"ensemble": [ensemble_member]})
        if u_latlvl_zm_comp is not None:
            results['u_latlev'] = u_latlvl_zm_comp.expand_dims({"ensemble": [ensemble_member]})
        if uv_latlev_comp is not None:
            results['uv_vi'] = uv_latlev_comp.expand_dims({"ensemble": [ensemble_member]})
        
        # Step 4: Composite AAM anomaly
        comp = composite_propagating_years_no_plot(
            AAM_da,
            wind_da=None,
            date_list=date_list,
            clim_da=clim_da,
            clim_start_yr=clim_start_yr,
            clim_end_yr=clim_end_yr,
            p_min_hpa=float(args.p_min),
            p_max_hpa=float(args.p_max),
            enso_state=str(args.enso_state),
            rolling_period=int(args.rolling_period),
            composite_months=int(args.composite_months),
            composite_start=str(args.composite_start),
            onset_season=str(args.onset_season),
        )
        if comp is not None:
            results['composite'] = comp.expand_dims({"ensemble": [ensemble_member]})
        
        # Step 5: Latitude×level composite
        aam_full = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
        aam_full = _select_region(aam_full, args.region)
        if "longitude" in aam_full.dims or "lon" in aam_full.dims:
            aam_full = _to_per_latitude_band(aam_full)
        
        clim_full = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
        # Normalize dimensions
        if 'latitude' in clim_full.dims and 'lat' not in clim_full.dims:
            clim_full = clim_full.rename({'latitude': 'lat'})
        if 'longitude' in clim_full.dims and 'lon' not in clim_full.dims:
            clim_full = clim_full.rename({'longitude': 'lon'})
        
        aam_full_reindexed, clim_on_time_full = _reindex_to_climatology_dims(aam_full, clim_full)
        anom_full = aam_full_reindexed - clim_on_time_full
        
        stacked_full = []
        seen_ev = set()
        for onset_str, _ in date_list:
            if not isinstance(onset_str, str):
                onset_str = _time_value_to_ymd_string(onset_str)
            ym = onset_str[:7]
            if ym in seen_ev:
                continue
            seen_ev.add(ym)
            window_start, window_end = _compute_composite_window_from_onset(
                onset_str,
                composite_months=int(args.composite_months),
                composite_start=str(args.composite_start),
            )
            evt = anom_full.sel(time=slice(window_start, window_end))
            if int(evt.sizes["time"]) < int(args.composite_months):
                continue
            evt = evt.isel(time=slice(0, int(args.composite_months)))
            evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1, dtype=int))
            if "month" in evt.coords:
                evt = evt.drop_vars("month")
            evt = evt.rename({"time": "month"})
            stacked_full.append(evt)
        
        if stacked_full:
            full_stack = xr.concat(stacked_full, dim="event")
            rp = int(args.rolling_period)
            if rp > 1:
                n_month = int(full_stack.sizes["month"])
                if rp <= n_month:
                    left = rp // 2
                    right = rp - left - 1
                    _rolled = [full_stack.roll(month=-offset, roll_coords=False) for offset in range(-left, right + 1)]
                    full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)
            
            composite_full = full_stack.mean("event", skipna=True)
            composite_full = composite_full.rename({"month": "time"})
            
            # Normalize dimension names before storing in results
            if 'latitude' in composite_full.dims and 'lat' not in composite_full.dims:
                composite_full = composite_full.rename({'latitude': 'lat'})
            
            results['lat_lev_composite'] = composite_full.expand_dims({"ensemble": [ensemble_member]})
        
        # Step 6: LAT×LON composite
        aam_full_latlon = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")['AAM']
        # Normalize dimension names immediately after loading
        if 'latitude' in aam_full_latlon.dims and 'lat' not in aam_full_latlon.dims:
            aam_full_latlon = aam_full_latlon.rename({'latitude': 'lat'})
        if 'longitude' in aam_full_latlon.dims and 'lon' not in aam_full_latlon.dims:
            aam_full_latlon = aam_full_latlon.rename({'longitude': 'lon'})
        
        clim_full_latlon = xr.open_dataset(f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
        # Normalize dimension names immediately after loading
        if 'latitude' in clim_full_latlon.dims and 'lat' not in clim_full_latlon.dims:
            clim_full_latlon = clim_full_latlon.rename({'latitude': 'lat'})
        if 'longitude' in clim_full_latlon.dims and 'lon' not in clim_full_latlon.dims:
            clim_full_latlon = clim_full_latlon.rename({'longitude': 'lon'})
        
        if ensemble_member == "r1i1p1f3":
            print(f"\n[LAT×LON DEBUG] Processing lat×lon composite for {ensemble_member}")
            lon_coord = 'lon' if 'lon' in aam_full_latlon.dims else 'longitude'
            print(f"  aam_full_latlon BEFORE region selection: shape={aam_full_latlon.shape}, lon range=[{float(aam_full_latlon[lon_coord].values.min()):.1f}, {float(aam_full_latlon[lon_coord].values.max()):.1f}]")
        
        aam_full_latlon = _select_region(aam_full_latlon, args.region)
        
        if ensemble_member == "r1i1p1f3":
            lon_coord = 'lon' if 'lon' in aam_full_latlon.dims else 'longitude'
            print(f"  aam_full_latlon AFTER region selection: shape={aam_full_latlon.shape}, lon range=[{float(aam_full_latlon[lon_coord].values.min()):.1f}, {float(aam_full_latlon[lon_coord].values.max()):.1f}]")
        aam_vs = vertical_sum_over_pressure_range(aam_full_latlon, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
        
        # CRITICAL: Apply region selection to climatology BEFORE vertical sum to match main data
        clim_full_data = clim_full_latlon["AAM"] if isinstance(clim_full_latlon, xr.Dataset) and "AAM" in clim_full_latlon else clim_full_latlon
        clim_full_data = _select_region(clim_full_data, args.region)
        clim_vs = vertical_sum_over_pressure_range(clim_full_data, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
        
        # Normalize dimensions to match main data
        if 'latitude' in aam_vs.dims and 'lat' not in aam_vs.dims:
            aam_vs = aam_vs.rename({'latitude': 'lat'})
        if 'longitude' in aam_vs.dims and 'lon' not in aam_vs.dims:
            aam_vs = aam_vs.rename({'longitude': 'lon'})
        if 'latitude' in clim_vs.dims and 'lat' not in clim_vs.dims:
            clim_vs = clim_vs.rename({'latitude': 'lat'})
        if 'longitude' in clim_vs.dims and 'lon' not in clim_vs.dims:
            clim_vs = clim_vs.rename({'longitude': 'lon'})
        
        aam_full_latlon_r, clim_on_time = _reindex_to_climatology_dims(aam_vs, clim_vs)
        anom_full_latlon = aam_full_latlon_r - clim_on_time
        
        stacked_full = []
        seen_ev = set()
        for onset_str, _ in date_list:
            if not isinstance(onset_str, str):
                onset_str = _time_value_to_ymd_string(onset_str)
            ym = onset_str[:7]
            if ym in seen_ev:
                continue
            seen_ev.add(ym)
            window_start, window_end = _compute_composite_window_from_onset(
                onset_str,
                composite_months=int(args.composite_months),
                composite_start=str(args.composite_start),
            )
            evt = anom_full_latlon.sel(time=slice(window_start, window_end))
            if int(evt.sizes["time"]) < int(args.composite_months):
                continue
            evt = evt.isel(time=slice(0, int(args.composite_months)))
            evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1, dtype=int))
            if "month" in evt.coords:
                evt = evt.drop_vars("month")
            evt = evt.rename({"time": "month"})
            stacked_full.append(evt)
        
        if stacked_full:
            full_stack = xr.concat(stacked_full, dim="event")
            rp = int(args.rolling_period)
            if rp > 1:
                n_month = int(full_stack.sizes["month"])
                if rp <= n_month:
                    left = rp // 2
                    right = rp - left - 1
                    _rolled = [full_stack.roll(month=-offset, roll_coords=False) for offset in range(-left, right + 1)]
                    full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)
            
            composite_full = full_stack.mean("event", skipna=True)
            composite_full = composite_full.rename({"month": "time"})
            
            # Normalize dimension names before storing in results
            if 'latitude' in composite_full.dims and 'lat' not in composite_full.dims:
                composite_full = composite_full.rename({'latitude': 'lat'})
            if 'longitude' in composite_full.dims and 'lon' not in composite_full.dims:
                composite_full = composite_full.rename({'longitude': 'lon'})
            
            results['lat_lon_composite'] = composite_full.expand_dims({"ensemble": [ensemble_member]})
        
    except Exception as e:
        print(f"Error processing ensemble member {ensemble_member}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def composite_propagating_years_no_plot(
    AAM_da,
    wind_da,
    date_list,
    clim_da=None,
    *,
    clim_start_yr: int = 1980,
    clim_end_yr: int = 2000,
    p_min_hpa: float = 150.0,
    p_max_hpa: float = 700.0,
    enso_state: str = "el_nino",
    rolling_period: int = 1,
    composite_months: int = 24,
    composite_start: str = "onset",
    nlevels: int = 13,
    onset_season: str = "all",
) -> xr.DataArray:
    """Composite AAM/variable anomalies for ENSO event onset windows.

    For each (onset_time, end_time) entry in date_list, a composite_months window
    starting at the onset month is extracted and stacked. Duplicate onset months
    are composited only once. The result is a mean anomaly aligned on
    relative month (1..N, where N = composite_months).

    Parameters
    ----------
    AAM_da : xr.DataArray or xr.Dataset  (time, level, latitude[, longitude])
    wind_da : xr.DataArray or xr.Dataset | None
    date_list : list of (onset, end) time-like tuples
    clim_da : xr.DataArray or xr.Dataset, optional
        Monthly climatology with a 'month' dim (1–12).  When None, a simple
        climatology is derived from the full AAM_da time range.
    """
    if not date_list:
        print("composite_propagating_years: date_list is empty, nothing to composite.")
        return

    rolling_period = int(rolling_period)
    if rolling_period < 1:
        raise ValueError("rolling_period must be >= 1")
    composite_months = int(composite_months)
    if composite_months < 1:
        raise ValueError("composite_months must be >= 1")
    if composite_start not in ("onset", "december_onset_year"):
        raise ValueError("composite_start must be 'onset' or 'december_onset_year'")

    # --- Unwrap Datasets ---
    AAM_field = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
    if isinstance(AAM_field, xr.Dataset):
        AAM_field = next(iter(AAM_field.data_vars.values()))
    
    # --- Normalize dimension names to ensure consistency ---
    if 'latitude' in AAM_field.dims and 'lat' not in AAM_field.dims:
        AAM_field = AAM_field.rename({'latitude': 'lat'})
    if 'longitude' in AAM_field.dims and 'lon' not in AAM_field.dims:
        AAM_field = AAM_field.rename({'longitude': 'lon'})

    wind_field = None
    if wind_da is not None:
        wind_field = wind_da["ua"] if isinstance(wind_da, xr.Dataset) and "ua" in wind_da else wind_da
        if isinstance(wind_field, xr.Dataset):
            wind_field = next(iter(wind_field.data_vars.values()))
        
        # --- Normalize wind field dimension names ---
        if 'latitude' in wind_field.dims and 'lat' not in wind_field.dims:
            wind_field = wind_field.rename({'latitude': 'lat'})
        if 'longitude' in wind_field.dims and 'lon' not in wind_field.dims:
            wind_field = wind_field.rename({'longitude': 'lon'})

    # --- Zonal integral for AAM per latitude band
    # and zonal mean for wind (intensive quantity) ---
    # if "longitude" in AAM_field.dims or "lon" in AAM_field.dims:
    #     AAM_field = _to_per_latitude_band(AAM_field)
    if wind_field is not None:
        for lon_name in ("longitude", "lon"):
            if lon_name in wind_field.dims:
                wind_field = wind_field.mean(dim=lon_name, skipna=True)

    # --- Vertical integration over the requested pressure range ---
    AAM_field = vertical_sum_over_pressure_range(
        AAM_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
    )
    if wind_field is not None:
        wind_field = vertical_sum_over_pressure_range(
            wind_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )

    # --- Compute anomalies ---
    if clim_da is not None:
        clim_field = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
        if isinstance(clim_field, xr.Dataset):
            clim_field = next(iter(clim_field.data_vars.values()))
        # # Only sum over longitude if it still exists (not already zonally integrated)
        # if "longitude" in clim_field.dims:
        #     clim_field = clim_field.sum(dim="longitude", skipna=True)
        # elif "lon" in clim_field.dims:
        #     clim_field = clim_field.sum(dim="lon", skipna=True)
        clim_field = vertical_sum_over_pressure_range(
            clim_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )
        AAM_anom = AAM_field.groupby("time.month") - clim_field
    else:
        clim_period = AAM_field.sel(time=slice(f"{clim_start_yr}-01", f"{clim_end_yr}-12"))
        clim_inline = clim_period.groupby("time.month").mean("time")
        AAM_anom = AAM_field.groupby("time.month") - clim_inline
    #import pdb; pdb.set_trace()
    
    # --- Extract composite window for each event ---
    stacked_AAM = []
    stacked_wind = []
    seen_onset_months: set = set()  # deduplicate by onset year-month, not calendar year
    for onset_time, _end_time in date_list:
        onset_str = (
            _time_value_to_ymd_string(onset_time)
            if not isinstance(onset_time, str)
            else onset_time
        )
        onset_ym = onset_str[:7]  # "YYYY-MM"
        if onset_ym in seen_onset_months:
            print(f"  composite: onset {onset_ym} already included, skipping duplicate.")
            continue
        seen_onset_months.add(onset_ym)

        window_start, window_end = _compute_composite_window_from_onset(
            onset_str,
            composite_months=composite_months,
            composite_start=composite_start,
        )

        aam_event = AAM_anom.sel(time=slice(window_start, window_end))
        n_avail = int(aam_event.sizes["time"])
        if n_avail < composite_months:
            print(f"  composite: onset {onset_ym} window has only {n_avail} months of data, skipping.")
            continue

        # Label as relative months 1..N
        aam_event = aam_event.isel(time=slice(0, composite_months))
        aam_event = aam_event.assign_coords(time=np.arange(1, composite_months + 1, dtype=int))
        if "month" in aam_event.coords:
            aam_event = aam_event.drop_vars("month")
        aam_event = aam_event.rename({"time": "month"})
        
        # Normalize latitude dimension name for consistency
        if 'latitude' in aam_event.dims and 'lat' not in aam_event.dims:
            aam_event = aam_event.rename({'latitude': 'lat'})
        
        stacked_AAM.append(aam_event)

        if wind_field is not None:
            wind_event = wind_field.sel(time=slice(window_start, window_end))
            n_w = int(wind_event.sizes["time"])
            if n_w < composite_months:
                print(f"  composite: wind onset {onset_ym} window incomplete ({n_w} months), skipping.")
                continue
            wind_event = wind_event.isel(time=slice(0, composite_months))
            wind_event = wind_event.assign_coords(time=np.arange(1, composite_months + 1, dtype=int))
            if "month" in wind_event.coords:
                wind_event = wind_event.drop_vars("month")
            wind_event = wind_event.rename({"time": "month"})
            
            # Normalize latitude dimension name for consistency
            if 'latitude' in wind_event.dims and 'lat' not in wind_event.dims:
                wind_event = wind_event.rename({'latitude': 'lat'})
            
            stacked_wind.append(wind_event)

    n_events = len(stacked_AAM)
    if n_events < 1:
        print("composite_propagating_years: no valid events to composite.")
        return

    _start_desc = "onset month" if composite_start == "onset" else "December of onset year"
    print(f"Compositing {n_events} {composite_months}-month window(s) from {_start_desc}.")
    aam_stack = xr.concat(stacked_AAM, dim="event")
    aam_stack_for_plot = _circular_rolling_mean(aam_stack, dim="month", window=rolling_period)
    composite_AAM = aam_stack_for_plot.mean("event", skipna=True)

    composite_wind = None
    if stacked_wind:
        wind_stack = xr.concat(stacked_wind, dim="event")
        wind_stack = _circular_rolling_mean(wind_stack, dim="month", window=rolling_period)
        composite_wind = wind_stack.mean("event", skipna=True)

    lat_dim = "lat" if "lat" in composite_AAM.dims else ("latitude" if "latitude" in composite_AAM.dims else None)
    if lat_dim is None:
        raise ValueError(f"No latitude dimension found in composite data with dims {composite_AAM.dims}")

    aam_vals = composite_AAM.values  # (month, lat) or (lat, month)
    # ensure shape is (lat, month)
    if composite_AAM.dims[0] == "month":
        aam_vals = aam_vals.T

    # --- T-test: significance vs zero along event dimension ---

    aam_for_ttest = aam_stack_for_plot.transpose("event", lat_dim, "month").values
    _, p_vals = _stats.ttest_1samp(aam_for_ttest, 0.0, axis=0, nan_policy="omit")
    # p_vals is (lat, month), matching aam_vals layout
    print(f"  t-test: shape={p_vals.shape}, min p={float(np.nanmin(p_vals)):.4f}, "
          f"p<0.05: {int(np.sum(p_vals < 0.05))} pts, "
          f"p<0.10: {int(np.sum(p_vals < 0.10))} pts")
    return composite_AAM


if __name__ == '__main__':

    clim_start_yr = 1980
    clim_end_yr = 2000

    ensemble_composites = []
    ensemble_lat_lev_composites = []
    ensemble_lat_lon_composites = []
    ensemble_u_latlon = []
    ensemble_u_latlev = []
    ensemble_uv_vi_lat = []
    available_members = []
    # Collect onset dates across ensemble members for histogram plotting
    onset_dates = []
    onset_members = []
    # Collect per-event peak Nino3.4 amplitudes (positive values)
    event_peak_amplitudes = []
    
    if not replot:
        
        for ensemble_member in [f"r{i}i1p1f3" for i in range(1, 61)]:
        # Use OS to see whether the nc file exists before trying to open with xarray, to avoid long error messages from xarray when files are missing.
            AAM_path = os.path.join(AAM_data_path_base, f"AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")
            AAM_exist = os.path.exists(AAM_path)
            clim_path = os.path.join(climatology_path_base, f"AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
            clim_exist = os.path.exists(clim_path)
            u_path = os.path.join(u_data_path_base, f"ua_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")
            u_exist = os.path.exists(u_path)
            uv_path = os.path.join(uv_data_path_base, f"uv_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")
            uv_exist = os.path.exists(uv_path)
            
            if not AAM_exist or not clim_exist or not u_exist or not uv_exist:
                if ensemble_member in ["r1i1p1f3"]:  # Only print for first member to avoid spam
                    print(f"DEBUG: First member file check: AAM={AAM_exist}, clim={clim_exist}, u={u_exist}, uv={uv_exist}")
                    print(f"  AAM_data_path_base: {AAM_data_path_base}")
                print(f"Skipping member {ensemble_member} because files not found.")
                continue
            else:
                available_members.append(ensemble_member)
                print(f"Processing ensemble member: {ensemble_member}")
        
        # Parallelize ensemble member processing using joblib
        n_jobs = int(n_cpus_to_use) if n_cpus_to_use != -1 else -1
        print(f"Running ensemble processing with {n_jobs} worker(s)...")
        
        if JOBLIB_AVAILABLE and n_jobs != 1:
            # Use parallel processing
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_ensemble_member)(
                    member,
                    clim_start_yr,
                    clim_end_yr,
                    args,
                    AAM_data_path_base,
                    climatology_path_base,
                    u_data_path_base,
                    uv_data_path_base,
                    u_level_to_plot,
                )
                for member in tqdm.tqdm(available_members)
            )
        else:
            # Sequential processing
            results_list = []
            for member in tqdm.tqdm(available_members):
                result = _process_single_ensemble_member(
                    member,
                    clim_start_yr,
                    clim_end_yr,
                    args,
                    AAM_data_path_base,
                    climatology_path_base,
                    u_data_path_base,
                    uv_data_path_base,
                    u_level_to_plot,
                )
                results_list.append(result)
        
        # Combine results from all ensemble members
        for result in results_list:
            if result['composite'] is not None:
                ensemble_composites.append(result['composite'])
            if result['lat_lev_composite'] is not None:
                ensemble_lat_lev_composites.append(result['lat_lev_composite'])
            if result['lat_lon_composite'] is not None:
                ensemble_lat_lon_composites.append(result['lat_lon_composite'])
            if result['u_latlon'] is not None:
                ensemble_u_latlon.append(result['u_latlon'])
            if result['u_latlev'] is not None:
                ensemble_u_latlev.append(result['u_latlev'])
            if result['uv_vi'] is not None:
                ensemble_uv_vi_lat.append(result['uv_vi'])
            
            onset_dates.extend(result['onset_dates'])
            onset_members.extend(result['onset_members'])
            event_peak_amplitudes.extend(result['peak_amplitudes'])
        
        number_of_available_members = len(available_members)
    
    else:
        # replot=True: Load pre-computed ensemble mean composites from netCDF files
        print("replot=True: Loading pre-computed ensemble mean composites from netCDF files...")
        
        # Construct filenames based on command-line args
        region_label = args.region.upper() if args.region != 'all' else 'GLOBAL'
        
        # Load main composite
        main_nc_path = os.path.join(
            ensemble_mean_output_path,
            f"AAM_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{region_label}_{args.onset_season}_start_{args.composite_start}_ninothres{float(args.nino_threshold)}.nc",
        )
        if os.path.exists(main_nc_path):
            print(f"  Loading main composite from {os.path.basename(main_nc_path)}")
            ens_mean_main = xr.open_dataset(main_nc_path)["AAMA"]
            ensemble_composites.append(ens_mean_main)
        else:
            print(f"  WARNING: Main composite file not found at {main_nc_path}")
        
        # Load lat×level composite
        latlev_nc_path = os.path.join(
            ensemble_mean_output_path,
            f"AAM_lat_lev_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
        )
        if os.path.exists(latlev_nc_path):
            print(f"  Loading lat×level composite from {os.path.basename(latlev_nc_path)}")
            ens_mean_latlev = xr.open_dataset(latlev_nc_path)["AAMA"]
            ensemble_lat_lev_composites.append(ens_mean_latlev)
            
            # Also load zonal wind for lat×level
            u_latlev_path = os.path.join(
                ensemble_mean_output_path,
                f"U_lat_lev_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
            )
            if os.path.exists(u_latlev_path):
                ens_mean_u_latlev_file = xr.open_dataset(u_latlev_path)["u"]
                ensemble_u_latlev.append(ens_mean_u_latlev_file)
        else:
            print(f"  WARNING: Lat×level composite file not found at {latlev_nc_path}")
        
        # Load lat×lon composite
        latlon_nc_path = os.path.join(
            ensemble_mean_output_path,
            f"AAM_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
        )
        if os.path.exists(latlon_nc_path):
            print(f"  Loading lat×lon composite from {os.path.basename(latlon_nc_path)}")
            ens_mean_latlon = xr.open_dataset(latlon_nc_path)["AAMA"]
            ensemble_lat_lon_composites.append(ens_mean_latlon)
            
            # Also load wind fields for lat×lon
            u_latlon_path = os.path.join(
                ensemble_mean_output_path,
                f"U_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
            )
            if os.path.exists(u_latlon_path):
                ens_mean_u_latlon_file = xr.open_dataset(u_latlon_path)["u"]
                ensemble_u_latlon.append(ens_mean_u_latlon_file)
            
            uv_latlon_path = os.path.join(
                ensemble_mean_output_path,
                f"UV_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
            )
            if os.path.exists(uv_latlon_path):
                ens_mean_uv_latlev_file = xr.open_dataset(uv_latlon_path)["uv"]
                ensemble_uv_vi_lat.append(ens_mean_uv_latlev_file)
        else:
            print(f"  WARNING: Lat×lon composite file not found at {latlon_nc_path}")
        
        number_of_available_members = 0  # placeholder for loaded replot mode
    
    _cmp = ">" if args.enso_state == "el_nino" else "<"
    _snap_season_label = "  |  NDJFM onsets only" if args.onset_season == "ndjfm" else ""
    _snap_suffix = (
                        f"{args.enso_state} state: Nino3.4{_cmp}{float(args.nino_threshold):.2f} "
                        f"for >= {int(args.min_elnino_months)} months"
                        f" | {int(args.composite_months)}-month composite from "
                        f"{'Dec of onset year' if args.composite_start == 'december_onset_year' else 'onset month'}"
                        f"{_snap_season_label}"
                    )
    
    # Determine region label for filenames (used in all composite plots)
    region_label = args.region.upper() if args.region != 'all' else 'GLOBAL'
    
    if ensemble_composites:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            import numpy as np
            from scipy import stats as _stats
            import os

            # -------------------------------
            # Ensemble mean
            # -------------------------------
            ens_stack = xr.concat(ensemble_composites, dim="ensemble")
            ens_mean = ens_stack.mean("ensemble", skipna=True)

            print(f"Plotting ENSEMBLE MEAN composite from {ens_stack.sizes['ensemble']} members...")

            # -------------------------------
            # Extract dims
            # -------------------------------
            lat_dim = "latitude" if "latitude" in ens_mean.dims else "lat"
            lat_vals = ens_mean[lat_dim].values
            month_vals = ens_mean["month"].values

            # Ensure (lat, month)
            aam_vals = ens_mean.values
            if ens_mean.dims[0] == "month":
                aam_vals = aam_vals.T

            # -------------------------------
            # FIXED t-test (CRITICAL)
            # -------------------------------
            # Ensure dimension order: (ensemble, lat, month)
            combined = ens_stack.transpose("ensemble", lat_dim, "month")

            # Now t-test across ensemble axis
            _, p_vals = _stats.ttest_1samp(
                combined.values,
                0.0,
                axis=0,
                nan_policy="omit"
            )

            # p_vals is now (lat, month) ✅

            print(
                f"Ensemble t-test: shape={p_vals.shape}, "
                f"min p={float(np.nanmin(p_vals)):.4f}, "
                f"p<0.05: {int(np.sum(p_vals < 0.05))} pts"
            )

            # Safety check
            if p_vals.shape != (len(lat_vals), len(month_vals)):
                raise ValueError(
                    f"Shape mismatch: p_vals {p_vals.shape}, "
                    f"expected ({len(lat_vals)}, {len(month_vals)})"
                )

            # -------------------------------
            # Plot
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.subplots_adjust(bottom=0.18)
            
            #vmax = 5e24
            vmax = float(np.nanpercentile(np.abs(aam_vals), 98))
            vmax = vmax if vmax > 0 else 1.0
            vmin = -vmax

            levels = np.linspace(vmin, vmax, 13)

            cf = ax.contourf(
                month_vals,
                lat_vals,
                aam_vals,
                levels=levels,
                cmap="RdBu_r",
                extend="both",
            )

            # -------------------------------
            # Colorbar
            # -------------------------------
            _abs = max(abs(vmin), abs(vmax))
            order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
            factor = 10 ** order

            cax = fig.add_axes([0.125, 0.06, 0.775, 0.015])
            cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")

            _sup = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
            _order_sup = str(order).translate(_sup)

            cbar.set_label(f"AAM anomaly (×10{_order_sup})", size=12)

            _tick_levels = cf.levels[::2]
            cbar.set_ticks(_tick_levels)
            cbar.set_ticklabels([f"{v / factor:.1f}" for v in _tick_levels])
            cbar.ax.tick_params(labelsize=11)

            # -------------------------------
            # Significance overlay (FIXED)
            # -------------------------------
            sig_lat_idx, sig_month_idx = np.where(p_vals < 0.05)

            if sig_lat_idx.size > 0:
                ax.scatter(
                    month_vals[sig_month_idx],   # correct
                    lat_vals[sig_lat_idx],       # correct
                    s=20,
                    c="k",
                    marker=".",
                    linewidths=0,
                    zorder=10,
                )
            else:
                print("No significant points (p < 0.05)")

            # -------------------------------
            # Labels
            # -------------------------------
            ax.set_xlabel("Month since onset")
            ax.set_ylabel("Latitude (°N)")

            ax.set_xlim(1, len(month_vals))
            ax.set_ylim(-60, 60)

            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

            state_pretty = "El Nino" if args.enso_state == "el_nino" else "La Nina"

            onset_season_label = f" | {args.onset_season.upper()} onsets only" if args.onset_season != "all" else ""
            
            ax.set_title(
                f"HadGEM3_GC31 {number_of_available_members} members ENSEMBLE MEAN Composite AAM anomaly\n"
                f"({args.p_min}–{args.p_max} hPa) {state_pretty} events  {args.start_year}–{args.end_year}"
                f"clim {clim_start_yr}–{clim_end_yr} {onset_season_label}"
            )

            # -------------------------------
            # Save
            # -------------------------------
            os.makedirs(output_dir, exist_ok=True)

            out_path = os.path.join(
                output_dir,
                f"AAM_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{region_label}_{args.onset_season}_start_{args.composite_start}_ninothres{float(args.nino_threshold)}.png",
            )

            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"Ensemble mean composite plot saved to {out_path}")
            
            if save_ensemble_mean_netcdf:
                # Output the ensemble mean composite data as netCDF for future analysis
                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"AAM_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{region_label}_{args.onset_season}_start_{args.composite_start}_ninothres{float(args.nino_threshold)}.nc",
                )
                ens_mean.name = "AAMA"
                ens_mean.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean.to_netcdf(out_nc_path)
                print(f"Ensemble mean composite data saved to {out_nc_path}")
        except Exception as e:
            print(f"Error plotting ensemble mean composite: {e}")
            import pdb; pdb.set_trace()
    
    if not ensemble_lat_lev_composites:
        print(f"WARNING: No ensemble lat×level composites computed for region '{args.region}'. Skipping lat×level plots.")
    
    if ensemble_lat_lev_composites:
        try:
            ens_stack = xr.concat(ensemble_lat_lev_composites, dim="ensemble")
            ens_mean = ens_stack.mean("ensemble", skipna=True)
            
            ens_stack = xr.concat(ensemble_u_latlev, dim="ensemble")
            ens_mean_u_latlev = ens_stack.mean("ensemble", skipna=True)
            # Rename month dimension to time for compatibility with plotting function
            if "month" in ens_mean_u_latlev.coords and "time" not in ens_mean_u_latlev.coords:
                ens_mean_u_latlev = ens_mean_u_latlev.rename({"month": "time"})

            print(f"Plotting ENSEMBLE MEAN lat×level composite from {ens_stack.sizes['ensemble']} members...")

            plot_latitude_level_snapshots_HadGEN3(
                ens_mean,
                zonal_wind_da=ens_mean_u_latlev,             
                ensemble_member="ENSEMBLE_MEAN",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                output_dir=output_dir,
                title_suffix= f"{number_of_available_members} Ensemble Mean {region_label} | " + _snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=f"_ensemble_mean_{args.enso_state}_{region_label.lower()}",
                dec_onset_month=args.composite_start,
                onset_season_ndjfm=args.onset_season,
                nino_threshold=float(args.nino_threshold),
            )
            
            if save_ensemble_mean_netcdf:
                # Output the ensemble mean composite data as netCDF for future analysis
                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"AAM_lat_lev_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
                )
                ens_mean.name = "AAMA"
                ens_mean.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean.to_netcdf(out_nc_path)
                print(f"Ensemble mean composite data saved to {out_nc_path}")

                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"U_lat_lev_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
                )
                ens_mean_u_latlev.name = "u"
                ens_mean_u_latlev.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean_u_latlev.to_netcdf(out_nc_path)
                print(f"Ensemble mean zonal wind composite data saved to {out_nc_path}")
                
        except Exception as e:
            print(f"Error plotting ensemble mean lat×level composite: {e}")
            import pdb; pdb.set_trace()
    
    if not ensemble_lat_lon_composites:
        print(f"WARNING: No ensemble lat×lon composites computed for region '{args.region}'. Skipping lat×lon plots.")
        
    
    if ensemble_lat_lon_composites:
        try:
            ens_stack = xr.concat(ensemble_lat_lon_composites, dim="ensemble")
            ens_mean = ens_stack.mean("ensemble", skipna=True)
            
            print(f"\n[ENSEMBLE MEAN DEBUG]")
            print(f"  ens_stack shape: {ens_stack.shape}")
            print(f"  ens_stack lon range: [{float(ens_stack['lon'].values.min()):.1f}, {float(ens_stack['lon'].values.max()):.1f}]")
            print(f"  ens_mean shape: {ens_mean.shape}")
            print(f"  ens_mean lon range BEFORE safety check: [{float(ens_mean['lon'].values.min()):.1f}, {float(ens_mean['lon'].values.max()):.1f}]")
            
            ens_stack = xr.concat(ensemble_u_latlon, dim="ensemble")
            ens_mean_u_latlon = ens_stack.mean("ensemble", skipna=True)
            # Rename month dimension to time for compatibility with plotting function
            if "month" in ens_mean_u_latlon.dims and "time" not in ens_mean_u_latlon.dims:
                ens_mean_u_latlon = ens_mean_u_latlon.rename({"month": "time"})
                
            ens_stack = xr.concat(ensemble_uv_vi_lat, dim="ensemble")
            ens_mean_uv_latlev = ens_stack.mean("ensemble", skipna=True)
            if "month" in ens_mean_uv_latlev.dims and "time" not in ens_mean_uv_latlev.dims:
                ens_mean_uv_latlev = ens_mean_uv_latlev.rename({"month": "time"})
            
            # Normalize dimension names (region selection was already applied in _process_single_ensemble_member)
            # Rename longitude → lon
            if 'longitude' in ens_mean.dims:
                ens_mean = ens_mean.rename({'longitude': 'lon'})
            if 'longitude' in ens_mean_u_latlon.dims:
                ens_mean_u_latlon = ens_mean_u_latlon.rename({'longitude': 'lon'})
            if 'longitude' in ens_mean_uv_latlev.dims:
                ens_mean_uv_latlev = ens_mean_uv_latlev.rename({'longitude': 'lon'})
            
            # Rename latitude → lat
            if 'latitude' in ens_mean.dims:
                ens_mean = ens_mean.rename({'latitude': 'lat'})
            if 'latitude' in ens_mean_u_latlon.dims:
                ens_mean_u_latlon = ens_mean_u_latlon.rename({'latitude': 'lat'})
            if 'latitude' in ens_mean_uv_latlev.dims:
                ens_mean_uv_latlev = ens_mean_uv_latlev.rename({'latitude': 'lat'})
            
            # Force region selection immediately before plotting to guarantee slice correctness.
            if args.region != 'all' and 'lon' in ens_mean.dims:
                ens_mean = _select_region(ens_mean, args.region)
                if 'lon' in ens_mean_u_latlon.dims:
                    ens_mean_u_latlon = _select_region(ens_mean_u_latlon, args.region)
                if 'lon' in ens_mean_uv_latlev.dims:
                    ens_mean_uv_latlev = _select_region(ens_mean_uv_latlev, args.region)

                lon_actual_min = float(ens_mean['lon'].values.min())
                lon_actual_max = float(ens_mean['lon'].values.max())
                n_lon_actual = int(ens_mean.sizes['lon'])
                print(f"\n[REGION DEBUG] Region requested: {args.region}")
                print(f"[REGION DEBUG] Final plot slice lon range [{lon_actual_min:.1f}, {lon_actual_max:.1f}], n_lon={n_lon_actual}")

            print(f"\n[PRE-PLOT CHECK] About to plot lat×lon composite:")
            print(f"  ens_mean shape: {ens_mean.shape}")
            print(f"  ens_mean lon range: [{float(ens_mean['lon'].values.min()):.1f}, {float(ens_mean['lon'].values.max()):.1f}]")
            print(f"  ens_mean lat range: [{float(ens_mean['lat'].values.min()):.1f}, {float(ens_mean['lat'].values.max()):.1f}]")
            print(f"  ens_mean_u_latlon shape: {ens_mean_u_latlon.shape}")
            print(f"  ens_mean_uv_latlev shape: {ens_mean_uv_latlev.shape}")
            print(f"  Passing region='{args.region}' to plot_lat_lon_snapshots()")

            print(f"Plotting ENSEMBLE MEAN lat×lon composite from {ens_stack.sizes['ensemble']} members...")
            
            plot_lat_lon_snapshots(
                ens_mean,
                zonal_wind_da=ens_mean_u_latlon,
                output_dir=output_dir,
                ensemble_member="ENSEMBLE_MEAN",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                title_suffix=f"{number_of_available_members} Ensemble Mean {region_label} | " + _snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=f"_ensemble_mean_{args.enso_state}_{region_label.lower()}",
                uv_latlev_profile=ens_mean_uv_latlev,
                pmin=float(args.p_min),
                pmax=float(args.p_max),
                nino_threshold=float(args.nino_threshold),
                region=args.region,
            )
            
            if save_ensemble_mean_netcdf:
                out_nc_path = os.path.join(
                ensemble_mean_output_path,
                f"AAM_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
                )
                ens_mean.name = "AAMA"
                ens_mean.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean.to_netcdf(out_nc_path)
                print(f"Ensemble mean composite data saved to {out_nc_path}")

                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"U_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
                )
                ens_mean_u_latlon.name = "u"
                ens_mean_u_latlon.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean_u_latlon.to_netcdf(out_nc_path)
                print(f"Ensemble mean zonal wind composite data saved to {out_nc_path}")
                
                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"UV_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_{region_label.lower()}_ninothres{float(args.nino_threshold)}.nc",
                )
                ens_mean_uv_latlev.name = "uv"
                ens_mean_uv_latlev.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean_uv_latlev.to_netcdf(out_nc_path)
                print(f"Ensemble mean UV composite data saved to {out_nc_path}")
                
        except Exception as e:
            print(f"Error plotting ensemble mean lat×lon composite: {e}")
            import pdb; pdb.set_trace()
    
    # =============================================================================
    # Plot histogram of ENSO onset months across ensemble members and years
    if 'onset_dates' in locals() and onset_dates:
        try:
            os.makedirs(output_dir, exist_ok=True)
            onset_pd = pd.to_datetime(onset_dates)
            months = onset_pd.month

            # Counts per calendar month (1=Jan .. 12=Dec)
            counts = months.value_counts().sort_index()
            month_idx = list(range(1, 13))
            counts_list = [int(counts.get(m, 0)) for m in month_idx]
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(month_names))
            ax.bar(x, counts_list, width=1.0, align='center', color='C1', edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(month_names)
            ax.set_xlim(-0.5, len(month_names) - 0.5)
            ax.set_xlabel('Onset month')
            ax.set_ylabel(f'Event count (across ensemble members and years)')
            ax.set_title(f'ENSO onset month histogram ({args.enso_state}) {number_of_available_members} members {args.start_year}-{args.end_year}')

            out_hist_path = os.path.join(
                output_dir,
                f"ENSO_onset_months_histogram_{args.enso_state}_{args.start_year}-{args.end_year}_ninothres{float(args.nino_threshold)}.png",
            )
            fig.savefig(out_hist_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved onset months histogram to {out_hist_path}")
        except Exception as e:
            print(f"Error generating onset month histogram: {e}")
            import traceback; traceback.print_exc()
        # Plot histogram of per-event peak Nino3.4 amplitudes (0.5 bins starting at 0.5)
        if event_peak_amplitudes:
            try:
                peaks = np.asarray(event_peak_amplitudes, dtype=float)
                if peaks.size > 0:
                    bin_width = 0.5
                    max_peak = float(np.nanmax(peaks))
                    # ensure at least one bin centered at 0.5
                    max_center = max(bin_width, np.ceil(max_peak / bin_width) * bin_width)
                    bin_centers = np.arange(bin_width, max_center + bin_width / 2, bin_width)
                    bin_edges = np.concatenate(([bin_centers[0] - bin_width / 2], bin_centers + bin_width / 2))
                    counts, _ = np.histogram(peaks, bins=bin_edges)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(bin_centers, counts, width=bin_width, align='center', color='C2', edgecolor='black', linewidth=0.5)
                    ax.set_xticks(bin_centers)
                    ax.set_xlabel('Event peak Nino3.4 (|°C|)')
                    ax.set_ylabel('Event count (across ensemble members and years)')
                    ax.set_title(f'ENSO event peak Nino3.4 histogram ({args.enso_state}) {number_of_available_members} members {args.start_year}-{args.end_year}')

                    out_hist_peaks = os.path.join(
                        output_dir,
                        f"ENSO_event_peak_nino34_histogram_{args.enso_state}_{args.start_year}-{args.end_year}_bin{bin_width}_ninothres{float(args.nino_threshold)}.png",
                    )
                    fig.savefig(out_hist_peaks, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved event peak Nino3.4 histogram to {out_hist_peaks}")
            except Exception as e:
                print(f"Error generating event-peak Nino3.4 histogram: {e}")
                import traceback; traceback.print_exc()
        