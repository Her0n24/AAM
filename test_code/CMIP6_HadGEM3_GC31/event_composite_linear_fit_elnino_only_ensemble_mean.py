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
import json
# Allow importing shared utilities from AAM/test_code
import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utilities import _to_per_latitude_band, _reindex_to_climatology_dims, vertical_sum_over_pressure_range, get_ENSO_index
from plotting_utils import plot_latitude_level_snapshots_HadGEN3, plot_lat_lon_snapshots
import tqdm
from scipy import stats as _stats
from matplotlib import pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=150.0, help='Minimum pressure level (hPa) to include (default: 0 hPa)')
parser.add_argument('--p-max', type=float, default=700, help='Maximum pressure level (hPa) to include (default: 1020 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year to plot (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year to plot (default: 2000)')
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
args = parser.parse_args()

base_dir = os.getcwd()
# AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"

#climatology_path_base = f"{base_dir}/climatology/"

# CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul" 
# nino34_directory = f"{CMIP6_path_base}/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/"
# output_dir = f"{base_dir}/figures/composites/non_tracking_algorithm/"

# Use scratch space and new directory structure due to workspace migration
CMIP6_path_base = "/work/scratch-nopw2/hhhn2/"
nino34_directory = f"{CMIP6_path_base}/HadGEM3-GC31-LL/ProcessedFlds/Omon/sst_indices/nino34/historical/"
output_dir = f"{base_dir}/figures/composites/non_tracking_algorithm/"
climatology_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/climatology/"
AAM_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/full/"

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
    output_dir: str = "figures/",
    enso_state: str = "el_nino",
    rolling_period: int = 1,
    composite_months: int = 24,
    composite_start: str = "onset",
    nlevels: int = 13,
    onset_season: str = "all",
) -> xr.DataArray:
    """Composite AAM anomalies for El Nino onset windows.

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

    # --- Unwrap Datasets ---
    AAM_field = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
    if isinstance(AAM_field, xr.Dataset):
        AAM_field = next(iter(AAM_field.data_vars.values()))

    wind_field = None
    if wind_da is not None:
        wind_field = wind_da["ua"] if isinstance(wind_da, xr.Dataset) and "ua" in wind_da else wind_da
        if isinstance(wind_field, xr.Dataset):
            wind_field = next(iter(wind_field.data_vars.values()))

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

    import os

    lat_dim = "latitude" if "latitude" in composite_AAM.dims else "lat"

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
    available_members = []
    
    for ensemble_member in [f"r{i}i1p1f3" for i in range(1, 61)]:
    # Use OS to see whether the nc file exists before trying to open with xarray, to avoid long error messages from xarray when files are missing.
        if not os.path.exists(os.path.join(AAM_data_path_base, f"AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")):
            print(f"Skipping member {ensemble_member} because AAM file not found.")
            continue
        else:
            available_members.append(ensemble_member)
            print(f"Processing ensemble member: {ensemble_member}")
    
    for ensemble_member in tqdm.tqdm(available_members):
        AAM_da = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")['AAM']
        # Ensure zonal integration (remove longitude) if present
        if "longitude" in AAM_da.dims or "lon" in AAM_da.dims:
            AAM_da = _to_per_latitude_band(AAM_da)  # (time, level, latitude)
            
        clim_da = xr.open_dataset(
            f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc"
            )  #dims: month, level, latitude, longitude 

        # IMPORTANT: use the per-latitude-band + zonal-integral climatology (matches _to_per_latitude_band convention)
        if 'longitude' in clim_da['AAM'].dims or 'lon' in clim_da['AAM'].dims:
            clim_da = _to_per_latitude_band(clim_da['AAM'])
        else:
            clim_da = clim_da['AAM']
        
        # --- Step 1: Detect ENSO state windows from Nino3.4 ---
        _onset_months_map: dict = {"all": None, "ndjfm": {11, 12, 1, 2, 3}}
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

        state_pretty = "El Nino" if args.enso_state == "el_nino" else "La Nina"

        # --- Step 2: Write JSON ---
        json_out = f"AAM_event_metrics_{args.start_year}_{args.end_year}.json"
        results_json = {
            "config": {
                "ensemble_member": ensemble_member,
                "start_year": int(args.start_year),
                "end_year": int(args.end_year),
                "p_min_hpa": float(args.p_min),
                "p_max_hpa": float(args.p_max),
                "enso_state": str(args.enso_state),
                "nino_threshold": float(args.nino_threshold),
                "min_elnino_months": int(args.min_elnino_months),
                "rolling_period": int(args.rolling_period),
                "onset_season": str(args.onset_season),
                "composite_months": int(args.composite_months),
                "composite_start": str(args.composite_start),
            },
            "events": [
                {
                    "event_id": int(i + 1),
                    "onset": onset,
                    "end": end,
                    "composite_window_start": _compute_composite_window_from_onset(
                        onset,
                        composite_months=int(args.composite_months),
                        composite_start=str(args.composite_start),
                    )[0],
                    "composite_window_end": _compute_composite_window_from_onset(
                        onset,
                        composite_months=int(args.composite_months),
                        composite_start=str(args.composite_start),
                    )[1],
                }
                for i, (onset, end) in enumerate(date_list)
            ],
        }
        with open(json_out, "w", encoding="utf-8") as _f:
            json.dump(results_json, _f, indent=2, sort_keys=False)
        print(f"Detected {len(date_list)} {state_pretty}-state onset event(s)")
        print(f"Wrote event JSON: {json_out}")

        # --- Step 4: Composite AAM anomaly from ENSO onset windows ---
        if date_list:
            comp = composite_propagating_years_no_plot(
                AAM_da,
                wind_da=None,
                date_list=date_list,
                clim_da=clim_da,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                p_min_hpa=float(args.p_min),
                p_max_hpa=float(args.p_max),
                output_dir=output_dir,
                enso_state=str(args.enso_state),
                rolling_period=int(args.rolling_period),
                composite_months=int(args.composite_months),
                composite_start=str(args.composite_start),
                onset_season=str(args.onset_season),
            )
            if comp is not None:
                comp = comp.expand_dims({"ensemble": [ensemble_member]})
                ensemble_composites.append(comp)
                
        #--- Step 5: Latitude×level composite ---
        #Re-uses the full multi-level AAM_da (all levels, all lats) but zonally integrates
        #(removes longitude) before compositing. No vertical integration here — the snapshot
        #function shows the full lat×level cross-section.
        if date_list:
            # Zonal integral only (keeps all pressure levels) → (time, level, latitude)
            aam_full = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
            if "longitude" in aam_full.dims or "lon" in aam_full.dims:
                aam_full = _to_per_latitude_band(aam_full)

            # Anomaly vs climatology (same pipeline as the rest of the script)
            clim_full = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
            aam_full, clim_on_time_full = _reindex_to_climatology_dims(aam_full, clim_full)
            anom_full = aam_full - clim_on_time_full  # (time, level, latitude)

            # Composite over events aligned to relative month 1..N
            stacked_full = []
            seen_ev: set = set()
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
                    print(f"  snapshot composite: onset {ym} window incomplete, skipping.")
                    continue
                evt = evt.isel(time=slice(0, int(args.composite_months)))
                evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1, dtype=int))
                if "month" in evt.coords:
                    evt = evt.drop_vars("month")
                evt = evt.rename({"time": "month"})
                stacked_full.append(evt)

            if stacked_full:
                n_ev = len(stacked_full)
                full_stack = xr.concat(stacked_full, dim="event")

                # Apply circular rolling over relative-month bins before averaging events.
                # This wraps across year-end, so NDJ uses Nov-Dec-Jan and DJF uses Dec-Jan-Feb.
                rp = int(args.rolling_period)
                if rp > 1:
                    n_month = int(full_stack.sizes["month"])
                    if rp > n_month:
                        raise ValueError(
                            f"rolling_period ({rp}) cannot exceed number of month bins ({n_month})"
                        )
                    left = rp // 2
                    right = rp - left - 1
                    _rolled = [
                        full_stack.roll(month=-offset, roll_coords=False)
                        for offset in range(-left, right + 1)
                    ]
                    full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)

                composite_full = full_stack.mean("event", skipna=True)
                # Rename month → time so plot_latitude_level_snapshots_HadGEN3 sees a 'time' dim
                composite_full = composite_full.rename({"month": "time"})
                composite_full.attrs["long_name"] = "AAM anomaly"
                _cmp = ">" if args.enso_state == "el_nino" else "<"
                _snap_season_label = "  |  NDJFM onsets only" if args.onset_season == "ndjfm" else ""
                _snap_suffix = (
                    f"{state_pretty} state: Nino3.4{_cmp}{float(args.nino_threshold):.2f} "
                    f"for >= {int(args.min_elnino_months)} months"
                    f" | {int(args.composite_months)}-month composite from "
                    f"{'Dec of onset year' if args.composite_start == 'december_onset_year' else 'onset month'}"
                    f"{_snap_season_label}"
                )
                _snap_file_season = f"_{args.onset_season}" if args.onset_season != "all" else ""
                composite_full = composite_full.expand_dims({"ensemble": [ensemble_member]})
                ensemble_lat_lev_composites.append(composite_full)
    
        # Step 6: LAT*Lon composite snapshots plots
        # IMPORTANT: Here we require different kind of Climatology not the latband version
        aam_full = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")['AAM']
        clim_full = clim_da = xr.open_dataset(
            f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc"
        )  # dims: month, level, latitude
        
        # Similar to Step 5 but without zonal integration, so the full lat×lon×level structure is retained.
        if date_list:
            # Vertical integral only (keeps all longitudes) → (time, latitude, longitude)
            aam_vs = vertical_sum_over_pressure_range(aam_full, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
            #import pdb; pdb.set_trace()
            clim_full = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
            clim_vs = vertical_sum_over_pressure_range(clim_full, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
            # Anomaly vs climatology (same pipeline as the rest of the script)
            clim_vs = clim_vs["AAM"] if isinstance(clim_vs, xr.Dataset) and "AAM" in clim_vs else clim_vs
            aam_full, clim_on_time = _reindex_to_climatology_dims(aam_vs, clim_vs)
            anom_full = aam_full - clim_on_time  # (time, latitude, longitude)
            
            #import pdb; pdb.set_trace()
            
            # Composite over events aligned to relative month 1..N
            stacked_full = []
            seen_ev: set = set()
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
                n_ev = len(stacked_full)
                full_stack = xr.concat(stacked_full, dim="event")

                # Apply circular rolling over relative-month bins before averaging events.
                # This wraps across year-end, so NDJ uses Nov-Dec-Jan and DJF uses Dec-Jan-Feb.
                rp = int(args.rolling_period)
                if rp > 1:
                    n_month = int(full_stack.sizes["month"])
                    if rp > n_month:
                        raise ValueError(
                            f"rolling_period ({rp}) cannot exceed number of month bins ({n_month})"
                        )
                    left = rp // 2
                    right = rp - left - 1
                    _rolled = [
                        full_stack.roll(month=-offset, roll_coords=False)
                        for offset in range(-left, right + 1)
                    ]
                    full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)

                composite_full = full_stack.mean("event", skipna=True)
                # Rename month → time so plot_latitude_level_snapshots_HadGEN3 sees a 'time' dim
                composite_full = composite_full.rename({"month": "time"})
                composite_full.attrs["long_name"] = "AAM anomaly"
                _cmp = ">" if args.enso_state == "el_nino" else "<"
                _snap_season_label = "  |  NDJFM onsets only" if args.onset_season == "ndjfm" else ""
                _snap_suffix = (
                    f"{state_pretty} state: Nino3.4{_cmp}{float(args.nino_threshold):.2f} "
                    f"for >= {int(args.min_elnino_months)} months"
                    f" | {int(args.composite_months)}-month composite from "
                    f"{'Dec of onset year' if args.composite_start == 'december_onset_year' else 'onset month'}"
                    f"{_snap_season_label}"
                )
                _snap_file_season = f"_{args.onset_season}" if args.onset_season != "all" else ""

                composite_full = composite_full.expand_dims({"ensemble": [ensemble_member]})
                ensemble_lat_lon_composites.append(composite_full)
        
    number_of_available_members = len(available_members)
    
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

            ax.set_title(
                f"HadGEM3_GC31 {number_of_available_members} members ENSEMBLE MEAN Composite AAM anomaly\n"
                f"({args.p_min}–{args.p_max} hPa) {state_pretty} events  {args.start_year}–{args.end_year}"
                f"clim {clim_start_yr}–{clim_end_yr}"
            )

            # -------------------------------
            # Save
            # -------------------------------
            os.makedirs(output_dir, exist_ok=True)

            out_path = os.path.join(
                output_dir,
                f"AAM_composite_ENSEMBLE_MEAN_{args.start_year}-{args.end_year}"
                f"_{args.p_min}-{args.p_max}hPa_{args.enso_state}_state.png",
            )

            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"Ensemble mean composite plot saved to {out_path}")

        except Exception as e:
            print(f"Error plotting ensemble mean composite: {e}")
            import pdb; pdb.set_trace()
    
    if ensemble_lat_lev_composites:
        try:
            ens_stack = xr.concat(ensemble_lat_lev_composites, dim="ensemble")
            ens_mean = ens_stack.mean("ensemble", skipna=True)

            print(f"Plotting ENSEMBLE MEAN lat×level composite from {ens_stack.sizes['ensemble']} members...")

            plot_latitude_level_snapshots_HadGEN3(
                ens_mean,
                ensemble_member="ENSEMBLE_MEAN",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                output_dir=output_dir,
                title_suffix= f"{number_of_available_members} Ensemble Mean " + _snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=f"_ensemble_mean_{args.enso_state}",
            )
        except Exception as e:
            print(f"Error plotting ensemble mean lat×level composite: {e}")
            import pdb; pdb.set_trace()
        
    if ensemble_lat_lon_composites:
        try:
            ens_stack = xr.concat(ensemble_lat_lon_composites, dim="ensemble")
            ens_mean = ens_stack.mean("ensemble", skipna=True)

            print(f"Plotting ENSEMBLE MEAN lat×lon composite from {ens_stack.sizes['ensemble']} members...")

            plot_lat_lon_snapshots(
                ens_mean,
                output_dir=output_dir,
                ensemble_member="ENSEMBLE_MEAN",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                title_suffix=f"{number_of_available_members} Ensemble Mean " + _snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=f"_ensemble_mean_{args.enso_state}",
            )
        except Exception as e:
            print(f"Error plotting ensemble mean lat×lon composite: {e}")
            import pdb; pdb.set_trace()
        