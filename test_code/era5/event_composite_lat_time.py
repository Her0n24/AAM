"""
Event composite of deterministic ERA5 AAM anomalies from ENSO event onset.

Example
-------
Run from AAM/test_code/era5 or AAM/test_code:

python era5/event_composite_lat_time.py \
  --start-year 1979 --end-year 2023 \
  --composite-months 24 --composite-start onset \
  --onset-season ndjfm --p-min 150 --p-max 700

The event setup follows the HadGEM3 event selection script:
- sustained ENSO state begins after at least N consecutive threshold months
- onset can be restricted to NDJFM
- composite windows start at onset month or December of onset year
- duplicate onset months are ignored
- the final composite is the simple mean across all complete events
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Optional

from scipy import stats as _stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import event_composite_all as composite_core
from plotting_utils import add_active_month_percent_labels, compute_active_month_percent

sys.path.append(str(Path(__file__).resolve().parents[1] / "CMIP6_HadGEM3_GC31"))
from CMIP6_HadGEM3_GC31.utilities import (
    _to_per_latitude_band,
    vertical_sum_over_pressure_range,
    REGION_BOUNDS
)

ACTIVE_MONTH_EL_NINO_THRESHOLD = 0.5
ACTIVE_MONTH_LA_NINA_THRESHOLD = -0.5

ERA5_DIR = Path(__file__).resolve().parent
NINO34_CSV = ERA5_DIR / "nino34" / "nino34_HadlSST.csv"
L137_CSV = ERA5_DIR / "l137_a_b.csv"
AAM_FIG_DIR = ERA5_DIR / "AAMA_fig"
AAM_DATA_DIR = Path("/work/scratch-nopw2/hhhn2/ERA5/monthly_mean/AAM/full")
CLIMATOLOGY_DIR = Path("/work/scratch-nopw2/hhhn2/ERA5/climatology")
OUTPUT_DIR = ERA5_DIR / "composite_non_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic ERA5 AAM event mean composite over ENSO events."
    )
    parser.add_argument("--p-min", type=float, default=150.0, help="Minimum pressure level in hPa.")
    parser.add_argument("--p-max", type=float, default=1000.0, help="Maximum pressure level in hPa.")
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
        help="Also save the full event stack used in the mean composite.",
    )
    return parser.parse_args()


def _resolve_climatology_file(args: argparse.Namespace) -> str:
    candidates = [
        CLIMATOLOGY_DIR / f"ERA5_AAM_climatology_{args.clim_start_year}-{args.clim_end_year}_full_level.nc",
        CLIMATOLOGY_DIR / f"AAM_ERA5_climatology_{args.clim_start_year}-{args.clim_end_year}_full_level.nc",
        CLIMATOLOGY_DIR / f"ERA5_AAM_full_climatology_{args.clim_start_year}-{args.clim_end_year}.nc",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"No ERA5 AAM climatology file found in fixed directory: {CLIMATOLOGY_DIR}"
    )

def _resolve_precomputed_anomaly_file(args: argparse.Namespace) -> Optional[str]:
    # Region-specific composites must start from the full field so longitude can be
    # trimmed before any zonal integration or latitude-band weighting.
    if args.region != "all":
        return None

    pattern = str(AAM_FIG_DIR / f"AAM_anomalies_*_p{args.p_min:g}-{args.p_max:g}hPa.nc")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None

    matching = []
    for path in candidates:
        try:
            ds = xr.open_dataset(path)
            try:
                if "time" not in ds.coords and "time" not in ds.dims:
                    continue
                time_index = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
                if time_index.empty:
                    continue
                file_start = int(time_index.min().year)
                file_end = int(time_index.max().year)
            finally:
                ds.close()
        except Exception:
            continue

        if file_start <= args.start_year and file_end >= args.end_year:
            try:
                with xr.open_dataset(path) as ds:
                    file_region = ds.attrs.get("region")
                    has_lon = "longitude" in ds.dims or "lon" in ds.dims
            except Exception:
                file_region = None
                has_lon = False

            matching.append((file_start, file_end, path))

    if not matching:
        return candidates[0]

    matching.sort(key=lambda item: (item[1] - item[0], item[0]))
    return matching[0][2]


def _first_data_var(ds: xr.Dataset, preferred: tuple[str, ...]) -> xr.DataArray:
    for name in preferred:
        if name in ds.data_vars:
            return ds[name]
    if len(ds.data_vars) == 1:
        return ds[next(iter(ds.data_vars))]
    raise KeyError(f"Could not resolve variable. Available variables: {list(ds.data_vars)}")


def _standardize_dims(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    aliases = {
        "latitude": ("lat", "y"),
        "longitude": ("lon", "x"),
        "level": ("lev", "plev", "pressure", "mid_level"),
    }
    for canonical, names in aliases.items():
        if canonical in da.dims:
            continue
        for name in names:
            if name in da.dims:
                rename[name] = canonical
                break
    if rename:
        da = da.rename(rename)
    return da

def _infer_latitude_band_width_deg(da: xr.DataArray) -> float:
    """Infer the latitude band width in degrees from the latitude coordinate."""
    lat_dim = "latitude" if "latitude" in da.dims else ("lat" if "lat" in da.dims else None)
    if lat_dim is None or da.sizes.get(lat_dim, 0) < 2:
        return float("nan")

    lat_vals = np.asarray(da[lat_dim].values, dtype=float)
    diffs = np.diff(np.sort(lat_vals))
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if diffs.size == 0:
        return float("nan")
    return float(np.nanmedian(np.abs(diffs)))


def _apply_latitude_band_weighting(da: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Apply latitude-band weighting when longitude exists; otherwise infer band width."""
    if "longitude" in da.dims or "lon" in da.dims:
        return _to_per_latitude_band(da)
    return da, _infer_latitude_band_width_deg(da)


def _select_region(da: xr.DataArray, region: str) -> xr.DataArray:
    if region == "all" or "longitude" not in da.dims:
        print("Region is set to all or Expected full AAM field not present for longitude trimming")
        return da
    lon_min, lon_max = REGION_BOUNDS[region]
    lon = da["longitude"]
    lon_360 = lon % 360.0
    lon_min_360 = lon_min % 360.0
    lon_max_360 = lon_max % 360.0
    
    if lon_min_360 > lon_max_360:
        mask = (lon_360 >= lon_min_360) | (lon_360 <= lon_max_360)
    else:
        mask = (lon_360 >= lon_min_360) & (lon_360 <= lon_max_360)
        
    # FIX 1: Extract pure boolean numpy array to prevent xarray alignment failures
    mask_vals = mask.values 
    
    selected_lon = np.asarray(lon_360.values[mask_vals], dtype=float)
    if lon_min_360 > lon_max_360:
        # Unwrap Greenwich-crossing regions so longitude remains monotonic
        selected_lon = np.where(selected_lon <= lon_max_360, selected_lon + 360.0, selected_lon)
        
    selected = da.isel(longitude=mask_vals).assign_coords(longitude=selected_lon)
    selected = selected.sortby("longitude")
    if selected.sizes.get("longitude", 0) == 0:
        raise ValueError(f"Region {region!r} selected zero longitudes.")
    return selected

def _infer_time_from_filenames(files: list[str]) -> list[pd.Timestamp]:
    times = []
    for fname in files:
        match = re.search(r"ERA5_\d{4}|AAM_ERA5_(\d{4})-(\d{2})", os.path.basename(fname))
        if match and match.lastindex and match.lastindex >= 2:
            times.append(pd.Timestamp(f"{match.group(1)}-{match.group(2)}-01"))
            continue
        match = re.search(r"(\d{4})-(\d{2})", os.path.basename(fname))
        if not match:
            raise ValueError(f"Could not infer YYYY-MM from filename: {fname}")
        times.append(pd.Timestamp(f"{match.group(1)}-{match.group(2)}-01"))
    return times


def load_era5_aam(args: argparse.Namespace) -> xr.DataArray:
    max_needed_year = args.end_year + int(np.ceil(args.composite_months / 12.0)) + 1
    files = []
    for year in range(args.start_year, max_needed_year + 1):
        files.extend(sorted(glob.glob(str(AAM_DATA_DIR / f"AAM_ERA5_{year}-*_full.nc"))))
    if not files:
        raise FileNotFoundError(f"No ERA5 AAM files found in fixed directory: {AAM_DATA_DIR}")

    print(f"Opening {len(files)} ERA5 AAM monthly files.")
    try:
        # Keep this lazy/chunked to avoid OOM on large regional composites.
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks={"time": 12},
            parallel=False,
        )
    except ValueError:
        datasets = [xr.open_dataset(path, decode_times=False) for path in files]
        ds = xr.concat(datasets, dim="time", coords="minimal", compat="override")

    da = _first_data_var(ds, ("AAM", "aam", "angular_momentum", "momentum"))
    da = _standardize_dims(da)
    if "time" in da.dims:
        times = _infer_time_from_filenames(files)
        if len(times) == da.sizes["time"]:
            da = da.assign_coords(time=pd.DatetimeIndex(times))
    return da


def load_era5_climatology(args: argparse.Namespace) -> xr.DataArray:
    path = _resolve_climatology_file(args)
    print(f"Opening ERA5 AAM climatology: {path}")
    ds = xr.open_dataset(path)
    da = _first_data_var(ds, ("AAM", "aam", "angular_momentum", "momentum", "AAMA"))
    da = _standardize_dims(da)
    if "month" not in da.dims and "month" not in da.coords:
        raise ValueError(f"Climatology must have a monthly 'month' dimension or coord: {path}")
    return da


def _prepare_monthly_anomalies(
    da: xr.DataArray,
    clim_da: xr.DataArray,
    args: argparse.Namespace,
    variable_name: str,
) -> xr.DataArray:
    data = _standardize_dims(da)
    clim = _standardize_dims(clim_da)

    if "month" not in clim.dims and "month" not in clim.coords:
        raise ValueError(f"{variable_name} climatology must have a monthly 'month' dimension or coord.")

    clim_months = clim["month"].values
    if np.size(clim_months) and np.nanmin(clim_months) == 0 and np.nanmax(clim_months) == 11:
        clim = clim.assign_coords(month=clim["month"] + 1)

    if args.region != "all" and "longitude" in data.dims:
        orig_lon_count = int(data.sizes.get("longitude", 0))
        data = _select_region(data, args.region)
        new_lon_count = int(data.sizes.get("longitude", 0))
        lon_vals = np.asarray(data["longitude"].values, dtype=float)
        lon_min = float(np.nanmin(lon_vals)) if lon_vals.size else float("nan")
        lon_max = float(np.nanmax(lon_vals)) if lon_vals.size else float("nan")
        print(
            f"Applied region slice '{args.region}' for {variable_name}: "
            f"longitude count {orig_lon_count} -> {new_lon_count}, range [{lon_min:.1f}, {lon_max:.1f}]"
        )
        clim = _select_region(clim, args.region)

        # Force climatology onto the same region grid as the data so the monthly
        # subtraction cannot silently align to an empty or mismatched longitude index.
        # Force climatology onto the same region grid as the data
        clim = clim.sortby("latitude") if "latitude" in clim.dims else clim
        data = data.sortby("latitude") if "latitude" in data.dims else data
        
        # FIX 2: Use .values and method="nearest" to avoid NaN-filled reindexing 
        # from floating-point misalignment
        if "longitude" in clim.dims and "longitude" in data.dims:
            clim = clim.sel(
                latitude=data["latitude"].values, 
                longitude=data["longitude"].values, 
                method="nearest"
            )
        elif "latitude" in clim.dims and "latitude" in data.dims:
            clim = clim.sel(
                latitude=data["latitude"].values, 
                method="nearest"
            )

        data, clim = xr.align(data, clim, join="override")
        print(
            f"Aligned {variable_name} region grid: data lon [{float(np.nanmin(data.longitude.values)):.1f}, "
            f"{float(np.nanmax(data.longitude.values)):.1f}], climatology lon [{float(np.nanmin(clim.longitude.values)):.1f}, "
            f"{float(np.nanmax(clim.longitude.values)):.1f}]"
        )

    anomalies = data.groupby("time.month") - clim
    if "time" not in anomalies.dims:
        raise ValueError(f"{variable_name} anomalies must have a time dimension.")
    return anomalies


def _debug_report(stage: str, da: xr.DataArray) -> None:
    """Always print detailed stage-by-stage diagnostics."""
    total_count = int(np.prod([int(da.sizes[d]) for d in da.dims], dtype=np.int64))

    finite_count = int(da.count().compute().item())
    zero_count = int((da == 0).sum(skipna=True).compute().item())
    nonzero_count = max(finite_count - zero_count, 0)

    if finite_count:
        vmin = float(da.min(skipna=True).compute().item())
        vmax = float(da.max(skipna=True).compute().item())
        vmean = float(da.mean(skipna=True).compute().item())
        vstd = float(da.std(skipna=True).compute().item())
        absmax = float(np.abs(da).max(skipna=True).compute().item())
    else:
        vmin = vmax = vmean = vstd = absmax = float("nan")

    print(
        f"[DEBUG][{stage}] dims={da.dims}, shape={da.shape}, "
        f"finite={finite_count}/{total_count}, zero={zero_count}, nonzero={nonzero_count}, "
        f"min={vmin:.6g}, max={vmax:.6g}, mean={vmean:.6g}, std={vstd:.6g}, absmax={absmax:.6g}"
    )

    if "time" in da.dims:
        reduce_dims = tuple(dim for dim in da.dims if dim != "time")
        month_absmax = np.abs(da).groupby("time.month").max(dim=reduce_dims, skipna=True).compute()
        try:
            sample_months = [1, 2, 3, 6, 9, 12]
            items = []
            for m in sample_months:
                if m in month_absmax["month"].values:
                    items.append(f"m{m}={float(month_absmax.sel(month=m).values):.6g}")
            if items:
                print(f"[DEBUG][{stage}] monthly absmax: " + ", ".join(items))
        except Exception:
            pass


def load_nino34(args: argparse.Namespace, end_year_buffer: int = 3) -> pd.Series:
    if not NINO34_CSV.exists():
        raise FileNotFoundError(f"Fixed Nino3.4 CSV not found: {NINO34_CSV}")

    df = pd.read_csv(NINO34_CSV)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {NINO34_CSV}")

    time_raw = df.iloc[:, 0]
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce").replace(-99.99, np.nan)

    time_vals = pd.to_datetime(time_raw, errors="coerce")
    if time_vals.isna().all():
        years = pd.to_numeric(time_raw, errors="coerce")
        if years.isna().any():
            raise ValueError(f"Could not parse first column as dates or years in {NINO34_CSV}")
        time_vals = pd.to_datetime(years.astype(int).astype(str) + "-01-01")

    valid = time_vals.notna() & values.notna()
    series = pd.Series(
        values.loc[valid].to_numpy(dtype=float),
        index=pd.DatetimeIndex(time_vals.loc[valid]).to_period("M"),
    ).sort_index()
    return series.loc[pd.Period(f"{args.start_year}-01", "M") : pd.Period(f"{args.end_year + end_year_buffer}-12", "M")]


def detect_enso_state_windows(args: argparse.Namespace, nino34: pd.Series) -> list[tuple[str, str]]:
    threshold = args.nino_threshold
    if threshold is None:
        threshold = ACTIVE_MONTH_EL_NINO_THRESHOLD if args.enso_state == "el_nino" else ACTIVE_MONTH_LA_NINA_THRESHOLD
    if args.enso_state == "la_nina" and threshold > -0.5:
        raise ValueError("For la_nina, --nino-threshold should be <= -0.5.")

    vals = nino34.values.astype(float)
    in_state = vals >= threshold if args.enso_state == "el_nino" else vals <= threshold
    periods = nino34.index
    onset_months = None if args.onset_season == "all" else {11, 12, 1, 2, 3}

    windows = []
    i = 0
    while i < in_state.size:
        if not in_state[i]:
            i += 1
            continue
        j = i
        while j < in_state.size and in_state[j]:
            j += 1
        run_len = j - i
        if run_len >= args.min_enso_months:
            onset_period = periods[i]
            if args.start_year <= onset_period.year <= args.end_year:
                if onset_months is None or onset_period.month in onset_months:
                    keep = True
                    if (
                        args.enso_state == "el_nino"
                        and not args.allow_reinitiation
                        and args.reinitiation_check_months > 0
                    ):
                        post_end = j + args.reinitiation_check_months
                        keep = post_end <= in_state.size and not bool(np.any(in_state[j:post_end]))
                    if keep:
                        onset_ts = onset_period.to_timestamp()
                        end_ts = onset_ts + pd.DateOffset(months=11)
                        windows.append((onset_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))
        i = j
    return windows


def _compute_composite_window_from_onset(
    onset_str: str,
    *,
    composite_months: int,
    composite_start: str,
) -> tuple[str, str]:
    onset = pd.Timestamp(onset_str)
    if composite_start == "december_onset_year":
        start = pd.Timestamp(f"{onset.year}-12-01")
    else:
        start = pd.Timestamp(f"{onset.year}-{onset.month:02d}-01")
    end = start + pd.DateOffset(months=composite_months - 1)
    return start.strftime("%Y-%m"), end.strftime("%Y-%m")


def _circular_rolling_mean(da: xr.DataArray, *, dim: str, window: int) -> xr.DataArray:
    if window <= 1:
        return da
    n = int(da.sizes[dim])
    if window > n:
        raise ValueError(f"rolling_period ({window}) cannot exceed number of {dim} bins ({n})")
    left = window // 2
    right = window - left - 1
    rolled = [da.roll({dim: -offset}, roll_coords=False) for offset in range(-left, right + 1)]
    return xr.concat(rolled, dim="_roll").mean("_roll", skipna=True)


def build_event_stack(
    aam_da: xr.DataArray,
    clim_da: xr.DataArray,
    date_list: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[xr.DataArray, list[str], float]:
    anomalies = _prepare_monthly_anomalies(aam_da, clim_da, args, "AAM")
    _debug_report("anomalies_after_subtraction", anomalies)

    anomalies, dphi_deg = _apply_latitude_band_weighting(anomalies)
    _debug_report("after_latitude_band_weighting", anomalies)

    if "level" in anomalies.dims:
        level_vals = np.asarray(anomalies["level"].values, dtype=float)
        if level_vals.size:
            # 1. Map requested pressure to model levels using the CSV table
            try:
                level_df = pd.read_csv(L137_CSV)
                
                # Clean up the table: remove non-numeric rows (like n=0 which has '-')
                level_df = level_df[level_df['ph [hPa]'] != '-'].copy()
                level_df['ph [hPa]'] = pd.to_numeric(level_df['ph [hPa]'])
                level_df['n'] = pd.to_numeric(level_df['n'])

                # Find the closest model levels using half-level pressures (ph)
                n_min_idx = (level_df['ph [hPa]'] - args.p_min).abs().idxmin()
                n_max_idx = (level_df['ph [hPa]'] - args.p_max).abs().idxmin()

                n_start = level_df.loc[n_min_idx, 'n']
                n_end = level_df.loc[n_max_idx, 'n']

                # Ensure lower 'n' is first for the slice (n=1 is the top of the atmosphere)
                target_n_min = min(n_start, n_end)
                target_n_max = max(n_start, n_end)

                print(
                    f"[DEBUG][level_coord] size={level_vals.size}, min={np.nanmin(level_vals):.6g}, "
                    f"max={np.nanmax(level_vals):.6g}, unit is model level (n).\n"
                    f"Mapped requested {args.p_min}-{args.p_max} hPa to model levels n={target_n_min} to n={target_n_max}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to map pressure to model levels using CSV: {e}")
            
            # 2. Prevent empty slices from reversed coordinates
            anomalies = anomalies.sortby("level")
            
            # 3. Subset to the mapped model level range
            # Xarray's slice is inclusive and will capture all points (like 1.5, 2.5) between the integer boundaries
            anomalies = anomalies.sel(level=slice(target_n_min, target_n_max))
            
            # 4. Perform the vertical sum
            anomalies = anomalies.sum(dim="level", skipna=True)
            
        _debug_report("after_vertical_sum", anomalies)

    stacked = []
    event_labels = []
    seen_onsets = set()
    for onset_str, _ in date_list:
        onset_ym = onset_str[:7]
        if onset_ym in seen_onsets:
            continue
        seen_onsets.add(onset_ym)

        window_start, window_end = _compute_composite_window_from_onset(
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
    event_stack = _circular_rolling_mean(event_stack, dim="month", window=args.rolling_period)
    _debug_report("event_stack", event_stack)
    event_stack.name = "AAMA_event_stack"
    return event_stack, event_labels, dphi_deg

def save_results(
    event_stack: xr.DataArray,
    composite: xr.DataArray,
    event_labels: list[str],
    active_pct: np.ndarray,
    args: argparse.Namespace,
    output_dir: str,
) -> tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    tag = (
        f"ERA5_Composite_AAM_{args.enso_state}_{args.start_year}-{args.end_year}"
        f"_{args.p_min:g}-{args.p_max:g}hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}"
    )
    result_path = os.path.join(output_dir, f"{tag}.nc")

    ds = xr.Dataset(
        {
            "composite_mean": composite,
            "active_month_percent": xr.DataArray(
                active_pct,
                dims=("month",),
                coords={"month": composite["month"]},
            ),
        }
    )
    ds.attrs.update(
        {
            "event_onsets": ",".join(event_labels),
            "n_events": int(event_stack.sizes["event"]),
            "enso_state": args.enso_state,
            "onset_season": args.onset_season,
            "composite_start": args.composite_start,
            "rolling_period": int(args.rolling_period),
            "region": args.region,
            "pressure_range_hpa": f"{args.p_min}-{args.p_max}",
        }
    )
    ds.to_netcdf(result_path)

    stack_path = None
    if args.save_event_stack:
        stack_path = os.path.join(output_dir, f"{tag}_event_stack.nc")
        event_stack.to_dataset(name="AAMA").to_netcdf(stack_path)
    return result_path, stack_path


def plot_composite(
    composite: xr.DataArray,
    significant: xr.DataArray,
    active_pct: np.ndarray,
    event_labels: list[str],
    args: argparse.Namespace,
    *,
    output_dir: str,
    dphi_deg: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    lat_dim = "latitude"
    comp = composite.transpose(lat_dim, "month")
    vals = comp.values
    lat_vals = comp[lat_dim].values
    month_vals = comp["month"].values

    vmax = float(np.nanpercentile(np.abs(vals), 99))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    levels = np.linspace(-vmax, vmax, 13)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.30)
    cf = ax.contourf(month_vals, lat_vals, vals, levels=levels, cmap="RdBu_r", extend="both")

    vmax = float(np.nanpercentile(np.abs(vals), 99))
    vmax = vmax if vmax > 0 else 1.0
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 13)
    
    _abs = max(abs(vmin), abs(vmax))
    order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
    factor = 10 ** order
    
    cax = fig.add_axes([0.125, 0.06, 0.775, 0.015])
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

    sig = significant.transpose(lat_dim, "month")
    insig = np.where(sig.values, 0, 1)
    if np.any(insig == 1):
        hatches = ax.contourf(
            month_vals,
            lat_vals,
            insig,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["//"],
            zorder=10,
        )
        import matplotlib.colors as mcolors
        line_color_with_alpha = mcolors.to_rgba('gray', alpha=0.4)
        for collection in hatches.collections:
            collection.set_facecolor("none")
            collection.set_edgecolor(line_color_with_alpha)
            if hasattr(collection, 'set_edgecolors'):
                collection.set_edgecolors([line_color_with_alpha])
            collection.set_linewidths([0.0])

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.8)
    for lat in (-40, -20, 20, 40):
        ax.axhline(lat, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=2)

    add_active_month_percent_labels(ax, month_vals, active_pct)

    state_pretty = "El Nino" if args.enso_state == "el_nino" else "La Nina"
    ax.set_xlim(1, args.composite_months)
    ax.set_ylim(-60, 60)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("Month since onset", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(
        f"ERA5 Reanalysis Mean Composite AAM anomaly\n"
        f"({args.p_min:g}-{args.p_max:g} hPa) {len(event_labels)} {state_pretty} events {args.start_year}-{args.end_year}clim {args.clim_start_year}-{args.clim_end_year}"
    )

    tag = (
        f"ERA5_AAM_composite_{args.enso_state}_{args.start_year}-{args.end_year}"
        f"_{args.p_min:g}-{args.p_max:g}hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}.png"
    )
    out_path = os.path.join(output_dir, tag)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.min_enso_months < 1:
        raise ValueError("--min-enso-months must be >= 1")
    if args.composite_months < 1:
        raise ValueError("--composite-months must be >= 1")
    if args.rolling_period < 1:
        raise ValueError("--rolling-period must be >= 1")

    output_dir = str(OUTPUT_DIR)

    nino34 = load_nino34(args)
    date_list = detect_enso_state_windows(args, nino34)
    if not date_list:
        raise RuntimeError("No ENSO events matched the requested criteria.")
    print(f"Detected {len(date_list)} ENSO event onset(s): {[onset[:7] for onset, _ in date_list]}")

    aam_da = load_era5_aam(args)
    clim_da = load_era5_climatology(args)
    event_stack, event_labels, dphi_float = build_event_stack(aam_da, clim_da, date_list, args)
    print(f"Built event stack with {event_stack.sizes['event']} complete event(s): {event_labels}")

    composite = event_stack.mean("event", skipna=True)
    composite.name = "AAMA"
    lat_dim = "latitude" if "latitude" in event_stack.dims else ("lat" if "lat" in event_stack.dims else None)
    if lat_dim is None:
        raise ValueError(f"No latitude dimension found in event stack with dims {event_stack.dims}")
    aam_for_ttest = event_stack.transpose("event", lat_dim, "month").values
    _, p_vals = _stats.ttest_1samp(aam_for_ttest, 0.0, axis=0, nan_policy="omit")
    significant = xr.DataArray(
        p_vals < 0.05,
        coords={lat_dim: event_stack[lat_dim], "month": event_stack["month"]},
        dims=(lat_dim, "month"),
        name="significant_mask",
    )
    active_pct = compute_active_month_percent(
        nino34,
        event_labels,
        composite_months=args.composite_months,
        composite_start=args.composite_start,
        enso_state=args.enso_state,
    )

    result_path, stack_path = save_results(
        event_stack,
        composite,
        event_labels,
        active_pct,
        args,
        output_dir,
    )
    plot_path = plot_composite(
        composite,
        significant,
        active_pct,
        event_labels,
        args,
        output_dir=output_dir,
        dphi_deg=dphi_float,
    )

    print(f"Saved composite results to {result_path}")
    if stack_path:
        print(f"Saved event stack to {stack_path}")
    print(f"Saved composite figure to {plot_path}")


if __name__ == "__main__":
    main()
