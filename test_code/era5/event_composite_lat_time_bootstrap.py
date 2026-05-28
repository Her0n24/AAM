"""
Bootstrap composite of deterministic ERA5 AAM anomalies from ENSO event onset.

Example
-------
Run from AAM/test_code/era5 or AAM/test_code:

python era5/event_composite_bootstrap.py \
  --start-year 1979 --end-year 2023 \
  --composite-months 24 --composite-start onset \
  --onset-season ndjfm --p-min 150 --p-max 700

The event setup follows the HadGEM3 bootstrap script:
- sustained ENSO state begins after at least N consecutive threshold months
- onset can be restricted to NDJFM
- composite windows start at onset month or December of onset year
- duplicate onset months are ignored
- bootstrap resamples the deterministic ERA5 event pool with replacement
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from plotting_utils import add_active_month_percent_labels, compute_active_month_percent

sys.path.append(str(Path(__file__).resolve().parents[1] / "CMIP6_HadGEM3_GC31"))
from utilities import (
    _to_per_latitude_band,
    vertical_sum_over_pressure_range,
)

try:
    import tqdm
except ImportError:  # pragma: no cover - convenience fallback for lean envs
    tqdm = None


ACTIVE_MONTH_EL_NINO_THRESHOLD = 0.5
ACTIVE_MONTH_LA_NINA_THRESHOLD = -0.5

REGION_BOUNDS = {
    "all": None,
    "pacific": (125.0, -110.0),
    "indian": (50.0, 100.0),
    "atlantic": (-60.0, 10.0),
}

ERA5_DIR = Path(__file__).resolve().parent
NINO34_CSV = ERA5_DIR / "nino34" / "nino34_HadlSST.csv"
AAM_FIG_DIR = ERA5_DIR / "AAMA_fig"
AAM_DATA_DIR = Path("/work/scratch-nopw2/hhhn2/ERA5/monthly_mean/AAM/full")
CLIMATOLOGY_DIR = ERA5_DIR / "climatology"
OUTPUT_DIR = ERA5_DIR / "composite_non_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic ERA5 AAM event composite and bootstrap significance over ENSO events."
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
    parser.add_argument("--n-iterations", type=int, default=2000, help="Bootstrap resample count.")
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Bootstrap confidence level.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for bootstrap resampling.")
    parser.add_argument(
        "--save-event-stack",
        action="store_true",
        help="Also save the full event stack used by the bootstrap.",
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


def _latitude_band_width_radians(lat_deg: np.ndarray) -> np.ndarray:
    lat_deg = np.asarray(lat_deg, dtype=float)
    if lat_deg.size < 2:
        return np.full_like(lat_deg, np.nan, dtype=float)
    lat_rad = np.deg2rad(lat_deg)
    edges = np.empty(lat_rad.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (lat_rad[1:] + lat_rad[:-1])
    edges[0] = lat_rad[0] - 0.5 * (lat_rad[1] - lat_rad[0])
    edges[-1] = lat_rad[-1] + 0.5 * (lat_rad[-1] - lat_rad[-2])
    return np.abs(np.diff(edges))


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
    selected = da.isel(longitude=mask)
    if selected.sizes.get("longitude", 0) == 0:
        raise ValueError(f"Region {region!r} selected zero longitudes.")
    return selected


# def _to_per_latitude_band(da: xr.DataArray) -> tuple[xr.DataArray, float]:
#     if "longitude" not in da.dims:
#         return da, _infer_latitude_band_width_deg(da)
#     da = da.sortby("latitude")
#     lon_rad = np.deg2rad(da["longitude"].astype(float))
#     da = da.assign_coords(longitude=lon_rad).sortby("longitude")
#     zonal_integral = da.integrate("longitude")
#     dphi = _latitude_band_width_radians(zonal_integral["latitude"].values)
#     band_width_deg = float(np.rad2deg(dphi)[0])
#     dphi_da = xr.DataArray(dphi, coords={"latitude": zonal_integral["latitude"]}, dims=("latitude",))
#     out = zonal_integral * dphi_da
#     out.attrs = dict(da.attrs)
#     out.attrs["zonal_reduction"] = "integral_radians"
#     out.attrs["lat_scaling"] = "dphi_radians"
#     return out, band_width_deg


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


def load_era5_aam(args: argparse.Namespace) -> tuple[xr.DataArray, bool]:
    anomaly_path = _resolve_precomputed_anomaly_file(args)
    if anomaly_path is not None:
        print(f"Opening precomputed ERA5 anomaly file: {anomaly_path}")
        ds = xr.open_dataset(anomaly_path)
        da = _first_data_var(ds, ("AAM_anomaly", "AAM", "aam", "AAMA")).load()
        da = _standardize_dims(da)
        if "time" not in da.dims:
            raise ValueError(f"Precomputed anomaly file must have a time dimension: {anomaly_path}")

        end_year_buffer = int(np.ceil(args.composite_months / 12.0)) + 1
        da = da.sel(time=slice(f"{args.start_year}-01-01", f"{args.end_year + end_year_buffer}-12-31"))
        ds.close()
        return da, True

    max_needed_year = args.end_year + int(np.ceil(args.composite_months / 12.0)) + 1
    files = []
    for year in range(args.start_year, max_needed_year + 1):
        files.extend(sorted(glob.glob(str(AAM_DATA_DIR / f"AAM_ERA5_{year}-*_full.nc"))))
    if not files:
        raise FileNotFoundError(f"No ERA5 AAM files found in fixed directory: {AAM_DATA_DIR}")

    print(f"Opening {len(files)} ERA5 AAM monthly files.")
    try:
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks={"time": 12, "level": 50, "latitude": 180, "longitude": 360},
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
    return da, False


def load_era5_climatology(args: argparse.Namespace) -> xr.DataArray:
    path = _resolve_climatology_file(args)
    print(f"Opening ERA5 AAM climatology: {path}")
    ds = xr.open_dataset(path)
    da = _first_data_var(ds, ("AAM", "aam", "angular_momentum", "momentum", "AAMA"))
    da = _standardize_dims(da)
    if "month" not in da.dims and "month" not in da.coords:
        raise ValueError(f"Climatology must have a monthly 'month' dimension or coord: {path}")
    return da


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
    clim_da: Optional[xr.DataArray],
    date_list: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[xr.DataArray, list[str], float]:
    aam = _select_region(aam_da, args.region)
    aam, dphi_deg = _apply_latitude_band_weighting(aam)
    if "level" in aam.dims:
        aam = vertical_sum_over_pressure_range(aam, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
    if clim_da is None:
        anomalies = aam
    else:
        clim = _select_region(clim_da, args.region)
        clim, clim_dphi_deg = _apply_latitude_band_weighting(clim)
        if not np.isfinite(dphi_deg):
            dphi_deg = clim_dphi_deg
        if "level" in clim.dims:
            clim = vertical_sum_over_pressure_range(clim, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")

        clim_months = clim["month"].values
        if np.size(clim_months) and np.nanmin(clim_months) == 0 and np.nanmax(clim_months) == 11:
            clim = clim.assign_coords(month=clim["month"] + 1)
        anomalies = aam.groupby("time.month") - clim

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
    event_stack.name = "AAMA_event_stack"
    return event_stack, event_labels, dphi_deg


def bootstrap_pooled_events(
    event_stack: xr.DataArray,
    *,
    n_iterations: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    n_events = int(event_stack.sizes["event"])
    if n_events < 2:
        raise ValueError("Bootstrap requires at least two complete events.")

    rng = np.random.default_rng(random_seed)
    means = []
    iterator = range(n_iterations)
    if tqdm is not None:
        iterator = tqdm.tqdm(iterator, desc=f"Bootstrapping {n_events} ERA5 events")
    for _ in iterator:
        indices = rng.choice(n_events, size=n_events, replace=True)
        means.append(event_stack.isel(event=indices).mean("event", skipna=True))

    boot = xr.concat(means, dim=pd.Index(np.arange(n_iterations), name="iteration"))
    alpha = (1.0 - confidence_level) / 2.0
    lower = boot.quantile(alpha, dim="iteration")
    upper = boot.quantile(1.0 - alpha, dim="iteration")
    significant = ((lower > 0) & (upper > 0)) | ((lower < 0) & (upper < 0))
    significant.name = "significance_mask"
    return boot, lower, upper, significant


def save_results(
    event_stack: xr.DataArray,
    composite: xr.DataArray,
    boot: xr.DataArray,
    lower: xr.DataArray,
    upper: xr.DataArray,
    significant: xr.DataArray,
    event_labels: list[str],
    active_pct: np.ndarray,
    args: argparse.Namespace,
    output_dir: str,
) -> tuple[str, Optional[str]]:
    os.makedirs(output_dir, exist_ok=True)
    tag = (
        f"ERA5_AAM_bootstrap_{args.enso_state}_{args.start_year}-{args.end_year}"
        f"_{args.p_min:g}-{args.p_max:g}hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}"
    )
    result_path = os.path.join(output_dir, f"{tag}.nc")

    if "quantile" in lower.coords:
        lower = lower.reset_coords("quantile", drop=True)
    if "quantile" in upper.coords:
        upper = upper.reset_coords("quantile", drop=True)

    ds = xr.Dataset(
        {
            "composite_mean": composite,
            "bootstrap_lower_bound": lower,
            "bootstrap_upper_bound": upper,
            "significance_mask": significant.astype(bool),
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
            "n_iterations": int(args.n_iterations),
            "confidence_level": float(args.confidence_level),
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
    sig = significant.transpose(lat_dim, "month")
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
        for collection in hatches.collections:
            collection.set_facecolor("none")
            collection.set_edgecolor((0.4, 0.4, 0.4, 0.35))
            collection.set_linewidth(0.0)

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
        f"ERA5 Reanalysis BOOTSTRAP Composite AAM anomaly\n"
        f"({args.p_min:g}-{args.p_max:g} hPa) {len(event_labels)} {state_pretty} events {args.start_year}-{args.end_year}clim {args.clim_start_year}-{args.clim_end_year}"
    )

    tag = (
        f"ERA5_AAM_bootstrap_{args.enso_state}_{args.start_year}-{args.end_year}"
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
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must be between 0 and 1")

    output_dir = str(OUTPUT_DIR)

    nino34 = load_nino34(args)
    date_list = detect_enso_state_windows(args, nino34)
    if not date_list:
        raise RuntimeError("No ENSO events matched the requested criteria.")
    print(f"Detected {len(date_list)} ENSO event onset(s): {[onset[:7] for onset, _ in date_list]}")

    aam_da, is_precomputed = load_era5_aam(args)
    clim_da = None if is_precomputed else load_era5_climatology(args)
    event_stack, event_labels, dphi_float = build_event_stack(aam_da, clim_da, date_list, args)
    print(f"Built event stack with {event_stack.sizes['event']} complete event(s): {event_labels}")

    composite = event_stack.mean("event", skipna=True)
    composite.name = "AAMA"
    boot, lower, upper, significant = bootstrap_pooled_events(
        event_stack,
        n_iterations=args.n_iterations,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
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
        boot,
        lower,
        upper,
        significant,
        event_labels,
        active_pct,
        args,
        output_dir,
    )
    plot_path = plot_composite(composite, significant, active_pct, event_labels, args, output_dir=output_dir, dphi_deg=dphi_float)

    print(f"Saved bootstrap results to {result_path}")
    if stack_path:
        print(f"Saved event stack to {stack_path}")
    print(f"Saved composite figure to {plot_path}")


if __name__ == "__main__":
    main()
