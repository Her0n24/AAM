
import numpy as np
import xarray as xr
import os 
import pandas as pd
import glob 
# Allow importing shared utilities from AAM/test_code
import sys
from pathlib import Path
from typing import Optional, Tuple
sys.path.append(str(Path(__file__).resolve().parents[1]))

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


def _zonal_integral_radians(da: xr.DataArray) -> xr.DataArray:
    lon_dim = 'longitude' if 'longitude' in da.dims else ('lon' if 'lon' in da.dims else None)
    if lon_dim is None:
        raise ValueError("Expected a longitude dimension ('longitude' or 'lon')")
    lon_rad = np.deg2rad(da[lon_dim].astype(float))
    da = da.assign_coords({lon_dim: lon_rad}).sortby(lon_dim)
    return da.integrate(lon_dim)


def _to_per_latitude_band(da: xr.DataArray) -> xr.DataArray:
    """
    Converts a zonally integrated xarray.DataArray to per-latitude-band
    values by multiplying with the latitude band width in radians. Expects a
    latitude dimension named latitude or lat.
    """
    lat_dim = 'latitude' if 'latitude' in da.dims else ('lat' if 'lat' in da.dims else None)
    if lat_dim is None:
        raise ValueError("Expected a latitude dimension ('latitude' or 'lat')")

    da = da.sortby(lat_dim)
    da_lonint = _zonal_integral_radians(da)

    dphi = _latitude_band_width_radians(da_lonint[lat_dim].values)
    dphi_da = xr.DataArray(dphi, coords={lat_dim: da_lonint[lat_dim]}, dims=(lat_dim,))
    dphi_da.attrs['units'] = 'radian'

    out = da_lonint * dphi_da
    out.attrs = dict(da.attrs)
    out.attrs['zonal_reduction'] = 'integral_radians'
    out.attrs['lat_scaling'] = 'dphi_radians'
    return out

def _reindex_to_climatology_dims(input_da: xr.DataArray, climatology_da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Align `input_da` to the non-time dims of `climatology_da`, and expand the climatology over `time`.

    Assumptions (common for monthly climatologies):
    - `input_da` has a `time` coordinate with datetime-like values.
    - `climatology_da` has a `month` dimension with values in 1..12 (or 0..11).

    Returns:
    - input_da_reindexed: `input_da` reindexed onto the climatology's shared dims (e.g., level/latitude).
    - climatology_on_time: `climatology_da` selected by `input_da.time.dt.month`, yielding a DataArray with `time`.
    """
    if "time" not in input_da.dims:
        raise ValueError("Expected `da` to have a 'time' dimension")
    if "month" not in climatology_da.dims and "month" not in climatology_da.coords:
        raise ValueError("Expected `clim_da` to have a 'month' dimension/coord")

    # Rename common dimension aliases in `input_da` to match `climatology_da`.
    rename_map: dict[str, str] = {}
    for src, dst in (("lat", "latitude"), ("lev", "level"), ("plev", "level"), ("lon", "longitude")):
        if dst in climatology_da.dims and src in input_da.dims and dst not in input_da.dims:
            rename_map[src] = dst
    if rename_map:
        input_da = input_da.rename(rename_map)

    # Reindex shared dims (excluding month/time) to the climatology grid.
    for dim in climatology_da.dims:
        if dim in ("month",):
            continue
        if dim in input_da.dims:
            input_da = input_da.reindex({dim: climatology_da[dim]})

    # Expand monthly climatology onto the input time axis.
    time_coord = input_da["time"]

    try:
        months = time_coord.dt.month
    except Exception as e:
        vals = time_coord.values
        if vals.size and hasattr(vals[0], "month"):
            months = xr.DataArray(
                [t.month for t in vals],
                coords={"time": time_coord},
                dims=("time",),
            )
        else:
            raise TypeError(
                "Expected `da.time` to be datetime-like (datetime64 or cftime)"
            ) from e
    months = time_coord.dt.month
    clim_months = climatology_da["month"].values
    if np.size(clim_months) and np.nanmin(clim_months) == 0 and np.nanmax(clim_months) == 11:
        months = months - 1

    climatology_on_time = climatology_da.sel(month=months)
    return input_da, climatology_on_time

def pressure_range_in_coord_units(
    level_coord: xr.DataArray, p_min_hpa: float, p_max_hpa: float
) -> tuple[float, float]:
    """Return (pmin, pmax) in same units as `level_coord` (Pa vs hPa)."""
    lev_vals = np.asarray(level_coord.values, dtype=float)
    lev_max = float(np.nanmax(lev_vals))
    if lev_max > 2000.0:  # heuristic: Pa
        return p_min_hpa * 100.0, p_max_hpa * 100.0
    return p_min_hpa, p_max_hpa


def vertical_sum_over_pressure_range(
    da: xr.DataArray,
    *,
    p_min_hpa: float,
    p_max_hpa: float,
    level_dim: str = "level",
) -> xr.DataArray:
    """Select pressure range and sum over `level_dim`.

    Note: this matches your current plotting scripts, which assume dp has already
    been applied upstream (so a plain sum is correct here).
    """
    if level_dim not in da.dims:
        raise ValueError(f"Expected {level_dim!r} dim in da, got {da.dims}")

    da = da.sortby(level_dim)
    pmin_u, pmax_u = pressure_range_in_coord_units(da[level_dim], p_min_hpa, p_max_hpa)

    da = da.sel({level_dim: slice(min(pmin_u, pmax_u), max(pmin_u, pmax_u))})
    out = da.sum(dim=level_dim, skipna=True)

    # optional metadata
    out.attrs = dict(da.attrs)
    out.attrs["vertical_sum_range_hpa"] = f"{p_min_hpa}-{p_max_hpa}"
    return out


def get_ENSO_index(start_year, end_year, ensemble_member, 
                   nino34_directory = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/") -> tuple[Optional[pd.DatetimeIndex], Optional[np.ndarray]]:
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