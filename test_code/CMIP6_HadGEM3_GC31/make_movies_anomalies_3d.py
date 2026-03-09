"""
This script plots the 3D structure of AAM anomalies from CMIP6 full field data.
Based on plot_momentum_anomalies_3d.py but adapted for CMIP6 data structure.

CMIP6 data structure:
- Single file with all years: AAM_CMIP6_HadGEM3_GC31_YYYY-MM_YYYY-MM.nc
- Dimensions: (time, level, latitude, longitude)
- Pressure levels in Pa
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
import matplotlib.cm as cm

# Allow importing shared utilities from AAM/test_code
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from plotting_utils import ensure_dir, plot_anomalies_3d_slices, plot_latitude_level_snapshots_HadGEN3  # noqa: E402


base_dir = os.getcwd()
AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"
climatology_path_base = f"{base_dir}/climatology/"

CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul"
u_directory = f"{CMIP6_path_base}/InterpolatedFlds/Amon/ua/historical/HadGEM3-GC31-LL/"
output_dir = f"{base_dir}/figures/"

# Create output directory if it doesn't exist
ensure_dir(output_dir)
ensure_dir(climatology_path_base)

# Default period 
start_yr, end_yr = 1980, 2000
clim_start_yr, clim_end_yr = 1980, 2000


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


def calculate_climatology(aam_file, clim_start_yr, clim_end_yr, ensemble_member, *, component='AAM'):
    """
    Calculate and save climatology for AAM variable.
    For CMIP6, we load the full field data and compute zonal mean then climatology.
    Returns the climatology DataArray.
    """
    # IMPORTANT: keep cache filename distinct from older zonal-mean climatologies.
    clim_kind = "latband_lonint"
    clim_file = (
        f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_"
        f"{clim_start_yr}-{clim_end_yr}_{clim_kind}.nc"
    )
    
    if os.path.exists(clim_file):
        print(f"Loading existing climatology from: {clim_file}")
        ds_climatology = xr.open_dataset(clim_file)
        try:
            if component not in ds_climatology.data_vars:
                raise KeyError(f"variable '{component}' not found in climatology file")
            da = ds_climatology[component]

            # Validate convention metadata to avoid mixing zonal-mean climatology with per-band anomalies.
            zr = str(da.attrs.get('zonal_reduction', ''))
            ls = str(da.attrs.get('lat_scaling', ''))
            kind = str(da.attrs.get('climatology_kind', ''))
            if (zr, ls, kind) != ('integral_radians', 'dphi_radians', clim_kind):
                print(
                    "Cached climatology exists but does not match per-lat-band convention; recomputing. "
                    f"(zonal_reduction={zr!r}, lat_scaling={ls!r}, kind={kind!r})"
                )
            else:
                return da
        finally:
            ds_climatology.close()
    
    print(f"Calculating climatology for AAM from {clim_start_yr} to {clim_end_yr}")
    
    # Load full field data
    ds = xr.open_dataset(aam_file)
    try:
        if component not in ds.data_vars:
            raise KeyError(f"variable '{component}' not found in dataset")
        aam_full = ds[component]

        # Verify time decoding
        print(f"Time coordinate info:")
        print(f"  First time: {aam_full.time.values[0]}")
        print(f"  Last time: {aam_full.time.values[-1]}")
        print(f"  Time dtype: {aam_full.time.dtype}")

        # Select climatology period using year filtering (works with cftime)
        years = np.array([t.year for t in aam_full.time.values])
        time_mask = (years >= clim_start_yr) & (years <= clim_end_yr)

        if time_mask.sum() > 0:
            aam_period = aam_full.isel(time=time_mask)
            print(f"Selected {time_mask.sum()} time steps from {clim_start_yr} to {clim_end_yr}")
        else:
            print(f"Warning: No data found for {clim_start_yr}-{clim_end_yr}, using full dataset")
            aam_period = aam_full

        # Convert to per-latitude-band magnitude: zonal integral (radians) × dphi (radians)
        aam_band = _to_per_latitude_band(aam_period)

        # Calculate monthly climatology
        da_climatology = aam_band.groupby('time.month').mean(dim='time')
        da_climatology.attrs = dict(aam_band.attrs)
        da_climatology.attrs['climatology_kind'] = clim_kind

        # Save climatology (as a dataset with compression)
        encoding = {component: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
        da_climatology.to_dataset(name=component).to_netcdf(clim_file, encoding=encoding)
        print(f"Climatology saved to: {clim_file}")

        return da_climatology
    finally:
        ds.close()


def plot_anomalies_3d(
    start_year,
    end_year,
    aam_file,
    clim_start_yr=1980,
    clim_end_yr=2000,
    *,
    ensemble_member='r1i1p1f3',
    find_extremum: str = 'max',
):
    """
    Plot 3D structure of AAM anomalies: multiple latitude×level slices at different times
    
    Parameters:
    -----------
    start_year : int
        Start year for plotting
    end_year : int
        End year for plotting
    aam_file : str
        Path to full field AAM file
    clim_start_yr : int
        Start year for climatology
    clim_end_yr : int
        End year for climatology
    """
    
    u_file = f'{u_directory}/ua_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc'
    ds_u = xr.open_dataset(u_file, mask_and_scale=True)
    
    # Calculate or load climatology (computed on per-latitude-band convention)
    da_climatology = calculate_climatology(aam_file, clim_start_yr, clim_end_yr, ensemble_member)
    
    # Load time series data
    ds = xr.open_dataset(aam_file)
    aam_full = ds['AAM']
    
    # Make missing values NaN explicitly to prevent artefacts
    fv = (
        da_climatology.encoding.get("_FillValue", None)
        or da_climatology.attrs.get("_FillValue", None)
        or da_climatology.attrs.get("missing_value", None)
    )
    print(f"_FillValue / missing_value detected in Climatology: {fv}")
    
    da_climatology = da_climatology.where(np.isfinite(da_climatology))  # drop inf/-inf if present
    
    if fv is not None:
        da_climatology = da_climatology.where(da_climatology != fv)
    
    fv = (
        aam_full.encoding.get("_FillValue", None)
        or aam_full.attrs.get("_FillValue", None)
        or aam_full.attrs.get("missing_value", None)
    )
    print(f"_FillValue / missing_value detected in AAM data: {fv}")
    
    aam_full = aam_full.where(np.isfinite(aam_full))  # drop inf/-inf if present
    
    if fv is not None:
        aam_full = aam_full.where(aam_full != fv)
        
    # Verify time coordinate decoding
    print(f"\n--- Time coordinate verification ---")
    time_var = ds['time']
    print(f"Time attributes: {dict(time_var.attrs)}")
    print(f"Time dtype: {time_var.dtype}")
    print(f"First 5 raw time values: {time_var.values[:5]}")
    print(f"Time range: {aam_full.time.values[0]} to {aam_full.time.values[-1]}")
    print(f"Total time steps in file: {len(aam_full.time)}\n")
    
    # Select time period using year filtering (works with cftime)
    years = np.array([t.year for t in aam_full.time.values])
    time_mask = (years >= start_year) & (years <= end_year)
    
    if time_mask.sum() > 0:
        aam_period = aam_full.isel(time=time_mask)
        print(f"Selected {time_mask.sum()} time steps from {start_year} to {end_year}")
    else:
        print(f"Warning: No data found for {start_year}-{end_year}, using full dataset")
        aam_period = aam_full
    
    # Convert to per-latitude-band magnitude: zonal integral (radians) × dphi (radians)
    aam_band = _to_per_latitude_band(aam_period)
    
    print(f"Time series shape: {aam_band.shape}")
    print(f"Climatology shape: {da_climatology.shape}")
    print(f"Climatology dimensions: {da_climatology.dims}")
    
    # Extract proper time coordinates
    proper_times = []
    for time_val in aam_band.time.values:
        # Convert cftime to pandas timestamp string
        year = time_val.year
        month = time_val.month
        proper_times.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    aam_band['time'] = proper_times
    print(f"Time range: {proper_times[0]} to {proper_times[-1]}")
    
    # Inflate climatology to match time series length
    months = aam_band.time.dt.month.values
    
    if 'month' in da_climatology.dims and len(da_climatology.month) == 12:
        # Expand climatology to full time series by selecting the right month for each timestep
        climatology_expanded = da_climatology.sel(month=xr.DataArray(months, dims='time'))
        climatology_expanded['time'] = aam_band.time
    else:
        # Climatology is incomplete - use simple broadcast/tile approach
        print(f"Warning: Climatology has only {len(da_climatology.month) if 'month' in da_climatology.dims else 0} month(s), using broadcast method")
        clim_single = da_climatology.squeeze(drop=True)
        climatology_expanded = xr.concat([clim_single] * len(aam_band.time), dim='time')
        climatology_expanded['time'] = aam_band.time
    
    # Compute anomalies
    anomalies = aam_band - climatology_expanded
    anomalies = anomalies.compute()
    
    print(f"Anomalies shape: {anomalies.shape}")

    # Quick sanity stats to diagnose sign/magnitude issues
    finite = np.isfinite(anomalies.values)
    n_finite = int(np.count_nonzero(finite))
    if n_finite > 0:
        vals = anomalies.values[finite]
        frac_pos = float(np.mean(vals > 0))
        print(
            "Anomalies finite stats: "
            f"min={np.nanmin(vals):.3e}, median={np.nanmedian(vals):.3e}, max={np.nanmax(vals):.3e}, "
            f"frac_pos={frac_pos:.2f}, n_finite={n_finite}"
        )
    else:
        print("Warning: anomalies has no finite values (all NaN/Inf)")

    # Build zonal-mean zonal wind for overlay (do NOT lon-integrate winds)
    try:
        wind_var = 'ua' if 'ua' in ds_u.data_vars else list(ds_u.data_vars)[0]
        u_full = ds_u[wind_var]

        # Apply same time selection by year and then compute zonal mean
        u_years = np.array([t.year for t in u_full.time.values])
        u_time_mask = (u_years >= start_year) & (u_years <= end_year)
        u_period = u_full.isel(time=u_time_mask) if u_time_mask.sum() > 0 else u_full

        if 'longitude' in u_period.dims and u_period.sizes['longitude'] > 1:
            u_zm = u_period.mean(dim='longitude')
        else:
            u_zm = u_period

        # Convert time coord to pandas Timestamp month-start so we can align to anomalies by coordinate
        u_times = []
        for t in u_zm.time.values:
            u_times.append(pd.Timestamp(f"{t.year}-{t.month:02d}-01"))
        u_zm = u_zm.assign_coords(time=u_times)

        try:
            u_zm = u_zm.sel(time=anomalies.time)
        except Exception:
            # Fallback: align by index length (still consistent with snapshot indexing)
            n = min(u_zm.sizes.get('time', 0), anomalies.sizes.get('time', 0))
            if n > 0:
                u_zm = u_zm.isel(time=slice(0, n))
    except Exception as exc:
        print(f"Warning: failed to prepare zonal wind overlay: {exc}")
        u_zm = None
    
    # Get coordinates
    lat_vals = anomalies.latitude.values
    level_vals = anomalies.level.values  # These are pressure levels in Pa
    
    # Convert pressure from Pa to hPa for plotting
    pressure_vals = level_vals / 100.0
    vertical_label = 'Pressure (hPa)'
    
    print(f"Data has {len(level_vals)} vertical levels")
    print(f"Pressure range: {pressure_vals.min():.1f} to {pressure_vals.max():.1f} hPa")
    
    # Check for NaN values in anomalies
    n_nans = np.isnan(anomalies.values).sum()
    if n_nans > 0:
        print(f"Warning: {n_nans} NaN values found in anomalies data")
    
    # === 3D SURFACE PLOT ===
    output_file = f"{output_dir}AAM_anomalies_3d_{ensemble_member}_{start_year}-{end_year}.png"
    plot_anomalies_3d_slices(
        anomalies,
        output_file=output_file,
        title=(
            f"CMIP6 HadGEM3_GC31 {ensemble_member} AAM Anomaly 3D Structure: Time × Latitude (mean) × Level\n"
            f"Climatology: {clim_start_yr}-{clim_end_yr}"
        ),
        time_step=3,
        vpercentile=95.0,
        cmap_name='RdBu_r',
    )
    print(f"3D figure saved to: {output_file}")
    
    # === SNAPSHOT PLOT ===
    # Show multiple latitude-level snapshots
    # Compute color limits from data (ignore NaN values)
    plot_latitude_level_snapshots_HadGEN3(
        anomalies,
        zonal_wind_da=u_zm,
        output_dir=output_dir,
        ensemble_member=ensemble_member,
        start_year=start_year,
        end_year=end_year,
        clim_start_yr=clim_start_yr,
        clim_end_yr=clim_end_yr,
        vpercentile=95.0,
        cmap_name='RdBu_r',
        find_extremum=find_extremum,
    )

    ds_u.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 3D structure of CMIP6 AAM anomalies')
    parser.add_argument(
        '--file',
        '-f',
        type=str,
        default=None,
        help=(
            'Path to AAM full field NetCDF file. If omitted, the script will use the default fixed path under '
            'monthly_mean/AAM/ for the selected ensemble member.'
        ),
    )
    parser.add_argument('--start', '-s', type=int, default=start_yr, help=f'start year for plotting (default: {start_yr})')
    parser.add_argument('--end', '-e', type=int, default=end_yr, help=f'end year for plotting (default: {end_yr})')
    parser.add_argument('--clim_start', type=int, default=clim_start_yr, help=f'climatology start year (default: {clim_start_yr})')
    parser.add_argument('--clim_end', type=int, default=clim_end_yr, help=f'climatology end year (default: {clim_end_yr})')
    parser.add_argument('--member', type=str, default='1', help='Ensemble member to plot (default: 1, control)')
    parser.add_argument('--find-min', action='store_true', help='Find and plot latitude of minimum AAM in northern hemisphere')
    args = parser.parse_args()
    
    ensemble_member = f"r{args.member}i1p1f3"

    if args.file is None:
        # Prefer the exact fixed filename if it exists; otherwise fall back to glob.
        fixed = f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc"
        if os.path.exists(fixed):
            aam_file = fixed
        else:
            pattern = f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_*.nc"
            matches = sorted(glob.glob(pattern))
            if len(matches) == 1:
                aam_file = matches[0]
            elif len(matches) == 0:
                raise FileNotFoundError(
                    f"No input file found. Looked for {fixed!r} and glob {pattern!r}. "
                    "Either place the file there or pass --file."
                )
            else:
                # Choose the last (lexicographically) to be deterministic.
                aam_file = matches[-1]
                print(f"Warning: Multiple matches for input file; using: {aam_file}")
    else:
        aam_file = args.file

    plot_anomalies_3d(
        args.start,
        args.end,
        aam_file,
        args.clim_start,
        args.clim_end,
        ensemble_member=ensemble_member,
        find_extremum=('min' if args.find_min else 'max'),
    )
