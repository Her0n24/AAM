"""Generate MP4 movies of AAM anomaly latitude×level structure from CMIP6 full field data.

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

# Allow importing shared utilities from AAM/test_code
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from plotting_utils import ensure_dir, plot_latitude_level_movie_HadGEM3  # noqa: E402
from utilities import vertical_sum_over_pressure_range, _reindex_to_climatology_dims, pressure_range_in_coord_units

# Vertical integration range for lat-lon movie (hPa)
P_MIN_HPA = 150.0
P_MAX_HPA = 700.0
MOVIE_FPS = 4


base_dir = os.getcwd()
# AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"
# climatology_path_base = f"{base_dir}/climatology/"

# CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul"
# u_directory = f"{CMIP6_path_base}/InterpolatedFlds/Amon/ua/historical/HadGEM3-GC31-LL/"
output_dir = f"{base_dir}/animation/"

CMIP6_path_base = "/work/scratch-nopw2/hhhn2"
u_directory = f"{CMIP6_path_base}/HadGEM3-GC31-LL/Amon/ua/historical/"
climatology_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/climatology/"
AAM_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/full/"

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
    clim_kind = ''
    clim_file = (
        f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_"
        f"{clim_start_yr}-{clim_end_yr}_{clim_kind}.nc"
    )
    
    if os.path.exists(clim_file):
        print(f"Loading existing climatology from: {clim_file}")
        ds_climatology = _safe_open_dataset(clim_file)
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
                    "Cached climatology exists but does not match per-lat-band convention; attempting conversion. "
                    f"(zonal_reduction={zr!r}, lat_scaling={ls!r}, kind={kind!r})"
                )

                # If the cached climatology is full-field (contains longitude), try to convert it
                lon_dim = 'longitude' if 'longitude' in da.dims else ('lon' if 'lon' in da.dims else None)
                if lon_dim is not None:
                    try:
                        # If climatology uses 'time' instead of 'month', aggregate into months
                        if 'month' not in da.dims and 'time' in da.dims:
                            try:
                                da_monthly = da.groupby('time.month').mean(dim='time')
                            except Exception:
                                da_monthly = da
                        else:
                            da_monthly = da

                        da_band = _to_per_latitude_band(da_monthly)
                        da_band.attrs = dict(da.attrs)
                        da_band.attrs['zonal_reduction'] = 'integral_radians'
                        da_band.attrs['lat_scaling'] = 'dphi_radians'
                        da_band.attrs['climatology_kind'] = clim_kind

                        # Save converted climatology (overwrite)
                        encoding = {component: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
                        da_band.to_dataset(name=component).to_netcdf(clim_file, encoding=encoding)
                        print(f"Converted full-field climatology saved to: {clim_file}")
                        return da_band
                    except Exception as exc:
                        print(f"Failed to convert cached full-field climatology: {exc}; will recompute from full-field AAM.")
                else:
                    print("Cached climatology is missing longitude dimension; will recompute from full-field AAM.")
            else:
                return da
        finally:
            ds_climatology.close()
    
    print(f"Calculating climatology for AAM from {clim_start_yr} to {clim_end_yr}")
    
    # Load full field data
    ds = _safe_open_dataset(aam_file)
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
def _safe_open_dataset(path, **kwargs):
    """
    Robustly open a dataset using xarray. Attempts:
      - glob pattern -> open_mfdataset
      - xr.open_dataset (default)
      - explicit engines: netcdf4, h5netcdf, scipy
    Raises FileNotFoundError if path not found.
    """
    if path is None:
        raise ValueError("No path provided to open")

    # Expand glob patterns
    matches = sorted(glob.glob(path))
    # include the literal path if it exists but wasn't matched by glob
    if os.path.exists(path) and os.path.isfile(path) and path not in matches:
        matches = [path]

    # If multiple files matched, try open_mfdataset
    if len(matches) > 1:
        try:
            return xr.open_mfdataset(matches, combine='by_coords', **kwargs)
        except Exception:
            # fall back to trying to open single file
            pass

    # Determine file to open
    file_to_open = matches[0] if len(matches) == 1 else path
    if not os.path.exists(file_to_open):
        raise FileNotFoundError(f"Input file not found: {file_to_open!r}")

    last_exc = None
    # Try default open first
    try:
        return xr.open_dataset(file_to_open, **kwargs)
    except Exception as exc:
        last_exc = exc

    # Try some common engines explicitly
    for engine in ('netcdf4', 'h5netcdf', 'scipy'):
        try:
            return xr.open_dataset(file_to_open, engine=engine, **kwargs)
        except Exception as exc:
            last_exc = exc

    raise ValueError(
        f"Failed to open dataset {file_to_open!r} with xarray; tried engines netcdf4/h5netcdf/scipy. "
        f"Original error: {last_exc}"
    )



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
    ds = _safe_open_dataset(aam_file)
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
    
    # === MOVIE PLOT ===
    plot_latitude_level_movie_HadGEM3(
        anomalies,
        zonal_wind_da=u_zm,
        output_dir=output_dir,
        ensemble_member=ensemble_member,
        start_year=start_year,
        end_year=end_year,
        clim_start_yr=clim_start_yr,
        clim_end_yr=clim_end_yr,
        vpercentile=99.0,
        cmap_name='RdBu_r',
        find_extremum=find_extremum,
    )

    # === LAT-LON MOVIE ===
    def render_latlon_movie(aam_period, u_dataset, pmin_hpa=P_MIN_HPA, pmax_hpa=P_MAX_HPA, fps=MOVIE_FPS):
        """Render a lat×lon MP4 movie of vertical-summed AAM anomalies.

        Parameters
        ----------
        aam_period : xr.DataArray
            Full-field AAM DataArray limited to the selected time period (time, level, lat, lon)
        u_dataset : xr.Dataset or xr.DataArray | None
            Zonal wind dataset (same time range) used for optional overlay.
        """
        import matplotlib.animation as animation
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # Vertical sum to produce lat×lon field (time, lat, lon)
        try:
            aam_vs = vertical_sum_over_pressure_range(aam_period, p_min_hpa=pmin_hpa, p_max_hpa=pmax_hpa, level_dim='level')
        except Exception as e:
            print(f"Could not compute vertical sum for lat-lon movie: {e}")
            return

        # Monthly climatology and anomaly
        try:
            clim_vs = aam_vs.groupby('time.month').mean('time')
            aam_vs, clim_on_time_vs = _reindex_to_climatology_dims(aam_vs, clim_vs)
            anom_vs = aam_vs - clim_on_time_vs
        except Exception as e:
            print(f"Could not compute lat-lon anomalies: {e}")
            return

        # Ensure time coords are pandas Timestamps (match earlier conversion)
        try:
            anom_vs = anom_vs.assign_coords(time=aam_band.time)
        except Exception:
            pass

        lat_vals = anom_vs['latitude'].values
        lon_vals = anom_vs['longitude'].values
        n_times = int(anom_vs.sizes['time'])

        # Patch longitudes if needed (add +180 column for seamless plotting)
        lon_step = np.round(np.diff(lon_vals).mean(), 6) if len(lon_vals) > 1 else None
        if lon_step is not None and np.isclose(lon_vals[0], -180) and not np.isclose(lon_vals[-1], 180):
            new_lon_vals = np.append(lon_vals, 180.0)
            arr = anom_vs.values
            arr_patched = np.concatenate([arr, arr[..., 0:1]], axis=-1)
            anom_vs = xr.DataArray(arr_patched, dims=anom_vs.dims, coords={**anom_vs.coords, 'longitude': new_lon_vals}, attrs=anom_vs.attrs)
            lon_vals = new_lon_vals

        # Prepare wind overlay at a single pressure level if possible
        wind_field_for_overlay = None
        if u_dataset is not None:
            try:
                wind_var = 'ua' if 'ua' in u_dataset.data_vars else list(u_dataset.data_vars)[0]
                u_full = u_dataset[wind_var]
                # select the same time slice as aam_period if possible
                try:
                    u_sel = u_full.sel(time=anom_vs.time)
                except Exception:
                    n = min(u_full.sizes.get('time', 0), anom_vs.sizes.get('time', 0))
                    u_sel = u_full.isel(time=slice(0, n)) if n > 0 else u_full
                # pick a representative level (nearest 250 hPa)
                try:
                    p_sel, _ = pressure_range_in_coord_units(u_sel[u_sel.dims[-1]] if 'plev' not in u_sel.dims else u_sel.plev, p_min_hpa=250.0, p_max_hpa=250.0)
                except Exception:
                    p_sel = None
                if p_sel is not None and 'plev' in u_sel.dims:
                    wind_field_for_overlay = u_sel.sel(plev=p_sel, method='nearest')
                else:
                    # if no plev, just try to use the first level/time-sliced array
                    wind_field_for_overlay = u_sel
            except Exception:
                wind_field_for_overlay = None

        # Build figure with explicit dpi and canvas attachment for headless rendering
        # Switch to Agg backend explicitly for headless environments
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plt.figure(figsize=(12, 6), dpi=200)
        
        # Attach Agg canvas immediately (before adding axes) to ensure dpi is accessible to writer
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            canvas = FigureCanvas(fig)
            fig.canvas = canvas
            # Force draw to initialize the canvas properly
            fig.canvas.draw()
        except Exception as e:
            print(f"Warning: Could not attach/draw FigureCanvas: {e}")
        
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))

        # Color limits
        vpercentile = 99.0
        vmin = -np.nanpercentile(np.abs(anom_vs.values), vpercentile)
        vmax = np.nanpercentile(np.abs(anom_vs.values), vpercentile)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
            vmin = -1.0

        levels = np.linspace(vmin, vmax, 21)

        output_file = Path(output_dir) / f"AAM_anomalies_lat_lon_movie_{ensemble_member}_{start_year}-{end_year}_{int(pmin_hpa)}-{int(pmax_hpa)}hPa.mp4"
        ensure_dir(output_file.parent)

        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        with writer.saving(fig, str(output_file), dpi=200):
            for t_idx in range(n_times):
                ax.cla()
                time_val = pd.to_datetime(anom_vs.time.values[t_idx])
                data_slice = anom_vs.isel(time=t_idx).values
                im = ax.contourf(lon_vals, lat_vals, data_slice, levels=levels, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
                ax.coastlines(resolution='110m', linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                ax.set_title(time_val.strftime('%Y-%m'))

                # wind overlay
                if wind_field_for_overlay is not None:
                    try:
                        w = wind_field_for_overlay.isel(time=t_idx) if 'time' in wind_field_for_overlay.dims else wind_field_for_overlay
                        # collapse extra dims if present
                        for d in [d for d in w.dims if d not in ('latitude', 'lat', 'longitude', 'lon')]:
                            if w.sizes.get(d, 0) == 1:
                                w = w.isel({d: 0})
                        wind_lat_dim = 'latitude' if 'latitude' in w.dims else ('lat' if 'lat' in w.dims else None)
                        wind_lon_dim = 'longitude' if 'longitude' in w.dims else ('lon' if 'lon' in w.dims else None)
                        if wind_lat_dim is not None and wind_lon_dim is not None:
                            wvals = w.transpose(wind_lat_dim, wind_lon_dim).values
                            ax.contour(w[w_lon_dim if False else wind_lon_dim].values, w[w_lat_dim if False else wind_lat_dim].values, wvals, levels=np.arange(-60,61,10), colors='k', linewidths=0.6, transform=ccrs.PlateCarree())
                    except Exception:
                        pass

                writer.grab_frame()
        plt.close(fig)
        print(f"Lat-lon movie saved to: {output_file}")

    # call to render lat-lon movie (produces an MP4)
    try:
        render_latlon_movie(aam_period=aam_period, u_dataset=ds_u, pmin_hpa=P_MIN_HPA, pmax_hpa=P_MAX_HPA, fps=MOVIE_FPS)
    except Exception as e:
        print(f"Failed to render lat-lon movie: {e}")

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
