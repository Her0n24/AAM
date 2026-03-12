"""Shared plotting utilities for AAM scripts.

This module is intentionally kept dependency-light (numpy/xarray/matplotlib/pandas)
so it can be imported by both ERA5 and CMIP6 plotting scripts.

Typical usage from a script inside e.g. `AAM/test_code/era5/`:

    from pathlib import Path
    import sys

    # Add `AAM/test_code` to import path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from plotting_utils import (
        load_or_compute_monthly_climatology_from_file,
        plot_anomalies_3d_slices,
    )

"""

from __future__ import annotations

from calendar import month
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class ClimatologyCacheSpec:
    """Specifies where/how to cache climatology."""

    cache_file: str | Path
    variable: str


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _extract_years(time_values: np.ndarray) -> np.ndarray:
    """Extract year numbers from numpy/cftime/pandas-like time objects."""
    # Works for cftime objects (have .year) and numpy datetime64 (via astype).
    if time_values.size == 0:
        return np.array([], dtype=int)

    sample = time_values.flat[0]
    if hasattr(sample, "year"):
        return np.array([t.year for t in time_values], dtype=int)

    # numpy datetime64
    try:
        years = time_values.astype("datetime64[Y]").astype(int) + 1970
        return years.astype(int)
    except Exception as exc:  # pragma: no cover
        raise TypeError("Unsupported time dtype for year extraction") from exc


def compute_monthly_climatology(
    da: xr.DataArray,
    clim_start_year: int,
    clim_end_year: int,
) -> xr.DataArray:
    """Compute monthly climatology (month=1..12) for a DataArray with `time` dim."""
    if "time" not in da.dims:
        raise ValueError(f"Expected 'time' dim, got {da.dims}")

    years = _extract_years(da["time"].values)
    time_mask = (years >= clim_start_year) & (years <= clim_end_year)
    if time_mask.sum() > 0:
        da = da.isel(time=time_mask)

    clim = da.groupby("time.month").mean(dim="time")
    clim.attrs = dict(da.attrs)
    clim.attrs["climatology_years"] = f"{clim_start_year}-{clim_end_year}"
    return clim


def load_or_compute_monthly_climatology_from_file(
    data_file: str | Path,
    *,
    component: str,
    clim_start_year: int,
    clim_end_year: int,
    cache: Optional[ClimatologyCacheSpec] = None,
    reduce_lon: Optional[str] = "mean",
    preprocess: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
) -> xr.DataArray:
    """Load cached monthly climatology or compute + cache it.

    Parameters
    ----------
    data_file:
        NetCDF file containing `component` variable.
    component:
        Variable name.
    cache:
        If provided, climatology will be loaded/saved at `cache.cache_file`.
    reduce_lon:
        If not None and `longitude` exists, reduce over longitude using 'mean' or 'integrate'.
        For a climatology meant to be used with zonal-mean anomalies, leave as 'mean'.
    preprocess:
        Optional hook applied after load and before climatology computation.

    Returns
    -------
    xr.DataArray with dims ('month', ...) suitable for broadcasting.
    """
    if cache is not None:
        cache_path = Path(cache.cache_file)
        if cache_path.exists():
            ds_cache = xr.open_dataset(cache_path)
            try:
                if cache.variable not in ds_cache.data_vars:
                    raise KeyError(f"{cache.variable!r} missing in {str(cache_path)}")
                return ds_cache[cache.variable]
            finally:
                ds_cache.close()

        ensure_dir(cache_path.parent)

    ds = xr.open_dataset(data_file)
    try:
        if component not in ds.data_vars:
            raise KeyError(f"variable {component!r} not found in {str(data_file)}")
        da = ds[component]

        if preprocess is not None:
            da = preprocess(da)

        if reduce_lon is not None and "longitude" in da.dims:
            if reduce_lon == "mean":
                da = da.mean(dim="longitude")
            elif reduce_lon == "integrate":
                lon_rad = np.deg2rad(da["longitude"].astype(float))
                da = da.assign_coords(longitude=lon_rad).sortby("longitude")
                da = da.integrate("longitude")
            else:
                raise ValueError("reduce_lon must be one of: None, 'mean', 'integrate'")

        clim = compute_monthly_climatology(da, clim_start_year, clim_end_year)

        if cache is not None:
            cache_path = Path(cache.cache_file)
            encoding = {cache.variable: {"zlib": True, "complevel": 4, "dtype": "float32"}}
            clim.to_dataset(name=cache.variable).to_netcdf(cache_path, encoding=encoding)

        return clim
    finally:
        ds.close()


def plot_anomalies_3d_slices(
    anomalies: xr.DataArray,
    *,
    output_file: str | Path,
    title: str,
    time_step: int = 3,
    vpercentile: float = 95.0,
    cmap_name: str = "RdBu_r",
) -> None:
    """Plot a 3D time×latitude×pressure visualization from anomalies.

    Expects anomalies with dims including ('time', 'level', 'latitude') and *no* longitude.

    This is a lightweight, reusable version of the surface plot used in your scripts.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    required = {"time", "latitude"}
    if not required.issubset(set(anomalies.dims)):
        raise ValueError(f"anomalies must include {required}, got {anomalies.dims}")

    level_dim = "level" if "level" in anomalies.dims else ("plev" if "plev" in anomalies.dims else None)
    if level_dim is None:
        raise ValueError("anomalies must have a vertical level dimension ('level' or 'plev')")

    lat_vals = anomalies["latitude"].values
    level_vals = anomalies[level_dim].values

    # Convert Pa->hPa for plotting if needed
    level_max = float(np.nanmax(level_vals.astype(float)))
    if level_max > 2000.0:
        pressure_vals = level_vals.astype(float) / 100.0
        vertical_label = "Pressure (hPa)"
    else:
        pressure_vals = level_vals.astype(float)
        vertical_label = f"{level_dim}"

    time_indices = np.arange(0, len(anomalies["time"]), max(1, int(time_step)))
    n_slices = len(time_indices)

    data = anomalies.transpose("time", level_dim, "latitude").values
    vmax = np.nanpercentile(np.abs(data), vpercentile)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection="3d")

    LAT, PRESS = np.meshgrid(lat_vals, pressure_vals)

    for i, t_idx in enumerate(time_indices):
        data_slice = data[t_idx, :, :]  # (level, lat)
        ax.plot_surface(
            np.ones_like(data_slice) * i,
            LAT,
            PRESS,
            facecolors=cmap(norm(data_slice)),
            shade=False,
            alpha=0.8,
        )

    ax.set_xlabel("Time Index", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_zlabel(vertical_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_zaxis()

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("Anomaly", fontsize=12)

    output_file = Path(output_file)
    ensure_dir(output_file.parent)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_latitude_level_snapshots_HadGEN3(
    anomalies: xr.DataArray,
    zonal_wind_da: Optional["xr.DataArray | xr.Dataset"] = None,
    *,
    ensemble_member: str,
    start_year: int,
    end_year: int,
    clim_start_yr: int,
    clim_end_yr: int,
    vpercentile: float = 95.0,
    cmap_name: str = "RdBu_r",
    output_dir: str | Path = "output/",
    find_extremum: str = "max",
    title_suffix: str = "",
) -> None:
    """Plot a grid of latitude×level snapshots from anomalies.

    Expects anomalies with dims including ('time', 'level', 'latitude') and *no* longitude.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import pandas as pd
    
    vmax = np.nanpercentile(np.abs(anomalies.values), vpercentile)
    vmin = -vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == 0:
        print(f"Warning: Invalid color limits (vmin={vmin}, vmax={vmax}), using fallback values")
        vmax = 1e22
        vmin = -vmax
        
    n_snapshots = min(24, len(anomalies.time))
    snapshot_step = max(1, len(anomalies.time) // n_snapshots)
    snapshot_indices = np.arange(0, len(anomalies.time), snapshot_step)[:n_snapshots]
    
    if zonal_wind_da is not None:
        print("Loading zonal wind data for overlay...")
    
    # Determine subplot layout
    n_cols = 4
    n_rows = int(np.ceil(n_snapshots / n_cols))
    
    # Create figure with GridSpec for paired subplots (contour + profile)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(22, 6*n_rows))
    gs = GridSpec(n_rows * 2, n_cols, figure=fig, height_ratios=[6, 1] * n_rows, hspace=0.3, wspace=0.2)
    
    # Contour levels
    levels = np.linspace(vmin, vmax, 21)
    
    lat_dim = "latitude" if "latitude" in anomalies.dims else ("lat" if "lat" in anomalies.dims else None)
    level_dim = "level" if "level" in anomalies.dims else ("plev" if "plev" in anomalies.dims else None)
    if lat_dim is None:
        raise ValueError("anomalies must have a latitude dimension ('latitude' or 'lat')")
    if level_dim is None:
        raise ValueError("anomalies must have a vertical dimension ('level' or 'plev')")

    lat_vals = anomalies[lat_dim].values
    level_vals = anomalies[level_dim].values

    # Convert pressure from Pa -> hPa for plotting if needed
    level_units = str(anomalies[level_dim].attrs.get("units", "")).lower()
    level_max = float(np.nanmax(level_vals.astype(float)))
    looks_like_pa = ("pa" in level_units) or (level_max > 2000.0)
    pressure_hpa = level_vals.astype(float) / 100.0 if looks_like_pa else level_vals.astype(float)
    vertical_label = "Pressure (hPa)" if looks_like_pa or ("hpa" in level_units) else level_dim

    # Ensure vertical coordinate is monotonic increasing for contouring.
    # We'll later invert the axis so surface (large pressure) is at the bottom.
    finite_mask = np.isfinite(pressure_hpa)
    if np.count_nonzero(finite_mask) < 2:
        raise ValueError("pressure/level coordinate has insufficient finite values")
    if pressure_hpa[finite_mask][0] > pressure_hpa[finite_mask][-1]:
        pressure_hpa = pressure_hpa[::-1]
        anomalies_for_plot = anomalies.isel({level_dim: slice(None, None, -1)})
    else:
        anomalies_for_plot = anomalies
    
    contour_axes = []
    profile_axes = []
    for row in range(n_rows):
        for col in range(n_cols):
            contour_axes.append(fig.add_subplot(gs[row*2, col]))
            profile_axes.append(fig.add_subplot(gs[row*2 + 1, col]))
    
    for i, t_idx in enumerate(snapshot_indices):
        if i >= len(contour_axes):
            break
        
        _raw_t = anomalies_for_plot.time.values[t_idx]
        _is_composite_month = isinstance(_raw_t, (int, np.integer)) or (
            isinstance(_raw_t, float) and _raw_t == int(_raw_t) and 1 <= int(_raw_t) <= 36
        )
        _panel_label = f"Composite-{int(_raw_t):02d}" if _is_composite_month else pd.to_datetime(_raw_t).strftime("%Y-%m")
        data_slice = anomalies_for_plot.isel(time=t_idx).transpose(level_dim, lat_dim)
        
        # Use explicit levels array to ensure consistent colorbar
        levels = np.linspace(vmin, vmax, 21)
        im = contour_axes[i].contourf(lat_vals, pressure_hpa, data_slice.values,
                             levels=levels, cmap='RdBu_r', extend='both')
        
        # Overlay zonal wind contours
        if zonal_wind_da is not None:
            try:
                wind_data = zonal_wind_da
                if isinstance(wind_data, xr.Dataset):
                    wind_var = "ua" if "ua" in wind_data.data_vars else list(wind_data.data_vars)[0]
                    wind_data = wind_data[wind_var]
                
                # Get wind data coordinates
                wind_lat_dim = "latitude" if "latitude" in wind_data.dims else ("lat" if "lat" in wind_data.dims else None)
                wind_level_dim = "level" if "level" in wind_data.dims else ("plev" if "plev" in wind_data.dims else None)
                if wind_lat_dim is None or wind_level_dim is None:
                    raise ValueError("zonal wind must have latitude and vertical dimensions")
                wind_lat = wind_data[wind_lat_dim].values

                if "time" in wind_data.dims:
                    wind_data = wind_data.isel(time=t_idx)
                
                # Reduce any remaining longitude dimension (wind files often use 'lon')
                lon_dim = None
                if 'longitude' in wind_data.dims:
                    lon_dim = 'longitude'
                elif 'lon' in wind_data.dims:
                    lon_dim = 'lon'

                if lon_dim is not None:
                    if wind_data.sizes[lon_dim] > 1:
                        wind_data = wind_data.mean(dim=lon_dim)
                    else:
                        wind_data = wind_data.isel({lon_dim: 0})

                # Drop/resolve any other leftover dims so we can safely transpose to (level, lat)
                extra_dims = [d for d in wind_data.dims if d not in (wind_level_dim, wind_lat_dim)]
                for d in extra_dims:
                    if wind_data.sizes.get(d, 0) == 1:
                        wind_data = wind_data.isel({d: 0})
                    else:
                        raise ValueError(
                            f"Unexpected extra wind dimension {d!r} with size {wind_data.sizes[d]}"
                        )
                
                # Align wind vertical order with the AAM plot coordinate
                wind_level_vals = wind_data[wind_level_dim].values.astype(float)
                wind_level_units = str(wind_data[wind_level_dim].attrs.get("units", "")).lower()
                wind_level_max = float(np.nanmax(wind_level_vals))
                wind_looks_like_pa = ("pa" in wind_level_units) or (wind_level_max > 2000.0)
                wind_pressure_hpa = wind_level_vals / 100.0 if wind_looks_like_pa else wind_level_vals
                if np.isfinite(wind_pressure_hpa[0]) and np.isfinite(wind_pressure_hpa[-1]) and wind_pressure_hpa[0] > wind_pressure_hpa[-1]:
                    wind_pressure_hpa = wind_pressure_hpa[::-1]
                    wind_data = wind_data.isel({wind_level_dim: slice(None, None, -1)})

                wind_values = wind_data.transpose(wind_level_dim, wind_lat_dim).to_numpy()

                # If wind is on full levels and AAM is on half levels, average consecutive levels.
                if wind_values.shape[0] == len(pressure_hpa) + 1:
                    wind_values = (wind_values[:-1, :] + wind_values[1:, :]) / 2
                    wind_pressure_hpa = (wind_pressure_hpa[:-1] + wind_pressure_hpa[1:]) / 2

                # If vertical coordinates still don't match, skip overlay instead of mis-plotting.
                if wind_values.shape[0] != len(pressure_hpa):
                    raise ValueError(
                        f"wind vertical size ({wind_values.shape[0]}) does not match AAM ({len(pressure_hpa)})"
                    )

                # Use the same pressure_vals array that was used for the anomaly plot
                # This ensures wind contours align with the same vertical coordinate system
                wind_contour_levels = np.arange(-60, 61, 10)  # Contours every 10 m/s
                wind_contour_levels = wind_contour_levels[np.abs(wind_contour_levels) >= 10]  # Only show |u| >= 15 m/s
                cs = contour_axes[i].contour(wind_lat, pressure_hpa, wind_values,
                                    levels=wind_contour_levels, colors='black', 
                                    linewidths=0.8, alpha=0.6)
                contour_axes[i].clabel(cs, inline=True, fontsize=7, fmt='%d')
            except Exception as e:
                print(f"Warning: Could not overlay wind data for snapshot {i}: {e}")
        
        contour_axes[i].set_xlabel('Latitude (°N)', fontsize=10)
        contour_axes[i].set_xlim(-60, 60)
        contour_axes[i].set_ylabel(vertical_label, fontsize=10)
        contour_axes[i].set_title(_panel_label, fontsize=11, pad=3)
        # Pressure axis: decreasing upward, with sensible pressure ticks
        if vertical_label.lower().startswith("pressure") and np.all(np.isfinite(pressure_hpa)) and np.all(pressure_hpa > 0):
            pmin = float(np.nanmin(pressure_hpa))
            pmax = float(np.nanmax(pressure_hpa))
            if (pmax / max(pmin, 1e-6)) > 3.0:
                common_ticks = np.array([1000, 850, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10], dtype=float)
                ticks = common_ticks[(common_ticks >= pmin) & (common_ticks <= pmax)]
                if ticks.size >= 3:
                    contour_axes[i].set_yticks(ticks)
                    contour_axes[i].yaxis.set_major_formatter(mticker.ScalarFormatter())
                    contour_axes[i].yaxis.set_minor_formatter(mticker.NullFormatter())
        contour_axes[i].invert_yaxis()  # surface (large p) at bottom
        
        # Find latitude of maximum or minimum AAM in northern hemisphere.
        # Restrict to the plotted latitude window so the marker tracks what you see.
        nh_mask = (lat_vals > 0) & (lat_vals >= -60) & (lat_vals <= 60)
        if vertical_label.lower().startswith("pressure"):
            lvl_mask = pressure_hpa > 100  # Pressure level constraint
        else:
            lvl_mask = np.ones_like(pressure_hpa, dtype=bool)
        
        # Apply masks using proper 2D indexing
        nh_data = data_slice.values[np.ix_(lvl_mask, nh_mask)]
        nh_lats = lat_vals[nh_mask]
        
        # Find the extremum value and its location (NaN-safe)
        if not np.any(np.isfinite(nh_data)):
            extreme_lat = np.nan
        else:
            if find_extremum == 'min':
                extreme_idx = np.unravel_index(np.nanargmin(nh_data), nh_data.shape)
            else:  # default to 'max'
                extreme_idx = np.unravel_index(np.nanargmax(nh_data), nh_data.shape)
            extreme_lat = nh_lats[extreme_idx[1]]
        
        # Add vertical line at the latitude of extremum AAM
        if np.isfinite(extreme_lat):
            contour_axes[i].axvline(extreme_lat, color='C1', linewidth=2, linestyle='-', alpha=0.8, zorder=10)
        
        # Add vertical line at equator
        # contour_axes[i].axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.9)
        
        # Create vertical profile plot below
        # Vertically integrate AAM anomaly at each latitude
        # Always use pressure levels for integration (even if plotting with model levels)
        # Integrate over pressure (use Pa for x-units so the integral is physically consistent)
        pressure_pa = pressure_hpa * 100.0
        col_vals = data_slice.to_numpy()  # (level, lat)
        vertical_integral = np.full(col_vals.shape[1], np.nan, dtype=float)
        for j in range(col_vals.shape[1]):
            col = col_vals[:, j]
            m = np.isfinite(col) & np.isfinite(pressure_pa)
            if np.count_nonzero(m) >= 2:
                vertical_integral[j] = float(np.trapz(col[m], x=pressure_pa[m]))
        vi_plot = vertical_integral.copy()
        vi_finite = np.isfinite(vi_plot)
        if np.count_nonzero(vi_finite) >= 2 and np.count_nonzero(~vi_finite) > 0:
            vi_plot[~vi_finite] = np.interp(lat_vals[~vi_finite], lat_vals[vi_finite], vi_plot[vi_finite])
        
        profile_axes[i].plot(lat_vals, vi_plot, 'C0-', linewidth=1.5)
        profile_axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        profile_axes[i].axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        profile_axes[i].set_xlim(-60, 60)
        finite_vi = vertical_integral[np.isfinite(vertical_integral)]
        if finite_vi.size >= 2:
            y0 = float(np.nanmin(finite_vi))
            y1 = float(np.nanmax(finite_vi))
            if not np.isfinite(y0) or not np.isfinite(y1):
                profile_axes[i].set_ylim(-1e25, 1e25)
            elif y0 == y1:
                pad = max(abs(y0) * 0.05, 1.0)
                profile_axes[i].set_ylim(y0 - pad, y1 + pad)
            else:
                profile_axes[i].set_ylim(y0, y1)
        else:
            # All-NaN/Inf (or single finite value) -> avoid matplotlib ValueError
            profile_axes[i].set_ylim(-1e25, 1e25)
        profile_axes[i].set_ylabel('Total', fontsize=9)
        profile_axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(snapshot_indices), len(contour_axes)):
        contour_axes[j].axis('off')
        profile_axes[j].axis('off')
    
    # Add a single colorbar at the bottom for all subplots
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.01])  # [left, bottom, width, height]
    
    variable = anomalies.attrs.get("long_name", anomalies.name if anomalies.name is not None else "Variable")
    
    # Create discrete levels matching the contour levels
    levels = np.linspace(vmin, vmax, 11)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    sm = cm.ScalarMappable(cmap=cm.get_cmap('RdBu_r'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both', spacing='proportional')
    cbar.set_label(f'{variable} (kg m² s⁻¹)', fontsize=12)
    # Set ticks at every other level boundary for clarity
    tick_indices = np.arange(0, 11, 1)
    cbar.set_ticks(list(levels[tick_indices]))
    
    _suffix_str = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f'CMIP6 HadGEM3_GC31 {ensemble_member} zonally integrated AAM Anomaly: Latitude × Level Snapshots\n'
        f'Climatology: {clim_start_yr}-{clim_end_yr}{_suffix_str}',
        fontsize=26, y=0.99,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])  # Leave space for colorbar at bottom and reduce top gap

    output_file = f'{output_dir}/AAM_anomalies_lat_level_snapshots_{ensemble_member}_{start_year}-{end_year}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Snapshot figure saved to: {output_file}")
    plt.close()
    

def plot_latitude_level_movie_HadGEM3(
    anomalies: xr.DataArray,
    zonal_wind_da: Optional["xr.DataArray | xr.Dataset"] = None,
    *,
    ensemble_member: str,
    start_year: int,
    end_year: int,
    clim_start_yr: int,
    clim_end_yr: int,
    vpercentile: float = 95.0,
    cmap_name: str = "RdBu_r",
    output_dir: str | Path = "output/",
    find_extremum: str = "max",
    fps: int = 4,
) -> None:
    """
    Animate latitude×level frames into an MP4 movie.

    Expects anomalies with dims including ('time', 'level', 'latitude') and *no* longitude.
    Each frame shows one time step: a contourf of the anomaly with optional zonal-wind
    overlay and a vertically-integrated profile strip below.

    Requires ffmpeg to be available on the system PATH."""
    
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.gridspec import GridSpec

    vmax = np.nanpercentile(np.abs(anomalies.values), vpercentile)
    vmin = -vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == 0:
        print(f"Warning: Invalid color limits (vmin={vmin}, vmax={vmax}), using fallback values")
        vmax = 1e22
        vmin = -vmax

    lat_dim = "latitude" if "latitude" in anomalies.dims else ("lat" if "lat" in anomalies.dims else None)
    level_dim = "level" if "level" in anomalies.dims else ("plev" if "plev" in anomalies.dims else None)
    if lat_dim is None:
        raise ValueError("anomalies must have a latitude dimension ('latitude' or 'lat')")
    if level_dim is None:
        raise ValueError("anomalies must have a vertical dimension ('level' or 'plev')")

    lat_vals = anomalies[lat_dim].values
    level_vals = anomalies[level_dim].values

    level_units = str(anomalies[level_dim].attrs.get("units", "")).lower()
    level_max = float(np.nanmax(level_vals.astype(float)))
    looks_like_pa = ("pa" in level_units) or (level_max > 2000.0)
    pressure_hpa = level_vals.astype(float) / 100.0 if looks_like_pa else level_vals.astype(float)
    vertical_label = "Pressure (hPa)" if looks_like_pa or ("hpa" in level_units) else level_dim

    finite_mask = np.isfinite(pressure_hpa)
    if np.count_nonzero(finite_mask) < 2:
        raise ValueError("pressure/level coordinate has insufficient finite values")
    if pressure_hpa[finite_mask][0] > pressure_hpa[finite_mask][-1]:
        pressure_hpa = pressure_hpa[::-1]
        anomalies_for_plot = anomalies.isel({level_dim: slice(None, None, -1)})
    else:
        anomalies_for_plot = anomalies

    n_times = len(anomalies_for_plot.time)
    pressure_pa = pressure_hpa * 100.0
    levels_cont = np.linspace(vmin, vmax, 21)

    # Pressure tick marks for contour axis
    if (vertical_label.lower().startswith("pressure")
            and np.all(np.isfinite(pressure_hpa))
            and np.all(pressure_hpa > 0)):
        pmin = float(np.nanmin(pressure_hpa))
        pmax = float(np.nanmax(pressure_hpa))
        common_ticks = np.array([1000, 850, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10], dtype=float)
        p_ticks = common_ticks[(common_ticks >= pmin) & (common_ticks <= pmax)]
        p_ticks = p_ticks if p_ticks.size >= 3 else None
    else:
        p_ticks = None
        pmin = float(np.nanmin(pressure_hpa))
        pmax = float(np.nanmax(pressure_hpa))

    if zonal_wind_da is not None:
        print(f"Using zonal wind overlay for {n_times} frames ...")

    # Pre-compute vertically integrated profiles for consistent y-axis range
    all_vi: list[np.ndarray] = []
    for t_idx in range(n_times):
        data_sl = anomalies_for_plot.isel(time=t_idx).transpose(level_dim, lat_dim).to_numpy()
        vi = np.full(data_sl.shape[1], np.nan, dtype=float)
        for j in range(data_sl.shape[1]):
            col = data_sl[:, j]
            m = np.isfinite(col) & np.isfinite(pressure_pa)
            if np.count_nonzero(m) >= 2:
                vi[j] = float(np.trapz(col[m], x=pressure_pa[m]))
        all_vi.append(vi)

    finite_vi_all = np.concatenate([v[np.isfinite(v)] for v in all_vi])
    if finite_vi_all.size >= 2:
        vi_ymin = float(np.nanmin(finite_vi_all))
        vi_ymax = float(np.nanmax(finite_vi_all))
    else:
        vi_ymin, vi_ymax = -1e25, 1e25

    # Build static figure layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[6, 1], hspace=0.35)
    ax_cont = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1])

    # Static colorbar
    norm = mcolors.BoundaryNorm(np.linspace(vmin, vmax, 11), ncolors=256)
    cmap = cm.get_cmap(cmap_name)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
    variable = anomalies.attrs.get("long_name", anomalies.name if anomalies.name is not None else "Variable")
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", extend="both", spacing="proportional")
    cbar.set_label(f"{variable} (kg m\u00b2 s\u207b\u00b9)", fontsize=10)
    cbar.set_ticks(list(np.linspace(vmin, vmax, 11)))

    fig.suptitle(
        f"CMIP6 HadGEM3_GC31 {ensemble_member} zonally integrated AAM Anomaly: Latitude \u00d7 Level\n"
        f"Climatology: {clim_start_yr}\u2013{clim_end_yr}",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    def _render_frame(t_idx: int) -> None:
        ax_cont.cla()
        ax_prof.cla()

        time_val = pd.to_datetime(anomalies_for_plot.time.values[t_idx])
        data_slice = anomalies_for_plot.isel(time=t_idx).transpose(level_dim, lat_dim)

        ax_cont.contourf(lat_vals, pressure_hpa, data_slice.values, levels=levels_cont, cmap=cmap_name, extend="both")

        # Zonal wind overlay
        if zonal_wind_da is not None:
            try:
                wind_data = zonal_wind_da
                if isinstance(wind_data, xr.Dataset):
                    wind_var = "ua" if "ua" in wind_data.data_vars else list(wind_data.data_vars)[0]
                    wind_data = wind_data[wind_var]

                wind_lat_dim = "latitude" if "latitude" in wind_data.dims else ("lat" if "lat" in wind_data.dims else None)
                wind_level_dim = "level" if "level" in wind_data.dims else ("plev" if "plev" in wind_data.dims else None)
                if wind_lat_dim is None or wind_level_dim is None:
                    raise ValueError("zonal wind must have latitude and vertical dimensions")

                wind_lat = wind_data[wind_lat_dim].values
                wind_data_t = wind_data.isel(time=t_idx) if "time" in wind_data.dims else wind_data

                lon_dim = "longitude" if "longitude" in wind_data_t.dims else ("lon" if "lon" in wind_data_t.dims else None)
                if lon_dim is not None:
                    wind_data_t = wind_data_t.mean(dim=lon_dim) if wind_data_t.sizes[lon_dim] > 1 else wind_data_t.isel({lon_dim: 0})

                for d in [dd for dd in wind_data_t.dims if dd not in (wind_level_dim, wind_lat_dim)]:
                    if wind_data_t.sizes.get(d, 0) == 1:
                        wind_data_t = wind_data_t.isel({d: 0})
                    else:
                        raise ValueError(f"Unexpected extra wind dimension {d!r}")

                wind_level_vals = wind_data_t[wind_level_dim].values.astype(float)
                wind_level_units = str(wind_data_t[wind_level_dim].attrs.get("units", "")).lower()
                wind_looks_like_pa = ("pa" in wind_level_units) or (float(np.nanmax(wind_level_vals)) > 2000.0)
                wind_pressure_hpa = wind_level_vals / 100.0 if wind_looks_like_pa else wind_level_vals
                if np.isfinite(wind_pressure_hpa[0]) and np.isfinite(wind_pressure_hpa[-1]) and wind_pressure_hpa[0] > wind_pressure_hpa[-1]:
                    wind_pressure_hpa = wind_pressure_hpa[::-1]
                    wind_data_t = wind_data_t.isel({wind_level_dim: slice(None, None, -1)})

                wind_values = wind_data_t.transpose(wind_level_dim, wind_lat_dim).to_numpy()
                if wind_values.shape[0] == len(pressure_hpa) + 1:
                    wind_values = (wind_values[:-1, :] + wind_values[1:, :]) / 2
                    wind_pressure_hpa = (wind_pressure_hpa[:-1] + wind_pressure_hpa[1:]) / 2
                if wind_values.shape[0] != len(pressure_hpa):
                    raise ValueError(f"wind vertical size ({wind_values.shape[0]}) != AAM ({len(pressure_hpa)})")

                wind_contour_levels = np.arange(-60, 61, 10)
                wind_contour_levels = wind_contour_levels[np.abs(wind_contour_levels) >= 10]
                cs = ax_cont.contour(wind_lat, pressure_hpa, wind_values, levels=wind_contour_levels,
                                     colors="black", linewidths=0.8, alpha=0.6)
                ax_cont.clabel(cs, inline=True, fontsize=7, fmt="%d")
            except Exception as exc:
                print(f"Warning: wind overlay failed for frame {t_idx}: {exc}")

        # NH extremum marker
        nh_mask = (lat_vals > 0) & (lat_vals >= -60) & (lat_vals <= 60)
        lvl_mask = pressure_hpa > 100 if vertical_label.lower().startswith("pressure") else np.ones_like(pressure_hpa, dtype=bool)
        nh_data = data_slice.values[np.ix_(lvl_mask, nh_mask)]
        nh_lats = lat_vals[nh_mask]
        if np.any(np.isfinite(nh_data)):
            extreme_idx = np.unravel_index(
                np.nanargmin(nh_data) if find_extremum == "min" else np.nanargmax(nh_data),
                nh_data.shape,
            )
            extreme_lat = nh_lats[extreme_idx[1]]
            if np.isfinite(extreme_lat):
                ax_cont.axvline(extreme_lat, color="C1", linewidth=2, linestyle="-", alpha=0.8, zorder=10)

        ax_cont.set_xlabel("Latitude (°N)", fontsize=10)
        ax_cont.set_xlim(-60, 60)
        ax_cont.set_ylabel(vertical_label, fontsize=10)
        ax_cont.set_title(f"{time_val.strftime('%Y-%m')}", fontsize=12, pad=4)
        if p_ticks is not None:
            ax_cont.set_yticks(p_ticks)
            ax_cont.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax_cont.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax_cont.invert_yaxis()

        # Profile strip
        vi_plot = all_vi[t_idx].copy()
        vi_finite = np.isfinite(vi_plot)
        if np.count_nonzero(vi_finite) >= 2 and np.count_nonzero(~vi_finite) > 0:
            vi_plot[~vi_finite] = np.interp(lat_vals[~vi_finite], lat_vals[vi_finite], vi_plot[vi_finite])
        ax_prof.plot(lat_vals, vi_plot, "C0-", linewidth=1.5)
        ax_prof.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_prof.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_prof.set_xlim(-60, 60)
        ax_prof.set_ylim(vi_ymin, vi_ymax)
        ax_prof.set_ylabel("Total", fontsize=9)
        ax_prof.grid(True, alpha=0.3)

    ensure_dir(Path(output_dir))
    output_file = Path(output_dir) / f"AAM_anomalies_lat_level_movie_{ensemble_member}_{start_year}-{end_year}.mp4"
    print(f"Rendering {n_times} frames to {output_file} ...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    with writer.saving(fig, str(output_file), dpi=300):
        for t_idx in range(n_times):
            _render_frame(t_idx)
            writer.grab_frame()
            if (t_idx + 1) % 12 == 0:
                print(f"  Rendered {t_idx + 1}/{n_times} frames")
    plt.close(fig)
    print(f"Movie saved to: {output_file}")


__all__ = [
    "ClimatologyCacheSpec",
    "compute_monthly_climatology",
    "ensure_dir",
    "load_or_compute_monthly_climatology_from_file",
    "plot_anomalies_3d_slices",
    "plot_latitude_level_movie_HadGEM3",
    "plot_latitude_level_snapshots_HadGEN3",
]
