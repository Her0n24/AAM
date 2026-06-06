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


class TaylorDiagram(object):
    """Taylor diagram for normalized standard deviation and correlation."""

    def __init__(
        self,
        refstd,
        fig=None,
        rect=111,
        label="_",
        min_corr=0.9,
        min_std=0.8,
        max_std=1.25,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        max_angle = np.arccos(min_corr)

        self.refstd = refstd
        self.min_std = min_std
        self.max_std = max_std
        self.max_angle = max_angle

        tr = PolarAxes.PolarTransform()
        rlocs = np.linspace(min_std, max_std, 5)
        corr_ticks = np.array([-0.5, 0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        corr_ticks = corr_ticks[corr_ticks >= min_corr]
        tlocs = np.arccos(corr_ticks)
        tlocs = tlocs[tlocs <= max_angle]
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, [f"{np.cos(t):.2f}" for t in tlocs])))
        gl2 = GF.FixedLocator(rlocs)
        tf2 = GF.DictFormatter(dict(zip(rlocs, [f"{val:.2f}" for val in rlocs])))

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, max_angle, min_std, max_std),
            grid_locator1=gl1,
            tick_formatter1=tf1,
            grid_locator2=gl2,
            tick_formatter2=tf2,
        )
        if fig is None:
            fig = plt.figure()
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].major_ticklabels.set_pad(10)
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_fontsize(16)

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["left"].label.set_fontsize(16)

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["left"].major_ticklabels.set_fontsize(12)
        ax.axis["top"].major_ticklabels.set_fontsize(12)
        ax.axis["right"].major_ticklabels.set_fontsize(12)
        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        self.ax.plot([0], self.refstd, "k*", ls="", ms=10, label=label)
        t = np.linspace(0, max_angle)
        self.ax.plot(t, np.zeros_like(t) + self.refstd, "k--", label="_")

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        return self.ax.plot(np.arccos(np.clip(corrcoef, -1.0, 1.0)), stddev, *args, **kwargs)


def compute_taylor_stats_against_reference(
    ref_da: xr.DataArray,
    sample_da: xr.DataArray,
    *,
    sample_dim: str = "iteration",
) -> "object":
    """Return a DataFrame with per-sample normalized std and correlation to ref_da."""
    import pandas as pd

    if sample_dim not in sample_da.dims:
        raise ValueError(f"sample_da must contain {sample_dim!r}; got dims {sample_da.dims}")

    ref_flat = np.asarray(ref_da.values, dtype=float).ravel()
    ref_mask = np.isfinite(ref_flat)
    ref_flat = ref_flat[ref_mask]
    if ref_flat.size == 0:
        return pd.DataFrame(columns=[sample_dim, "std", "corr"])

    ref_mean = float(np.mean(ref_flat))
    ref_scale = float(np.std(ref_flat))
    if not np.isfinite(ref_scale) or ref_scale <= 0:
        return pd.DataFrame(columns=[sample_dim, "std", "corr"])

    ref_norm = (ref_flat - ref_mean) / ref_scale
    rows = []
    for sample_idx in range(int(sample_da.sizes[sample_dim])):
        sample_flat = np.asarray(sample_da.isel({sample_dim: sample_idx}).values, dtype=float).ravel()
        if sample_flat.size != ref_mask.size:
            continue
        sample_flat = sample_flat[ref_mask]
        valid = np.isfinite(sample_flat) & np.isfinite(ref_norm)
        if np.count_nonzero(valid) < 3:
            continue
        sample_norm = (sample_flat[valid] - ref_mean) / ref_scale
        corr = float(np.corrcoef(ref_norm[valid], sample_norm)[0, 1])
        std = float(np.std(sample_norm))
        if np.isfinite(corr) and np.isfinite(std):
            rows.append({sample_dim: int(sample_idx), "std": std, "corr": corr})

    return pd.DataFrame(rows)


def plot_taylor_diagram_from_stats(
    stats_df,
    *,
    output_path: str | Path,
    stats_csv_path: str | Path | None = None,
    percentiles_csv_path: str | Path | None = None,
    title: str,
    reference_label: str = "Reference",
    sample_label: str = "Bootstrap samples",
    min_corr: float = 0.9,
    min_std: float = 0.8,
    max_std: float = 1.2,
    sample_color: str = "black",
    sample_alpha: float = 0.8,
    sample_ms: float = 1.0,
    draw_percentiles: bool = True,
    label_column: str | None = None,
    label_fontsize: float = 12.0,
) -> bool:
    """Plot a Taylor diagram from a stats DataFrame containing std and corr columns."""
    import pandas as pd
    import matplotlib.pyplot as plt

    required_cols = {"std", "corr"}
    if not required_cols.issubset(stats_df.columns):
        print(f"Taylor stats missing required columns {required_cols}; skipping {output_path}")
        return False

    df = stats_df.copy()
    df = df[np.isfinite(df["std"]) & np.isfinite(df["corr"])]
    df = df[(df["corr"] >= -1.0) & (df["corr"] <= 1.0)]
    if df.empty:
        print(f"Taylor stats are empty after filtering; skipping {output_path}")
        return False

    corr_vals = df["corr"].values
    std_vals = df["std"].values
    corr_low = float(np.nanpercentile(corr_vals, 1))
    std_low = float(np.nanpercentile(std_vals, 1))
    std_high = float(np.nanpercentile(std_vals, 99))

    if np.isfinite(corr_low) and corr_low < min_corr:
        min_corr = max(-1.0, np.floor((corr_low - 0.02) * 20.0) / 20.0)
        print(f"Taylor diagram axis expanded to min_corr={min_corr:.2f} for {output_path}")
    if np.isfinite(std_low) and std_low < min_std:
        min_std = max(0.0, np.floor((std_low - 0.02) * 10.0) / 10.0)
        print(f"Taylor diagram axis expanded to min_std={min_std:.2f} for {output_path}")
    if np.isfinite(std_high) and std_high > max_std:
        max_std = np.ceil((std_high + 0.02) * 10.0) / 10.0
        print(f"Taylor diagram axis expanded to max_std={max_std:.2f} for {output_path}")
    if max_std <= min_std:
        max_std = min_std + 0.2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_csv_path is not None:
        stats_csv_path = Path(stats_csv_path)
        stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(stats_csv_path, index=False)
        print(f"Saved Taylor statistics ({len(df)} samples) to {stats_csv_path}")

    corr_pct = np.nanpercentile(df["corr"].values, [5, 50, 95])
    std_pct = np.nanpercentile(df["std"].values, [5, 50, 95])

    fig = plt.figure(figsize=(10, 8))
    dia = TaylorDiagram(
        1.0,
        fig=fig,
        label=reference_label,
        min_corr=min_corr,
        min_std=min_std,
        max_std=max_std,
    )
    dia.ax.plot(
        np.arccos(np.clip(df["corr"].values, -1.0, 1.0)),
        df["std"].values,
        marker="o",
        color=sample_color,
        alpha=sample_alpha,
        ms=sample_ms,
        ls="",
        label="_nolegend_",
    )
    if label_column is not None and label_column in df.columns:
        
        for _, row in df.iterrows():
            corr_val = float(row["corr"])
            std_val = float(row["std"])
            label_text = str(row[label_column])
            if not label_text:
                continue
            theta = float(np.arccos(np.clip(corr_val, -1.0, 1.0)))
            
            # --- CUSTOM HIGHLIGHT FOR THE CONTROL RUN (MEMBER 1) ---
            if label_text == "1":
                # Plot the special orange diamond directly over its standard black dot
                dia.ax.plot(
                    theta,
                    std_val,
                    marker="D",          
                    color="darkorange",  
                    mec="black",         
                    mew=1.2,             
                    ms=sample_ms * 2.2,  # Make it larger to completely engulf/hide the black dot underneath
                    ls="",
                    zorder=5,            # Bring to front layer
                    label="HadGEM3 CTRL" 
                )
                
                # Create text object for member 1
                dia.ax.text(
                    theta,
                    std_val,
                    f" {label_text}",    
                    fontsize=label_fontsize + 2.0, 
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    color="darkorange",
                    zorder=6
                )
            
    # --- ADD MINOR RADIAL ARC AT 0.5 STD DEV ---
    theta_grid = np.linspace(0, dia.max_angle, 240)
    dia.ax.plot(
        theta_grid,
        np.full_like(theta_grid, 0.5),  # Creates an array of 0.5 values matching theta_grid
        color="gray",                   # Neutral color for a minor gridline
        linestyle="--",                 # Dashed line style
        linewidth=1.0,                  # Slightly thinner than the main percentile lines
        alpha=0.3,                      # Faint transparency requested
        label="_nolegend_",             # Keeps it out of the legend
        zorder=1                        # Puts it in the background behind data points
    )

    if draw_percentiles:
        percentile_specs = [
            ("5th", "royalblue", corr_pct[0], std_pct[0]),
            ("median", "seagreen", corr_pct[1], std_pct[1]),
            ("95th", "firebrick", corr_pct[2], std_pct[2]),
        ]
        theta_grid = np.linspace(0, dia.max_angle, 240)

        radial_span = max(float(dia.max_std - dia.min_std), 1e-6)

        # Place the correlation annotations on an inner radial band so they do
        # not collide with the right-hand correlation tick labels.
        corr_label_rs = np.linspace(
            dia.min_std + 0.18 * radial_span,
            dia.min_std + 0.34 * radial_span,
            len(percentile_specs),
        )
        std_label_thetas = np.linspace(
            0.02 * dia.max_angle,
            0.12 * dia.max_angle,
            len(percentile_specs),
        )
        
        for idx, (pct_label, color, corr_val, std_val) in enumerate(percentile_specs):
            theta = float(np.arccos(np.clip(corr_val, -1.0, 1.0)))
            dia.ax.plot(
                [theta, theta],
                [dia.min_std, dia.max_std],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label="_nolegend_",
            )
            dia.ax.plot(
                theta_grid,
                np.full_like(theta_grid, float(std_val)),
                color=color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label="_nolegend_",
            )

            # Nudge the text slightly inward along the angular direction too.
            corr_label_theta = min(theta - 0.015 * dia.max_angle, dia.max_angle - 0.03 * dia.max_angle)
            
            # Plot correlation labels with a protective white background box
            dia.ax.text(
                corr_label_theta,
                float(corr_label_rs[idx]),  # Removed the +0.04 overshoot pushing it outside
                f"r={float(corr_val):.3f}",
                ha="left",
                va="center",
                color=color,
                fontsize=13,
                clip_on=False,
            )

            # Plot std dev labels cleanly cutting through the dashed lines using the bbox
            std_label_r = float(np.clip(std_val, dia.min_std + 0.01 * radial_span, dia.max_std - 0.01 * radial_span))
            dia.ax.text(
                float(std_label_thetas[idx]),
                std_label_r,
                f"{float(std_val):.3f}",
                ha="right",  # Center alignment looks much cleaner inside a bbox
                va="center",
                color=color,
                fontsize=13,
                clip_on=False,
            )

        if percentiles_csv_path is not None:
            percentiles_csv_path = Path(percentiles_csv_path)
            percentiles_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "percentile": [5, 50, 95],
                    "correlation": corr_pct,
                    "standard_deviation": std_pct,
                }
            ).to_csv(percentiles_csv_path, index=False)
            print(f"Saved Taylor percentiles to {percentiles_csv_path}")

    handles, labels = dia.ax.get_legend_handles_labels()
    visible = [(handle, label) for handle, label in zip(handles, labels) if label and not label.startswith("_")]
    if visible:
        legend_handles, legend_labels = zip(*visible)
        plt.legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(0.98, 1.02), fontsize=12, frameon=False)

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.84, right=0.82, left=0.08, bottom=0.08)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Taylor diagram to {output_path}")
    return True


def add_active_month_percent_labels(
    ax,
    month_vals,
    active_month_percent,
    *,
    label_text: str = "ENSO active (%)",
    label_x: float = -0.06,
    label_y: float = -0.155,
    text_y: float = -0.19,
    label_fontsize: int = 14,
    text_fontsize: int = 10,
    every_n: int = 2,
    label_colors: tuple[str, ...] = ("#7c7b66", "#9e833b", "#f39c34", "#e85d3a", "#bd491f"),
) -> None:
    """Draw colored ENSO-active percentage labels under a monthly axis."""
    if active_month_percent is None:
        return

    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    pct_cmap = LinearSegmentedColormap.from_list("enso_active_pct", list(label_colors))
    pct_norm = Normalize(vmin=0, vmax=100)

    ax.text(
        label_x,
        label_y,
        label_text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=label_fontsize,
    )

    for month_idx, (month_val, pct_val) in enumerate(zip(month_vals, active_month_percent)):
        if every_n > 1 and month_idx % every_n != 0:
            continue
        color = pct_cmap(pct_norm(int(pct_val)))
        if isinstance(color, np.ndarray):
            color = mcolors.to_rgba(color)
        ax.text(
            month_val,
            text_y,
            f"{int(pct_val)}%",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=text_fontsize,
            fontweight="bold",
            color=color,
            clip_on=False,
        )


def compute_active_month_percent(
    nino34,
    event_labels,
    *,
    composite_months: int,
    composite_start: str,
    enso_state: str,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Compute the percentage of ENSO-active months across event windows."""
    import pandas as pd

    if threshold is None:
        threshold = 0.5 if enso_state == "el_nino" else -0.5

    total = np.zeros(int(composite_months), dtype=int)
    n_valid = 0

    for onset_ym in event_labels:
        onset_ts = pd.Timestamp(f"{onset_ym}-01")
        if composite_start == "december_onset_year":
            window_start = pd.Timestamp(f"{onset_ts.year}-12-01")
        else:
            window_start = pd.Timestamp(f"{onset_ts.year}-{onset_ts.month:02d}-01")
        window_end = window_start + pd.DateOffset(months=int(composite_months) - 1)

        vals = nino34.loc[pd.Period(window_start.strftime("%Y-%m"), "M") : pd.Period(window_end.strftime("%Y-%m"), "M")].values
        if vals.size < int(composite_months):
            continue

        if enso_state == "el_nino":
            active = vals[: int(composite_months)] >= float(threshold)
        else:
            active = vals[: int(composite_months)] <= float(threshold)

        total += np.asarray(active, dtype=int)
        n_valid += 1

    if n_valid == 0:
        return np.zeros(int(composite_months), dtype=int)

    return np.rint(100.0 * total / float(n_valid)).astype(int)

def plot_lat_lon_snapshots(
anomalies: xr.DataArray,
zonal_wind_da: Optional["xr.DataArray | xr.Dataset"] = None,
uv_latlev_profile: Optional[xr.DataArray] = None,
p_values: Optional[np.ndarray] = None,
significance_mask: Optional[np.ndarray] = None,
*,
ensemble_member: str,
start_year: int,
end_year: int,
clim_start_yr: int,
clim_end_yr: int,
vpercentile: float = 99.0,
cmap_name: str = "RdBu_r",
output_dir: str | Path = "output/",
find_extremum: str = "max",
title_suffix: str = "",
rolling_period: int | None = None,
filename_suffix: str = "",
dec_onset_month: Optional[str] = None,
onset_season_ndjfm: Optional[str] = None,
pmin: Optional[float] = None,
pmax: Optional[float] = None,
nino_threshold: Optional[float] = None,
region: str = "all",
) -> None:
    
    """Plot a grid of latitude × longitude anomaly snapshots for CMIP6 composites (HadGEM3 style).
    
    Parameters
    ----------
    anomalies : xr.DataArray
        Data with dims (time, latitude, longitude)
    p_values : np.ndarray, optional
        P-values with shape (time, latitude, longitude). Where p > 0.05, regions will be hatched.
    significance_mask : np.ndarray, optional
        Boolean mask with shape (time, latitude, longitude). Where True, regions will be hatched.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm
    from scipy.ndimage import gaussian_filter

    coastlines, borders = True, True
    lat_extent: tuple = (-60, 60)
    n_snapshots = min(24, anomalies.sizes["time"])
    tick_fontsize = 35
    title_fontsize = 25
    suptitle_fontsize = 32
    
    # Detect and normalize dimension names
    lat_dim = "latitude" if "latitude" in anomalies.dims else ("lat" if "lat" in anomalies.dims else None)
    lon_dim = "longitude" if "longitude" in anomalies.dims else ("lon" if "lon" in anomalies.dims else None)
    
    if not {"time", lat_dim, lon_dim}.issubset(set(anomalies.dims)) or lat_dim is None or lon_dim is None:
        raise ValueError(f"anomalies must have dims (time, latitude|lat, longitude|lon), got {anomalies.dims}")

    # Normalize dimension names to standard forms for internal use
    if lat_dim != "latitude" or lon_dim != "longitude":
        anomalies = anomalies.rename({lat_dim: "latitude", lon_dim: "longitude"})
        lat_dim, lon_dim = "latitude", "longitude"

    lat_vals = anomalies["latitude"].values
    lon_vals = anomalies["longitude"].values
    time_vals = anomalies["time"].values

    # Debug: report region/filename inputs so runs clearly indicate intended output
    print(f"plot_lat_lon_snapshots: region='{region}', filename_suffix='{filename_suffix}'")

    # --- Patch: Add +180° longitude column for seamless plotting if needed ---
    # If longitude runs from -180 to 177.5 (step 2.5), add a +180 column duplicating -180 values
    # Only patch if longitude is regularly spaced and missing +180 endpoint
    lon_step = np.round(np.diff(lon_vals).mean(), 6) if len(lon_vals) > 1 else None
    if lon_step is not None and np.isclose(lon_vals[0], -180) and not np.isclose(lon_vals[-1], 180):
        # Add +180 column
        new_lon_vals = np.append(lon_vals, 180.0)
        # Patch DataArray
        if isinstance(anomalies, xr.DataArray):
            arr = anomalies.values
            arr_patched = np.concatenate([arr, arr[..., 0:1]], axis=-1)
            # Create new DataArray with updated longitude
            anomalies = xr.DataArray(
                arr_patched,
                dims=anomalies.dims,
                coords={**anomalies.coords, "longitude": new_lon_vals},
                attrs=anomalies.attrs,
                name=anomalies.name,
            )
            lon_vals = new_lon_vals
        else:
            # Dataset: patch each variable with lon dim
            for v in anomalies.data_vars:
                if "longitude" in anomalies[v].dims:
                    arr = anomalies[v].values
                    arr_patched = np.concatenate([arr, arr[..., 0:1]], axis=-1)
                    anomalies[v].data_vars[v].values[...] = arr_patched
            anomalies = anomalies.assign_coords(longitude=new_lon_vals)
            lon_vals = new_lon_vals

    # Choose snapshots
    n_times = len(time_vals)
    n_snapshots = min(n_snapshots, n_times)
    snapshot_indices = np.linspace(0, n_times - 1, n_snapshots, dtype=int)

    # Color limits (symmetric, data-driven) so weak but real anomalies remain visible.
    vmax = float(np.nanpercentile(np.abs(anomalies.values), vpercentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(anomalies.values))) if np.size(anomalies.values) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 21)

    from matplotlib.gridspec import GridSpec
    # Layout
    n_cols = 3
    n_rows = int(np.ceil(n_snapshots / n_cols))
    
    fig = plt.figure(figsize=(35, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols * 2, figure=fig, width_ratios=[0.6, 6] * n_cols,
              hspace=0.25, wspace=0.15)
    
    map_axes = []
    profile_axes = []
    
    # Center map based on region: use dateline (180°) for Pacific and global (all); use Greenwich (0°) for others
    region_central_lons = {
        "pacific": 180,
        "all": 180,
        "indian": 75,      # Center Indian Ocean (50-100°E)
        "atlantic": -30,   # Center Atlantic (-60-10°E)
    }
    central_lon = region_central_lons.get(region, 0)
    map_crs = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()

    for i, t_idx in enumerate(snapshot_indices):
        if i >= n_snapshots:
            break
        
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col * 2 + 1], projection=map_crs)
        ax_profile = fig.add_subplot(gs[row, col * 2])
        
        map_axes.append(ax)
        profile_axes.append(ax_profile)
        
        data_slice = anomalies.isel(time=t_idx)
        time_val = pd.to_datetime(time_vals[t_idx])
        
        _raw_t = anomalies.time.values[t_idx]
        _is_composite_month = isinstance(_raw_t, (int, np.integer)) or (
            isinstance(_raw_t, float) and _raw_t == int(_raw_t) and 1 <= int(_raw_t) <= 36
        )
        if _is_composite_month:
            if rolling_period is not None and rolling_period > 1:
                _panel_label = f"Composite-Rolling-{int(_raw_t):02d}"
            else:
                _panel_label = f"Composite-Month-{int(_raw_t):02d}"
        else:
            _panel_label = pd.to_datetime(_raw_t).strftime("%Y-%m")
        
        data_values = data_slice.values
        if data_values.shape != (len(lat_vals), len(lon_vals)) and data_values.T.shape == (len(lat_vals), len(lon_vals)):
            data_values = data_values.T

        im = ax.contourf(lon_vals, lat_vals, data_values, levels=levels, cmap=cmap_name, extend="both", transform=data_crs)
        
        # -------------------------------
        # SIGNIFICANCE HATCHING (IPCC STYLE)
        # -------------------------------
        hatch_mask = None

        # Case 1: Using Precomputed Boolean Mask (True = Significant)
        if significance_mask is not None and significance_mask.shape[0] > t_idx:
            sig_slice = significance_mask[t_idx, :, :]
            
            # 1. Handle Transpose
            if sig_slice.shape != data_values.shape:
                if sig_slice.T.shape == data_values.shape:
                    sig_slice = sig_slice.T

            # 2. Handle Longitude Wrap-around patching
            if sig_slice.shape[1] != data_values.shape[1]:
                sig_slice = np.concatenate([sig_slice, sig_slice[:, 0:1]], axis=1)

            # 3. Create Hatch Mask: 1.0 where NOT significant (sig_slice is False)
            # We want to hatch where sig_slice is False.
            hatch_mask = np.where(sig_slice == False, 1.0, 0.0)

        # Case 2: Fallback to Raw P-Values (Float)
        elif p_values is not None and p_values.shape[0] > t_idx:
            p_slice = p_values[t_idx, :, :]
            
            # 1. Handle Transpose
            if p_slice.shape != data_values.shape:
                if p_slice.T.shape == data_values.shape:
                    p_slice = p_slice.T
            
            # 2. Handle Longitude Wrap-around patching
            if p_slice.shape[1] != data_values.shape[1]:
                p_slice = np.concatenate([p_slice, p_slice[:, 0:1]], axis=1)
            
            # 3. Create Hatch Mask: 1.0 where p > 0.05 OR p is NaN
            hatch_mask = np.where((p_slice > 0.05) | np.isnan(p_slice), 1.0, 0.0)

        def _align_hatch_mask(mask: np.ndarray, target_shape: tuple[int, int], label: str) -> Optional[np.ndarray]:
            mask = np.asarray(mask)
            if mask.shape == target_shape:
                return mask
            if mask.T.shape == target_shape:
                return mask.T
            if mask.shape[0] == target_shape[0] and mask.shape[1] > target_shape[1]:
                return mask[:, : target_shape[1]]
            if mask.shape[1] == target_shape[1] and mask.shape[0] > target_shape[0]:
                return mask[: target_shape[0], :]
            print(f"Skipping {label} hatch overlay due to shape mismatch: mask {mask.shape}, data {target_shape}")
            return None

        if hatch_mask is not None:
            hatch_mask = _align_hatch_mask(hatch_mask, data_values.shape, "significance")

        # Draw the hatches only where the mask marks insignificant cells.
        # pcolor applies the hatch to every rendered patch, so we mask out the
        # significant cells first and let Matplotlib draw only the remaining ones.
        if hatch_mask is not None:
            hatch_data = np.ma.masked_where(hatch_mask != 1, hatch_mask)
            if hatch_data.shape != data_values.shape and hatch_data.T.shape == data_values.shape:
                hatch_data = hatch_data.T

            # Use contourf to target only the '1.0' (insignificant) values
            # levels=[0.5, 1.5] ensures we only hatch the insignificant regions
            h_plot = ax.contourf(
                lon_vals, 
                lat_vals, 
                hatch_data,
                levels=[0.5, 1.5], 
                hatches=['/'],  # Dense hatches
                colors='none',     # No background fill
                zorder=10,
                transform=ccrs.PlateCarree() # Ensure transform is included for Cartopy
            )

            # Style the hatches and remove the contour boundaries
            for collection in h_plot.collections:
                collection.set_edgecolor((0.4, 0.4, 0.4, 0.6)) # Gray hatches
                collection.set_linewidth(0.0) # Removes the boundary of the hatched area
        
        # Overlay zonal wind contours
        if zonal_wind_da is not None:
            try:
                #import pdb; pdb.set_trace()
                if isinstance(zonal_wind_da, xr.Dataset):
                    wind_var = "ua" if "ua" in zonal_wind_da.data_vars else list(zonal_wind_da.data_vars)[0]
                    wind_data = zonal_wind_da[wind_var]
                else:
                    wind_data = zonal_wind_da
                
                # Get wind data coordinates
                wind_lat_dim = "latitude" if "latitude" in wind_data.dims else ("lat" if "lat" in wind_data.dims else None)
                wind_lon_dim = "longitude" if "longitude" in wind_data.dims else ("lon" if "lon" in wind_data.dims else None)
                if wind_lat_dim is None or wind_lon_dim is None:
                    raise ValueError("zonal wind must have latitude and longitude dimensions")
                wind_lat = wind_data[wind_lat_dim].values
                wind_lon = wind_data[wind_lon_dim].values
                
                if "time" in wind_data.dims:
                    wind_data = wind_data.isel(time=t_idx)
                else:
                    raise ValueError("zonal wind DataArray must have a time dimension for snapshot plotting")
                
                # Drop/resolve any other leftover dims so we can safely transpose to (lat, lon)
                extra_dims = [d for d in wind_data.dims if d not in (wind_lon_dim, wind_lat_dim)]
                for d in extra_dims:
                    if wind_data.sizes.get(d, 0) == 1:
                        wind_data = wind_data.isel({d: 0})
                    else:
                        raise ValueError(
                            f"Unexpected extra wind dimension {d!r} with size {wind_data.sizes[d]}"
                        )
                
                wind_vals = gaussian_filter(wind_data.values, sigma=1)  # shape: (lat, lon)
                wind_peak = float(np.nanmax(np.abs(wind_vals)))
                if np.isfinite(wind_peak) and wind_peak >= 30:
                    # 1. Set the step size to 15
                    wind_step = 15.0
                    # 2. Find the maximum limit required to cover the peak wind speed
                    wind_max = max(30.0, wind_step * np.ceil(wind_peak / wind_step))
                    
                    # 3. Generate steps: e.g., [-45, -30, -15, 0, 15, 30, 45]
                    wind_contour_levels = np.arange(-wind_max, wind_max + wind_step, wind_step)
                    
                    # 4. Filter: Keep only values where absolute magnitude is >= 30
                    # This drops 0, 15, and -15 entirely
                    wind_contour_levels = wind_contour_levels[np.abs(wind_contour_levels) >= 30.0]
                else:
                    wind_contour_levels = np.array([])
                if wind_contour_levels.size == 0:
                    raise ValueError("wind field has no finite amplitude to contour")
                cs = ax.contour(wind_lon, wind_lat, wind_vals,
                                    levels=wind_contour_levels, colors='C4', 
                                    linewidths=1.5, alpha=1.0)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
            except Exception as e: 
                raise ValueError("Failed to identify zonal wind variable and coordinates") from e
        
        if uv_latlev_profile is not None:
            uv_vals = uv_latlev_profile.isel(time=t_idx).values
            ax_profile.plot(uv_vals, lat_vals, color="C1", lw=2)
            ax_profile.axhline(0, color="black", linestyle="-", lw=1)
            ax_profile.set_xlabel("uv", fontsize=10)
            if i % n_cols == 0:
                ax_profile.set_ylabel("Latitude (°N)", fontsize=10)
            else: 
                ax_profile.set_ylabel("")
                ax_profile.tick_params(labelleft=False)
            ax_profile.tick_params(labelsize=10)
            ax_profile.set_xlim(-300,300)
            ax_profile.set_ylim(-60,60)
            ax_profile.grid(True, linestyle="--", alpha=0.5)
                
        if coastlines:
            ax.coastlines(resolution="110m", linewidth=0.5)
        if borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        # Set geographic extent in data CRS so region slices are displayed in correct location.
        if region == "indian":
            ax.set_extent([50, 100, lat_extent[0], lat_extent[1]], crs=data_crs)
        elif region == "atlantic":
            ax.set_extent([-60, 10, lat_extent[0], lat_extent[1]], crs=data_crs)
        elif region == "pacific":
            # 125E to 110W (250E) across dateline.
            ax.set_extent([125, 250, lat_extent[0], lat_extent[1]], crs=data_crs)
        else:
            lon_min, lon_max = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
            ax.set_extent([lon_min, lon_max, lat_extent[0], lat_extent[1]], crs=data_crs)
        ax.set_xlabel("Longitude (°E)", fontsize=tick_fontsize)
        ax.set_ylabel("Latitude (°N)", fontsize=tick_fontsize)
        ax.set_title(_panel_label, fontsize=title_fontsize, pad=3)
        ax.tick_params(labelsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        # Mark extremum in NH
        nh_mask = lat_vals > 0
        data_nh = data_slice.values[nh_mask, :]
        lat_nh = lat_vals[nh_mask]
        if np.any(np.isfinite(data_nh)):
            if find_extremum == "min":
                extreme_idx = np.unravel_index(np.nanargmin(data_nh), data_nh.shape)
            else:
                extreme_idx = np.unravel_index(np.nanargmax(data_nh), data_nh.shape)
            extreme_lat = lat_nh[extreme_idx[0]]
            extreme_lon = lon_vals[extreme_idx[1]]
            ax.plot(extreme_lon, extreme_lat, color="C2", marker="x", markersize=12, markeredgewidth=2, transform=data_crs)

    # Colorbar
    # Make colorbar shorter in height and increase tick label font size
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.01])  # [left, bottom, width, height]
    
    variable = anomalies.attrs.get("long_name", anomalies.name if anomalies.name is not None else "Variable")

    # Create discrete levels matching the contour levels
    levels = np.linspace(vmin, vmax, 13)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    sm = cm.ScalarMappable(cmap=cm.get_cmap('RdBu_r'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both', spacing='proportional')
    cbar.set_label(f'{variable} (kg m² s⁻¹)', fontsize=35)
    # Set ticks at every other level boundary for clarity
    tick_indices = np.arange(0, 13, 2)
    cbar.set_ticks(list(levels[tick_indices]))
    # Increase colorbar tick label size and scientific notation font size
    cbar.ax.tick_params(labelsize=32)
    # Make scientific notation (e.g., 1e24) larger
    cbar.ax.yaxis.get_offset_text().set_fontsize(32)
    cbar.ax.xaxis.get_offset_text().set_fontsize(32)

    _suffix_str = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f'CMIP6 HadGEM3_GC31 {ensemble_member} vertically summed {pmin:.1f}-{pmax:.1f}hPa\n'
        f'AAM anomaly: Latitude × Longitude Snapshots Climatology: {clim_start_yr}-{clim_end_yr}\n'
        f'{_suffix_str}',
        fontsize=suptitle_fontsize, y=0.96,
    )
    
    rolling_tag = ""
    if rolling_period is not None:
        rp = int(rolling_period)
        if rp > 1:
            rolling_tag = f"_rolling{rp}"

    file_suffix = ""
    if filename_suffix:
        file_suffix = f"_{str(filename_suffix).strip('_')}"
        
    if dec_onset_month == "onset":
        pass
    elif dec_onset_month == "december_onset_year":
        file_suffix += "_december_onset_year"
    
    if onset_season_ndjfm == "ndjfm":
        file_suffix += "_onset_season_NDJFM"
    else:
        file_suffix += "_onset_season_all"
        
    nino_thres_tag = "nino_thres" + str(float(nino_threshold)) if nino_threshold is not None else ""

    # Compose filename including region explicitly and ensure output directory exists
    fname = (
        f'AAM_anomalies_lat_lon_snapshots_{ensemble_member}_{start_year}-{end_year}_{pmin:.1f}-{pmax:.1f}hPa_{region}{rolling_tag}_{nino_thres_tag}{file_suffix}.svg'
    )
    output_path = Path(output_dir) / fname
    ensure_dir(output_path.parent)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])  # Leave space for colorbar at bottom and title at top
    plt.savefig(output_path, format = "svg", bbox_inches="tight")
    print(f"Lat-lon snapshot figure saved to: {output_path}")
    print(f"Lat-lon snapshot figure saved to: {output_path}")
    plt.close(fig)


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
    p_values: Optional[np.ndarray] = None,
    significance_mask: Optional[np.ndarray] = None,
    *,
    ensemble_member: str,
    start_year: int,
    end_year: int,
    clim_start_yr: int,
    clim_end_yr: int,
    vpercentile: float = 99.0,
    cmap_name: str = "RdBu_r",
    output_dir: str | Path = "output/",
    find_extremum: str = "max",
    title_suffix: str = "",
    rolling_period: int | None = None,
    filename_suffix: str = "",
    dec_onset_month: Optional[str] = "",
    onset_season_ndjfm: Optional[str] = "",
    nino_threshold: Optional[float] = 0.5) -> None:
    """Plot a grid of latitude×level snapshots from anomalies.

    Parameters
    ----------
    anomalies : xr.DataArray
        Data with dims (time, level, latitude)
    zonal_wind_da : xr.DataArray or xr.Dataset, optional
        Zonal wind for overlay
    p_values : np.ndarray, optional
        P-values with shape (time, level, latitude). Where p > 0.05, regions will be hatched.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import pandas as pd
    
    vmax = np.nanpercentile(np.abs(anomalies.values), vpercentile)
    # vmax = 1e23
    # vmax = 1e23
    vmin = -vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == 0:
        print(f"Warning: Invalid color limits (vmin={vmin}, vmax={vmax}), using fallback values")
        vmax = 2.5e23
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
    fig = plt.figure(figsize=(18, 4*n_rows))
    gs = GridSpec(n_rows * 2, n_cols, figure=fig, height_ratios=[6, 1] * n_rows, hspace=0.3, wspace=0.1)
    
    # Contour levels
    levels = np.linspace(vmin, vmax, 13)
    
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
    
    all_vertical_integral_maxima = []
    
    for i, t_idx in enumerate(snapshot_indices):
        if i >= len(contour_axes):
            break
        
        row = i // n_cols
        col = i % n_cols
        
        if col == 0:
            contour_axes[i].set_ylabel(vertical_label, fontsize=10)
        else:
            contour_axes[i].set_ylabel("")
            contour_axes[i].tick_params(labelleft=False)  # also hide tick labels if you want
                    
        _raw_t = anomalies_for_plot.time.values[t_idx]
        _is_composite_month = isinstance(_raw_t, (int, np.integer)) or (
            isinstance(_raw_t, float) and _raw_t == int(_raw_t) and 1 <= int(_raw_t) <= 36
        )
        if _is_composite_month:
            if rolling_period is not None and rolling_period > 1:
                _panel_label = f"Composite-Rolling-month-{int(_raw_t):02d}"
            else:
                _panel_label = f"Composite-month-{int(_raw_t):02d}"
        else:
            _panel_label = pd.to_datetime(_raw_t).strftime("%Y-%m")
        
        data_slice = anomalies_for_plot.isel(time=t_idx).transpose(level_dim, lat_dim)
        data_values = data_slice.values
        
        # Use explicit levels array to ensure consistent colorbar
        levels = np.linspace(vmin, vmax, 13)
        im = contour_axes[i].contourf(lat_vals, pressure_hpa, data_values,
                             levels=levels, cmap='RdBu_r', extend='both')
        
        hatch_mask = None
        
        # Case A: We have a precomputed boolean mask (True = significant)
        if significance_mask is not None and significance_mask.shape[0] > t_idx:
            sig_slice = significance_mask[t_idx, :, :]
            # Check shape/transpose logic
            if sig_slice.shape != data_values.shape:
                if sig_slice.T.shape == data_values.shape:
                    sig_slice = sig_slice.T
            
            # IPCC: Hatch where NOT significant. 
            # Since sig_mask is True for significant, we hatch where it is False.
            hatch_mask = np.where(sig_slice == False, 1.0, 0.0)

        # Case B: We have raw p-values (Float)
        elif p_values is not None and p_values.shape[0] > t_idx:
            p_slice = p_values[t_idx, :, :]
            if p_slice.shape != data_values.shape:
                if p_slice.T.shape == data_values.shape:
                    p_slice = p_slice.T
            
            # Hatch where p > 0.05 or NaN (insignificant)
            hatch_mask = np.where((p_slice > 0.05) | np.isnan(p_slice), 1.0, 0.0)
        
        if hatch_mask is not None:
            # Create a masked version of your lat/lon grid for the pcolor.
            # We only want to 'plot' the grid cells where significance is False (hatch = 1).
            hatch_data = np.ma.masked_where(hatch_mask != 1, hatch_mask)
            if hatch_data.shape != data_values.shape and hatch_data.T.shape == data_values.shape:
                hatch_data = hatch_data.T

            # Use pcolor which allows more granular control over edges.
            h_plot = contour_axes[i].pcolor(
                lat_vals,
                pressure_hpa,
                hatch_data,
                hatch='/',
                alpha=0,
                zorder=10,
            )

            for patch in h_plot.get_children():
                # In pcolor, setting edgecolor to a color BUT linewidth to a value
                # allows the hatches to show up while the cell borders remain faint/invisible.
                h_plot.set_edgecolor((0.5, 0.5, 0.5, 0.7))
                h_plot.set_linewidth(0.5)
        
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
                wind_peak = float(np.nanmax(np.abs(wind_values)))
                if np.isfinite(wind_peak) and wind_peak > 0:
                    wind_step = 10.0 if wind_peak >= 40.0 else 5.0
                    wind_max = max(wind_step, wind_step * np.ceil(wind_peak / wind_step))
                    wind_contour_levels = np.arange(-wind_max, wind_max + wind_step, wind_step)
                    wind_contour_levels = wind_contour_levels[np.abs(wind_contour_levels) >= wind_step]
                else:
                    wind_contour_levels = np.array([])
                if wind_contour_levels.size == 0:
                    raise ValueError("wind field has no finite amplitude to contour")
                cs = contour_axes[i].contour(wind_lat, pressure_hpa, wind_values,
                                    levels=wind_contour_levels, colors='black', 
                                    linewidths=0.8, alpha=0.6)
                contour_axes[i].clabel(cs, inline=True, fontsize=7, fmt='%d')
            except Exception as e:
                print(f"Warning: Could not overlay wind data for snapshot {i}: {e}")
        
        # Set axis label only for the left column
        if i % n_cols == 0:
            contour_axes[i].set_ylabel(vertical_label, fontsize=10)
        else:
            contour_axes[i].set_ylabel("")
            contour_axes[i].tick_params(labelleft=False)
            
        contour_axes[i].set_xlabel('Latitude (°N)', fontsize=10)
        contour_axes[i].set_xlim(-60, 60)
        contour_axes[i].set_title(_panel_label, fontsize=10, pad=3)
        # Pressure axis: decreasing upward, with sensible pressure ticks
        if vertical_label.lower().startswith("pressure") and np.all(np.isfinite(pressure_hpa)) and np.all(pressure_hpa > 0):
            pmin = float(np.nanmin(pressure_hpa))
            pmax = float(np.nanmax(pressure_hpa))
            if (pmax / max(pmin, 1e-6)) > 3.0:
                common_ticks = np.array([1000, 850, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10], dtype=float)
                ticks = common_ticks[(common_ticks >= pmin) & (common_ticks <= pmax)]
                if ticks.size >= 3:
                    contour_axes[i].set_yticks(ticks)
                    contour_axes[i].set_yticklabels([f"{int(t)}" for t in ticks])
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
            contour_axes[i].axvline(extreme_lat, color='C2', linewidth=2, linestyle='-', alpha=0.8, zorder=10)
        
        # Add vertical line at equator
        contour_axes[i].axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.7)
        
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
        
        # choose vmax from actual data, not the mask
        finite_vi = vertical_integral[np.isfinite(vertical_integral)]
        if finite_vi.size > 0:
            snapshot_max = np.nanmax(np.abs(finite_vi))
            all_vertical_integral_maxima.append(snapshot_max)
        else:
            snapshot_max = 5e27
            all_vertical_integral_maxima.append(snapshot_max)# or 0, depending on how you want to handle empty

        if np.count_nonzero(vi_finite) >= 2 and np.count_nonzero(~vi_finite) > 0:
            vi_plot[~vi_finite] = np.interp(lat_vals[~vi_finite], lat_vals[vi_finite], vi_plot[vi_finite])
        profile_axes[i].plot(lat_vals, vi_plot, color="C0", lw=2)
        if i % n_cols == 0:
            profile_axes[i].set_ylabel('Total', fontsize=9)
        else:
            contour_axes[i].set_ylabel("")
            contour_axes[i].tick_params(labelleft=False)
        profile_axes[i].set_xlim(-60, 60)
        profile_axes[i].grid(True, alpha=0.3)
    
    # Compute global vmax from all snapshots
    finite_maxima = np.array([x for x in all_vertical_integral_maxima if np.isfinite(x)])
    if finite_maxima.size > 0:
        vmax_global = np.nanmax(finite_maxima)
    else:
        vmax_global = 5e27
    
    # Apply same limits to all profile axes
    for ax in profile_axes[:len(snapshot_indices)]:
        ax.set_ylim(-vmax_global, vmax_global)
        
    # Hide unused subplots
    for j in range(len(snapshot_indices), len(contour_axes)):
        contour_axes[j].axis('off')
        profile_axes[j].axis('off')
    
    # Add a single colorbar at the bottom for all subplots
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.01])  # [left, bottom, width, height]
    
    variable = anomalies.attrs.get("long_name", anomalies.name if anomalies.name is not None else "Variable")
    
    # Create discrete levels matching the contour levels
    levels = np.linspace(vmin, vmax, 13)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    sm = cm.ScalarMappable(cmap=cm.get_cmap('RdBu_r'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both', spacing='proportional')
    cbar.set_label(f'{variable} (kg m² s⁻¹)', fontsize=15)
    # Set ticks at every other level boundary for clarity
    tick_indices = np.arange(0, 13, 2)
    cbar.set_ticks(list(levels[tick_indices]))
    
    _suffix_str = f"  |  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f'CMIP6 HadGEM3_GC31 {ensemble_member} zonally integrated AAM Anomaly\n'
        f'Latitude × Level Snapshots Climatology: {clim_start_yr}-{clim_end_yr}\n'
        f'{_suffix_str}',
        fontsize=20, y=0.95,
    )
    plt.tight_layout(rect=[0.00, 0.1, 0.98, 0.96])  # Leave space for colorbar at bottom and reduce top gap

    rolling_tag = ""
    if rolling_period is not None:
        rp = int(rolling_period)
        if rp > 1:
            rolling_tag = f"_rolling{rp}"

    file_suffix = ""
    if filename_suffix:
        file_suffix = f"_{str(filename_suffix).strip('_')}"
        
    if dec_onset_month == "onset":
        pass
    elif dec_onset_month == "december_onset_year":
        file_suffix += "_december_onset_year"
    
    if onset_season_ndjfm == "ndjfm":
        file_suffix += "_onset_season_NDJFM"
    else:
        file_suffix += "_onset_season_all"
        
    nino_thres_tag = "nino_thres" + str(float(nino_threshold)) if nino_threshold is not None else ""

    output_file = (
        f'{output_dir}AAM_anomalies_lat_level_snapshots_{ensemble_member}_{start_year}-{end_year}{rolling_tag}_{nino_thres_tag}{file_suffix}.svg'
    )
    plt.savefig(output_file, format="svg", bbox_inches='tight')
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
    vpercentile: float = 99.0,
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
        vmax = 1e23
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
                ax_cont.axvline(extreme_lat, color="C2", linewidth=2, linestyle="-", alpha=0.8, zorder=10)

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


def plot_lat_level_postage_stamp_5x12(
    member_composites: list,
    *,
    output_dir: str | Path = "output/",
    start_year: int = 1850,
    end_year: int = 2010,
    clim_start_yr: int = 1980,
    clim_end_yr: int = 2000,
    region: str = "all",
    enso_state: str = "el_nino",
    nino_threshold: float = 0.5,
    title_suffix: str = "",
) -> None:
    """Create a postage stamp grid of lat×level composites for all ensemble members.
    
    5 columns × 12 rows = 60 subplots (one per ensemble member).
    
    Parameters
    ----------
    member_composites : list of tuples (member_name, composite_da)
        Each composite_da has dims (lat, level) or (level, lat)
    output_dir : str | Path
        Directory to save the figure
    region : str
        Region label for filename
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_members = len(member_composites)
    n_cols = 5
    n_rows = int(np.ceil(n_members / n_cols))
    
    # Collect all composites (time-averaged) to determine global color limits
    all_values = []
    for _, comp in member_composites:
        # Average over time dimension if present
        if "time" in comp.dims:
            comp_avg = comp.mean("time")
        elif "month" in comp.dims:
            comp_avg = comp.mean("month")
        else:
            comp_avg = comp
        all_values.append(comp_avg.values)
    all_values = np.concatenate([v.flatten() for v in all_values])
    all_values = all_values[np.isfinite(all_values)]
    
    if all_values.size > 0:
        vmax = float(np.nanpercentile(np.abs(all_values), 99))
        vmax = vmax if vmax > 0 else 1.0
    else:
        vmax = 1.0
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 21)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Normalize dimension names
    for idx, (member_name, comp) in enumerate(member_composites):
        # Average over time dimension if present to get single representative image
        if "time" in comp.dims:
            comp = comp.mean("time")
        elif "month" in comp.dims:
            comp = comp.mean("month")
        
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Ensure standard dimension names
        if "latitude" in comp.dims and "lat" not in comp.dims:
            comp = comp.rename({"latitude": "lat"})
        if "level" in comp.dims or "plev" in comp.dims:
            lev_dim = "level" if "level" in comp.dims else "plev"
        else:
            lev_dim = None
        
        lat_dim = "lat" if "lat" in comp.dims else ("latitude" if "latitude" in comp.dims else None)
        
        if lat_dim is None or lev_dim is None:
            ax.text(0.5, 0.5, f"Missing dims\n{comp.dims}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(member_name, fontsize=8)
            continue
        
        lat_vals = comp[lat_dim].values
        lev_vals = comp[lev_dim].values
        
        # Ensure data has shape (lat, level) for contourf with swapped axes
        data_vals = comp.transpose(lat_dim, lev_dim).values
        
        cf = ax.contourf(lev_vals, lat_vals, data_vals, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_ylabel("Latitude (°N)", fontsize=8)
        ax.set_xlabel("Level (hPa)", fontsize=8)
        ax.set_title(member_name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.invert_yaxis()
    
    # Hide empty subplots
    for idx in range(n_members, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
    
    # Add single colorbar
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.01])
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    sm = cm.ScalarMappable(cmap=cm.get_cmap('RdBu_r'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label('AAM anomaly (kg m² s⁻¹)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    region_label = region.upper() if region != 'all' else 'GLOBAL'
    fig.suptitle(
        f'HadGEM3_GC31 {n_members} Ensemble Members - Latitude×Level Composites\n'
        f'{region_label} | {enso_state} (Nino3.4{">=" if enso_state=="el_nino" else "<="}{nino_threshold})\n'
        f'{title_suffix}',
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    out_path = output_dir / f"postage_stamp_lat_level_ensemble_{enso_state}_{region_label.lower()}_{n_members}members.svg"
    fig.savefig(out_path, format='svg', bbox_inches="tight")
    plt.close(fig)
    print(f"Postage stamp (lat×level) saved to {out_path}")


def plot_lat_level_monthly_stamps_10col(
    member_composites: list,
    *,
    output_dir: str | Path = "output/",
    region: str = "all",
    enso_state: str = "el_nino",
    nino_threshold: float = 0.5,
    title_suffix: str = "",
) -> None:
    """Create monthly postage stamp figures for lat×level composites (10 columns per figure).
    
    For each month in the composite, creates a separate figure with 10 columns × 6 rows = 60 members.
    
    Parameters
    ----------
    member_composites : list of tuples (member_name, composite_da)
        Each composite_da has dims (lat, level, month) or permutation thereof
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_members = len(member_composites)
    if n_members == 0:
        print("No member composites to plot")
        return
    
    # Get number of months and collect all values for color scaling
    first_comp = member_composites[0][1]
    
    # Find time/month dimension
    time_dim = None
    for dim in first_comp.dims:
        if dim in ("time", "month"):
            time_dim = dim
            break
    
    if time_dim is None:
        print("No time/month dimension found in composites")
        return
    
    n_months = first_comp.sizes[time_dim]
    
    # Collect all values for global color scaling
    all_values = []
    for _, comp in member_composites:
        all_values.append(comp.values)
    all_values = np.concatenate([v.flatten() for v in all_values])
    all_values = all_values[np.isfinite(all_values)]
    
    if all_values.size > 0:
        vmax = float(np.nanpercentile(np.abs(all_values), 99))
        vmax = vmax if vmax > 0 else 1.0
    else:
        vmax = 1.0
    vmax = 5e23
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 21)
    
    # Process each month
    for month_idx in range(n_months):
        n_cols = 10
        n_rows = int(np.ceil(n_members / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for member_idx, (member_name, comp) in enumerate(member_composites):
            # Normalize dimension names
            if "latitude" in comp.dims and "lat" not in comp.dims:
                comp = comp.rename({"latitude": "lat"})
            if "level" in comp.dims or "plev" in comp.dims:
                lev_dim = "level" if "level" in comp.dims else "plev"
            else:
                lev_dim = None
            
            lat_dim = "lat" if "lat" in comp.dims else ("latitude" if "latitude" in comp.dims else None)
            
            row = member_idx // n_cols
            col = member_idx % n_cols
            ax = axes[row, col]
            
            if lat_dim is None or lev_dim is None:
                ax.text(0.5, 0.5, f"Missing dims", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(member_name, fontsize=7)
                continue
            
            # Ensure time_dim is present in comp
            if time_dim not in comp.dims:
                ax.text(0.5, 0.5, f"No {time_dim} dim\n({comp.dims})", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(member_name, fontsize=7)
                continue
            
            # Extract this month's data
            comp_month = comp.isel({time_dim: month_idx})
            
            lat_vals = comp_month[lat_dim].values
            lev_vals = comp_month[lev_dim].values
            
            # Ensure data has shape (level, lat) for contourf
            data_vals = comp_month.transpose(lev_dim, lat_dim).values
            
            cf = ax.contourf(lat_vals, lev_vals, data_vals, levels=levels, cmap="RdBu_r", extend="both")
            ax.set_ylabel("Level", fontsize=6)
            ax.set_xlabel("Lat", fontsize=6)
            ax.set_title(member_name, fontsize=7)
            ax.tick_params(labelsize=5)
            ax.invert_yaxis()
        
        # Hide empty subplots
        for idx in range(n_members, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")
        
        # Colorbar
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.01])
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
        sm = cm.ScalarMappable(cmap=cm.get_cmap('RdBu_r'), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label('AAM anomaly (kg m² s⁻¹)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        region_label = region.upper() if region != 'all' else 'GLOBAL'
        fig.suptitle(
            f'HadGEM3_GC31 Lat×Level - Month {month_idx+1:02d} | {region_label} {enso_state}\n{title_suffix}',
            fontsize=11, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        out_path = output_dir / f"postage_stamp_lat_level_month{month_idx+1:02d}_{region_label.lower()}_{enso_state}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Month {month_idx+1:02d} lat×level stamps saved to {out_path}")


def plot_lat_time_postage_stamp_all_members(
    member_composites: list,
    *,
    output_dir: str | Path = "output/",
    region: str = "all",
    enso_state: str = "el_nino",
    nino_threshold: float = 0.5,
    title_suffix: str = "",
) -> None:
    """Create a postage stamp grid of lat×time composites for all ensemble members.
    
    Parameters
    ----------
    member_composites : list of tuples (member_name, composite_da)
        Each composite_da has dims (lat, time/month) or (time/month, lat)
    output_dir : str | Path
        Directory to save the figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.ticker as mticker
    import numpy as np
    from pathlib import Path
    import math
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_members = len(member_composites)
    if n_members == 0:
        return
        
    n_cols = 5
    n_rows = int(np.ceil(n_members / n_cols))
    
    # Collect all values to determine global color limits
    all_values = []
    for _, comp in member_composites:
        all_values.append(comp.values.flatten())
    all_values = np.concatenate(all_values)
    all_values = all_values[np.isfinite(all_values)]
    
    if all_values.size > 0:
        vmax = float(np.nanpercentile(np.abs(all_values), 99))
        vmax = vmax if vmax > 0 else 1.0
    else:
        vmax = 1.0
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 13)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.15)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate subplots
    for idx, (member_name, comp) in enumerate(member_composites):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Normalize dimension names
        if "latitude" in comp.dims and "lat" not in comp.dims:
            comp = comp.rename({"latitude": "lat"})
        
        time_dim = next((d for d in comp.dims if d in ("time", "month")), None)
        lat_dim = "lat" if "lat" in comp.dims else ("latitude" if "latitude" in comp.dims else None)
        
        if time_dim is None or lat_dim is None:
            ax.text(0.5, 0.5, f"Missing dims", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(member_name, fontsize=8)
            continue
            
        lat_vals = comp[lat_dim].values
        time_vals = comp[time_dim].values
        n_times = len(time_vals)
        
        # Ensure (lat, time) ordering
        data_vals = comp.values
        if comp.dims[0] == time_dim:
            data_vals = data_vals.T
            
        cf = ax.contourf(time_vals, lat_vals, data_vals, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_ylabel("Latitude (°N)", fontsize=8)
        
        if row == n_rows - 1 or idx >= n_members - n_cols:
            ax.set_xlabel("Month since onset", fontsize=8)
            
        ax.set_xlim(1, len(time_vals))
        ax.set_ylim(-60, 60)
        
        # Consistent ticks and grid
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1)) # Every 1 month like normal plot
        ax.yaxis.set_major_locator(mticker.MultipleLocator(30)) # -60, -30, 0, 30, 60
        
        # Grid and equator
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        
        ax.set_title(member_name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for idx in range(n_members, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
        
    # Add identical style single colorbar
    _abs = max(abs(vmin), abs(vmax))
    order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
    factor = 10 ** order
    
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.01])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', extend='both')
    
    _sup = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
    _order_sup = str(order).translate(_sup)
    
    cbar.set_label(f"AAM anomaly (×10{_order_sup})", fontsize=12)
    
    _tick_levels = cf.levels[::2]
    cbar.set_ticks(_tick_levels)
    cbar.set_ticklabels([f"{v / factor:.1f}" for v in _tick_levels])
    cbar.ax.tick_params(labelsize=10)
    
    region_label = region.upper() if region != 'all' else 'GLOBAL'
    fig.suptitle(
        f'HadGEM3_GC31 {n_members} Ensemble Members - Lat×Time Evolution Composites\n'
        f'{region_label} | {enso_state} (Nino3.4{">=" if enso_state=="el_nino" else "<="}{nino_threshold})\n'
        f'{title_suffix}',
        fontsize=14, fontweight="bold", y=0.98
    )
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    out_path = output_dir / f"postage_stamp_lat_time_ensemble_{enso_state}_{region_label.lower()}_{n_members}members.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Lat×time evolution postage stamp saved to {out_path}")


def plot_lat_lon_monthly_stamps_10col(
    member_composites: list,
    *,
    output_dir: str | Path = "output/",
    region: str = "all",
    enso_state: str = "el_nino",
    nino_threshold: float = 0.5,
    title_suffix: str = "",
) -> None:
    """Create monthly postage stamp figures for lat×lon composites (10 columns per figure).
    
    For each month in the composite, creates a separate figure with 10 columns × 6 rows = 60 members.
    
    Parameters
    ----------
    member_composites : list of tuples (member_name, composite_da)
        Each composite_da has dims (lat, lon, month) or permutation thereof
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_members = len(member_composites)
    if n_members == 0:
        print("No member composites to plot")
        return
    
    # Get number of months and collect all values for color scaling
    first_comp = member_composites[0][1]
    
    # Find time/month dimension
    time_dim = None
    for dim in first_comp.dims:
        if dim in ("time", "month"):
            time_dim = dim
            break
    
    if time_dim is None:
        print("No time/month dimension found in composites")
        return
    
    n_months = first_comp.sizes[time_dim]
    
    # Collect all values for global color scaling (average over time first)
    all_values = []
    for _, comp in member_composites:
        if "time" in comp.dims:
            comp_avg = comp.mean("time")
        elif "month" in comp.dims:
            comp_avg = comp.mean("month")
        else:
            comp_avg = comp
        all_values.append(comp_avg.values)
    all_values = np.concatenate([v.flatten() for v in all_values])
    all_values = all_values[np.isfinite(all_values)]
    
    if all_values.size > 0:
        vmax = float(np.nanpercentile(np.abs(all_values), 99))
        vmax = vmax if vmax > 0 else 1.0
    else:
        vmax = 1.0
    vmax = 1e25
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 21)
    
    # Use PlateCarree projection (region-aware): center at dateline for Pacific and global
    region_central_lons = {
        "pacific": 180,
        "all": 180,
        "indian": 75,      # Center Indian Ocean (50-100°E)
        "atlantic": -30,   # Center Atlantic (-60-10°E)
    }
    central_lon = region_central_lons.get(region, 0)
    map_crs = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()
    
    # Process each month
    for month_idx in range(n_months):
        n_cols = 10
        n_rows = int(np.ceil(n_members / n_cols))
        
        fig = plt.figure(figsize=(24, 2.5*n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
        for member_idx, (member_name, comp) in enumerate(member_composites):
            # Normalize dimension names
            if "latitude" in comp.dims and "lat" not in comp.dims:
                comp = comp.rename({"latitude": "lat"})
            if "longitude" in comp.dims and "lon" not in comp.dims:
                comp = comp.rename({"longitude": "lon"})
            
            lat_dim = "lat" if "lat" in comp.dims else ("latitude" if "latitude" in comp.dims else None)
            lon_dim = "lon" if "lon" in comp.dims else ("longitude" if "longitude" in comp.dims else None)
            
            row = member_idx // n_cols
            col = member_idx % n_cols
            ax = fig.add_subplot(gs[row, col], projection=map_crs)
            
            if lat_dim is None or lon_dim is None:
                ax.text(0.5, 0.5, f"Missing dims", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(member_name, fontsize=7)
                continue
            
            # Ensure time_dim is present in comp
            if time_dim not in comp.dims:
                ax.text(0.5, 0.5, f"No {time_dim} dim\n({comp.dims})", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(member_name, fontsize=7)
                continue
            
            # Extract this month's data
            comp_month = comp.isel({time_dim: month_idx})
            
            lat_vals = comp_month[lat_dim].values
            lon_vals = comp_month[lon_dim].values
            
            data_vals = comp_month.values
            # Ensure (lat, lon)
            if comp_month.dims[0] == lon_dim:
                data_vals = data_vals.T
            
            cf = ax.contourf(lon_vals, lat_vals, data_vals, levels=levels, cmap="RdBu_r", 
                            extend="both", transform=data_crs)
            ax.coastlines(resolution="110m", linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.2)
            
            # Set geographic extent based on region
            lat_extent = (-60, 60)
            if region == "indian":
                ax.set_extent([50, 100, lat_extent[0], lat_extent[1]], crs=data_crs)
            elif region == "atlantic":
                ax.set_extent([-60, 10, lat_extent[0], lat_extent[1]], crs=data_crs)
            elif region == "pacific":
                ax.set_extent([125, 250, lat_extent[0], lat_extent[1]], crs=data_crs)
            else:
                lon_min, lon_max = float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))
                ax.set_extent([lon_min, lon_max, lat_extent[0], lat_extent[1]], crs=data_crs)
            
            ax.set_title(member_name, fontsize=7)
            ax.tick_params(labelsize=5)
        
        # Hide empty subplots
        for idx in range(n_members, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")
        
        # Colorbar
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.01])
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
        sm = cm.ScalarMappable(cmap=cm.get_cmap('RdBu_r'), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.set_label('AAM anomaly (kg m² s⁻¹)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        region_label = region.upper() if region != 'all' else 'GLOBAL'
        fig.suptitle(
            f'HadGEM3_GC31 Lat×Lon - Month {month_idx+1:02d} | {region_label} {enso_state}\n{title_suffix}',
            fontsize=11, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        out_path = output_dir / f"postage_stamp_lat_lon_month{month_idx+1:02d}_{region_label.lower()}_{enso_state}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Month {month_idx+1:02d} lat×lon stamps saved to {out_path}")


__all__ = [
    "ClimatologyCacheSpec",
    "compute_monthly_climatology",
    "ensure_dir",
    "load_or_compute_monthly_climatology_from_file",
    "plot_anomalies_3d_slices",
    "plot_latitude_level_movie_HadGEM3",
    "plot_latitude_level_snapshots_HadGEN3",
    "plot_lat_level_postage_stamp_5x12",
    "plot_lat_level_monthly_stamps_10col",
    "plot_lat_lon_monthly_stamps_10col",
]
