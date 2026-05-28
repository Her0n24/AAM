# %%
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap
import os
import glob
import pandas as pd
import argparse
from typing import Optional
import sys


def _latitude_band_width_radians(lat_deg: np.ndarray) -> np.ndarray:
    """Return dphi (radians) for latitude bands centered on the latitude points."""
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=float))
    n = lat_rad.size
    edges = np.empty(n + 1, dtype=float)
    edges[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    edges[0] = -0.5 * np.pi
    edges[-1] = 0.5 * np.pi
    return np.abs(np.diff(edges))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=150, help='Minimum pressure level (hPa) to include (default: 100 hPa)')
parser.add_argument('--p-max', type=float, default=1000, help='Maximum pressure level (hPa) to include (default: 1000 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year for analysis (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year for analysis (default: 2000)')
parser.add_argument('--input-nc', type=str, default=None, help='Precomputed AAM anomaly NetCDF to replot (default: derived from start/end/p-range)')
parser.add_argument('--enso-csv', type=str, default=None, help='Niño3.4 CSV for the lower panel (default: nino34/nino34_HadlSST.csv)')
if "ipykernel" in sys.modules:
    args, _ = parser.parse_known_args([
        '--start-year', '1979',
        '--end-year', '2000',
    ])
else:
    args, _ = parser.parse_known_args()

scratch_path = "/work/scratch-nopw2/hhhn2"
base_dir = os.getcwd()

clim_start = 1981
clim_end = 2010
AAM_data_path = f"{scratch_path}/ERA5/monthly_mean/AAM/full/"
climatology_path = f"{scratch_path}/ERA5/climatology/"
output_dir = f"{base_dir}/AAMA_fig/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load sigma level coefficients for pressure calculation
sigma_coeff = pd.read_csv(f'{base_dir}/l137_a_b.csv')
a_coeff = sigma_coeff['a [Pa]'].values[1:]  # omit level 0
b_coeff = sigma_coeff['b'].values[1:]

# Calculate mid-level coefficients (for the AAM data which is on mid-levels)
a_mid = 0.5 * (a_coeff[:-1] + a_coeff[1:])
b_mid = 0.5 * (b_coeff[:-1] + b_coeff[1:])

start_yr = args.start_year
end_yr = args.end_year
p_min = args.p_min * 100  # Convert hPa to Pa
p_max = args.p_max * 100
analysis_start_yr = start_yr
analysis_end_yr = end_yr
DEFAULT_ENSO_CSV = os.path.join(base_dir, 'nino34', 'nino34_HadlSST.csv')

# Resolve requested variable name to an actual data variable in the dataset.
def _resolve_var(ds, var):
    # Common candidate names
    alt_map = {
        'u': ['u', 'eastward_wind', 'u_component_of_wind'],
        'v': ['v', 'northward_wind', 'v_component_of_wind'],
        'sp': ['sp', 'surface_pressure', 'surface_air_pressure'],
        'ps': ['sp', 'surface_pressure', 'surface_air_pressure'],
        'AAM': ['AAM', 'atmospheric_angular_momentum', 'aam'],
        'u_momentum': ['u_momentum', 'u_aam', 'zonal_momentum'],
        'v_momentum': ['v_momentum', 'v_aam', 'meridional_momentum'],
    }
    candidates = []
    if var in alt_map:
        candidates.extend(alt_map[var])
    candidates.append(var)
    
    # Try candidates in order
    for c in candidates:
        if c in ds.data_vars:
            return ds[c]
    
    # Fallback: if only one data var present, return it
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars.keys())[0]]
    
    raise KeyError(f"Variable '{var}' not found in dataset; tried: {candidates}")


def _load_precomputed_aam_anomaly(nc_path: str, component: str = 'AAM') -> xr.DataArray:
    """Load a precomputed AAM anomaly NetCDF file produced by this workspace."""
    ds = xr.open_dataset(nc_path)
    try:
        candidates = (
            f'{component}_anomaly',
            component,
            'AAM_anomaly',
            'AAMA',
        )
        for name in candidates:
            if name in ds.data_vars:
                return ds[name].load()
        if len(ds.data_vars) == 1:
            return next(iter(ds.data_vars.values())).load()
        raise KeyError(f"Could not identify anomaly variable in {nc_path}; available={list(ds.data_vars)}")
    finally:
        ds.close()


def _load_nino34_csv(csv_path: str, start_year: int, end_year: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Load monthly Niño3.4 values from the provided CSV file."""
    df = pd.read_csv(csv_path, skiprows=1, names=['Date', 'NINO34'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['NINO34'] = pd.to_numeric(df['NINO34'], errors='coerce')
    df = df.dropna(subset=['Date', 'NINO34'])
    df = df[(df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)]
    if df.empty:
        return pd.DatetimeIndex([]), np.array([], dtype=float)
    return pd.DatetimeIndex(df['Date']), df['NINO34'].to_numpy(dtype=float)


def _find_precomputed_aam_anomaly(output_dir: str, start_year: int, end_year: int, p_min_hpa: float, p_max_hpa: float) -> Optional[str]:
    """Return a saved anomaly NetCDF that covers the requested years, if one exists."""
    pattern = os.path.join(output_dir, f"AAM_anomalies_*_p{int(p_min_hpa)}-{int(p_max_hpa)}hPa.nc")
    candidates = sorted(glob.glob(pattern))
    matching = []

    for nc_path in candidates:
        try:
            with xr.open_dataset(nc_path) as ds:
                if 'time' not in ds.coords and 'time' not in ds.dims:
                    continue
                time_index = pd.DatetimeIndex(pd.to_datetime(ds['time'].values))
                if time_index.empty:
                    continue
                file_start = int(time_index.min().year)
                file_end = int(time_index.max().year)
        except Exception:
            continue

        if file_start <= start_year and file_end >= end_year:
            matching.append((file_start, file_end, nc_path))

    if matching:
        matching.sort(key=lambda item: (item[1] - item[0], item[0]))
        return matching[0][2]

    return None


def plot_AAM_anomalies(start_year, end_year, component='AAM', *, nlevels=11, cmap_name='RdBu_r', 
                       vmin=None, vmax=None, savefile=None, show=True, input_nc_path=None, enso_csv_path=None):
    """Plot AAM anomalies (variation from climatological mean) for a given period.

    Loads files matching `AAM_ERA5_{year}*.nc` from `AAM_data/monthly_mean/` for all years
    in the range [start_year, end_year], subtracts the precomputed monthly climatology,
    and plots the anomalies as latitude vs time using a discrete colormap.

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
    
    precomputed_nc = input_nc_path
    if precomputed_nc is None:
        precomputed_nc = _find_precomputed_aam_anomaly(output_dir, start_year, end_year, args.p_min, args.p_max)
        if precomputed_nc is None:
            precomputed_nc = os.path.join(output_dir, f"AAM_anomalies_{start_year}-{end_year}_p{int(p_min/100)}-{int(p_max/100)}hPa.nc")

    if enso_csv_path is None:
        enso_csv_path = DEFAULT_ENSO_CSV

    if os.path.exists(precomputed_nc):
        print(f"Loading precomputed AAM anomalies from {precomputed_nc}")
        anomalies = _load_precomputed_aam_anomaly(precomputed_nc, component=component)

        # Normalize common dim names for plotting.
        rename = {}
        if 'lat' in anomalies.dims and 'latitude' not in anomalies.dims:
            rename['lat'] = 'latitude'
        if 'lon' in anomalies.dims and 'longitude' not in anomalies.dims:
            rename['lon'] = 'longitude'
        if rename:
            anomalies = anomalies.rename(rename)

        if 'time' not in anomalies.dims or 'latitude' not in anomalies.dims:
            raise ValueError(f'Precomputed anomaly file must have dims (time, latitude); got {anomalies.dims}')

        anomaly_attrs = anomalies.attrs
        times = pd.DatetimeIndex(pd.to_datetime(anomalies['time'].values))
        time_mask = (times.year >= start_year) & (times.year <= end_year)
        if time_mask.any() and not time_mask.all():
            anomalies = anomalies.isel(time=np.where(time_mask)[0])
            times = pd.DatetimeIndex(pd.to_datetime(anomalies['time'].values))
        lats = np.asarray(anomalies['latitude'].values, dtype=float)
        data = np.asarray(anomalies.values, dtype=float)
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            data = data[:, ::-1]

        if data.ndim != 2:
            raise ValueError(f'Expected a 2D anomaly field (time, latitude); got shape {data.shape}')

        if vmin is None:
            vmin = np.nanpercentile(data, 1)
        if vmax is None:
            vmax = np.nanpercentile(data, 99)
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

        print(f"Initial color limits: vmin={vmin:.2e}, vmax={vmax:.2e}")
        print(f"Final symmetric color limits: vmin={vmin:.2e}, vmax={vmax:.2e}")

        levels = np.linspace(vmin, vmax, nlevels)
        colormaps = getattr(mpl, 'colormaps', None)
        base_cmap = colormaps.get_cmap(cmap_name) if colormaps is not None else cm.get_cmap(cmap_name)
        cmap_disc = ListedColormap(base_cmap(np.linspace(0, 1, nlevels - 1)))
        norm = BoundaryNorm(levels, ncolors=cmap_disc.N, clip=True)

        fig = plt.figure(figsize=(16, 6), constrained_layout=False)
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[6.0, 1.0, 0.18], hspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        ax_enso = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[2, 0])

        times_num = np.asarray(mdates.date2num(times.to_pydatetime()), dtype=float)
        im = ax.imshow(
            data.T,
            origin='lower',
            aspect='auto',
            cmap=cmap_disc,
            norm=norm,
            extent=[times_num[0], times_num[-1], lats[0], lats[-1]],
            interpolation='bilinear',
        )

        ax.xaxis_date()
        years = np.unique(times.year)
        n_years = len(years)
        font_size = 14
        chars = 4
        char_width_pt = font_size * 0.6
        label_width_in = (char_width_pt / 72.0) * chars
        axis_width_in = fig.get_size_inches()[0] * ax.get_position().width
        required_width_in = n_years * label_width_in * 1.05
        major_locator = mdates.YearLocator(5) if required_width_in > axis_width_in else mdates.YearLocator(5)
        # ax.xaxis.set_major_locator(major_locator)
        # ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 3,5,7,9,11)))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), ha='center', size=font_size)
        plt.setp(ax.get_yticklabels(), size=14)

        ax.set_ylim(-60,60)
        ax.set_ylabel('Latitude (°)', size=16)
        ax.set_xlabel('', size=16)
        ax.set_title(
            f"ERA5 zonally-integrated {component} fluctuations from Climatology ({start_year}-{end_year})\n"
            f"(Summed: {p_min/100:.0f}-{p_max/100:.0f} hPa)",
            size=18,
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=10)

        cbar = fig.colorbar(
            im,
            cax=cax,
            boundaries=levels,
            extend='both',
            orientation='horizontal',
            spacing='proportional',
        )
        tick_spacing = max(1, len(levels)//8)
        cbar.set_ticks(levels[::tick_spacing].tolist())

        max_abs_value = max(abs(vmin), abs(vmax))
        if max_abs_value >= 1e3 or max_abs_value <= 1e-3:
            order = int(np.floor(np.log10(max_abs_value)))
            factor = 10**order

            def _superscript(exp):
                superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
                return ''.join(superscript_map.get(c, c) for c in str(exp))

            units = anomaly_attrs.get('units', '')
            if units:
                units_formatted = units.replace('**', '^').replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
                units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
                units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
                units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
                label_text = f"{component} Anomalies 10{_superscript(order)} {units_formatted} per latitude band"
            else:
                label_text = f"{component} Anomalies 10{_superscript(order)} (kg m² s⁻¹ per latitude band)"
            cbar.set_ticklabels([f'{val/factor:.1f}' for val in levels[::tick_spacing]])
        else:
            units = anomaly_attrs.get('units', '')
            if units:
                units_formatted = units.replace('**', '^').replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
                units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
                units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
                units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
                label_text = f"{component} Anomalies ({units_formatted} per latitude band)"
            else:
                label_text = f"{component} Anomalies (kg m² s⁻¹ per latitude band)"

        cbar.set_label(label_text, rotation=0, labelpad=10, size=14)
        cbar.ax.tick_params(labelsize=12)

        enso_times, enso_vals = _load_nino34_csv(enso_csv_path, start_year, end_year)
        if len(enso_vals) > 0:
            enso_times_num = np.asarray(mdates.date2num(enso_times.to_pydatetime()), dtype=float)
            enso_vals = np.asarray(enso_vals, dtype=float).reshape(-1)
            ax_enso.plot(enso_times_num, enso_vals, color='black', linewidth=1.5)
            ax_enso.fill_between(enso_times_num, 0.0, enso_vals, where=enso_vals >= 0, color='red', alpha=0.35, interpolate=True)
            ax_enso.fill_between(enso_times_num, 0.0, enso_vals, where=enso_vals < 0, color='blue', alpha=0.35, interpolate=True)
            ax_enso.axhline(0.0, color='black', linewidth=1.0, alpha=0.8)
            ax_enso.set_ylabel('Niño3.4', size=14)
            ax_enso.set_ylim(-2.5, 2.5)
            ax_enso.grid(True, alpha=0.3)
        else:
            ax_enso.text(0.5, 0.5, 'No Niño3.4 data available', ha='center', va='center', transform=ax_enso.transAxes)
            ax_enso.set_ylabel('Niño3.4', size=14)

        plt.setp(ax.get_xticklabels(), visible=False)
        ax_enso.xaxis_date()
        # ax_enso.xaxis.set_major_locator(major_locator)
        # ax_enso.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 3,5,7,9,11)))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax_enso.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_enso.set_xlabel('Year', size=16)
        plt.setp(ax_enso.get_xticklabels(), ha='center', size=font_size)

        fig.align_ylabels([ax, ax_enso])
        fig.subplots_adjust(top=0.92, bottom=0.08)
        cax_pos = cax.get_position()
        cax.set_position([cax_pos.x0, cax_pos.y0 - 0.1, cax_pos.width, cax_pos.height * 0.75])

        if savefile:
            save_path = os.path.join(output_dir, savefile)
            fig.savefig(save_path, dpi=500, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        if show:
            plt.show()

        return fig, ax

    # Load all data needed for either the analysis window or climatology window.
    data_start_year = min(start_year, clim_start)
    data_end_year = max(end_year, clim_end)

    # Load all data for the specified period
    all_files = []
    for year in range(data_start_year, data_end_year + 1):
        pattern = f"{AAM_data_path}AAM_ERA5_{year}*.nc" # Excludes files starting with 'f' after year
        year_files = [f for f in glob.glob(pattern) if '_full_level' not in f]
        if not year_files:
            print(f"Warning: No files found for year {year}")
            continue
        all_files.extend(year_files)
    
    if not all_files:
        raise FileNotFoundError(f"No files found for period {start_year}-{end_year}")
    
    # Load all data with chunking for better memory management
    print(f"Loading {len(all_files)} AAM files...")
    ds = xr.open_mfdataset(all_files, combine='by_coords', chunks={'time': 12, 'latitude': 180, 'longitude': 360})
    da = _resolve_var(ds, component)
    
    # print(f"AAM data shape: {da.shape}, dims: {da.dims}")
    # print(f"AAM data range: min={np.nanmin(da.values):.2e}, max={np.nanmax(da.values):.2e}, mean={np.nanmean(da.values):.2e}")
    # print(f"AAM units: {da.attrs.get('units', 'N/A')}")

    selected_levels = None

    # Check if this is 3D/4D data with level dimension
    if 'level' in da.dims:
        # Load ONE surface pressure file to calculate pressure levels
        # Since a/b coefficients are constant, we only need one representative file
        sp_pattern = f"{AAM_data_path.replace('/AAM/', '/variables/')}ERA5_sp_{start_year}-*.nc"
        sp_files = sorted(glob.glob(sp_pattern))[:1]  # Just take first file
        
        if sp_files:
            print(f"Loading surface pressure file: {sp_files[0]}")
            ds_sp = xr.open_dataset(sp_files[0])
            # Determine the surface pressure variable name
            if 'sp' in ds_sp.data_vars:
                ps = ds_sp['sp']
            elif 'surface_pressure' in ds_sp.data_vars:
                ps = ds_sp['surface_pressure']
            elif 'surface_air_pressure' in ds_sp.data_vars:
                ps = ds_sp['surface_air_pressure']
            else:
                raise KeyError("Cannot find surface pressure variable in dataset")
            
            # Calculate pressure at each mid-level: p = a_mid + b_mid * ps
            # ps may have dims (lat, lon) or (time, lat, lon) depending on file format
            # Result should have dims (level, time, lat, lon) or (level, lat, lon)
            
            # Check if level ordering and count matches
            level_coords = da['level'].values
            n_aam_levels = len(level_coords)
            n_coeff_levels = len(a_mid)
            
            print(f"AAM data has {n_aam_levels} levels, coefficients have {n_coeff_levels} mid-levels")
            
            if n_aam_levels != n_coeff_levels:
                raise ValueError(f"Level count mismatch: AAM has {n_aam_levels} levels but coefficients have {n_coeff_levels} mid-levels")
            
            if np.all(np.diff(level_coords) < 0):
                # Levels are decreasing, reverse coefficients
                a_mid_use = a_mid[::-1]
                b_mid_use = b_mid[::-1]
                print("Reversed coefficients to match decreasing level order")
            else:
                a_mid_use = a_mid
                b_mid_use = b_mid
            
            # Get surface pressure values
            ps_expanded = ps.values
            print(f"Surface pressure shape: {ps_expanded.shape}, dimensions: {ps.dims}")
            print(f"Surface pressure range: min={np.min(ps_expanded):.2e}, max={np.max(ps_expanded):.2e}, mean={np.mean(ps_expanded):.2e}")
            print(f"Surface pressure units: {ps.attrs.get('units', 'N/A')}, long_name: {ps.attrs.get('long_name', 'N/A')}")
            
            # Check if values are suspiciously small (indicating they're log values)
            if np.mean(ps_expanded) < 100:
                print("WARNING: Surface pressure values appear to be logarithmic! Mean < 100")
                print("Expected range for surface pressure is ~50,000-110,000 Pa")
                raise ValueError("Surface pressure appears to be in log form. Please check the data preprocessing.")
            
            # Broadcast based on ps dimensions
            if ps_expanded.ndim == 2:
                # ps is (lat, lon) - single time point per file
                # Broadcasting: (level,1,1) + (level,1,1) * (1,lat,lon) = (level,lat,lon)
                pressure = (a_mid_use[:, None, None] + 
                           b_mid_use[:, None, None] * ps_expanded[None, :, :])
                print(f"Pressure array shape: {pressure.shape} (level, lat, lon)")
                # Average pressure across space for each level
                mean_pressure_per_level = np.mean(pressure, axis=(1, 2))
            elif ps_expanded.ndim == 3:
                # ps is (time, lat, lon)
                # Broadcasting: (level,1,1,1) + (level,1,1,1) * (1,time,lat,lon) = (level,time,lat,lon)
                pressure = (a_mid_use[:, None, None, None] + 
                           b_mid_use[:, None, None, None] * ps_expanded[None, :, :, :])
                print(f"Pressure array shape: {pressure.shape} (level, time, lat, lon)")
                # Average pressure across time and space for each level
                mean_pressure_per_level = np.mean(pressure, axis=(1, 2, 3))
            else:
                raise ValueError(f"Unexpected ps dimensions: {ps_expanded.shape}")
            
            # Find levels within the pressure range
            print(f"Pressure per level range: {mean_pressure_per_level.min()/100:.1f} - {mean_pressure_per_level.max()/100:.1f} hPa")
            level_mask = (mean_pressure_per_level >= p_min) & (mean_pressure_per_level <= p_max)
            selected_levels = level_coords[level_mask]
            
            if len(selected_levels) == 0:
                raise ValueError(f"No levels found in pressure range {p_min/100:.0f}-{p_max/100:.0f} hPa")
            
            print(f"Selected {len(selected_levels)} levels (indices: {np.where(level_mask)[0][:5]}...) in pressure range {p_min/100:.0f}-{p_max/100:.0f} hPa")
            print(f"Selected level pressure range: {mean_pressure_per_level[level_mask].min()/100:.1f} - {mean_pressure_per_level[level_mask].max()/100:.1f} hPa")
            
            # Select and integrate over these levels
            print("Integrating over selected pressure levels...")
            da = da.sel(level=selected_levels).sum(dim='level')
            # Use compute() to force evaluation with dask if needed
            if hasattr(da, 'compute'):
                da = da.compute()
            print(f"After level integration: shape={da.shape}")
            
            ds_sp.close()
        else:
            print("Warning: No surface pressure files found. Using all levels.")
            da = da.sum(dim='level')
    
    # Integrate over longitude and apply latitude band width.
    # AAM from compute step includes dp but NOT dλ dφ, so to get totals per latitude band
    # we need ∫ AAM dλ and then multiply by band width dφ.
    print(f"===== LONGITUDE INTEGRATION =====")
    print(f"Before longitude check - shape: {da.shape}, dims: {da.dims}")
    print(f"Has longitude dimension: {'longitude' in da.dims}")
    if 'longitude' in da.dims:
        print("Longitude dimension found - performing zonal integral (radians)...")
        # Convert longitude coordinate to radians so integrate() gives a true ∫ dλ
        lon_rad = np.deg2rad(da['longitude'].astype(float))
        da = da.assign_coords(longitude=lon_rad).sortby('longitude')
        try:
            dlon = np.diff(da['longitude'].values)
            if dlon.size:
                print(f"Mean dλ (rad): {float(np.nanmean(dlon)):.3e} ; 2π={2*np.pi:.3e}")
        except Exception:
            pass
        da = da.integrate('longitude')
        print(f"After .integrate('longitude') - sh: {da.shape}, dims: {da.dims}")
        # Force computation if using dask
        if hasattr(da, 'compute'):
            print("Computing dask array...") 
            da = da.compute()
            print(f"After .compute() - shape: {da.shape}, dims: {da.dims}")
    else:
        print("No longitude dimension found - skipping longitude integration")
    print(f"=================================")

    #Multiply by latitude band width dφ to get kg m^2 s^-1 per latitude band
    if 'latitude' in da.dims:
        da = da.sortby('latitude')
        dphi = _latitude_band_width_radians(da['latitude'].values)
        orig_name = da.name
        dphi_deg = np.rad2deg(dphi)
        print(f"Median Δφ (deg): {float(np.nanmedian(dphi_deg)):.3f} ; min/max: {float(np.nanmin(dphi_deg)):.3f}/{float(np.nanmax(dphi_deg)):.3f}")
        dphi_da = xr.DataArray(dphi, coords={'latitude': da['latitude']}, dims=('latitude',))
        da = da * dphi_da
        da.name = orig_name
        da.attrs['units'] = 'kg m**2 s**-1'
        da.attrs['description'] = 'AAM per latitude band (dp, dλ, dφ applied)'
    
    # ensure dims include time and latitude
    if 'time' not in da.dims or 'latitude' not in da.dims:
        raise ValueError('dataarray must have dims (time, latitude)')
    
    # Split analysis and climatology periods explicitly.
    analysis_period = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    clim_period = da.sel(time=slice(f"{clim_start}-01-01", f"{clim_end}-12-31"))

    if analysis_period.sizes.get('time', 0) == 0:
        raise ValueError(f"No data found in analysis period {start_year}-{end_year}")
    if clim_period.sizes.get('time', 0) == 0:
        raise ValueError(f"No data found in climatology period {clim_start}-{clim_end}")

    # Calculate climatological mean
    print("Computing climatology...")
    climatology = clim_period.groupby('time.month').mean('time')
    if hasattr(climatology, 'compute'):
        climatology = climatology.compute()
    print(f"Climatology computed: shape={climatology.shape}")
    
    # Calculate anomalies by subtracting climatology from each month
    print("Computing anomalies...")
    anomalies = analysis_period.groupby('time.month') - climatology
    if hasattr(anomalies, 'compute'):
        anomalies = anomalies.compute()
    print(f"Anomalies computed: shape={anomalies.shape}")
    
    times = anomalies['time'].to_index()
    lats = anomalies['latitude'].values
    data = anomalies.values  # shape (time, lat)
    
    # Check if latitudes are decreasing (90 to -90) and flip if needed
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[:, ::-1]

    nc_save_path = os.path.join(output_dir, savefile.replace('.png', '.nc'))
    # Convert DataArray to a Dataset and give it a clear variable name
    anomalies_ds = anomalies.to_dataset(name=f"{component}_anomaly")
    anomalies_ds.to_netcdf(nc_save_path)
    print(f"Computed numerical anomalies saved to NetCDF: {nc_save_path}")

    if vmin is None:
        vmin = np.nanpercentile(data, 2)  # Use percentiles for better scaling
    if vmax is None:
        vmax = np.nanpercentile(data, 98)
    
    print(f"Initial color limits: vmin={vmin:.2e}, vmax={vmax:.2e}")
    
    # Make colorbar symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    print(f"Final symmetric color limits: vmin={vmin:.2e}, vmax={vmax:.2e}")

    levels = np.linspace(vmin, vmax, nlevels)
    base_cmap = plt.get_cmap(cmap_name)
    cmap_disc = ListedColormap(base_cmap(np.linspace(0, 1, nlevels - 1)))
    norm = BoundaryNorm(levels, ncolors=cmap_disc.N, clip=True)

    fig, ax = plt.subplots(figsize=(16, 6))

    # imshow: transpose so y is latitude (data shape: time x lat)
    times_num = mdates.date2num(times)
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
        major_locator = mdates.YearLocator(1)
    ax.xaxis.set_major_locator(major_locator)
    # ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1,3,5,7,9,11)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), ha='center', size=font_size)
    plt.setp(ax.get_yticklabels(), size=14)

    ax.set_ylabel('Latitude (°)', size=16)
    ax.set_xlabel('Year', size=16)
    title = f"ERA5 Reanalysis {component} fluctuations from Climatology ({start_year}-{end_year})"
    if p_min > 0 and p_max > 0:
        title += f"\n(Integrated: {p_min/100:.0f}-{p_max/100:.0f} hPa)"
    ax.set_title(title, size=18)
    ax.grid(True, alpha=0.3)
    
    # Add black solid line at equator (latitude = 0)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=10)

    # Colorbar with better formatting
    # Place colorbar horizontally at the bottom to span the width
    # Use `shrink` close to 1.0 so the colorbar occupies most of the axis width
    cbar = fig.colorbar(
        im,
        ax=ax,
        boundaries=levels,
        extend='both',
        orientation='horizontal',
        spacing='proportional',
        pad=0.16,
        fraction=0.03,
        shrink=1.5,
        aspect=100
    )
    
    # Determine if we need to factor out scientific notation
    max_abs_value = max(abs(vmin), abs(vmax))
    if max_abs_value >= 1e3 or max_abs_value <= 1e-3:
        # Calculate the order of magnitude
        order = int(np.floor(np.log10(max_abs_value)))
        factor = 10**order
        
        # Format units with superscripts and factor
        units = da.attrs.get('units', '')
        if units:
            units_formatted = units.replace('**', '^')  # Convert ** to ^ first
            units_formatted = units_formatted.replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
            units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
            units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
            units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
            label_text = f"{component} Anomalies (10^{{{order}}} {units_formatted})"
        else:
            label_text = f"{component} Anomalies (×10²⁴?)"
        
        # Scale tick labels by the factor
        scaled_levels = levels / factor
        cbar.set_ticks(levels[::max(1, len(levels)//8)])
        cbar.set_ticklabels([f'{val/factor:.1f}' for val in levels[::max(1, len(levels)//8)]])
    else:
        # Normal formatting without factoring
        units = da.attrs.get('units', '')
        if units:
            units_formatted = units.replace('**', '^')  # Convert ** to ^ first
            units_formatted = units_formatted.replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
            units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
            units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
            units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
            label_text = f"{component} Anomalies ({units_formatted})"
        else:
            label_text = f"{component} Anomalies"
        
        # Normal tick formatting
        tick_spacing = max(1, len(levels)//8)
        cbar.set_ticks(levels[::tick_spacing])
    
    # Label placement depends on colorbar orientation
    if getattr(cbar, 'orientation', 'vertical') == 'horizontal':
        cbar.set_label(label_text, rotation=0, labelpad=10, size=14)
        cbar.ax.tick_params(labelsize=12)
        # Make space at bottom for horizontal colorbar
        fig.subplots_adjust(bottom=0.15)
    else:
        cbar.set_label(label_text, rotation=270, labelpad=20, size=16)

    fig.tight_layout()
    
    if savefile:
        # Save to output directory
        save_path = os.path.join(output_dir, savefile)
        fig.savefig(save_path, dpi=500, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()

    ds.close()
    return fig, ax

# %%
# Plot AAM anomalies for the specified climatology period
if __name__ == '__main__':
    savefile = f'AAM_anomalies_{start_yr}-{end_yr}_p{int(p_min/100)}-{int(p_max/100)}hPa.png'
    plot_AAM_anomalies(start_yr, end_yr, component='AAM', 
                       savefile=savefile, 
                       show=True,
                       input_nc_path=args.input_nc,
                       enso_csv_path=args.enso_csv)
    
    
# %%
