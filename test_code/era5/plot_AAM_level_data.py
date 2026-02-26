# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import os
import glob
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=100, help='Minimum pressure level (hPa) to include (default: 100 hPa)')
parser.add_argument('--p-max', type=float, default=1000, help='Maximum pressure level (hPa) to include (default: 1000 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year for analysis (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year for analysis (default: 2000)')
args = parser.parse_args()

scratch_path = "/work/scratch-nopw2/hhhn2"
base_dir = os.getcwd()
AAM_data_path = f"{base_dir}/monthly_mean/AAM/"
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

def plot_AAM_anomalies(start_year, end_year, component='AAM', *, nlevels=11, cmap_name='RdBu_r', 
                       vmin=None, vmax=None, savefile=None, show=True):
    """Plot AAM anomalies (variation from climatological mean) for a given period.

    Loads files matching `AAM_ERA5_{year}*.nc` from `AAM_data/monthly_mean/` for all years
    in the range [start_year, end_year], calculates the climatological mean, and plots
    the anomalies as latitude vs time using a discrete colormap.

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
    
    # Load all data for the specified period
    all_files = []
    for year in range(start_year, end_year + 1):
        pattern = f"{AAM_data_path}AAM_ERA5_{year}*_full_level.nc"
        year_files = glob.glob(pattern)
        if not year_files:
            print(f"Warning: No files found for year {year}")
            continue
        all_files.extend(year_files)
    
    if not all_files:
        raise FileNotFoundError(f"No files found for period {start_year}-{end_year}")
    
    # Load all data with chunking for better memory management
    print(f"Loading {len(all_files)} AAM files...")
    ds = xr.open_mfdataset(all_files, combine='by_coords', chunks={'time': 12, 'latitude': 180, 'longitude': 360})
    
    if component not in ds.data_vars:
        raise KeyError(f"variable '{component}' not found in dataset")

    da = ds[component]
    
    print(f"===== INITIAL DATA =====")
    print(f"AAM data shape: {da.shape}, dims: {da.dims}")
    print(f"AAM dimensions: {dict(da.sizes)}")
    print(f"Has level? {'level' in da.dims}")
    print(f"Has mid_level? {'mid_level' in da.dims}")
    print(f"Has longitude? {'longitude' in da.dims}")
    print(f"AAM units: {da.attrs.get('units', 'N/A')}")
    print(f"========================")

    # Check if this is 3D/4D data with level dimension (can be 'level' or 'mid_level')
    level_dim = None
    if 'level' in da.dims:
        level_dim = 'level'
    elif 'mid_level' in da.dims:
        level_dim = 'mid_level'
    
    if level_dim:
        # Load ONE surface pressure file to calculate pressure levels
        # Since a/b coefficients are constant, we only need one representative file
        sp_pattern = f"{scratch_path}/ERA5/monthly_mean/variables/ERA5_sp_{start_year}-*.nc"
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
            level_coords = da[level_dim].values
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
            print(f"===== LEVEL INTEGRATION (SP files found) =====")
            print(f"Before level selection - shape: {da.shape}, dims: {da.dims}")
            print(f"Using level dimension: '{level_dim}'")
            da = da.sel({level_dim: selected_levels}).sum(dim=level_dim)
            print(f"After .sum(dim='{level_dim}') -  shape: {da.shape}, dims: {da.dims}")
            # Use compute() to force evaluation with dask if needed
            if hasattr(da, 'compute'):
                print("Computing dask array...")
                da = da.compute()
                print(f"After .compute() - shape: {da.shape}, dims: {da.dims}")
            print(f"===============================================")
            
            ds_sp.close()
        else:
            print(f"===== LEVEL INTEGRATION (NO SP files) =====")
            print(f"SP pattern searched: {sp_pattern}")
            print(f"Data shape before level sum: {da.shape}, dims: {da.dims}")
            print(f"Level dimension: '{level_dim}'")
            da = da.sum(dim=level_dim)
            print(f"Data shape after level sum: {da.shape}, dims: {da.dims}")
            if hasattr(da, 'compute'):
                print("Computing dask array...")
                da = da.compute()
                print(f"After .compute() - shape: {da.shape}, dims: {da.dims}")
            print(f"============================================")
    
    # Integrate over longitude if it exists (to get zonal mean/integral)
    print(f"===== LONGITUDE INTEGRATION =====")
    print(f"Before longitude check - shape: {da.shape}, dims: {da.dims}")
    print(f"Has longitude dimension: {'longitude' in da.dims}")
    if 'longitude' in da.dims:
        print("Longitude dimension found - integrating...")
        da = da.sum(dim='longitude')
        print(f"After .sum(dim='longitude') - sh: {da.shape}, dims: {da.dims}")
        # Force computation if using dask
        if hasattr(da, 'compute'):
            print("Computing dask array...") 
            da = da.compute()
            print(f"After .compute() - shape: {da.shape}, dims: {da.dims}")
    else:
        print("No longitude dimension found - skipping longitude integration")
    print(f"=================================")
    
    # ensure dims include time and latitude
    print(f"===== FINAL DATA CHECK =====")
    print(f"Data shape: {da.shape}, dims: {da.dims}")
    print(f"Data dimensions: {dict(da.sizes)}")
    if 'time' not in da.dims or 'latitude' not in da.dims:
        raise ValueError(f'dataarray must have dims (time, latitude), but has dims: {da.dims}')
    if 'level' in da.dims or 'mid_level' in da.dims:
        raise ValueError(f'dataarray still has level dimension! Shape: {da.shape}, dims: {da.dims}')
    if 'longitude' in da.dims:
        raise ValueError(f'dataarray still has longitude dimension! Shape: {da.shape}, dims: {da.dims}')
    print(f"Data is valid for plotting")
    print(f"============================")
    
    # Calculate climatological mean
    # Group by month and calculate mean across all years
    print("Computing climatology...")
    climatology = da.groupby('time.month').mean('time')
    if hasattr(climatology, 'compute'):
        climatology = climatology.compute()
    print(f"Climatology computed: shape={climatology.shape}")
    
    # Calculate anomalies by subtracting climatology from each month
    print("Computing anomalies...")
    anomalies = da.groupby('time.month') - climatology
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
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
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
        
        # Convert order to superscript for display
        order_str = str(abs(order))
        superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
                          '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        order_superscript = ''.join(superscript_map[d] for d in order_str)
        if order < 0:
            order_superscript = '⁻' + order_superscript
        
        # Format units with superscripts and factor
        units = da.attrs.get('units', '')
        if units:
            units_formatted = units.replace('**', '^')  # Convert ** to ^ first
            units_formatted = units_formatted.replace('m^2', 'm²').replace('s^-1', 's⁻¹').replace('kg^-1', 'kg⁻¹')
            units_formatted = units_formatted.replace('m^-2', 'm⁻²').replace('s^-2', 's⁻²').replace('kg^-2', 'kg⁻²')
            units_formatted = units_formatted.replace('^2', '²').replace('^-1', '⁻¹').replace('^3', '³')
            units_formatted = units_formatted.replace('^-2', '⁻²').replace('^-3', '⁻³').replace('^1', '¹')
            label_text = f"{component} Anomalies (10{order_superscript} {units_formatted})"
        else:
            label_text = f"{component} Anomalies (×10{order_superscript})"
        
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
    savefile = f'AAM_anomalies_{start_yr}-{end_yr}_p{int(p_min/100)}-{int(p_max/100)}hPa_new.png'
    plot_AAM_anomalies(start_yr, end_yr, component='AAM', 
                       savefile=savefile, 
                       show=True, vmin=-0.2e24, vmax=0.2e24)
# %%
