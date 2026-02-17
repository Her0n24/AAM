# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap
import os
import glob

base_dir = os.getcwd()
AAM_data_path = f"{base_dir}/monthly_mean/AAM/"
output_dir = f"{base_dir}/AAMA_fig/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

start_yr = 1980
end_yr = 2000

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
        pattern = f"{AAM_data_path}AAM_ERA5_{year}*.nc" # Excludes files starting with 'f' after year
        year_files = [f for f in glob.glob(pattern) if '_full_level' not in f]
        if not year_files:
            print(f"Warning: No files found for year {year}")
            continue
        all_files.extend(year_files)
    
    if not all_files:
        raise FileNotFoundError(f"No files found for period {start_year}-{end_year}")
    
    # Load all data
    ds = xr.open_mfdataset(all_files, combine='by_coords')
    
    if component not in ds.data_vars:
        raise KeyError(f"variable '{component}' not found in dataset")

    da = ds[component]

    # ensure dims include time and latitude
    if 'time' not in da.dims or 'latitude' not in da.dims:
        raise ValueError('dataarray must have dims (time, latitude)')
    
    # Calculate climatological mean
    # Group by month and calculate mean across all years
    climatology = da.groupby('time.month').mean('time')
    
    # Calculate anomalies by subtracting climatology from each month
    anomalies = da.groupby('time.month') - climatology
    
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
    
    # Make colorbar symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

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
    ax.set_title(f"ERA5 Reanalysis {component} fluctuations from Climatology ({start_year}-{end_year})", size=18)
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
            label_text = f"{component} Anomalies (10²⁴ {units_formatted})"
        else:
            label_text = f"{component} Anomalies (×10²⁴)"
        
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
plot_AAM_anomalies(start_yr, end_yr, component='AAM', 
                   savefile=f'AAM_anomalies_{start_yr}-{end_yr}.png', 
                   show=True, vmin=-0.2e24, vmax=0.2e24)
# %%
