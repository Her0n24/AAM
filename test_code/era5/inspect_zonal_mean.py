# %%
import re
import xarray as xr
import matplotlib.pyplot as plt

file = "monthly_mean/variables/zonal_mean_ERA5_u_1998-11.nc"
output_path = "monthly_mean/variables/figs"
# Extract month and year from filename

match = re.search(r'_(\d{4})-(\d{2})\.nc$', file)
if match:
    year = int(match.group(1))
    month = int(match.group(2))
else:
    year = None
    month = None

ds= xr.open_dataset(file)

# Check data statistics to diagnose blank areas
print("=== DATA DIAGNOSTICS ===")
if 'u_zonal_mean' in ds.variables:
    data = ds.u_zonal_mean
    print(f"Data shape: {data.shape}")
    print(f"Data min: {data.min().values:.2f}")
    print(f"Data max: {data.max().values:.2f}")
    print(f"Data mean: {data.mean().values:.2f}")
    print(f"Number of NaN values: {data.isnull().sum().values}")
    print(f"Total data points: {data.size}")
    print(f"Values outside colorbar range (-60 to 60):")
    print(f"  Below -60: {(data < -60).sum().values} points")
    print(f"  Above 60: {(data > 60).sum().values} points")
    print(f"  Within -60 to 60: {((data >= -60) & (data <= 60)).sum().values} points")

# %%
# Detect whether u or lnsp is present
if 'u_zonal_mean' in ds.variables:
    # Create heatmap with contour lines
    plt.figure(figsize=(12, 8))
    
    # Filled contours (color fill)
    import numpy as np
    levels = np.linspace(-60, 60, 21)  # 21 levels from -60 to 60 (15 intervals)
    cs_fill = plt.contourf(ds.latitude, ds.level, ds.u_zonal_mean, levels=levels, cmap='RdBu_r',
                           vmin=-60, vmax=60, extend='both')
    cbar = plt.colorbar(cs_fill, label='U wind (m/s)')
    cbar.set_label('U wind (m/s)', fontsize=14)
    
    # Force colorbar to use the specified vmin/vmax range
    cbar.mappable.set_clim(vmin=-60, vmax=60)

    # Increase colorbar tick text size
    cbar.ax.tick_params(labelsize=12)

    # Line contours with spacing of 15
    cs_lines = plt.contour(ds.latitude, ds.level, ds.u_zonal_mean, 
                          levels=range(-150, 151, 15), colors='black', linewidths=0.8, alpha=0.8,
                          extend='both')
    
    # Add contour labels
    plt.clabel(cs_lines, inline=True, fontsize=14, fmt='%d')
    
    plt.gca().invert_yaxis()  # Invert y-axis. Level 1 (surface) is at bottom, level 137 (top) is at top

    plt.xlabel('Latitude', size=14)
    plt.ylabel('Level', size=14)
    plt.title(f'ERA5 U Zonal Mean {year}-{month:02d} Monthly Mean', size =20)

    # Increase x and y axis tick text size
    plt.tick_params(axis='both', labelsize=14)

    plt.savefig(f'{output_path}/ERA5_u_monthly_zonal_mean_{year}_{month:02d}.png', dpi=400, bbox_inches='tight')
    plt.show()
elif 'surface_pressure_zonal_mean' in ds.variables:
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.plot(ds.latitude, ds.surface_pressure_zonal_mean/100, label='Surface Pressure')
    plt.xlabel('Latitude', size=14)
    plt.ylabel('Surface Pressure (hPa)', size=14)
    plt.title(f'ERA5 Surface Pressure Zonal Mean {year}-{month:02d} Monthly Mean', size=20)
    plt.grid(which='major', alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    # Increase x and y axis tick text size
    plt.tick_params(axis='both', labelsize=14)
    plt.legend()

    plt.savefig(f'{output_path}/ERA5_surface_pressure_monthly_zonal_mean_{year}_{month:02d}.png', dpi=400, bbox_inches='tight')
    plt.show()


# test which .nc files are valid within a certain year range 
# report ones that are not valid
import os
import xarray as xr
base_path = "monthly_mean/variables/"

start_year = 1980
end_year = 1999

for year in range(start_year, end_year + 1):
    for month in range(1,13):
        filename_u = f"zonal_mean_ERA5_u_{year:04d}-{month:02d}.nc"
        filepath_u = os.path.join(base_path, filename_u)
        try:
            ds = xr.open_dataset(filepath_u)
            _ = ds.u_zonal_mean.mean().values  # attempt to access data
        except Exception as e:
            print(f"Invalid file: {filename_u} -- Error: {e}")
        filename_sp = f"zonal_mean_ERA5_sp_{year:04d}-{month:02d}.nc"
        filepath_sp = os.path.join(base_path, filename_sp)
        try:
            ds = xr.open_dataset(filepath_sp)
            _ = ds.surface_pressure_zonal_mean.mean().values  # attempt to access data
        except Exception as e:
            print(f"Invalid file: {filename_sp} -- Error: {e}")

