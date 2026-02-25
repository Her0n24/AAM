import os
import xarray as xr
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Check for missing or corrupt ERA5 monthly mean files')
parser.add_argument('--zonal-mean', action='store_true', 
                    help='Check zonal mean files instead of full field files')
args = parser.parse_args()

scratch_path = "/work/scratch-nopw2/hhhn2"
base_path = f"{scratch_path}/ERA5/monthly_mean/variables"
start_year = 1980
end_year = 1999

missing_files = []

# Create list of all year-month combinations for progress bar
year_months = [(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)]

for year, month in tqdm(year_months, desc="Checking files"):
    if args.zonal_mean:
        filename_u = f"zonal_mean_ERA5_u_{year:04d}-{month:02d}.nc"
        filename_sp = f"zonal_mean_ERA5_sp_{year:04d}-{month:02d}.nc"
        u_varname = 'u_zonal_mean'
        sp_varname = 'surface_pressure_zonal_mean'
    else:
        filename_u = f"ERA5_u_{year:04d}-{month:02d}.nc"
        filename_sp = f"ERA5_sp_{year:04d}-{month:02d}.nc"
        u_varname = 'u'
        sp_varname = 'sp'
    
    filepath_u = os.path.join(base_path, filename_u)
    filepath_sp = os.path.join(base_path, filename_sp)

    # Check if files exist and are valid
    for filename, filepath, var, varname in [
        (filename_u, filepath_u, 'u', u_varname), 
        (filename_sp, filepath_sp, 'sp', sp_varname)
    ]:
        try:
            if not os.path.exists(filepath):
                missing_files.append(f"{year} {month} {var}")
            else:
                ds = xr.open_dataset(filepath)
                # Just check if variable exists and has data (faster than computing mean)
                _ = ds[varname]
                ds.close()
        except Exception as e:
            missing_files.append(f"{year} {month} {var}")

# Write to file
with open('missing_files_scratch.txt', 'w') as f:
    for item in missing_files:
        f.write(f"{item}\n")

print(f"Found {len(missing_files)} missing files")

