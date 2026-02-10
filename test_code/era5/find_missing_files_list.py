import os
import xarray as xr
base_path = "monthly_mean/variables/"
start_year = 1980
end_year = 1999

missing_files = []

for year in range(start_year, end_year + 1):
    for month in range(1,13):
        filename_u = f"zonal_mean_ERA5_u_{year:04d}-{month:02d}.nc"
        filepath_u = os.path.join(base_path, filename_u)
        filename_sp = f"zonal_mean_ERA5_sp_{year:04d}-{month:02d}.nc"
        filepath_sp = os.path.join(base_path, filename_sp)

        # Check if files exist and are valid
        for filename, filepath, var in [(filename_u, filepath_u, 'u'), (filename_sp, filepath_sp, 'sp')]:
            try:
                if not os.path.exists(filepath):
                    missing_files.append(f"{year} {month} {var}")
                else:
                    ds = xr.open_dataset(filepath)
                    if var == 'u':
                        _ = ds.u_zonal_mean.mean().values
                    else:
                        _ = ds.surface_pressure_zonal_mean.mean().values
                    ds.close()
            except Exception as e:
                missing_files.append(f"{year} {month} {var}")

# Write to file
with open('missing_files.txt', 'w') as f:
    for item in missing_files:
        f.write(f"{item}\n")

print(f"Found {len(missing_files)} missing files")

