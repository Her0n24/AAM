"""
The code uses monthly mean values of surface pressure (ps) and zonal winds (u)
to compute vertically integrated angular momentum at each latitude and longitude, 
and saves the result as a netCDF file for each month.
This version computes vertically integrated AAM (not zonal mean) with dimensions (time, lat, lon).

Usage:
    python compute_AAM_full_field.py
"""
import time
time_start = time.time()
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
import os
import xarray as xr
import tqdm

base_path = os.getcwd()
monthly_mean_path_base = f"{base_path}/monthly_mean/variables" 
save_path = f"{base_path}/monthly_mean/AAM"
print(monthly_mean_path_base)

start_yr = 1980
end_yr = 2000

# Load the a and b sigma level coefficients
sigma_coeff = pd.read_csv(f'{base_path}/l137_a_b.csv')
a = sigma_coeff['a [Pa]'].values[1:] #omit level 0
b = sigma_coeff['b'].values[1:]

# layer differences (for dp = da + db * ps)
da = a[1:] - a[:-1]
db = b[1:] - b[:-1]

# constants
r = 6371229.0 # m
g = 9.80665 # m/s2
omega = 7.292116e-5 # rad/s
r3g = r**3 / g

# Flag to check if da/db have been reversed (do this only once)
da_db_reversed = False

for year in range(start_yr, end_yr + 1):
    for m in range(1, 13):
        # Inspect whether the file exist or not
        print(f"Processing year={year}, month={m:02d}")
        # Define file paths - use FULL fields (not zonal mean)
        u_file = f'{monthly_mean_path_base}/ERA5_u_{year}-{str(m).zfill(2)}.nc'
        sp_file = f'{monthly_mean_path_base}/ERA5_sp_{year}-{str(m).zfill(2)}.nc'
        
        # Check if files exist
        if not os.path.exists(u_file):
            print(f"  Skipping: {u_file} does not exist")
            continue
        if not os.path.exists(sp_file):
            print(f"  Skipping: {sp_file} does not exist")
            continue
        
        # Try to load files and check for corruption
        try : 
            # Load zonal winds (full fields with lon dimension)
            ds_u_mm = xr.open_mfdataset(u_file)
            # Load surface pressure
            ds_sp_mm = xr.open_mfdataset(sp_file)

            # Test basic data access to catch corruption
            _ = ds_u_mm.dims
            _ = ds_sp_mm.dims
            _ = list(ds_u_mm.data_vars.keys())
            _ = list(ds_sp_mm.data_vars.keys())

            print(f"  Files loaded successfully")
            
        except Exception as e:
            print(f"  Skipping: Error loading files - {str(e)}")
            continue

        try:
            print(ds_sp_mm)
            print(ds_u_mm)
        except Exception as e:
            raise RuntimeError("Error accessing data variables in datasets: " + str(e))

        # variable name detection
        if 'u' in ds_u_mm.data_vars:
            u_var = ds_u_mm['u']
        elif 'eastward_wind' in ds_u_mm.data_vars:
            u_var = ds_u_mm['eastward_wind']
        elif 'u_component_of_wind' in ds_u_mm.data_vars:
            u_var = ds_u_mm['u_component_of_wind']
        else:
            raise KeyError("Cannot find zonal wind variable (u or eastward_wind) in dataset")

        # surface pressure detection
        if 'sp' in ds_sp_mm.data_vars:
            ps_var = ds_sp_mm['sp']
        elif 'surface_air_pressure' in ds_sp_mm.data_vars:
            ps_var = ds_sp_mm['surface_air_pressure']
        elif 'surface_pressure' in ds_sp_mm.data_vars:
            ps_var = ds_sp_mm['surface_pressure']
        else:
            raise KeyError("Cannot find surface pressure variable (sp or surface_air_pressure)")

        # Ensure the ordering of the `da`/`db` coefficients matches the dataset `level` ordering.
        # Coefficients `a` and `b` are provided top->bottom (level 1 is top). If the dataset's
        # `level` coordinate runs in the opposite direction, reverse `da`/`db` so `dp` aligns
        # with `u_mid` when we compute layer thickness as `da + db * ps`.
        # Only check and reverse once (first iteration)
        if not da_db_reversed:
            try:
                level_coords = u_var['level'].values
                # If level coordinates are decreasing (e.g., 137..1), reverse da/db
                if np.all(np.diff(level_coords) < 0):
                    da = da[::-1]
                    db = db[::-1]
                    print("Reversed da/db to match dataset level ordering")
                da_db_reversed = True
            except Exception:
                # If we can't determine ordering, warn but continue; user should verify alignment
                print("Warning: could not determine `level` ordering; verify da/db alignment manually")
                da_db_reversed = True

        try:
            # compute u on mid-levels: shape (time, n_mid, lat, lon)
            # compute mid-levels by averaging adjacent model levels
            u_mid = 0.5 * (u_var.isel(level=slice(1, None)).values + u_var.isel(level=slice(0, -1)).values)

            # Squeeze out the time dimension if we only have one time point
            if u_mid.shape[0] == 1:
                u_mid = u_mid.squeeze(axis=0)  # (n_mid, lat, lon)

            # Extract coordinates
            time_coords = u_var['time'].values
            lat_coords = u_var['latitude'].values
            lon_coords = u_var['longitude'].values
            full_lvl_coords = u_var['level'].values
            # Create mid-level coordinates (average of adjacent full levels)
            mid_lvl_coords = 0.5 * (full_lvl_coords[:-1] + full_lvl_coords[1:])

            # Ensure time_coords is always an array, even if it's a single value
            if time_coords.ndim == 0:
                time_coords = np.array([time_coords])

            # prepare dp: da, db arrays have length n_mid
            # ps values shape (time, lat, lon) -> squeeze to (lat, lon) if single time
            ps_vals = ps_var.values  # (time, lat, lon) surface pressure in Pa
            if ps_vals.ndim == 3 and ps_vals.shape[0] == 1:
                ps_vals = ps_vals.squeeze(axis=0)  # (lat, lon)

            # Broadcast: dp = da[:, None, None] + db[:, None, None] * ps_vals[None, :, :]
            # Result: (n_mid, lat, lon)
            dp = da[:, None, None] + db[:, None, None] * ps_vals[None, :, :]  # (n_mid, lat, lon)

            # latitude parameters
            lat_rad = np.radians(lat_coords)
            cos_lat = np.cos(lat_rad)
            cossq = cos_lat**2
            rocos = omega * r * cos_lat  # (lat,)

            # broadcast rocos, cossq to (n_mid, lat, lon) shapes as needed
            rocos_b = rocos[None, :, None]  # (1, lat, 1)
            cossq_b = cossq[None, :, None]  # (1, lat, 1)

            # AAM at each grid point (no longitudinal or latitudinal integration)
            # integrand: (rocos + u_mid) * cossq * r3g * dp
            # Shape: (n_mid, lat, lon)
            AAM = (rocos_b + u_mid) * cossq_b * r3g * dp

            # Vertically integrate by summing over mid-levels
            # Shape: (lat, lon)
            AAM_vertint = np.sum(AAM, axis=0)

            # create xarray DataArray and save
            aam_da = xr.DataArray(
                AAM_vertint[None, :, :], # Add time dimension back: (1, lat, lon)
                coords={'time': time_coords, 'latitude': lat_coords, 'longitude': lon_coords},
                dims=['time', 'latitude', 'longitude'],
                name='AAM'
            )
            aam_da.attrs['long_name'] = 'vertically_integrated_atmospheric_angular_momentum'
            aam_da.attrs['units'] = 'kg m**2 s**-1'
            aam_da.attrs['description'] = 'Vertically integrated AAM at each lat, lon (not zonal mean)'

            out_fname = f"{save_path}/AAM_ERA5_{np.datetime_as_string(time_coords[0], unit='M')}_vertint.nc"
            aam_da.to_dataset(name='AAM').to_netcdf(out_fname)
            print("Wrote", out_fname)
            
        except Exception as e:
            print(f"  Skipping: Error computing AAM - {str(e)}")
            print(f"  This file may be corrupted: {u_file}")
        
        finally:
            # Close datasets to free file handles
            ds_u_mm.close()
            ds_sp_mm.close()

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
