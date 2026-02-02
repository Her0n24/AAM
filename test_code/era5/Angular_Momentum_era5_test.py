"""
The code uses monthly mean values of surface pressure (ps) and zonal winds (u)
to compute vertically integrated angular momentum summed over longitude
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

start_yr = 1990
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
two_pi = 2.0 * np.pi


for year in range(start_yr, end_yr + 1):
    for m in range(1, 13):
        # Inspect whether the file exist or not
        print(f"Processing year={year}, month={m:02d}")
        # Define file paths
        u_file = f'{monthly_mean_path_base}/zonal_mean_ERA5_u_{year}-{str(m).zfill(2)}.nc'
        sp_file = f'{monthly_mean_path_base}/zonal_mean_ERA5_sp_{year}-{str(m).zfill(2)}.nc'
        
        # Check if files exist
        if not os.path.exists(u_file):
            print(f"  Skipping: {u_file} does not exist")
            continue
        if not os.path.exists(sp_file):
            print(f"  Skipping: {sp_file} does not exist")
            continue
        
        # Try to load files and check for corruption
        try : 
            # Load zonal winds 
            ds_u_mm_zm = xr.open_mfdataset(u_file)
            # Load natural log of surface pressure
            ds_lnsp_mm_zm = xr.open_mfdataset(sp_file)

            # Test basic data access to catch corruption
            _ = ds_u_mm_zm.dims
            _ = ds_lnsp_mm_zm.dims
            _ = list(ds_u_mm_zm.data_vars.keys())
            _ = list(ds_lnsp_mm_zm.data_vars.keys())

            print(f"  Files loaded successfully")
            
        except Exception as e:
            print(f"  Skipping: Error loading files - {str(e)}")
            continue

        try:
            print(ds_lnsp_mm_zm)
            print(ds_u_mm_zm)
        except Exception as e:
            raise RuntimeError("Error accessing data variables in datasets: " + str(e))

        # variable name detection
        if 'u_zonal_mean' in ds_u_mm_zm.data_vars:
            u_var = ds_u_mm_zm['u_zonal_mean']
        elif 'u' in ds_u_mm_zm.data_vars:
            u_var = ds_u_mm_zm['u']
        elif 'eastward_wind' in ds_u_mm_zm.data_vars:
            u_var = ds_u_mm_zm['eastward_wind']
        else:
            raise KeyError("Cannot find zonal wind variable (u or eastward_wind) in dataset")

        # surface pressure detection (lnsp -> exp, sp -> direct)
        if 'surface_pressure_zonal_mean' in ds_lnsp_mm_zm.data_vars:
            ps_var = ds_lnsp_mm_zm['surface_pressure_zonal_mean']
        elif 'sp' in ds_lnsp_mm_zm.data_vars:
            ps_var = ds_lnsp_mm_zm['sp']
        elif 'surface_air_pressure' in ds_lnsp_mm_zm.data_vars:
            ps_var = ds_lnsp_mm_zm['surface_air_pressure']
        else:
            raise KeyError("Cannot find surface pressure variable (lnsp, sp, or surface_air_pressure)")

        # compute u on mid-levels: shape (time, n_mid, lat)
        # compute mid-levels by averaging adjacent model levels; keep as numpy for the later ops
        u_mid = 0.5 * (u_var.isel(level=slice(1, None)).values + u_var.isel(level=slice(0, -1)).values)

        # Squeeze out the time dimension since we only have one time point
        if u_mid.shape[0] == 1:
            u_mid = u_mid.squeeze(axis=0)  # (n_mid, lat)

        # u_mid dims: (time, n_mid, lat)
        time_coords = u_var['time'].values
        lat_coords = u_var['latitude'].values

        # Ensure time_coords is always an array, even if it's a single value
        if time_coords.ndim == 0:
            time_coords = np.array([time_coords])

        # prepare dp: da, db arrays have length n_mid
        # ps_zm values shape (time, lat)
        ps_vals = ps_var.values  # (time, lat) surface pressure in Pa
        if ps_vals.ndim == 2 and ps_vals.shape[0] == 1:
            ps_vals = ps_vals.squeeze(axis=0)  # (lat,)

        dp = da[:, None] + db[:, None] * ps_vals[None, :]  # (n_mid, lat)

        # latitude parameters
        lat_rad = np.radians(lat_coords)
        cos_lat = np.cos(lat_rad)
        cossq = cos_lat**2
        rocos = omega * r * cos_lat  # (lat,)
        # delta_phi: use gradient (per-lat spacing)
        delta_phi = np.abs(np.gradient(lat_rad))  # (lat,)

        # broadcast rocos, cossq, delta_phi to (n_mid, lat) shapes as needed
        rocos_b = rocos[None, :]          # (1,lat)
        cossq_b = cossq[None, :]          # (1,lat)
        delta_phi_b = delta_phi[None, :]  # (1,lat)

        # integrand: (rocos + u_mid) * cossq * r3g * dp * 2*pi * delta_phi
        integrand = (rocos_b + u_mid) * cossq_b * r3g * dp * two_pi * delta_phi_b  # (n_mid,lat), multiply 2pi for the full longitudinal integral (after zonal mean)
        # integrand shape (n_min, lat)

        # sum over vertical mid-levels (axis=0) -> result shape (lat,)
        aam_lat = np.sum(integrand, axis=0)

        # create xarray DataArray and save
        aam_da = xr.DataArray(
            aam_lat[None,:], # Add time dimension back, date is from the input nc: (1, lat)
            coords={'time': time_coords, 'latitude': lat_coords},
            dims=['time', 'latitude'],
            name='AAM'
        )
        aam_da.attrs['long_name'] = 'absolute_atmospheric_angular_momentum'
        aam_da.attrs['units'] = 'kg m**-1 s**-1'

        out_fname = f"{save_path}/AAM_ERA5_{np.datetime_as_string(time_coords[0], unit='M')}.nc"
        aam_da.to_dataset(name='AAM').to_netcdf(out_fname)
        print("Wrote", out_fname)

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")