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
ear5_path_base = "/badc/ecmwf-era5/data/oper/an_ml/" #an_ml for analysis on model levels
# an_sfc for analysis on surface levels
save_path = f"{base_path}/AAM_data/monthly_mean/"
# base path hhhn2@sci-vm-01:/badc/ecmwf-era5/data/oper/an_ml/2022/01/01$
# full file name ecmwf-era5_oper_an_ml_197901011100.u.nc
# ecmwf-era5_oper_an_ml_197901011100.lnsp.nc
# two variables needed: u (eastward_wind) and lnsp ln(surface_air_pressure)

# Load the a and b sigma level coefficients
# exec(open('vertical_levels.py').read())
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

year = 1980
m = 1

# Load zonal winds 
ds_u = xr.open_mfdataset(f'{ear5_path_base}{year}/{str(m).zfill(2)}/*/*.u.nc')
# print(ds_u)
ds_u_monthly_mean = ds_u.resample(time='1MS').mean()
ds_u_mm_zm = ds_u_monthly_mean.mean(dim='longitude')
# print(ds_u_mm_zm)

# Load natural log of surface pressure
ds_lnsp = xr.open_mfdataset(f'{ear5_path_base}{year}/{str(m).zfill(2)}/*/*.lnsp.nc')
ds_lnsp_monthly_mean = ds_lnsp.resample(time='1MS').mean()
ds_lnsp_mm_zm = ds_lnsp_monthly_mean.mean(dim='longitude')
#ds_lnsp_mm_zm.lnsp = np.exp(ds_lnsp_mm_zm.lnsp) # convert ln(sp) to sp in Pa !! # Conversion below

print(ds_lnsp_mm_zm.lnsp.values)
try:
    print(ds_lnsp_mm_zm.lnsp.shape)
except Exception as e:
    pass

# variable name detection
if 'u' in ds_u_mm_zm.data_vars:
    u_var = ds_u_mm_zm['u']
elif 'eastward_wind' in ds_u_mm_zm.data_vars:
    u_var = ds_u_mm_zm['eastward_wind']
else:
    raise KeyError("Cannot find zonal wind variable (u or eastward_wind) in dataset")

# surface pressure detection (lnsp -> exp, sp -> direct)
if 'lnsp' in ds_lnsp_mm_zm.data_vars:
    ps_var = np.exp(ds_lnsp_mm_zm['lnsp'])
elif 'sp' in ds_lnsp_mm_zm.data_vars:
    ps_var = ds_lnsp_mm_zm['sp']
elif 'surface_air_pressure' in ds_lnsp_mm_zm.data_vars:
    ps_var = ds_lnsp_mm_zm['surface_air_pressure']
else:
    raise KeyError("Cannot find surface pressure variable (lnsp, sp, or surface_air_pressure)")

# compute u on mid-levels: shape (time, n_mid, lat)
# compute mid-levels by averaging adjacent model levels; keep as numpy for the later ops
u_mid = 0.5 * (u_var.isel(level=slice(1, None)).values + u_var.isel(level=slice(0, -1)).values)
# u_mid dims: (time, n_mid, lat)
time_coords = u_var['time'].values
lat_coords = u_var['latitude'].values

# prepare dp: da, db arrays have length n_mid
# ps_zm values shape (time, lat)
# ps_var already holds exp(lnsp) above, so do NOT exp() again — use the values directly
ps_vals = ps_var.values  # (time, lat) surface pressure in Pa
dp = da[:, None, None] + db[:, None, None] * ps_vals[None, :, :]  # (n_mid, time, lat)
# reorder dp to (time, n_mid, lat)
dp = np.transpose(dp, (1, 0, 2))

# latitude parameters
lat_rad = np.radians(lat_coords)
cos_lat = np.cos(lat_rad)
cossq = cos_lat**2
rocos = omega * r * cos_lat  # (lat,)
# delta_phi: use gradient (per-lat spacing)
delta_phi = np.abs(np.gradient(lat_rad))  # (lat,)

# broadcast rocos, cossq, delta_phi to (time, n_mid, lat) shapes as needed
rocos_b = rocos[None, None, :]          # (1,1,lat)
cossq_b = cossq[None, None, :]          # (1,1,lat)
delta_phi_b = delta_phi[None, None, :]  # (1,1,lat)

# integrand: (rocos + u_mid) * cossq * r3g * dp * 2*pi * delta_phi
integrand = (rocos_b + u_mid) * cossq_b * r3g * dp * two_pi * delta_phi_b  # (time,n_mid,lat), multiply 2pi for the full longitudinal integral (after zonal mean)

# sum over vertical mid-levels (axis=1) -> result shape (time, lat)
aam_time_lat = np.sum(integrand, axis=1)

# create xarray DataArray and save
aam_da = xr.DataArray(
    aam_time_lat,
    coords={'time': time_coords, 'latitude': lat_coords},
    dims=['time', 'latitude'],
    name='AAM'
)
aam_da.attrs['long_name'] = 'absolute_atmospheric_angular_momentum'
aam_da.attrs['units'] = 'kg m**-1 s**-1'

out_fname = f"{save_path}AAM_ERA5_{np.datetime_as_string(time_coords[0], unit='M')}.nc"
aam_da.to_dataset(name='AAM').to_netcdf(out_fname)
print("Wrote", out_fname)

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")