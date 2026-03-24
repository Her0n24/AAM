"""
The code uses monthly mean values of surface pressure (ps) and zonal winds (u)
to compute the full 4D AAM field at each pressure level without integration.
Output dimensions are (time, level, latitude, longitude) saved as a single netCDF file.

Note that we are using pressure levels.
nc files naming format 
- ps_mon_historical_HadGEM3-GC31-LL_r37i1p1f3_interp.nc
- ua_mon_historical_HadGEM3-GC31-LL_r11i1p1f3_interp.nc

Expected input dimensions (will be transposed if different):
- u: (time, level, lat, lon)
- ps: (time, lat, lon)

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
# Use scratch space and new directory structure due to workspace migration
CMIP6_path_base = "/work/scratch-nopw2/hhhn2/HadGEM3-GC31-LL"
u_directory = f"{CMIP6_path_base}/Amon/ua/historical/"
ps_directory = f"{CMIP6_path_base}/Amon/ps/historical/"
output_dir = f"{CMIP6_path_base}/AAM/full/"

# output_dir = f"{base_dir}/figures/composites/non_tracking_algorithm/"
# climatology_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/climatology/AAM/"
# u_directory = f"{CMIP6_path_base}/InterpolatedFlds/Amon/ua/historical/HadGEM3-GC31-LL/"
# ps_directory = f"{CMIP6_path_base}/InterpolatedFlds/Amon/ps/historical/HadGEM3-GC31-LL/"

ensemble_members = [f'r{i}i1p1f3' for i in range(6, 12)]

save_path = output_dir

print("u directory:", u_directory)
print("ps directory:", ps_directory)
print("save_path:", save_path)

start_yr = 1980 # Time dimension is days since 1850-01-01-00-00-00, starting at 15, 45, 75, ..., up to 59385 (around 2015)
end_yr = 2000

def _mask_fill_and_nonfinite(da: xr.DataArray) -> xr.DataArray:
    """Mask common fill/missing sentinels and non-finite values."""
    da = da.where(np.isfinite(da))

    for key in ("_FillValue", "missing_value"):
        fv = da.encoding.get(key, None)
        if fv is None:
            fv = da.attrs.get(key, None)

        # Only mask finite numeric sentinels (e.g., 1e20)
        if fv is not None:
            try:
                fv_float = float(fv)
            except Exception:
                fv_float = None

            if fv_float is not None and np.isfinite(fv_float):
                da = da.where(da != fv_float)

    return da


# constants
r = 6371229.0 # m
g = 9.80665 # m/s2
omega = 7.292116e-5 # rad/s
r3g = r**3 / g
two_pi = 2.0 * np.pi

for ensemble_member in tqdm.tqdm(ensemble_members):
    print(f"\n=== Processing ensemble member: {ensemble_member} ===")

    # Define file paths
    u_file = f'{u_directory}ua_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc'
    sp_file = f'{ps_directory}ps_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc'

    # Check if files exist
    if not os.path.exists(u_file):
        print(f"  Skipping: {u_file} does not exist")
        continue
    if not os.path.exists(sp_file):
        print(f"  Skipping: {sp_file} does not exist")
        continue

    # Try to load files and check for corruption
    try:
        # Load zonal winds and surface pressure
        # mask_and_scale=True (default) automatically converts _FillValue to NaN
        ds_u_mm_zm = xr.open_dataset(u_file, mask_and_scale=True)
        ds_lnsp_mm_zm = xr.open_dataset(sp_file, mask_and_scale=True)

        # Test basic data access to catch corruption
        _ = ds_u_mm_zm.dims
        _ = ds_lnsp_mm_zm.dims
        _ = list(ds_u_mm_zm.data_vars.keys())
        _ = list(ds_lnsp_mm_zm.data_vars.keys())

        print(f" Files loaded successfully")

    except Exception as e:
        print(f"  Skipping: Error loading files - {str(e)}")
        continue

    try:
        print(ds_lnsp_mm_zm)
        print(ds_u_mm_zm)
    except Exception as e:
        raise RuntimeError("Error accessing data variables in datasets: " + str(e))

    # variable name detection
    if 'ua' in ds_u_mm_zm.data_vars:
        u_var = ds_u_mm_zm['ua']
    elif 'Eastward_wind' in ds_u_mm_zm.data_vars:
        u_var = ds_u_mm_zm['Eastward_wind']
    elif 'eastward_wind' in ds_u_mm_zm.data_vars:
        u_var = ds_u_mm_zm['eastward_wind']
    else:
        raise KeyError("Cannot find zonal wind variable (ua or eastward_wind) in dataset")

    # surface pressure detection (lnsp -> exp, sp -> direct)
    if 'surface_air_pressure' in ds_lnsp_mm_zm.data_vars:
        ps_var = ds_lnsp_mm_zm['surface_air_pressure']
    elif 'sp' in ds_lnsp_mm_zm.data_vars:
        ps_var = ds_lnsp_mm_zm['sp']
    elif 'ps' in ds_lnsp_mm_zm.data_vars:
        ps_var = ds_lnsp_mm_zm['ps']
    else:
        raise KeyError("Cannot find surface pressure variable (surface_pressure, sp or ps)")

    # Check for fill values in the data
    # CMIP6 data uses _FillValue = 1e20 for missing values
    # xarray's mask_and_scale=True should automatically convert these to NaN
    print("\n=== Verifying fill value handling ===")
    u_fillvalue = u_var.encoding.get('_FillValue', None) or u_var.attrs.get('_FillValue', None)
    ps_fillvalue = ps_var.encoding.get('_FillValue', None) or ps_var.attrs.get('_FillValue', None)
    print(f"u _FillValue from file: {u_fillvalue}")
    print(f"ps _FillValue from file: {ps_fillvalue}")

    # Check and verify dimension order
    print(f"\n=== Verifying dimension order ===")
    print(f"u dimensions from file: {u_var.dims}")
    print(f"u shape from file: {u_var.shape}")
    print(f"ps dimensions from file: {ps_var.dims}")
    print(f"ps shape from file: {ps_var.shape}")

    # Expected dimension order: (time, level, lat, lon)
    expected_dims = ('time', 'plev', 'lat', 'lon')
    if u_var.dims != expected_dims:
        print(f"WARNING: u dimension order is {u_var.dims}, expected {expected_dims}")
        print(f"Will transpose to expected order...")
        u_var = u_var.transpose('time', 'plev', 'lat', 'lon')
        print(f"After transpose: {u_var.dims}")

    expected_ps_dims = ('time', 'lat', 'lon')
    if ps_var.dims != expected_ps_dims:
        print(f"WARNING: ps dimension order is {ps_var.dims}, expected {expected_ps_dims}")
        print(f"Will transpose to expected order...")
        ps_var = ps_var.transpose('time', 'lat', 'lon')
        print(f"After transpose: {ps_var.dims}")

    print("====================================\n")

    # After selecting u_var and ps_var and transposing:
    u_var = _mask_fill_and_nonfinite(u_var)
    ps_var = _mask_fill_and_nonfinite(ps_var)

    # u_mid dims: (time, n_mid, lat)
    time_coords = u_var['time'].values
    lat_coords = u_var['lat'].values
    lon_coords = u_var['lon'].values
    level_coords = u_var['plev'].values

    # Ensure time_coords is always an array, even if it's a single value
    if time_coords.ndim == 0:
        time_coords = np.array([time_coords])

    # Get u and ps values (now in correct dimension order after transpose check)
    u_vals = u_var.values  # (time, level, lat, lon)
    ps_vals = ps_var.values  # (time, lat, lon)

    print(f"u_vals shape: {u_vals.shape}")
    print(f"ps_vals shape: {ps_vals.shape}")

    # Verify xarray properly converted fill values to NaN
    print(f"u_vals: min={np.nanmin(u_vals):.3e}, max={np.nanmax(u_vals):.3e}, NaN count={np.isnan(u_vals).sum()}")
    print(f"ps_vals: min={np.nanmin(ps_vals):.3e}, max={np.nanmax(ps_vals):.3e}, NaN count={np.isnan(ps_vals).sum()}")

    # Check if 1e20 values remain (would indicate mask_and_scale didn't work)
    if u_fillvalue == 1e20:
        n_fill_u = np.sum(np.abs(u_vals - 1e20) < 1e18)  # Check for values near 1e20
        if n_fill_u > 0:
            print(f"WARNING: Found {n_fill_u} values still equal to fill value 1e20 in u - mask_and_scale may have failed")
        else:
            print(f"✓ Fill values properly converted to NaN in u")

    if ps_fillvalue == 1e20:
        n_fill_ps = np.sum(np.abs(ps_vals - 1e20) < 1e18)
        if n_fill_ps > 0:
            print(f"WARNING: Found {n_fill_ps} values still equal to fill value 1e20 in ps - mask_and_scale may have failed")
        else:
            print(f"✓ Fill values properly converted to NaN in ps")

    print("====================================\n")

    # For PRESSURE LEVELS: dp is just the difference between levels
    # level_coords are in Pa (e.g., [100000, 92500, 85000, ...])
    dp_1d = np.abs(np.diff(level_coords))  # Pressure differences (positive)
    print(f"Calculated dp from pressure level differences")
    print(f"  Example: dp[0] = {dp_1d[0]/100:.1f} hPa between levels {level_coords[0]/100:.0f} and {level_coords[1]/100:.0f} hPa")

    # Broadcast dp to match data shape: (time, level-1, lat, lon)
    dp = dp_1d[np.newaxis, :, np.newaxis, np.newaxis] * np.ones((u_vals.shape[0], len(dp_1d), u_vals.shape[2], u_vals.shape[3]))

    # Average u to mid-levels (between pressure level interfaces)
    # u_vals: (time, 19 levels, lat, lon) → (time, 18 mid-levels, lat, lon)
    u_vals = 0.5 * (u_vals[:, :-1, ...] + u_vals[:, 1:, ...])
    print(f"Averaged u to {u_vals.shape[1]} mid-levels")

    print(f"dp shape: {dp.shape}")
    print(f"u_vals shape after averaging: {u_vals.shape}")

    # Latitude parameters
    lat_rad = np.radians(lat_coords)
    cos_lat = np.cos(lat_rad)
    cossq = cos_lat**2
    rocos = omega * r * cos_lat  # (lat,)

    # Check that u_vals and dp have compatible shapes
    if u_vals.shape != dp.shape:
        raise ValueError(f"Shape mismatch: u_vals {u_vals.shape} vs dp {dp.shape}")

    # AAM at each grid point (no longitudinal or latitudinal integration)
    # integrand: (rocos + u_mid) * cossq * r3g * dp
    # Input shape after mid-level averaging: (time, level, lat, lon)
    # Broadcast to match dimensions

    rocos_b = rocos[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, lat, 1)
    cossq_b = cossq[np.newaxis, np.newaxis, :, np.newaxis]  # (1, 1, lat, 1)

    AAM = (rocos_b + u_vals) * cossq_b * r3g * dp

    AAM_4d = AAM

    # Verify AAM calculation results
    print(f"\n=== AAM calculation results ===")
    print(f"AAM_4d shape: {AAM_4d.shape}")
    print(f"AAM_4d min: {np.nanmin(AAM_4d):.3e}, max: {np.nanmax(AAM_4d):.3e}")
    print(f"AAM_4d mean: {np.nanmean(AAM_4d):.3e}, std: {np.nanstd(AAM_4d):.3e}")
    print(f"AAM_4d NaN count: {np.isnan(AAM_4d).sum()} / {AAM_4d.size} ({100*np.isnan(AAM_4d).sum()/AAM_4d.size:.2f}%)")
    print("================================\n")

    # Create xarray DataArray and save
    # AAM_4d shape: (time, level, latitude, longitude)
    try:
        aam_da = xr.DataArray(
            AAM_4d,
            coords={'time': time_coords, 'level': level_coords[:-1], 'latitude': lat_coords, 'longitude': lon_coords},
            dims=['time', 'level', 'latitude', 'longitude'],
            name='AAM'
        )
    except Exception as e:
        import pdb; pdb.set_trace()

    aam_da.attrs['long_name'] = 'atmospheric_angular_momentum'
    aam_da.attrs['units'] = 'kg m**2 s**-1'
    aam_da.attrs['description'] = 'AAM at each pressure level, lat, lon and time since 1850'
    aam_da.attrs['ensemble_member'] = ensemble_member
    aam_da.attrs['model'] = 'CMIP6 HadGEM3-GC31'

    # Set encoding to properly handle NaN values in output
    encoding = {
        'AAM': {
            '_FillValue': np.nan,
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4
        }
    }

    # Format time for filename (handle cftime objects with non-standard calendars)
    time_start_str = f"{time_coords[0].year:04d}-{time_coords[0].month:02d}"
    time_end_str = f"{time_coords[-1].year:04d}-{time_coords[-1].month:02d}"

    out_fname = f"{save_path}/AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_{time_start_str}_{time_end_str}.nc"
    aam_da.to_dataset(name='AAM').to_netcdf(out_fname, encoding=encoding)
    print("Wrote", out_fname)
    print(f"  Encoding: _FillValue=NaN, dtype=float32, compressed")

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")