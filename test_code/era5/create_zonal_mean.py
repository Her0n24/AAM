"""
The code obtain ERA5/ ERA-Interim modelled level output files from share folder and computes monthly zonal mean of some physical properties. 
Multiprocessing is introduced to use multiple cores on Lotus to speed up computation.
The variables we're looking for are surface pressure (ps)/ natural log of surface pressure (lnsp) and zonal wind (u).
The zonal mean .nc files are then saved to local folder for later use in computing Atmospheric Angular Momentum (AAM).
"""
import time
time_start = time.time()

import numpy as np
import os
import xarray as xr
import tqdm
import argparse
import sys
import gc

base_path = os.getcwd()
ear5_path_base = "/badc/ecmwf-era5/data/oper/an_ml/" #an_ml for analysis on model levels
# an_sfc for analysis on surface levels
save_path = f"{base_path}/AAM_data/monthly_mean/"
os.makedirs(save_path, exist_ok=True)
# base path hhhn2@sci-vm-01:/badc/ecmwf-era5/data/oper/an_ml/2022/01/01$
# full file name ecmwf-era5_oper_an_ml_197901011100.u.nc
# ecmwf-era5_oper_an_ml_197901011100.lnsp.nc
# two variables needed: u (eastward_wind) and lnsp ln(surface_air_pressure)

def get_available_years_months(base=ear5_path_base):
    """Scan ear5_path_base for available years and months."""
    years = {}
    if not os.path.isdir(base):
        return years
    for year_dir in sorted(os.listdir(base)):
        year_path = os.path.join(base, year_dir)
        if os.path.isdir(year_path) and year_dir.isdigit():
            months = []
            for month_dir in sorted(os.listdir(year_path)):
                if month_dir.isdigit():
                    months.append(int(month_dir))
            if months:
                years[year_dir] = months
    return years

def compute_zonal_mean_from_nc(year, month, variable):
    """
    Load a single variable for a given year/month, compute monthly zonal mean,
    and write to .nc file.
    """
    if month < 1 or month > 12:
        print(f"Invalid month: {month}")
        return

    if variable == 'u':
        glob_pattern = f"{ear5_path_base}{year}/{str(month).zfill(2)}/*/*.u.nc"
        var_name = 'u'
        output_name = 'u_zonal_mean'
    elif variable == 'lnsp':
        glob_pattern = f"{ear5_path_base}{year}/{str(month).zfill(2)}/*/*.lnsp.nc"
        var_name = 'lnsp'
        output_name = 'surface_pressure_zonal_mean'
    
    try:
        ds = xr.open_mfdataset(glob_pattern, combine='by_coords',
        parallel=False # Only useful when requesting 2 or more cores in LOTUS
        )
    except Exception as e:
        print(f"Error loading {variable} for {year}/{month:02d}: {e}")
        return
    
    print(f"Processing {variable} for {year}/{month:02d}")

    # monthly mean and zonal mean
    ds_mm = ds.resample(time='1MS').mean(skipna=True).mean(dim='longitude', skipna=True)
    ds.close()
    del ds
    gc.collect()

    # detect variable name if different
    if var_name not in ds_mm.data_vars:
        var_name = list(ds_mm.data_vars)[0]
    
    print(f"Variable found: {var_name}")

    times = ds_mm['time'].values
    print(f"Number of times: {len(times)}", flush=True)
    
    for t in times:
        tstr = np.datetime_as_string(t, unit='M')  # YYYY-MM
        data_t = ds_mm.sel(time=t)[var_name]
        print(f"Processing time: {tstr}", flush=True)
        # Convert lnsp to pressure 
        if variable == 'lnsp':
            data_t = np.exp(data_t)
        
        # write to file
        fname = os.path.join(save_path, f"zonal_mean_ERA5_{variable}_{tstr}.nc")
        ds_out = xr.Dataset({output_name: data_t})
        ds_out.to_netcdf(fname)
        print(f"Wrote {fname}")
        del ds_out
    
    ds_mm.close()
    del ds_mm
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ERA5 zonal means. If no year/ month specified, process all available years/ months.')
    parser.add_argument('--year', type=int, help='Year to process')
    parser.add_argument('--month', type=int, help='Month to process')
    args = parser.parse_args()

    year = args.year
    month = str(args.month).zfill(2)  # Convert to string first, then pad with zeros 
    
    if args.year and args.month:
        compute_zonal_mean_from_nc(year, args.month, 'u')  # Pass integer month
        compute_zonal_mean_from_nc(year, args.month, 'lnsp')  # Pass integer monthnsp')  # Pass integer month
    else:
        avail = get_available_years_months()
        disable_tqdm = not sys.stdout.isatty()
        for variable in ['u', 'lnsp']:
            for year in tqdm.tqdm(avail, disable=disable_tqdm):
                for month in avail[year]:
                    compute_zonal_mean_from_nc(year, month, variable)

    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")