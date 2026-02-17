"""
The code obtains ERA5 model level output files from share folder and computes monthly mean of physical properties. 
This version preserves the full spatial structure (lat, lon, level) - no zonal averaging.
Multiprocessing is introduced to use multiple cores on Lotus to speed up computation.
The variables we're looking for are surface pressure (ps)/ natural log of surface pressure (lnsp) and zonal wind (u).
The monthly mean .nc files are then saved to local folder for later use in computing Atmospheric Angular Momentum (AAM).
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
era5_path_base = "/badc/ecmwf-era5/data/oper/an_ml/" #an_ml for analysis on model levels
# an_sfc for analysis on surface levels
save_path = f"{base_path}/monthly_mean/variables/"
os.makedirs(save_path, exist_ok=True)
# base path hhhn2@sci-vm-01:/badc/ecmwf-era5/data/oper/an_ml/2022/01/01$
# full file name ecmwf-era5_oper_an_ml_197901011100.u.nc
# ecmwf-era5_oper_an_ml_197901011100.lnsp.nc
# two variables needed: u (eastward_wind) and lnsp ln(surface_air_pressure)

def get_available_years_months(base=era5_path_base):
    """Scan era5_path_base for available years and months."""
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

def compute_monthly_mean_from_nc(year, month, variable):
    """
    Load a single variable for a given year/month, compute monthly mean (preserving full spatial structure),
    and write to .nc file.
    """
    if month < 1 or month > 12:
        print(f"Invalid month: {month}")
        return

    if variable == 'u':
        glob_pattern = f"{era5_path_base}{year}/{str(month).zfill(2)}/*/*.u.nc"
        var_name = 'u'
        output_name = 'u'
    elif variable == 'lnsp':
        glob_pattern = f"{era5_path_base}{year}/{str(month).zfill(2)}/*/*.lnsp.nc"
        var_name = 'lnsp'
        output_name = 'sp'  # Will convert to surface pressure
    
    try:
        ds = xr.open_mfdataset(glob_pattern, combine='by_coords',
        parallel=False # Only useful when requesting 2 or more cores in LOTUS
        )
    except Exception as e:
        print(f"Error loading {variable} for {year}/{month:02d}: {e}")
        return
    
    print(f"Processing {variable} for {year}/{month:02d}")

    # monthly mean (NO zonal averaging - preserve lon dimension)
    ds_mm = ds.resample(time='1MS').mean(skipna=True)
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
        
        output_variable = 'sp' if variable == 'lnsp' else variable
        
        # write to file (no "zonal_mean_" prefix)
        fname = os.path.join(save_path, f"ERA5_{output_variable}_{tstr}.nc")
        ds_out = xr.Dataset({output_variable: data_t})
        ds_out.to_netcdf(fname)
        print(f"Wrote {fname}")
        del ds_out
    
    ds_mm.close()
    del ds_mm
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute ERA5 monthly means (full fields, not zonal mean). '
                    'If no year/month specified, process all available years/months.'
    )
    parser.add_argument('--year', type=int, help='Year to process')
    parser.add_argument('--month', type=int, help='Month to process')
    args = parser.parse_args()

    year = args.year
    
    if args.year and args.month:
        compute_monthly_mean_from_nc(year, args.month, 'u')
        compute_monthly_mean_from_nc(year, args.month, 'lnsp')
    else:
        avail = get_available_years_months()
        disable_tqdm = not sys.stdout.isatty()
        for variable in ['u', 'lnsp']:
            for year in tqdm.tqdm(avail, disable=disable_tqdm):
                for month in avail[year]:
                    compute_monthly_mean_from_nc(year, month, variable)

    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
