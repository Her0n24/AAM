
"""
This script computes the monthly climatology of the full AAM field (not zonal mean) for a given ensemble member and period, and stores it as a netCDF file.
"""
import numpy as np
import xarray as xr
import os
import argparse
import tqdm

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
CMIP6_path_base = "/work/scratch-nopw2/hhhn2/HadGEM3-GC31-LL"
monthly_mean_dir = os.path.join(CMIP6_path_base, "AAM", "full")
output_dir = os.path.join(CMIP6_path_base, "AAM", "climatology")
os.makedirs(output_dir, exist_ok=True)

def make_fullfield_climatology(start_year, end_year, member):
    """
    Compute monthly climatology for the full AAM field for a given ensemble member.
    """
    # Find the input file
    input_file = os.path.join(
        monthly_mean_dir,
        f"AAM_CMIP6_HadGEM3_GC31_{member}_1850-01_2014-12.nc"
    )
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"AAM file not found: {input_file}")
    ds = xr.open_dataset(input_file)
    if "AAM" not in ds.data_vars:
        raise KeyError("AAM variable not found in dataset")
    da = ds["AAM"]
    # Restrict to years in range
    years = xr.DataArray(da["time.year"].values, dims="time")
    mask = (years >= start_year) & (years < end_year)
    da = da.isel(time=mask)
    # Compute monthly climatology
    clim = da.groupby("time.month").mean("time", skipna=True)
    clim.attrs = da.attrs
    clim.attrs["climatology_years"] = f"{start_year}-{end_year}"
    # Save
    output_file = os.path.join(
        output_dir,
        f"AAM_Climatology_CMIP6_HadGEM3_GC31_{member}_{start_year}-{end_year}.nc"
    )
    clim.to_dataset(name="AAM").to_netcdf(output_file)
    print(f"Climatology saved to: {output_file}")
    print(f"Shape: {clim.shape}, Dims: {clim.dims}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make full-field AAM climatology for a member over a year range")
    parser.add_argument("--member", default=None, help="ensemble member (e.g. r1i1p1f3)")
    parser.add_argument("--start", "-s", type=int, default=1980, help="start year (default: 1980)")
    parser.add_argument("--end", "-e", type=int, default=2000, help="end year (default: 2000)")
    args = parser.parse_args()
    
    if args.member is None:
        for ensemble_member in tqdm.tqdm(np.arange(6,61), desc="Processing ensemble members"):
            member_str = f"r{ensemble_member}i1p1f3"
            try:
                make_fullfield_climatology(args.start, args.end, member_str)
            except FileNotFoundError as e:
                print(f"Skipping Member {member_str}: {e}")
                pass
    else:
        make_fullfield_climatology(args.start, args.end, args.member)
