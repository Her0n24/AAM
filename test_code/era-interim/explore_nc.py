# %%
import xarray as xr

nc_base_path = "/badc/ecmwf-era-interim/data/monthly-means/gg/as/"
year = 1980

ds = xr.open_mfdataset(f'{nc_base_path}{year}/*.nc', combine='by_coords')
# %%
# For ERA-Interim, surface pressure is stored in surface levels, 
# U winds are stored at pressure/ modelled levels
