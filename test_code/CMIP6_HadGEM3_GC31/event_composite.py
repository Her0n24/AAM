"""


Reference 
Hardiman et al., 2025
https://doi.org/10.1038/s41612-025-01283-7
"""
import xarray as xr
import numpy as np
import json
# Allow importing shared utilities from AAM/test_code
import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utilities import _to_per_latitude_band, _reindex_to_climatology_dims, vertical_sum_over_pressure_range, get_ENSO_index

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=0.0, help='Minimum pressure level (hPa) to include (default: 100 hPa)')
parser.add_argument('--p-max', type=float, default=1020, help='Maximum pressure level (hPa) to include (default: 1000 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year to plot (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year to plot (default: 2000)')
parser.add_argument('--member', type=str, default='1', help='Ensemble member to plot (default: 1, control)')
args = parser.parse_args()

base_dir = os.getcwd()
AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"
output_dir = f"{base_dir}/figures/"

climatology_path_base = f"{base_dir}/climatology/"
CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul"
nino34_directory = f"{CMIP6_path_base}/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/"
output_dir = f"{base_dir}/figures/"


# 1. Calculate deviation (in AAM, or U) from annual mean for each poleward propagating year. Can be for each month
# or for each season or rolling season
# 2. Plot anomalies for multi-year (composite) 
# 3. Repeat but for this time the criteria is El Niño years (ENSO3.4 > threshold)
# 4. What about non-propagating years?
# 5. Propagating characteristics: speed, amplitude, height, latitude extent. How should we define propagation? 
# Do we set up a hard threshold? Is this robust and fair ? 

# Strategies: Given El Niño event, look for northward propagation?
# or Northward propagation regardless of El Niño/ La Niña and seasons
# Can also constraint base on propagation ceased, no consistent signal, hemispheric symmetry (prior knowledge/ assumption?)

# If (intermittancy) criteria has been satisfied, do a fit line to the maximum values across latitude in time.


def detect_poleward_propagation_time(da, clim_da, start_yr, end_yr, sym_constraint, intermittency_constraint, winter_constraint, el_nino_constraint, anomaly_thres: float = 0.0) -> dict:
    """
    Given a dataarray of zonally integrated AAM in latitude and time, detect poleward propagating events in the NH.
    Return a list of time period that satisfy the criteria for poleward propagation in the NH for each event.
    
    Parameters:
    - dataarray: xarray DataArray of zonally integrated AAM (time, level, latitude, longitude)
    - clim_dataarray: xarray DataArray of climatological AAM (level, latitude, month)
    - symmetry_constraint: boolean, constraint to events where poleward propagation in the SH 
    can also be found in the similar time frame
    - intermittency_constraint: int, the month tolerance for intermitting events. 0 for the method used in
    Hardiman et al., 2025, which requires the sign of the poleward propagating AAM to be consistent across all months.
    - winter_start_constraint: boolean, constraint the start of the event to be in winter (DJF)
    - el_nino_constraint: boolean, constraint to events where El Niño is 
    active at the time (Nino 3.4 is above 0.5 for all months in DJF)
    - anomaly_threshold: float, the threshold for a positive AAM anomaly to be considered. 
    Default is 0 following Hardiman et al., 2025
    
    Return:
    - dict: a dictionary containing the centre position (event_no, time, latitude, *params) of the anomaly, speed, mean amplitude, and latitude extent for each event.
    """
    # limit to equator to 60 degrees in both hemispheres
    da = da.sel(latitude=slice(-60, 60), time=slice(f"{start_yr}-01", f"{end_yr -1}-12"))
    da = da['AAM'] if isinstance(da, xr.Dataset) else da
    clim_da = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) else clim_da

    
    #reduce da to zonally and vertically integrated first (tiem, latitude)
    da = _to_per_latitude_band(da)
    
    da = vertical_sum_over_pressure_range(da, p_min_hpa = 0.0, p_max_hpa= 1020, level_dim = 'level')
    clim_da = vertical_sum_over_pressure_range(clim_da, p_min_hpa=0.0, p_max_hpa=1020, level_dim="level")
    
    da, clim_on_time = _reindex_to_climatology_dims(da, clim_da)

    anomaly = da - clim_on_time

    # Look for any onset above anomaly_threshold within the tropics (10S to 10N)
    tropics = anomaly.sel(latitude=slice(-10, 10))
    
    # Cosine-lat weighted mean over latitude
    w = xr.DataArray(
    np.cos(np.deg2rad(tropics["latitude"].values)),
    coords={"latitude": tropics["latitude"]},
    dims=("latitude"))
    
    # 1D tropical index: number for each time step that summarises how strong the anomaly is in the tropics
    I_wmean = tropics.weighted(w).mean("latitude")  # dims: time

    # Threshold exceedance mask
    T = anomaly_thres
    above = I_wmean > T 
    
       
    # Track the northward propagation by argmax at each timestep. 
    # Enforce monotonicity in the final position of the anomalies centre (i.e., the centre of the anomaly cannot move southward in time).
    # Use centre of mass approach. Brief southward movement can be noise. Don't reject the event. Instead, use a rolling window to 
    # access the movement of the anomalies.
    def centre_of_mass_latitude(
    anom_event: xr.DataArray,
    *,
    threshold: float = 0.0,
    use_coslat_area_weight: bool = True,
    ) -> xr.DataArray:
        """
        Returns: com_lat(time) in degrees.
        """
        # limit to NH
        sub = anom_event.sel(latitude=slice(0,))

        # keep only positive anomalies above threshold (this avoids negative tails pulling COM)
        A = sub.where(sub > threshold).fillna(0.0)

        # latitude weights
        lat = sub["latitude"]
        if use_coslat_area_weight:
            w = xr.DataArray(
                np.cos(np.deg2rad(lat.values)),
                coords={"latitude": lat},
                dims=("latitude",),
            )
        else:
            w = xr.ones_like(lat)

        # COM latitude: sum(lat * A * w) / sum(A * w)
        num = (A * lat * w).sum("latitude")
        den = (A * w).sum("latitude")

        com_lat = num / den
        com_lat = com_lat.where(den > 0)  # avoid 0/0 when no positive anomaly
        com_mono_tone = com_lat.copy()
        
        # For valid segment, enforce non-decreasing:
        com_mono_tone.values= np.maximum.accumulate(com_lat.values)
        com_mono_tone.name = "com_latitude"
        return com_mono_tone
    
    def _label_events(flag: np.ndarray, *, max_gap: int = intermittency_constraint, min_len: int = 3) -> np.ndarray:
        """
        Returns a 1D NumPy array of integers with length = number of time steps. Each 1 to n integer is a separate event identifier. 0 means not part of any events.
        Parameters:
        - flag: boolean array of shape (time,) indicating where the condition is met (e.g., anomaly above threshold).
        - max_gap: maximum number of consecutive False values allowed within an event. This is the "intermittency constraint". 
        For example, if max_gap=1, then a single month below threshold within an otherwise continuous event would still be considered part of that event. 
        - min_len: minimum number of consecutive True values required to consider a sequence as an event. This is to filter out very short events that may not be meaningful.
        """
        
        flag = np.asarray(flag, dtype=bool)

        # bridge gaps up to max_gap
        if max_gap > 0:
            bridged = flag.copy()
            true_idx = np.where(flag)[0]
            for i in range(true_idx.size - 1):
                a, b = true_idx[i], true_idx[i + 1]
                gap = (b - a) - 1
                if 0 < gap <= max_gap:
                    bridged[a:b + 1] = True
            flag = bridged

        labels = np.zeros(flag.size, dtype=int)
        eid = 0
        i = 0
        while i < flag.size:
            if not flag[i]:
                i += 1
                continue
            j = i
            while j < flag.size and flag[j]:
                j += 1
            if (j - i) >= min_len:
                eid += 1
                labels[i:j] = eid
            i = j
        return labels

    
    event_id = xr.DataArray(
        _label_events(above.values, max_gap=intermittency_constraint, min_len=3),
        coords={"time": anomaly["time"]},
        dims=("time",),
    )
    #import pdb; pdb.set_trace()
    
    # Enforce El_nino_constraint 
    djf_all_above = None
    if el_nino_constraint:
        enso_times, enso_vals = get_ENSO_index(start_yr, end_yr - 1)
        ENSO_da = xr.DataArray(enso_vals, coords={"time": enso_times}, dims=("time",))
        
        # Check for each event, if the onset year DJF has a value above 0.5 for that winter
        month = ENSO_da["time"].dt.month
        year = ENSO_da["time"].dt.year
        winter_year = xr.where(month == 12, year + 1, year)
        
        djf = ENSO_da.where(month.isin([12, 1, 2]), drop=True)
        djf_winter_year = winter_year.sel(time=djf["time"])
        djf = djf.assign_coords(winter_year = djf_winter_year)
        # For each winter, require ALL DJF months > threshold:
        djf_all_above = (djf > 0.5).groupby(djf_winter_year).all(dim="time")
        
    results = {}
    
    for eid in np.unique(event_id.values):
        if eid == 0: # non event
            continue
        
        mask = (event_id == eid)
        anom_evt = anomaly.where(mask, drop=True)   # keeps only event times
        
        onset = anom_evt["time"].values[0]          # first month in the event
        
        # Enforce winter start constraint, (onset must be in DJF)
        if winter_constraint and onset.month not in (12, 1, 2):
            continue
        
        if el_nino_constraint:
            onset_month = onset.month
            onset_year = onset.year
            
            # same winter-year convention
            evt_winter_year = onset_year + 1 if onset_month == 12 else onset_year     
            is_el_nino = bool(djf_all_above.sel(winter_year=evt_winter_year))
            
            if not is_el_nino:
                continue
        
        # passed constraints -> now track
        com = centre_of_mass_latitude(anom_evt, threshold=anomaly_thres)  # dims: time, values=lat
        amp_at_com = anom_evt.interp(latitude=com)                         # dims: time

        # speed (deg per time-step; deg/month if monthly)
        tt = np.arange(com.size, dtype=float)
        ok = np.isfinite(com.values)
        slope, intercept = np.polyfit(tt[ok], com.values[ok], 1)

        # poleward extent based on threshold edge (NH)
        nh = anom_evt.sel(latitude=slice(0, None))
        lat_where = xr.where(nh > anomaly_thres, nh["latitude"], np.nan)
        poleward_edge = lat_where.max("latitude", skipna=True)             # dims: time
        max_lat_extent_threshold = float(poleward_edge.max("time", skipna=True))

        results[int(eid)] = {
            "anomaly": anom_evt,                         # time x latitude
            "com_latitude": com,                         # time
            "amp_at_com": amp_at_com,                    # time
            "speed_deg_per_month": float(slope),
            "mean_amp_at_com": float(amp_at_com.mean("time", skipna=True)),
            "max_amp_at_com": float(amp_at_com.max("time", skipna=True)),
            "max_lat_reached_com": float(com.max("time", skipna=True)),
            "max_lat_extent_threshold": max_lat_extent_threshold,
        }
        
    return results


# def track_poleward_propagation() -> dict:
#     """
#     Given a dataarray of zonally integrated AAM in latitude and time, and the list that contains 
#     the poleward propagating years, track the poleward propagation of events.
#     The function then tracks the maximum over a period of time, then conduct a linear fit to the maximum values across latitude in time.
#     The centre of the maximums cannot go backward in time, hence the slope of the fitted line must be strictly positive.
#     The function then return a dictionary containing the centre position of the anomaly, speed, mean amplitude, height and latitude extent for each event.

#     Parameters:
#     - dataarray: xarray DataArray of zonally integrated AAM (time, level, latitude, longitude)
#     - poleward_propagating_times: list of initial times that satisfy the criteria for poleward propagation

    
#     """
#     return {}

if __name__ == '__main__':
    
    clim_start_yr = 1980
    clim_end_yr = 2000
    ensemble_member = f"r{args.member}i1p1f3"
    
    # Load data
    AAM_da = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc"
    )['AAM']  
    # IMPORTANT: use the per-latitude-band + zonal-integral climatology (matches _to_per_latitude_band convention)
    clim_kind = "latband_lonint"
    clim_da = xr.open_dataset(
        f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}_{clim_kind}.nc"
    )  # dims: month, level, latitude
    results = detect_poleward_propagation_time(
        AAM_da,
        clim_da,
        start_yr=args.start_year,
        end_yr=args.end_year,
        sym_constraint=False,
        intermittency_constraint=0,
        winter_constraint=False,
        el_nino_constraint=False,
        anomaly_thres=0.0
    )
    print(results)
    # Store it as json output
    sym_str = "_sym_" if sym_constraint else "" 
    interm_str = "_int_" if intermittency_constraint else ""
    win_str = "_win_" if winter_constraint else ""
    nino_str = "_nino_" if el_nino_constraint else ""
    
    with open(f"AAM_events_results_{args.start_year}_{args.end_year}.json", "w") as f:
        json.dump(results, f)