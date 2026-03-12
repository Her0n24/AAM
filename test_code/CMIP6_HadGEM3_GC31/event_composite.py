"""


Reference 
Hardiman et al., 2025
https://doi.org/10.1038/s41612-025-01283-7
"""
import xarray as xr
import numpy as np
import json
from typing import Optional
# Allow importing shared utilities from AAM/test_code
import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utilities import _to_per_latitude_band, _reindex_to_climatology_dims, vertical_sum_over_pressure_range, get_ENSO_index

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=150.0, help='Minimum pressure level (hPa) to include (default: 0 hPa)')
parser.add_argument('--p-max', type=float, default=700, help='Maximum pressure level (hPa) to include (default: 1020 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year to plot (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year to plot (default: 2000)')
parser.add_argument('--member', type=str, default='1', help='Ensemble member to plot (default: 1, control)')
parser.add_argument('--json-out', type=str, default='', help='Optional path to write a metrics-only JSON summary')
parser.add_argument('--intermittency', type=int, default=9, help='Month tolerance for terminating an event when COM latitude is stuck (default: 6)')
parser.add_argument('--gap-bridge', type=int, default=3, help='Max consecutive months allowed below threshold when labeling events (0 disables, 1-2 recommended)')
parser.add_argument('--tropics-band', type=float, default=20.0, help='Latitude band (±deg) used to compute the tropical detection index')
parser.add_argument('--onset-com-lat-max', type=float, default=20.0, help='Reject events whose onset COM latitude exceeds this (deg N); default 10')
parser.add_argument('--max-com-jump', type=float, default=50.0, help='Optional: reject events with any month-to-month COM jump above this (deg)')
parser.add_argument('--rolling-smooth', type=int, default=1, help='Rolling window size (months) for smoothing COM trajectory (default: 3; use 1 to disable)')
parser.add_argument('--tracker', type=str, default='dp', choices=['com', 'dp', 'argmax'], help='Latitude tracker to use: com, dp, or argmax (default: dp)')
parser.add_argument('--argmax-continuity-deg', type=float, default=20.0, help='Max latitude jump (deg) for argmax tracker continuity constraint (default: 20)')
parser.add_argument('--dp-jump-penalty', type=float, default=0.01, help='DP penalty per degree of month-to-month latitude jump (default: 0.01)')
parser.add_argument('--dp-south-penalty', type=float, default=2.0, help='DP extra penalty per degree for southward moves (default: 2.0)')
parser.add_argument('--dp-max-step-deg', type=float, default=30.0, help='DP hard cap on monthly latitude step in degrees (default: 15)')
parser.add_argument('--dp-scale-quantile', type=float, default=0.5, help='Quantile of positive anomaly used for DP emission scaling A0 (default: 0.5)')
parser.add_argument('--track-require-tropical', action='store_true', help='If set, apply tropical threshold gating during tracking (default: off; onset-only gating)')
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


def detect_poleward_propagation_time(
    da,
    clim_da,
    start_yr,
    end_yr,
    sym_constraint,
    intermittency_constraint,
    winter_constraint,
    el_nino_constraint,
    anomaly_thres: float = 0.0,
    *,
    p_min_hpa: float = 0.0,
    p_max_hpa: float = 1020.0,
    tropics_band_deg: float = 10.0,
    onset_com_lat_max_deg=None,
    max_com_jump_deg=None,
    gap_bridge_months=None,
    rolling_smooth_months: int = 3,
    tracker: str = 'dp',
    dp_jump_penalty: float = 0.01,
    dp_south_penalty: float = 0.03,
    dp_max_step_deg: float = 15.0,
    dp_scale_quantile: float = 0.5,
    track_require_tropical: bool = False,
) -> dict:
    """
    Given a dataarray of zonally integrated AAM in latitude and time, detect poleward propagating events in the NH.
    Return a list of time period that satisfy the criteria for poleward propagation in the NH for each event.
    
    Parameters:
    - dataarray: xarray DataArray of zonally integrated AAM (time, level, latitude, longitude)
    - clim_dataarray: xarray DataArray of climatological AAM (level, latitude, month)
    - symmetry_constraint: boolean, constraint to events where poleward propagation in the SH 
    can also be found in the similar time frame
        - intermittency_constraint: int, month tolerance for terminating an event when the COM latitude becomes
            stuck (repeats for too long).
        - gap_bridge_months: int, maximum number of consecutive months allowed below the detection threshold
            when labeling events. Use 0 to disable gap-bridging.
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
    
    da = vertical_sum_over_pressure_range(da, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim='level')
    clim_da = vertical_sum_over_pressure_range(clim_da, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level")
    
    da, clim_on_time = _reindex_to_climatology_dims(da, clim_da)

    anomaly = da - clim_on_time

    # Look for onset above anomaly_threshold within the tropics (±tropics_band_deg)
    tropics = anomaly.sel(latitude=slice(-float(tropics_band_deg), float(tropics_band_deg)))
    
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
    enforce_monotonic: bool = True,
    rolling_window: int = 3,
    ) -> xr.DataArray:
        """
        Returns: com_lat(time) in degrees.
        
        Parameters:
        - rolling_window: number of months for rolling mean smoothing (default: 3).
                         Set to 1 to disable smoothing.
        """
        # limit to NH (order-independent; .sel(slice(0, ...)) can select SH if latitude is descending)
        lat0 = anom_event["latitude"]
        sub = anom_event.where(lat0 >= 0, drop=True)

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
        
        # Apply rolling mean to smooth out noise
        if rolling_window > 1:
            # Use pandas rolling with min_periods=1 so we don't lose edge values
            com_series = com_lat.to_series()
            com_smoothed = com_series.rolling(
                window=rolling_window,
                center=True,
                min_periods=1
            ).mean()
            com_lat = xr.DataArray(
                com_smoothed.values,
                coords=com_lat.coords,
                dims=com_lat.dims
            )
        
        if enforce_monotonic:
            # Enforce non-decreasing, but ignore NaNs (e.g., months we purposely
            # mask out because the tropical detection index is below threshold).
            # NaNs remain NaNs in the output and do not poison subsequent values.
            vals = np.asarray(com_lat.values, dtype=float)
            out = vals.copy()
            last = np.nan
            for i in range(vals.size):
                v = vals[i]
                if not np.isfinite(v):
                    out[i] = np.nan
                    continue
                if np.isfinite(last):
                    last = max(last, v)
                else:
                    last = v
                out[i] = last

            com_mono_tone = com_lat.copy(data=out)
            com_mono_tone.name = "com_latitude"
            return com_mono_tone

        com_lat.name = "com_latitude"
        return com_lat

    def dp_track_latitude(
        anom_event: xr.DataArray,
        *,
        threshold: float = 0.0,
        jump_penalty: float = 0.01,
        south_penalty: float = 0.03,
        max_step_deg: float = 15.0,
        scale_quantile: float = 0.5,
        min_latitude: Optional[float] = None,
        start_latitude: Optional[float] = None,
        start_tolerance_deg: float = 2.5,
    ) -> xr.DataArray:
        """Track a single latitude path through time using dynamic programming.

        Objective per path y_t:
        sum_t emission(t, y_t) - sum_t transition(y_{t-1}, y_t)
        
        Parameters:
        -----------
        min_latitude : float, optional
            Minimum allowed latitude (deg N). Prevents tracking from going south
            of this threshold. Typically set to onset latitude to enforce monotonic
            poleward propagation.
        start_latitude : float, optional
            Latitude anchor for t0. When provided, the DP path is forced to start
            near this latitude (within start_tolerance_deg).
        """
        lat0 = anom_event["latitude"]
        sub = anom_event.where(lat0 >= 0, drop=True)
        
        # Apply minimum latitude constraint: drop latitudes below threshold
        if min_latitude is not None and np.isfinite(min_latitude):
            sub = sub.where(sub["latitude"] >= float(min_latitude), drop=True)
        
        lat = np.asarray(sub["latitude"].values, dtype=float)

        A = sub.where(sub > threshold).fillna(0.0)
        Avals = np.asarray(A.values, dtype=float)  # (time, lat)
        nt, nl = Avals.shape

        out = np.full(nt, np.nan, dtype=float)
        if nt == 0 or nl == 0:
            trk = sub.isel(latitude=0).copy(data=out) if nt > 0 else xr.DataArray(out, dims=("time",), coords={"time": sub["time"]})
            trk.name = "com_latitude"
            return trk

        # Emission scaling A0 from positive values
        pos = Avals[Avals > 0.0]
        if pos.size == 0:
            trk = sub.isel(latitude=0).copy(data=out)
            trk.name = "com_latitude"
            return trk

        q = float(np.clip(scale_quantile, 0.05, 0.95))
        A0 = float(np.nanquantile(pos, q))
        if not np.isfinite(A0) or A0 <= 0.0:
            A0 = float(np.nanmedian(pos)) if pos.size else 1.0
        if not np.isfinite(A0) or A0 <= 0.0:
            A0 = 1.0

        # emission reward: stronger positive anomalies get larger reward
        E = np.log1p(np.maximum(Avals, 0.0) / A0)

        # transition penalty matrix P[i, j] from lat i -> lat j
        lat_i = lat[:, None]
        lat_j = lat[None, :]
        d = np.abs(lat_j - lat_i)
        south = np.maximum(0.0, lat_i - lat_j)
        P = float(jump_penalty) * d + float(south_penalty) * south
        if max_step_deg is not None and float(max_step_deg) > 0.0:
            P = np.where(d > float(max_step_deg), 1.0e12, P)

        V = np.full((nt, nl), -np.inf, dtype=float)
        B = np.zeros((nt, nl), dtype=np.int32)
        V[0, :] = E[0, :]

        # Anchor the first month near the detected onset latitude so trajectories
        # don't artificially begin far north due to a stronger mid-latitude lobe.
        if start_latitude is not None and np.isfinite(start_latitude):
            d0 = np.abs(lat - float(start_latitude))
            tol = max(float(start_tolerance_deg), 0.0)
            allowed = d0 <= tol
            if np.any(allowed):
                V[0, ~allowed] = -np.inf
            else:
                k = int(np.argmin(d0))
                mask0 = np.ones(nl, dtype=bool)
                mask0[k] = False
                V[0, mask0] = -np.inf

        for t in range(1, nt):
            # score[i, j] = V[t-1, i] - P[i, j]
            score = V[t - 1, :][:, None] - P
            best_prev = np.argmax(score, axis=0)
            best_score = score[best_prev, np.arange(nl)]
            V[t, :] = E[t, :] + best_score
            B[t, :] = best_prev

        # backtrack best terminal latitude
        j = int(np.argmax(V[-1, :]))
        idx = np.empty(nt, dtype=np.int32)
        idx[-1] = j
        for t in range(nt - 1, 0, -1):
            idx[t - 1] = B[t, idx[t]]

        # Emit NaN on months with no positive NH anomaly to avoid fake positions
        has_pos = np.any(Avals > 0.0, axis=1)
        out = lat[idx].astype(float)
        if start_latitude is not None and np.isfinite(start_latitude) and out.size > 0:
            # Report the true detected onset latitude for t0 rather than snapping
            # to the discrete latitude grid, which biases starts poleward.
            out[0] = float(start_latitude)
        out[~has_pos] = np.nan

        trk = sub.isel(latitude=0).copy(data=out)
        trk.name = "com_latitude"
        return trk
    
    def argmax_track_latitude(
        anom_event: xr.DataArray,
        *,
        threshold: float = 0.0,
        continuity_deg: float = 20.0,
        start_latitude: Optional[float] = None,
    ) -> xr.DataArray:
        """Track latitude of maximum anomaly at each timestep with continuity constraint.
        
        Simple algorithm:
        1. Find argmax of positive anomalies in NH at each timestep
        2. Constrain argmax to be within continuity_deg of previous timestep
        3. Returns latitude trajectory suitable for linear fit
        
        Parameters:
        -----------
        threshold : float
            Minimum anomaly value to consider
        continuity_deg : float
            Maximum latitude jump allowed between consecutive timesteps (degrees)
        start_latitude : float, optional
            Starting latitude anchor for first timestep
        """
        lat0 = anom_event["latitude"]
        sub = anom_event.where(lat0 >= 0, drop=True)  # NH only
        
        # Keep only positive anomalies above threshold
        A = sub.where(sub > threshold)
        lat = np.asarray(sub["latitude"].values, dtype=float)
        Avals = np.asarray(A.values, dtype=float)  # (time, lat)
        nt, nl = Avals.shape
        
        out = np.full(nt, np.nan, dtype=float)
        if nt == 0 or nl == 0:
            trk = sub.isel(latitude=0).copy(data=out) if nt > 0 else xr.DataArray(
                out, dims=("time",), coords={"time": sub["time"]}
            )
            trk.name = "com_latitude"
            return trk
        
        prev_lat = None
        if start_latitude is not None and np.isfinite(start_latitude):
            prev_lat = float(start_latitude)
        
        for t in range(nt):
            row = Avals[t, :]
            finite = np.isfinite(row)
            
            if not np.any(finite):
                # No valid data at this timestep
                out[t] = np.nan
                continue
            
            # Find all valid latitudes with positive anomalies
            valid_idx = np.where(finite)[0]
            
            if prev_lat is None:
                # First timestep: find global maximum
                max_idx = valid_idx[np.argmax(row[valid_idx])]
                out[t] = lat[max_idx]
                prev_lat = out[t]
            else:
                # Subsequent timesteps: find max within continuity constraint
                lat_dist = np.abs(lat[valid_idx] - prev_lat)
                within_range = lat_dist <= continuity_deg
                
                if np.any(within_range):
                    # Find max among points within continuity constraint
                    constrained_idx = valid_idx[within_range]
                    max_idx = constrained_idx[np.argmax(row[constrained_idx])]
                    out[t] = lat[max_idx]
                    prev_lat = out[t]
                else:
                    # No points within range: find closest point and use that
                    closest_idx = valid_idx[np.argmin(lat_dist)]
                    out[t] = lat[closest_idx]
                    prev_lat = out[t]
        
        trk = sub.isel(latitude=0).copy(data=out)
        trk.name = "com_latitude"
        return trk
    
    def _label_events(flag: np.ndarray, *, max_gap: int, min_len: int = 3) -> np.ndarray:
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

    
    # IMPORTANT: Do not implicitly gap-bridge (which can chain/merge distinct events).
    # Gap-bridging is only enabled when gap_bridge_months is explicitly provided (> 0).
    max_gap = 0 if gap_bridge_months is None else int(gap_bridge_months)

    # Minimum event length (months). This must be enforced both at labeling time
    # and after any trimming/truncation steps.
    min_event_len = 1
    event_id = xr.DataArray(
        _label_events(above.values, max_gap=max_gap, min_len=min_event_len),
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

    # By construction, events are detected from a tropical (±10°) index. Enforce
    # a consistent interpretation: if the caller doesn't specify an onset latitude
    # cap, require onset COM latitude to be within 10°N.
    if onset_com_lat_max_deg is None:
        onset_com_lat_max_deg = 10.0

    def _truncate_on_stuck_latitude(
        com_lat: xr.DataArray,
        *,
        max_stuck_len: int,
        eps: float = 1.0e-6,
    ) -> int:
        """Return number of time steps to keep before latitude is 'stuck' too long.

        A run is considered "stuck" when |com[t] - com[t-1]| <= eps.
        If the run length exceeds `max_stuck_len`, terminate *before* the first
        offending time step.

        Example: max_stuck_len=3 allows [a,a,a] but truncates [a,a,a,a] to 3.
        """
        if max_stuck_len is None:
            return int(com_lat.size)
        max_stuck_len = int(max_stuck_len)
        if max_stuck_len <= 0:
            return int(com_lat.size)
        if com_lat.size <= 1:
            return int(com_lat.size)

        vals = np.asarray(com_lat.values, dtype=float)
        run_len = 1
        for i in range(1, vals.size):
            if not (np.isfinite(vals[i]) and np.isfinite(vals[i - 1])):
                run_len = 1
                continue
            if abs(vals[i] - vals[i - 1]) <= eps:
                run_len += 1
                if run_len > max_stuck_len:
                    return int(i)  # truncate before this index
            else:
                run_len = 1
        return int(vals.size)
    
    for eid in np.unique(event_id.values):
        if eid == 0: # non event
            continue
        
        mask = (event_id == eid)
        anom_evt = anomaly.where(mask, drop=True)   # keeps only event times
        
        onset = anom_evt["time"].values[0]          # first month in the event
        
        print(f"\n=== Processing event {eid}: onset={onset.year}-{onset.month:02d}, length={anom_evt.sizes.get('time')} ===")
        
        # Enforce winter start constraint, (onset must be in DJF)
        if winter_constraint and onset.month not in (12, 1, 2):
            print(f"  ❌ FILTERED: winter_constraint (month={onset.month})")
            continue
        
        if el_nino_constraint:
            if djf_all_above is None:
                raise ValueError("el_nino_constraint=True but ENSO DJF mask was not computed")
            onset_month = onset.month
            onset_year = onset.year
            
            # same winter-year convention
            evt_winter_year = onset_year + 1 if onset_month == 12 else onset_year     
            is_el_nino = bool(djf_all_above.sel(winter_year=evt_winter_year))
            
            if not is_el_nino:
                continue
        
        # passed constraints -> now track
        # 1) Compute a raw (non-monotone, non-smoothed) COM series for onset finding.
        com_raw = centre_of_mass_latitude(
            anom_evt,
            threshold=anomaly_thres,
            enforce_monotonic=False,
            rolling_window=1,  # NO smoothing for onset finding
        )

        # 2) Define the event onset within the labeled segment as the first month
        #    where the tropical index is above threshold AND the COM is within the
        #    onset latitude cap.
        if onset_com_lat_max_deg is not None:
            evt_above = above.where(mask, drop=True)
            vals = np.asarray(com_raw.values, dtype=float)
            ok = np.isfinite(vals)
            in_tropics = vals <= float(onset_com_lat_max_deg)
            valid = np.asarray(evt_above.values, dtype=bool) & ok & in_tropics
            print(f"  Onset COM check: raw_com[0]={vals[0]:.2f}°N, onset_cap={onset_com_lat_max_deg}°N")
            if not np.any(valid):
                print(f"  ❌ FILTERED: no valid onset (COM always > {onset_com_lat_max_deg}°N or below threshold)")
                continue
            onset_i0 = int(np.where(valid)[0][0])
            if onset_i0 > 0:
                print(f"  Trimmed {onset_i0} month(s) from start (onset shifted from idx 0 to {onset_i0})")
                anom_evt = anom_evt.isel(time=slice(onset_i0, None))
                com_raw = com_raw.isel(time=slice(onset_i0, None))
                mask = mask.sel(time=anom_evt["time"])  # keep alignment for any later uses

        # 3) Recompute track on the trimmed event.
        # Design choice: tropical threshold is used for event initiation (onset)
        # and labeling. Tracking itself should follow NH anomalies even when the
        # tropical index temporarily dips. Set --track-require-tropical to recover
        # the previous gated behavior.
        evt_above_track = above.where(mask, drop=True)
        if track_require_tropical:
            anom_evt_track = anom_evt.where(evt_above_track)
        else:
            anom_evt_track = anom_evt
        
        # Get onset latitude to use as floor for DP tracking
        onset_lat = None
        if com_raw.size > 0 and np.isfinite(com_raw.values[0]):
            onset_lat = float(com_raw.values[0])
            print(f"  Setting DP min_latitude={onset_lat:.2f}°N (prevents southward tracking)")
        
        if tracker == 'dp':
            com = dp_track_latitude(
                anom_evt_track,
                threshold=anomaly_thres,
                jump_penalty=float(dp_jump_penalty),
                south_penalty=float(dp_south_penalty),
                max_step_deg=float(dp_max_step_deg),
                scale_quantile=float(dp_scale_quantile),
                min_latitude=onset_lat,
                start_latitude=onset_lat,
            )
        elif tracker == 'argmax':
            com = argmax_track_latitude(
                anom_evt_track,
                threshold=anomaly_thres,
                continuity_deg=float(args.argmax_continuity_deg),
                start_latitude=onset_lat,
            )
        else:
            com = centre_of_mass_latitude(
                anom_evt_track,
                threshold=anomaly_thres,
                enforce_monotonic=True,
                rolling_window=rolling_smooth_months,  # Smoothing applied here for COM tracker
            )

        # If latitude is stuck too long, regard the event as terminated.
        # Reuse intermittency_constraint as the maximum allowed consecutive months
        # with no poleward movement in the COM latitude.
        original_len = int(anom_evt.sizes.get("time", 0))
        keep_n = _truncate_on_stuck_latitude(com, max_stuck_len=intermittency_constraint)
        if keep_n < original_len:
            print(f"  Stuck-lat truncation: {original_len} → {keep_n} months (≤{intermittency_constraint} stuck allowed)")
            anom_evt = anom_evt.isel(time=slice(0, keep_n))
            anom_evt_track = anom_evt_track.isel(time=slice(0, keep_n))
            evt_above_track = evt_above_track.isel(time=slice(0, keep_n))
            com = com.isel(time=slice(0, keep_n))

        # If trimming/truncation makes the event too short, drop it.
        final_len = int(anom_evt.sizes.get("time", 0))
        if final_len < min_event_len:
            print(f"  ❌ FILTERED: too short after trimming ({final_len} < {min_event_len} months)")
            continue

        # Optional quality constraints
        # For onset COM check, use the RAW (unsmoothed) value to avoid bias from smoothing
        if onset_com_lat_max_deg is not None:
            onset_idx = 0
            # Compute raw COM at onset to check against constraint
            com_raw_final = centre_of_mass_latitude(
                anom_evt_track.isel(time=slice(0, 1)),  # Just the onset month
                threshold=anomaly_thres,
                enforce_monotonic=False,
                rolling_window=1,
            )
            onset_com = float(com_raw_final.values[0]) if np.isfinite(com_raw_final.values[0]) else np.nan
            if not np.isfinite(onset_com) or onset_com > float(onset_com_lat_max_deg):
                print(f"  ❌ FILTERED: raw onset COM {onset_com:.2f}°N > {onset_com_lat_max_deg}°N")
                continue

        if max_com_jump_deg is not None and com.size >= 2:
            dv = np.diff(np.asarray(com.values, dtype=float))
            max_jump = float(np.nanmax(dv))
            if np.isfinite(max_jump) and max_jump > float(max_com_jump_deg):
                print(f"  ❌ FILTERED: max COM jump {max_jump:.2f}° > {max_com_jump_deg}°")
                continue
        
        print(f"  ✅ ACCEPTED: {final_len} months, COM: {com.values[0]:.2f}° → {com.max().values:.2f}°N")

        # Compute amplitude at COM consistently with COM definition:
        # - restrict to NH
        # - after interpolation, mask out values below threshold
        nh_evt = anom_evt_track.where(anom_evt_track["latitude"] >= 0, drop=True)
        amp_raw = nh_evt.interp(latitude=com)  # dims: time
        amp_at_com = amp_raw.where(amp_raw > anomaly_thres)

        # speed (deg per time-step; deg/month if monthly)
        # For very short events (<2 valid points), speed is undefined.
        slope = None
        tt = np.arange(com.size, dtype=float)
        ok = np.isfinite(com.values)
        if int(np.sum(ok)) >= 2:
            slope, _intercept = np.polyfit(tt[ok], com.values[ok], 1)
            slope = float(slope)

        # poleward extent based on threshold edge (NH; order-independent)
        nh = anom_evt_track.where(anom_evt_track["latitude"] >= 0, drop=True)
        lat_where = xr.where(nh > anomaly_thres, nh["latitude"], np.nan)
        poleward_edge = lat_where.max("latitude", skipna=True)             # dims: time
        max_lat_extent_threshold = float(poleward_edge.max("time", skipna=True))

        results[int(eid)] = {
            "anomaly": anom_evt,                         # time x latitude
            "com_latitude": com,                         # time
            "amp_at_com": amp_at_com,                    # time
            "speed_deg_per_month": slope,
            "mean_amp_at_com": float(amp_at_com.mean("time", skipna=True)),
            "max_amp_at_com": float(amp_at_com.max("time", skipna=True)),
            "max_lat_reached_com": float(com.max("time", skipna=True)),
            "max_lat_extent_threshold": max_lat_extent_threshold,
        }
        
    return results


def _time_value_to_ymd_string(t) -> str:
    """Convert numpy/cftime-like time scalar to YYYY-MM-DD string."""
    if isinstance(t, np.datetime64):
        return str(np.datetime_as_string(t, unit="D"))
    if hasattr(t, "year") and hasattr(t, "month"):
        day = getattr(t, "day", 1)
        return f"{int(t.year):04d}-{int(t.month):02d}-{int(day):02d}"
    return str(t)


def _fill_nan_latitudes(vals: list[float]) -> list[float]:
    """Fill NaN latitudes so JSON export has no nulls.

    Strategy:
    - forward-fill from previous valid value
    - back-fill any leading NaNs from the first valid value
    - if all are NaN, fill with 0.0 as a safe fallback
    """
    arr = np.asarray(vals, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return [0.0] * int(arr.size)

    # forward fill
    out = arr.copy()
    for i in range(1, out.size):
        if not np.isfinite(out[i]) and np.isfinite(out[i - 1]):
            out[i] = out[i - 1]

    # back fill leading NaNs
    first_valid = int(np.where(np.isfinite(out))[0][0])
    if first_valid > 0:
        out[:first_valid] = out[first_valid]

    # any remaining NaNs (rare) -> nearest previous if available, else first valid
    for i in range(out.size):
        if np.isfinite(out[i]):
            continue
        if i > 0 and np.isfinite(out[i - 1]):
            out[i] = out[i - 1]
        else:
            out[i] = out[first_valid]

    return [float(v) for v in out.tolist()]


def export_results_metrics_json(
    results: dict,
    *,
    json_path: str,
    run_config: dict,
    include_monthly_latitude: bool = True,
) -> None:
    """Write a JSON file with scalar/metadata outputs.

    If `include_monthly_latitude=True`, also export the per-time-step centre-of-mass
    latitude series for each event (aligned to event months).
    """
    events_out: list[dict] = []
    for eid, evt in sorted(results.items(), key=lambda kv: int(kv[0])):
        anom = evt.get("anomaly")
        if isinstance(anom, xr.DataArray) and "time" in anom.coords and anom["time"].size:
            onset = _time_value_to_ymd_string(anom["time"].values[0])
            end = _time_value_to_ymd_string(anom["time"].values[-1])
            event_times = anom["time"].values
        else:
            onset, end = None, None
            event_times = None

        monthly = None
        if include_monthly_latitude:
            com = evt.get("com_latitude")
            if isinstance(com, xr.DataArray) and event_times is not None and com.size:
                # Ensure JSON-serializable types and preserve alignment with event months.
                months = [_time_value_to_ymd_string(t) for t in event_times]
                raw_lats: list[float] = []
                for v in com.values.tolist():
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        fv = np.nan
                    raw_lats.append(fv if np.isfinite(fv) else np.nan)

                # User requested no null latitudes in JSON.
                lats = _fill_nan_latitudes(raw_lats)
                monthly = {"months": months, "com_latitude_deg": lats}

        event_payload = {
            "event_id": int(eid),
            "onset": onset,
            "end": end,
            "speed_deg_per_month": evt.get("speed_deg_per_month"),
            "mean_amp_at_com": evt.get("mean_amp_at_com"),
            "max_amp_at_com": evt.get("max_amp_at_com"),
            "max_lat_reached_com": evt.get("max_lat_reached_com"),
            "max_lat_extent_threshold": evt.get("max_lat_extent_threshold"),
        }
        if monthly is not None:
            event_payload["monthly_com_latitude"] = monthly

        events_out.append(event_payload)

    payload = {"config": run_config, "events": events_out}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


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
        intermittency_constraint=int(args.intermittency),
        winter_constraint=False,
        el_nino_constraint=False,
        anomaly_thres=0.0,
        p_min_hpa=float(args.p_min),
        p_max_hpa=float(args.p_max),
        tropics_band_deg=float(args.tropics_band),
        onset_com_lat_max_deg=args.onset_com_lat_max,
        max_com_jump_deg=args.max_com_jump,
        gap_bridge_months=args.gap_bridge,
        rolling_smooth_months=int(args.rolling_smooth),
        tracker=str(args.tracker),
        dp_jump_penalty=float(args.dp_jump_penalty),
        dp_south_penalty=float(args.dp_south_penalty),
        dp_max_step_deg=float(args.dp_max_step_deg),
        dp_scale_quantile=float(args.dp_scale_quantile),
        track_require_tropical=bool(args.track_require_tropical),
    )

    # Metrics-only JSON export (exclude DataArrays)
    json_out = args.json_out.strip() or f"AAM_event_metrics_{args.start_year}_{args.end_year}.json"
    export_results_metrics_json(
        results,
        json_path=json_out,
        run_config={
            "ensemble_member": ensemble_member,
            "start_year": int(args.start_year),
            "end_year": int(args.end_year),
            "p_min_hpa": float(args.p_min),
            "p_max_hpa": float(args.p_max),
            "anomaly_threshold": float(0.0),
            "intermittency_constraint": int(args.intermittency),
            "gap_bridge_months": (None if args.gap_bridge is None else int(args.gap_bridge)),
            "winter_constraint": bool(False),
            "el_nino_constraint": bool(False),
            "sym_constraint": bool(False),
            "tropics_band_deg": float(args.tropics_band),
            "onset_com_lat_max_deg": args.onset_com_lat_max,
            "max_com_jump_deg": args.max_com_jump,
            "tracker": str(args.tracker),
            "rolling_smooth_months": int(args.rolling_smooth),
            "dp_jump_penalty": float(args.dp_jump_penalty),
            "dp_south_penalty": float(args.dp_south_penalty),
            "dp_max_step_deg": float(args.dp_max_step_deg),
            "dp_scale_quantile": float(args.dp_scale_quantile),
            "track_require_tropical": bool(args.track_require_tropical),
        },
    )
    print(f"Detected {len(results)} events")
    print(f"Wrote metrics-only JSON: {json_out}")