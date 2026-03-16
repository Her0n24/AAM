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
from plotting_utils import plot_latitude_level_snapshots_HadGEN3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=150.0, help='Minimum pressure level (hPa) to include (default: 0 hPa)')
parser.add_argument('--p-max', type=float, default=700, help='Maximum pressure level (hPa) to include (default: 1020 hPa)')
parser.add_argument('--start-year', type=int, default=1980, help='Start year to plot (default: 1980)')
parser.add_argument('--end-year', type=int, default=2000, help='End year to plot (default: 2000)')
parser.add_argument('--member', type=str, default='1', help='Ensemble member to plot (default: 1, control)')
parser.add_argument('--intermittency', type=int, default=1, help='Months ahead used to validate northward movement at onset; also max stall before event termination (default: 6)')
parser.add_argument('--onset-lat-max', type=float, default=25.0, help='Latitude cap (deg N) for onset argmax search (default: 25)')
parser.add_argument('--max-com-jump', type=float, default=20.0, help='Optional: reject events with any month-to-month COM jump above this (deg)')
parser.add_argument('--argmax-continuity-deg', type=float, default=20.0, help='Max latitude jump (deg) allowed between consecutive timesteps for argmax tracker (default: 20)')
parser.add_argument('--max-southward-jump', type=float, default=5.0, help='Terminate event when argmax latitude drops by more than this (deg) in one month (default: 10)')
parser.add_argument('--winter-constraint', action='store_true', default=False, help='Only accept events whose onset is in NDJFM (default: off)')
parser.add_argument('--el-nino-constraint', action='store_true', default=False, help='Only accept events coinciding with El Nino (Nino3.4 > 0.5 in DJF; default: off)')
parser.add_argument('--sym-constraint', action='store_true', default=False, help='Only accept events with concurrent SH poleward propagation (default: off)')
parser.add_argument('--rolling-period', type=int, default=1, help='Rolling-month window for composite analysis (default: 1 = no rolling; 3 gives DJF/JFM/FMA labels)')
args = parser.parse_args()

base_dir = os.getcwd()
AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"

climatology_path_base = f"{base_dir}/climatology/"
CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul"
nino34_directory = f"{CMIP6_path_base}/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/"
output_dir = f"{base_dir}/figures/composites"


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


def _rolling_mean_time(da: xr.DataArray, window: int) -> xr.DataArray:
    """Centered rolling mean over the time axis for chronological monthly data."""
    window = int(window)
    if window < 1:
        raise ValueError("rolling_period must be >= 1")
    if window == 1:
        return da
    return da.rolling(time=window, center=True, min_periods=window).mean()


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
    onset_lat_max_deg: float = 25.0,
    max_com_jump_deg=None,
    argmax_continuity_deg: float = 20.0,
    max_southward_jump_deg: float = 15.0,
    rolling_period: int = 1,
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
    anomaly = _rolling_mean_time(anomaly, rolling_period)

    def argmax_track_latitude(
        anom_event: xr.DataArray,
        *,
        threshold: float = 0.0,
        continuity_deg: float = 20.0,
        max_southward_jump_deg: float = 10.0,
        start_latitude: Optional[float] = None,
    ) -> xr.DataArray:
        """Track latitude of maximum anomaly at each timestep with an asymmetric window.

        Search window: [prev_lat - max_southward_jump_deg,  prev_lat + continuity_deg]
        This prevents equatorial anomalies from corrupting prev_lat while still
        allowing fast northward propagation.
        """
        lat0 = anom_event["latitude"]
        sub = anom_event.where(lat0 >= 0, drop=True)  # NH only

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
                out[t] = np.nan
                continue

            valid_idx = np.where(finite)[0]

            if prev_lat is None:
                # First timestep: find global maximum in NH
                max_idx = valid_idx[np.argmax(row[valid_idx])]
                out[t] = lat[max_idx]
                prev_lat = out[t]
            else:
                within_range = (
                    (lat[valid_idx] >= prev_lat - max_southward_jump_deg) &
                    (lat[valid_idx] <= prev_lat + continuity_deg)
                )

                if np.any(within_range):
                    constrained_idx = valid_idx[within_range]
                    max_idx = constrained_idx[np.argmax(row[constrained_idx])]
                    out[t] = lat[max_idx]
                    prev_lat = out[t]
                else:
                    # No anomaly within continuity_deg of last position.
                    # Output NaN and hold prev_lat so the search resumes from
                    # the last known position next month (avoids equatorial snapping).
                    out[t] = np.nan

        # Anchor t=0 to start_latitude (true onset, not grid-snapped)
        if start_latitude is not None and np.isfinite(start_latitude) and nt > 0:
            out[0] = float(start_latitude)

        # Linear interpolation over NaN gaps (average of surrounding valid values).
        # Only interior NaNs are filled; leading/trailing NaNs are left as-is so
        # _truncate_on_stuck_latitude can still budget them against intermittency.
        finite_idx = np.where(np.isfinite(out))[0]
        if finite_idx.size >= 2:
            for gap_start in range(finite_idx[0], finite_idx[-1]):
                if np.isfinite(out[gap_start]):
                    continue
                # find the next finite value after this gap
                after = finite_idx[finite_idx > gap_start]
                if after.size == 0:
                    break
                gap_end = int(after[0])
                # linearly interpolate between out[gap_start-1] and out[gap_end]
                v0 = out[gap_start - 1]
                v1 = out[gap_end]
                gap_len = gap_end - (gap_start - 1)
                for k in range(1, gap_len):
                    out[gap_start - 1 + k] = v0 + (v1 - v0) * k / gap_len

        trk = sub.isel(latitude=0).copy(data=out)
        trk.name = "com_latitude"
        return trk
    
    # Pre-compute NH argmax at every timestep for onset scanning and northward validation.
    # Two flavours:
    #   _argmax_capped : only considers latitudes <= onset_lat_max_deg (onset band)
    #   _argmax_free   : full NH (0–60°N) — used to check future position
    _lat_arr = np.asarray(anomaly["latitude"].values, dtype=float)
    _nh_idx  = np.where(_lat_arr >= 0.0)[0]
    _lat_nh  = _lat_arr[_nh_idx]
    _Avals_nh = np.asarray(anomaly.values, dtype=float)[:, _nh_idx]  # (time, lat_nh)
    _nt_full  = _Avals_nh.shape[0]
    time_coords = anomaly["time"].values

    def _argmax_lat(t_idx, lat_max=None):
        row = _Avals_nh[t_idx, :]
        if lat_max is not None:
            cand = (_lat_nh <= lat_max) & np.isfinite(row) & (row > anomaly_thres)
        else:
            cand = np.isfinite(row) & (row > anomaly_thres)
        if not np.any(cand):
            return np.nan
        idx = np.where(cand)[0]
        return float(_lat_nh[idx[np.argmax(row[idx])]])

    _argmax_capped = np.array([_argmax_lat(ti, lat_max=float(onset_lat_max_deg)) for ti in range(_nt_full)])
    _argmax_free   = np.array([_argmax_lat(ti)                                    for ti in range(_nt_full)])

    # Pre-compute SH onset-band argmax for sym_constraint (equatorial SH, mirrors NH capped logic)
    _sh_argmax_capped = None
    if sym_constraint:
        _sh_idx   = np.where(_lat_arr < 0.0)[0]
        _lat_sh   = _lat_arr[_sh_idx]
        _Avals_sh = np.asarray(anomaly.values, dtype=float)[:, _sh_idx]  # (time, lat_sh)

        def _sh_argmax_lat(t_idx):
            """Latitude of max anomaly in equatorial SH onset band [-onset_lat_max_deg, 0)."""
            row = _Avals_sh[t_idx, :]
            cand = (_lat_sh >= -float(onset_lat_max_deg)) & np.isfinite(row) & (row > anomaly_thres)
            if not np.any(cand):
                return np.nan
            idx = np.where(cand)[0]
            return float(_lat_sh[idx[np.argmax(row[idx])]])

        _sh_argmax_capped = np.array([_sh_argmax_lat(ti) for ti in range(_nt_full)])

    # Enforce El_nino_constraint 
    djf_all_above = None
    if el_nino_constraint:
        _enso_member = f"r{args.member}i1p1f3"
        enso_times, enso_vals = get_ENSO_index(start_yr, end_yr - 1, ensemble_member=_enso_member)
        if enso_times is None or enso_vals is None:
            raise RuntimeError(f"El Niño constraint enabled but no Nino3.4 file found for member {_enso_member}")
        ENSO_da = xr.DataArray(enso_vals, coords={"time": enso_times}, dims=("time",))
        
        # Check for each event, if the onset year DJF has a value above 0.5 for that winter
        month = ENSO_da["time"].dt.month
        year = ENSO_da["time"].dt.year
        winter_year = xr.where(month == 12, year + 1, year)
        
        djf = ENSO_da.where(month.isin([11, 12, 1, 2, 3]), drop=True)
        djf_winter_year = winter_year.sel(time=djf["time"])
        djf = djf.assign_coords(winter_year = djf_winter_year)
        # For each winter, require the mean of DJF months >= 0.5 (El Niño threshold):
        djf_all_above = djf.groupby(djf_winter_year).mean(dim="time") >= 0.5
        
    results = {}
    eid = 0
    t = 0
    min_event_len = 3

    def _truncate_on_stuck_latitude(
        com_lat: xr.DataArray,
        *,
        max_stuck_len: int,
        eps: float = 1.0e-6,
        max_southward_jump_deg: Optional[float] = None,
    ) -> int:
        """Return number of time steps to keep.

        Terminates *before* the first timestep where either:
        - latitude is 'stuck' (|lat[t] - lat[t-1]| <= eps) for more than
          `max_stuck_len` consecutive months, OR
        - latitude drops southward by more than `max_southward_jump_deg` in one step.
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
            if not np.isfinite(vals[i]):
                # NaN (tracker lost anomaly within continuity window) counts toward
                # the intermittency budget just like a stall.
                run_len += 1
                if run_len > max_stuck_len:
                    return int(i)
                continue
            if not np.isfinite(vals[i - 1]):
                # Valid step resuming after a NaN gap — reset stall counter.
                run_len = 1
                continue
            # Both vals[i] and vals[i-1] are finite.
            # Southward jump check: large drop → event has ended
            if max_southward_jump_deg is not None and np.isfinite(max_southward_jump_deg):
                jump = vals[i] - vals[i - 1]
                if jump < -float(max_southward_jump_deg):
                    print(f"  Southward-jump truncation at step {i}: {vals[i-1]:.2f}° → {vals[i]:.2f}°N (drop={-jump:.2f}°)")
                    return int(i)  # keep up to i-1
            if abs(vals[i] - vals[i - 1]) <= eps:
                run_len += 1
                if run_len > max_stuck_len:
                    return int(i)  # truncate before this index
            else:
                run_len = 1
        return int(vals.size)
    
    while t < _nt_full:
        # Need at least intermittency_constraint months ahead to validate northward onset
        if t + intermittency_constraint >= _nt_full:
            break

        lat_onset = _argmax_capped[t]
        if not np.isfinite(lat_onset):
            t += 1
            continue

        # Northward validation: argmax (anywhere in NH) intermittency months later
        # must be north of the onset argmax but within a physically reachable distance
        # (continuity_deg per step × number of steps ahead)
        lat_future = _argmax_free[t + intermittency_constraint]
        max_reachable = lat_onset + argmax_continuity_deg * intermittency_constraint
        if not np.isfinite(lat_future) or lat_future <= lat_onset or lat_future > max_reachable:
            t += 1
            continue

        # Valid onset candidate — apply optional constraints
        onset_time = time_coords[t]
        print(f"\n=== Event candidate: onset={onset_time}, argmax={lat_onset:.2f}°N, "
              f"t+{intermittency_constraint}m={lat_future:.2f}°N ===")

        if winter_constraint and onset_time.month not in (11, 12, 1, 2, 3):
            print(f"  ❌ FILTERED: winter_constraint (month={onset_time.month})")
            t += 1
            continue

        if el_nino_constraint:
            if djf_all_above is None:
                raise ValueError("el_nino_constraint=True but ENSO DJF mask was not computed")
            evt_winter_year = onset_time.year + 1 if onset_time.month == 12 else onset_time.year
            is_el_nino = bool(djf_all_above.sel(group=evt_winter_year))
            if not is_el_nino:
                t += 1
                continue

        if sym_constraint:
            # Run the same argmax_track_latitude tracking on the SH.
            # Poleward in SH = southward = decreasing latitude.
            # To reuse argmax_track_latitude (which is NH-only, lat >= 0) without
            # modification, we negate the SH latitudes so that -60° → +60°,
            # making southward propagation appear as northward in the flipped space.
            sh_lat_onset = _sh_argmax_capped[t]
            if not np.isfinite(sh_lat_onset):
                print(f"  \u274c FILTERED: sym_constraint (no SH onset anomaly at t={t})")
                t += 1
                continue

            anom_sh = anomaly.isel(time=slice(t, None))
            anom_sh = anom_sh.where(anom_sh["latitude"] < 0, drop=True)
            anom_sh = anom_sh.assign_coords(
                latitude=(-anom_sh["latitude"].values)
            ).sortby("latitude")

            com_sh = argmax_track_latitude(
                anom_sh,
                threshold=anomaly_thres,
                continuity_deg=float(argmax_continuity_deg),
                max_southward_jump_deg=float(max_southward_jump_deg),
                start_latitude=float(-sh_lat_onset),  # negate: e.g. -15°S → +15° in flipped space
            )
            keep_sh = _truncate_on_stuck_latitude(
                com_sh,
                max_stuck_len=intermittency_constraint,
                max_southward_jump_deg=max_southward_jump_deg,
            )
            _com_sh_vals = np.asarray(com_sh.values[:keep_sh], dtype=float)
            _sh_finite   = np.where(np.isfinite(_com_sh_vals))[0]
            _sh_final_neg = float(_com_sh_vals[_sh_finite[-1]]) if _sh_finite.size > 0 else np.nan
            sh_onset_neg  = float(-sh_lat_onset)  # onset in flipped space
            if (
                keep_sh < min_event_len
                or not np.isfinite(_sh_final_neg)
                or _sh_final_neg <= sh_onset_neg  # did not propagate poleward
            ):
                sh_final_real = -_sh_final_neg if np.isfinite(_sh_final_neg) else float("nan")
                print(
                    f"  \u274c FILTERED: sym_constraint "
                    f"(SH tracking: {keep_sh} months, "
                    f"{sh_lat_onset:.2f}\u00b0 \u2192 {sh_final_real:.2f}\u00b0)"
                )
                t += 1
                continue
            sh_final_real = -_sh_final_neg
            print(
                f"  \u2713 sym_constraint passed "
                f"(SH: {sh_lat_onset:.2f}\u00b0 \u2192 {sh_final_real:.2f}\u00b0)"
            )

        # Track from onset with continuity constraint
        anom_evt = anomaly.isel(time=slice(t, None))
        com = argmax_track_latitude(
            anom_evt,
            threshold=anomaly_thres,
            continuity_deg=float(argmax_continuity_deg),
            max_southward_jump_deg=float(max_southward_jump_deg),
            start_latitude=float(lat_onset),
        )

        # Terminate event on stall or large southward retreat
        keep_n = _truncate_on_stuck_latitude(
            com,
            max_stuck_len=intermittency_constraint,
            max_southward_jump_deg=max_southward_jump_deg,
        )

        if keep_n < min_event_len:
            print(f"  ❌ FILTERED: too short after truncation ({keep_n} < {min_event_len} months)")
            t += max(1, keep_n)
            continue

        anom_evt = anom_evt.isel(time=slice(0, keep_n))
        com     = com.isel(time=slice(0, keep_n))

        if max_com_jump_deg is not None and com.size >= 2:
            dv = np.diff(np.asarray(com.values, dtype=float))
            max_jump = float(np.nanmax(dv))
            if np.isfinite(max_jump) and max_jump > float(max_com_jump_deg):
                print(f"  ❌ FILTERED: max COM jump {max_jump:.2f}° > {max_com_jump_deg}°")
                t += keep_n
                continue

        # Discard south-moving events: final latitude must be north of onset
        _com_vals = np.asarray(com.values, dtype=float)
        _finite_idx = np.where(np.isfinite(_com_vals))[0]
        _final_lat = float(_com_vals[_finite_idx[-1]]) if _finite_idx.size > 0 else np.nan
        if not np.isfinite(_final_lat) or _final_lat <= lat_onset:
            print(f"  ❌ FILTERED: south-moving event (onset {lat_onset:.2f}°N → end {_final_lat:.2f}°N)")
            t += keep_n
            continue

        eid += 1
        final_len = int(anom_evt.sizes.get("time", 0))
        print(f"  ✅ ACCEPTED event {eid}: {final_len} months, "
              f"{lat_onset:.2f}° → {float(com.max()):.2f}°N")

        # Amplitude at argmax position
        nh_evt = anom_evt.where(anom_evt["latitude"] >= 0, drop=True)
        amp_raw = nh_evt.interp(latitude=com)
        amp_at_com = amp_raw.where(amp_raw > anomaly_thres)

        # Speed: linear fit slope (deg/month)
        slope = None
        tt = np.arange(com.size, dtype=float)
        ok = np.isfinite(com.values)
        if int(np.sum(ok)) >= 2:
            slope, _intercept = np.polyfit(tt[ok], com.values[ok], 1)
            slope = float(slope)

        # Poleward extent based on threshold edge
        nh = anom_evt.where(anom_evt["latitude"] >= 0, drop=True)
        lat_where = xr.where(nh > anomaly_thres, nh["latitude"], np.nan)
        poleward_edge = lat_where.max("latitude", skipna=True)
        max_lat_extent_threshold = float(poleward_edge.max("time", skipna=True))

        results[int(eid)] = {
            "anomaly": anom_evt,
            "com_latitude": com,
            "amp_at_com": amp_at_com,
            "speed_deg_per_month": slope,
            "mean_amp_at_com": float(amp_at_com.mean("time", skipna=True)),
            "max_amp_at_com": float(amp_at_com.max("time", skipna=True)),
            "max_lat_reached_com": float(com.max("time", skipna=True)),
            "max_lat_extent_threshold": max_lat_extent_threshold,
        }

        t += keep_n

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


def plot_hovmoller_with_events(
    AAM_da,
    clim_da,
    results: dict,
    *,
    start_yr: int,
    end_yr: int,
    p_min_hpa: float = 150.0,
    p_max_hpa: float = 700.0,
    ensemble_member: str = "r1i1p1f3",
    output_dir: str = "figures/",
    nlevels: int = 11,
    cmap_name: str = "RdBu_r",
    rolling_period: int = 1,
) -> None:
    """Hovmöller (latitude–time) plot of AAM anomalies with detected event trajectories."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.cm as cm
    import pandas as pd

    # --- Compute anomaly field (time × latitude) using the same pipeline as detection ---
    field = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
    clim = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da

    field = field.sel(latitude=slice(-60, 60), time=slice(f"{start_yr}-01", f"{end_yr - 1}-12"))
    field = _to_per_latitude_band(field)
    field = vertical_sum_over_pressure_range(field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level")
    clim  = vertical_sum_over_pressure_range(clim,  p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level")
    field, clim_on_time = _reindex_to_climatology_dims(field, clim)
    anomaly = field - clim_on_time
    anomaly = _rolling_mean_time(anomaly, rolling_period)

    times = pd.DatetimeIndex(
        [pd.Timestamp(f"{t.year:04d}-{t.month:02d}-01") for t in anomaly["time"].values]
    )
    lats = np.asarray(anomaly["latitude"].values, dtype=float)
    data = np.asarray(anomaly.values, dtype=float)  # (time, lat)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[:, ::-1]

    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    levels = np.linspace(vmin, vmax, nlevels)
    base_cmap = (mpl.colormaps.get_cmap(cmap_name) if hasattr(mpl, 'colormaps')
                 else cm.get_cmap(cmap_name))
    cmap_disc = ListedColormap(list(base_cmap(np.linspace(0, 1, nlevels - 1))))
    norm = BoundaryNorm(levels, ncolors=nlevels - 1, clip=True)

    fig = plt.figure(figsize=(16, 6), constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[6.0, 0.18], hspace=0.08)
    ax  = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[1, 0])

    times_num = np.asarray(mdates.date2num(times.to_pydatetime()), dtype=float)
    im = ax.imshow(
        data.T,
        origin='lower',
        aspect='auto',
        cmap=cmap_disc,
        norm=norm,
        extent=[times_num[0], times_num[-1], lats[0], lats[-1]],
        interpolation='bilinear',
    )
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), ha='center', size=13)
    plt.setp(ax.get_yticklabels(), size=13)
    ax.set_xlabel('Year', size=15)
    ax.set_ylabel('Latitude (°)', size=15)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', zorder=10)
    if rolling_period > 1:
        ax.set_title(
            f"CMIP6 HadGEM3-GC3.1 {ensemble_member} AAM anomalies ({start_yr}\u2013{end_yr})  "
            f"{p_min_hpa:.0f}\u2013{p_max_hpa:.0f} hPa  | {rolling_period} months rolling |  {len(results)} events detected",
            size=12,
        )
    else:
        ax.set_title(
            f"CMIP6 HadGEM3-GC3.1 {ensemble_member} AAM anomalies ({start_yr}\u2013{end_yr})  "
            f"{p_min_hpa:.0f}\u2013{p_max_hpa:.0f} hPa |  {len(results)} events detected",
            size=15,
        )
        
    ax.grid(True, alpha=0.3)

    # Overlay detected event trajectories
    for eid, evt in sorted(results.items(), key=lambda kv: int(kv[0])):
        com = evt.get("com_latitude")
        if not isinstance(com, xr.DataArray) or "time" not in com.coords:
            continue
        tt = pd.DatetimeIndex(
            [pd.Timestamp(f"{t.year:04d}-{t.month:02d}-01") for t in com["time"].values]
        )
        ll = np.asarray(com.values, dtype=float)
        mask = (tt >= times.min()) & (tt <= times.max())
        if mask.sum() < 1:
            continue
        x = np.asarray(mdates.date2num(tt[mask].to_pydatetime()), dtype=float)
        y = ll[mask]
        ax.plot(x, y, color='lime', linewidth=2.5, alpha=0.9, zorder=20)
        finite = np.isfinite(y)
        if finite.any():
            i0 = int(np.where(finite)[0][0])
            ax.text(
                x[i0], float(y[i0]), str(eid),
                fontsize=9, color='white', weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='none', alpha=0.7),
                zorder=21,
            )

    # Colorbar
    cbar = fig.colorbar(
        im, cax=cax, boundaries=levels, extend='both',
        orientation='horizontal', spacing='proportional',
    )
    _abs = max(abs(vmin), abs(vmax))
    order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
    factor = 10 ** order
    tick_spacing = max(1, len(levels) // 8)
    cbar.set_ticks(levels[::tick_spacing].tolist())
    cbar.set_ticklabels([f'{v / factor:.1f}' for v in levels[::tick_spacing]])
    cbar.set_label(f'AAM anomaly \u00d710\u207b\u00b9 kg m\u00b2 s\u207b\u00b9 per lat band (\u00d710^{order})', size=12)
    cbar.ax.tick_params(labelsize=11)

    fig.subplots_adjust(top=0.90, bottom=0.10)
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, cax_pos.y0 - 0.05, cax_pos.width, cax_pos.height * 0.75])

    os.makedirs(output_dir, exist_ok=True)
    if rolling_period > 1:
        savepath = os.path.join(
            output_dir,
            f"AAM_hovmoller_{ensemble_member}_{start_yr}-{end_yr}"
            f"_{p_min_hpa:.0f}-{p_max_hpa:.0f}hPa_rolling{rolling_period}wEvents.png",
        )
    else:
        savepath = os.path.join(
            output_dir,
            f"AAM_hovmoller_{ensemble_member}_{start_yr}-{end_yr}"
            f"_{p_min_hpa:.0f}-{p_max_hpa:.0f}hPa_Events.png",
        )
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Hovmöller plot saved to: {savepath}")


def composite_propagating_years(
    AAM_da,
    wind_da,
    date_list,
    clim_da=None,
    *,
    clim_start_yr: int = 1980,
    clim_end_yr: int = 2000,
    ensemble_member: str = args.member,
    p_min_hpa: float = 150.0,
    p_max_hpa: float = 700.0,
    output_dir: str = "figures/",
    events_meta=None,
    winter_constraint: bool = False,
    el_nino_constraint: bool = False,
    sym_constraint: bool = False,
    rolling_period: int = 1,
    nlevels: int = 11,
):
    """Composite AAM anomalies for detected propagating event years.

    For each (onset_time, end_time) entry in date_list, the full calendar year
    (Jan–Dec) of the onset year is extracted and stacked.  Duplicate onset years
    are composited only once.  The result is a 12-month mean anomaly aligned on
    calendar month.

    Parameters
    ----------
    AAM_da : xr.DataArray or xr.Dataset  (time, level, latitude[, longitude])
    wind_da : xr.DataArray or xr.Dataset | None
    date_list : list of (onset, end) time-like tuples
    clim_da : xr.DataArray or xr.Dataset, optional
        Monthly climatology with a 'month' dim (1–12).  When None, a simple
        climatology is derived from the full AAM_da time range.
    """
    if not date_list:
        print("composite_propagating_years: date_list is empty, nothing to composite.")
        return

    rolling_period = int(rolling_period)
    if rolling_period < 1:
        raise ValueError("rolling_period must be >= 1")

    def _circular_rolling_mean(da: xr.DataArray, *, dim: str, window: int) -> xr.DataArray:
        """Circular rolling mean over `dim` to preserve Jan/Dec continuity."""
        if window <= 1:
            return da
        n = int(da.sizes[dim])
        if window > n:
            raise ValueError(f"rolling_period ({window}) cannot exceed number of {dim} bins ({n})")
        left = window // 2
        right = window - left - 1
        rolled = [da.roll({dim: -offset}, roll_coords=False) for offset in range(-left, right + 1)]
        return xr.concat(rolled, dim="_roll").mean("_roll", skipna=True)

    def _rolling_tick_labels(window: int, n_bins: int = 12) -> list[str]:
        """Return month-window labels for axis ticks (e.g., DJF, JFM, FMA for window=3)."""
        month_initials = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        if window <= 1:
            return [f"M{m:02d}" for m in range(1, n_bins + 1)]
        left = window // 2
        right = window - left - 1
        labels = []
        for center in range(n_bins):
            parts = [month_initials[(center + off) % 12] for off in range(-left, right + 1)]
            labels.append("".join(parts))
        return labels

    # --- Unwrap Datasets ---
    AAM_field = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
    if isinstance(AAM_field, xr.Dataset):
        AAM_field = next(iter(AAM_field.data_vars.values()))

    wind_field = None
    if wind_da is not None:
        wind_field = wind_da["ua"] if isinstance(wind_da, xr.Dataset) and "ua" in wind_da else wind_da
        if isinstance(wind_field, xr.Dataset):
            wind_field = next(iter(wind_field.data_vars.values()))

    # --- Zonal integral for AAM (per-latitude-band, matches detect_poleward_propagation_time)
    # and zonal mean for wind (intensive quantity) ---
    AAM_field = _to_per_latitude_band(AAM_field)
    if wind_field is not None:
        for lon_name in ("longitude", "lon"):
            if lon_name in wind_field.dims:
                wind_field = wind_field.mean(dim=lon_name, skipna=True)

    # --- Vertical integration over the requested pressure range ---
    AAM_field = vertical_sum_over_pressure_range(
        AAM_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
    )
    if wind_field is not None:
        wind_field = vertical_sum_over_pressure_range(
            wind_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )

    # --- Compute anomalies ---
    if clim_da is not None:
        clim_field = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
        if isinstance(clim_field, xr.Dataset):
            clim_field = next(iter(clim_field.data_vars.values()))
        clim_field = vertical_sum_over_pressure_range(
            clim_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )
        AAM_anom = AAM_field.groupby("time.month") - clim_field
    else:
        clim_period = AAM_field.sel(time=slice(f"{clim_start_yr}-01", f"{clim_end_yr}-12"))
        clim_inline = clim_period.groupby("time.month").mean("time")
        AAM_anom = AAM_field.groupby("time.month") - clim_inline

    # --- Extract 12-month window starting from onset month ---
    stacked_AAM = []
    stacked_wind = []
    seen_onset_months: set = set()  # deduplicate by onset year-month, not calendar year
    included_event_meta = []  # events_meta entries that were actually included

    for idx, (onset_time, _end_time) in enumerate(date_list):
        onset_str = (
            _time_value_to_ymd_string(onset_time)
            if not isinstance(onset_time, str)
            else onset_time
        )
        onset_ym = onset_str[:7]  # "YYYY-MM"
        if onset_ym in seen_onset_months:
            print(f"  composite: onset {onset_ym} already included, skipping duplicate.")
            continue
        seen_onset_months.add(onset_ym)

        onset_year = int(onset_str[:4])
        onset_month = int(onset_str[5:7])

        # Window: onset month → onset month + 11 months (crosses year boundary if needed)
        import pandas as _pd
        t_start = _pd.Timestamp(f"{onset_year}-{onset_month:02d}-01")
        t_end = t_start + _pd.DateOffset(months=11)
        window_start = t_start.strftime("%Y-%m")
        window_end = t_end.strftime("%Y-%m")

        aam_event = AAM_anom.sel(time=slice(window_start, window_end))
        n_avail = int(aam_event.sizes["time"])
        if n_avail < 12:
            print(f"  composite: onset {onset_ym} window has only {n_avail} months of data, skipping.")
            continue

        # Label as relative months 1–12 (1 = onset month)
        aam_event = aam_event.isel(time=slice(0, 12))
        aam_event = aam_event.assign_coords(time=np.arange(1, 13, dtype=int))
        if "month" in aam_event.coords:
            aam_event = aam_event.drop_vars("month")
        aam_event = aam_event.rename({"time": "month"})
        stacked_AAM.append(aam_event)
        if events_meta is not None and idx < len(events_meta):
            included_event_meta.append(events_meta[idx])

        if wind_field is not None:
            wind_event = wind_field.sel(time=slice(window_start, window_end))
            n_w = int(wind_event.sizes["time"])
            if n_w < 12:
                print(f"  composite: wind onset {onset_ym} window incomplete ({n_w} months), skipping.")
                continue
            wind_event = wind_event.isel(time=slice(0, 12))
            wind_event = wind_event.assign_coords(time=np.arange(1, 13, dtype=int))
            if "month" in wind_event.coords:
                wind_event = wind_event.drop_vars("month")
            wind_event = wind_event.rename({"time": "month"})
            stacked_wind.append(wind_event)

    n_events = len(stacked_AAM)
    if n_events < 1:
        print("composite_propagating_years: no valid events to composite.")
        return

    print(f"Compositing {n_events} event year(s) (full Jan–Dec).")
    aam_stack = xr.concat(stacked_AAM, dim="event")
    aam_stack_for_plot = _circular_rolling_mean(aam_stack, dim="month", window=rolling_period)
    composite_AAM = aam_stack_for_plot.mean("event", skipna=True)

    composite_wind = None
    if stacked_wind:
        wind_stack = xr.concat(stacked_wind, dim="event")
        wind_stack = _circular_rolling_mean(wind_stack, dim="month", window=rolling_period)
        composite_wind = wind_stack.mean("event", skipna=True)

    # --- Plot lat×time Hovmöller of composite ---
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import os

    lat_dim = "latitude" if "latitude" in composite_AAM.dims else "lat"
    lat_vals = composite_AAM[lat_dim].values
    month_vals = composite_AAM["month"].values  # 1–12

    aam_vals = composite_AAM.values  # (month, lat) or (lat, month)
    # ensure shape is (lat, month)
    if composite_AAM.dims[0] == "month":
        aam_vals = aam_vals.T

    # --- T-test: significance vs zero along event dimension ---
    from scipy import stats as _stats
    aam_for_ttest = aam_stack_for_plot.transpose("event", lat_dim, "month").values
    _, p_vals = _stats.ttest_1samp(aam_for_ttest, 0.0, axis=0, nan_policy="omit")
    # p_vals is (lat, month), matching aam_vals layout
    print(f"  t-test: shape={p_vals.shape}, min p={float(np.nanmin(p_vals)):.4f}, "
          f"p<0.05: {int(np.sum(p_vals < 0.05))} pts, "
          f"p<0.10: {int(np.sum(p_vals < 0.10))} pts")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.18)
    vmax = float(np.nanpercentile(np.abs(aam_vals), 95))
    vmin = -vmax
    vmax = vmax if vmax > 0 else 1.0
    cf = ax.contourf(
        month_vals, lat_vals, aam_vals,
        levels=np.linspace(-vmax, vmax, nlevels),
        cmap="RdBu_r", extend="both",
    )
    
    _abs = max(abs(vmin), abs(vmax))
    order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
    factor = 10 ** order
    cax = fig.add_axes([0.125, 0.06, 0.775, 0.015])  # [left, bottom, width, height] #thin colorbar
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='both')
    _sup = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
    _order_sup = str(order).translate(_sup)
    cbar.set_label(f"AAM anomaly (\u00d710{_order_sup} kg m\u00b2 s\u207b\u00b9 per lat band)", size=12)
    _tick_levels = cf.levels[::2]
    cbar.set_ticks(_tick_levels)
    cbar.set_ticklabels([f'{v / factor:.1f}' for v in _tick_levels])
    cbar.ax.tick_params(labelsize=11)

    if composite_wind is not None:
        wind_vals = composite_wind.values
        if composite_wind.dims[0] == "month":
            wind_vals = wind_vals.T
        ax.contour(
            month_vals, lat_vals, wind_vals,
            levels=[-10, -5, 5, 10], colors="k", linewidths=0.8,
        )

    # Dot significant points (p < 0.05)
    sig_lat_idx, sig_month_idx = np.where(p_vals < 0.05)
    if sig_lat_idx.size > 0:
        ax.scatter(
            month_vals[sig_month_idx], lat_vals[sig_lat_idx],
            s=20, c="k", marker=".", linewidths=0, zorder=10,
        )
    else:
        print("  No grid points reach p<0.05 significance.")

    # Propagation speed line
    legend_handles = []
    if included_event_meta:
        from datetime import datetime as _dt
        import matplotlib.lines as _mlines
        onset_months = [_dt.strptime(e["onset"][:7], "%Y-%m").month for e in included_event_meta]
        onset_lats = [e["monthly_com_latitude"]["com_latitude_deg"][0] for e in included_event_meta]
        speeds = [e["speed_deg_per_month"] for e in included_event_meta]
        # Circular mean of onset months (handles Dec/Jan boundary correctly)
        _angles = np.array(onset_months, dtype=float) * 2.0 * np.pi / 12.0
        _circ = float(np.arctan2(np.mean(np.sin(_angles)), np.mean(np.cos(_angles))) / (2.0 * np.pi) * 12.0) % 12.0
        mean_onset_month = _circ if _circ >= 0.5 else _circ + 12.0
        mean_onset_lat = float(np.mean(onset_lats))
        mean_speed = float(np.mean(speeds))
        # x=1 is always the onset month in the relative-month composite
        x_line = np.array([1.0, 12.0])
        y_line = mean_onset_lat + mean_speed * (x_line - 1.0)
        (speed_line,) = ax.plot(
            x_line, y_line, color="lime", linewidth=2, linestyle="--", zorder=6,
        )
        ax.scatter([1.0], [mean_onset_lat], color="lime", s=50, zorder=7)
        legend_handles.append(
            _mlines.Line2D([], [], color="lime", linewidth=2, linestyle="--",
                           label=f"mean speed {mean_speed:.1f}°/month")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    if rolling_period == 3:
        ax.set_xlabel("Rolling 3-month window")
    elif rolling_period > 1:
        ax.set_xlabel(f"Rolling {rolling_period}-month window")
    else:
        ax.set_xlabel("Month since onset (1 = onset month)")
    ax.set_ylabel("Latitude (°N)")
    _constraint_parts = []
    if winter_constraint:   _constraint_parts.append("winter")
    if el_nino_constraint:  _constraint_parts.append("El Niño")
    if sym_constraint:      _constraint_parts.append("SH")
    _constraint_str = ", ".join(_constraint_parts) + " constraints" if _constraint_parts else "No constraints"
    
    if rolling_period > 1:
        ax.set_title(
            f"HadGEM3_GC31 {ensemble_member} Composite AAM anomaly ({p_min_hpa}–{p_max_hpa} hPa)\n"
            f"{n_events} events  {args.start_year}–{args.end_year}  clim {clim_start_yr}–{clim_end_yr}  |  {rolling_period}-month rolling  |  {_constraint_str}"
        )
    else:
        ax.set_title(
            f"HadGEM3_GC31 {ensemble_member} Composite AAM anomaly ({p_min_hpa}–{p_max_hpa} hPa)\n"
            f"{n_events} events  {args.start_year}–{args.end_year}  clim {clim_start_yr}–{clim_end_yr}  |  {_constraint_str}"
        )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    if rolling_period >= 3:
        ax.set_xticks(month_vals)
        ax.set_xticklabels(_rolling_tick_labels(rolling_period, n_bins=len(month_vals)), rotation=0, ha="center")
    ax.set_xlim(1, 12)
    ax.set_ylim(-60, 60)

    _constraint_tag = ""
    if winter_constraint:
        _constraint_tag += "_winter"
    if el_nino_constraint:
        _constraint_tag += "_elnino"
    if sym_constraint:
        _constraint_tag += "_sym"
    os.makedirs(output_dir, exist_ok=True)
    if rolling_period > 1:
        out_path = os.path.join(
            output_dir,
            f"AAM_composite_{ensemble_member}_{args.start_year}-{args.end_year}"
            f"_{p_min_hpa}-{p_max_hpa}hPa_rolling{rolling_period}_{_constraint_tag}.png",
        )
    else:
        out_path = os.path.join(
            output_dir,
            f"AAM_composite_{ensemble_member}_{args.start_year}-{args.end_year}"
            f"_{p_min_hpa}-{p_max_hpa}hPa{_constraint_tag}.png",
        )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out_path}")
    


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

    # --- Step 1: Detect events ---
    results = detect_poleward_propagation_time(
        AAM_da,
        clim_da,
        start_yr=args.start_year,
        end_yr=args.end_year,
        sym_constraint=bool(args.sym_constraint),
        intermittency_constraint=int(args.intermittency),
        winter_constraint=bool(args.winter_constraint),
        el_nino_constraint=bool(args.el_nino_constraint),
        anomaly_thres=0.0,
        p_min_hpa=float(args.p_min),
        p_max_hpa=float(args.p_max),
        onset_lat_max_deg=float(args.onset_lat_max),
        max_com_jump_deg=args.max_com_jump,
        argmax_continuity_deg=float(args.argmax_continuity_deg),
        max_southward_jump_deg=float(args.max_southward_jump),
        rolling_period=int(args.rolling_period),
    )

    # Build constraint tag for filename and title
    _tags = []
    if args.winter_constraint:   _tags.append("winter")
    if args.el_nino_constraint:  _tags.append("elnino")
    if args.sym_constraint:      _tags.append("sym")
    constraint_tag = ("_" + "_".join(_tags)) if _tags else ""

    # --- Step 2: Write JSON ---
    json_out = f"AAM_event_metrics_{args.start_year}_{args.end_year}.json"
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
            "winter_constraint": bool(args.winter_constraint),
            "el_nino_constraint": bool(args.el_nino_constraint),
            "sym_constraint": bool(args.sym_constraint),
            "onset_lat_max_deg": float(args.onset_lat_max),
            "max_com_jump_deg": args.max_com_jump,
            "argmax_continuity_deg": float(args.argmax_continuity_deg),
            "max_southward_jump_deg": float(args.max_southward_jump),
            "rolling_period": int(args.rolling_period),
        },
    )
    print(f"Detected {len(results)} events")
    print(f"Wrote metrics-only JSON: {json_out}")

    # --- Step 3: Hovmöller plot with event trajectories ---
    plot_hovmoller_with_events(
        AAM_da,
        clim_da,
        results,
        start_yr=args.start_year,
        end_yr=args.end_year,
        p_min_hpa=float(args.p_min),
        p_max_hpa=float(args.p_max),
        ensemble_member=ensemble_member,
        output_dir=output_dir,
        rolling_period=int(args.rolling_period),
    )

    # --- Step 4: Read JSON back and composite ---
    with open(json_out, "r", encoding="utf-8") as _f:
        results_json = json.load(_f)
    date_list = [(evt["onset"], evt["end"]) for evt in results_json["events"]]
    if date_list:
        composite_propagating_years(
            AAM_da,
            wind_da=None,
            date_list=date_list,
            clim_da=clim_da,
            clim_start_yr=clim_start_yr,
            clim_end_yr=clim_end_yr,
            ensemble_member=ensemble_member,
            p_min_hpa=float(args.p_min),
            p_max_hpa=float(args.p_max),
            output_dir=output_dir,
            events_meta=results_json["events"],
            winter_constraint=bool(args.winter_constraint),
            el_nino_constraint=bool(args.el_nino_constraint),
            sym_constraint=bool(args.sym_constraint),
            rolling_period=int(args.rolling_period),
        )

    # --- Step 5: Latitude×level composite snapshot plot ---
    # Re-uses the full multi-level AAM_da (all levels, all lats) but zonally integrates
    # (removes longitude) before compositing. No vertical integration here — the snapshot
    # function shows the full lat×level cross-section.
    if date_list:
        import pandas as _pd

        # Zonal integral only (keeps all pressure levels) → (time, level, latitude)
        aam_full = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
        aam_full = _to_per_latitude_band(aam_full)

        # Anomaly vs climatology (same pipeline as the rest of the script)
        clim_full = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
        aam_full, clim_on_time_full = _reindex_to_climatology_dims(aam_full, clim_full)
        anom_full = aam_full - clim_on_time_full  # (time, level, latitude)

        # Composite over events aligned to relative month 1–12 (1 = onset month)
        stacked_full = []
        seen_ev: set = set()
        for onset_str, _ in date_list:
            if not isinstance(onset_str, str):
                onset_str = _time_value_to_ymd_string(onset_str)
            ym = onset_str[:7]
            if ym in seen_ev:
                continue
            seen_ev.add(ym)
            onset_year = int(onset_str[:4])
            onset_month = int(onset_str[5:7])
            t_start = _pd.Timestamp(f"{onset_year}-{onset_month:02d}-01")
            window_end = (t_start + _pd.DateOffset(months=11)).strftime("%Y-%m")
            evt = anom_full.sel(time=slice(ym, window_end))
            if int(evt.sizes["time"]) < 12:
                print(f"  snapshot composite: onset {ym} window incomplete, skipping.")
                continue
            evt = evt.isel(time=slice(0, 12))
            evt = evt.assign_coords(time=np.arange(1, 13, dtype=int))
            if "month" in evt.coords:
                evt = evt.drop_vars("month")
            evt = evt.rename({"time": "month"})
            stacked_full.append(evt)

        if stacked_full:
            n_ev = len(stacked_full)
            full_stack = xr.concat(stacked_full, dim="event")

            # Apply circular rolling over the 12 relative months before averaging events.
            # This wraps across year-end, so NDJ uses Nov-Dec-Jan and DJF uses Dec-Jan-Feb.
            rp = int(args.rolling_period)
            if rp > 1:
                n_month = int(full_stack.sizes["month"])
                if rp > n_month:
                    raise ValueError(
                        f"rolling_period ({rp}) cannot exceed number of month bins ({n_month})"
                    )
                left = rp // 2
                right = rp - left - 1
                _rolled = [
                    full_stack.roll(month=-offset, roll_coords=False)
                    for offset in range(-left, right + 1)
                ]
                full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)

            composite_full = full_stack.mean("event", skipna=True)
            # Rename month → time so plot_latitude_level_snapshots_HadGEN3 sees a 'time' dim
            composite_full = composite_full.rename({"month": "time"})
            composite_full.attrs["long_name"] = "AAM anomaly"
            _constraint_parts_snap = []
            if args.winter_constraint:  _constraint_parts_snap.append("winter")
            if args.el_nino_constraint: _constraint_parts_snap.append("El Niño")
            if args.sym_constraint:     _constraint_parts_snap.append("SH")
            _snap_suffix = ", ".join(_constraint_parts_snap) + " constraints" if _constraint_parts_snap else ""
            print(f"Plotting lat×level composite snapshots for {n_ev} events...")
            plot_latitude_level_snapshots_HadGEN3(
                composite_full,
                ensemble_member=ensemble_member,
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                vpercentile = 99.0,
                clim_end_yr=clim_end_yr,
                output_dir=output_dir,
                title_suffix=_snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=constraint_tag,
            )