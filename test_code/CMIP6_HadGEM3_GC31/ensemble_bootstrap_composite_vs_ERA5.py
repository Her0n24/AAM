"""
This code compares the HadGEM3 bootstrap composites from a pool of events per member
with ERA5 observational reanalysis (regarded as reference/truth) all event simple mean composites. 

It will first perform a regrid (bilinear interpolation) to upscale the ERA5 data.
Then it will compute statistics (pattern correlation, normalized std, RMSE) for each HadGEM3 member 
bootstrap composite against the reference ERA5 composite, and plot Taylor diagrams to visualize the distribution of skill across the bootstrap samples.
Default bootstrap settings: 2000 iteratiosn, 95% confidence level, pooled resampling across all events (not block bootstrapping).
"""
# %%
import xarray as xr
import numpy as np
import pandas as pd
import json
import traceback
import re
# Allow importing shared utilities from AAM/test_code
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utilities import _to_per_latitude_band, _reindex_to_climatology_dims, vertical_sum_over_pressure_range, get_ENSO_index, pressure_range_in_coord_units, REGION_BOUNDS
from plotting_utils import (
    compute_taylor_stats_against_reference,
    plot_latitude_level_snapshots_HadGEN3,
    plot_lat_lon_snapshots,
    plot_taylor_diagram_from_stats,
)
import tqdm
from scipy import stats as _stats
from matplotlib import pyplot as plt

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Note: joblib not installed. For CPU-bound parallelization, install: pip install joblib")
    sys.exit(1)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot CMIP6 AAM anomalies integrated over specified pressure levels')
parser.add_argument('--p-min', type=float, default=150.0, help='Minimum pressure level (hPa) to include (default: 0 hPa)')
parser.add_argument('--p-max', type=float, default=700, help='Maximum pressure level (hPa) to include (default: 1020 hPa)')
parser.add_argument('--start-year', type=int, default=1850, help='Start year to plot (default: 1850)')
parser.add_argument('--end-year', type=int, default=2010, help='End year to plot (default: 2010)')
parser.add_argument('--enso-state', type=str, default='el_nino', choices=['el_nino', 'la_nina'], help='ENSO state to detect (default: el_nino)')
parser.add_argument('--min-elnino-months', type=int, default=3, help='Minimum consecutive months above threshold to define El Nino state (default: 3)')
parser.add_argument('--rolling-period', type=int, default=1, help='Rolling-month window for composite analysis (default: 1 = no rolling; 3 gives DJF/JFM/FMA labels)')
parser.add_argument('--composite-months', type=int, default=24,
                    help='Number of months to composite for each event window (default: 24)')
parser.add_argument('--composite-start', type=str, default='onset', choices=['onset', 'december_onset_year'],
                    help='Composite window start mode: onset month, or December of onset year (default: onset)')
parser.add_argument('--onset-season', type=str, default='all', choices=['all', 'ndjfm'],
                    help='Only composite events whose onset month falls in the given season (default: all; ndjfm = Nov–Mar only)')
parser.add_argument(
    '--region',
    type=str,
    default='all',
    choices=['all', 'pacific', 'indian', 'atlantic'],
    help='Geographic region to analyze (default: all). Pacific: 125–(-110)°, Indian: 50–100°, Atlantic: -60–10°',
)
parser.add_argument(
    '--member-bootstrap-taylor-vs-era5',
    action='store_true',
    help=(
        'Also create one bootstrapped AAM composite per HadGEM3 member from that member event pool, '
        'then compare those member bootstrap composites to ERA5 in a labeled Taylor diagram.'
    ),
)

if "ipykernel" in sys.modules:
    args = parser.parse_args([
    "--composite-start", "onset",
    "--onset-season", "ndjfm",
    "--region", "all",
    ])
else:
    args = parser.parse_args()
# args = parser.parse_args()
replot = False
if args.member_bootstrap_taylor_vs_era5:
    replot = False

n_cpus_to_use = 8
# Define region longitude bounds (degrees East; negative = West)
REGION_BOUNDS = {
    'all': None,  # no filtering
    'pacific': (125, -110),      # 125°E to 110°W
    'indian': (50, 100),         # 50°E to 100°E
    'atlantic': (-60, 10),       # 60°W to 10°E
}

import matplotlib.pyplot as plt
import numpy as np
ERA5_RESULTS_DIR = Path(__file__).resolve().parents[1] / "era5" / "composite_non_tracking"

def _safe_label_float(value):
    return f"{float(value):g}"


def ml_to_pl_era5 (era5_regridded, m_da) -> xr.DataArray:
    
    # Ensure matching dimensions
    m_da = m_da.copy()

    # Match dimension names if target uses 'lat' instead of 'latitude'
    if "lat" in m_da.dims and "latitude" not in m_da.dims:
        m_da = m_da.rename({"lat": "latitude"})

    # Convert target levels from Pascals (Pa) to hPa if they are in the 100,000 range
    if m_da["level"].max() > 2000:
        print("  [INFO] Target levels appear to be in Pascals (Pa). Converting to hPa...")
        m_da = m_da.assign_coords(level=m_da["level"] / 100.0)

    # CRITICAL: groupby_bins requires bins/coordinates to be strictly ascending (increasing)
    m_da = m_da.sortby("level", ascending=True)
    
    
    # Extract the raw level values from the netCDF
    incoming_levels = era5_regridded['level'].values
    
    # Detect if the levels dimension is corrupted, if it is more than 137 levels
    incoming_levels = era5_regridded['level'].values
    
    if len(incoming_levels) > 137:
        print(f"  [INFO] Mangled ERA5 axis detected ({len(incoming_levels)} levels). Cleaning integer artifacts...")
        valid_pressure_levels = [
            lv for lv in incoming_levels 
            if not (float(lv).is_integer() and 0 <= lv <= 137)
        ]
        era5_regridded = era5_regridded.sel(level=valid_pressure_levels)
        print(f"  [INFO] Axis cleaned. Reduced to {len(era5_regridded['level'])} physical pressure levels.")
    elif float(era5_regridded['level'].max()) <= 137:
        print("  [INFO] Standard ERA5 model levels (1-137) detected. Applying CSV mapping...")
        level_map_df = pd.read_csv("/home/users/hhhn2/AAM/test_code/era5/l137_a_b.csv")
        level_to_hpa = dict(zip(level_map_df['n'], level_map_df['ph [hPa]']))
        nominal_pressures = [level_to_hpa[int(lev)] for lev in era5_regridded['level'].values]
        era5_regridded = era5_regridded.assign_coords(level=("level", nominal_pressures))

    # Ensure ERA5 levels are also strictly ascending before binning
    era5_regridded = era5_regridded.sortby("level", ascending=True)
    
    # -------------------------------------------------------------------------
    # CONSERVATIVE BINNING: Group into HadGEM3 levels
    # -------------------------------------------------------------------------
    target_levels = m_da["level"].values

    # Calculate edges for the HadGEM3 bins
    midpoints = (target_levels[1:] + target_levels[:-1]) / 2.0
    top_edge = target_levels[0] - (target_levels[1] - target_levels[0]) / 2.0
    bottom_edge = target_levels[-1] + (target_levels[-1] - target_levels[-2]) / 2.0
    bin_edges = np.concatenate(([top_edge], midpoints, [bottom_edge]))
    bin_edges = np.sort(bin_edges) 

    # Sum the mass-weighted ERA5 layers into the coarse HadGEM3 bins
    era5_coarsened = era5_regridded.groupby_bins(
        "level", 
        bins=bin_edges, 
        labels=np.sort(target_levels),  
        right=False
    ).sum(dim="level")

    # Clean up dimensions to match HadGEM3 expectations
    era5_coarsened = era5_coarsened.rename({"level_bins": "level"})
    era5_coarsened["level"] = era5_coarsened["level"].astype(m_da["level"].dtype)
    
    return era5_coarsened
    
def _era5_composite_path_candidates(comp_type: str) -> list[Path]:
    """Return ERA5 composite candidates, preferring an exact run-argument match."""
    if comp_type not in ["aam", "latlon", "latlev"]:
        return []

    era5_dir = Path(ref_ERA5_composite_dir)
    # Handle the filename convention split
    if comp_type == "aam":
        prefix = "ERA5_Composite_AAM"
    else:
        # Both latlon and latlev are stored in the event_composite file
        prefix = "ERA5_event_composite"

    exact_path = era5_dir / (
        f"{prefix}_{args.enso_state}_{args.start_year}-{args.end_year}_"
        f"{args.p_min:g}-{args.p_max:g}hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}.nc"
    )

    pattern = (
        f"{prefix}_{args.enso_state}_*hPa_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}.nc"
    )
    
    candidates = [exact_path]
    if era5_dir.exists():
        candidates.extend(sorted(era5_dir.glob(pattern)))

    seen: set[Path] = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    def _sort_key(path: Path) -> tuple[float, float, str]:
        if path == exact_path:
            return (0.0, 0.0, path.name)
        match = re.search(
            r"_(?P<start>\d{4})-(?P<end>\d{4})_(?P<pmin>[\d.]+)-(?P<pmax>[\d.]+)hPa_",
            path.name,
        )
        if not match:
            return (1.0e9, 1.0e9, path.name)
        pressure_penalty = (
            abs(float(match.group("pmin")) - float(args.p_min))
            + abs(float(match.group("pmax")) - float(args.p_max))
        )
        year_penalty = (
            abs(int(match.group("start")) - int(args.start_year))
            + abs(int(match.group("end")) - int(args.end_year))
        )
        return (pressure_penalty, year_penalty, path.name)

    return sorted(unique_candidates, key=_sort_key)


def _load_era5_composite_truth(comp_type: str) -> Optional[xr.DataArray]:
    """Load the single ERA5 deterministic composite used as ground truth."""
    candidates = _era5_composite_path_candidates(comp_type)

    era5_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if era5_path is None:
        expected = candidates[0] if candidates else Path(ref_ERA5_composite_dir)
        print(f"ERA5 mean composite not found: {expected}")
        return None
    if candidates and era5_path != candidates[0]:
        print(
            "ERA5 exact composite not found for the CMIP6 run arguments; "
            f"using available ERA5 composite instead: {era5_path}"
        )

    try:
        ds = xr.open_dataset(era5_path)
        
        # Route to the exact variable name based on your ERA5 ncdump
        target_vars = []
        if comp_type == "latlon":
            target_vars = ["lat_lon_composite"]
        elif comp_type == "latlev":
            target_vars = ["height_lat_composite"]
        else: # aam
            target_vars = ["composite_mean", "ens_mean", "AAMA", "aam_mean"]

        for var_name in target_vars:
            if var_name in ds:
                truth = ds[var_name].copy()
                ds.close()
                truth.attrs["source_path"] = str(era5_path)
                return truth
        # Fallback
        if len(ds.data_vars) == 1:
            var_name = next(iter(ds.data_vars))
            truth = ds[var_name].copy()
            ds.close()
            truth.attrs["source_path"] = str(era5_path)
            return truth    
        available = list(ds.data_vars)
        ds.close()
        print(f"Could not identify ERA5 truth variable for {comp_type} in {era5_path}; variables={available}")
        return None
    except Exception as e:
        print(f"Error loading ERA5 truth from {era5_path}: {e}")
        return None


def _regrid_reference_to_target_grid(
    reference_da: xr.DataArray,
    target_da: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Interpolate the fixed reference field onto the target bootstrap grid.

    The Taylor comparison should keep the reference deterministic. This helper
    only remaps ERA5 to the HadGEM3 coordinates; it does not create a bootstrap
    ensemble from ERA5.
    
    We will use an area-weighted block average to transition from high (Era5) to low (HadGEM3) resolution. 
    """
    reference = _standardize_taylor_dims(reference_da)
    target = _standardize_taylor_dims(target_da)

    # 1. Dynamically identify if the dataset uses "lat/lon" or "latitude/longitude"
    lat_dim = next((d for d in reference.dims if d in ["lat", "latitude"]), None)
    lon_dim = next((d for d in reference.dims if d in ["lon", "longitude"]), None)
    
    # 1. Check for resolution factor downscaling (e.g., 0.25 to 2.5 deg = factor of 10)
    # We apply an area-weighted coarsen to smooth high-res features into the coarse grid footprint.
    coarsen_dict = {}
    for dim in [lat_dim, lon_dim]:
        if dim in reference.dims and dim in target.dims:
            ref_res = abs(float(reference[dim].diff(dim).median()))
            tgt_res = abs(float(target[dim].diff(dim).median()))
            factor = int(round(tgt_res / ref_res))
            if factor > 1:
                coarsen_dict[dim] = factor
    #import pdb; pdb.set_trace()
    if coarsen_dict:
        # AAM is an EXTENSIVE quantity (total momentum). 
        # When moving from a 0.25 deg grid to a 2.5 deg grid, we must SUM 
        # the smaller bands to match the total mass/momentum of the wider band.
        if lat_dim in coarsen_dict:
            lat_coarsen = {lat_dim: coarsen_dict[lat_dim]}
            reference = reference.coarsen(lat_coarsen, boundary="trim").sum()
        else:
            # Fallback if only longitude is being coarsened (unlikely since it's zonally integrated)
            reference = reference.coarsen(coarsen_dict, boundary="trim").sum()
            
    #import pdb; pdb.set_trace()
    interp_coords: dict[str, xr.DataArray] = {}
    for dim in reference.dims:
        if dim == "iteration" or dim not in target.dims:
            continue
        if dim in target.coords:
            interp_coords[dim] = target[dim]

    ### This is gonna produce bunch of issues if not handled properly, i.e.
    ## the end point values and the ordering of the lat/lon
    if interp_coords:
        try:
            reference = reference.interp(interp_coords, method="linear")
        except Exception:
            reference = reference.reindex(interp_coords, method="nearest")

    shared_dims = [dim for dim in target.dims if dim in reference.dims and dim != "iteration"]
    if not shared_dims:
        raise ValueError(
            f"No shared dimensions between reference {reference.dims} and target {target.dims}"
        )

    return reference.transpose(*shared_dims), target.transpose("iteration", *shared_dims)


def _standardize_taylor_dims(da: xr.DataArray) -> xr.DataArray:
    rename_map = {}
    if "lat" in da.dims and "latitude" not in da.dims:
        rename_map["lat"] = "latitude"
    if "lon" in da.dims and "longitude" not in da.dims:
        rename_map["lon"] = "longitude"
    if "time" in da.dims and "month" not in da.dims:
        rename_map["time"] = "month"
    # Ensure vertical pressure dimensions match
    if "plev" in da.dims and "level" not in da.dims:
        rename_map["plev"] = "level"
        
    if rename_map:
        da = da.rename(rename_map)
    return da


def _align_bootstrap_to_truth(truth_da: xr.DataArray, boot_ds: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Align HadGEM bootstrap iterations to the ERA5 truth grid before flattening."""
    truth = _standardize_taylor_dims(truth_da)
    boot = _standardize_taylor_dims(boot_ds)

    for dim in truth.dims:
        if dim == "quantile":
            truth = truth.squeeze(dim, drop=True)
            continue
        if dim not in boot.dims:
            continue
        if dim in truth.coords and dim in boot.coords:
            try:
                boot = boot.interp({dim: truth[dim]}, kwargs={"fill_value": "extrapolate"})
            except Exception:
                boot = boot.reindex({dim: truth[dim]}, method="nearest")

    shared_dims = [dim for dim in truth.dims if dim in boot.dims and dim != "iteration"]
    if not shared_dims:
        raise ValueError(f"No shared dimensions between ERA5 truth {truth.dims} and HadGEM bootstrap {boot.dims}")
    return truth.transpose(*shared_dims), boot.transpose("iteration", *shared_dims)


def plot_hadgem_bootstrap_taylor_vs_era5(
    boot_ds: xr.DataArray,
    comp_type: str,
    label: str,
    output_dir: str,
) -> Optional[str]:
    """Taylor diagram of all HadGEM3 bootstrap composites against ERA5 truth."""
    import os

    era5_truth = _load_era5_composite_truth(comp_type)
    if era5_truth is None:
        return None

    try:
        ref_da, boot_aligned = _regrid_reference_to_target_grid(era5_truth, boot_ds)
    except Exception as e:
        print(f"Skipping ERA5 Taylor diagram for {comp_type} {label}: alignment failed: {e}")
        return None
    
    # --- ADD LATITUDE TRIMMING HERE ---
    if ref_da["latitude"].values[0] > ref_da["latitude"].values[-1]:
        ref_da = ref_da.sel(latitude=slice(60, -60))
        boot_aligned = boot_aligned.sel(latitude=slice(60, -60))
    else:
        ref_da = ref_da.sel(latitude=slice(-60, 60))
        boot_aligned = boot_aligned.sel(latitude=slice(-60, 60))
    # ----------------------------------

    df = compute_taylor_stats_against_reference(ref_da, boot_aligned, sample_dim="iteration")
    if df.empty:
        print(f"Skipping ERA5 Taylor diagram for {comp_type} {label}: no valid bootstrap samples.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"Taylor_Stats_HadGEM3_bootstrap_vs_ERA5_{comp_type}_{label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv",
    )
    pct_path = os.path.join(
        output_dir,
        f"Taylor_Percentiles_HadGEM3_bootstrap_vs_ERA5_{comp_type}_{label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv",
    )
    era5_source = era5_truth.attrs.get("source_path", "ERA5 composite")
    save_path = os.path.join(
        output_dir,
        f"Taylor_HadGEM3_bootstrap_vs_ERA5_{comp_type}_{label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.svg",
    )
    ok = plot_taylor_diagram_from_stats(
        df,
        output_path=save_path,
        stats_csv_path=csv_path,
        percentiles_csv_path=pct_path,
        title=(
            f"Taylor Diagram: HadGEM3 bootstrap {comp_type.upper()} vs ERA5 Reanalysis\n"
            f"{label}; ERA5={os.path.basename(str(era5_source))}"
        ),
        reference_label="ERA5 Reanalysis",
        sample_label="HadGEM3 bootstrap",
        min_corr=0.0,
        min_std=0.0,
        max_std=1.0,
    )
    return save_path if ok else None


def plot_member_bootstrap_taylor_vs_era5(
    member_bootstrap_ds: xr.DataArray,
    comp_type: str,
    label: str,
    output_dir: str,
) -> Optional[str]:
    """Taylor diagram for one bootstrapped composite per HadGEM3 member vs ERA5."""
    if comp_type != "aam":
        return None

    era5_truth = _load_era5_composite_truth(comp_type)
    if era5_truth is None:
        return None

    try:
        ref_da, boot_aligned = _regrid_reference_to_target_grid(era5_truth, member_bootstrap_ds)
    except Exception as e:
        print(f"Skipping member-bootstrap ERA5 Taylor diagram for {label}: alignment failed: {e}")
        return None
    
    # --- ADD LATITUDE TRIMMING HERE ---
    if ref_da["latitude"].values[0] > ref_da["latitude"].values[-1]:
        ref_da = ref_da.sel(latitude=slice(60, -60))
        boot_aligned = boot_aligned.sel(latitude=slice(60, -60))
    else:
        ref_da = ref_da.sel(latitude=slice(-60, 60))
        boot_aligned = boot_aligned.sel(latitude=slice(-60, 60))
    # ----------------------------------

    df = compute_taylor_stats_against_reference(ref_da, boot_aligned, sample_dim="iteration")
    if df.empty:
        print(f"Skipping member-bootstrap ERA5 Taylor diagram for {label}: no valid member samples.")
        return None
    
    if "member" in boot_aligned.coords:
        # Split on 'i' to drop the suffix, then use [1:] to skip the leading 'r'
        member_labels = [str(v).split("i")[0][1:] for v in boot_aligned["member"].values]
    else:
        member_labels = [f"{i + 1}" for i in range(int(boot_aligned.sizes["iteration"]))]
    df["member"] = [member_labels[int(i)] for i in df["iteration"].values]

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"Taylor_Stats_HadGEM3_member_bootstrap_vs_ERA5_{comp_type}_{label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv",
    )
    pct_path = os.path.join(
        output_dir,
        f"Taylor_Percentiles_HadGEM3_member_bootstrap_vs_ERA5_{comp_type}_{label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv",
    )
    era5_source = era5_truth.attrs.get("source_path", "ERA5 composite")
    save_path = os.path.join(
        output_dir,
        f"Taylor_HadGEM3_member_bootstrap_vs_ERA5_{comp_type}_{label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.svg",
    )
    ok = plot_taylor_diagram_from_stats(
        df,
        output_path=save_path,
        stats_csv_path=csv_path,
        percentiles_csv_path=pct_path,
        title=(
            f"Taylor Diagram: HadGEM3 member bootstraps {comp_type.upper()} vs ERA5 Reanalysis\n"
            f"{label}; ERA5={os.path.basename(str(era5_source))}"
        ),
        reference_label="ERA5 Reanalysis",
        sample_label="HadGEM3 member bootstrap",
        min_corr=0.0,
        min_std=0.0,
        max_std=1.0,
        sample_ms=3.5,
        sample_alpha=0.85,
        label_column="member",
        label_fontsize=10.0,
    )
    return save_path if ok else None


def _era5_taylor_output_paths(
    results_dir: str,
    comp_type: str,
    strength_label: str,
) -> tuple[str, str, str]:
    stats_csv = os.path.join(
        results_dir,
        f"Taylor_Stats_HadGEM3_bootstrap_vs_ERA5_{comp_type}_{strength_label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv",
    )
    percentiles_csv = os.path.join(
        results_dir,
        f"Taylor_Percentiles_HadGEM3_bootstrap_vs_ERA5_{comp_type}_{strength_label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv",
    )
    plot_path = os.path.join(
        results_dir,
        f"Taylor_HadGEM3_bootstrap_vs_ERA5_{comp_type}_{strength_label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.svg",
    )
    return stats_csv, percentiles_csv, plot_path


def _plot_or_rebuild_era5_taylor(
    *,
    results_dir: str,
    comp_type: str,
    strength_label: str,
    boot_ds: Optional[xr.DataArray] = None,
) -> bool:
    """Plot HadGEM3-vs-ERA5 Taylor stats from CSV, or rebuild from bootstrap samples."""
    if comp_type != "aam" or strength_label != "all":
        return False

    stats_csv, percentiles_csv, plot_path = _era5_taylor_output_paths(
        results_dir,
        comp_type,
        strength_label,
    )

    if os.path.exists(stats_csv):
        print("[ERA5 TAYLOR] Replotting HadGEM3 bootstrap vs ERA5 from saved CSV:")
        print(f"  CSV: {stats_csv}")
        print(f"  SVG: {plot_path}")
        return _plot_taylor_diagram_from_csv(
            stats_csv_path=stats_csv,
            output_path=plot_path,
            title=(
                f"Taylor Diagram: HadGEM3 bootstrap {comp_type.upper()} vs ERA5 Reanalysis\n"
                f"{strength_label}"
            ),
            reference_label="ERA5 Reanalysis",
            sample_label="HadGEM3 bootstrap",
            percentiles_csv_path=percentiles_csv,
        )

    if boot_ds is not None:
        print("[ERA5 TAYLOR] Saved CSV missing; rebuilding HadGEM3 bootstrap vs ERA5 from bootstrap samples.")
        return plot_hadgem_bootstrap_taylor_vs_era5(
            boot_ds,
            comp_type,
            strength_label,
            results_dir,
        ) is not None

    print(
        "[ERA5 TAYLOR] Cannot create HadGEM3 bootstrap vs ERA5 Taylor diagram in replot mode: "
        f"missing stats CSV and no saved bootstrap samples at {stats_csv}. "
        "Run once with --no-replot to generate the ERA5 stats."
    )
    return False


def plot_bootstrap_taylor_diagram(ref_da, boot_ds, comp_type, label, output_dir, external_ref_name="Ens Mean"):
    """Plot a Taylor diagram for bootstrap iterations against a reference field."""
    os.makedirs(output_dir, exist_ok=True)
    df = compute_taylor_stats_against_reference(ref_da, boot_ds, sample_dim="iteration")
    csv_path = os.path.join(
        output_dir,
        f"Taylor_Stats_{comp_type}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}_{label}.csv",
    )
    save_path = os.path.join(
        output_dir,
        f"Taylor_Bootstrap_{comp_type}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}_{label}.svg",
    )
    return plot_taylor_diagram_from_stats(
        df,
        output_path=save_path,
        stats_csv_path=csv_path,
        title=f"Taylor Diagram: {comp_type.upper()} {label}\nBootstrap iterations vs {external_ref_name}",
        reference_label=external_ref_name,
        sample_label="HadGEM3 bootstrap",
    )


def _plot_taylor_diagram_from_csv(
    *,
    stats_csv_path: str,
    output_path: str,
    title: str,
    reference_label: str,
    sample_label: str,
    percentiles_csv_path: Optional[str] = None,
    min_corr: float = 0.0,
    min_std: float = 0.0,
    max_std: float = 1.1,
) -> bool:
    """Recreate a Taylor diagram from a saved stats CSV."""
    if not os.path.exists(stats_csv_path):
        print(f"[TAYLOR] ERROR: CSV not found: {stats_csv_path}")
        return False

    try:
        stats_df = pd.read_csv(stats_csv_path)
    except Exception as e:
        print(f"[TAYLOR] ERROR: Could not load CSV {stats_csv_path}: {e}")
        return False

    return plot_taylor_diagram_from_stats(
        stats_df,
        output_path=output_path,
        stats_csv_path=stats_csv_path,
        percentiles_csv_path=percentiles_csv_path,
        title=title,
        reference_label=reference_label,
        sample_label=sample_label,
        min_corr=min_corr,
        min_std=min_std,
        max_std=max_std,
        label_column="member"
    )

def bootstrap_pooled_events(event_stack, dim='event', n_iterations=2000, confidence_level=0.95):
    """
    Performs bootstrap resampling across the pooled event dimension.
    
    Parameters:
        event_stack (xr.DataArray): Array containing all individual events.
        dim (str): The name of the dimension containing the pooled events.
        n_iterations (int): Number of bootstrap resamples.
    """
    n_events = event_stack.sizes[dim]
    bootstrap_means = []
    
    print(f"Starting Pooled Event Bootstrap ({n_iterations} iterations over {n_events} events)...")
    
    for i in tqdm.tqdm(range(n_iterations)):
        # 1. Randomly select EVENT indices with replacement
        resample_indices = np.random.choice(n_events, size=n_events, replace=True)
        
        # 2. Extract those specific events and calculate the mean
        # Using **{dim: ...} allows us to dynamically use whatever dimension name you have
        iteration_mean = event_stack.isel(**{dim: resample_indices}).mean(dim, skipna=True)
        
        bootstrap_means.append(iteration_mean)
    
    # 3. Concatenate all 1,000 means into a single DataArray
    boot_ds = xr.concat(bootstrap_means, dim='iteration')
    
    # 4. Calculate Percentiles
    alpha = (1 - confidence_level) / 2
    lower_bound = boot_ds.quantile(alpha, dim='iteration')
    upper_bound = boot_ds.quantile(1 - alpha, dim='iteration')
    
    # Significant if BOTH bounds are positive OR BOTH bounds are negative
    significant_mask = (lower_bound > 0) & (upper_bound > 0) | (lower_bound < 0) & (upper_bound < 0)
    
    return boot_ds, lower_bound, upper_bound, significant_mask


def _select_region(da, region):
    """Select a geographic region from a DataArray by longitude bounds.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data with a longitude dimension
    region : str
        Region name ('all', 'pacific', 'indian', 'atlantic')
    
    Returns
    -------
    xr.DataArray
        Subset of da containing only the selected region's longitudes
    """
    if region == 'all':
        return da
    
    # Normalize dimension names
    da_renamed = da.copy()
    if 'longitude' in da_renamed.dims and 'lon' not in da_renamed.dims:
        da_renamed = da_renamed.rename({'longitude': 'lon'})
    if 'latitude' in da_renamed.dims and 'lat' not in da_renamed.dims:
        da_renamed = da_renamed.rename({'latitude': 'lat'})
    
    lon_min, lon_max = REGION_BOUNDS[region]
    
    if 'lon' not in da_renamed.dims:
        # No longitude dimension, return as-is
        return da_renamed
    
    lon_vals = da_renamed['lon'].values
    orig_lon_range = (float(lon_vals.min()), float(lon_vals.max()))
    
    # Handle wrapping: if lon_min > lon_max and crosses dateline (Pacific case)
    if lon_min > lon_max:
        # Select lon >= lon_min OR lon <= lon_max (crosses dateline)
        mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)
    else:
        # Standard case: lon_min < lon_max
        mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    
    result = da_renamed.isel({'lon': mask})
    new_lon_range = (float(result['lon'].values.min()), float(result['lon'].values.max())) if result.sizes['lon'] > 0 else (None, None)
    
    # Debug output
    if result.sizes['lon'] > 0:
        print(f"[SELECT_REGION] {region}: original lon {orig_lon_range}, target [{lon_min}, {lon_max}], selected {result.sizes['lon']} lons in range {new_lon_range}")
    
    return result

base_dir = os.getcwd()
# AAM_data_path_base = f"{base_dir}/monthly_mean/AAM/"

#climatology_path_base = f"{base_dir}/climatology/"

# CMIP6_path_base = "/gws/nopw/j04/leader_epesc/CMIP6_SinglForcHistSimul" 
# nino34_directory = f"{CMIP6_path_base}/ProcessedFlds/Omon/sst_indices/nino34/historical/HadGEM3-GC31-LL/"
# output_dir = f"{base_dir}/figures/composites/non_tracking_algorithm/"

# Use scratch space and new directory structure due to workspace migration
CMIP6_path_base = "/work/scratch-nopw2/hhhn2"
nino34_directory = f"{CMIP6_path_base}/HadGEM3-GC31-LL/ProcessedFlds/Omon/sst_indices/nino34/historical/"
output_dir = f"{base_dir}/figures/may_2026/bootstrap_per_memberc_vs_era5/"
climatology_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/climatology/"
AAM_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/full/"
u_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/Amon/ua/historical"
uv_data_path_base = f"{CMIP6_path_base}/HadGEM3-GC31-LL/Amon/uv/historical"
ensemble_mean_output_path = f"{CMIP6_path_base}/HadGEM3-GC31-LL/AAM/ensemble_mean_composite/" 
ensemble_results_dir = f"{base_dir}/event_composite_per_member_bootstrap_vs_era5_results/"  # Store composites, significance, active-month data
ref_ERA5_composite_dir = f"/home/users/hhhn2/AAM/test_code/era5/composite_non_tracking/" 


u_level_to_plot = 250.0  # hPa
save_ensemble_mean_netcdf = True  # Save region-specific netCDF files for each run

if replot:
    save_ensemble_mean_netcdf = False

ACTIVE_MONTH_EL_NINO_THRESHOLD = 0.5
ACTIVE_MONTH_LA_NINA_THRESHOLD = -0.5

EVENT_STRENGTH_BINS = (
    ("weak", 0.5, 1.0),
    ("moderate", 1.0, 1.5),
    ("strong", 1.5, None),
)


def _classify_event_strength(peak_amplitude: float) -> Optional[str]:
    """Map an ENSO event peak amplitude to a strength bin."""
    peak_amplitude = float(peak_amplitude)
    if 0.5 <= peak_amplitude < 1.0:
        return "weak"
    if 1.0 <= peak_amplitude < 1.5:
        return "moderate"
    if peak_amplitude >= 1.5:
        return "strong"
    return None


def _filter_events_by_strength(
    date_list: list[tuple[str, str]],
    peak_amplitudes: list[float],
    strength_label: str,
) -> list[tuple[str, str]]:
    """Return only events whose peak amplitude falls inside the requested bin."""
    filtered_events: list[tuple[str, str]] = []
    for event_window, peak_amplitude in zip(date_list, peak_amplitudes):
        if _classify_event_strength(peak_amplitude) == strength_label:
            filtered_events.append(event_window)
    return filtered_events

def _bootstrap_results_filename(
    composite_type: str,
    strength_label: str,
    *,
    results_dir: str,
) -> str:
    strength_str = strength_label if strength_label in ['weak', 'moderate', 'strong', 'all'] else 'all'
    filename = (
        f"composite_{composite_type}_{strength_str}_onset_{args.onset_season}"
        f"_start_{args.composite_start}_region_{args.region}_bootstrap_results.nc"
    )
    return os.path.join(results_dir, filename)


def save_composite_results(
    results_dir: str,
    composite_type: str,  # 'aam', 'latlon', or 'latlev'
    strength_label: str,
    ens_stack: xr.DataArray,
    ens_mean: xr.DataArray,
    significance_mask: np.ndarray,
    bootstrap_lower_bound: Optional[xr.DataArray] = None,
    bootstrap_upper_bound: Optional[xr.DataArray] = None,
    bootstrap_samples: Optional[xr.DataArray] = None,
    active_month_percent: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Save composite results to netCDF with clear filenames.
    
    Parameters
    ----------
    results_dir : str
        Directory to save results (will be created if it doesn't exist)
    composite_type : str
        Type of composite ('aam', 'latlon', 'latlev')
    strength_label : str
        Event strength ('weak', 'moderate', 'strong', or 'all')
    ens_stack : xr.DataArray
        Full ensemble stack (n_members, lat, months) or similar
    ens_mean : xr.DataArray
        Ensemble mean field
    significance_mask : np.ndarray
        Boolean bootstrap significance mask (same shape as ens_mean)
    bootstrap_lower_bound, bootstrap_upper_bound : Optional[xr.DataArray]
        Bootstrap confidence interval bounds
    active_month_percent : Optional[np.ndarray]
        Active-month percentages per month
    metadata : Optional[dict]
        Dictionary of metadata to store as attributes
    
    Returns
    -------
    str
        Path to saved netCDF file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = _bootstrap_results_filename(composite_type, strength_label, results_dir=results_dir)
    
    # Create a dataset to store all results
    ds = xr.Dataset()
    
    # Store ensemble mean
    ens_mean_copy = ens_mean.copy()
    ens_mean_copy.name = f"{composite_type}_mean"
    ds['ens_mean'] = ens_mean_copy
    
    # Store bootstrap significance mask in the same file
    significance_mask_da = xr.DataArray(
        np.asarray(significance_mask, dtype=bool),
        dims=ens_mean.dims,
        coords=ens_mean.coords,
        name='significance_mask'
    )
    ds['significance_mask'] = significance_mask_da

    if bootstrap_lower_bound is not None:
        ds['bootstrap_lower_bound'] = bootstrap_lower_bound
    if bootstrap_upper_bound is not None:
        ds['bootstrap_upper_bound'] = bootstrap_upper_bound
    if bootstrap_samples is not None:
        bootstrap_samples_copy = bootstrap_samples.copy()
        bootstrap_samples_copy.name = "bootstrap_samples"
        ds["bootstrap_samples"] = bootstrap_samples_copy
    
    # Store active-month percentages if available
    if active_month_percent is not None:
        active_pct_da = xr.DataArray(
            active_month_percent,
            dims=['month'],
            coords={'month': ens_mean.coords['month']} if 'month' in ens_mean.coords else None,
            name='active_month_percent'
        )
        ds['active_month_percent'] = active_pct_da
    
    # Store ensemble stack metadata (shape and dims)
    ds.attrs['ensemble_n_members'] = int(ens_stack.sizes['ensemble'])
    ds.attrs['composite_type'] = str(composite_type)
    ds.attrs['strength_label'] = str(strength_label)
    
    # Store additional metadata
    if metadata is not None:
        for key, val in metadata.items():
            try:
                ds.attrs[f'meta_{key}'] = str(val)
            except:
                pass
    
    # Save to netCDF
    ds.to_netcdf(filepath)
    print(f"Saved bootstrap composite results to {filepath}")

    return filepath


def load_composite_results(
    results_dir: str,
    composite_type: str,
    strength_label: str,
) -> Optional[dict]:
    """Load composite results from netCDF file.
    
    Parameters
    ----------
    results_dir : str
        Directory containing results
    composite_type : str
        Type of composite ('aam', 'latlon', 'latlev')
    strength_label : str
        Event strength label
    
    Returns
    -------
    Optional[dict]
        Dictionary with keys: 'ens_mean', 'p_values', 'active_month_percent', 'metadata'
        Returns None if file not found
    """
    strength_str = strength_label if strength_label in ['weak', 'moderate', 'strong', 'all'] else 'all'
    onset_season = getattr(args, 'onset_season', 'all')
    composite_start = getattr(args, 'composite_start', 'onset')
    region = getattr(args, 'region', 'all')
    filepath = _bootstrap_results_filename(composite_type, strength_label, results_dir=results_dir)

    if not os.path.exists(filepath):
        print(f"Results file not found: {filepath}")
        return None
    
    try:
        ds = xr.open_dataset(filepath)
        results = {
            'ens_mean': ds['ens_mean'].copy() if 'ens_mean' in ds else None,
            'significance_mask': ds['significance_mask'].copy() if 'significance_mask' in ds else None,
            'bootstrap_lower_bound': ds['bootstrap_lower_bound'].copy() if 'bootstrap_lower_bound' in ds else None,
            'bootstrap_upper_bound': ds['bootstrap_upper_bound'].copy() if 'bootstrap_upper_bound' in ds else None,
            'bootstrap_samples': ds['bootstrap_samples'].copy() if 'bootstrap_samples' in ds else None,
            'active_month_percent': ds['active_month_percent'].values if 'active_month_percent' in ds else None,
            'metadata': dict(ds.attrs),
        }
        ds.close()
        return results
    except Exception as e:
        print(f"Error loading results from {filepath}: {e}")
        return None


def compute_and_save_composite_significance(
    ens_stack: xr.DataArray,
    composite_type: str,
    strength_label: str,
    results_dir: str,
    active_month_percent: Optional[np.ndarray] = None,
    args: Optional[object] = None,
) -> Tuple[np.ndarray, str, xr.DataArray]:
    """Compute bootstrap significance for a composite and save results.
    
    Parameters
    ----------
    ens_stack : xr.DataArray
        Full ensemble stack with dimension 'ensemble' plus spatial dims
    composite_type : str
        Type of composite ('aam', 'latlon', 'latlev')
    strength_label : str
        Event strength label
    results_dir : str
        Directory to save results
    active_month_percent : Optional[np.ndarray]
        Active-month percentages
    args : Optional[object]
        Arguments object for metadata
    
    Returns
    -------
        Tuple[np.ndarray, str]
        (boolean significance mask array, filepath to saved results)
    """
    
    print(f"\n[BOOTSTRAP SIGNIFICANCE - {composite_type}]")
    
    # 1. Compute the actual Ensemble Mean (this is your reference)
    # Important: If your stack dim is named 'ensemble', we average over it.
    # In a pooled approach, this 'ensemble' dim actually contains ALL events.
    stack_dim = 'ensemble' if 'ensemble' in ens_stack.dims else 'event'
    ens_mean = ens_stack.mean(stack_dim, skipna=True)
    
    if replot:
        print("Replot mode: skipping significance calculation and loading from saved results.")
        loaded_results = load_composite_results(results_dir, composite_type, strength_label)
        if loaded_results is None:
            raise RuntimeError("No saved results found for replotting.")
        significance_mask = loaded_results['significance_mask']
        if significance_mask is None:
            raise RuntimeError("Significance mask not found in loaded results for replotting.")

        print(f"[REPLOT TAYLOR] Entering replot Taylor generation for {composite_type} {strength_label}")
        
        bootstrap_taylor_stats_csv = os.path.join(
            results_dir,
            f"Taylor_Stats_{composite_type}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}_{strength_label}.csv",
        )
        bootstrap_taylor_png = os.path.join(
            results_dir,
            f"Taylor_Bootstrap_{composite_type}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}_{strength_label}.svg",
        )
        print(f"[REPLOT TAYLOR] Attempting bootstrap Taylor replot:")
        print(f"  CSV: {bootstrap_taylor_stats_csv}")
        print(f"  PNG: {bootstrap_taylor_png}")
        
        _plot_taylor_diagram_from_csv(
            stats_csv_path=bootstrap_taylor_stats_csv,
            output_path=bootstrap_taylor_png,
            title=f"Taylor Diagram: {composite_type.upper()} {strength_label}\nBootstrap iterations vs Ensemble Mean",
            reference_label="Ensemble Mean",
            sample_label="HadGEM3 bootstrap",
        )

        _plot_or_rebuild_era5_taylor(
            results_dir=results_dir,
            comp_type=composite_type,
            strength_label=strength_label,
            boot_ds=loaded_results.get("bootstrap_samples"),
        )
        # ADD THIS: Extract the bootstrap mean from loaded results
        bootstrap_mean = None
        if loaded_results.get("bootstrap_samples") is not None:
            bootstrap_mean = loaded_results["bootstrap_samples"].mean("iteration", skipna=True)
            
        return significance_mask, "loaded_from_saved_results", bootstrap_mean

    
    else:
        # 1. Run the Pooled Event Bootstrap
        # Make sure 'dim' matches whatever you named the concatenated dimension 
        # that holds all ~1,300 events (e.g., 'event' or 'ensemble')
        boot_ds, lower_bound, upper_bound, sig_mask = bootstrap_pooled_events(
            ens_stack, dim=stack_dim, n_iterations=2000, confidence_level=0.95 
        )

        plot_bootstrap_taylor_diagram(
            ens_mean,
            boot_ds,
            composite_type,
            strength_label,
            results_dir,
            external_ref_name="Ensemble Mean",
        )

        _plot_or_rebuild_era5_taylor(
            results_dir=results_dir,
            comp_type=composite_type,
            strength_label=strength_label,
            boot_ds=boot_ds,
        )
        
        # Calculate the mean of the bootstrap iterations
        bootstrap_mean = boot_ds.mean("iteration", skipna=True)

        # Save results to NetCDF (including the mask and bounds)
        filepath = save_composite_results(
            results_dir=results_dir,
            composite_type=composite_type,
            strength_label=strength_label,
            ens_stack=ens_stack,
            ens_mean=ens_mean,
            significance_mask=sig_mask,
            bootstrap_lower_bound=lower_bound,
            bootstrap_upper_bound=upper_bound,
            bootstrap_samples=boot_ds if composite_type == 'aam' else None,
            active_month_percent=active_month_percent,
            metadata={'bootstrap': 'pooled_events', 'iterations': 2000}
        )
    
    # RETURN ONLY TWO THINGS: the bootstrap significance mask and the path.
    return sig_mask.values, filepath, bootstrap_mean
    

def detect_enso_state_windows(
    *,
    start_yr: int,
    end_yr: int,
    ensemble_member: str,
    enso_state: str = "el_nino",
    threshold: float = 0.5,
    min_consecutive_months: int = 3,
    require_no_reinitiation: bool = True,
    reinitiation_check_months: int = 12,
    onset_months: "set[int] | None" = None,
) -> list[tuple[str, str]]:
    """Return (onset, end) windows from first month of sustained ENSO state.

    For `enso_state='el_nino'`, onset is the first month in a run where
    Nino3.4 >= threshold for at least `min_consecutive_months`.
    For `enso_state='la_nina'`, onset is the first month in a run where
    Nino3.4 <= threshold for at least `min_consecutive_months`.

    The returned window spans onset month through onset+11.

    When `require_no_reinitiation` is True, El Niño events are further filtered
    so that the first `reinitiation_check_months` months after the event ends
    never rise back above `threshold`.

    If `onset_months` is provided (e.g. {11, 12, 1, 2, 3} for NDJFM), only
    events whose onset falls in one of those calendar months are returned.
    """
    import pandas as pd

    if min_consecutive_months < 1:
        raise ValueError("min_consecutive_months must be >= 1")
    if reinitiation_check_months < 1:
        raise ValueError("reinitiation_check_months must be >= 1")
    if enso_state not in ("el_nino", "la_nina"):
        raise ValueError("enso_state must be 'el_nino' or 'la_nina'")
    if enso_state == "la_nina" and float(threshold) > -0.5:
        raise ValueError("For la_nina, threshold must be lower than -0.5")

    # Need enough data to cover the latest possible event end plus the
    # 12-month no-reinitiation check. For events starting in `end_yr`, end_yr+2
    # is sufficient; this adds a small buffer without pulling unnecessary years.
    search_end_yr = int(end_yr) + 2
    enso_times, enso_vals = get_ENSO_index(start_yr, search_end_yr, ensemble_member=ensemble_member)
    if enso_times is None or enso_vals is None:
        raise RuntimeError(f"No Nino3.4 file found for member {ensemble_member}")

    enso_da = xr.DataArray(enso_vals, coords={"time": enso_times}, dims=("time",)).sortby("time")
    enso_vals_arr = np.asarray(enso_da.values, dtype=float)
    if enso_state == "el_nino":
        in_state = enso_vals_arr >= 0.5
    else:
        in_state = enso_vals_arr <= -0.5

    windows: list[tuple[str, str]] = []
    i = 0
    n = in_state.size
    while i < n:
        if not in_state[i]:
            i += 1
            continue
        j = i
        while j < n and in_state[j]:
            j += 1
        run_len = j - i
        if run_len >= int(min_consecutive_months):
            onset_ts = pd.Timestamp(enso_da["time"].values[i]).to_period("M").to_timestamp()
            if int(onset_ts.year) > int(end_yr):
                i = j
                continue
            if require_no_reinitiation and enso_state == "el_nino":
                post_start = j
                post_end = post_start + int(reinitiation_check_months)
                if post_end > n:
                    i = j
                    continue
                if np.any(in_state[post_start:post_end]):
                    i = j
                    continue
            end_ts = onset_ts + pd.DateOffset(months=11)
            windows.append((onset_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))
        i = j

    if onset_months is not None:
        windows = [
            (onset, end) for onset, end in windows
            if int(onset[5:7]) in onset_months
        ]

    return windows


def _time_value_to_ymd_string(t) -> str:
    """Convert numpy/cftime-like time scalar to YYYY-MM-DD string."""
    if isinstance(t, np.datetime64):
        return str(np.datetime_as_string(t, unit="D"))
    if hasattr(t, "year") and hasattr(t, "month"):
        day = getattr(t, "day", 1)
        return f"{int(t.year):04d}-{int(t.month):02d}-{int(day):02d}"
    return str(t)


def _compute_composite_window_from_onset(
    onset_str: str,
    *,
    composite_months: int,
    composite_start: str,
) -> tuple[str, str]:
    """Return (window_start_ym, window_end_ym) for a detected onset string."""
    import pandas as pd

    if int(composite_months) < 1:
        raise ValueError("composite_months must be >= 1")
    if composite_start not in ("onset", "december_onset_year"):
        raise ValueError("composite_start must be 'onset' or 'december_onset_year'")

    onset_year = int(onset_str[:4])
    onset_month = int(onset_str[5:7])

    if composite_start == "december_onset_year":
        t_start = pd.Timestamp(f"{onset_year}-12-01")
    else:
        t_start = pd.Timestamp(f"{onset_year}-{onset_month:02d}-01")

    t_end = t_start + pd.DateOffset(months=int(composite_months) - 1)
    return t_start.strftime("%Y-%m"), t_end.strftime("%Y-%m")

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

def _prepare_field(da, p_min_hpa, p_max_hpa):
    # zonal mean ONLY if lon exists
    for lon_name in ("longitude", "lon"):
        if lon_name in da.dims:
            da = da.mean(dim=lon_name, skipna=True)

    # vertical integration if level exists
    if "level" in da.dims:
        da = vertical_sum_over_pressure_range(
            da, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )
    return da


def _match_plot_orientation(reference_values, candidate_values):
    """Return candidate_values oriented to match reference_values for contouring."""
    reference_array = np.asarray(reference_values)
    candidate_array = np.asarray(candidate_values)

    if candidate_array.shape == reference_array.shape:
        return candidate_array
    if candidate_array.T.shape == reference_array.shape:
        return candidate_array.T
    if candidate_array.size == reference_array.size:
        return candidate_array.reshape(reference_array.shape)

    raise ValueError(
        f"Cannot align array of shape {candidate_array.shape} to reference shape {reference_array.shape}"
    )


def _plot_and_save_ensemble_mean_aam_composite(
    *,
    ens_stack: xr.DataArray,
    args,
    clim_start_yr: int,
    clim_end_yr: int,
    output_dir: str,
    ensemble_mean_output_path: str,
    region_label: str,
    reinitiation_suffix: str,
    number_of_available_members: int,
    strength_label: str,
    strength_desc: str,
    save_ensemble_mean_netcdf: bool,
    ensemble_results_dir: str,
    active_month_percent: Optional[np.ndarray] = None,
):
    """Plot and optionally save an ensemble-mean AAM composite for one strength bin."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from scipy import stats as _stats
    import os

    ens_mean = ens_stack.mean("ensemble", skipna=True)
    
    #ens_mean = _to_per_latitude_band(ens_mean)

    print(
        f"Plotting ENSEMBLE MEAN {strength_label} composite from "
        f"{ens_stack.sizes['ensemble']} members..."
    )

    lat_dim = "latitude" if "latitude" in ens_mean.dims else "lat"
    lat_vals = ens_mean[lat_dim].values
    month_vals = ens_mean["month"].values

    aam_vals = ens_mean.values
    if ens_mean.dims[0] == "month":
        aam_vals = aam_vals.T

    # -------------------------------
    # SIGNIFICANCE HANDLING (FIXED)
    # -------------------------------
    if not replot:
        # Only compute t-test if we have the full multi-member stack
        combined = ens_stack.transpose("ensemble", lat_dim, "month")
        ttest_result = _stats.ttest_1samp(combined.values, 0.0, axis=0, nan_policy="omit")
        p_vals = np.asarray(ttest_result.pvalue, dtype=float)

        print(
            f"{strength_label.title()} ensemble t-test: shape={p_vals.shape}, "
            f"min p={float(np.nanmin(p_vals)):.4f}, "
            f"p<0.05: {int(np.sum(p_vals < 0.05))} pts"
        )
        
        # Save strength-binned composite significance
        sig_mask_saved, save_path, _ = compute_and_save_composite_significance(
            ens_stack=ens_stack,
            composite_type='aam',
            strength_label=strength_label,
            results_dir=ensemble_results_dir,
            active_month_percent=active_month_percent,
            args=args,
        )
        print(f"AAM {strength_label} composite significance saved to {save_path}")
        
        # Create mask for plotting (1 = insignificant, hatch it)
        insig_mask = np.where(sig_mask_saved, 0, 1)
    else:
        # Replot Mode: Load the boolean mask directly and regenerate Taylor diagrams
        strength_results = load_composite_results(ensemble_results_dir, 'aam', strength_label)
        if strength_results is not None and strength_results.get('significance_mask') is not None:
            sig_mask = strength_results['significance_mask']
            insig_mask = np.where(sig_mask, 0, 1)
        else:
            print(f"  WARNING: Combined bootstrap results not found for {strength_label}. Hatching everything.")
            insig_mask = np.ones_like(aam_vals)
        
        # TAYLOR REPLOT: Generate Taylor diagrams from saved CSV statistics
        if args is not None:
            bootstrap_taylor_stats_csv = os.path.join(
                ensemble_results_dir,
                f"Taylor_Stats_aam_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}_{strength_label}.csv",
            )
            bootstrap_taylor_png = os.path.join(
                ensemble_results_dir,
                f"Taylor_Bootstrap_aam_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}_{strength_label}.svg",
            )
            _plot_taylor_diagram_from_csv(
                stats_csv_path=bootstrap_taylor_stats_csv,
                output_path=bootstrap_taylor_png,
                title=f"Taylor Diagram: AAM {strength_label}\nBootstrap iterations vs Ensemble Mean",
                reference_label="Ensemble Mean",
                sample_label="HadGEM3 bootstrap",
            )

    insig_mask = _match_plot_orientation(aam_vals, insig_mask)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.30)

    vmax = float(np.nanpercentile(np.abs(aam_vals), 98))
    vmax = vmax if vmax > 0 else 1.0
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 13)

    cf = ax.contourf(
        month_vals,
        lat_vals,
        aam_vals,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )

    _abs = max(abs(vmin), abs(vmax))
    order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
    factor = 10 ** order

    cax = fig.add_axes([0.125, 0.06, 0.775, 0.015])
    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")
    _sup = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
    _order_sup = str(order).translate(_sup)
    cbar.set_label(f"AAM anomaly (×10{_order_sup})", size=14)
    tick_levels = list(levels[::2])
    cbar.set_ticks(tick_levels)
    cbar.ax.tick_params(labelsize=11)
    
    # 2. Apply hatching over the areas marked as 1
    if np.any(insig_mask == 1):
        hatches = ax.contourf(
            month_vals,          # Your full X-axis array
            lat_vals,            # Your full Y-axis array
            insig_mask,          # The 2D grid of 0s and 1s
            levels=[0.5, 1.5],   # Draw boundaries only around the 1s
            colors='none',       # Keep background transparent
            hatches=['//'],      # Use dense slashes
            zorder=10
        )
        # 'edgecolor' controls the color of the lines themselves
        import matplotlib.colors as mcolors
        line_color_with_alpha = mcolors.to_rgba('gray', alpha=0.4)
        
        for collection in hatches.collections:
            collection.set_facecolor('none')
            collection.set_edgecolor(line_color_with_alpha)

            if hasattr(collection, 'set_edgecolors'):
                collection.set_edgecolors([line_color_with_alpha])
            collection.set_linewidths([0.0])
    else:
        print("No insignificant points to hatch (all p <= 0.05)")
        
    #Solid black line at the Equator
    ax.axhline(
        y=0, 
        color='black', 
        linestyle='-',    # Solid line
        linewidth=1, 
        zorder=2,
        alpha=0.8
    )
    
    for lat in [-40, -20, 20, 40]:
        ax.axhline(
            y=lat, 
            color='grey', 
            linestyle='--',   # Dashed line
            linewidth=0.8, 
            alpha=0.4,        # Less opacity (lower alpha means more transparent)
            zorder=2
        )

    ax.set_xlabel("Month since onset", fontsize=14)
    ax.set_ylabel("Latitude (°N)", fontsize=14)

    if active_month_percent is not None:
        pct_cmap = LinearSegmentedColormap.from_list(
            "enso_active_pct",
            ["#7c7b66", "#9e833b", "#f39c34", "#e85d3a", "#bd491f"],
        )
        pct_norm = Normalize(vmin=0, vmax=100)
        ax.text(
            -0.06,
            -0.155,
            "ENSO active (%)",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=14,
        )
        for month_idx, (month_val, pct_val) in enumerate(zip(month_vals, active_month_percent)):
            if month_idx % 2 != 0:
                continue
            ax.text(
                month_val,
                -0.19,
                f"{int(pct_val)}%",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=pct_cmap(pct_norm(int(pct_val))),
                clip_on=False,
            )

    ax.set_xlim(1, len(month_vals))
    ax.set_ylim(-60, 60)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

    state_pretty = "El Nino" if args.enso_state == "el_nino" else "La Nina"
    onset_season_label = f" | {args.onset_season.upper()} onsets only" if args.onset_season != "all" else ""
    ax.set_title(
        f"HadGEM3_GC31 {number_of_available_members} members BOOTSTRAP Composite AAM anomaly\n"
        f"({args.p_min}–{args.p_max} hPa) {state_pretty} {strength_desc} events  {args.start_year}–{args.end_year}"
        f"clim {clim_start_yr}–{clim_end_yr} {onset_season_label}"
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"AAM_composite_ENSEMBLE_MEAN_{strength_label}_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.svg",
    )
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Ensemble mean composite plot saved to {out_path}")

    if save_ensemble_mean_netcdf:
        out_nc_path = os.path.join(
            ensemble_mean_output_path,
            f"AAM_composite_ENSEMBLE_MEAN_{strength_label}_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc",
        )
        ens_mean.name = "AAMA"
        ens_mean.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
        ens_mean.to_netcdf(out_nc_path)
        print(f"Ensemble mean composite data saved to {out_nc_path}")

from scipy.ndimage import gaussian_filter

def compute_composite_field(field, date_list, args) -> xr.DataArray:
    """Compute composite field for some variables excluding AAM anomaly 
    for given date_list and args, without plotting."""
    stacked = []
    seen = set()

    for onset_str, _ in date_list:
        if not isinstance(onset_str, str):
            onset_str = _time_value_to_ymd_string(onset_str)

        ym = onset_str[:7]
        if ym in seen:
            continue
        seen.add(ym)

        window_start, window_end = _compute_composite_window_from_onset(
            onset_str,
            composite_months=int(args.composite_months),
            composite_start=str(args.composite_start),
        )

        evt = field.sel(time=slice(window_start, window_end))
        if int(evt.sizes["time"]) < int(args.composite_months):
            continue

        evt = evt.isel(time=slice(0, int(args.composite_months)))
        evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1))
        evt = evt.rename({"time": "month"})

        stacked.append(evt)

    if not stacked:
        return None

    stack = xr.concat(stacked, dim="event")

    # rolling
    if args.rolling_period > 1:
        stack = _circular_rolling_mean(stack, dim="month", window=args.rolling_period)

    return stack.mean("event", skipna=True)

def _process_single_ensemble_member(
    ensemble_member,
    clim_start_yr,
    clim_end_yr,
    args,
    AAM_data_path_base,
    climatology_path_base,
    u_data_path_base,
    uv_data_path_base,
    u_level_to_plot,
):
    """Process a single ensemble member and return results as a dictionary.
    
    Returns dict with keys: 'composite', 'lat_lev_composite', 'lat_lon_composite',
    'u_latlon', 'u_latlev', 'uv_vi', 'onset_dates', 'onset_members', 'peak_amplitudes'
    """
    results = {
        'ensemble_member': ensemble_member,
        'composite': None,
        'lat_lev_composite': None,
        'lat_lon_composite': None,
        'u_latlon': None,
        'u_latlev': None,
        'uv_vi': None,
        'onset_dates': [],
        'onset_members': [],
        'peak_amplitudes': [],
        'member_bootstrap_composite': None,
        'member_bootstrap_lat_lev_composite': None,
        'member_bootstrap_lat_lon_composite': None,
        'composites_by_strength': {label: None for label, _, _ in EVENT_STRENGTH_BINS},
        'lat_lev_composites_by_strength': {label: None for label, _, _ in EVENT_STRENGTH_BINS},
        'lat_lon_composites_by_strength': {label: None for label, _, _ in EVENT_STRENGTH_BINS},
        'enso_active_profile': np.zeros(int(args.composite_months), dtype=int),
        'enso_active_member_count': 0,
    }
    
    try:
        import gc
        # Load AAM dataset (full, will be reused for all composites)
        aam_dataset = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")
        AAM_da = aam_dataset['AAM']
        
        # Normalize dimension names immediately after loading
        if 'latitude' in AAM_da.dims and 'lat' not in AAM_da.dims:
            AAM_da = AAM_da.rename({'latitude': 'lat'})
        if 'longitude' in AAM_da.dims and 'lon' not in AAM_da.dims:
            AAM_da = AAM_da.rename({'longitude': 'lon'})
        
        # DEBUG: Print what region is requested
        if ensemble_member == "r1i1p1f3":
            print(f"\n[ENSEMBLE MEMBER DEBUG] Processing {ensemble_member}")
            print(f"  Region requested: {args.region}")
            print(f"  Full AAM data shape before selection: {AAM_da.shape}")
            lon_coord = 'lon' if 'lon' in AAM_da.dims else 'longitude'
            print(f"  Full AAM lon range: [{float(AAM_da[lon_coord].values.min()):.1f}, {float(AAM_da[lon_coord].values.max()):.1f}]")
        
        # Apply region selection BEFORE zonal integration so ensemble mean is also region-specific
        AAM_da = _select_region(AAM_da, args.region)
        
        print(AAM_da)
        print(AAM_da.dims)
        print(AAM_da.attrs)
        
        #import pdb; pdb.set_trace()
        
        if "longitude" in AAM_da.dims or "lon" in AAM_da.dims:
            AAM_da, _ = _to_per_latitude_band(AAM_da)
            
        u_da = xr.open_dataset(f"{u_data_path_base}/ua_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")['ua']
        # Normalize dimension names
        if 'latitude' in u_da.dims and 'lat' not in u_da.dims:
            u_da = u_da.rename({'latitude': 'lat'})
        if 'longitude' in u_da.dims and 'lon' not in u_da.dims:
            u_da = u_da.rename({'longitude': 'lon'})
        
        uv_da = xr.open_dataset(f"{uv_data_path_base}/uv_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")['uv']
        # Normalize dimension names
        if 'latitude' in uv_da.dims and 'lat' not in uv_da.dims:
            uv_da = uv_da.rename({'latitude': 'lat'})
        if 'longitude' in uv_da.dims and 'lon' not in uv_da.dims:
            uv_da = uv_da.rename({'longitude': 'lon'})

        # Climatology
        clim_da = xr.open_dataset(
            f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
        
        # Normalize dimension names immediately after loading
        if 'latitude' in clim_da.dims and 'lat' not in clim_da.dims:
            clim_da = clim_da.rename({'latitude': 'lat'})
        if 'longitude' in clim_da.dims and 'lon' not in clim_da.dims:
            clim_da = clim_da.rename({'longitude': 'lon'})

        # Select region before zonal mean so that zonal mean is computed over regional lons only
        u_da_regional = _select_region(u_da, args.region)
        u_da_zonal_band = u_da_regional.mean(dim="longitude", skipna=True) if "longitude" in u_da_regional.dims else u_da_regional
        
        # Use the per-latitude-band + zonal-integral climatology
        # CRITICAL: Apply region selection to climatology BEFORE zonal integration to match main AAM data
        clim_aam_data = clim_da['AAM']
        clim_aam_data = _select_region(clim_aam_data, args.region)
        if 'longitude' in clim_aam_data.dims or 'lon' in clim_aam_data.dims:
            clim_da, _ = _to_per_latitude_band(clim_aam_data)
        else:
            clim_da = clim_aam_data
        
        # Detect ENSO state windows from Nino3.4
        _onset_months_map = {"all": None, "ndjfm": {11, 12, 1, 2, 3}}
        onset_months_filter = _onset_months_map[args.onset_season]
        # Hardcoded detection: always use 0.5 threshold + 12-month reinitiation lookahead
        try:
            date_list = detect_enso_state_windows(
                start_yr=int(args.start_year),
                end_yr=int(args.end_year),
                ensemble_member=ensemble_member,
                enso_state=str(args.enso_state),
                threshold=0.5,  # hardcoded: always use 0.5 for state boundary
                min_consecutive_months=int(args.min_elnino_months),
                require_no_reinitiation=True,  # hardcoded: always exclude events with reactivation
                reinitiation_check_months=12,  # hardcoded: check 12 months ahead for reactivation
                onset_months=onset_months_filter,
            )
        except RuntimeError as e:
            print(f"Skipping member {ensemble_member}: {e}")
            return results

        if not date_list:
            return results
            
        # Collect onset dates for histogram
        for _onset, _end in date_list:
            onset_str = _onset if isinstance(_onset, str) else _time_value_to_ymd_string(_onset)
            results['onset_dates'].append(onset_str)
            results['onset_members'].append(ensemble_member)
        
        # Record event peak Nino3.4 amplitudes
        try:
            enso_times_all, enso_vals_all = get_ENSO_index(int(args.start_year), int(args.end_year) + 3, ensemble_member=ensemble_member)
            if enso_times_all is not None and enso_vals_all is not None:
                enso_index = pd.to_datetime(enso_times_all)
                enso_series = pd.Series(np.asarray(enso_vals_all, dtype=float), index=enso_index.to_period('M'))
                member_has_valid_event = False
                member_active_profile = np.zeros(int(args.composite_months), dtype=bool)
                member_active_profile_total = np.zeros(int(args.composite_months), dtype=int)
                member_valid_event_count = 0
                valid_event_windows: list[tuple[str, str]] = []
                for _onset, _end in date_list:
                    onset_str = _onset if isinstance(_onset, str) else _time_value_to_ymd_string(_onset)
                    window_start, window_end = _compute_composite_window_from_onset(
                        onset_str,
                        composite_months=int(args.composite_months),
                        composite_start=str(args.composite_start),
                    )
                    start_period = pd.Period(window_start, freq='M')
                    end_period = pd.Period(window_end, freq='M')
                    ev_vals = enso_series.loc[start_period:end_period].values
                    if ev_vals.size < int(args.composite_months):
                        continue
                    member_has_valid_event = True
                    if args.enso_state == 'el_nino':
                        peak = float(np.nanmax(ev_vals))
                        active_profile = np.asarray(ev_vals[:int(args.composite_months)] >= ACTIVE_MONTH_EL_NINO_THRESHOLD, dtype=bool)
                    else:
                        peak = float(abs(np.nanmin(ev_vals)))
                        active_profile = np.asarray(ev_vals[:int(args.composite_months)] <= ACTIVE_MONTH_LA_NINA_THRESHOLD, dtype=bool)
                    results['peak_amplitudes'].append(peak)
                    member_active_profile_total += np.asarray(active_profile, dtype=int)
                    member_valid_event_count += 1
                    member_active_profile |= active_profile
                    valid_event_windows.append((onset_str, _end))
                results['enso_active_profile'] = member_active_profile.astype(int)
                results['enso_active_member_count'] = 1 if member_has_valid_event else 0
                results['enso_active_profile_total'] = member_active_profile_total
                results['enso_valid_event_count'] = member_valid_event_count
                results['_valid_event_windows'] = valid_event_windows
        except Exception:
            pass

        valid_event_windows = list(results.get('_valid_event_windows', []))
        strength_date_lists = {label: [] for label, _, _ in EVENT_STRENGTH_BINS}
        strength_active_profiles = {label: np.zeros(int(args.composite_months), dtype=int) for label, _, _ in EVENT_STRENGTH_BINS}
        strength_event_counts = {label: 0 for label, _, _ in EVENT_STRENGTH_BINS}
        
        # Classify events into strength bins and compute active profiles per bin
        for event_idx, (event_window, peak_amplitude) in enumerate(zip(valid_event_windows, results['peak_amplitudes'])):
            strength_label = _classify_event_strength(peak_amplitude)
            if strength_label is not None:
                strength_date_lists[strength_label].append(event_window)
                strength_event_counts[strength_label] += 1
                # Recompute active profile for this strength bin
                onset_str = event_window[0]
                window_start, window_end = _compute_composite_window_from_onset(
                    onset_str,
                    composite_months=int(args.composite_months),
                    composite_start=str(args.composite_start),
                )
                start_period = pd.Period(window_start, freq='M')
                end_period = pd.Period(window_end, freq='M')
                ev_vals = enso_series.loc[start_period:end_period].values
                if ev_vals.size == int(args.composite_months):
                    if args.enso_state == 'el_nino':
                        active_profile = np.asarray(ev_vals[:int(args.composite_months)] >= ACTIVE_MONTH_EL_NINO_THRESHOLD, dtype=int)
                    else:
                        active_profile = np.asarray(ev_vals[:int(args.composite_months)] <= ACTIVE_MONTH_LA_NINA_THRESHOLD, dtype=int)
                    strength_active_profiles[strength_label] += active_profile
        
        results['strength_active_profiles'] = strength_active_profiles
        results['strength_event_counts'] = strength_event_counts
        
        # Process fields
        p_selected, _ = pressure_range_in_coord_units(u_da.plev, p_min_hpa=u_level_to_plot, p_max_hpa=float(args.p_max))
        u_level = u_da.sel(plev=p_selected, method="nearest")
        u_zm = u_da_zonal_band
        
        aam_latlon = vertical_sum_over_pressure_range(AAM_da, p_min_hpa=float(args.p_min), p_max_hpa=float(args.p_max), level_dim="level")
        aam_latlev = AAM_da.sum(dim="longitude", skipna=True) if "longitude" in AAM_da.dims else AAM_da
        uv_vi = vertical_sum_over_pressure_range(uv_da, p_min_hpa=float(args.p_min), p_max_hpa=float(args.p_max), level_dim="plev")
        
        u_latlon_comp = compute_composite_field(u_level, date_list=date_list, args=args)
        u_latlvl_zm_comp = compute_composite_field(u_zm, date_list=date_list, args=args)
        uv_latlev_comp = compute_composite_field(uv_vi, date_list, args)
        
        if u_latlon_comp is not None:
            results['u_latlon'] = u_latlon_comp.expand_dims({"ensemble": [ensemble_member]})
        if u_latlvl_zm_comp is not None:
            results['u_latlev'] = u_latlvl_zm_comp.expand_dims({"ensemble": [ensemble_member]})
        if uv_latlev_comp is not None:
            results['uv_vi'] = uv_latlev_comp.expand_dims({"ensemble": [ensemble_member]})
        
        # Step 4: Composite AAM anomaly
        comp = composite_propagating_years_no_plot(
            AAM_da,
            wind_da=None,
            date_list=date_list,
            clim_da=clim_da,
            clim_start_yr=clim_start_yr,
            clim_end_yr=clim_end_yr,
            p_min_hpa=float(args.p_min),
            p_max_hpa=float(args.p_max),
            enso_state=str(args.enso_state),
            rolling_period=int(args.rolling_period),
            composite_months=int(args.composite_months),
            composite_start=str(args.composite_start),
            onset_season=str(args.onset_season),
        )
        if comp is not None:
            results['composite'] = comp.expand_dims({"ensemble": [ensemble_member]})

        if comp is not None and getattr(args, "member_bootstrap_taylor_vs_era5", False):
            member_number_match = re.search(r"r(\d+)i", str(ensemble_member))
            member_number = int(member_number_match.group(1)) if member_number_match else 0
            member_bootstrap_comp = composite_propagating_years_no_plot(
                AAM_da,
                wind_da=None,
                date_list=date_list,
                clim_da=clim_da,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                p_min_hpa=float(args.p_min),
                p_max_hpa=float(args.p_max),
                enso_state=str(args.enso_state),
                rolling_period=int(args.rolling_period),
                composite_months=int(args.composite_months),
                composite_start=str(args.composite_start),
                onset_season=str(args.onset_season),
                bootstrap_events=True,
                bootstrap_seed=10_000 + member_number,
            )
            if member_bootstrap_comp is not None:
                results['member_bootstrap_composite'] = member_bootstrap_comp.expand_dims({"iteration": [0]})

        for strength_label, strength_dates in strength_date_lists.items():
            if not strength_dates:
                continue
            strength_comp = composite_propagating_years_no_plot(
                AAM_da,
                wind_da=None,
                date_list=strength_dates,
                clim_da=clim_da,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                p_min_hpa=float(args.p_min),
                p_max_hpa=float(args.p_max),
                enso_state=str(args.enso_state),
                rolling_period=int(args.rolling_period),
                composite_months=int(args.composite_months),
                composite_start=str(args.composite_start),
                onset_season=str(args.onset_season),
            )
            if strength_comp is not None:
                results['composites_by_strength'][strength_label] = strength_comp.expand_dims({"ensemble": [ensemble_member]})
        
        def _extract_and_average_spatial_windows(anom_data, dates):
            stacked = []
            seen = set()
            for onset_str, _ in dates:
                if not isinstance(onset_str, str):
                    onset_str = _time_value_to_ymd_string(onset_str)
                ym = onset_str[:7]
                if ym in seen: continue
                seen.add(ym)
                
                w_start, w_end = _compute_composite_window_from_onset(
                    onset_str,
                    composite_months=int(args.composite_months),
                    composite_start=str(args.composite_start),
                )
                
                evt = anom_data.sel(time=slice(w_start, w_end))
                if int(evt.sizes["time"]) < int(args.composite_months): continue
                
                evt = evt.isel(time=slice(0, int(args.composite_months)))
                evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1, dtype=int))
                if "month" in evt.coords: evt = evt.drop_vars("month")
                evt = evt.rename({"time": "month"})
                stacked.append(evt)
            
            if not stacked: return None
            
            stack = xr.concat(stacked, dim="event")
            rp = int(args.rolling_period)
            if rp > 1:
                n_month = int(stack.sizes["month"])
                if rp <= n_month:
                    left = rp // 2
                    right = rp - left - 1
                    _rolled = [stack.roll(month=-offset, roll_coords=False) for offset in range(-left, right + 1)]
                    stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)
            
            comp = stack.mean("event", skipna=True)
            comp = comp.rename({"month": "time"})
            
            # Normalize dims
            if 'latitude' in comp.dims and 'lat' not in comp.dims: comp = comp.rename({'latitude': 'lat'})
            if 'longitude' in comp.dims and 'lon' not in comp.dims: comp = comp.rename({'longitude': 'lon'})
            
            return comp.expand_dims({"ensemble": [ensemble_member]})
        
        # Step 5: Latitude×level composite
        aam_full = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
        aam_full = _select_region(aam_full, args.region)
        if "longitude" in aam_full.dims or "lon" in aam_full.dims:
            aam_full = _to_per_latitude_band(aam_full)
        
        clim_full = clim_da["AAM"] if isinstance(clim_da, xr.Dataset) and "AAM" in clim_da else clim_da
        # Normalize dimensions
        if 'latitude' in clim_full.dims and 'lat' not in clim_full.dims:
            clim_full = clim_full.rename({'latitude': 'lat'})
        if 'longitude' in clim_full.dims and 'lon' not in clim_full.dims:
            clim_full = clim_full.rename({'longitude': 'lon'})
        
        aam_full_reindexed, clim_on_time_full = _reindex_to_climatology_dims(aam_full, clim_full)
        anom_full_latlev = aam_full_reindexed - clim_on_time_full
        
        # Generate 'all' events composite
        results['lat_lev_composite'] = _extract_and_average_spatial_windows(anom_full_latlev, date_list)
        
        # Generate strength-binned composites
        for label, dates in strength_date_lists.items():
            if dates:
                results['lat_lev_composites_by_strength'][label] = _extract_and_average_spatial_windows(anom_full_latlev, dates)
        
        # Explicit memory flush
        del aam_full, clim_full, aam_full_reindexed, clim_on_time_full, anom_full_latlev
        gc.collect()
        
        # Step 6: LAT×LON composite - RELOAD fresh data (AAM_da has been modified by earlier operations)

        try:
            aam_full_latlon_dataset = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")
            aam_full_latlon = aam_full_latlon_dataset['AAM']
        except Exception as e:
            print(f"Error loading full AAM data for lat×lon composite for {ensemble_member}: {e}")
            if hasattr(aam_dataset, 'close'): aam_dataset.close()
            del aam_dataset
            gc.collect()
            return results
        
        # Normalize dimension names immediately after loading
        if 'latitude' in aam_full_latlon.dims and 'lat' not in aam_full_latlon.dims:
            aam_full_latlon = aam_full_latlon.rename({'latitude': 'lat'})
        if 'longitude' in aam_full_latlon.dims and 'lon' not in aam_full_latlon.dims:
            aam_full_latlon = aam_full_latlon.rename({'longitude': 'lon'})
            
        clim_latlon_dataset = xr.open_dataset(f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
        clim_full_latlon = clim_latlon_dataset['AAM']
        if 'latitude' in clim_full_latlon.dims and 'lat' not in clim_full_latlon.dims: clim_full_latlon = clim_full_latlon.rename({'latitude': 'lat'})
        if 'longitude' in clim_full_latlon.dims and 'lon' not in clim_full_latlon.dims: clim_full_latlon = clim_full_latlon.rename({'longitude': 'lon'})
        
        aam_full_latlon = _select_region(aam_full_latlon, args.region)
        aam_vs = vertical_sum_over_pressure_range(aam_full_latlon, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
        
        clim_full_data = _select_region(clim_full_latlon, args.region)
        clim_vs = vertical_sum_over_pressure_range(clim_full_data, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
        
        # Explicit file close to free RAM overhead
        aam_full_latlon_dataset.close()
        clim_latlon_dataset.close()
        del aam_full_latlon_dataset, clim_latlon_dataset
        
        if 'latitude' in aam_vs.dims and 'lat' not in aam_vs.dims: aam_vs = aam_vs.rename({'latitude': 'lat'})
        if 'longitude' in aam_vs.dims and 'lon' not in aam_vs.dims: aam_vs = aam_vs.rename({'longitude': 'lon'})
        if 'latitude' in clim_vs.dims and 'lat' not in clim_vs.dims: clim_vs = clim_vs.rename({'latitude': 'lat'})
        if 'longitude' in clim_vs.dims and 'lon' not in clim_vs.dims: clim_vs = clim_vs.rename({'longitude': 'lon'})
        
        aam_full_latlon_r, clim_on_time = _reindex_to_climatology_dims(aam_vs, clim_vs)
        anom_full_latlon = aam_full_latlon_r - clim_on_time
        
        # Generate 'all' events composite
        results['lat_lon_composite'] = _extract_and_average_spatial_windows(anom_full_latlon, date_list)
        
        # Generate strength-binned composites
        for label, dates in strength_date_lists.items():
            if dates:
                results['lat_lon_composites_by_strength'][label] = _extract_and_average_spatial_windows(anom_full_latlon, dates)
                
        # **Explicit cleanup to free memory before returning results**
        del aam_full_latlon, clim_full_latlon, aam_vs, clim_vs, aam_full_latlon_r, clim_on_time, anom_full_latlon
        try:
            if hasattr(aam_dataset, 'close'): aam_dataset.close()
        except: pass
        try:
            if hasattr(clim_da, 'close'): clim_da.close()
        except: pass
        
        # Final GC run for the member
        gc.collect()       
        
        # stacked_full = []
        # seen_ev = set()
        # for onset_str, _ in date_list:
        #     if not isinstance(onset_str, str):
        #         onset_str = _time_value_to_ymd_string(onset_str)
        #     ym = onset_str[:7]
        #     if ym in seen_ev:
        #         continue
        #     seen_ev.add(ym)
        #     window_start, window_end = _compute_composite_window_from_onset(
        #         onset_str,
        #         composite_months=int(args.composite_months),
        #         composite_start=str(args.composite_start),
        #     )
        #     evt = anom_full.sel(time=slice(window_start, window_end))
        #     if int(evt.sizes["time"]) < int(args.composite_months):
        #         continue
        #     evt = evt.isel(time=slice(0, int(args.composite_months)))
        #     evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1, dtype=int))
        #     if "month" in evt.coords:
        #         evt = evt.drop_vars("month")
        #     evt = evt.rename({"time": "month"})
        #     stacked_full.append(evt)
        
        # if stacked_full:
        #     full_stack = xr.concat(stacked_full, dim="event")
        #     rp = int(args.rolling_period)
        #     if rp > 1:
        #         n_month = int(full_stack.sizes["month"])
        #         if rp <= n_month:
        #             left = rp // 2
        #             right = rp - left - 1
        #             _rolled = [full_stack.roll(month=-offset, roll_coords=False) for offset in range(-left, right + 1)]
        #             full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)
            
        #     composite_full = full_stack.mean("event", skipna=True)
        #     composite_full = composite_full.rename({"month": "time"})
            
        #     # Normalize dimension names before storing in results
        #     if 'latitude' in composite_full.dims and 'lat' not in composite_full.dims:
        #         composite_full = composite_full.rename({'latitude': 'lat'})
            
        #     results['lat_lev_composite'] = composite_full.expand_dims({"ensemble": [ensemble_member]})
        
        # # Step 6: LAT×LON composite - RELOAD fresh data (AAM_da has been modified by earlier operations)
        # # We need the full unmodified data with longitude dimension for lat×lon composite
        # try:
        #     aam_full_latlon_dataset = xr.open_dataset(f"{AAM_data_path_base}AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")
        #     aam_full_latlon = aam_full_latlon_dataset['AAM']
        # except Exception as e:
        #     print(f"Error loading full AAM data for lat×lon composite for {ensemble_member}: {e}")
        #     # Explicit cleanup even on error
        #     if hasattr(aam_dataset, 'close'):
        #         aam_dataset.close()
        #     del aam_dataset
        #     gc.collect()
        #     return results
        
        # # Normalize dimension names immediately after loading
        # if 'latitude' in aam_full_latlon.dims and 'lat' not in aam_full_latlon.dims:
        #     aam_full_latlon = aam_full_latlon.rename({'latitude': 'lat'})
        # if 'longitude' in aam_full_latlon.dims and 'lon' not in aam_full_latlon.dims:
        #     aam_full_latlon = aam_full_latlon.rename({'longitude': 'lon'})
        
        # # Load climatology for lat×lon composite
        # clim_latlon_dataset = xr.open_dataset(f"{climatology_path_base}AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
        # clim_full_latlon = clim_latlon_dataset['AAM']
        # # Normalize dimension names immediately after loading
        # if 'latitude' in clim_full_latlon.dims and 'lat' not in clim_full_latlon.dims:
        #     clim_full_latlon = clim_full_latlon.rename({'latitude': 'lat'})
        # if 'longitude' in clim_full_latlon.dims and 'lon' not in clim_full_latlon.dims:
        #     clim_full_latlon = clim_full_latlon.rename({'longitude': 'lon'})
        
        # if ensemble_member == "r1i1p1f3":
        #     print(f"\n[LAT×LON DEBUG] Processing lat×lon composite for {ensemble_member}")
        #     lon_coord = 'lon' if 'lon' in aam_full_latlon.dims else ('longitude' if 'longitude' in aam_full_latlon.dims else None)
        #     if lon_coord:
        #         print(f"  aam_full_latlon BEFORE region selection: shape={aam_full_latlon.shape}, lon range=[{float(aam_full_latlon[lon_coord].values.min()):.1f}, {float(aam_full_latlon[lon_coord].values.max()):.1f}]")
        #     else:
        #         print(f"  aam_full_latlon BEFORE region selection: shape={aam_full_latlon.shape}, dimensions={aam_full_latlon.dims}")
        
        # aam_full_latlon = _select_region(aam_full_latlon, args.region)
        
        # if ensemble_member == "r1i1p1f3":
        #     lon_coord = 'lon' if 'lon' in aam_full_latlon.dims else ('longitude' if 'longitude' in aam_full_latlon.dims else None)
        #     if lon_coord:
        #         print(f"  aam_full_latlon AFTER region selection: shape={aam_full_latlon.shape}, lon range=[{float(aam_full_latlon[lon_coord].values.min()):.1f}, {float(aam_full_latlon[lon_coord].values.max()):.1f}]")
        #     else:
        #         print(f"  aam_full_latlon AFTER region selection: shape={aam_full_latlon.shape}, dimensions={aam_full_latlon.dims}")
        # aam_vs = vertical_sum_over_pressure_range(aam_full_latlon, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
        
        # # CRITICAL: Apply region selection to climatology BEFORE vertical sum to match main data
        # clim_full_data = clim_full_latlon  # already extracted above
        # clim_full_data = _select_region(clim_full_data, args.region)
        # clim_vs = vertical_sum_over_pressure_range(clim_full_data, p_min_hpa=args.p_min, p_max_hpa=args.p_max, level_dim="level")
        
        # # Explicitly close datasets to free memory
        # aam_full_latlon_dataset.close()
        # clim_latlon_dataset.close()
        # del aam_full_latlon_dataset, clim_latlon_dataset
        
        # # Normalize dimensions to match main data
        # if 'latitude' in aam_vs.dims and 'lat' not in aam_vs.dims:
        #     aam_vs = aam_vs.rename({'latitude': 'lat'})
        # if 'longitude' in aam_vs.dims and 'lon' not in aam_vs.dims:
        #     aam_vs = aam_vs.rename({'longitude': 'lon'})
        # if 'latitude' in clim_vs.dims and 'lat' not in clim_vs.dims:
        #     clim_vs = clim_vs.rename({'latitude': 'lat'})
        # if 'longitude' in clim_vs.dims and 'lon' not in clim_vs.dims:
        #     clim_vs = clim_vs.rename({'longitude': 'lon'})
        
        # aam_full_latlon_r, clim_on_time = _reindex_to_climatology_dims(aam_vs, clim_vs)
        # anom_full_latlon = aam_full_latlon_r - clim_on_time
        
        # stacked_full = []
        # seen_ev = set()
        # for onset_str, _ in date_list:
        #     if not isinstance(onset_str, str):
        #         onset_str = _time_value_to_ymd_string(onset_str)
        #     ym = onset_str[:7]
        #     if ym in seen_ev:
        #         continue
        #     seen_ev.add(ym)
        #     window_start, window_end = _compute_composite_window_from_onset(
        #         onset_str,
        #         composite_months=int(args.composite_months),
        #         composite_start=str(args.composite_start),
        #     )
        #     evt = anom_full_latlon.sel(time=slice(window_start, window_end))
        #     if int(evt.sizes["time"]) < int(args.composite_months):
        #         continue
        #     evt = evt.isel(time=slice(0, int(args.composite_months)))
        #     evt = evt.assign_coords(time=np.arange(1, int(args.composite_months) + 1, dtype=int))
        #     if "month" in evt.coords:
        #         evt = evt.drop_vars("month")
        #     evt = evt.rename({"time": "month"})
        #     stacked_full.append(evt)
        
        # if stacked_full:
        #     full_stack = xr.concat(stacked_full, dim="event")
        #     rp = int(args.rolling_period)
        #     if rp > 1:
        #         n_month = int(full_stack.sizes["month"])
        #         if rp <= n_month:
        #             left = rp // 2
        #             right = rp - left - 1
        #             _rolled = [full_stack.roll(month=-offset, roll_coords=False) for offset in range(-left, right + 1)]
        #             full_stack = xr.concat(_rolled, dim="_roll").mean("_roll", skipna=True)
            
        #     composite_full = full_stack.mean("event", skipna=True)
        #     composite_full = composite_full.rename({"month": "time"})
            
        #     # Normalize dimension names before storing in results
        #     if 'latitude' in composite_full.dims and 'lat' not in composite_full.dims:
        #         composite_full = composite_full.rename({'latitude': 'lat'})
        #     if 'longitude' in composite_full.dims and 'lon' not in composite_full.dims:
        #         composite_full = composite_full.rename({'longitude': 'lon'})
            
        #     results['lat_lon_composite'] = composite_full.expand_dims({"ensemble": [ensemble_member]})
        
        # # **Explicit cleanup to free memory before returning results**
        # # Close all opened datasets
        # try:
        #     if hasattr(aam_dataset, 'close'):
        #         aam_dataset.close()
        # except:
        #     pass
        # try:
        #     if hasattr(clim_da, 'close'):
        #         clim_da.close()
        # except:
        #     pass
        
        # # Force garbage collection to free intermediate arrays
        # gc.collect()
    
        
    except Exception as e:
        print(f"Error processing ensemble member {ensemble_member}: {e}")
        import traceback
        traceback.print_exc()
        # Attempt cleanup even on error
        try:
            if hasattr(aam_dataset, 'close'):
                aam_dataset.close()
        except:
            pass
        gc.collect()
    
    return results

def compute_spatial_per_member_taylor(
    *,
    member_composites_list: list,
    era5_reference: xr.DataArray,
    comp_type: str,
    output_dir: str,
    strength_label: str = "all",
) -> None:
    """
    Computes per-member Taylor statistics against ERA5 for 3D spatial fields 
    (latlon or latlev) and outputs the corresponding Taylor diagram and CSVs.
    """
    import os
    print(f"\n[SPATIAL MEMBER TAYLOR] Generating per-member Taylor stats for {comp_type}...")
    
    if not member_composites_list:
        print(f"[SPATIAL MEMBER TAYLOR] No member composites available for {comp_type}. Skipping.")
        return

    member_stats = []
    
    # 1. Standardize and Regrid ERA5 to HadGEM3 grid ONCE
    try:
        print("  [SPATIAL MEMBER TAYLOR] Regridding ERA5 to match HadGEM3 resolution...")
        template_da = _standardize_taylor_dims(member_composites_list[0].copy())
        era5_std = _standardize_taylor_dims(era5_reference.copy())
        
        # Build interpolation dictionary for shared spatial dimensions
        interp_coords = {}
        for dim in era5_std.dims:
            if dim in template_da.coords:
                interp_coords[dim] = template_da[dim]
                
        # Interpolate ERA5 onto HadGEM3 coordinates
        try:
            era5_regridded = era5_std.interp(interp_coords, method="linear")
        except Exception:
            era5_regridded = era5_std.reindex(interp_coords, method="nearest")
        
        # Apply the exact same latitude trimming used in your AAM Taylor functions
        if era5_regridded["latitude"].values[0] > era5_regridded["latitude"].values[-1]:
            era5_regridded = era5_regridded.sel(latitude=slice(60, -60))
        else:
            era5_regridded = era5_regridded.sel(latitude=slice(-60, 60))
            
        if comp_type == "latlev":
            try:
                # template_da contains the 18 HadGEM3 levels
                era5_regridded = ml_to_pl_era5(era5_regridded, template_da)
                print("  [SPATIAL MEMBER TAYLOR] Vertically aligned ERA5 L137 to HadGEM3 pressure levels.")
            except Exception as e:
                print(f"  [WARNING] Vertical alignment failed for latlev: {e}")
            
    except Exception as e:
        print(f"  [SPATIAL MEMBER TAYLOR] Regridding failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Loop through every individual member composite
    for idx, mem_da in enumerate(member_composites_list):
        m_da = mem_da.copy()
        m_da = _standardize_taylor_dims(m_da)
        
        # Apply the matching latitude trim to the HadGEM3 member
        if m_da["latitude"].values[0] > m_da["latitude"].values[-1]:
            m_da = m_da.sel(latitude=slice(60, -60))
        else:
            m_da = m_da.sel(latitude=slice(-60, 60))
            
        try:
            #import pdb; pdb.set_trace()
            # Align them precisely to guarantee flat arrays match point-for-point
            m_da, ref_aligned = xr.align(m_da, era5_regridded, join="inner")
            
            # Flatten to 1D arrays for Taylor statistics
            member_vector = m_da.values.flatten()
            ref_vector = ref_aligned.values.flatten()
            
            # Remove any NaN values safely
            valid_mask = ~np.isnan(member_vector) & ~np.isnan(ref_vector)
            if not np.any(valid_mask):
                continue
            
            
            # 1. Reference array stays 1D over space ('points')
            ref_da_wrapped = xr.DataArray(ref_vector[valid_mask], dims=("points",))
            # 2. Expand member array to 2D: adding a dummy 'iteration' dimension of size 1
            m_da_wrapped = xr.DataArray(
                member_vector[valid_mask][np.newaxis, :], 
                dims=("iteration", "points"),
                coords={"iteration": [0]}
            )
            
            # 3. Pass to the function in the correct order: (Reference, Sample)
            stats_df = compute_taylor_stats_against_reference(
                ref_da_wrapped, 
                m_da_wrapped, 
                sample_dim="iteration"
            )
            
            # 2. Extract the first row as a Series
            stats = stats_df.iloc[0]
            
            # 3. Safely read your function's actual keys ('corr' and 'std')
            corr_val = float(stats["corr"])
            std_val = float(stats["std"])
            
            # 4. Calculate the missing normalized RMSE geometrically
            rmse_val = np.sqrt(max(0.0, 1.0 + std_val**2 - 2.0 * std_val * corr_val))
            
            member_stats.append({
                "iteration": idx,
                "corr": corr_val,
                "std": std_val,
                "rmse": rmse_val,
                "centered_rms": np.nan,
                "member": str(int(idx+1)),
            })
        except Exception as e:
            print(f"  [SPATIAL MEMBER TAYLOR] Alignment failed for member {idx}: {e}")
            continue

    if not member_stats:
        print(f"  [SPATIAL MEMBER TAYLOR] No valid stats calculated for {comp_type}.")
        return

    df_stats = pd.DataFrame(member_stats)
    
    # 3. Save Statistics to CSV
    csv_path = os.path.join(
        output_dir, 
        f"Taylor_Stats_HadGEM3_member_bootstrap_vs_ERA5_{comp_type}_{strength_label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.csv"
    )
    df_stats.to_csv(csv_path, index=False)
    print(f"  Saved member Taylor statistics to {csv_path}")
    
    # 4. Plot and Save Taylor Diagram SVG
    svg_path = os.path.join(
        output_dir, 
        f"Taylor_HadGEM3_member_bootstrap_vs_ERA5_{comp_type}_{strength_label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.svg"
    )
    
    try:
        plot_taylor_diagram_from_stats(
            stats_df=df_stats,
            output_path=svg_path,
            title=f"HadGEM3 Members vs ERA5 ({comp_type.upper()})",
            reference_label="ERA5 Reanalysis",
            sample_label="HadGEM3 Member",
            min_corr=0.0,
            min_std=0.0,
            max_std=1.2,
            sample_ms=3.5,
            sample_alpha=0.85,
            label_column="member"
        )
        print(f"  Saved member Taylor diagram to {svg_path}")
    except Exception as e:
        print(f"  Error plotting member Taylor diagram for {comp_type}: {e}")

def composite_propagating_years_no_plot(
    AAM_da,
    wind_da,
    date_list,
    clim_da=None,
    *,
    clim_start_yr: int = 1981,
    clim_end_yr: int = 2010,
    p_min_hpa: float = 150.0,
    p_max_hpa: float = 700.0,
    enso_state: str = "el_nino",
    rolling_period: int = 1,
    composite_months: int = 24,
    composite_start: str = "onset",
    nlevels: int = 13,
    onset_season: str = "all",
    bootstrap_events: bool = False,
    bootstrap_seed: Optional[int] = None,
) -> xr.DataArray:
    """Composite AAM/variable anomalies for ENSO event onset windows.

    For each (onset_time, end_time) entry in date_list, a composite_months window
    starting at the onset month is extracted and stacked. Duplicate onset months
    are composited only once. The result is a mean anomaly aligned on
    relative month (1..N, where N = composite_months).

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
    composite_months = int(composite_months)
    if composite_months < 1:
        raise ValueError("composite_months must be >= 1")
    if composite_start not in ("onset", "december_onset_year"):
        raise ValueError("composite_start must be 'onset' or 'december_onset_year'")

    # --- Unwrap Datasets ---
    AAM_field = AAM_da["AAM"] if isinstance(AAM_da, xr.Dataset) and "AAM" in AAM_da else AAM_da
    if isinstance(AAM_field, xr.Dataset):
        AAM_field = next(iter(AAM_field.data_vars.values()))
    
    # --- Normalize dimension names to ensure consistency ---
    if 'latitude' in AAM_field.dims and 'lat' not in AAM_field.dims:
        AAM_field = AAM_field.rename({'latitude': 'lat'})
    if 'longitude' in AAM_field.dims and 'lon' not in AAM_field.dims:
        AAM_field = AAM_field.rename({'longitude': 'lon'})

    wind_field = None
    if wind_da is not None:
        wind_field = wind_da["ua"] if isinstance(wind_da, xr.Dataset) and "ua" in wind_da else wind_da
        if isinstance(wind_field, xr.Dataset):
            wind_field = next(iter(wind_field.data_vars.values()))
        
        # --- Normalize wind field dimension names ---
        if 'latitude' in wind_field.dims and 'lat' not in wind_field.dims:
            wind_field = wind_field.rename({'latitude': 'lat'})
        if 'longitude' in wind_field.dims and 'lon' not in wind_field.dims:
            wind_field = wind_field.rename({'longitude': 'lon'})

    # --- Zonal integral for AAM per latitude band
    # and zonal mean for wind (intensive quantity) ---
    # if "longitude" in AAM_field.dims or "lon" in AAM_field.dims:
    #     AAM_field = _to_per_latitude_band(AAM_field)
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
        # # Only sum over longitude if it still exists (not already zonally integrated)
        # if "longitude" in clim_field.dims:
        #     clim_field = clim_field.sum(dim="longitude", skipna=True)
        # elif "lon" in clim_field.dims:
        #     clim_field = clim_field.sum(dim="lon", skipna=True)
        clim_field = vertical_sum_over_pressure_range(
            clim_field, p_min_hpa=p_min_hpa, p_max_hpa=p_max_hpa, level_dim="level"
        )
        AAM_anom = AAM_field.groupby("time.month") - clim_field
    else:
        clim_period = AAM_field.sel(time=slice(f"{clim_start_yr}-01", f"{clim_end_yr}-12"))
        clim_inline = clim_period.groupby("time.month").mean("time")
        AAM_anom = AAM_field.groupby("time.month") - clim_inline
    #import pdb; pdb.set_trace()
    
    # --- Extract composite window for each event ---
    stacked_AAM = []
    stacked_wind = []
    seen_onset_months: set = set()  # deduplicate by onset year-month, not calendar year
    for onset_time, _end_time in date_list:
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

        window_start, window_end = _compute_composite_window_from_onset(
            onset_str,
            composite_months=composite_months,
            composite_start=composite_start,
        )

        aam_event = AAM_anom.sel(time=slice(window_start, window_end))
        n_avail = int(aam_event.sizes["time"])
        if n_avail < composite_months:
            print(f"  composite: onset {onset_ym} window has only {n_avail} months of data, skipping.")
            continue

        # Label as relative months 1..N
        aam_event = aam_event.isel(time=slice(0, composite_months))
        aam_event = aam_event.assign_coords(time=np.arange(1, composite_months + 1, dtype=int))
        if "month" in aam_event.coords:
            aam_event = aam_event.drop_vars("month")
        aam_event = aam_event.rename({"time": "month"})
        
        # Normalize latitude dimension name for consistency
        if 'latitude' in aam_event.dims and 'lat' not in aam_event.dims:
            aam_event = aam_event.rename({'latitude': 'lat'})
        
        stacked_AAM.append(aam_event)

        if wind_field is not None:
            wind_event = wind_field.sel(time=slice(window_start, window_end))
            n_w = int(wind_event.sizes["time"])
            if n_w < composite_months:
                print(f"  composite: wind onset {onset_ym} window incomplete ({n_w} months), skipping.")
                continue
            wind_event = wind_event.isel(time=slice(0, composite_months))
            wind_event = wind_event.assign_coords(time=np.arange(1, composite_months + 1, dtype=int))
            if "month" in wind_event.coords:
                wind_event = wind_event.drop_vars("month")
            wind_event = wind_event.rename({"time": "month"})
            
            # Normalize latitude dimension name for consistency
            if 'latitude' in wind_event.dims and 'lat' not in wind_event.dims:
                wind_event = wind_event.rename({'latitude': 'lat'})
            
            stacked_wind.append(wind_event)

    n_events = len(stacked_AAM)
    if n_events < 1:
        print("composite_propagating_years: no valid events to composite.")
        return

    _start_desc = "onset month" if composite_start == "onset" else "December of onset year"
    print(f"Compositing {n_events} {composite_months}-month window(s) from {_start_desc}.")
    aam_stack = xr.concat(stacked_AAM, dim="event")
    aam_stack_for_plot = _circular_rolling_mean(aam_stack, dim="month", window=rolling_period)
    if bootstrap_events:
        rng = np.random.default_rng(bootstrap_seed)
        resample_indices = rng.integers(0, n_events, size=n_events)
        composite_AAM = aam_stack_for_plot.isel(event=resample_indices).mean("event", skipna=True)
        print(f"  member bootstrap: resampled {n_events} events with replacement.")
    else:
        composite_AAM = aam_stack_for_plot.mean("event", skipna=True)

    composite_wind = None
    if stacked_wind:
        wind_stack = xr.concat(stacked_wind, dim="event")
        wind_stack = _circular_rolling_mean(wind_stack, dim="month", window=rolling_period)
        composite_wind = wind_stack.mean("event", skipna=True)

    lat_dim = "lat" if "lat" in composite_AAM.dims else ("latitude" if "latitude" in composite_AAM.dims else None)
    if lat_dim is None:
        raise ValueError(f"No latitude dimension found in composite data with dims {composite_AAM.dims}")

    aam_vals = composite_AAM.values  # (month, lat) or (lat, month)
    # ensure shape is (lat, month)
    if composite_AAM.dims[0] == "month":
        aam_vals = aam_vals.T

    # --- T-test: significance vs zero along event dimension ---

    aam_for_ttest = aam_stack_for_plot.transpose("event", lat_dim, "month").values
    _, p_vals = _stats.ttest_1samp(aam_for_ttest, 0.0, axis=0, nan_policy="omit")
    # p_vals is (lat, month), matching aam_vals layout
    print(f"  t-test: shape={p_vals.shape}, min p={float(np.nanmin(p_vals)):.4f}, "
          f"p<0.05: {int(np.sum(p_vals < 0.05))} pts, "
          f"p<0.10: {int(np.sum(p_vals < 0.10))} pts")
    return composite_AAM


if __name__ == '__main__':

    clim_start_yr = 1981
    clim_end_yr = 2010

    ensemble_composites = []
    ensemble_lat_lev_composites = []
    ensemble_lat_lon_composites = []
    ensemble_u_latlon = []
    ensemble_u_latlev = []
    ensemble_uv_vi_lat = []
    member_bootstrap_composites = []
    member_bootstrap_lat_lev_composites = []
    member_bootstrap_lat_lon_composites = []
    member_bootstrap_labels = []
    ensemble_composites_by_strength = {label: [] for label, _, _ in EVENT_STRENGTH_BINS}
    ensemble_lat_lev_composites_by_strength = {label: [] for label, _, _ in EVENT_STRENGTH_BINS}
    ensemble_lat_lon_composites_by_strength = {label: [] for label, _, _ in EVENT_STRENGTH_BINS}
    available_members = []
    # Collect onset dates across ensemble members for histogram plotting
    onset_dates = []
    onset_members = []
    # Collect per-event peak Nino3.4 amplitudes (positive values)
    event_peak_amplitudes = []
    
    if not replot:
        
        for ensemble_member in [f"r{i}i1p1f3" for i in range(1, 61)]:
        # Use OS to see whether the nc file exists before trying to open with xarray, to avoid long error messages from xarray when files are missing.
            AAM_path = os.path.join(AAM_data_path_base, f"AAM_CMIP6_HadGEM3_GC31_{ensemble_member}_1850-01_2014-12.nc")
            AAM_exist = os.path.exists(AAM_path)
            clim_path = os.path.join(climatology_path_base, f"AAM_Climatology_CMIP6_HadGEM3_GC31_{ensemble_member}_{clim_start_yr}-{clim_end_yr}.nc")
            clim_exist = os.path.exists(clim_path)
            u_path = os.path.join(u_data_path_base, f"ua_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")
            u_exist = os.path.exists(u_path)
            uv_path = os.path.join(uv_data_path_base, f"uv_mon_historical_HadGEM3-GC31-LL_{ensemble_member}_interp.nc")
            uv_exist = os.path.exists(uv_path)
            
            if not AAM_exist or not clim_exist or not u_exist or not uv_exist:
                if ensemble_member in ["r1i1p1f3"]:  # Only print for first member to avoid spam
                    print(f"DEBUG: First member file check: AAM={AAM_exist}, clim={clim_exist}, u={u_exist}, uv={uv_exist}")
                    print(f"  AAM_data_path_base: {AAM_data_path_base}")
                print(f"Skipping member {ensemble_member} because files not found.")
                continue
            else:
                available_members.append(ensemble_member)
                print(f"Processing ensemble member: {ensemble_member}")
        
        # Parallelize ensemble member processing using joblib
        n_jobs = int(n_cpus_to_use) if n_cpus_to_use != -1 else -1
        print(f"Running ensemble processing with {n_jobs} worker(s)...")
        
        if JOBLIB_AVAILABLE and n_jobs != 1:
            # Use parallel processing
            results_list = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_ensemble_member)(
                    member,
                    clim_start_yr,
                    clim_end_yr,
                    args,
                    AAM_data_path_base,
                    climatology_path_base,
                    u_data_path_base,
                    uv_data_path_base,
                    u_level_to_plot,
                )
                for member in tqdm.tqdm(available_members)
            )
        else:
            # Sequential processing
            results_list = []
            for member in tqdm.tqdm(available_members):
                result = _process_single_ensemble_member(
                    member,
                    clim_start_yr,
                    clim_end_yr,
                    args,
                    AAM_data_path_base,
                    climatology_path_base,
                    u_data_path_base,
                    uv_data_path_base,
                    u_level_to_plot,
                )
                results_list.append(result)
        
        # Combine results from all ensemble members
        for result in results_list:
            if result['composite'] is not None:
                ensemble_composites.append(result['composite'])
            if result.get('member_bootstrap_composite') is not None:
                member_bootstrap_composites.append(result['member_bootstrap_composite'])
                member_bootstrap_labels.append(str(result.get('ensemble_member', f"member_{len(member_bootstrap_labels) + 1}")))
            for strength_label in ensemble_composites_by_strength:
                strength_composite = result['composites_by_strength'].get(strength_label)
                if strength_composite is not None:
                    ensemble_composites_by_strength[strength_label].append(strength_composite)
            if result['lat_lev_composite'] is not None:
                ensemble_lat_lev_composites.append(result['lat_lev_composite'])
            if result['lat_lon_composite'] is not None:
                ensemble_lat_lon_composites.append(result['lat_lon_composite'])
            if result['u_latlon'] is not None:
                ensemble_u_latlon.append(result['u_latlon'])
            if result['u_latlev'] is not None:
                ensemble_u_latlev.append(result['u_latlev'])
            if result['uv_vi'] is not None:
                ensemble_uv_vi_lat.append(result['uv_vi'])
            
            onset_dates.extend(result['onset_dates'])
            onset_members.extend(result['onset_members'])
            event_peak_amplitudes.extend(result['peak_amplitudes'])
            if 'enso_active_profile_total' not in locals():
                enso_active_profile_total = np.zeros(int(args.composite_months), dtype=int)
                enso_active_event_total = 0
                # Track active-month profiles per strength bin (SUM of counts, not bitwise OR)
                active_profile_by_strength = {label: np.zeros(int(args.composite_months), dtype=int) for label, _, _ in EVENT_STRENGTH_BINS}
                active_event_count_by_strength = {label: 0 for label, _, _ in EVENT_STRENGTH_BINS}
            
            enso_active_profile_total += np.asarray(result.get('enso_active_profile_total', 0), dtype=int)
            enso_active_event_total += int(result.get('enso_valid_event_count', 0))
            
            # Accumulate per-strength active profiles as COUNTS, not OR operations
            strength_profiles = result.get('strength_active_profiles', {})
            strength_event_counts = result.get('strength_event_counts', {})

            for strength_label, _, _ in EVENT_STRENGTH_BINS:

                # Use the FULL key name: weak/moderate/strong
                strength_profile = strength_profiles.get(strength_label, None)

                latlev_strength = (
                    result.get('lat_lev_composites_by_strength', {})
                    .get(strength_label)
                )

                if latlev_strength is not None:
                    ensemble_lat_lev_composites_by_strength[strength_label].append(
                        latlev_strength
                    )

                latlon_strength = (
                    result.get('lat_lon_composites_by_strength', {})
                    .get(strength_label)
                )

                if latlon_strength is not None:
                    ensemble_lat_lon_composites_by_strength[strength_label].append(
                        latlon_strength
                    )

                if strength_profile is not None and np.any(strength_profile):
                    active_profile_by_strength[strength_label] += np.asarray(
                        strength_profile,
                        dtype=int,
                    )

                    active_event_count_by_strength[strength_label] += int(
                        strength_event_counts.get(strength_label, 0)
                    )
        
        for lbl in ensemble_lat_lon_composites_by_strength:
            print(
                f"[DEBUG] {lbl}: "
                f"{len(ensemble_lat_lon_composites_by_strength[lbl])} latlon members, "
                f"{len(ensemble_lat_lev_composites_by_strength[lbl])} latlev members"
            )
        number_of_available_members = len(available_members)
        print("number of weak events:", len(ensemble_lat_lev_composites_by_strength['weak']))

        if getattr(args, "member_bootstrap_taylor_vs_era5", False):
            for strength_label, _, _ in EVENT_STRENGTH_BINS:
                print(f"\n=====================================================================")
                print(f"[HOOK] FIRST LOOP HERE - Generating Spatial Per-Member Taylor Diagrams for {strength_label.upper()}...")
                
                era5_latlon_truth = _load_era5_composite_truth("latlon")
                era5_latlev_truth = _load_era5_composite_truth("latlev")
                
                # 1. LAT-LON SPATIAL TAYLOR
                latlon_list = ensemble_lat_lon_composites_by_strength[strength_label]
                if len(latlon_list) > 0:
                    compute_spatial_per_member_taylor(
                        member_composites_list=latlon_list,
                        era5_reference=era5_latlon_truth,
                        comp_type="latlon",
                        output_dir=ensemble_results_dir,
                        strength_label=strength_label
                    )
                elif len(latlon_list) == 0:
                    print(f"  [SKIP] No latlon member composites found for '{strength_label}'.")

                # 2. LAT-LEV SPATIAL TAYLOR
                latlev_list = ensemble_lat_lev_composites_by_strength[strength_label]
                if len(latlev_list) > 0:
                    compute_spatial_per_member_taylor(
                        member_composites_list=latlev_list,
                        era5_reference=era5_latlev_truth,
                        comp_type="latlev",
                        output_dir=ensemble_results_dir,
                        strength_label=strength_label,
                    )
                elif len(latlev_list) == 0:
                    print(f"  [SKIP] No latlev member composites found for '{strength_label}'.")
                    
            member_bootstrap_stack = xr.concat(
                member_bootstrap_composites,
                dim=xr.IndexVariable("iteration", np.arange(len(member_bootstrap_composites))))
            member_bootstrap_stack = member_bootstrap_stack.assign_coords(
                member=("iteration", member_bootstrap_labels)
            )
            
            # svg_path = os.path.join(
            #     ensemble_results_dir, 
            #     f"Taylor_HadGEM3_member_bootstrap_vs_ERA5_new_{strength_label}_onset_{args.onset_season}_start_{args.composite_start}_region_{args.region}.svg"
            # )
            
            plot_member_bootstrap_taylor_vs_era5(
                member_bootstrap_stack,
                "aam",
                "all",
                ensemble_results_dir,
            )
    
    else:
        # replot=True: Load pre-computed ensemble mean composites and boolean masks
        print("replot=True: Loading pre-computed ensemble mean composites and significance masks...")

        region_label = args.region.upper() if args.region != 'all' else 'GLOBAL'
        reinitiation_label = "no_reinitiation"
        reinitiation_suffix = f"_{reinitiation_label}"
        
        # Initialise to 0; will be updated from metadata file
        number_of_available_members = 0
        active_month_percent = None
        strength_active_month_percent_by_label = {}
        
        main_results = load_composite_results(ensemble_results_dir, 'aam', 'all')
        if main_results is not None:
            if 'metadata' in main_results and 'ensemble_n_members' in main_results['metadata']:
                number_of_available_members = int(main_results['metadata']['ensemble_n_members'])
            if main_results.get('active_month_percent') is not None:
                active_month_percent = main_results['active_month_percent']
            if main_results.get('ens_mean') is not None:
                ensemble_composites.append(main_results['ens_mean'])
            significance_mask_aam = main_results.get('significance_mask')
            print(f"  Loaded bootstrap AAM results for {number_of_available_members} members.")
            _plot_or_rebuild_era5_taylor(
                results_dir=ensemble_results_dir,
                comp_type='aam',
                strength_label='all',
                boot_ds=main_results.get('bootstrap_samples'),
            )
        else:
            print("  WARNING: No bootstrap AAM results found for replotting.")

        latlev_results = load_composite_results(ensemble_results_dir, 'latlev', 'all')
        if latlev_results is not None and latlev_results.get('ens_mean') is not None:
            ensemble_lat_lev_composites.append(latlev_results['ens_mean'])
            significance_mask_latlev = latlev_results.get('significance_mask')
            print("  Loaded bootstrap lat×level results.")
        else:
            significance_mask_latlev = None
                
        latlon_results = load_composite_results(ensemble_results_dir, 'latlon', 'all')
        if latlon_results is not None and latlon_results.get('ens_mean') is not None:
            ensemble_lat_lon_composites.append(latlon_results['ens_mean'])
            significance_mask_latlon = latlon_results.get('significance_mask')
            print("  Loaded bootstrap lat×lon results.")
        else:
            significance_mask_latlon = None

        # Load strength-binned composites
        for strength_label, _, _ in EVENT_STRENGTH_BINS:
            strength_nc_path = os.path.join(
                ensemble_mean_output_path,
                f"AAM_composite_ENSEMBLE_MEAN_{strength_label}_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc",
            )
            loaded_strength = load_composite_results(ensemble_results_dir, 'aam', strength_label)
            if loaded_strength is not None and loaded_strength.get('ens_mean') is not None:
                ensemble_composites_by_strength[strength_label].append(loaded_strength['ens_mean'])
                if loaded_strength.get('active_month_percent') is not None:
                    strength_active_month_percent_by_label[strength_label] = loaded_strength['active_month_percent']
                    print(f"  Loaded active_month_percent for {strength_label} from bootstrap results")

    _cmp = ">" if args.enso_state == "el_nino" else "<"
    _snap_season_label = "  |  NDJFM onsets only" if args.onset_season == "ndjfm" else ""
    _snap_suffix = (
                        f"{args.enso_state} state: Nino3.4{_cmp}{ACTIVE_MONTH_EL_NINO_THRESHOLD:.2f} "
                        f"for >= {int(args.min_elnino_months)} months"
                        f" | {int(args.composite_months)}-month composite from "
                        f"{'Dec of onset year' if args.composite_start == 'december_onset_year' else 'onset month'}"
                        f"{_snap_season_label}"
                    )
    
    # Determine region label for filenames (used in all composite plots)
    region_label = args.region.upper() if args.region != 'all' else 'GLOBAL'
    # Hardcoded: always require no reactivation within 12 months after event ends
    reinitiation_label = "no_reinitiation"
    reinitiation_suffix = f"_{reinitiation_label}"
    
    if ensemble_composites:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            from matplotlib.colors import LinearSegmentedColormap, Normalize
            import numpy as np
            from scipy import stats as _stats
            import os

            # -------------------------------
            # Ensemble mean
            # -------------------------------
            ens_stack = xr.concat(ensemble_composites, dim="ensemble")
            ens_mean = ens_stack.mean("ensemble", skipna=True)

            print(f"Plotting ENSEMBLE MEAN composite from {ens_stack.sizes['ensemble']} members...")

            # -------------------------------
            # Extract dims
            # -------------------------------
            lat_dim = "latitude" if "latitude" in ens_mean.dims else "lat"
            lat_vals = ens_mean[lat_dim].values
            month_vals = ens_mean["month"].values

            # Ensure (lat, month)
            aam_vals = ens_mean.values
            if ens_mean.dims[0] == "month":
                aam_vals = aam_vals.T

            # -------------------------------
            # FIXED t-test (CRITICAL) - Bypass in replot
            # -------------------------------
            if not replot:
                active_month_percent = None
                if 'enso_active_event_total' in locals() and enso_active_event_total > 0:
                    active_month_percent = np.rint(
                        100.0 * np.asarray(enso_active_profile_total, dtype=float) / float(enso_active_event_total)
                    ).astype(int)

                significance_mask_aam, aam_save_path, _ = compute_and_save_composite_significance(
                    ens_stack=ens_stack,
                    composite_type='aam',
                    strength_label='all',
                    results_dir=ensemble_results_dir,
                    active_month_percent=active_month_percent,
                    args=args,
                )
                insig_mask = np.where(significance_mask_aam, 0, 1)
            else:
                if 'significance_mask_aam' in locals() and significance_mask_aam is not None:
                    insig_mask = np.where(significance_mask_aam, 0, 1)
                else:
                    print("WARNING: No significance mask loaded. Hatching everything.")
                    insig_mask = np.ones_like(aam_vals)

            insig_mask = _match_plot_orientation(aam_vals, insig_mask)

            insig_mask = _match_plot_orientation(aam_vals, insig_mask)

            # -------------------------------
            # Plot
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.subplots_adjust(bottom=0.30)
            
            #vmax = 5e24
            vmax = float(np.nanpercentile(np.abs(aam_vals), 98))
            vmax = vmax if vmax > 0 else 1.0
            vmin = -vmax

            levels = np.linspace(vmin, vmax, 13)

            cf = ax.contourf(
                month_vals,
                lat_vals,
                aam_vals,
                levels=levels,
                cmap="RdBu_r",
                extend="both",
            )

            # -------------------------------
            # Colorbar
            # -------------------------------
            _abs = max(abs(vmin), abs(vmax))
            order = int(np.floor(np.log10(_abs))) if _abs > 0 else 0
            factor = 10 ** order

            cax = fig.add_axes([0.125, 0.06, 0.775, 0.015])
            cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", extend="both")

            _sup = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
            _order_sup = str(order).translate(_sup)

            cbar.set_label(f"AAM anomaly (×10{_order_sup})", size=14)

            _tick_levels = cf.levels[::2]
            cbar.set_ticks(_tick_levels)
            cbar.set_ticklabels([f"{v / factor:.1f}" for v in _tick_levels])
            cbar.ax.tick_params(labelsize=11)
            
            #Solid black line at the Equator
            ax.axhline(
                y=0, 
                color='black', 
                linestyle='-',    # Solid line
                linewidth=1, 
                zorder=2,
                alpha=0.8
            )
            
            for lat in [-40, -20, 20, 40]:
                ax.axhline(
                    y=lat, 
                    color='grey', 
                    linestyle='--',   # Dashed line
                    linewidth=0.8, 
                    alpha=0.4,        # Less opacity (lower alpha means more transparent)
                    zorder=2
                )
            
            # -------------------------------
            # Significance overlay (FIXED) --update hash insignificant regions
            # -------------------------------            
            # 2. Apply hatching over the areas marked as 1
            if np.any(insig_mask == 1):
                hatches = ax.contourf(
                    month_vals,          # Your full X-axis array
                    lat_vals,            # Your full Y-axis array
                    insig_mask,          # The 2D grid of 0s and 1s
                    levels=[0.5, 1.5],   # Draw boundaries only around the 1s
                    colors='none',       # Keep background transparent
                    hatches=['//'],      # Use dense slashes
                    zorder=10
                )
                # 'edgecolor' controls the color of the lines themselves
                import matplotlib.colors as mcolors
                line_color_with_alpha = mcolors.to_rgba('gray', alpha=0.4)
                
                for collection in hatches.collections:
                    collection.set_facecolor('none')
                    collection.set_edgecolor(line_color_with_alpha)

                    if hasattr(collection, 'set_edgecolors'):
                        collection.set_edgecolors([line_color_with_alpha])
                    collection.set_linewidths([0.0])
            else:
                print("No insignificant points to hatch (all p <= 0.05)")

            # -------------------------------
            # Labels
            # -------------------------------
            ax.set_xlabel("Month since onset", fontsize=14)
            ax.set_ylabel("Latitude (°N)", fontsize=14)

            if active_month_percent is not None:
                pct_cmap = LinearSegmentedColormap.from_list(
                    "enso_active_pct",
                    ["#7c7b66", "#9e833b", "#f39c34", "#e85d3a", "#bd491f"],
                )
                pct_norm = Normalize(vmin=0, vmax=100)
                ax.text(
                    -0.06,
                    -0.155,
                    "ENSO active (%)",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=14,
                )
                for month_idx, (month_val, pct_val) in enumerate(zip(month_vals, active_month_percent)):
                    if month_idx % 2 != 0:
                        continue
                    ax.text(
                        month_val,
                        -0.19,
                        f"{int(pct_val)}%",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=10,
                        fontweight="bold",
                        color=pct_cmap(pct_norm(int(pct_val))),
                        clip_on=False,
                    )


            ax.set_xlim(1, len(month_vals))
            ax.set_ylim(-60, 60)

            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))

            state_pretty = "El Nino" if args.enso_state == "el_nino" else "La Nina"

            onset_season_label = f" | {args.onset_season.upper()} onsets only" if args.onset_season != "all" else ""
            
            ax.set_title(
                f"HadGEM3_GC31 {number_of_available_members} members BOOTSTRAP Composite AAM anomaly\n"
                f"({args.p_min}–{args.p_max} hPa) {state_pretty} events  {args.start_year}–{args.end_year}"
                f"clim {clim_start_yr}–{clim_end_yr} {onset_season_label}"
            )

            # -------------------------------
            # Save
            # -------------------------------
            os.makedirs(output_dir, exist_ok=True)

            out_path = os.path.join(
                output_dir,
                f"AAM_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{region_label}_{args.onset_season}_start_{args.composite_start}{reinitiation_suffix}.svg",
            )

            fig.savefig(out_path, format="svg", bbox_inches="tight")
            plt.close(fig)

            print(f"Ensemble mean composite plot saved to {out_path}")
            
            if save_ensemble_mean_netcdf:
                # Output the ensemble mean composite data as netCDF for future analysis
                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"AAM_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc",
                )
                ens_mean.name = "AAMA"
                ens_mean.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean.to_netcdf(out_nc_path)
                print(f"Ensemble mean composite data saved to {out_nc_path}")
        except Exception as e:
            print(f"Error plotting ensemble mean composite: {e}")
            import traceback
            traceback.print_exc()
    
    if not ensemble_lat_lev_composites:
        print(f"WARNING: No ensemble lat×level composites computed for region '{args.region}'. Skipping lat×level plots.")
    
    # %%
    
    if ensemble_lat_lev_composites:
        try:
            print("\n[HOOK] Loading ERA5 truth for lat-lev Taylor...")
            era5_latlev_truth = _load_era5_composite_truth("latlev")
            
            if era5_latlev_truth is not None:
                compute_spatial_per_member_taylor(
                    member_composites_list=ensemble_lat_lev_composites,
                    era5_reference=era5_latlev_truth,
                    comp_type="latlev",
                    output_dir=ensemble_results_dir,
                    strength_label="all"
                )
            
            #import pdb; pdb.set_trace()
            
            ens_stack = xr.concat(ensemble_lat_lev_composites, dim="ensemble")
            #ens_mean = ens_stack.mean("ensemble", skipna=True)
            if not replot:
                sig_mask_latlev, _, boot_mean_latlev = compute_and_save_composite_significance(
                    ens_stack=ens_stack,
                    composite_type='latlev',
                    strength_label='all',
                    results_dir=ensemble_results_dir,
                    active_month_percent=None,
                    args=args,
                )
            else:
                sig_mask_latlev = significance_mask_latlev if 'significance_mask_latlev' in locals() else None
                # If you want to use it in replot, make sure you loaded it above!
                #boot_mean_latlev = loaded_latlev_results['bootstrap_samples'].mean('iteration') if loaded_latlev_results else ens_mean

            ens_mean_u_latlev = None
            if ensemble_u_latlev:
                ens_stack_u = xr.concat(ensemble_u_latlev, dim="ensemble")
                ens_mean_u_latlev = ens_stack_u.mean("ensemble", skipna=True)
                if "month" in ens_mean_u_latlev.coords and "time" not in ens_mean_u_latlev.coords:
                    ens_mean_u_latlev = ens_mean_u_latlev.rename({"month": "time"})

            print(f"Plotting BOOTSTRAP COMPOSITE lat×level composite from {ens_stack.sizes['ensemble']} members...")

            boot_mean_latlev = boot_mean_latlev.rename({"month": "time"}) if "month" in boot_mean_latlev.dims else boot_mean_latlev
            
            plot_latitude_level_snapshots_HadGEN3(
                boot_mean_latlev,
                zonal_wind_da=ens_mean_u_latlev,
                p_values=None,
                significance_mask=sig_mask_latlev,
                ensemble_member="BOOTSTRAP_MEAN",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                output_dir=output_dir,
                title_suffix= f"{number_of_available_members} BOOTSTRAP {region_label} | " + _snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=f"_bootstrap_composite_{args.enso_state}_{region_label.lower()}{reinitiation_suffix}",
                dec_onset_month=args.composite_start,
                onset_season_ndjfm=args.onset_season,
                nino_threshold=float(ACTIVE_MONTH_EL_NINO_THRESHOLD),
            )
            
            if save_ensemble_mean_netcdf:
                # Output the ensemble mean composite data as netCDF for future analysis
                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"AAM_lat_lev_composite_BOOTSTRAP_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc",
                )
                boot_mean_latlev.name = "AAMA"
                boot_mean_latlev.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                boot_mean_latlev.to_netcdf(out_nc_path)
                print(f"Ensemble mean composite data saved to {out_nc_path}")

                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"U_lat_lev_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc",
                )
                ens_mean_u_latlev.name = "u"
                ens_mean_u_latlev.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean_u_latlev.to_netcdf(out_nc_path)
                print(f"Ensemble mean zonal wind composite data saved to {out_nc_path}")
                
        except Exception as e:
            print(f"Error plotting ensemble mean lat×level composite: {e}")
            import traceback
            traceback.print_exc()
    
    if not ensemble_lat_lon_composites:
        print(f"WARNING: No ensemble lat×lon composites computed for region '{args.region}'. Skipping lat×lon plots.")
        
    # %%
    if ensemble_lat_lon_composites:
        try:
            print("\n[HOOK] Loading ERA5 truth for lat-lon Taylor...")
            era5_latlon_truth = _load_era5_composite_truth("latlon")
            if era5_latlon_truth is not None:
                compute_spatial_per_member_taylor(
                    member_composites_list=ensemble_lat_lon_composites,
                    era5_reference=era5_latlon_truth,
                    comp_type="latlon",
                    output_dir=ensemble_results_dir,
                    strength_label="all"
                )
            ens_stack = xr.concat(ensemble_lat_lon_composites, dim="ensemble")
            #ens_mean = ens_stack.mean("ensemble", skipna=True)
            
            if not replot:
                sig_mask_latlon, _, boot_mean_latlon = compute_and_save_composite_significance(
                    ens_stack=ens_stack,
                    composite_type='latlon',
                    strength_label='all',
                    results_dir=ensemble_results_dir,
                    active_month_percent=None,
                    args=args,
                )
            else:
                sig_mask_latlon = significance_mask_latlon if 'significance_mask_latlon' in locals() else None
                
            # print(f"Lat-lon t-test: min p={float(np.nanmin(p_vals_latlon)):.4f}, p<0.05: {int(np.sum(p_vals_latlon < 0.05))} pts")
            
            # print(f"\n[ENSEMBLE MEAN DEBUG]")
            # print(f"  ens_stack shape: {ens_stack.shape}")
            # print(f"  ens_stack lon range: [{float(ens_stack['lon'].values.min()):.1f}, {float(ens_stack['lon'].values.max()):.1f}]")
            # print(f"  ens_mean shape: {ens_mean.shape}")
            # print(f"  ens_mean lon range BEFORE safety check: [{float(ens_mean['lon'].values.min()):.1f}, {float(ens_mean['lon'].values.max()):.1f}]")
            # print(f"  p_vals_latlon shape: {p_vals_latlon.shape}")
            # print(f"  p_vals_latlon non-nan points: {np.sum(~np.isnan(p_vals_latlon))}")
            # print(f"  p_vals_latlon < 0.05: {np.sum(p_vals_latlon < 0.05)} / {p_vals_latlon.size} points")
            
            ens_mean_u_latlon = None
            if ensemble_u_latlon:
                ens_stack_u = xr.concat(ensemble_u_latlon, dim="ensemble")
                ens_mean_u_latlon = ens_stack_u.mean("ensemble", skipna=True)
                if "month" in ens_mean_u_latlon.dims and "time" not in ens_mean_u_latlon.dims:
                    ens_mean_u_latlon = ens_mean_u_latlon.rename({"month": "time"})

            ens_mean_uv_latlev = None
            if ensemble_uv_vi_lat:
                ens_stack_uv = xr.concat(ensemble_uv_vi_lat, dim="ensemble")
                ens_mean_uv_latlev = ens_stack_uv.mean("ensemble", skipna=True)
                if "month" in ens_mean_uv_latlev.dims and "time" not in ens_mean_uv_latlev.dims:
                    ens_mean_uv_latlev = ens_mean_uv_latlev.rename({"month": "time"})
            
            # Normalize dimension names (region selection was already applied in _process_single_ensemble_member)
            # Rename longitude → lon
            if 'longitude' in boot_mean_latlon.dims:
                boot_mean_latlon = boot_mean_latlon.rename({'longitude': 'lon'})
            if ens_mean_u_latlon is not None and 'longitude' in ens_mean_u_latlon.dims:
                ens_mean_u_latlon = ens_mean_u_latlon.rename({'longitude': 'lon'})
            if ens_mean_uv_latlev is not None and 'longitude' in ens_mean_uv_latlev.dims:
                ens_mean_uv_latlev = ens_mean_uv_latlev.rename({'longitude': 'lon'})
            
            # Rename latitude → lat
            if 'latitude' in boot_mean_latlon.dims:
                boot_mean_latlon = boot_mean_latlon.rename({'latitude': 'lat'})
            if ens_mean_u_latlon is not None and 'latitude' in ens_mean_u_latlon.dims:
                ens_mean_u_latlon = ens_mean_u_latlon.rename({'latitude': 'lat'})
            if ens_mean_uv_latlev is not None and 'latitude' in ens_mean_uv_latlev.dims:
                ens_mean_uv_latlev = ens_mean_uv_latlev.rename({'latitude': 'lat'})
            
            # Force region selection immediately before plotting to guarantee slice correctness.
            if args.region != 'all' and 'lon' in boot_mean_latlon.dims:
                boot_mean_latlon = _select_region(boot_mean_latlon, args.region)
                if ens_mean_u_latlon is not None and 'lon' in ens_mean_u_latlon.dims:
                    ens_mean_u_latlon = _select_region(ens_mean_u_latlon, args.region)
                if ens_mean_uv_latlev is not None and 'lon' in ens_mean_uv_latlev.dims:
                    ens_mean_uv_latlev = _select_region(ens_mean_uv_latlev, args.region)

                lon_actual_min = float(boot_mean_latlon['lon'].values.min())
                lon_actual_max = float(boot_mean_latlon['lon'].values.max())
                n_lon_actual = int(boot_mean_latlon.sizes['lon'])
                print(f"\n[REGION DEBUG] Region requested: {args.region}")
                print(f"[REGION DEBUG] Final plot slice lon range [{lon_actual_min:.1f}, {lon_actual_max:.1f}], n_lon={n_lon_actual}")

            print(f"\n[PRE-PLOT CHECK] About to plot lat×lon composite:")
            # print(f"  bootstrap shape: {boot_mean_latlon.shape}")
            # print(f"  bootstrap lon range: [{float(boot_mean_latlon['lon'].values.min()):.1f}, {float(boot_mean_latlon['lon'].values.max()):.1f}]")
            # print(f"  bootstrap lat range: [{float(boot_mean_latlon['lat'].values.min()):.1f}, {float(boot_mean_latlon['lat'].values.max()):.1f}]")
            # print(f"  ens_mean_u_latlon shape: {None if ens_mean_u_latlon is None else ens_mean_u_latlon.shape}")
            # print(f"  ens_mean_uv_latlev shape: {None if ens_mean_uv_latlev is None else ens_mean_uv_latlev.shape}")
            print(f"  Passing region='{args.region}' to plot_lat_lon_snapshots()")
            
            boot_mean_latlon = boot_mean_latlon.rename({"month": "time"}) if "month" in boot_mean_latlon.dims else boot_mean_latlon
            
            print(f"Plotting ENSEMBLE MEAN lat×lon composite from {ens_stack.sizes['ensemble']} members...")
            
            # --- Slice both AAM and U arrays to -60 to 60 latitude ---
            # Check if latitude goes north-to-south or south-to-north to prevent empty slices
            if boot_mean_latlon.latitude[0] > boot_mean_latlon.latitude[-1]:
                lat_slice = slice(60, -60)
            else:
                lat_slice = slice(-60, 60)
                
            boot_mean_latlon = boot_mean_latlon.sel(latitude=lat_slice)
            if ens_mean_u_latlon is not None:
                ens_mean_u_latlon = ens_mean_u_latlon.sel(latitude=lat_slice)
            
            plot_lat_lon_snapshots(
                boot_mean_latlon,
                zonal_wind_da=ens_mean_u_latlon,
                uv_latlev_profile=ens_mean_uv_latlev,
                p_values=None,
                significance_mask=sig_mask_latlon,
                output_dir=output_dir,
                ensemble_member="BOOTSTRAP_MEAN",
                start_year=args.start_year,
                end_year=args.end_year,
                clim_start_yr=clim_start_yr,
                clim_end_yr=clim_end_yr,
                title_suffix=f"{number_of_available_members} BOOTSTRAP {region_label} | " + _snap_suffix,
                rolling_period=int(args.rolling_period),
                filename_suffix=f"_bootstrap_composite_{args.enso_state}_{region_label.lower()}_start_{args.composite_start}",
                pmin=float(args.p_min),
                pmax=float(args.p_max),
                nino_threshold=float(ACTIVE_MONTH_EL_NINO_THRESHOLD),
                region=args.region,
            )
            
            if save_ensemble_mean_netcdf:
                out_nc_path = os.path.join(
                ensemble_mean_output_path,
                f"AAM_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc",
                )
                boot_mean_latlon.name = "AAMA"
                boot_mean_latlon.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                boot_mean_latlon.to_netcdf(out_nc_path)
                print(f"Ensemble mean composite data saved to {out_nc_path}")

                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"U_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc")
                
                ens_mean_u_latlon.name = "u"
                ens_mean_u_latlon.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean_u_latlon.to_netcdf(out_nc_path)
                print(f"Ensemble mean zonal wind composite data saved to {out_nc_path}")
                
                out_nc_path = os.path.join(
                    ensemble_mean_output_path,
                    f"UV_lat_lon_composite_ENSEMBLE_MEAN_{args.enso_state}_state_{args.start_year}-{args.end_year}_{args.p_min}-{args.p_max}hPa_{args.onset_season}_start_{args.composite_start}_region_{args.region}{reinitiation_suffix}.nc")
                
                ens_mean_uv_latlev.name = "uv"
                ens_mean_uv_latlev.attrs["n_ensemble_members"] = int(ens_stack.sizes["ensemble"])
                ens_mean_uv_latlev.to_netcdf(out_nc_path)
                print(f"Ensemble mean UV composite data saved to {out_nc_path}")
                
        except Exception as e:
            print(f"Error plotting ensemble mean lat×lon composite: {e}")
            import traceback
            traceback.print_exc()

    strength_desc_map = {
        "weak": "weak (0.5-0.9)",
        "moderate": "moderate (1.0-1.4)",
        "strong": "strong (>=1.5)",
    }
    strength_outputs_found = False
    for strength_label, _, _ in EVENT_STRENGTH_BINS:
        print(f"\n  SECOND LOOP HERE --- Processing Spatial Taylor for strength: {strength_label.upper()} ---")
        # Load the specific ERA5 truth for this strength bin
        era5_latlon_truth = _load_era5_composite_truth("latlon")
        era5_latlev_truth = _load_era5_composite_truth("latlev")

        # Get the populated list from your new dictionary
        latlon_list = ensemble_lat_lon_composites_by_strength[strength_label]
        latlev_list = ensemble_lat_lev_composites_by_strength[strength_label]
        
        # Compute LatLon if data exists
        if era5_latlon_truth is not None and len(latlon_list) > 0:
            compute_spatial_per_member_taylor(
                member_composites_list=latlon_list,
                era5_reference=era5_latlon_truth,
                comp_type="latlon",
                output_dir=output_dir,
                strength_label=strength_label
            )
        elif len(latlon_list) == 0:
            print(f"  [SKIP] No latlon member composites found for '{strength_label}'.")

        # Compute LatLev if data exists
        if era5_latlev_truth is not None and len(latlev_list) > 0:
            compute_spatial_per_member_taylor(
                member_composites_list=latlev_list,
                era5_reference=era5_latlev_truth,
                comp_type="latlev",
                output_dir=output_dir,
                strength_label=strength_label
            )
        elif len(latlev_list) == 0:
            print(f"  [SKIP] No latlev member composites found for '{strength_label}'.")
    
        
        strength_composites = ensemble_composites_by_strength.get(strength_label, [])
        if not strength_composites:
            continue
        strength_outputs_found = True
        strength_stack = xr.concat(strength_composites, dim="ensemble")
        
        # Compute active-month percentages for this strength bin
        strength_active_month_percent = None
        if 'active_event_count_by_strength' in locals() and active_event_count_by_strength.get(strength_label, 0) > 0:
            strength_active_month_percent = np.rint(
                100.0 * np.asarray(active_profile_by_strength[strength_label], dtype=float) / float(active_event_count_by_strength[strength_label])
            ).astype(int)
        elif replot and strength_label in strength_active_month_percent_by_label:
            strength_active_month_percent = strength_active_month_percent_by_label[strength_label]
        # Fallback: if we still don't have an active-month percent array, try loading it
        # from the saved ensemble_results file (useful when rerunning overwrote in-memory counts)
        if strength_active_month_percent is None:
            try:
                loaded_strength = load_composite_results(ensemble_results_dir, 'aam', strength_label)
                if loaded_strength is not None and loaded_strength.get('active_month_percent') is not None:
                    strength_active_month_percent = loaded_strength.get('active_month_percent')
                    print(f"  Fallback-loaded active_month_percent for {strength_label} from saved results")
            except Exception:
                pass
        # Diagnostic: print what active-month percent array will be used for this strength
        print(f"[REplot-DIAG] strength_label={strength_label}, strength_active_month_percent={strength_active_month_percent}")
        
        _plot_and_save_ensemble_mean_aam_composite(
            ens_stack=strength_stack,
            args=args,
            clim_start_yr=clim_start_yr,
            clim_end_yr=clim_end_yr,
            output_dir=output_dir,
            ensemble_mean_output_path=ensemble_mean_output_path,
            region_label=region_label,
            reinitiation_suffix=reinitiation_suffix,
            number_of_available_members=number_of_available_members,
            strength_label=strength_label,
            strength_desc=strength_desc_map[strength_label],
            save_ensemble_mean_netcdf=save_ensemble_mean_netcdf,
            ensemble_results_dir=ensemble_results_dir,
            active_month_percent=strength_active_month_percent,
        )

    if not strength_outputs_found:
        print("WARNING: No weak/moderate/strong composites were available to plot.")
    
    # =============================================================================
    # Plot histogram of ENSO onset months across ensemble members and years
    if 'onset_dates' in locals() and onset_dates:
        try:
            os.makedirs(output_dir, exist_ok=True)
            onset_pd = pd.to_datetime(onset_dates)
            months = onset_pd.month

            # Counts per calendar month (1=Jan .. 12=Dec)
            counts = months.value_counts().sort_index()
            month_idx = list(range(1, 13))
            counts_list = [int(counts.get(m, 0)) for m in month_idx]
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(month_names))
            ax.bar(x, counts_list, width=1.0, align='center', color='C1', edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(month_names)
            ax.set_xlim(-0.5, len(month_names) - 0.5)
            ax.set_xlabel('Onset month')
            ax.set_ylabel(f'Event count (across ensemble members and years)')
            ax.set_title(f'ENSO onset month histogram ({args.enso_state}) {number_of_available_members} members {args.start_year}-{args.end_year}')

            out_hist_path = os.path.join(
                output_dir,
                f"ENSO_onset_months_histogram_{args.enso_state}_{args.start_year}-{args.end_year}{reinitiation_suffix}.svg",
            )
            fig.savefig(out_hist_path, format="svg", bbox_inches='tight')
            plt.close(fig)
            print(f"Saved onset months histogram to {out_hist_path}")
        except Exception as e:
            print(f"Error generating onset month histogram: {e}")
            import traceback; traceback.print_exc()
        # Plot histogram of per-event peak Nino3.4 amplitudes (0.5 bins starting at 0.5)
        if event_peak_amplitudes:
            try:
                peaks = np.asarray(event_peak_amplitudes, dtype=float)
                if peaks.size > 0:
                    bin_width = 0.5
                    max_peak = float(np.nanmax(peaks))
                    # ensure at least one bin centered at 0.5
                    max_center = max(bin_width, np.ceil(max_peak / bin_width) * bin_width)
                    bin_centers = np.arange(bin_width, max_center + bin_width / 2, bin_width)
                    bin_edges = np.concatenate(([bin_centers[0] - bin_width / 2], bin_centers + bin_width / 2))
                    counts, _ = np.histogram(peaks, bins=bin_edges)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(bin_centers, counts, width=bin_width, align='center', color='C2', edgecolor='black', linewidth=0.5)
                    ax.set_xticks(bin_centers)
                    ax.set_xlabel('Event peak Nino3.4 (|°C|)')
                    ax.set_ylabel('Event count (across ensemble members and years)')
                    ax.set_title(f'ENSO event peak Nino3.4 histogram ({args.enso_state}) {number_of_available_members} members {args.start_year}-{args.end_year}')

                    out_hist_peaks = os.path.join(
                        output_dir,
                        f"ENSO_event_peak_nino34_histogram_{args.enso_state}_{args.start_year}-{args.end_year}{reinitiation_suffix}_bin{bin_width}.svg",
                    )
                    fig.savefig(out_hist_peaks, format="svg", bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved event peak Nino3.4 histogram to {out_hist_peaks}")
            except Exception as e:
                print(f"Error generating event-peak Nino3.4 histogram: {e}")
                import traceback; traceback.print_exc()
        
