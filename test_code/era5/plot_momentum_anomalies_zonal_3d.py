"""
This script plots snapshots of the vertically integrated AAM on north - east 
plane at different time steps. Access to the full AAM data is required. 
This script is intended to visualise individual events. 
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

base_dir = os.getcwd()
Variable_data_path_base = f"{base_dir}/monthly_mean/AAM/"
zonal_wind_path_base = f"{base_dir}/monthly_mean/variables/"
climatology_path_base = f"{base_dir}/climatology/"
output_dir = f"{base_dir}/figures/"
pressure_lvl_dir = f"{base_dir}/l137_a_b.csv"

