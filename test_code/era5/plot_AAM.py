# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, ListedColormap

AAM_data_path = "AAM_data/monthly_mean/"

def plot_AAM(year, component='AAM', *, nlevels=21, cmap_name='bwr', vmin=None, vmax=None, savefile=None, show=True):
	"""Plot AAM (latitude vs time) for a given year.

	Loads files matching `AAM_ERA5_{year}*.nc` from `AAM_data/monthly_mean/` and
	plots the given `component` as latitude vs time using a discrete blue->red
	colormap.

	Args:
		year (int or str): Year to plot (e.g. 1980).
		component (str): Variable name inside the netCDF files (default 'AAM').
		nlevels (int): Number of discrete color levels.
		cmap_name (str): Matplotlib colormap name.
		vmin, vmax (float): Color limits (optional).
		savefile (str): If provided, save figure to this path.
		show (bool): If True, call `plt.show()`.

	Returns:
		(fig, ax): Matplotlib figure and axis.
	"""

	pattern = f"{AAM_data_path}AAM_ERA5_{int(year)}*.nc"
	ds = xr.open_mfdataset(pattern, combine='by_coords')
	if component not in ds.data_vars:
		raise KeyError(f"variable '{component}' not found in {pattern}")

	da = ds[component]

	# ensure dims include time and latitude
	if 'time' not in da.dims or 'latitude' not in da.dims:
		raise ValueError('dataarray must have dims (time, latitude)')

	times = da['time'].to_index()
	lats = da['latitude'].values
	data = da.values  # shape (time, lat)

	if vmin is None:
		vmin = np.nanmin(data)
	if vmax is None:
		vmax = np.nanmax(data)

	levels = np.linspace(vmin, vmax, nlevels)
	base_cmap = plt.get_cmap(cmap_name)
	cmap_disc = ListedColormap(base_cmap(np.linspace(0, 1, nlevels - 1)))
	norm = BoundaryNorm(levels, ncolors=cmap_disc.N, clip=True)

	fig, ax = plt.subplots(figsize=(10, 4))

	# imshow: transpose so y is latitude (data shape: time x lat)
	times_num = mdates.date2num(times)
	im = ax.imshow(
		data.T,
		origin='lower',
		aspect='auto',
		cmap=cmap_disc,
		norm=norm,
		extent=[times_num[0], times_num[-1], lats[0], lats[-1]]
	)

	ax.xaxis_date()
	ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

	ax.set_ylabel('Latitude')
	ax.set_xlabel('Time')
	ax.set_title(f"{component} latitude-time ({int(year)})")

	cbar = fig.colorbar(im, ax=ax, boundaries=levels, ticks=levels[::max(1, len(levels)//10)])
	cbar.set_label(da.attrs.get('units', ''))

	fig.tight_layout()
	if savefile:
		fig.savefig(savefile, dpi=150)
	if show:
		plt.show()

	ds.close()
	return fig, ax

# %%
plot_AAM(1980, component='AAM', savefile='AAM_ERA5_1980.png', vmin=1e24, vmax=5e24, show=True)
# %%
