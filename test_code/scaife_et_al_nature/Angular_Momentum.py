
"""
The code uses monthly mean values of surface pressure (ps) and zonal winds (u)
to compute vertically integrated angular momentum summed over longitude
"""
import iris
import iris_grib
import numpy as np
from iris.time import PartialDateTime
import datetime
from multiprocessing import Pool




# Load the a and b sigma level coefficients
exec(open('vertical_levels.py').read())


for year in np.arange(2008,2011):
    for m in np.arange(1,13):
        pdt = iris.Constraint(time=PartialDateTime(month=m))
        # load zonal winds on model sigma cooridnates
        u = iris.load_cube('ERA20C_U_%04d.nc' % (year), 'eastward_wind').extract(pdt)
        # load surface pressure
        ps = iris.load_cube('ERA20C_SP_%04d.nc' % (year),'surface_air_pressure').extract(pdt).data
        # get u on middle height levels
        u_mid = (u[:,1:].data + u[:,:-1].data)*0.5
        # difference of a and b sigma coefficients
        db = b[1:] - b[:-1]
        da = a[1:] - a[:-1]
        # constants
        r   = 6371229.0
        g   = 9.80665
        omega  = 7.292116e-5
        
        lat = np.radians(u.coord('latitude').points)
        lon = np.radians(u.coord('longitude').points)
        
        cos_theta_latitude=np.cos(lat)
        cossq=cos_theta_latitude*cos_theta_latitude
        r3g=r*r*r/g
        rocos=omega*r*cos_theta_latitude
        delta_phi = lat[0] - lat[1]

        am_m3 = np.zeros(ps[:,:,0].shape)
        for k in range(90):
            for j in range(lat.shape[0]):
                am_m3[:,j]=am_m3[:,j]+np.mean((rocos[j] + u_mid[:,k,j])*cossq[j]*r3g*(da[k] + db[k]*ps[:,j])*2*np.pi*delta_phi ,axis=-1) ## (omega*a*cosphi + u) * cos^2 phi * r^3 * dp/g *2*pi*dphi

        am_cube = u[:,0,:,0].copy(am_m3)
        am_cube.var_name='AAM'
        am_cube.long_name='absolute_atmospheric_angular_momentum'
        am_cube.units = 'kg m**-1 s**-1'
        iris.save(am_cube, 'AAM_ERA20C_%04d%02d.nc' % (year,m))
