import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic
import astroquery
from intensityInterferometry import IIdata, IItools, IIdisplay, IImodels
from astroquery.gaia import Gaia
T1_mcgill=np.array([135.48,-8.61,12.23])
T2_mcgill=np.array([44.1,-47.7,4.4])
T3_mcgill=np.array([29.4,60.1,9.8])
T4_mcgill=np.array([-35.9,11.3,7.0])

# IItools.airy_disk2D((200,200), 100, 100, 8)
# IItools.gaussian_disk2D()

veritas_locs = [T1_mcgill, T2_mcgill, T3_mcgill, T4_mcgill]

baselines, tel_names = IItools.array_baselines(veritas_locs)

veritas_telLat = 31.6749557
veritas_telLon = -110.9507624
veritas_telElv = 1268
veritas_tels = [IIdata.IItelescope(telLat=veritas_telLat, telLon=veritas_telLon, telElv=veritas_telElv) for base in baselines]

# source_table.get_coulumns()


time = '2019-6-10 07:00:00'
obs_time = Time(time)
skypos = SkyCoord(0*u.deg, 0*u.deg)
star_name = "SPICA"
delta_time = np.linspace(-2, 4, 144*5) * u.hour

telFrame = AltAz(obstime=obs_time+delta_time, location=veritas_tels[0].tel_loc)

loc = EarthLocation(lat=veritas_telLat * u.deg, lon=veritas_telLon * u.deg, height=veritas_telElv * u.m)

skyAltAz = skypos.transform_to(telFrame)



asdf=1223
