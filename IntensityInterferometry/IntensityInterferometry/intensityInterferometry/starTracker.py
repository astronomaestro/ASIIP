import astropy
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic

starName = "CAPELLA"
starToTrack = SkyCoord.from_name(starName)

slc = EarthLocation(lat=40.76629 * u.deg, lon=-111.85005 * u.deg, height=1334 * u.m)
utcoffset = -6*u.hour  # Eastern Daylight Time

midnight = Time('2019-5-31 00:00:00', location=slc) - utcoffset
delta_midnight = np.linspace(-12, 12, 1000)*u.hour



slcFrame = AltAz(obstime=midnight + delta_midnight, location=slc)
# eclipFrame = BaseEclipticFrame(obstime=midnight+delta_midnight, location=slc)

starLoc = starToTrack.transform_to(slcFrame)
geoCentLoc = starToTrack.transform_to('geocentrictrueecliptic')
# eclipLoc = starToTrack.transform_to()
# plt.plot(delta_midnight, spicaSLC.alt)

from astropy.coordinates import get_sun

sunaltazs = get_sun(delta_midnight+midnight).transform_to(slcFrame)

hourAngle = starToTrack.ra.to("hourangle")

plt.scatter(slcFrame.obstime.sidereal_time('apparent') - starToTrack.ra, starLoc.alt.to("deg"),
            c=delta_midnight.to('hr'), label="%s HourAngle: %s"%(starName, hourAngle), lw=0, s=78,
            cmap='viridis')

# plt.fill_between(delta_midnight.to('hr').value, 0, 90,
#                  sunaltazs.alt < -0*u.deg, color='0.5', zorder=0)
# plt.fill_between(delta_midnight.to('hr').value, 0, 90,
#                  sunaltazs.alt < -18*u.deg, color='k', zorder=0)

plt.legend(loc='upper left')


plt.xlabel('Time from midnight')
plt.ylabel('Deg')
plt.show()



coordFrame = AltAz(obstime=midnight + delta_midnight,
                   location=slc)
SLCaltazs_July13night = starToTrack.transform_to(coordFrame)

SLCairmasss_July13night = SLCaltazs_July13night.secz
plt.plot(delta_midnight, SLCairmasss_July13night)
plt.show()

