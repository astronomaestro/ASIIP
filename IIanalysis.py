from II import IImodels, IIdisplay, IItools, IIdata
import asiip
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.visualization as viz
from astropy.table import Table
from astropy.table import Column as col
from astropy.io import ascii
import os
from matplotlib import pyplot as plt
from astropy.coordinates import Angle
import json
import sys
import time as timer


jsondat = open("IIparameters.json")
IIparam = json.load(jsondat)

steps = 800

tel_array = IIdata.IItelescope(telLat=IIparam['telLat'],
                               telLon=IIparam['telLon'],
                               telElv=IIparam['telElv'],
                               time=IIparam['time'],
                               steps=steps,
                               sig1=IIparam['sigmaTel'],
                               m1=IIparam['sigmaMag'],
                               t1=IIparam['sigmaTime'],
                               mag_range=IIparam['magRange'],
                               dec_range=IIparam['decRange'],
                               ra_range=IIparam['raRange'],
                               max_sun_alt=IIparam['maxSunAltitude'],
                               timestep=IIparam['integrationTime'])

relative_tel_locs = np.array(IIparam["telLocs"])
baselines = IItools.array_baselines(relative_tel_locs)
[tel_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]

ang_diam = (.5 * u.mas).to('rad').value


dec_angle_rad = (5.6 * u.hourangle).to('rad').value
lat = tel_array.telLat.to('rad').value


hour_angle_rad = [-1.06417879, -0.93292069, -0.80166259, -0.67040449, -0.53914639, -0.40788829,
 -0.27663019, -0.14537209, -0.01411399,  0.11714411,  0.24840222,  0.37966032,
  0.51091842,  0.64217652,  0.77343462,  0.90469272,  1.03595082]

tel_tracks = [IItools.uv_tracks(lat=lat,
                                dec=dec_angle_rad,
                                hours=hour_angle_rad,
                                Bn=Bns,
                                Be=Bew,
                                Bu=Bud) for Bns, Bew, Bud in
              zip(tel_array.Bnss, tel_array.Bews, tel_array.Buds)]

airy_disk, airy_func = IImodels.airy_disk2D(shape=(tel_array.xlen, tel_array.ylen),
                                            xpos=tel_array.xlen,
                                            ypos=tel_array.ylen,
                                            angdiam=ang_diam,
                                            wavelength=IIparam['wavelength'])

binary_model = IImodels.binary_visibility2D(shape=(tel_array.xlen, tel_array.ylen),
                                            flux_ratio=.5,
                                            separation=5*u.mas,
                                            wavelength=IIparam['wavelength']*u.m,
                                            arcsec1=ang_diam,
                                            arcsec2=ang_diam)

asdf=234

