from intensityInterferometry import IIdata, IItools, IIdisplay, IImodels
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic, Angle, concatenate, concatenate_representations
import astropy.visualization as viz

def star_rater(catalog, cat_name, tel_array, xlen, ylen, wavelength, cutoff_obs_time = 0.5):
    good_stars=[]
    coverages=[]
    ang_diams = []
    calc_ra=[]
    calc_dec=[]
    for i, star in enumerate(catalog):
        total_obs_time = 0 * u.hour
        ra, dec, ang_diam = tel_array.ra_dec_diam_getter(cat_name, star)

        if ra.value not in calc_ra and dec.value not in calc_dec:
            calc_ra.append(ra.value)
            calc_dec.append(dec.value)
            print("\nAnalyzing Star RA %s DEC %s at %s" % (ra, dec, i))

        else:
            print('|', end="")
            continue
        tel_array.star_track(ra=ra, dec=dec)
        if np.alen(tel_array.observable_times) > 0:
            total_obs_time = np.ptp(tel_array.observable_times)

        if total_obs_time < cutoff_obs_time * u.hour:
            print("Skipping star RA %s DEC %s as it is observable only %s" % (ra, dec, total_obs_time))
            continue

        # if i== 13:
        #     asdf=123

        tel_tracks = [IItools.uv_tracks(lat=tel_array.telLat.to('rad').value,
                                        dec=tel_array.dec.to('rad').value,
                                        hours=tel_array.sidereal_times.to('rad').value,
                                        Bn=Bns,
                                        Be=Bew,
                                        Bu=Bud) for Bns, Bew, Bud in
                      zip(tel_array.Bnss, tel_array.Bews, tel_array.Buds)]

        airy_disk, airy_func = IImodels.airy_disk2D(shape=(xlen, ylen),
                                                    xpos=xlen / 2,
                                                    ypos=ylen / 2,
                                                    arcsec=ang_diam,
                                                    wavelength=wavelength)

        percent_coverage = IItools.track_coverage(tel_tracks=tel_tracks,
                                                  airy_func=airy_func)

        coverages.append(percent_coverage)
        good_stars.append(i)
        ang_diams.append(ang_diam.value)
    return np.array(good_stars), np.array(coverages), np.array(ang_diams)

norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())

save_dir = "/home/jonathandd/Pictures/IIgraphs/IIPlaneCoverage"

T1_mcgill=np.array([135.48,-8.61,12.23])
T2_mcgill=np.array([44.1,-47.7,4.4])
T3_mcgill=np.array([29.4,60.1,9.8])
T4_mcgill=np.array([-35.9,11.3,7.0])

# IItools.airy_disk2D((200,200), 100, 100, 8)
# IItools.gaussian_disk2D()

veritas_locs = [T1_mcgill, T2_mcgill, T3_mcgill, T4_mcgill]

baselines = IItools.array_baselines(veritas_locs)


# xlen=500
# ylen=500
# wavelength = 415e-9 * u.m
# binary = IImodels.binary_visibility2D(shape=(xlen, ylen),
#                             flux_ratio=1,
#                             separation=.8e-3 *u.arcsec,
#                             arcsec1 = 3e-3 * u.arcsec,
#                             arcsec2 = 3e-3 * u.arcsec,
#                             wavelength=wavelength)
time = '2019-6-13 07:00:00'
star_name = "SPICA"

veritas_telLat = 31.6749557
veritas_telLon = 360 - 110.9507624
veritas_telElv = 1268
veritas_array = IIdata.IItelescope(telLat=veritas_telLat, telLon=veritas_telLon, telElv=veritas_telElv, time=time, steps=1440)
veritas_tels = [veritas_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]


mag_range = (0, 3)
dec_range = (-20, 90)
ra_range = ((veritas_array.dark_times.min()-4*u.hourangle).to('deg').value, veritas_array.dark_times.max().to('deg').value)
xlen=500
ylen=500
wavelength = 410e-9 * u.m
u_arcsec = 1e-3 * u.arcsec
# get stars that appeary during the specified night

veritas_array.make_gaia_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)

gaia_stars, gaia_coverages, gaia_diam = star_rater(catalog=veritas_array.gaia,
                                          cat_name="GAIA",
                                          tel_array=veritas_array,
                                          xlen=xlen,
                                          ylen=ylen,
                                          wavelength=wavelength)

veritas_array.make_tess_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)

tess_stars, tess_coverages, tess_diam = star_rater(catalog=veritas_array.tess,
                                          cat_name="TESS",
                                          tel_array=veritas_array,
                                          xlen=xlen,
                                          ylen=ylen,
                                          wavelength=wavelength)


veritas_array.make_cadars_query(from_database=True,
                                mag_range=mag_range,
                                ra_range=ra_range,
                                dec_range=dec_range)

cedar_stars, cedar_coverages, cedar_diam = star_rater(catalog=veritas_array.cedars,
                                          cat_name="CEDARS",
                                          tel_array=veritas_array,
                                          xlen=xlen,
                                          ylen=ylen,
                                          wavelength=wavelength)

veritas_array.make_charm2_query(mag_range=mag_range,
                                ra_range=ra_range,
                                dec_range=dec_range)

charm2_stars, charm2_coverages, charm2_diam = star_rater(catalog=veritas_array.charm2,
                                          cat_name="CHARM2",
                                          tel_array=veritas_array,
                                          xlen=xlen,
                                          ylen=ylen,
                                          wavelength=wavelength)




veritas_array.make_jmmc_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)
jmmc_stars, jmmc_coverages, jmmc_diam = star_rater(catalog=veritas_array.jmmc,
                                          cat_name="JMMC",
                                          tel_array=veritas_array,
                                          xlen=xlen,
                                          ylen=ylen,
                                          wavelength=wavelength)



gaia_coords = SkyCoord(veritas_array.gaia[gaia_stars]["RAJ2000"], veritas_array.gaia[gaia_stars]["DEJ2000"])
tess_coords = SkyCoord(veritas_array.tess[tess_stars]["RAJ2000"], veritas_array.tess[tess_stars]["DEJ2000"])
charm2_coords = SkyCoord(veritas_array.charm2[charm2_stars]["RAJ2000"], veritas_array.charm2[charm2_stars]["DEJ2000"], unit=(u.hourangle, u.deg))
cedars_coords = SkyCoord(veritas_array.cedars[cedar_stars]["RAJ2000"], veritas_array.cedars[cedar_stars]["DEJ2000"], unit=(u.hourangle, u.deg))
jmmc_coords = SkyCoord(veritas_array.jmmc[jmmc_stars]["RAJ2000"], veritas_array.jmmc[jmmc_stars]["DEJ2000"], unit=(u.hourangle, u.deg))



coords_ = [tess_coords, charm2_coords, cedars_coords, jmmc_coords]
one_mighty_catalog = SkyCoord(gaia_coords)
for cat in coords_:
    closest_star, skydis, distance3d = one_mighty_catalog.match_to_catalog_sky(cat)
    one_mighty_catalog = SkyCoord([c.data for c in one_mighty_catalog[skydis > 1*u.arcsec]] + [c.data for c in cat])
print()
print(np.sort(one_mighty_catalog.data._values))

# coords_ = [tess_coords, charm2_coords, cedars_coords, jmmc_coords]
#
# one_mighty_catalog = SkyCoord(gaia_coords)
#
# closest_star, skydis, distance3d = gaia_coords.match_to_catalog_sky(tess_coords)
# unique_idx = closest_star[skydis > 1*u.arcsec]
#
# not_in_gaia = tess_coords[unique_idx]
# concatenate([gaia_coords,not_in_gaia])
#
# one_mighty_catalog = SkyCoord(gaia_coords)




# for cat in coords_:
#     closest_star, skydis, distance3d = one_mighty_catalog.match_to_catalog_sky(cat)
#     unique_idx = closest_star[skydis > 1 * u.arcsec]
#
#     not_in_coords = cat[unique_idx]
#     one_mighty_catalog = concatenate([one_mighty_catalog, not_in_coords])



# coords_ = [tess_coords, charm2_coords, cedars_coords, jmmc_coords]
# one_mighty_catalog = SkyCoord(gaia_coords)
#
# for cat in coords_:
#     closest_star, skydis, distance3d = one_mighty_catalog.match_to_catalog_sky(cat)
#     unique_idx = closest_star[skydis > 1 * u.arcsec]
#     print(unique_idx)
#
#     not_in_coords = cat[unique_idx]
#     if np.alen(not_in_coords) > 0:
#         one_mighty_catalog = SkyCoord([c.data for c in one_mighty_catalog] + [c.data for c in not_in_coords])


np_cov = np.array(coverages)
np_goodStar = np.array(good_stars)
cov_sort = np.argsort(np_cov)
asdbvasdfv=23123

name = catalog[np_goodStar[cov_sort][-1]]["Name"]


for i in range(len(coverages)):
    print("%s\t%s\t%s\t%s" % (catalog[np_goodStar][cov_sort]["Name"][i], np_cov[cov_sort][i],
                              catalog[np_goodStar][cov_sort]["RAJ2000"][i], catalog[np_goodStar][cov_sort]["DEJ2000"][i]))







#
# IIdisplay.uvtracks_airydisk2D(tel_tracks, veritas_array, baselines, airy_disk, arcsec=u_arcsec, wavelength=wavelength,
#                               save_dir=save_dir)
# IIdisplay.uvtracks_amplitudes(tel_tracks, baselines, airy_func, arcsec=u_arcsec, wavelength=wavelength,
#                               save_dir=save_dir)
# IIdisplay.radial_profile_plot(airy_disk, center=None, data_name="Airy Disk", arcsec=u_arcsec, wavelength=wavelength,
#                               save_dir=save_dir)

#
# veritas_array.make_gaia_query(from_database=True,
#                                 mag_range=(1,3),
#                                 ra_range=(0,360),
#                                 dec_range=(-20,90))

# #
# # veritas_array.bright_star_cat(ra_range=ra_range,
#                                 dec_range=(-20,90))
#

# mag_range = (1, 3)
# dec_range = (-20, 90)
# veritas_array.tess_star_cat(mag_range=mag_range,
#                             ra_range=ra_range,
#                             dec_range=dec_range)
# veritas_array.make_jmmc_query(from_database=True,
#                               mag_range=mag_range,
#                               ra_range=ra_range,
#                               dec_range=dec_range,
#                               load_vizier=True)
#
# veritas_array.make_charm2_query(from_database=True,
#                                 mag_range=mag_range,
#                                 ra_range=ra_range,
#                                 dec_range=dec_range,
#                                 load_vizier=True)

# veritas_array.make_cadars_query(from_database=True,
#                                 mag_range=mag_range,
#                                 ra_range=(0,360),
#                                 dec_range=dec_range)

# veritas_array.make_gaia_query(from_database=True,
#                               mag_range=mag_range,
#                               ra_range=(0,360),
#                               dec_range=dec_range)
#
# plt.figure(figsize=(18,8))
# plt.hist(veritas_array.JMMC_stars["Bmag"])
# plt.title("Historgram from JMMC catalogue V2")
# plt.xlabel("B magnitude")
# plt.ylabel("Number of objects")


asfadsf=1234123

