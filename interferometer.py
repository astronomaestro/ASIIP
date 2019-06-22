from intensityInterferometry import IIdata, IItools, IIdisplay, IImodels
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic, Angle, concatenate, concatenate_representations
import astropy.visualization as viz
from astropy.table import Table


def star_rater(tel_array, xlen, ylen, wavelength, cutoff_obs_time = 0.5):
    coverages=[]
    good_diams = []
    good_ra = []
    good_dec = []
    calc_ra=[]
    calc_dec=[]
    good_mags = []
    good_mag_names = []
    cat_names = []
    coords_done = SkyCoord([(0*u.deg,0*u.deg)])
    i = 0
    for catalog, cat_name in zip(tel_array.catalogs, tel_array.cat_names):
        ras, decs, ang_diams, mags, mag_name = tel_array.ra_dec_diam_getter(cat_name, catalog)
        pos_cat=SkyCoord(ras,decs)
        closest_star, skydis, distance3d = pos_cat.match_to_catalog_sky(coords_done)
        clo_idx = skydis > 1 * u.arcsec
        coords_done = SkyCoord([c.data for c in pos_cat[clo_idx]] + [c.data for c in coords_done])

        for ra,dec,ang_diam, mag in zip(ras[clo_idx], decs[clo_idx], ang_diams[clo_idx], mags[clo_idx]):
            total_obs_time = 0 * u.hour

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

            i = i+1
            # if i== 36:
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
            if percent_coverage > .95:
                asdf=23

            coverages.append(percent_coverage)
            good_diams.append(ang_diam.value)
            good_ra.append(ra.value)
            good_dec.append(dec.value)
            good_mag_names.append(mag_name)
            good_mags.append(mag)
            cat_names.append(cat_name)
    data_table = Table([np.array(good_ra), np.array(good_dec), np.array(good_diams), np.array(coverages),
                        np.array(good_mag_names), np.array(good_mags), np.array(cat_names)],
                       names=("RA","DEC","ANGDIAM","UVCOV","FILT","MAG","CAT"))
    return data_table

norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())

save_dir = "/home/jonathandd/Pictures/IIgraphs/IIPlaneCoverage"

T1_mcgill=np.array([135.48,-8.61,12.23])
T2_mcgill=np.array([44.1,-47.7,4.4])
T3_mcgill=np.array([29.4,60.1,9.8])
T4_mcgill=np.array([-35.9,11.3,7.0])


veritas_locs = [T1_mcgill, T2_mcgill, T3_mcgill, T4_mcgill]

baselines = IItools.array_baselines(veritas_locs)

time = '2019-6-13 07:00:00'
star_name = "SPICA"

veritas_telLat = 31.6749557
veritas_telLon = 360 - 110.9507624
veritas_telElv = 1268
veritas_array = IIdata.IItelescope(telLat=veritas_telLat, telLon=veritas_telLon, telElv=veritas_telElv, time=time, steps=1440)
veritas_tels = [veritas_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]


mag_range = (-3, 3)
dec_range = (-20, 90)
ra_range = ((veritas_array.dark_times.min()-4*u.hourangle).to('deg').value, veritas_array.dark_times.max().to('deg').value)
xlen=500
ylen=500
wavelength = 410e-9 * u.m
u_arcsec = 1e-3 * u.arcsec
# get stars that appeary during the specified night

# veritas_array.make_simbad_query(mag_range=mag_range,
#                               ra_range=ra_range,
#                               dec_range=dec_range)

veritas_array.make_gaia_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)



veritas_array.make_tess_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)



veritas_array.make_cadars_query(from_database=True,
                                mag_range=mag_range,
                                ra_range=ra_range,
                                dec_range=dec_range)



veritas_array.make_charm2_query(mag_range=mag_range,
                                ra_range=ra_range,
                                dec_range=dec_range)


veritas_array.make_jmmc_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)




veritas_cat = star_rater(tel_array=veritas_array,
                                          xlen=xlen,
                                          ylen=ylen,
                                          wavelength=wavelength)

np.set_printoptions(threshold=np.inf)
gcov = np.where(veritas_cat["UVCOV"] > 0)
bstidx = np.argsort(veritas_cat[gcov]["UVCOV"])
veritas_cat[gcov][bstidx][::-1].pprint(max_lines=10000000000, max_width=10000000000000)
from astroquery.simbad import Simbad
Simbad.add_votable_fields('flux(B)', 'flux(G)')
sim = Simbad.query_region(SkyCoord(veritas_cat[gcov][bstidx]["RA"], veritas_cat[gcov][bstidx]["RA"], unit=(u.deg, u.deg)))

print()











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

asfadsf=1234123

