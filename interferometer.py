from IntensityInterferometry import IImodels, IIdisplay, IItools, IIdata
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.visualization as viz
from astropy.table import Table
from astropy.table import Column as col
from astropy.io import ascii
import os
norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())

def star_analysis(tel_array,ang_diam, wavelength):
    tel_tracks = [IItools.uv_tracks(lat=tel_array.telLat.to('rad').value,
                                    dec=tel_array.dec.to('rad').value,
                                    hours=tel_array.sidereal_times.to('rad').value,
                                    Bn=Bns,
                                    Be=Bew,
                                    Bu=Bud) for Bns, Bew, Bud in
                  zip(tel_array.Bnss, tel_array.Bews, tel_array.Buds)]

    airy_disk, airy_func = IImodels.airy_disk2D(shape=(tel_array.xlen, tel_array.ylen),
                                                xpos=tel_array.xlen / 2,
                                                ypos=tel_array.ylen / 2,
                                                arcsec=ang_diam,
                                                wavelength=wavelength)
    return tel_tracks, airy_disk, airy_func

def star_rater(tel_array, xlen, ylen, wavelength, cutoff_obs_time = 0.5, obs_t=None):
    coverages=[]
    good_diams = []
    good_ra = []
    good_dec = []
    calc_ra=[]
    calc_dec=[]
    good_mags = []
    good_mag_names = []
    cat_names = []
    tracks = []
    airy_funcs = []
    total_obs_times = []
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
            tel_tracks, airy_disk, airy_func = star_analysis(tel_array=tel_array,
                                                             ang_diam=ang_diam,
                                                             wavelength=wavelength)

            r0_percent, r1_percent, r2_percent, r0_amp, r1_amp = IItools.track_coverage(tel_tracks=tel_tracks,
                                                                                        airy_func=airy_func)

            if r0_percent > .95:
                asdf=23

            coverages.append([r0_percent,r1_percent,r2_percent, r0_amp, r1_amp])
            good_diams.append(ang_diam.to("mas").value)
            good_ra.append(ra.value)
            good_dec.append(dec.value)
            good_mag_names.append(mag_name)
            good_mags.append(mag)
            cat_names.append(cat_name)
            total_obs_times.append(total_obs_time.to('s').value)
            tracks.append(tel_tracks)
            airy_funcs.append(airy_func)
    np_covs = np.array(coverages)

    bs_mat, bs_dis, bs_3dis = SkyCoord(good_ra, good_dec,
                                       unit=(u.hourangle, u.deg)).match_to_catalog_sky(
        SkyCoord(tel_array.BS_stars["RAJ2000"], tel_array.BS_stars["DEJ2000"], unit=(u.hourangle, u.deg)))
    bs_info = tel_array.BS_stars[bs_mat]

    simbad_matches, simd = tel_array.simbad_matcher(good_ra, good_dec)
    sim_rotV = simbad_matches["ROT_Vsini"]
    sim_sptype = simbad_matches["SP_TYPE"]
    sim_bflux = simbad_matches["FLUX_B"]
    sim_id = simbad_matches["MAIN_ID"]
    vmag = bs_info["Vmag"]
    bmag = bs_info["B-V"] + vmag

    if obs_t == None:
        obs_t = np.array(total_obs_times)
    else:
        obs_t = np.full(np.alen(total_obs_times),obs_t)
    amp_errs = IItools.track_error(sig1=tel_array.err_sig,
                                   m1=tel_array.err_mag,
                                   m2=bmag,
                                   t1=tel_array.err_t1,
                                   t2=obs_t)
    amp_err_ratio = np_covs[:,3]/amp_errs

    data_table = Table([sim_id, col(good_ra, unit=u.hourangle), col(good_dec, unit=u.deg), col(good_diams, unit=u.mas),
                        np_covs[:,0], np_covs[:,1], np_covs[:,3], np_covs[:,4], good_mag_names, good_mags, cat_names, bmag,vmag,
                        bs_dis.to("mas"), bs_info["SpType"], bs_info["RotVel"], sim_bflux, sim_sptype, simd.to('mas'), sim_rotV,
                        col(amp_errs),col(total_obs_times, unit=u.second), col(obs_t, unit=u.second), amp_err_ratio],
                       names=("SIMID","RA","DEC","ANGD",
                              "R0COV","R1COV","R0AMP","R1AMP","FILT","MAG","CAT","BS_BMAG","BS_VMAG",
                              "BSSkyD", "BSSpT", "BSRV","SimBMAG", "SIMSpT", "SIMSkyD", "SIMRV",
                              "ErrAmp", "TotObsTime", "ObsTime", "AmpErrRatio"))
    return data_table, np.array(tracks), np.array(airy_funcs)

def veritas_catalog_builder(veritas_array, wavelength = 410e-9 * u.m, cat_name="veritasCatalog"):
    mag_range = veritas_array.mag_range
    dec_range = veritas_array.dec_range
    ra_range = veritas_array.ra_range
    xlen=veritas_array.xlen
    ylen=veritas_array.ylen


    # get stars that appeary during the specified night
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

    veritas_array.bright_star_cat(ra_range=ra_range,
                                  dec_range=dec_range)



    veritas_cat, tracks, airy_funcs = star_rater(tel_array=veritas_array,
                                              xlen=xlen,
                                              ylen=ylen,
                                              wavelength=wavelength,
                                                 obs_t=1800)

    print()
    for col in veritas_cat.colnames:
        if veritas_cat[col].dtype in [np.float32, np.float64]:
            veritas_cat[col].format = "%6.4f"
    np.set_printoptions(threshold=np.inf)
    bstidx = np.argsort(veritas_cat, order=["AmpErrRatio", "R1COV"])
    ascii.write(veritas_cat[bstidx][::-1], cat_name)
    absdf=123



if __name__ == "__main__":
    print("Welcome to the VERITAS catalog builder. Please make sure you are running the catalog for the desire night.\n")
    time = '2019-6-30 07:00:00'
    cat_name = "veritasCatalog%s.dat"%(time)
    curdir = os.path.dirname(__file__)
    save_dir = os.path.join(curdir, "IIGraphs")

    wavelength = 410e-9 * u.m
    veritas_cat = None

    T1_mcgill = np.array([135.48, -8.61, 12.23])
    T2_mcgill = np.array([44.1, -47.7, 4.4])
    T3_mcgill = np.array([29.4, 60.1, 9.8])
    T4_mcgill = np.array([-35.9, 11.3, 7.0])
    veritas_baselines = [T1_mcgill, T2_mcgill, T3_mcgill, T4_mcgill]

    steps = 600
    veritas_telLat = 31.6749557
    veritas_telLon = 360 - 110.9507624
    veritas_telElv = 1268
    veritas_array = IIdata.IItelescope(telLat=veritas_telLat,
                                       telLon=veritas_telLon,
                                       telElv=veritas_telElv,
                                       time=time,
                                       steps=steps)

    baselines = IItools.array_baselines(veritas_baselines)
    [veritas_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]

    #Figure out if there is a catalog created already and/or if the user wants to create a catalog
    if cat_name in os.listdir():
        while not veritas_cat:
            response = input("It seems you have a catalog called %s, do you wish to use it? y or n\n"%(cat_name))
            if response == 'y':
                veritas_cat = ascii.read(cat_name)
            elif response == "n":
                print("Creating new catalog")
                veritas_catalog_builder(veritas_array=veritas_array,
                                        wavelength=wavelength,
                                        cat_name=cat_name)
                veritas_cat = ascii.read(cat_name)
            else:
                print("You have to type y for yes or n for no.\n")
    else:
        print("No catalog found for time %s, creating a new catalog."%(time))
        veritas_catalog_builder(veritas_array=veritas_array,
                                wavelength=wavelength,
                                cat_name=cat_name)
        veritas_cat = ascii.read(cat_name)

    ind_col = col(np.arange(np.alen(veritas_cat)), name="Index")
    veritas_cat.add_column(ind_col, index=0)
    veritas_cat.pprint(max_lines=-1, max_width=-1)
    stop_sel = 'n'



    while stop_sel != 'q':
        try:
            index = int(input("Please enter the index of the star you want to analyze\n"))
            selected_star = veritas_cat[index]
            print("\nNow analyzing your selected star\n%s\n"%(selected_star))

            veritas_array.star_track(ra=selected_star["RA"]*u.hourangle,
                                     dec=selected_star["DEC"]*u.deg)

            ang_diam = selected_star["ANGD"] * u.mas
            star_name = selected_star["SIMID"]
            star_err = selected_star["ErrAmp"]

            tel_tracks, airy_disk, airy_func = star_analysis(tel_array=veritas_array,
                                                             ang_diam=ang_diam,
                                                             wavelength=wavelength)
            IIdisplay.uvtracks_airydisk2D(tel_tracks=tel_tracks,
                                          veritas_tels=veritas_array,
                                          baselines=baselines,
                                          airy_disk=airy_disk,
                                          arcsec=ang_diam,
                                          wavelength=wavelength,
                                          save_dir=save_dir,
                                          name=star_name,
                                          err=star_err)
            IIdisplay.uvtracks_amplitudes(tel_tracks=tel_tracks,
                                          baselines=baselines,
                                          airy_func=airy_func,
                                          arcsec=ang_diam,
                                          wavelength=wavelength,
                                          save_dir=save_dir,
                                          name=star_name,
                                          err=star_err)
            stop_sel = input("Do you wish to quit? Enter q to quit, anything else to coninue.")
        except Exception as e:
            print(e)
            print("You must enter a valid index. If a star has an index of 0 and you want to analyze it, enter in a 0")

    possible_plots = ["UV tracks overlay with airydisk"]
    asdf=123







# veritas_cat[np.argsort(veritas_cat, order=["RA", "DEC"])].pprint(max_lines=10000000000, max_width=10000000000000)

# np.set_printoptions(threshold=np.inf)
# bstidx = np.argsort(veritas_cat, order=["R0COV", "R1COV", "R2COV"])
# bluei = np.where(veritas_cat[bstidx]["BS_BMAG"]<3)
# veritas_cat[bstidx][bluei][::-1].pprint(max_lines=10000000000, max_width=10000000000000)



# veritas_cat[np.where(np.abs(veritas_cat["MAG"]-veritas_cat["BS_VMAG"])>.2)].pprint(max_lines=10000000000, max_width=10000000000000)

# veritas_cat[np.where(np.abs(veritas_cat["MAG"]-veritas_cat["BS_VMAG"])>.2)].pprint(max_lines=10000000000, max_width=10000000000000)



# IIdisplay.radial_profile_plot(airy_disk, center=None, data_name="Airy Disk", arcsec=u_arcsec, wavelength=wavelength,
#                               save_dir=save_dir)

#
# veritas_array.make_gaia_query(from_database=True,
#                                 mag_range=(1,3),
#                                 ra_range=(0,360),
#                                 dec_range=(-20,90))

asfadsf=1234123

