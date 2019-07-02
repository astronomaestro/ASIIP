from IntensityInterferometry import IImodels, IIdisplay, IItools, IIdata
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
from scipy.stats import chisquare
import json
norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())


def deep_star_analysis(star, boot_runs = 100):
    ang_diam = star["ANGD"] * u.mas
    star_name = star["SIMID"]
    star_mag = star["BS_BMAG"]
    guess_r = wavelength.to('m').value / ang_diam.to('rad').value
    I_time = np.diff(veritas_array.integration_times)[1].value * u.h.to("s")

    star_err = IItools.track_error(sig1=veritas_array.err_sig,
                                   m1=veritas_array.err_mag,
                                   m2=star_mag,
                                   t1=veritas_array.err_t1,
                                   t2=I_time)

    tel_tracks, airy_disk, airy_func = star_airy_track(tel_array=veritas_array,
                                                       ang_diam=ang_diam,
                                                       wavelength=wavelength)

    fitdiams, fiterrs, failed_fits = IItools.IIbootstrap_analysis(tel_tracks=tel_tracks,
                                                                  airy_func=airy_func,
                                                                  star_err=star_err,
                                                                  guess_r=guess_r,
                                                                  wavelength=wavelength,
                                                                  runs=boot_runs)

    fdiam_mean = np.mean(fitdiams)
    ferr_mean = np.mean(fiterrs)
    ferr_std = np.std(fiterrs)


    print("%s: mean mas of 1000 fits is %s +- %.4f percent.\tfiterr std is %.4f +- %.4f percent with %s failed fits"
          % (star_name, fdiam_mean, ferr_mean / fdiam_mean * 100, ferr_std,
             ferr_std / ferr_mean * 100, failed_fits))
    return fdiam_mean, ferr_mean, ferr_std, failed_fits

def changing_veritas_values(varray, star):
    times = varray.sidereal_times
    name = star["SIMID"]

    print("So current sideral observation times for %s are %s to %s, with integration times of %s"
          %(name,np.min(times), np.max(times), times[1]-times[0]))

    chng_ver_vals = input("Do you want to change these? y for yes, anything else for no.\n")
    if chng_ver_vals.lower() == 'y':
        while True:
            try:
                print("Make sure you enter your times in the exact same format as printed before like -> '.2h.2m.2s' unless you want unexpected behavior.\n")
                mod_start = Angle(input("Please enter when you want the observation to start.\n"))
                mod_end = Angle(input("Please enter when you want the observation to end.\n"))
                integration_time = Angle(input("Please enter the integration time you desire IN THE SAME FORMAT\n"))
                varray.modify_obs_times(mod_start, mod_end, integration_time)
                break
            except Exception as e:
                print("Make sure you entered in the time in a format like %s and you chose a valid starting and end time"%(np.min(times)))

def star_airy_track(tel_array, ang_diam, wavelength):
    tel_tracks = [IItools.uv_tracks(lat=tel_array.telLat.to('rad').value,
                                    dec=tel_array.dec.to('rad').value,
                                    hours=tel_array.integration_times.to('rad').value,
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
    chis = []
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
            tel_tracks, airy_disk, airy_func = star_airy_track(tel_array=tel_array,
                                                               ang_diam=ang_diam,
                                                               wavelength=wavelength)

            r0_percent, r1_percent, r2_percent, r0_amp, r1_amp = IItools.track_coverage(tel_tracks=tel_tracks,
                                                                                        airy_func=airy_func)

            star_err = IItools.track_error(sig1=tel_array.err_sig,
                                           m1=tel_array.err_mag,
                                           m2=mag,
                                           t1=tel_array.err_t1,
                                           t2=obs_t)
            tr_amp, tr_rad, tr_Ints, tr_Irad, tr_aerrs, tr_xerr = IImodels.airy_disk1D(tel_tracks=tel_tracks,
                                                                                       airy_func=airy_func,
                                                                                       err=star_err)

            rIsort = np.argsort(tr_Irad)
            chi = chisquare(tr_Ints[rIsort] + tr_aerrs[rIsort], tr_Ints[rIsort])[0]

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
            chis.append(chi)
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
    amp_err_ratio = np_covs[:,3]/(amp_errs)

    data_table = Table([sim_id, col(good_ra, unit=u.hourangle), col(good_dec, unit=u.deg), col(good_diams, unit=u.mas),
                        np_covs[:,0], np_covs[:,1], np_covs[:,3], np_covs[:,4], good_mag_names, good_mags, cat_names, bmag,vmag,
                        bs_dis.to("mas"), bs_info["SpType"], bs_info["RotVel"], sim_bflux, sim_sptype, simd.to('mas'), sim_rotV,
                        col(amp_errs),col(total_obs_times, unit=u.second), col(obs_t, unit=u.second), amp_err_ratio, col(chis)],
                       names=("SIMID","RA","DEC","ANGD",
                              "R0COV","R1COV","R0AMP","R1AMP","FILT","MAG","CAT","BS_BMAG","BS_VMAG",
                              "BSSkyD", "BSSpT", "BSRV","SimBMAG", "SIMSpT", "SIMSkyD", "SIMRV",
                              "ErrAmp", "TotObsTime", "ObsTime", "AmpErrRatio", "chiSquare"))
    return data_table, np.array(tracks), np.array(airy_funcs)

def catalog_builder(veritas_array, wavelength =410e-9 * u.m, cat_name="veritasCatalog"):
    mag_range = veritas_array.mag_range
    dec_range = veritas_array.dec_range
    ra_range = veritas_array.ra_range
    xlen=veritas_array.xlen
    ylen=veritas_array.ylen


    # get stars that appeary during the specified night
    veritas_array.make_gaia_query(mag_range=mag_range,
                                  ra_range=ra_range,
                                  dec_range=dec_range)


    veritas_array.make_jmmc_query(mag_range=mag_range,
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
    with open("IIparameters.json") as param:
        #I use json as it is clear from the file itself what the variables are.
        IIparam = json.load(param)
        #The time is meant to be midnight on the day that you are observing. Make sure you correct for your timezone and daylight savings
        time = IIparam["time"]
        #this is the name of the output save file
        cat_name = "veritasCatalog%s.dat"%(time)
        curdir = os.path.dirname(__file__)
        #this is the directory where all analysis graphs will be saved.
        save_dir = os.path.join(curdir, "IIGraphs")

        #The wavelength of your filter
        wavelength = IIparam["wavelength"] * u.m

        #locations of all the different VERITAS telescopes relative to each other, used to construct the baselines
        veritas_baselines = IIparam["telLocs"]
        int_time = IIparam["integrationTime"]
        # steps details how fine you want the baseline graphs to be, fewer steps means faster processing but less detail in the models
        steps = IIparam['steps']


        #The latitude, longitude, and elevation of the center point of your array
        veritas_telLat = IIparam["telLat"]
        veritas_telLon = IIparam["telLon"]
        veritas_telElv = IIparam["telElv"]
        #How many bootstrapping runs you want to do per target. If it is
        boot_runs = IIparam["bootStrapRuns"]
        skip_prompts = IIparam["skipScript"]

    veritas_array = IIdata.IItelescope(telLat=veritas_telLat,
                                       telLon=veritas_telLon,
                                       telElv=veritas_telElv,
                                       time=time,
                                       steps=steps)

    baselines = IItools.array_baselines(veritas_baselines)
    [veritas_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]

    #allows for quicker running when set to true by skipping script prompts and setting them to values which are given
    #within the code itself
    if skip_prompts:
        print("Skipping script prompts and running the analysis with default values.")
        veritas_cat = ascii.read(cat_name)
        ind_col = col(np.arange(np.alen(veritas_cat)), name="Index")
        veritas_cat.add_column(ind_col, index=0)
        veritas_cat.pprint(max_lines=-1, max_width=-1)
        stop_sel = 'n'

        deep_analysis = []
        for selected_star in veritas_cat:
            veritas_array.star_track(ra=selected_star["RA"] * u.hourangle,
                                     dec=selected_star["DEC"] * u.deg)


            tims = veritas_array.integration_times
            obs_start = np.min(tims)
            obs_end = np.max(tims)
            veritas_array.modify_obs_times(start=obs_start,
                                           end=obs_end,
                                           int_time=Angle(int_time))

            fitdiam_mean, fiterr_mean, fiterr_std, failed_fits = deep_star_analysis(star=selected_star,
                                                                                    boot_runs=boot_runs)
            deep_analysis.append([fitdiam_mean, fiterr_mean, fiterr_std, failed_fits])
            abs=2323


        deep_analysis = np.array(deep_analysis)
        diam_percent_err = deep_analysis[:,1]/deep_analysis[:,0]
        err_of_diam_err = deep_analysis[:,2]/deep_analysis[:,1]
        veritas_cat.add_column(col(deep_analysis[:,0], name="MeanDiamFit", format="%.4f"))
        veritas_cat.add_column(col(diam_percent_err, name="PercentDiamFitErr", format="%.4f"))
        veritas_cat.add_column(col(err_of_diam_err, name="PercentErrOfDiamFitErr", format="%.4f"))
        veritas_cat.add_column(col(deep_analysis[:,3] / boot_runs, name="PercentOfFailedFits", format="%.4f"))

        lowerr = np.argsort(veritas_cat, order=["PercentOfFailedFits","PercentDiamFitErr"])
        veritas_cat[lowerr].pprint(max_lines=-1, max_width=-1)


        deep_analysis = np.array(deep_analysis)
        fit_failures = deep_analysis[:,3]
        not_complete_failures = np.where(np.array(fit_failures) < boot_runs / 4)


        asdqwer=1239


    #If skip_prompts is false, ask the user what values they desire to be analyzed
    else:
        veritas_cat = None
        #Figure out if there is a catalog created already and/or if the user wants to create a catalog
        if cat_name in os.listdir():
            while not veritas_cat:
                response = input("It seems you have a catalog called %s, do you wish to use it? y or n\n"%(cat_name))
                if response == 'y':
                    veritas_cat = ascii.read(cat_name)
                elif response == "n":
                    print("Creating new catalog")
                    catalog_builder(veritas_array=veritas_array,
                                    wavelength=wavelength,
                                    cat_name=cat_name)
                    veritas_cat = ascii.read(cat_name)
                else:
                    print("You have to type y for yes or n for no.\n")
        else:
            print("No catalog found for time %s, creating a new catalog."%(time))
            catalog_builder(veritas_array=veritas_array,
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

                changing_veritas_values(veritas_array, selected_star)



                ang_diam = selected_star["ANGD"] * u.mas
                star_name = selected_star["SIMID"]
                star_mag = selected_star["BS_BMAG"]
                I_time = np.diff(veritas_array.integration_times)[0].value*u.h.to("s")
                star_err = IItools.track_error(sig1=veritas_array.err_sig,
                                               m1=veritas_array.err_mag,
                                               m2=star_mag,
                                               t1=veritas_array.err_t1,
                                               t2=I_time)

                tel_tracks, airy_disk, airy_func = star_airy_track(tel_array=veritas_array,
                                                                   ang_diam=ang_diam,
                                                                   wavelength=wavelength)

                IIdisplay.uvtracks_integrated(varray=veritas_array,
                                              tel_tracks=tel_tracks,
                                              airy_func=airy_func,
                                              save_dir=save_dir,
                                              name="%sIntegrationTime%.4f"%(star_name, I_time),
                                              err=star_err,
                                              noise=False)

                IIdisplay.uvtracks_integrated(varray=veritas_array,
                                              tel_tracks=tel_tracks,
                                              airy_func=airy_func,
                                              save_dir=save_dir,
                                              name="%sIntegrationTimeWNoise%.4f"%(star_name, I_time),
                                              err=star_err,
                                              noise=True)

                IIdisplay.uvtracks_airydisk2D(tel_tracks=tel_tracks,
                                              veritas_tels=veritas_array,
                                              baselines=baselines,
                                              airy_disk=airy_disk,
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

# selected_star = veritas_cat[63]
# int_time = ".5h"
# veritas_array.star_track(ra=selected_star["RA"] * u.hourangle,
#                          dec=selected_star["DEC"] * u.deg)
#
# tims = veritas_array.integration_times
# obs_start = np.min(tims)
# obs_end = np.max(tims)
# veritas_array.modify_obs_times(start=obs_start - 0*u.hourangle,
#                                end=obs_end,
#                                int_time=Angle(int_time))
#
# ang_diam = selected_star["ANGD"] * u.mas
# star_name = selected_star["SIMID"]
# star_mag = selected_star["BS_BMAG"]
# guess_r = wavelength.to('m').value / ang_diam.to('rad').value
# I_time = np.diff(veritas_array.integration_times)[1].value * u.h.to("s")
#
# star_err = IItools.track_error(sig1=veritas_array.err_sig,
#                                m1=veritas_array.err_mag,
#                                m2=star_mag,
#                                t1=veritas_array.err_t1,
#                                t2=I_time)
#
# tel_tracks, airy_disk, airy_func = star_airy_track(tel_array=veritas_array,
#                                                    ang_diam=ang_diam,
#                                                    wavelength=wavelength)
#
# fitdiams, fiterrs, failed_fits = IItools.IIbootstrap_analysis(tel_tracks=tel_tracks,
#                                                               airy_func=airy_func,
#                                                               star_err=star_err,
#                                                               guess_r=guess_r,
#                                                               wavelength=wavelength,
#                                                               runs=light_boot)
# tr_amp, tr_rad, tr_Ints, tr_Irad, tr_aerrs, tr_xerr = IImodels.airy_disk1D(tel_tracks=tel_tracks,
#                                                                            airy_func=airy_func,
#                                                                            err=star_err)
# rIsort = np.argsort(tr_Irad)
#
# plt.figure(figsize=(28, 24))
# # plt.ylim(-.1,.1)
#
# plt.errorbar(x=tr_Irad,
#              y=tr_Ints + tr_aerrs,
#              fmt='o',
#              yerr=np.full(np.alen(tr_Ints), star_err),
#              xerr=tr_xerr,
#              label="Model w/ err")
# rsort = np.argsort(tr_rad)
# plt.plot(tr_rad[rsort], tr_amp[rsort], '-', label="Actual Function")
# # plt.figure(figsize=(28,24))
# airy_r1D = np.linspace(0, np.max(tr_rad), np.alen(tr_rad))
# airy_mod = IImodels.airy1D(airy_r1D, r=guess_r)
# plt.plot(airy_r1D, airy_mod)
# plt.plot(tr_Irad, tr_Ints, 'o', label="Acutal Integration")
# airy_fitr, airy_fiterr = IImodels.fit_airy1D(rx=tr_Irad[rIsort],
#                                              airy_amp=tr_Ints[rIsort] + tr_aerrs[rIsort],
#                                              guess_r=guess_r,
#                                              errs=np.full(np.alen(tr_Ints), star_err))
# fit_diam = (wavelength.to('m').value / airy_fitr[0] * u.rad).to('mas')
# fit_err = np.sqrt(np.diag(airy_fiterr))[0] / airy_fitr[0] * fit_diam
# plt.plot(airy_r1D, IImodels.airy1D(airy_r1D, airy_fitr[0]), label="Fitted Model")
# title = "Star %s Integration time of %s\n fit mas of data is %s +- %s or %.2f percent" % (
#     star_name, int_time, fit_diam, fit_err, fit_err / fit_diam * 100)
# plt.title(title, fontsize=28)
# plt.legend()