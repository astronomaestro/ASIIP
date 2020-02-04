from II import IImodels, IIdisplay, IItools, IIdata
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
#norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())
sep_line = "-----------------------------------------------------------------------------------------------------------"
red_line = '\n\x1b[1;31;40m' + sep_line + '\x1b[0m\n'




def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def do_plots(tel_array, baselines, tel_tracks, ang_diam, airy_func, star_err, guess_r, wavelength, name, I_time, pererr, star_save):
    IIdisplay.uvtrack_model_run(tel_tracks=tel_tracks,
                                airy_func=airy_func,
                                star_err=star_err,
                                guess_r=guess_r,
                                wavelength=wavelength,
                                star_name=name,
                                ITime=I_time,
                                save_dir=star_save,
                                pererr=pererr,
                                fullAiry=True)

    IIdisplay.uvtracks_airydisk2D(tel_tracks=tel_tracks,
                                  veritas_tels=tel_array,
                                  baselines=baselines,
                                  airy_func=airy_func,
                                  guess_r=guess_r,
                                  wavelength=wavelength,
                                  save_dir=star_save,
                                  star_name=name)

    IIdisplay.chi_square_anal(tel_tracks=tel_tracks,
                              airy_func=airy_func,
                              star_err=star_err,
                              guess_r=guess_r,
                              ang_diam=ang_diam,
                              star_name=name,
                              save_dir=star_save)

def star_info(star, wavelength):
    if star["NAME"]:
        name = str.strip(star["NAME"], " *")
    else:
        name = "NONAME"
    ra = star["RA"] * u.hourangle
    dec = star["DEC"] * u.deg

    if "SimBMAG" in star.colnames:
        star_mag = star["SimBMAG"]
    else:
        star_mag = star["MAG"]

    ang_diam = (star["ANGD"] * u.mas).to('rad').value
    guess_diam = wavelength / ang_diam
    star_id = str(ra) + str(dec)
    return name, ra, dec, star_mag, ang_diam, guess_diam, star_id

def star_model(tel_array, I_time, star_mag, ang_diam, wavelength, star_id):
    star_err = IItools.track_error(sig0=tel_array.err_sig,
                                   m0=tel_array.err_mag,
                                   m=star_mag,
                                   T_0=tel_array.err_t1,
                                   T=I_time.to("s").value)

    hour_angle_rad = Angle(tel_array.star_dict[star_id]["IntSideTimes"]).to('rad').value
    dec_angle_rad = tel_array.star_dict[star_id]["DEC"].to('rad').value
    lat = tel_array.telLat.to('rad').value

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
                                                wavelength=wavelength)
    return star_err, hour_angle_rad, dec_angle_rad, lat, tel_tracks, airy_disk, airy_func

def siicat_constructor(tel_array, cutoff_obs_time=0, Int_obst=None):
    """
    This function takes every entry from every target catalog and combines the information into one big master catalog.
    It will prioritize which catalogs to analyze first based upon the order the catalogs were queried. The way it was
    written was for optimization along with an ability to keep the sizing of the arrays dynamic, which is why python
    lists were used, along with how duplicates are identified. It was kept seperate from the main part of the script
    to keep the script more readable.
    :param tel_array: The object which includes all of the information from the catalogs that have been queried
    :param cutoff_obs_time: The minimum amount of time you wish a star to be observable. Any star that is below or
    equal to this value is skipped. 0 means if it can't be observed at anytime during the night, it is skipped.
    The units the script expects is hours
    :param Int_obst: The integration time used for the data you will be analyzing. This is used to give a reference
    to how much error an observation will have
    :return: The completed catalog as an Astropy table
    """
    c = 0

    good_diams = []
    good_ra = []
    good_dec = []
    calc_ra=[]
    calc_dec=[]
    good_mags = []
    good_mag_names = []
    cat_names = []
    total_obs_times = []
    dup_count = []
    coords_done = None
    for catalog, cat_name in zip(tel_array.catalogs, tel_array.cat_names):
        ras, decs, ang_diams, mags, mag_name = tel_array.ra_dec_diam_getter(cat_name, catalog)
        pos_cat=SkyCoord(ras,decs)



        if np.alen(pos_cat) ==0:continue
        if coords_done == None:
            closest_star, skydis, distance3d = (np.array([]), Angle(np.ones(np.alen(ras)),unit=u.rad), np.full(np.alen(ras),True))
            coords_done = SkyCoord(good_ra, good_dec, unit=("hourangle", "deg"))

        else:
            coords_done = SkyCoord(good_ra, good_dec, unit=("hourangle","deg"))
            # match pos_cat to the closest star in coords_done returning the closest coords_done indicies

            closest_star, skydis, distance3d = pos_cat.match_to_catalog_sky(coords_done)


        # the indicies where the matched pos_cat star is large enough to be considered a unique star
        unq_idx = skydis > 1 * u.arcsec

        dists = np.full(np.alen(ras),np.inf)
        rs = np.argsort(ras)
        dup_diams = []
        i = 0
        for ra,dec,ang_diam, mag, unique in zip(ras, decs, ang_diams, mags, unq_idx):
            total_obs_time = 0 * u.hour

            dis = (ra.to('deg').value + dec.to("deg").value)
            if unique and (ra.value not in calc_ra and dec.value not in calc_dec):
                print("\nAnalyzing Star RA %s DEC %s at %s" % (ra, dec, i))

                star_id = str(ra) + str(dec)
                tel_array.star_track(ra=ra,
                                     dec=dec,
                                     alt_cut=alt_cut)

                if tel_array.star_dict[star_id]["ObsTimes"]:
                    if np.alen(tel_array.star_dict[star_id]["ObsTimes"]) > 1:
                        # total_obs_time = np.ptp(tel_array.star_dict[star_id]["ObsTimes"])
                        total_obs_time = (tel_array.time_delt * np.alen(tel_array.star_dict[star_id]["ObsTimes"])).to('s')
                    else: total_obs_time=0*u.hour

                if total_obs_time <= cutoff_obs_time * u.hour:
                    print("Skipping star RA %s DEC %s as it is observable only %s" % (ra, dec, total_obs_time))
                    dup_diams = []
                    continue

                dists[i] = dis
                calc_ra.append(ra.value)
                calc_dec.append(dec.value)
                dup_diams.append([ang_diam.to("mas").value, ra, dec, 1, cat_name])
                if i== 14:
                    asdf=123
                i = i+1


                good_diams.append(ang_diam.to("mas").value)
                good_ra.append(ra.value)
                good_dec.append(dec.value)
                good_mag_names.append(mag_name)
                good_mags.append(mag)
                cat_names.append(cat_name)
                total_obs_times.append(total_obs_time.to('s').value)

                if np.alen(dup_diams) > 1:
                    adsf=234
                dup_count.append(dup_diams)
                dup_diams = []

            elif not unique:
                cat_idx = closest_star[i]
                # print((np.array(good_ra) - ra.value)[cat_idx])
                dup_count[cat_idx].append([ang_diam.to("mas").value, good_ra[cat_idx], good_dec[cat_idx],2,cat_name])
                i = i + 1
            else:
                print('|', end="")
                dupind = calc_dec.index(dec.value)
                dup_count[dupind].append([ang_diam.to("mas").value, ra, dec, 3,cat_name])
                c=c+1
                i=i+1

    diamstd = np.array([np.std(np.array(np.array(r)[:,0],float)) for r in dup_count])
    diammean = np.array([np.median(np.array(np.array(r)[:,0],float)) for r in dup_count])

    if Int_obst == None:
        Int_obst = np.array(total_obs_times)
    else:
        Int_obst = np.full(np.alen(total_obs_times), Int_obst)

    bs_mat, bs_dis, bs_3dis = SkyCoord(good_ra,
                                       good_dec,
                                       unit=(u.hourangle, u.deg)).match_to_catalog_sky(
        SkyCoord(tel_array.BS_stars["RAJ2000"], tel_array.BS_stars["DEJ2000"], unit=(u.hourangle, u.deg)))

    bs_info = tel_array.BS_stars[bs_mat]
    vmag = bs_info["Vmag"]
    bmag = bs_info["B-V"] + vmag
    amp_errs = IItools.track_error(sig0=tel_array.err_sig,
                                   m0=tel_array.err_mag,
                                   m=bmag,
                                   T_0=tel_array.err_t1,
                                   T=Int_obst)

    try:
        simbad_matches, simd = tel_array.simbad_matcher(good_ra, good_dec)
        sim_rotV = simbad_matches["ROT_Vsini"]
        sim_sptype = simbad_matches["SP_TYPE"]
        sim_bflux = simbad_matches["FLUX_B"]
        sim_id = simbad_matches["MAIN_ID"]
        data_table = Table([sim_id.astype("U13"), good_ra, good_dec, good_diams, diammean, diamstd,
                            good_mag_names, good_mags, cat_names, bmag, vmag,
                            bs_dis.to("mas"), bs_info["SpType"], bs_info["RotVel"], sim_bflux, sim_sptype, simd.to('mas'), sim_rotV,
                            col(amp_errs), col(total_obs_times, unit=u.second), col(Int_obst, unit=u.second)],
                           names=("NAME","RA","DEC","ANGD","DiamMedian","DiamStd",
                                  "FILT","MAG","CAT","BS_BMAG","BS_VMAG",
                                  "BSSkyD", "BSSpT", "BSRV","SimBMAG", "SIMSpT", "SIMSkyD", "SIMRV",
                                  "ErrAmp", "TotObsTime", "ObsTime"))
    except Exception as e:
        print(e)
        print("SIMBAD probably failed, using the Bright Star Catalog only")
        data_table = Table([bs_info["Name"], good_ra, good_dec, good_diams, diammean, diamstd,
                            good_mag_names, good_mags, cat_names, bmag, vmag,
                            bs_dis.to("mas"), bs_info["SpType"], bs_info["RotVel"],
                            col(amp_errs), col(total_obs_times, unit=u.second), col(Int_obst, unit=u.second)],
                           names=("NAME","RA","DEC","ANGD","DiamMedian","DiamStd",
                                  "FILT","MAG","CAT","BS_BMAG","BS_VMAG",
                                  "BSSkyD", "BSSpT", "BSRV",
                                  "ErrAmp", "TotObsTime", "ObsTime"))


    return data_table

def catalog_builder(tel_array, cat_name="MasterSIICatalog"):
    """
    Makes the the quries needed from each of the individual catalogs and then constructs the master SII catalog from
    the information returned from the queries
    :param tel_array: The object which contains all the information about which telescopes you will be using, the
    location of the observatory itself,along with the constraints defined in the parameter file
    :param cat_name: The name of the catalog the software will save the
    :return: The function saves the catalog as an ASCII file which can be then be loaded.
    """
    mag_range = tel_array.mag_range
    dec_range = tel_array.dec_range
    ra_range = tel_array.ra_range


    # get stars that appeary during the specified night
    tel_array.make_gaia_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)


    tel_array.make_jmmc_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)

    tel_array.make_tess_query(mag_range=mag_range,
                              ra_range=ra_range,
                              dec_range=dec_range)

    tel_array.make_charm2_query(mag_range=mag_range,
                                ra_range=ra_range,
                                dec_range=dec_range)

    tel_array.make_cadars_query(mag_range=mag_range,
                                ra_range=ra_range,
                                dec_range=dec_range)

    tel_array.bright_star_cat(ra_range=ra_range,
                              dec_range=dec_range)



    masterSII_cat = siicat_constructor(tel_array=tel_array, Int_obst=1800)

    print()
    for colu in masterSII_cat.colnames:
        if masterSII_cat[colu].dtype in [np.float32, np.float64]:
            #Since it is unlikely there is a precision greater then 4 decimal places in any of the values, round to
            #four decimal places as it makes the catalog much more ledgible
            masterSII_cat[colu].format = "%6.4f"
    np.set_printoptions(threshold=np.inf)

    errAsort = np.argsort(masterSII_cat['ErrAmp'])
    masterSII_cat = masterSII_cat[errAsort]
    #Save the completed master SII catalog
    ind_col = col(np.arange(np.alen(masterSII_cat)), name="Index")
    masterSII_cat.add_column(ind_col, index=0)
    ascii.write(masterSII_cat, cat_name)
    absdf=123

def catalog_interaction(master_SII_cat):

    cls()
    truncated_print = True
    if "PerFitErr" in master_SII_cat.columns:
        truncvals = ["Index", "NAME", "RA", "DEC", "ANGD", "BS_BMAG", "BSSpT", "PerFitErr", "PerFailFit"]
    else:
        truncvals = ["Index", "NAME", "RA", "DEC", "ANGD", "BS_BMAG", "BSSpT"]


    master_SII_cat[truncvals].pprint(max_lines=-1,max_width=-1)
    stop_sel = 'n'



    while True:
        mode = input("\nEnter a star's index value to do a single analysis\n"
                     "Enter 'rankall' to do a full catalog ranking\n"
                     "Enter 'toggleinfo' to show/hide all available catalog information\n"
                     "Enter 'q' to quit\n"
                     ": ")

        if mode.lower() == 'rankall':
            print("Ranking all stars")
            break
        elif mode.lower() == "q":
            print("quitting ASIIP")
            sys.exit()

        elif mode.lower() == "toggleinfo":
            truncated_print = not truncated_print
            if truncated_print:
                print("\nTruncated printing is now activated\n")
                master_SII_cat["Index", "NAME", "RA", "DEC", "ANGD", "BS_BMAG", "BSSpT"].pprint(max_lines=-1, max_width=-1)
            else:
                print("\nTruncated printing has been deactivated\n")
                master_SII_cat.pprint(max_lines=-1, max_width=-1)


        else:
            try:
                selection = int(mode)
                star = master_SII_cat[selection]
                name, ra, dec, star_mag, ang_diam, guess_diam, star_id = star_info(star, wavelength)
                tel_array.star_track(ra=ra,
                                     dec=dec,
                                     alt_cut=alt_cut,
                                     obs_start=obs_start,
                                     obs_end=obs_end,
                                     Itime=int_time)
                I_time = tel_array.star_dict[star_id]["IntDelt"]

                if I_time:
                    star_err, hour_angle_rad, dec_angle_rad, lat, tel_tracks, airy_disk, airy_func = \
                        star_model(tel_array=tel_array, I_time=I_time, star_mag=star_mag, ang_diam=ang_diam,
                                   wavelength=wavelength, star_id=star_id)
                    # This is where the custom Monte Carlo analysis is performed. If you wish to add an analytical function, you
                    # can use this function as a template to create another analysis technique
                    runs = boot_runs * 2
                    fdiams, ferrs, ffit = \
                        IItools.IIbootstrap_analysis_airyDisk(tel_tracks=tel_tracks,
                                                              airy_func=airy_func,
                                                              star_err=star_err,
                                                              guess_diam=guess_diam,
                                                              wavelength=wavelength,
                                                              runs=runs)
                    fit_err = np.nanstd(fdiams) / np.nanmedian(fdiams) * 100
                    failed_fit = ffit / runs * 100

                    cls()

                    if truncated_print: master_SII_cat[truncvals].pprint(max_lines=-1,max_width=-1)
                    else: master_SII_cat.pprint(max_lines=-1, max_width=-1)


                    print(red_line)
                    if truncated_print: print(star[truncvals])
                    else: print(star)
                    print("\x1b[1;32;40m' The fit error is %.3f%%, the failed fit is %.3f%% \x1b[0m\n" % (
                        fit_err, failed_fit))
                    print(red_line)

                    if save_plots: star_save = os.path.join(save_dir, name)
                    else: star_save = False

                    guess_r = airy_func.radius.value

                    do_plots(tel_array=tel_array,
                             baselines=baselines,
                             tel_tracks=tel_tracks,
                             ang_diam=star["ANGD"],
                             airy_func=airy_func,
                             star_err=star_err,
                             guess_r=guess_r,
                             wavelength=wavelength,
                             name=name,
                             I_time=I_time,
                             pererr=fit_err,
                             star_save=star_save)


                else:
                    print("\nSorry, that star isn't visible for the times you defined.\n")

            except Exception as e:
                print(e)
                print("\nI'm sorry, but that response doesn't actually mean anything to ASIIP.\n")

    return mode.lower()


if __name__ == "__main__":
    cls()
    print("Welcome to ASIIP (A Stellar Intensity Interferometry Planner). Please make sure you are running the catalog for the desire night.\n")

    try:
        if np.alen(sys.argv) > 1: param_file_name = sys.argv[1]
        else: param_file_name = "ExampleSIIparameters.json"

        #Read in all parameters from the parameter file to make sure everything will run correctly
        with open(param_file_name) as param:
            #I use json as it is clear from the file itself what the variables are.
            IIparam = json.load(param)
            #The time is meant to be midnight on the day that you are observing. Make sure you correct for your timezone and daylight savings
            time = IIparam["time"]
            observatory_name = IIparam["obsName"]

            hour_ra = IIparam["raRange"]
            ra_range = Angle(IIparam["raRange"], unit='hourangle').to('deg').value.tolist()
            dec_range = IIparam["decRange"]
            mag_range = IIparam["magRange"]
            #specifies the lowest altitude your observatory can observe
            alt_cut = IIparam["altitudeCutoff"]
            #specifies that maximum altitude the sun is allowed to be at to determine when observing can start
            max_sun_alt = IIparam["maxSunAltitude"]



            #The wavelength of your filter in meters
            wavelength = IIparam["wavelength"]

            #locations of all the different VERITAS telescopes relative to each other, used to construct the baselines
            relative_tel_locs = np.array(IIparam["telLocs"])


            #THESE SHOULD BE DEFINED IN HOURS AND ONLY FRACTIONS OF HOURS, 0 is defined as midnight
            int_time = IIparam["integrationTime"] * u.h

            if ~np.isnan(np.array(IIparam["observationStart"]).astype(float)): obs_start = IIparam["observationStart"]*u.h
            else: obs_start = IIparam["observationStart"]

            if ~np.isnan(np.array(IIparam["observationEnd"]).astype(float)): obs_end = IIparam["observationEnd"]*u.h
            else: obs_end = IIparam["observationEnd"]

            # steps details how fine you want the baseline graphs to be, fewer steps means faster processing but less detail in the models
            steps = 800


            #The latitude, longitude, and elevation of the center point of your array
            observatory_lat = IIparam["telLat"]
            observatory_lon = IIparam["telLon"]
            observatory_elv = IIparam["telElv"]
            #How many bootstrapping runs you want to do per target.
            boot_runs = IIparam["bootStrapRuns"]

            #The pre-determined values to calculate your error
            sigma_tel = IIparam["sigmaTel"]
            sigma_mag = IIparam["sigmaMag"]
            sigma_time = IIparam["sigmaTime"]

            save_plots = IIparam["savePlots"]

            #this is the name of the output save file
            cat_name = "%sCatalog%smag%.3fto%.3f_obsstart_%s_obsend_%sRA%.4fTo%.4f.siicat"%\
                       (observatory_name,time, mag_range[0], mag_range[1], obs_start, obs_end, hour_ra[0], hour_ra[1])
            curdir = os.path.dirname(__file__)
            #this is the directory where all analysis graphs will be saved.
            save_dir = os.path.join(curdir, "IIGraphs")



    except Exception as e:
        print(e)
        print("So the parameters could not be read in correctly. Make sure you have formatted your parameter file correctly, "
              "that you have named it correctly and it's located in the same directory as the script. "
              "Default parameter file name is IIparameters.json")
        raise e


    tel_array = IIdata.IItelescope(telLat=observatory_lat,
                                   telLon=observatory_lon,
                                   telElv=observatory_elv,
                                   time=time,
                                   steps=steps,
                                   sig1=sigma_tel,
                                   m1=sigma_mag,
                                   t1=sigma_time,
                                   mag_range=mag_range,
                                   dec_range=dec_range,
                                   ra_range=ra_range,
                                   max_sun_alt=max_sun_alt,
                                   timestep=int_time.value)

    baselines = IItools.array_baselines(relative_tel_locs)
    [tel_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]


    print("Now running analysis.")
    cats = [ca for ca in os.listdir() if ".siicat" in ca]

    if np.alen(cats) > 0:

        print(red_line)
        for i, cat in enumerate(cats):
            print("%s: %s" % (i, cat))
        print(red_line)

        cat_choice = input("\nCatalogs have been found "
                           "Enter a catalogs index number printed above and to the catalogs left to load one\n"
                           "Enter anything else to create a new catalog\n"
                           ": ")
        try:
            cat_choice = int(cat_choice)
            master_SII_cat = ascii.read(cats[cat_choice])

        except:
            print("That input wasn't a number, creating a new catalog.")
            catalog_builder(tel_array=tel_array, cat_name=cat_name)
            master_SII_cat = ascii.read(cat_name)
    else:
        print("\nNo catalog %s found, creating a new catalog.\n"%(cat_name))
        catalog_builder(tel_array=tel_array, cat_name=cat_name)
        master_SII_cat = ascii.read(cat_name)





    catalog_interaction(master_SII_cat)



    fit_diams = []
    fit_errs = []
    failed_fits = []
    guess_diams = []
    star_tracks = []
    star_funcs = []
    star_errs = []
    s = timer.time()
    #Start the full ranking
    for star in master_SII_cat:
        name, ra, dec, star_mag, ang_diam, guess_diam, star_id = star_info(star, wavelength)

        tel_array.star_track(ra=ra,
                             dec=dec,
                             alt_cut=alt_cut,
                             obs_start=obs_start,
                             obs_end=obs_end,
                             Itime=int_time)

        I_time = tel_array.star_dict[star_id]["IntDelt"]

        # if "bet CMa" in name:
        #     adsf = 123
        #
        #     tel_array.star_track(ra=ra,
        #                          dec=dec,
        #                          alt_cut=alt_cut,
        #                          obs_start=obs_start,
        #                          obs_end=obs_end,
        #                          Itime=int_time)


        # if there are any valid points, begin the Monte Carlo Analysis
        if I_time:


            star_err, hour_angle_rad, dec_angle_rad, lat, tel_tracks, airy_disk, airy_func = \
                star_model(tel_array=tel_array, I_time=I_time, star_mag=star_mag, ang_diam=ang_diam,
                           wavelength=wavelength, star_id=star_id)
            #This is where the custom Monte Carlo analysis is performed. If you wish to add an analytical function, you
            #can use this function as a template to create another analysis technique
            fdiams, ferrs, ffit = \
                IItools.IIbootstrap_analysis_airyDisk(tel_tracks=tel_tracks,
                                                      airy_func=airy_func,
                                                      star_err=star_err,
                                                      guess_diam=guess_diam,
                                                      wavelength=wavelength,
                                                      runs=boot_runs)
            guess_diam = wavelength / ang_diam

            fit_diams.append(wavelength/fdiams * u.rad.to('mas'))
            fit_errs.append(ferrs)
            failed_fits.append(ffit/boot_runs*100)
            guess_diams.append(star["ANGD"])
            star_tracks.append(tel_tracks)
            star_funcs.append(airy_func)
            star_errs.append(star_err)

            print("Completed %s" % (name))
            abs=2323
        else:

            #becasue of indexing and performance concerncs with Astropy Tables, the array are constructed and then
            #appended to the final table
            print("For your selected times %s to %s, the star %s cannot be observed" % (obs_start, obs_end, star["NAME"]))
            fit_diams.append(np.full(boot_runs, np.nan))
            fit_errs.append(np.full(boot_runs, np.nan))
            failed_fits.append(boot_runs)
            guess_diams.append(guess_diam)

            star_tracks.append(np.nan)
            star_funcs.append(np.nan)
            star_errs.append(np.nan)


    e = timer.time()
    total_time = e-s
    fit_diams = np.array(fit_diams)
    fit_errs = np.array(fit_errs)
    failed_fits = np.array(failed_fits)
    guess_diams = np.array(guess_diams)

    medianFits = np.nanmedian(fit_diams, axis=1)
    medianerr = np.nanmedian(fit_errs, axis=1)
    true_err = (1-medianFits/guess_diams)*100
    stdFits = np.nanstd(fit_diams, axis=1)
    # stdFits = np.nansum((fit_diams-guess_diams[:,None])**2,axis=1)/np.count_nonzero(~np.isnan(fit_diams), axis=1)
    diam_percent_err = medianerr
    err_of_diam_err = stdFits/medianFits*100

    if "MeanDiamFit" not in master_SII_cat.colnames:
        master_SII_cat.add_column(col(medianFits, name="MeanDiamFit", format="%.2f"))
        master_SII_cat.add_column(col(diam_percent_err, name="PerDiamErr", format="%.2f"))
        master_SII_cat.add_column(col(err_of_diam_err, name="PerFitErr", format="%.2f"))
        master_SII_cat.add_column(col(failed_fits, name="PerFailFit", format="%.2f"))

    lowerr = np.argsort(master_SII_cat, order=["PerFailFit", "PerFitErr"])
    rasort = np.argsort(master_SII_cat, order=["RA"])


    # asdqwer=1239

    # max_radius = 5*u.arcsec

    # cep_res, cep_mat, cep_dis = tel_array.cephied_finder(ras=master_SII_cat["RA"],
    #                                              decs=master_SII_cat["DEC"],
    #                                                     radius=max_radius)
    #
    # #
    # # varInfo = np.full(np.alen(lowerr), "---------")
    # # varInfo[cep_mat] = cep_res["VarType"]
    # # gcvsNames = np.full(np.alen(lowerr),"---------")
    # # gcvsNames[cep_mat] = cep_res["GCVS"]
    # # matchDis = np.full(np.alen(lowerr), "---------")
    # # matchDis[cep_mat] = cep_dis.to("mas")
    # # # gcvsSpTy = np.full(np.alen(lowerr), "---------")
    # # # gcvsSpTy[cep_mat] = cep_res["SpType"]
    # #
    # # master_SII_cat.add_column(col(varInfo), name="VariableInfo")
    # # master_SII_cat.add_column(col(matchDis, name="VariableDis"))
    # # master_SII_cat.add_column(col(gcvsNames, name="GCVSName"))
    # # master_SII_cat.add_column(col(gcvsSpTy, name="GCVSSpType"))
    #
    # master_SII_cat[cep_mat]["VariableDis"] = cep_dis.to("mas")
    #
    # stable_targs = np.where((master_SII_cat[lowerr]["PerFitErr"] < 20) & (master_SII_cat[lowerr]["MAG"] < 4))


    ind_col = col(np.arange(np.alen(master_SII_cat)), name="Index")

    master_SII_cat = master_SII_cat[lowerr]

    master_SII_cat["Index"] = ind_col




    response = catalog_interaction(master_SII_cat)

    if response == "rankall":
        print("\nYou just ranked all these stars silly! Restart ASIIP if you want to do the ranking all over again.")

    master_SII_cat["Index"] = ind_col


    ascii.write(master_SII_cat, "Ranked_%s"%(cat_name), overwrite=True)


# xvals=[]
# yvals=[]
# ramps=[]
# xn= "ErrAmp"
# yn="PerFitErr"
# for star in master_SII_cat[lowerr]:
#         n = star["Index"]
#         trac = star_tracks[n]
#         func = star_funcs[n]
#         xvals.append(star[xn])
#         yvals.append(star[yn])
#         ramps.append(r_amp)
# # plt.xlabel(xn)
# plt.ylabel("Signal to Noise Ratio")
# plt.xlabel("Star Rank")
# plt.plot(np.array(ramps)/np.array(yvals),'o')

# xvals=[]
# yvals=[]
# midx=[]
# midy=[]
# badx=[]
# bady=[]
# ramps=[]
# xn= "ANGD"
# yn="BS_BMAG"
# lowsky = np.where(master_SII_cat[lowerr]["BSSkyD"]<3000)
# for star in master_SII_cat[lowerr]:
#     if star["PerFitErr"]:
#         n = star["Index"]
#         trac = star_tracks[n]
#         func = star_funcs[n]
#         if star["PerFitErr"]>30:
#             badx.append(star[xn])
#             bady.append(star[yn])
#         elif star["PerFitErr"]>20 and star["PerFitErr"]<30:
#             midx.append(star[xn])
#             midy.append(star[yn])
#         elif star["PerFitErr"]<20:
#             xvals.append(star[xn])
#             yvals.append(star[yn])
# # plt.xlabel(xn)
# plt.title("B-Magnitude Vs Angular Diameter Monte-Carlo Results")
# plt.ylabel("Bright Star Catalogue B magnitude")
# plt.xlabel("%s(mas)"%(xn))
# plt.semilogx(xvals,yvals,'o',label="PerFitErr < 20%",color="b",marker='o')
# plt.semilogx(midx,midy,'o',label="PerFitErr > 20% & < 30% ",color="y",marker="^")
# plt.semilogx(badx,bady,'o',label="PerFitErr > 30%",color="r",marker="x")
# plt.legend()


# xvals=[]
# yvals=[]
# midx=[]
# midy=[]
# badx=[]
# bady=[]
# ramps=[]
# xn= "ANGD"
# yn="MAG"
# for i in range(80):
#     n=3
#     star = master_SII_cat[lowerr][n]
#     func = np.array(star_funcs)[lowerr][n]
#     trac = np.array(star_tracks)[lowerr][n]
#
#     rads, amps, avg_rad, avg_amp  = IImodels.airy2dTo1d(trac,func)
#
#     err = star["ErrAmp"]
#     r0 = func.radius.value
#     modhi = IImodels.airy1D(rads.ravel(),r0+r0*i*.01)
#     modlo = IImodels.airy1D(rads.ravel(),r0-r0*i*.01)
#     xvals.append(r0+r0*i*.01)
#     xvals.append(r0-r0*i*.01)
#
#     chihi = np.sum((modhi-err)**2/err**2)
#     chilow = np.sum((modlo-err)**2/err**2)
#     yvals.append(chihi)
#     yvals.append(chilow)
#
# plt.plot(xvals,yvals,'o')
# plt.title(r0)
# plt.show()
# plt.xlabel(xn)
# plt.ylabel(yn)
# plt.xlabel(xn)
#
# plt.legend()
asfadsf=1234123
