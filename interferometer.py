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
norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())


def deep_star_analysis(star, star_id, boot_runs = 100, I_time = 1800*u.s):
    ang_diam = star["ANGD"] * u.mas
    star_name = star["SIMID"]
    star_mag = star["SimBMAG"]
    guess_r = wavelength.to('m').value / ang_diam.to('rad').value

    star_err = IItools.track_error(sig0=veritas_array.err_sig,
                                   m0=veritas_array.err_mag,
                                   m=star_mag,
                                   t0=veritas_array.err_t1,
                                   t=I_time.value)

    tel_tracks, airy_disk, airy_func = star_airy_track(tel_array=veritas_array, ang_diam=ang_diam,
                                                       wavelength=wavelength, star_id=star_id)

    fitdiams, fiterrs, failed_fits = IItools.IIbootstrap_analysis(tel_tracks=tel_tracks,
                                                                  airy_func=airy_func,
                                                                  star_err=star_err,
                                                                  guess_r=guess_r,
                                                                  wavelength=wavelength,
                                                                  runs=boot_runs)

    fdiam_mean = np.mean(fitdiams)
    ferr_mean = np.mean(fiterrs)
    ferr_std = np.std(fiterrs)


    # print("%s: mean mas of 1000 fits is %s +- %.4f percent.\tfiterr std is %.4f +- %.4f percent with %s failed fits"
    #       % (star_name, fdiam_mean, ferr_mean / fdiam_mean * 100, ferr_std,
    #          ferr_std / ferr_mean * 100, failed_fits))
    print("Completed %s"%(str.strip(star_name, '* ')))
    reg_analysis = [tel_tracks, airy_func, star_err]
    boot_analysis = [fdiam_mean, ferr_mean, ferr_std, failed_fits]
    return boot_analysis, reg_analysis

def star_airy_track(tel_array, ang_diam, wavelength, star_id):
    hour_angle_rad = Angle(tel_array.star_dict[star_id]["IntTimes"]).to('rad').value
    dec_angle_rad = tel_array.star_dict[star_id]["DEC"].to('rad').value
    tel_tracks = [IItools.uv_tracks(lat=tel_array.telLat.to('rad').value,
                                    dec=dec_angle_rad,
                                    hours=hour_angle_rad,
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

def star_rater(tel_array, xlen, ylen, wavelength, cutoff_obs_time = 0, obs_t=None):
    tablenames = ("RA","DEC","ANGD","FILT","MAG","CAT","TotObsTime")
    dtype = (np.float, np.float, np.float, np.unicode, np.float, np.unicode, np.float)
    dtable = Table(names=tablenames, dtype=dtype)
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
    coords_done = SkyCoord([(0*u.deg,0*u.deg)])
    i = 0
    for catalog, cat_name in zip(tel_array.catalogs, tel_array.cat_names):
        ras, decs, ang_diams, mags, mag_name = tel_array.ra_dec_diam_getter(cat_name, catalog)
        pos_cat=SkyCoord(ras,decs)

        #match pos_cat to the closest star in coords_done returning the closest coords_done indicies
        closest_star, skydis, distance3d = pos_cat.match_to_catalog_sky(coords_done)

        #the indicies where the matched pos_cat star is large enough to be considered a unique star
        clo_idx = skydis > 1 * u.arcsec

        #the indicies of pos_cat which are under 1 arcsec and are considered a duplicate
        dup_idx = np.where(~clo_idx)

        #the indicies of the full catalog that has the duplicate value
        dup_star = closest_star[dup_idx]

        coords_done = SkyCoord([c.data for c in coords_done] + [c.data for c in pos_cat[clo_idx]])



        for ra,dec,ang_diam, mag in zip(ras[clo_idx], decs[clo_idx], ang_diams[clo_idx], mags[clo_idx]):
            total_obs_time = 0 * u.hour

            if ra.value not in calc_ra and dec.value not in calc_dec:
                calc_ra.append(ra.value)
                calc_dec.append(dec.value)
                print("\nAnalyzing Star RA %s DEC %s at %s" % (ra, dec, i))
            else:
                print('|', end="")
                c=c+1

                continue
            star_id = str(ra) + str(dec)
            tel_array.star_track(ra=ra,
                                 dec=dec,
                                 alt_cut=alt_cut)

            if tel_array.star_dict[star_id]["ObsTimes"]:
                if np.alen(tel_array.star_dict[star_id]["ObsTimes"]) > 1:
                    total_obs_time = np.ptp(tel_array.star_dict[star_id]["ObsTimes"])
                else: total_obs_time=0*u.hour

            if total_obs_time <= cutoff_obs_time * u.hour:
                print("Skipping star RA %s DEC %s as it is observable only %s" % (ra, dec, total_obs_time))
                continue

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
            dup_count.append([ang_diam.value])

        if np.alen(dup_star) > 0:
            try:
                for mcati, ccati in zip(dup_star, dup_idx[0]):
                    dup_count[mcati].append(ang_diams[ccati].value)
            except Exception as e:
                print(e,c)

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
    amp_errs = IItools.track_error(sig0=tel_array.err_sig,
                                   m0=tel_array.err_mag,
                                   m=bmag,
                                   t0=tel_array.err_t1,
                                   t=obs_t)
    dtable.pprint(max_lines=-1, max_width=-1)
    data_table = Table([sim_id.astype("U13"), col(good_ra, unit=u.hourangle), col(good_dec, unit=u.deg), col(good_diams, unit=u.mas),
                         good_mag_names, good_mags, cat_names, bmag,vmag,
                        bs_dis.to("mas"), bs_info["SpType"], bs_info["RotVel"], sim_bflux, sim_sptype, simd.to('mas'), sim_rotV,
                        col(amp_errs),col(total_obs_times, unit=u.second), col(obs_t, unit=u.second)],
                       names=("SIMID","RA","DEC","ANGD",
                              "FILT","MAG","CAT","BS_BMAG","BS_VMAG",
                              "BSSkyD", "BSSpT", "BSRV","SimBMAG", "SIMSpT", "SIMSkyD", "SIMRV",
                              "ErrAmp", "TotObsTime", "ObsTime"))
    return data_table

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



    veritas_array.make_cadars_query(mag_range=mag_range,
                                    ra_range=ra_range,
                                    dec_range=dec_range)



    veritas_array.make_charm2_query(mag_range=mag_range,
                                    ra_range=ra_range,
                                    dec_range=dec_range)


    veritas_array.bright_star_cat(ra_range=ra_range,
                                  dec_range=dec_range)



    veritas_cat = star_rater(tel_array=veritas_array,
                                              xlen=xlen,
                                              ylen=ylen,
                                              wavelength=wavelength,
                                                 obs_t=1800)

    print()
    for col in veritas_cat.colnames:
        if veritas_cat[col].dtype in [np.float32, np.float64]:
            veritas_cat[col].format = "%6.4f"
    np.set_printoptions(threshold=np.inf)
    ascii.write(veritas_cat, cat_name)
    absdf=123


if __name__ == "__main__":
    print("Welcome to the VERITAS catalog builder. Please make sure you are running the catalog for the desire night.\n")
    try:
        if np.alen(sys.argv) > 1: param_file_name = sys.argv[1]
        else: param_file_name = "IIparameters.json"
        with open(param_file_name) as param:
            #I use json as it is clear from the file itself what the variables are.
            IIparam = json.load(param)
            #The time is meant to be midnight on the day that you are observing. Make sure you correct for your timezone and daylight savings
            time = IIparam["time"]
            ra_range = IIparam["raRange"]
            dec_range = IIparam["decRange"]
            mag_range = IIparam["magRange"]
            #specifies the lowest altitude your observatory can observe
            alt_cut = IIparam["altitudeCutoff"]

            #this is the name of the output save file
            cat_name = "veritasCatalog%smag%sto%s.dat"%(time, mag_range[0], mag_range[1])
            curdir = os.path.dirname(__file__)
            #this is the directory where all analysis graphs will be saved.
            save_dir = os.path.join(curdir, "IIGraphs")

            #The wavelength of your filter
            wavelength = IIparam["wavelength"] * u.m

            #locations of all the different VERITAS telescopes relative to each other, used to construct the baselines
            veritas_baselines = np.array(IIparam["telLocs"])


            #THESE SHOULD BE DEFINED IN HOURS AND ONLY FRACTIONS OF HOURS, 0 is defined as midnight
            int_time = IIparam["integrationTime"] * u.h

            if ~np.isnan(np.array(IIparam["observationStart"]).astype(float)): obs_start = IIparam["observationStart"]*u.h
            else: obs_start = IIparam["observationStart"]

            if ~np.isnan(np.array(IIparam["observationEnd"]).astype(float)): obs_end = IIparam["observationEnd"]*u.h
            else: obs_end = IIparam["observationEnd"]

            # steps details how fine you want the baseline graphs to be, fewer steps means faster processing but less detail in the models
            steps = 600


            #The latitude, longitude, and elevation of the center point of your array
            veritas_telLat = IIparam["telLat"]
            veritas_telLon = IIparam["telLon"]
            veritas_telElv = IIparam["telElv"]
            #How many bootstrapping runs you want to do per target.
            boot_runs = IIparam["bootStrapRuns"]

            #The pre-determined values to calculate your error
            sigma_tel = IIparam["sigmaTel"]
            sigma_mag = IIparam["sigmaMag"]
            sigma_time = IIparam["sigmaTime"]



    except Exception as e:
        print(e)
        print("So the parameters could not be read in correctly. Make sure you have formatted your parameter file correctly, "
              "that you have named it correctly and it's located in the same directory as the script. "
              "Default parameter file name is IIparameters.json")
        raise e


    veritas_array = IIdata.IItelescope(telLat=veritas_telLat,
                                       telLon=veritas_telLon,
                                       telElv=veritas_telElv,
                                       time=time,
                                       steps=steps,
                                       sig1=sigma_tel,
                                       m1=sigma_mag,
                                       t1=sigma_time,
                                       mag_range=mag_range,
                                       dec_range=dec_range,
                                       ra_range=ra_range)

    baselines = IItools.array_baselines(veritas_baselines)
    [veritas_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]


    print("Now running analysis.")
    if cat_name in os.listdir():
        print("Catalog %s has been found and will be used"%(cat_name))
        veritas_cat = ascii.read(cat_name)
    else:
        print("No catalog %s found, creating a new catalog."%(cat_name))
        catalog_builder(veritas_array=veritas_array,
                        wavelength=wavelength,
                        cat_name=cat_name)
        veritas_cat = ascii.read(cat_name)

    if "Index" not in veritas_cat.colnames:
        ind_col = col(np.arange(np.alen(veritas_cat)), name="Index")
        veritas_cat.add_column(ind_col, index=0)
    veritas_cat.pprint(max_lines=-1, max_width=-1)
    stop_sel = 'n'

    deep_analysis = []
    reg_analysis = []
    for selected_star in veritas_cat:
        ra = selected_star["RA"] * u.hourangle
        dec = selected_star["DEC"] * u.deg
        if 'eps Vir' in selected_star["SIMID"]:
            asdf=234
        veritas_array.star_track(ra=ra,
                                 dec=dec,
                                 alt_cut=alt_cut,
                                 obs_start=obs_start,
                                 obs_end=obs_end,
                                 Itime=int_time)
        star_id = str(ra) + str(dec)

        I_time = veritas_array.star_dict[star_id]["IntDelt"]


        if I_time:
            deep_anal, reg_anal = deep_star_analysis(star=selected_star, star_id=star_id, boot_runs=boot_runs, I_time=I_time.to('s'))
            deep_analysis.append(deep_anal)
            reg_analysis.append(reg_anal)
            abs=2323
        else:
            print("For your selected times %s to %s, the star %s cannot be observed"%(obs_start, obs_end, selected_star["SIMID"]))
            deep_analysis.append([np.nan, np.nan, np.nan, np.nan])
            reg_analysis.append([np.nan,np.nan,np.nan])


    deep_analysis = np.array(deep_analysis)
    reg_analysis = np.array(reg_analysis)
    diam_percent_err = deep_analysis[:,1]/deep_analysis[:,0]*100
    err_of_diam_err = deep_analysis[:,2]/deep_analysis[:,1]*100

    if "MeanDiamFit" not in veritas_cat.colnames:
        veritas_cat.add_column(col(deep_analysis[:,0], name="MeanDiamFit", format="%.2f"))
        veritas_cat.add_column(col(diam_percent_err, name="PerDiamErr", format="%.2f"))
        veritas_cat.add_column(col(err_of_diam_err, name="PerErrDiamFitErr", format="%.2f"))
        veritas_cat.add_column(col(deep_analysis[:,3] / boot_runs * 100, name="PerFailFit", format="%.2f"))

    lowerr = np.argsort(veritas_cat, order=["PerFailFit","PerDiamErr"])
    rasort = np.argsort(veritas_cat, order=["RA"])
    veritas_cat[lowerr].pprint(max_lines=-1, max_width=-1)


    fit_failures = deep_analysis[:,3]
    not_complete_failures = np.where(np.array(fit_failures) < boot_runs / 4)


    asdqwer=1239

    response = input("\nEnter an index you wish to make graphs of or enter 'q' to quit\n")
    while(response!='q'):
        try:
            n = int(response)
            guess_r = (wavelength.to('m').value / (veritas_cat[n]["ANGD"]*u.mas).to('rad').value)
            star_tracks = reg_analysis[n][0]
            star_func = reg_analysis[n][1]
            star_err = reg_analysis[n][2]
            star_name = str.strip(veritas_cat[n]["SIMID"], "* ")
            star_save = os.path.join(save_dir, star_name)
            IIdisplay.uvtrack_model_run(tel_tracks=star_tracks,
                                        airy_func=star_func,
                                        star_err=star_err,
                                        guess_r=guess_r,
                                        wavelength= wavelength,
                                        star_name=star_name,
                                        ITime=I_time,
                                        save_dir=star_save,
                                        fullAiry=False)
            IIdisplay.uvtrack_model_run(tel_tracks=star_tracks,
                                        airy_func=star_func,
                                        star_err=star_err,
                                        guess_r=guess_r,
                                        wavelength= wavelength,
                                        star_name=star_name,
                                        ITime=I_time,
                                        save_dir=star_save,
                                        fullAiry=True)
            IIdisplay.uvtracks_airydisk2D(tel_tracks=star_tracks,
                                          veritas_tels=veritas_array,
                                          baselines=baselines,
                                          airy_func=star_func,
                                          guess_r=guess_r,
                                          wavelength=wavelength,
                                          save_dir=star_save,
                                          star_name=star_name)
        except Exception as e:
            print(e)
            print("Make sure to enter a valid response\n")
        response = input("enter an index you wish to make graphs of or enter 'q' to quit\n")





asfadsf=1234123
