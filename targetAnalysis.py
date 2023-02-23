import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from scipy.signal import savgol_filter
import sys, os
sys.path.append("ASIIP")
from II import IImodels, IIdisplay, IItools, IIdata
from asiip import star_model
import astropy.units as u
import json
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astroquery.simbad import Simbad
import webbrowser
import matplotlib as mpl
import multiprocessing
import traceback
if os.name=='nt': mpl.use('Qt5Agg')
# else: mpl.use('TkAgg')


Simbad.add_votable_fields('flux(B)', 'flux(G)', 'flux(V)', 'sptype', 'rot', "v*", "velocity", "distance",
                          "diameter", "pm",
                          "morphtype")

sep_line = "-----------------------------------------------------------------------------------------------------------"
red_line = '\n\x1b[1;31;40m' + sep_line + '\x1b[0m\n'
same_input_output_warning = "!!!!The input and output directory are the same. This is likely a big mistake so the script will abort!!!!"
red_sameio_warning = '\n\x1b[1;31;40m' + same_input_output_warning + '\x1b[0m\n'
targetmessage = "This is a printed list of the targets in the given directory\n" \
                "!!!!!!!!!!ONLY TARGET DIRECTORIES SHOULD BE LISTED HERE!!!!!!!!!!!!!!!!!"
graphsFoldername = "analysisGraphs"
reduceDataDir = "reducedTargets"

mes_delays = (np.array([677.7, 663.4662619, 599.93064474, 957.49610466, 1060.5799357]) +
              np.array([677.7, 663.20958087, 600.5387132, 960.00093392, 1057.31916855]) +
              np.array([677.7, 664.93901958, 600.1011103, 954.39544037, 1062.15499002])) / 3
pair_delays = []
Ts = []
n = 5
for ii in range(n):
    for j in range(1, n - ii):
        pair_delays.append((mes_delays[j + ii] - mes_delays[ii]) / 4)
        Ts.append("T%sT%s" % (ii + 0, ii + j + 0))

preamp_gain=20.e3 #V/A
adc_gain_14bit=2**14/1.23 #dc/V
adc_gain_8bit=2**8/1.23 #dc/V
amp_per_adc_14bit=1./(preamp_gain*adc_gain_14bit) #amp/dc
amp_per_adc_8bit=1./(preamp_gain*adc_gain_8bit) #amp

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def array_spliner(x_old, y_old, x_new):
    """
    takes a function with x and y data and pieceiwse interpolates y to the values in x_new
    :param x_old: original x data
    :param y_old: original y data
    :param x_new: the x values you want to fit new y values too
    :return: the interpolated
    """
    spliner = CubicSpline(x_old, y_old)
    splined_array = spliner(x_new)
    return splined_array

def star_simulator(star_ha = 5.4382 * u.hourangle, star_dec = 28.6075 * u.deg, teldir="IIparameters2.json",
                   ang_diam = (1.2 * u.mas).to('rad').value, st=None, et=None, date="2020-12-26 07:00:00", starmag=1,
                   starerr=1, equinox="J2022"):
    with open(teldir) as jsondat:
    # jsondat = open(teldir)
        IIparam = json.load(jsondat)

    steps = 4000

    tel_array = IIdata.IItelescope(telLat=IIparam['telLat'],
                                   telLon=IIparam['telLon'],
                                   telElv=IIparam['telElv'],
                                   time=date,
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
    baselines, tel_names = IItools.array_baselines(relative_tel_locs)
    [tel_array.add_baseline(Bew=base[0], Bns=base[1], Bud=base[2]) for base in baselines]

    if st == None:
        tel_array.star_track(star_ha, star_dec,
                             obs_start=None,
                             obs_end=None,
                             sunangle=IIparam["maxSunAltitude"],
                             equinox=equinox)
    else:
        tel_array.star_track(star_ha, star_dec,
                             obs_start=st*u.hour,
                             obs_end=et*u.hour,
                             sunangle=IIparam["maxSunAltitude"],
                             equinox=equinox)
    ra_dec_str = str(star_ha) + str(star_dec)
    I_time = tel_array.star_dict[ra_dec_str]["IntDelt"]
    star_err, hour_angle_rad, dec_angle_rad, lat, tel_tracks, airy_disk, airy_func = star_model(tel_array,
                                                                                                I_time,
                                                                                                starmag,
                                                                                                ang_diam,
                                                                                                IIparam['wavelength'],
                                                                                                ra_dec_str)

    rads, amps, avgrad, avgamp = IImodels.visibility2dTo1d(tel_tracks=tel_tracks, visibility_func=airy_func,
                                                           x_0=airy_func.x_0.value, y_0=airy_func.y_0.value)

    utrack = [tracku[0][:, 0] for tracku in tel_tracks]

    vtrack = [trackv[0][:, 1] for trackv in tel_tracks]


    skytimes_ha = tel_array.star_dict[ra_dec_str]["IntTimes"].value

    my_odp_corr = np.array([(tel_tracks[i][0][:, 2]) / 299792458 for i in range(len(tel_tracks))]) / 4e-9
    my_odp_corr1 = np.array([(tel_tracks[i][1][:, 2]) / 299792458 for i in range(len(tel_tracks))]) / 4e-9
    return rads, amps, skytimes_ha, tel_array, ra_dec_str, [my_odp_corr,my_odp_corr1], utrack, vtrack, tel_array


def open_veritas_sii(fuldir, timesum=60, datcut=3):

    try:
        datraw = np.loadtxt(fuldir, dtype=float, skiprows=3, usecols=range(2, 132))
        rawvolt1 = datraw[:, 0]
        rawvolt2 = datraw[:, 1]
        voltcomb = datraw[:, 3:-1].mean(axis=1)
        datVnorm = datraw[:, 2:] / voltcomb[:,None]

    except:
        datraw = np.loadtxt(fuldir, dtype=float, skiprows=3, usecols=range(2, 70))
        rawvolt1 = datraw[:, 0]
        rawvolt2 = datraw[:, 1]
        voltcomb = datraw[:, 3:-1].mean(axis=1)
        datVnorm = datraw[:, 4:] / voltcomb[:,None]

    time = np.loadtxt(fuldir, dtype=str, skiprows=3, usecols=[0, 1])

    try:
        unixtime = datetime.datetime.strptime(time[0][0] + " " + time[0][1], '%Y-%m-%d %H:%M:%S')
        rowdt = 1
        midnight_date = str(unixtime.date()) + " 07:00:00"
        rowcon = 1
    except:
        unixtime = datetime.datetime.strptime(time[0][0] + " " + time[0][1], '%Y-%m-%d %H:%M:%S.%f')
        rowdt = 0.125
        midnight_date = str(unixtime.date()) + " 07:00:00"
        rowcon = 1

    ht = (unixtime.hour + unixtime.minute / 60 + unixtime.second / 60 ** 2)
    if ht > 12:
        hourcon = ht - 24
    else:
        hourcon = ht

    if datraw.shape[1] > 80:
        chunk = timesum * 8
        g2conscorr = 1e6#.5e-16
    else:
        chunk = timesum
        g2conscorr = -1e-12

    if datraw.shape[0] < chunk:
        raise Exception("The file %s you are trying to read is too small"%(fuldir))

    if voltcomb[0] == 0:
        raise Exception("The file %s has bad voltages"%(fuldir))

    chunkavail = int(np.floor(datraw.shape[0] / chunk))

    mintime = hourcon
    maxtime = hourcon + (rowdt * chunk * chunkavail) / 60 ** 2

    err = np.std(datVnorm, axis=1)
    compress_err = (np.array([(err[ii * chunk:(ii + 1) * chunk]).mean(axis=0)/chunk**.5 for ii in range(chunkavail)]))*g2conscorr
    compress_datVnorm = (np.array([(datVnorm[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])-1)*g2conscorr
    compress_voltcomb = np.array([(voltcomb[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
    rawdatvn = np.array([(datraw[:, 2:][ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
    return compress_datVnorm, compress_voltcomb, rawvolt1, rawvolt2, mintime, maxtime, chunk, chunkavail, g2conscorr, \
           midnight_date, unixtime, rawdatvn, compress_err,datraw

def data_reduce(g2_surface, g2_filtered, g2_surface_raw,skytimes,radstar,my_odp_corrs, unixtime, rawvolt1, rawvolt2,
                telname, target_name, runname, save_dir, utrack, vtrack, airmass, voltcomb, compress_err):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scrub_save_dir = os.path.join(save_dir, "g2surfaceFourierFiltered.siidat")
    np.savetxt(scrub_save_dir, g2_surface)
    current_tel_index = Ts.index(telname)

    rawg2surf_dir = os.path.join(save_dir, "g2surfaceraw.siidat")
    np.savetxt(rawg2surf_dir, g2_surface_raw)

    filtg2surf_dir = os.path.join(save_dir, "g2surfaceRadioFiltered.siidat")
    np.savetxt(filtg2surf_dir, g2_filtered)

    skytime_interpol = np.linspace(skytimes.min(), skytimes.max(), g2_surface.shape[0])
    telbaseline = array_spliner(skytimes, radstar[current_tel_index], skytime_interpol)
    baseline_save_dir = os.path.join(save_dir, "baseline.siidat")
    np.savetxt(baseline_save_dir, telbaseline)

    voltagefull1_save_dir = os.path.join(save_dir, "full%sVoltage.siidat" % (telname[:2]))
    np.savetxt(voltagefull1_save_dir, rawvolt1)

    voltagefull2_save_dir = os.path.join(save_dir, "full%sVoltage.siidat" % (telname[2:]))
    np.savetxt(voltagefull2_save_dir, rawvolt2)

    chunk = 480
    chunkavail = int(np.floor(rawvolt1.shape[0] / chunk))
    voltagecomp1_save_dir = os.path.join(save_dir, "comp%sVoltage.siidat" % (telname[:2]))
    # vmeans1 = np.zeros(chunkavail, dtype=float)
    # IItools.datcompress(rawvolt1,vmeans1, chunk, chunkavail)
    np.savetxt(voltagecomp1_save_dir, IItools.datcompress(rawvolt2, chunk, chunkavail))

    # vmeans2 = np.zeros(chunkavail, dtype=float)
    # IItools.datcompress(rawvolt2,vmeans2, chunk, chunkavail)
    voltagecomp2_save_dir = os.path.join(save_dir, "comp%sVoltage.siidat" % (telname[2:]))
    np.savetxt(voltagecomp2_save_dir, IItools.datcompress(rawvolt2, chunk, chunkavail))

    delay_pos = str.split(runname, "_")[-1]
    if delay_pos[0] == 'p':
        corr_delay = float(delay_pos[1:-8]) * 4
        my_odp_corr = my_odp_corrs[1]
        opd_tel = my_odp_corr[current_tel_index]
        pair_delay = pair_delays[current_tel_index]
        telopdcorr = array_spliner(skytimes, opd_tel, skytime_interpol)
        if telname == "T3T4":
            telopdcorr = -telopdcorr
        avg_telopdcorr = telopdcorr + pair_delay + corr_delay + 64
    elif delay_pos[0] == 'm':
        corr_delay = -float(delay_pos[1:-8]) * 4
        my_odp_corr = my_odp_corrs[1]
        opd_tel = my_odp_corr[current_tel_index]
        pair_delay = pair_delays[current_tel_index]
        telopdcorr = array_spliner(skytimes, opd_tel, skytime_interpol)
        if telname == "T3T4":
            telopdcorr = -telopdcorr
        avg_telopdcorr = telopdcorr + pair_delay + corr_delay + 64

    # avg_telopdcorr = avg_telopdcorr-avg_telopdcorr[0] +71

    opd_save_dir = os.path.join(save_dir, "telOPD.siidat")
    np.savetxt(opd_save_dir, avg_telopdcorr)

    opdgeo_save_dir = os.path.join(save_dir, "geoOPD.siidat")
    np.savetxt(opdgeo_save_dir, telopdcorr*4)

    utrackdir = os.path.join(save_dir, "utrack.siidat")
    vtrackdir = os.path.join(save_dir, "vtrack.siidat")

    utrackspl = array_spliner(skytimes, utrack[current_tel_index], skytime_interpol)
    np.savetxt(utrackdir, np.array(utrackspl))
    vtrackspl = array_spliner(skytimes, vtrack[current_tel_index], skytime_interpol)
    np.savetxt(vtrackdir, np.array(vtrackspl))

    amspl_tim = array_spliner(np.arange(len(skytimes)), skytimes, np.linspace(0,len(skytimes), len(airmass)))
    airmass_spline = array_spliner(amspl_tim, airmass, skytime_interpol)
    airmass_dir = os.path.join(save_dir, "airmass.siidat")
    np.savetxt(airmass_dir, airmass_spline)

    skytime_dir = os.path.join(save_dir, "skytime.siidat")
    np.savetxt(skytime_dir, np.array(skytime_interpol))

    voltcomb_dir = os.path.join(save_dir, "voltAB.siidat")
    np.savetxt(voltcomb_dir, np.array(voltcomb))

    compress_err_dir = os.path.join(save_dir, "compresserror.siidat")
    np.savetxt(compress_err_dir, np.array(compress_err))


    return telbaseline, avg_telopdcorr, skytime_interpol,current_tel_index, airmass_spline




def index_selector(indexable_things):
    notfinishedselecting = True
    while (notfinishedselecting):
        for i, thing in enumerate(indexable_things):
            print("%s: %s" % (i, thing))
        selected_index = input("Please enter the index number for the entry you wish to select\n")

        try:
            selected_index = int(selected_index)
            selected_thing = indexable_things[selected_index]
            notfinishedselecting = False
        except Exception as e:
            print("You must enter in a valid index else this won't work. Just in case the error is printed below")
            print(e)
    return selected_thing, selected_index

def directory_crawler(basdir, mode, multired=False):
    continue_crawl = True
    currentdir = basdir
    if mode==0:
        reduction_instruc = "Navigate to the desired file and enter it's index to reduce it\n"
    if mode==1:
        reduction_instruc = "Navigate to the desired parent directory and enter 'mult' to reduce every file in the sub-directories\n"
    notfinishedselecting = True
    while (notfinishedselecting):
        cls()
        try:
            if '.txt' in currentdir:
                notfinishedselecting = False
                break
            if len(currentdir) == 0:
                print("Reseting directory to starting directory")
                currentdir = basdir
            current_dirs = os.listdir(currentdir)
            for i, thing in enumerate(current_dirs):
                print("%s: %s" % (i, thing))

            print(sep_line)
            print("Current Directory = %s"%(currentdir))
            selected_index = input("%s"
                                   "Enter b to go up a directory\n"
                                   "Enter s to return to the starting directory\n"
                                   "Enter q to quit\n"%(reduction_instruc)).lower()

            if selected_index == 'b':
                splitdir = currentdir.split(os.sep)
                currentdir = os.sep.join(splitdir[:-1])
            elif selected_index == 's':
                currentdir = basdir
            elif selected_index == 'q':
                quit()
            elif multired and selected_index == 'mult':
                return currentdir

            else:
                selected_index = int(selected_index)
                selected_thing = current_dirs[selected_index]
                pastdir = currentdir
                currentdir = os.path.join(currentdir, selected_thing)
                if '.txt' in currentdir:
                    break
                current_dirs = os.listdir(currentdir)
                for i, thing in enumerate(current_dirs):
                    if mode == 1 and 'corrOut' in thing:
                        notfinishedselecting = False
                        currentdir = pastdir
        except Exception as e:
            print("You must enter in a valid response. The error is printed below")
            print(e)
    return currentdir

def datcompress(A,chunk):
    chunkavail = int(np.floor(A.shape[0] / chunk))
    return np.array([(A[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])

def single_target_analysis(args):
    fuldir, timesum, savdat, savplots = args
    try:
        truncdir = fuldir.split(os.sep)
        targetname = truncdir[-4]
        dayname = truncdir[-3]
        telname = truncdir[-2]
        runname = truncdir[-1]
        unclean_g2surface, voltcomb, rawvolt1, rawvolt2, mintime, maxtime, chunk, chunkavail, \
        g2conscorr, midnight_date, unixtime, rawdatvn, compress_err, raw_dat = \
            open_veritas_sii(fuldir, timesum=timesum)

        clean_g2surface0, fourier_pars0 = IItools.fourier_radio_clean(unclean_g2surface[:, :-1])
        clean_g2surface, fourier_pars = IItools.fourier_radio_clean(clean_g2surface0, [.1, 1])

        radio_clean_g2_res = IItools.radio_clean_sigmoid(unclean_g2surface[:, :-1], 65, 3)
        radiofilt_g2surface, radioindexfft, freqs, sigmoid_window = radio_clean_g2_res
        voltcut = np.where(voltcomb > voltcomb.max() / 2)

        try:
            cortargname = targetname[0:3] + ' ' + targetname[3:]
            if targetname == 'mulep':
                cortargname = "mu lep"
            result_table = Simbad.query_object(cortargname)
            raraw = result_table["RA"]
            decraw = result_table["DEC"]
            skycoord = SkyCoord(raraw, decraw, unit=("hourangle", "deg"))
            ra = skycoord.ra.to("hourangle").value[0]
            dec = skycoord.dec.to("degree").value[0]
            targmag = result_table["FLUX_B"].value[0]
        except Exception as e:
            print(e)
            print()
            ra = float(input("auto RA failed, please input target RA in hourangle i.e. 16.323"))
            dec = float(input("auto dec failed, please input target dec in degrees 68.444"))
            targmag = -9999

        mas = 1
        hoff = 0
        radstar, ampstar, skytimes, telarr, ra_dec_str, my_odp_corrs, utrack, vtrack, tel_arr = \
            star_simulator(ra * u.hourangle, dec * u.deg,
                           ang_diam=(mas * u.mas).to('rad').value,
                           st=mintime+hoff,
                           et=maxtime+hoff,
                           date=midnight_date)

        meang2surfraw = np.array([g - g.mean() for g in unclean_g2surface[:,:-1]])/g2conscorr

        datadate = unixtime.strftime('%Yx%mx%dX%Hx%Mx%S')

        save_dir = os.path.join(reduceDataDir, targetname, telname, datadate)
        graph_save_dir = os.path.join(save_dir, graphsFoldername)


        airmassfull = tel_arr.star_dict[ra_dec_str]["Airmass"]
        telbaseline, avg_telopdcorr, skytime_interpol, current_tel_index, airmass_spl = data_reduce(clean_g2surface, radiofilt_g2surface,
                                                                    meang2surfraw,skytimes,radstar,my_odp_corrs, unixtime,
                                                                    rawvolt1, rawvolt2, telname, targetname,
                                                                    runname, save_dir, utrack, vtrack, airmassfull, voltcomb,
                                                                                                    compress_err)


        cutvolts = np.where(np.abs(radiofilt_g2surface.std(axis=1)) < 2.5)

        g2opdrange=50
        shifted_g2, roundopd = IItools.g2_shifter_mid(radiofilt_g2surface, avg_telopdcorr)
        timedelay_tocheck = np.linspace(-g2opdrange,g2opdrange,1000)
        ampmeans, tshift, tlin_cor, bestfitamp, bestfitamp_highpoly = IItools.opd_correction_new(
            shifted_g2[cutvolts],
            roundopd[cutvolts],
            telbaseline[cutvolts] - telbaseline[
                cutvolts].min(),
            timedelay_tocheck,
            datstart=0,
            datend=radiofilt_g2surface[cutvolts].shape[1],
            order=1)

        savename = "%s %s %s" % (targetname, telname, datadate)
        measstd = radiofilt_g2surface.std(axis=1)
        tim = np.arange(len(measstd))
        stdpolyfit = np.poly1d(np.polyfit(tim, measstd, 4))(tim)
        g2amps_rowbyrow = IItools.g2_amps_rbr(shifted_g2[:, :-1], roundopd + tshift)

        if savdat:
            opd_save_dir = os.path.join(save_dir, "telOPDcorr.siidat")
            np.savetxt(opd_save_dir, avg_telopdcorr + tshift)

            opd_save_dir = os.path.join(save_dir, "telOPD.siidat")
            np.savetxt(opd_save_dir, avg_telopdcorr)


            v2rbr_save_dir = os.path.join(save_dir, "V2rowbyrow.siidat")
            np.savetxt(v2rbr_save_dir, g2amps_rowbyrow)

            err_save_dir = os.path.join(save_dir, "errorRowbyRow.siidat")
            np.savetxt(err_save_dir, measstd)

            rawdatdir = os.path.join(save_dir, "rawg2counts.siidat")
            np.savetxt(rawdatdir, rawdatvn)

            polyerr_save_dir = os.path.join(save_dir, "polyerrorRowbyRow.siidat")
            np.savetxt(polyerr_save_dir, stdpolyfit)

            opdstart_save_dir = os.path.join(save_dir, "opdStartMaxCorr.siidat")
            np.savetxt(opdstart_save_dir, tlin_cor)

            g2shift_save_dir = os.path.join(save_dir, "g2shifted.siidat")
            np.savetxt(g2shift_save_dir, shifted_g2)

            opdshift_save_dir = os.path.join(save_dir, "shiftedopd.siidat")
            np.savetxt(opdshift_save_dir, roundopd)

        if savplots:
            IIdisplay.analysis_graphs(clean_g2surface[:, :-1],
                            shifted_g2,
                            radiofilt_g2surface,
                            avg_telopdcorr,
                            roundopd,
                            telbaseline,
                            meang2surfraw[:, :-1],
                            measstd,
                            stdpolyfit,
                            timesum,
                            savename,
                            targmag,
                            graph_save_dir,
                            tlin_cor,
                            ampmeans,
                            tshift,
                            fourier_pars,
                            cutvolts,
                            radio_clean_g2_res)

            IIdisplay.fit_graph(radiofilt_g2surface[:, :-1][cutvolts],
                      (avg_telopdcorr + tshift)[cutvolts],
                      telbaseline[cutvolts],
                      savename,
                      graph_save_dir,
                      4)

            webbrowser.open(os.path.relpath(graph_save_dir))
            startposition = (avg_telopdcorr[0] + tshift)
            successmsg = "Success in reading %s %s %s %s" % (targetname, dayname, telname, runname)
            return successmsg
            # return tshift, radiofilt_g2surface, telbaseline, avg_telopdcorr, skytime_interpol, compress_err, g2amps_rowbyrow, tshift
        else:
            startposition = (avg_telopdcorr[0] + tshift)
            successmsg = "Success in reading %s %s %s %s" % (targetname, dayname, telname, runname)
            return successmsg
    except Exception as e:
        exception_message = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        failmessage = "Failed in reading %s Here is the Traceback \n %s" % (fuldir, exception_message)
        return failmessage

def multi_pair_run_analysis(fuldir, timesum = 60):
    analysis_args = []
    for root, dirs, files in os.walk(fuldir, topdown=False):
        for file in files:
            if "corrOut" in file and '.txt' in file:
                file_path = os.path.join(root, file)
                analysis_arg = [file_path, timesum, True, False]
                analysis_args.append(analysis_arg)
    with multiprocessing.Pool() as analysis_pool:
        s=time.time()
        analysis_messages = analysis_pool.map(single_target_analysis, analysis_args)
        analysis_pool.close()
        analysis_pool.join()
        e=time.time()
    s1=time.time()
    for ar in analysis_args:
        single_target_analysis(ar)
    e1 = time.time()
    print(e-s)
    print(e1-s1)
    for mesges in analysis_messages:
        print(mesges)


    # with multiprocessing.Pool() as analysis_pool:
    #     for root, dirs, files in os.walk(fuldir, topdown=False):
    #         for file in files:
    #             if "corrOut" in file and '.txt' in file:
    #                 file_path = os.path.join(root, file)
    #                 analysis_arg = [file_path, timesum, True, False]
    #                 # single_target_analysis(*analysis_args)
    #                 analysis_args.append(analysis_arg)
    #                 # analysis_pool.apply_async(singtarganal, analysis_arg)
    #                 print("Started %s" % (file_path))
    #     analysis_pool.close()
    #     analysis_pool.join()

    asdf=333



def main():
    continue_user_input = True

    basedirnotfound = True
    # stdir = "datasii"

    if basedirnotfound:
        while(basedirnotfound):
            print(sep_line)
            stdir = input("Please enter the full data directory (folder directory where all target data is located)\n")
            print()
            try:
                targs_in_dir = os.listdir(stdir)
                for targ in targs_in_dir:
                    print(targ)
                print(sep_line)
                print(targetmessage)
                print(sep_line)
                do_directory = input("Are these the correct targets you wish to analyze? y for yes, anything else for no\n")
                if do_directory.lower() == 'y':
                    basedirnotfound = False
                else:
                    print("Asking for another directory")
                    continue

            except Exception as e:
                print(e)
                print("You probably didn't enter the directory correctly, try again")
                continue


    splitstdir = stdir.split(os.sep)
    current_dir = stdir
    user_defined_mode = False

    while not user_defined_mode:
        try:
            print("0: Single Pair Analysis")
            print("1: Multi Pair, Multi Run, Analysis\n")
            user_defined_mode = input("Please select analysis mode:\n")
            user_mode_sel = int(user_defined_mode)
            if np.abs(user_mode_sel) > 1:
                print("Sorry that mode doesn't exist, please make a different selection\n")
                user_defined_mode = False
            else:
                user_defined_mode = True

        except Exception as e:
            print(e)
            print("You need to enter the corresponding index for the mode you desire")
            user_defined_mode = False

    if continue_user_input:
        userseldir = ""
        while (continue_user_input):
            print(sep_line)
            try:
                if user_mode_sel == 0:
                    fuldir = directory_crawler(current_dir, user_mode_sel, multired=False)
                    truncdir = fuldir.split(os.sep)
                    args = [fuldir, 60, True, True]
                    message = single_target_analysis(args)
                    print(message)

                elif user_mode_sel == 1:
                    fuldir = directory_crawler(current_dir, user_mode_sel, multired=True)
                    multi_pair_run_analysis(fuldir, 60)

                do_directory = input("Do you wish to quit? Enter 'y' for yes, anything else for no\n")
                if do_directory.lower() == 'y':
                    continue_user_input = False
                else:
                    print("Asking for another directory")
                    current_dir = os.path.split(fuldir)[0]
                    continue

            except Exception as e:
                print(e)
                print("Something went wrong with the reduction")
                print(sys.exc_info()[2])
                current_dir = os.path.split(fuldir)[0]
                # stdir = False
                continue

if __name__ == "__main__":
    pair_delays = []
    Ts = []
    n = 5
    for ii in range(n):
        for j in range(1, n - ii):
            pair_delays.append((mes_delays[j + ii] - mes_delays[ii]) / 4)
            Ts.append("T%sT%s" % (ii + 0, ii + j + 0))

    pair_delays[-1] = -np.abs(pair_delays[-1])
    main()

# import numba
# @numba.njit(parallel=True)
# def gaussian(x, mu, sig, amp):
#     return amp * np.exp(-0.5 * (x-mu)**2 / sig**2)
#
# @numba.njit(parallel=True)
# def g2_sig_surface(g2width, g2amp, g2position, g2shape):
#     g2frame = np.zeros(g2shape)
#     timechunks = g2shape[0]
#
#     timedelsaysize = g2shape[1]
#     x = np.arange(timedelsaysize)
#     for i in range(timechunks):
#         g2frame[i] = g2frame[i] + gaussian(x, g2position[i], g2width, g2amp[i])
#     return g2frame
#
# def amp_anal(data, odp_corr, baseline, start, end, order=3, g2width=0.9):
#
#     cut_opd_correction = odp_corr - start
#     cutdata = data[:, start:end]
#     g2shape = cutdata.shape
#
#     def g2_sig_amp_ravel(x, *args):
#         polyterms = [term for term in args]
#         g2amp_poly = np.poly1d(polyterms)
#         g2amp_model = g2amp_poly(x)
#         ravelg2 = g2_sig_surface(g2width, g2amp_model, cut_opd_correction, g2shape).ravel()
#         return ravelg2
#
#     time = np.arange(cutdata.shape[0])
#
#     guess_par = np.zeros(order)
#     g2fitpar, g2fiterr = curve_fit(f=g2_sig_amp_ravel,
#                                    xdata=baseline,
#                                    ydata=cutdata.ravel(),
#                                    p0=guess_par)
#     amps = np.poly1d(g2fitpar)(baseline)
#     return amps, cut_opd_correction, cutdata, g2fitpar
# def opd_correction_new(data, opd, baseline, tcors, datstart=40, datend=90, order=4, highorder=4, g2width=0.9):
#     steps = len(tcors)
#     amp_means = np.zeros(steps)
#     opdstart_tcor = np.zeros(len(tcors))
#     for i in range(steps):
#         opdcorrected = opd + tcors[i]
#         amps, cut_opd_correction, cutdata , polypars = amp_anal(data, opdcorrected, baseline, datstart, datend, order=order, g2width=g2width)
#         amp_means[i] = amps.mean()
#         opdstart_tcor[i] = opdcorrected[0]
#     peak_arg = np.argmax(amp_means)
#
#     tcorr = opdstart_tcor[peak_arg]
#     bestfitamp, _, _, polypars = amp_anal(data, opd + tcorr, baseline, datstart, datend, order=order)
#     bestfitamp_highpoly, _, _, polypars_high = amp_anal(data, opd + tcorr, baseline, datstart, datend,
#                                                         order=highorder)
#     tshift = tcorr - opd[0]
#
#
#     return amp_means, tshift, opdstart_tcor, bestfitamp,bestfitamp_highpoly
# timedelay_tocheck = np.linspace(-g2opdrange, g2opdrange, 10)
# s=time.time()
# ampmeans, tshift, tlin_cor, bestfitamp, bestfitamp_highpoly = opd_correction_new(
#     shifted_g2[cutvolts],
#     roundopd[cutvolts],
#     telbaseline[cutvolts] - telbaseline[
#         cutvolts].min(),
#     timedelay_tocheck,
#     datstart=0,
#     datend=radiofilt_g2surface[cutvolts].shape[1],
#     order=1)
# e=time.time()
# print('009')
# print(e-s)