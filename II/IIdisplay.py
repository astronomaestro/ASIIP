import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization as viz
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from II import IItools, IImodels
import numpy as np
import os
from scipy.stats import norm
from scipy.optimize import curve_fit


def target_moon_location(tel_array, star_id, name, save_dir):

    moon_loc = tel_array.moonaltazs
    star_info = tel_array.star_dict[star_id]


    moon_loc = SkyCoord(moon_loc.az, moon_loc.alt)
    star_loc = SkyCoord(star_info['fullAz'], star_info['fullAlt'])
    moon_star_separation = moon_loc.separation(star_loc)

    plt.figure(figsize=(16,16))
    plt.title("%s on %s"%(name, tel_array.time_info), fontsize=32)
    plt.xlabel("Hours from Midnight (hours)", fontsize=28)
    plt.ylabel("Sky distance from Moon for %s (degrees)"%(name), fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.plot(tel_array.delta_time, moon_star_separation, linewidth=6)


    if save_dir:
        graph_saver(save_dir, "MoonStarDistance")
    else:
        plt.show()

def display_airy_disk(veritas_array, angd, wavelength, save_dir):
    airy_disk, airy_func = IImodels.airy_disk2D(shape=(veritas_array.xlen, veritas_array.ylen),
                                                xpos=veritas_array.xlen / 2,
                                                ypos=veritas_array.ylen / 2,
                                                angdiam=angd,
                                                wavelength=wavelength)
    # norm = viz.ImageNormalize(1, stretch=viz.LogStretch())

    plt.figure(figsize=(80, 80))

    plt.title("The Airy disk of a %s Point Source"%(angd))
    plt.xlabel("Meters")
    plt.ylabel("Meters")
    norm = viz.ImageNormalize(airy_disk, stretch=viz.SqrtStretch())

    plt.imshow(airy_disk, norm=norm, cmap="gray")

    if save_dir:
        graph_saver(save_dir, "AiryDisk")
    else:
        plt.show()



def chi_square_anal(tel_tracks, airy_func, star_err, guess_r, ang_diam, star_name, save_dir):
    angdiams, chis = IItools.chi_square_anal(airy_func=airy_func,
                                             tel_tracks=tel_tracks,
                                             guess_r=guess_r,
                                             star_err=star_err,
                                             ang_diam=ang_diam)


    plt.figure(figsize=(16,16))
    title = "%s Chi square analysis" % (star_name)
    plt.title(title, fontsize=28)
    plt.xlabel("Fit value (mas)", fontsize=22)
    plt.ylabel("ChiSquare", fontsize=22)
    plt.ylim((0,10))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)

    plt.plot(angdiams, chis, linewidth=6)

    if save_dir:
        graph_saver(save_dir, title)
    else:
        plt.show()

def uvtrack_model_run(tel_tracks, airy_func, star_err, guess_r, wavelength, star_name, ITime, save_dir, pererr,
                      fullAiry=False):
    rads, amps, avgrad, avgamp = IImodels.visibility2dTo1d(tel_tracks=tel_tracks, visibility_func=airy_func,
                                                           x_0=airy_func.x_0.value, y_0=airy_func.y_0.value)
    yerr = np.random.normal(0, star_err, avgamp.shape)
    rerr = np.random.normal(0, guess_r / 5)
    airy_fitr, airy_fiterr, sig = IImodels.fit_airy_avg(rads=rads, avg_rads=avgrad, avg_amps=avgamp + yerr,
                                                        err=star_err, guess_r=guess_r + rerr)
    fit_diam = (wavelength / airy_fitr[0] * u.rad).to('mas')
    fit_err = np.sqrt(np.diag(airy_fiterr))[0] / airy_fitr[0] * fit_diam
    tr_Irad = avgrad.ravel()
    tr_Ints = avgamp.ravel()
    tr_rad = rads.ravel()
    tr_amp = amps.ravel()
    rs = np.argsort(tr_rad)
    plt.figure(figsize=(18, 12))
    plt.errorbar(x=tr_Irad,
                 y=tr_Ints + yerr.ravel(),
                 fmt='o',
                 yerr=np.full(len(tr_Ints), star_err),
                 label="Simulated Data",
                 linewidth=2,
                 markersize=10)
    # plt.scatter(tr_Irad, tr_Ints, label="Actual Integration", s=120, color="r")
    #
    # plt.plot(tr_rad[rs], IImodels.airy1D(tr_rad, airy_fitr)[rs], linestyle="--", label="Fitted Airy Function",
    #          linewidth=4)
    if fullAiry:
        full_x = np.linspace(start=0,
                             stop=np.max(tr_rad) + 5,
                             num=1000)
        full_y = IImodels.airy1D(full_x, guess_r)
        plt.plot(full_x, full_y, '-', label="Uniform Disk Model", linewidth=3, color='black')
        title = "Star %s" % (star_name)
    else:
        plt.plot(tr_rad[rs], tr_amp[rs], '-', label="Uniform Disk Model", linewidth=3)
        title = "Star %s Integration time of %s\n fit mas of data is %s +- %s or %.2f percent" % (
            star_name, ITime, fit_diam, fit_err, fit_err / fit_diam * 100)
    highy = IImodels.airy1D(full_x, guess_r + pererr/100*guess_r)
    lowy = IImodels.airy1D(full_x, guess_r - pererr/100*guess_r)
    plt.fill_between(full_x, lowy, highy, color='grey', alpha=0.5)
    # plt.title(title, fontsize=28)
    plt.legend(fontsize=28)
    plt.xlabel("Projected Baseline (m)", fontsize=36)
    plt.ylabel("$|V(r)|^2$", fontsize=36)
    plt.xlim([-1, np.max(tr_rad)])
    plt.tick_params(axis='both', which='major', labelsize=28, length=10, width=4)
    plt.tick_params(axis='both', which='minor', labelsize=28, length=10, width=4)
    plt.tick_params(which="major", labelsize=24, length=8, width=3)
    plt.tick_params(which="minor", length=6, width=2)


    if save_dir:
        graph_saver(save_dir, title+"1D")
    else:
        plt.show()

def uvtracks_airydisk2D(tel_tracks, veritas_tels, baselines, airy_func, guess_r, wavelength, save_dir, star_name):
    x_0 = int(np.max(np.abs(tel_tracks))*1.2)
    y_0 = int(np.max(np.abs(tel_tracks))*1.2)
    airy_disk, airy_funcd = IImodels.airy_disk2D(shape=(x_0, y_0),
                                                 xpos=x_0,
                                                 ypos=y_0,
                                                 angdiam=1.22 * wavelength / airy_func.radius.value,
                                                 wavelength=wavelength)
    y, x = np.mgrid[:x_0 * 2, :y_0 * 2]
    y, x = np.mgrid[:x_0 * 2, :y_0 * 2]
    airy_disk = airy_funcd(x, y)
    fig = plt.figure(figsize=(18, 12))

    plt.imshow(airy_disk,
               norm=viz.ImageNormalize(airy_disk, stretch=viz.LogStretch()),
               extent=[-x_0, x_0, -y_0, y_0],
               cmap='gray')
    for i, track in enumerate(tel_tracks):
        plt.plot(track[0][:, 0], track[0][:, 1], linewidth=6, color='b')
        # plt.text(track[0][:, 0][5], track[0][:, 1][5], "Baseline %s" % (baselines[i]), fontsize=14, color='w')
        plt.plot(track[1][:, 0], track[1][:, 1], linewidth=6, color='b')
        # plt.text(track[1][:, 0][5], track[1][:, 1][5], "Baseline %s" % (-baselines[i]), fontsize=14, color='w')
    starttime = veritas_tels.time_info.T + veritas_tels.observable_times[0]
    endtime = veritas_tels.time_info.T + veritas_tels.observable_times[-1]
    title = "Coverage of %s at VERITAS \non %s UTC" % (
        star_name, veritas_tels.time_info.T)
    # plt.title(star_name, fontsize=28)

    plt.xlabel("U (m)", fontsize=36)
    plt.ylabel("V (m)", fontsize=36)
    plt.tick_params(axis='both', which='major', labelsize=28, length=10, width=4)
    plt.tick_params(axis='both', which='minor', labelsize=28, length=10, width=4)
    plt.tick_params(which="major", labelsize=24, length=8, width=3)
    plt.tick_params(which="minor", length=6, width=2)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=24, length=6, width=3)



    if save_dir:
        graph_saver(save_dir, "CoverageOf%sOn%sUTC" % (star_name, veritas_tels.time_info.T))
    else:
        plt.show()

# def uvtracks_integrated(varray, tel_tracks, airy_func,save_dir, name, err, noise=None):
#     noise_array = 0
#     nlen = len(varray.star_dict[star_id]["IntTimes"])-1
#     if noise:
#         noise_array = np.random.normal(0,err,nlen)
#     x_0 = airy_func.x_0.value
#     y_0 = airy_func.y_0.value
#     plt.figure(figsize=(22, 16))
#     for i, track in enumerate(tel_tracks):
#         utrack = track[0][:, 0] + x_0
#         vtrack = track[0][:, 1] + y_0
#         airy_amp = airy_func(utrack, vtrack)
#         airy_radius = np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2)
#         airy_I, trap_err, Irads = IItools.trap_w_err(airy_amp, airy_radius, err, err)
#         airy_err = np.ones(len(airy_I)) * err
#         plt.errorbar(x=.5*Irads + airy_radius[:-1], y=airy_I / Irads + noise_array, yerr=airy_err,xerr=Irads, fmt='o')
#         plt.plot(airy_radius, airy_amp)
#     title = "UV integration times vs integration for %s" % (name)
#     plt.title(title)
#     plt.xlabel('Radius')
#     plt.ylabel("normalized amplitude")
#     plt.xlim(0, 180)
#     plt.ylim(0)
#     graph_saver(save_dir, title+"1D")



# def uv_tracks_plot(tel_tracks, veritas_tels, baselines, arcsec, save_dir):
#     plt.figure(figsize=(18, 15))
#     plt.tight_layout()
#
#     for i, track in enumerate(tel_tracks):
#         plt.plot(track[0][:, 0], track[0][:, 1], linewidth=4)
#         plt.text(track[0][:, 0][5], track[0][:, 1][5], "Baseline %s" % (baselines[i]), fontsize=14)
#         plt.plot(track[1][:, 0], track[1][:, 1], linewidth=4)
#         plt.text(track[1][:, 0][5], track[1][:, 1][5], "Baseline %s" % (-baselines[i]), fontsize=14)
#
#     starttime = veritas_tels[0].time_info.T + veritas_tels[0].observable_times[0] - 6 * u.hour
#     endtime = veritas_tels[0].time_info.T + veritas_tels[0].observable_times[-1] - 6 * u.hour
#
#     plt.xlabel("U (meters)")
#     plt.ylabel("V (meters)")
#     title = "UV plane coverage at times \n %s to %s of %s and %s" % (starttime.T, endtime.T, arcsec, wavelength)
#     plt.title(title, fontsize=18)
#     graph_saver(save_dir, title)

def uvtracks_amplitudes(tel_tracks, baselines, airy_func, arcsec, wavelength, save_dir, name, err):
    plt.figure(figsize=(28, 28))
    subplot_num = np.ceil(len(baselines) ** .5)
    x_0 = airy_func.x_0.value
    y_0 = airy_func.y_0.value
    for i, track in enumerate(tel_tracks):
        airy_amp = airy_func(track[0][:, 0] + x_0, track[0][:, 1] + y_0)
        airy_amp_neg = airy_func(track[1][:, 0] + x_0, track[1][:, 1] + y_0)
        plt.subplot(subplot_num, subplot_num - 1, i + 1)
        plt.plot(airy_amp, linewidth=3, label='pos baseline')
        plt.plot(airy_amp_neg, linewidth=3, label='neg baseline')
        plt.title("+-%s" % (baselines[i]), fontsize=18)
        plt.ylabel("Amplitude", fontsize=14)
        plt.xlabel("Track Position", fontsize=14)
        plt.legend()

    title = "UV plane tracks %s for %s and %s"%(name, arcsec, wavelength)
    plt.suptitle(title, fontsize=18)
    graph_saver(save_dir, title)

    plt.figure(figsize=(14, 10))
    ebar_density = 40
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        airy_amp = airy_func(utrack, vtrack)
        airy_radius = np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2)
        yerr = np.full(len(airy_radius), err)
        plt.errorbar(x=airy_radius[0::ebar_density], y=airy_amp[0::ebar_density], yerr=yerr[0::ebar_density], fmt='o')
        plt.plot(airy_radius, airy_amp, 'o')
    plt.title("UV radius Vs ampltiude coverage")
    plt.xlabel('UV radius')
    plt.ylabel("normalized amplitude")
    plt.xlim(0, 180)
    plt.ylim(0)
    graph_saver(save_dir, title+"1D")

def radial_profile_plot(data, center, data_name, arcsec, wavelength, save_dir):
    rad_prof = IItools.radial_profile(data)
    plt.figure(figsize=(18,12))
    plt.xlabel("Radius (m)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.plot(rad_prof)
    title = "Radial profile of %s for %s and wavelength %s" % (data_name, arcsec, wavelength)
    plt.title(title, fontsize=24)
    graph_saver(save_dir, title)



def graph_saver(save_dir, plot_name, fig = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_type = ".png"
    plot_name = file_name_cleaner(plot_name,"")
    final_dir = os.path.join(save_dir,plot_name + file_type)
    if fig:
        fig.savefig(final_dir, bbox_inches = "tight")
        fig.clf()
        fig.close()
    else:
        plt.savefig(final_dir, bbox_inches = "tight")
        plt.clf()
        plt.close()
    print(final_dir)

def file_name_cleaner(file_name, file_ext):
    if os.name == "nt":
        print("You are running in windows, removing illegal characters from catalog name.")
        clean_name = "".join(x for x in file_name if x.isalnum()) + file_ext
    else:
        clean_name = file_name + file_ext
    return clean_name


def pdf_fit_and_plot(dat, title, bins):
    mean, sigma = norm.fit(dat)
    plt.figure(figsize=(14, 10))
    bincount, hbins, bcontainer = plt.hist(dat, bins=bins, density=True)
    p = norm.pdf(hbins, mean, sigma)
    plt.plot(hbins, p, linewidth=2, label="sigma = %.8f mean = %.8f" % (sigma, mean))
    plt.title("Normal Distribution PDF fit to %s" % (title), fontsize=20)
    plt.legend(fontsize=16)


def analysis_graphs(g2_surface, g2_shifted, g2_filtered,telopd, roundopd, baseline, g2_surface_raw, measstd, fitstd, chunk_time,
                    savename, targmag, graph_save_dir, tlin_cor, ampmeans,tshift, fourier_pars, cutvolts, radio_clean_g2_res):
    myg2 = g2_shifted[cutvolts].mean(axis=0)
    tim = np.arange(len(myg2))
    midpoint = roundopd.mean()
    driftVnoDriftTitle = "OPD shifted g2 mean Fourier Series Subtracted\n %s" % (savename)
    plt.figure(figsize=(12, 6))
    plt.title(driftVnoDriftTitle, fontsize=16)
    plt.xlabel("Time bin (4ns)", fontsize=14)
    plt.ylabel("g2 Amplitude (unNormalized coherence 1X10^-6)", fontsize=14)
    startpos = midpoint + tshift

    fitg, fiterr = curve_fit(IItools.gaussian, tim, myg2, [startpos, .85, .0],
                             bounds=[[startpos * .999, .85, -10],
                                     [startpos * 1.0001, .8501, 10]])

    fitgorig, fiterrorig = curve_fit(IItools.gaussian, tim, myg2, [midpoint, .85, .0],
                                     bounds=[[midpoint * .8, .85, -10],
                                             [midpoint * 1.2, .8501, 10]])


    plt.plot((tim - midpoint)[20:-20], (myg2)[20:-20], label="DFT filtered g2 mean \nMean Baseline = %.4f" % (baseline.mean()))
    timedelay = np.linspace(20, 108, 1000)
    plt.plot(timedelay - midpoint,
             IItools.gaussian(timedelay, *fitg), '-',
             label="Max Method g2 Amplitude = %.4f X 10^-6  \ntshift=%.4f (4ns)\ntloc=%.4f" % (
                 fitg[-1], tshift, startpos))
    plt.plot(timedelay - midpoint,
             IItools.gaussian(timedelay, *fitgorig), '-',
             label="Orig Method g2 Amplitude = %.4f X 10^-6\ntshift=%.4f" % (fitgorig[-1], fitgorig[0]))
    plt.vlines(0, 0, myg2.max() * 1.15, 'red', linestyles="--", label="Default OPD bin")
    plt.legend(fontsize=12)


    graph_saver(graph_save_dir, driftVnoDriftTitle)


    shifted_g2_raw, _ = IItools.g2_shifter_mid(g2_surface_raw, telopd)
    myg2_raw = shifted_g2_raw[cutvolts].mean(axis=0) * 1e6
    tim = np.arange(len(myg2_raw))
    driftVnoDriftTitle_rawg2 = "OPD shifted g2 mean subtraction no cleaning\n %s" % (savename)
    plt.figure(figsize=(12, 6))
    plt.title(driftVnoDriftTitle_rawg2, fontsize=14)
    plt.xlabel("Time bin (4ns)", fontsize=14)
    plt.ylabel("g2 Amplitude (unNormalized coherence 1X10^-6)", fontsize=14)
    plt.plot(tim[20:-20] - midpoint, myg2_raw[20:-20], label="Mean Raw Data\nMean Baseline = %.4f" % (baseline.mean()))
    opdpos = startpos
    fitgraw, fiterrraw = curve_fit(IItools.gaussian, tim, myg2_raw, [opdpos, .85, .2],
                                   bounds=[[opdpos * .999, .85, -10],
                                           [opdpos * 1.0001, .8501, 10]])
    plt.plot(timedelay - midpoint, IItools.gaussian(timedelay, *fitgraw), '--',
             label="g2 Amplitude = %.4f X 10^-6" % (fitgraw[-1]))
    plt.plot(timedelay - midpoint,
             IItools.gaussian(timedelay, *fitg), '-',
             label="Max Method g2 Amplitude = %.4f X 10^-6  \ntshift=%.4f (4ns)\ntloc=%.4f" % (
                 fitg[-1], tshift, startpos))
    plt.legend(fontsize=14)
    graph_saver(graph_save_dir, driftVnoDriftTitle_rawg2)


    g2polymeanshift_radfilt = "OPD shifter g2 poly mean %s" % (savename)
    plt.figure(figsize=(12, 6))
    plt.title(g2polymeanshift_radfilt, fontsize=14)
    plt.xlabel("Time bin shift (4ns)", fontsize=14)
    plt.ylabel("g2 mean Polynomial Amplitude (unNormalized coherence 1X10^-6) ", fontsize=14)
    plt.plot(tlin_cor, ampmeans, label="Gaussian amplitude of data \nMean Baseline = %.4f\ntshift=%.4f" % (baseline.mean(), tshift))
    plt.vlines(64, 0, myg2.max() * 1.25, 'red', linestyles="--", label="Default OPD bin")
    # startpos = 64 + tshift
    plt.vlines(startpos, 0, myg2.max() * 1.25, 'blue', linestyles="--",
               label="Corrected OPD bin \ntshift=%.4f (4ns)\ntloc=%.4f" % (tshift, startpos))
    # plt.vlines(telopd.max(), 0,myg2.max()*1.05,'red',linestyles="--",label="Default OPD bin")
    # plt.vlines(telopd.max()+tshift, 0,myg2.max()*1.05,'blue',linestyles="--",label="Corrected OPD bin")
    # plt.vlines(telopd.max()+tshift, 0,1,'red',label="Corrected OPD g2 center")
    plt.legend()
    graph_saver(graph_save_dir, g2polymeanshift_radfilt)

    errorFitTitle = "g2 autocorrelation Surface histogram %.4f \n %s" % (targmag,savename)
    ravelsurface = (g2_surface[cutvolts]).ravel()
    bin_no = int(len(ravelsurface) / 200)
    pdf_fit_and_plot(ravelsurface, errorFitTitle, bin_no)
    graph_saver(graph_save_dir, errorFitTitle)

    plt.xlabel("Time Delay bin (4ns)", fontsize=14)
    plt.ylabel("Time from Observation start (%s sec)" % (chunk_time), fontsize=14)
    corHeatMapTitle = "OPD Centered DFT g2 Correlogram Heatmap \n %s" % (savename)
    plt.title(corHeatMapTitle, fontsize=14)
    plt.imshow(g2_shifted[cutvolts])
    plt.vlines([startpos - 2, startpos + 2], 0, len(g2_shifted[cutvolts]) - 1, linewidth=2, color="red")
    plt.colorbar()
    graph_saver(graph_save_dir, corHeatMapTitle)

    plt.xlabel("Time Delay bin (4ns)", fontsize=14)
    plt.ylabel("Time from Observation start (%s sec)" % (chunk_time), fontsize=14)
    corFourierHeatMapTitle = "Fourier Series Corrected g2 Heatmap \n %s" % (savename)
    plt.title(corFourierHeatMapTitle, fontsize=14)
    plt.imshow(g2_surface[cutvolts])
    plt.colorbar()
    graph_saver(graph_save_dir, corFourierHeatMapTitle)

    plt.xlabel("Time from Observation start (%s sec)" % (chunk_time), fontsize=14)
    plt.ylabel("Coherence bin STD", fontsize=14)
    plt.plot(measstd, label="Measured STD")
    plt.plot(fitstd, label="Poly Fit STD")
    plt.legend(fontsize=12)
    errorOvertimeTitle = "Bin Error of Correlogram Overtime \n %s" % (savename)
    plt.title(errorOvertimeTitle, fontsize=14)
    graph_saver(graph_save_dir, errorOvertimeTitle)

    rawHeatMapTitle = "Mean Subtracted Raw g2 signal \n %s" % (savename)
    plt.title(rawHeatMapTitle, fontsize=14)
    plt.xlabel("Time Delay bin (4ns)", fontsize=14)
    plt.ylabel("Time from Observation start (%s sec)" % (chunk_time), fontsize=14)
    plt.imshow(g2_surface_raw[cutvolts])
    plt.colorbar()
    graph_saver(graph_save_dir, rawHeatMapTitle)


    g2fouriertitle = "Fourier Parameters"
    plt.figure(figsize=(10, 8))
    plt.suptitle(g2fouriertitle, fontsize=17)
    plt.subplot(3, 1, 1)
    plt.ylabel("Fourier Period", fontsize=14)
    plt.plot(fourier_pars[:, 0])
    plt.subplot(3, 1, 2)
    plt.ylabel('Fourier Phase', fontsize=14)
    plt.plot(fourier_pars[:, 1])
    plt.subplot(3, 1, 3)
    plt.ylabel('Fourier Amplitude', fontsize=14)
    plt.plot(fourier_pars[:, 2])
    plt.xlabel("Time-Avaerage correlogram row (1 minute)")

    graph_saver(graph_save_dir, g2fouriertitle)

    g2DFTFilteringTitle = "DFT Filtering"
    radiofilt_g2surface, radioindexfft, freqs, dft_window = radio_clean_g2_res
    plt.figure(figsize=(14, 12))
    plt.subplot(3, 1, 1)
    mytim = np.arange(len(myg2))
    plt.plot((mytim - midpoint), (myg2), label="DFT filtered g2 mean")
    plt.plot(tim - midpoint, myg2_raw, label="Raw g2 mean")
    plt.xlabel("Time bin shift (4ns)", fontsize=14)
    plt.ylabel("g2 mean  1X10^-6", fontsize=14)
    plt.legend(fontsize=14)

    plt.subplot(3, 1, 2)
    abs_dft = np.abs(radioindexfft.mean(axis=0))
    plt.plot(freqs, abs_dft, label="FFT of a raw data")
    plt.xlabel("Frequency (MHz)", fontsize=15)
    plt.ylabel("Absolute Magnitude", fontsize=15)
    plt.legend(fontsize=14)

    plt.subplot(3, 1, 3)
    plt.plot(freqs, abs_dft * dft_window, label="(Mean g2 DFT)X(DFT filter)")
    plt.xlabel("Frequency (MHz)", fontsize=15)
    plt.ylabel("Absolute Magnitude", fontsize=15)
    plt.legend(fontsize=14)
    graph_saver(graph_save_dir, g2DFTFilteringTitle)

def fit_graph(g2surface, opd, baselines, savename,graph_save_dir, order=8):
    allamps, allopdcut, alldatacut, polypars = IItools.amp_anal(g2surface, opd, baselines, 0, 800, order)
    airyallamps, fitpar, g2fiterr = IItools.amp_anal_airy_limb(data=g2surface,
                                                       odp_corr=opd,
                                                       baseline=baselines,
                                                       start=0,
                                                       end=800,
                                                       guess=[120, 1, .0, 0.867],
                                                       bounds=[[60, 1, .0, 0.867], [400, 1.0001, .0001, 0.867000111]])

    plt.figure(figsize=(10, 5))
    plt.plot(baselines, allamps, '.', label="Poly fit to VERITAS data")
    modrad = np.linspace(0, baselines.max(), 400)
    modamp = IImodels.airynormLimb(modrad, *fitpar[:-1])

    mas = 1.22 * 415e-9 / fitpar[0] * 2.063e8

    plt.plot(modrad, modamp, label="Airy fit diam: %.4f mas, %.4f norm" % (mas, fitpar[1]))
    plt.xlabel("Baseline (meter)", fontsize=15)
    plt.ylabel("|V|^2 (meter)", fontsize=15)
    plt.legend(fontsize=14)
    savetitle = "Norm Fixed Projected Baseline\n %s" % (savename)
    plt.title(savetitle, fontsize=18)
    graph_saver(graph_save_dir, savetitle)

    plt.plot(opd)
    plt.xlabel("Time (min)", fontsize=15)
    plt.ylabel("Time Delay (4ns)", fontsize=15)
    savetitle = "Calculated OPD\n %s" % (savename)
    plt.title(savetitle, fontsize=18)
    graph_saver(graph_save_dir, savetitle)

    plt.plot(baselines)
    plt.xlabel("Time (min)", fontsize=15)
    plt.ylabel("Projected Baseline (meter)", fontsize=15)
    savetitle = "Calculated Projected Baseline\n %s" % (savename)
    plt.title(savetitle, fontsize=18)
    graph_saver(graph_save_dir, savetitle)

    g2model = IItools.g2_sig_surface(0.85, airyallamps, opd, g2surface.shape)
    opdmin = int(opd.min() - 7)
    opdmax = int(opd.max() + 7)
    plt.figure(figsize=(14, 10))
    savetitle = "g2 data, model, and residuals\n %s" % (savename)
    plt.suptitle(savetitle, fontsize=20)
    plt.subplot(1, 3, 1)
    plt.title('g2 data', fontsize=18)
    plt.imshow(g2surface[:, opdmin:opdmax])
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title('g2 model', fontsize=18)
    plt.imshow(g2model[:, opdmin:opdmax])
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow((g2surface - g2model)[:, opdmin:opdmax])
    plt.colorbar()
    plt.title('g2 residuals', fontsize=18)
    graph_saver(graph_save_dir, savetitle)

def multi_fit_graph(g2surface, opd, baselines, order=6):
    all_g2surf = np.concatenate(g2surface)
    all_opd = np.concatenate(opd)
    all_baseline = np.concatenate(baselines)

    allamps, allopdcut, alldatacut, polypars = IItools.amp_anal(all_g2surf, all_opd, all_baseline, 0, 800, order)
    airyallamps, fitpar, g2fiterr = IItools.amp_anal_airy_limb_convo(data=all_g2surf,
                                                                              odp_corr=all_opd,
                                                                              baseline=all_baseline,
                                                                              start=0,
                                                                              end=800,
                                                                              guess=[120, .1, .0, 0.867],
                                                                              bounds=[[60, 1, .0, 0.867],
                                                                                      [400, 11.0001, .0001,
                                                                                       0.867000111]],
                                                                              radfilt=[0, 10,
                                                                                     66, 70, 73, 75,
                                                                                     77, 78, 79, 80,
                                                                                     81, 83, 86, 87,
                                                                                     89, 91, 93, 94,
                                                                                     95, 97, 99,
                                                                                     101, 103, 105,
                                                                                     107, 108, 109,
                                                                                     110, 111, 113,
                                                                                     115, 117, 118,
                                                                                     120, 121, 122,
                                                                                     124, 125],
                                                                              fwidth=1)
    mas = 1.22*410e-9 / fitpar[0] * 2.063e8
    plt.figure(figsize=(10, 5))
    plt.plot(all_baseline, allamps, '.', label="Poly fit to VERITAS data")
    modrad = np.linspace(all_baseline.min(), all_baseline.max(), 400)
    modamp = IImodels.airynormLimb(modrad, *fitpar[:-1])
    mas = 1.22*410e-9 / fitpar[0] * 2.063e8
    plt.plot(modrad, modamp, label="Airy fit diam: %.4f mas, %.4f norm" % (mas, fitpar[1]))
    plt.xlabel("Baseline (meter)", fontsize=15)
    plt.ylabel("|V|^2 (meter)", fontsize=15)
    plt.legend(fontsize=14)
    # savetitle = "Norm Fixed Projected Baseline\n %s" % (savename)
    # plt.title(savetitle, fontsize=18)
    plt.show()