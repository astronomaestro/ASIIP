import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization as viz
from astropy.coordinates import Angle
from II import IItools, IImodels
import numpy as np
import os

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

def uvtracks_integrated(varray, tel_tracks, airy_func,save_dir, name, err, noise=None):
    noise_array = 0
    nlen = len(varray.star_dict[star_id]["IntTimes"])-1
    if noise:
        noise_array = np.random.normal(0,err,nlen)
    x_0 = airy_func.x_0.value
    y_0 = airy_func.y_0.value
    plt.figure(figsize=(22, 16))
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        airy_amp = airy_func(utrack, vtrack)
        airy_radius = np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2)
        airy_I, trap_err, Irads = IItools.trap_w_err(airy_amp, airy_radius, err, err)
        airy_err = np.ones(len(airy_I)) * err
        plt.errorbar(x=.5*Irads + airy_radius[:-1], y=airy_I / Irads + noise_array, yerr=airy_err,xerr=Irads, fmt='o')
        plt.plot(airy_radius, airy_amp)
    title = "UV integration times vs integration for %s" % (name)
    plt.title(title)
    plt.xlabel('Radius')
    plt.ylabel("normalized amplitude")
    plt.xlim(0, 180)
    plt.ylim(0)
    graph_saver(save_dir, title+"1D")



def uv_tracks_plot(tel_tracks, veritas_tels, baselines, arcsec, save_dir):
    plt.figure(figsize=(18, 15))
    plt.tight_layout()

    for i, track in enumerate(tel_tracks):
        plt.plot(track[0][:, 0], track[0][:, 1], linewidth=4)
        plt.text(track[0][:, 0][5], track[0][:, 1][5], "Baseline %s" % (baselines[i]), fontsize=14)
        plt.plot(track[1][:, 0], track[1][:, 1], linewidth=4)
        plt.text(track[1][:, 0][5], track[1][:, 1][5], "Baseline %s" % (-baselines[i]), fontsize=14)

    starttime = veritas_tels[0].time_info.T + veritas_tels[0].observable_times[0] - 6 * u.hour
    endtime = veritas_tels[0].time_info.T + veritas_tels[0].observable_times[-1] - 6 * u.hour

    plt.xlabel("U (meters)")
    plt.ylabel("V (meters)")
    title = "UV plane coverage at times \n %s to %s of %s and %s" % (starttime.T, endtime.T, arcsec, wavelength)
    plt.title(title, fontsize=18)
    graph_saver(save_dir, title)

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


