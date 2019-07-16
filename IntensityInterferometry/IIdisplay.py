import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization as viz
from astropy.coordinates import Angle
from IntensityInterferometry import IItools, IImodels
norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())
import numpy as np
import os


def uvtrack_model_run(tel_tracks, airy_func, star_err, guess_r, wavelength, star_name, intTime, save_dir):
    rads, amps, avgrad, avgamp = IImodels.avg_air1D(tel_tracks=tel_tracks,
                                                    airy_func=airy_func,
                                                    err=star_err)
    yerr = np.random.normal(0, star_err, avgamp.shape)
    rerr = np.random.normal(0, guess_r / 5)
    airy_fitr, airy_fiterr, df, der, sig = IImodels.fit_airy_avg(rads=rads,
                                                                 amps=amps,
                                                                 avg_rads=avgrad,
                                                                 avg_amps=avgamp + yerr,
                                                                 func=airy_func,
                                                                 err=star_err,
                                                                 guess_r=guess_r + rerr,
                                                                 real_r=guess_r)
    fit_diam = (wavelength.to('m').value / airy_fitr[0] * u.rad).to('mas')
    fit_err = np.sqrt(np.diag(airy_fiterr))[0] / airy_fitr[0] * fit_diam
    tr_Irad = avgrad.ravel()
    tr_Ints = avgamp.ravel()
    tr_rad = rads.ravel()
    tr_amp = amps.ravel()
    rs = np.argsort(tr_rad)
    plt.figure(figsize=(32, 20))
    plt.errorbar(x=tr_Irad,
                 y=tr_Ints + yerr.ravel(),
                 fmt='o',
                 yerr=np.full(np.alen(tr_Ints), star_err),
                 label="Model w/ err",
                 linewidth=2)
    plt.scatter(tr_Irad, tr_Ints, label="Acutal Integration", s=120, color="r")
    plt.plot(tr_rad[rs], IImodels.airy1D(tr_rad, airy_fitr)[rs], linestyle="--", label="Fitted Airy Function",
             linewidth=4)
    plt.plot(tr_rad[rs], tr_amp[rs], '-', label="True Airy Function", linewidth=3)
    title = "Star %s Integration time of %s\n fit mas of data is %s +- %s or %.2f percent" % (
        star_name, intTime, fit_diam, fit_err, fit_err / fit_diam * 100)
    plt.title(title, fontsize=28)
    plt.legend(fontsize=28)
    plt.xlabel("Meters", fontsize=22)
    plt.ylabel("Amplitude", fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=18)

    graph_saver(save_dir, title+"1D")


def uvtracks_integrated(varray, tel_tracks, airy_func,save_dir, name, err, noise=None):
    noise_array = 0
    nlen = np.alen(varray.star_dict[star_id]["IntTimes"])-1
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
        airy_err = np.ones(np.alen(airy_I)) * err
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



def uvtracks_airydisk2D(tel_tracks, veritas_tels, baselines, airy_disk, arcsec, wavelength, save_dir, name, err):
    half_xlen = np.alen(airy_disk[0]) / 2
    half_ylen = np.alen(airy_disk[1]) / 2

    plt.figure(figsize=(28, 28))
    plt.imshow(airy_disk,
               norm=viz.ImageNormalize(1, stretch=viz.LogStretch()),
               extent=[-half_xlen, half_xlen, -half_ylen, half_ylen])
    for i, track in enumerate(tel_tracks):
        plt.plot(track[0][:, 0], track[0][:, 1], linewidth=4, color='r')
        plt.text(track[0][:, 0][5], track[0][:, 1][5], "Baseline %s" % (baselines[i]), fontsize=14, color='gray')
        plt.plot(track[1][:, 0], track[1][:, 1], linewidth=4, color='r')
        plt.text(track[1][:, 0][5], track[1][:, 1][5], "Baseline %s" % (-baselines[i]), fontsize=14, color='gray')
    starttime = veritas_tels.time_info.T + veritas_tels.observable_times[0] - 6 * u.hour
    endtime = veritas_tels.time_info.T + veritas_tels.observable_times[-1] - 6 * u.hour
    plt.xlabel("U (meters)", fontsize=22)
    plt.ylabel("V (meters)", fontsize=22)
    title = "UV plane coverage of %s at %s and %s Airy Disk at times \n %s to %s" % (name, wavelength,arcsec, starttime.T, endtime.T)
    plt.title(title, fontsize=18)
    plt.colorbar()
    graph_saver(save_dir, title)

def uvtracks_amplitudes(tel_tracks, baselines, airy_func, arcsec, wavelength, save_dir, name, err):
    plt.figure(figsize=(28, 28))
    subplot_num = np.ceil(np.alen(baselines) ** .5)
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
        yerr = np.full(np.alen(airy_radius), err)
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


