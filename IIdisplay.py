import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import LogNorm
import astropy.visualization as viz
from intensityInterferometry import IItools
norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())
import numpy as np
import os



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



def uvtracks_airydisk2D(tel_tracks, veritas_tels, baselines, airy_disk, arcsec, wavelength, save_dir):
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
    title = "UV plane coverage of %s and %s Airy Disk at times \n %s to %s" % (wavelength,arcsec, starttime.T, endtime.T)
    plt.title(title, fontsize=18)
    graph_saver(save_dir, title)


def uvtracks_amplitudes(tel_tracks, baselines, airy_func, arcsec, wavelength, save_dir):
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

    title = "UV plane tracks Amplitude for %s and %s"%(arcsec, wavelength)
    plt.suptitle(title, fontsize=18)
    graph_saver(save_dir, title)

    plt.figure(figsize=(14, 10))
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        airy_amp = airy_func(utrack, vtrack)
        airy_radius = np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2)
        plt.plot(airy_radius, airy_amp, 'o')

    plt.title("UV radius Vs ampltiude coverage")
    plt.xlabel('UV radius')
    plt.ylabel("normalized amplitude")
    plt.xlim(0, 180)
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


