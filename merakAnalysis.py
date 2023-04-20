import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
# mpl.use('Qt5Agg')
import numba
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.stats import chisquare
from scipy.optimize import curve_fit
import sys, os
sys.path.append("ASIIP")
from II import IImodels, IIdisplay, IItools, IIdata
from scipy.interpolate import CubicSpline
from astropy.convolution import convolve_fft
from astropy.visualization import (MinMaxInterval, LinearStretch, ImageNormalize, PowerStretch)
import scipy
from scipy.signal import find_peaks
import datetime
from astropy.io import ascii
import time
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
preamp_gain = 20.e3  # V/A
adc_gain_14bit = 2 ** 14 / 1.23  # dc/V
adc_gain_8bit = 2 ** 8 / 1.23  # dc/V
amp_per_adc_14bit = 1. / (preamp_gain * adc_gain_14bit)  # amp/dc
amp_per_adc_8bit = 1. / (preamp_gain * adc_gain_8bit)  # amp
def extract_background(x, norm, back):
    return norm * (x - back)

def pdf_fit_and_plot(dat, title, bins):
    mean, sigma = norm.fit(dat)
    plt.figure(figsize=(14, 10))
    bincount, hbins, bcontainer = plt.hist(dat, bins=bins, density=True)
    p = norm.pdf(hbins, mean, sigma)
    plt.plot(hbins, p, linewidth=2, label="sigma = %.8f mean = %.8f" % (sigma, mean))
    plt.title("Density Normal PDF fit to %s" % (title), fontsize=20)
    plt.legend(fontsize=16)

def radial_profile(data, center=None, theta=0, ellip=1, mask=None):
    if center == None:
        center = (data.shape[0] / 2, data.shape[1] / 2)
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)


    # shape_mask = (x / ellip) ** 2 + (y * ellip) ** 2
    sinth = np.sin(theta)
    costh = np.cos(theta)

    # shape_mask = ((x * costh + y * sinth) / ellip) ** 2 + ((x * sinth - y * costh) * ellip) ** 2

    y, x = np.indices((data.shape))
    x = x - center[0]
    y = y - center[1]

    r = np.sqrt((x) ** 2 + (y) ** 2)
    r = np.sqrt(((x * costh + y * sinth)/ ellip) ** 2 + ((x * sinth - y * costh) * ellip) ** 2)
    r = r.astype(np.int)


    tbin = np.bincount(r.ravel()[~mask.ravel()], data.ravel()[~mask.ravel()])
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def airynormoff(r, fzero, norm):
    airymod = (IImodels.airy1D(r, fzero)) * norm
    return airymod

def airynormLimb(r, fzero, norm, lcon=0.3):
    airymod = (IImodels.airy1DLimb(r, fzero, lcon)) * norm
    return airymod

def binary_v2_r1r2(r, r1, r2, fratio, sep, norm):
    v1 = airynormoff(r, r1, norm)
    v2 = airynormoff(r, r2, norm)
    return ((v1) + (fratio ** 2) * (v2) + 2 * fratio * ((v1 ** .5) * (v2 ** .5)) * np.cos(2 * np.pi * r * sep)) / (1 + fratio) ** 2

def sgolay2d (data, window_size, order, mask, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = data.shape[0] + 2 * half_size, data.shape[1] + 2 * half_size
    Z = np.zeros( (new_shape) )
    # top band
    top_band = data[0, :]
    Z[:half_size, half_size:-half_size] =  top_band -  np.abs(np.flipud(data[1:half_size + 1, :]) - top_band)
    # bottom band
    bot_band = data[-1, :]
    Z[-half_size:, half_size:-half_size] = bot_band  + np.abs(np.flipud(data[-half_size - 1:-1, :]) - bot_band)
    # left band
    left_band = np.tile(data[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = left_band - np.abs(np.fliplr(data[:, 1:half_size + 1]) - left_band)
    # right band
    right_band = np.tile(data[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] =  right_band + np.abs(np.fliplr(data[:, -half_size - 1:-1]) - right_band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = data

    # top left corner
    topl_corner = data[0, 0]
    Z[:half_size,:half_size] = topl_corner - np.abs(np.flipud(np.fliplr(data[1:half_size + 1, 1:half_size + 1])) - topl_corner)
    # bottom right corner
    botr_corner = data[-1, -1]
    Z[-half_size:,-half_size:] = botr_corner + np.abs(np.flipud(np.fliplr(data[-half_size - 1:-1, -half_size - 1:-1])) - botr_corner)

    # top right corner
    topr_corner = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = topr_corner - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - topr_corner )
    # bottom left corner
    botl_corner = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = botl_corner - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - botl_corner )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        norm = ImageNormalize(data, vmin=-.2, vmax=.2, stretch=LinearStretch())
        # plt.imshow(np.ma.masked_array(data - convolve_fft(data, m, mask=mask), mask=mask), norm=LogNorm(), cmap='gray')
        # result = scipy.signal.fftconvolve(Z, m, mode='valid')
        result = convolve_fft(data, m, mask=mask)
        return result
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def savgol_2Dgal(gal, window, order, mask=None, deriv=None):

    # interp_gal = nan_interp(gal, mask)
    smoothed_gal = sgolay2d(gal, window, order, mask, None)
    resid = gal - smoothed_gal
    norm = ImageNormalize(gal, vmin=-.2, vmax=.2, stretch=LinearStretch())

    # plt.figure(0)
    # plt.imshow(gal, norm=LogNorm())
    #
    # plt.figure(1)
    # plt.imshow(smoothed_gal, norm=LogNorm())
    #
    # plt.figure(2)
    # plt.imshow(gal - smoothed_gal, norm=norm)
    return smoothed_gal, resid

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

@numba.jit()
def gaussian(x, mu, sig, amp):
    return amp * np.exp(-0.5 * (x-mu)**2 / sig**2)

fgx = np.linspace(-64,64,128)
@numba.jit()
def gaussian_msub(x, mu, sig, amp):
    gausmod = amp * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)
    fullgmod = amp * np.exp(-0.5 * (fgx - mu) ** 2 / sig ** 2)
    return gausmod - fullgmod.mean()

@numba.jit()
def g2_sig_surface(g2width, g2amp, g2position, g2shape):
    g2frame = np.zeros(g2shape)
    timechunks = g2shape[0]

    timedelsaysize = g2shape[1]
    x = np.arange(timedelsaysize)
    for i in range(timechunks):
        g2frame[i] = g2frame[i] + gaussian(x, g2position[i], g2width, g2amp[i])
    return g2frame

@numba.jit()
def g2_sig_surface_gsub(g2width, g2amp, g2position, g2shape):
    g2frame = np.zeros(g2shape)
    timechunks = g2shape[0]

    timedelsaysize = g2shape[1]
    x = np.arange(timedelsaysize)
    for i in range(timechunks):
        g2frame[i] = g2frame[i] + gaussian_msub(x, g2position[i], g2width, g2amp[i])
    return g2frame

def fourier(x, *a):
    tau=a[0]
    phi=a[1]
    ret = a[2] * np.cos(tau * x + phi)
    for deg in range(3, len(a)-1):
        ret += a[deg] * np.cos(tau*(deg+1) * 2 * x + phi)
    return ret

def amp_anal(data, odp_corr, baseline, start, end, order=3):

    cut_opd_correction = odp_corr - start
    cutdata = data[:, start:end]
    g2shape = cutdata.shape

    def g2_sig_amp_ravel(x, *args):
        polyterms = [term for term in args]
        g2amp_poly = np.poly1d(polyterms)
        g2amp_model = g2amp_poly(x)
        ravelg2 = g2_sig_surface(0.9, g2amp_model, cut_opd_correction, g2shape).ravel()
        return ravelg2

    time = np.arange(cutdata.shape[0])

    guess_par = np.zeros(order)
    g2fitpar, g2fiterr = curve_fit(f=g2_sig_amp_ravel,
                                   xdata=baseline,
                                   ydata=cutdata.ravel(),
                                   p0=guess_par)
    amps = np.poly1d(g2fitpar)(baseline)
    return amps, cut_opd_correction, cutdata, g2fitpar


def amp_anal_airy(data, odp_corr, baseline, start, end, guessr, guessnorm, bounds=[[60,.4],[400,1.2]]):

    cut_opd_correction = odp_corr - start
    cutdata = data[:, start:end]
    g2shape = cutdata.shape

    def g2_sig_amp_ravel(x, *args):
        airyarg = [term for term in args]
        g2amp_model = airynormoff(baseline, *airyarg)
        ravelg2 = g2_sig_surface(0.85, g2amp_model, cut_opd_correction, g2shape).ravel()
        return ravelg2

    guess_par = [guessr, guessnorm]
    g2fitpar, g2fiterr = curve_fit(f=g2_sig_amp_ravel,
                                   xdata=baseline,
                                   ydata=cutdata.ravel(),
                                   p0=guess_par,
                                   bounds=bounds)
    amps = airynormoff(baseline, *g2fitpar)
    return amps, g2fitpar[0], g2fitpar[1], g2fiterr

def amp_anal_airy_limb(data, odp_corr, baseline, start, end, guess=[120,1,0.3,0.85], bounds=[[60,.4, 0.3,0.84],[400,1.2, 0.301,0.86]]):
    cut_opd_correction = odp_corr - start
    cutdata = data[:, start:end]
    g2shape = cutdata.shape
    def g2_sig_amp_ravel(x, *args):
        airyarg = [term for term in args[:-1]]
        g2amp_model = airynormLimb(baseline, *airyarg)
        g2amp_model = g2amp_model
        g2mod = g2_sig_surface(args[-1], g2amp_model, cut_opd_correction, g2shape)
        ravelg2 = (g2mod - g2mod.mean(axis=1)[:,None]).ravel()
        return ravelg2
    g2fitpar, g2fiterr = curve_fit(f=g2_sig_amp_ravel,
                                   xdata=baseline,
                                   ydata=cutdata.ravel(),
                                   p0=guess,
                                   bounds=bounds)
    amps = airynormLimb(baseline, *g2fitpar[:-1])
    return amps, g2fitpar, g2fiterr

def amp_anal_airy_limb_convo(data, odp_corr, baseline, start, end, guess=[120,1,0.3,0.85], bounds=[[60,.4, 0.3,0.84],[400,1.2, 0.301,0.86]],
                             radfilt=[79,107.5], fwidth=2):
    cut_opd_correction = odp_corr - start
    cutdata = data[:, start:end]
    g2shape = cutdata.shape
    def g2_sig_amp_ravel(x, *args):
        airyarg = [term for term in args[:-1]]
        g2amp_model = airynormLimb(baseline, *airyarg)
        modframe = g2_sig_surface(args[-1], g2amp_model, cut_opd_correction, g2shape)
        conmodframe, radioindexfft, gauwin = radio_clean(modframe, radfilt, fwidth)
        ravelg2 = conmodframe.ravel()
        return ravelg2
    g2fitpar, g2fiterr = curve_fit(f=g2_sig_amp_ravel,
                                   xdata=baseline,
                                   ydata=cutdata.ravel(),
                                   p0=guess,
                                   bounds=bounds)
    amps = airynormLimb(baseline, *g2fitpar[:-1])
    return amps, g2fitpar, g2fiterr

from scipy.optimize import minimize
def opd_correction(data, opd, baseline, start, end, steps, datstart=40, datend=90, order=4, highorder=4):
    amp_means = np.zeros(steps)
    tcors = np.linspace(start, end, steps)
    for i in range(steps):
        amps, cut_opd_correction, cutdata , polypars = amp_anal(data, opd + tcors[i], baseline, datstart, datend, order=order)
        amp_means[i] = amps.mean()
    peak_arg = np.argmax(amp_means)

    tcut = tcors[peak_arg-10:peak_arg+10]
    ampcut = amp_means[peak_arg-10:peak_arg+10]
    polyfunc = np.poly1d(np.polyfit(tcut, (ampcut), 2))
    tcorr = -polyfunc.deriv()[0]/polyfunc.deriv()[1]
    bestfitamp, _, _ ,polypars = amp_anal(data, opd + tcorr, baseline, datstart, datend, order=order)
    bestfitamp_highpoly, _, _ ,polypars_high = amp_anal(data, opd + tcorr, baseline, datstart, datend, order=highorder)

    # plt.plot(tcors, amp_means)
    # plt.show()
    return amp_means, tcorr, tcors, bestfitamp,bestfitamp_highpoly
asdf=7474747474747

def opd_plotter():
    peakwidth = 2
    modelHeatMap = g2_sig_surface(peakwidth, 0 * ampmeans + 1, T3T4_odp + 32, dataT3T4.shape)

    plt.xlabel('Time (minutes)')
    plt.ylabel("Max g2 peak")
    plt.title("Gaussian g2 Peak Width = %s X 4ns" % (peakwidth))
    plt.plot(modelHeatMap.max(axis=1), '.-')
    plt.show()

def calc_back_norm(startratio, polyerr):
    return (1 + startratio * (polyerr / polyerr[0])) ** 2


def get_angular_diameter(r):
    return 1.22*416e-9 / r * 2.063e8

def get_angular_diameter_r1(r1):
    return 1.22*416e-9 / r1 * 2.063e8

def get_angular_diameter_r2(r2):
    return 1.22*416e-9 / r2 * 2.063e8




def g2surface_airy_backratio_prob(r, B_0):
    totsprob = 0
    for i in range(len(invcovs)):
        g2amps_model = airynormoff(baselines[i], r, 1)
        backcor = calc_back_norm(B_0, polyerrs[i])
        g2_surface_model = g2_sig_surface(0.87, g2amps_model/backcor, cutopds[i], cutdats[i].shape)
        diff = savgoldataravel[i] - g2_surface_model.ravel()
        dotprob = np.dot(diff, np.dot(invcovs[i], diff)) /2
        totsprob = dotprob + totsprob
    return -totsprob

from scipy import special
def fold_gaus_airy(r, r_zero, sigmatel, norm=1):
    mu_V1 = airynormoff(r, r_zero, norm)
    part1 = sigmatel * np.sqrt(2 / np.pi) * np.exp(-(mu_V1 ** 2) / (2 * sigmatel ** 2))
    part2 = mu_V1 * special.erf(mu_V1 / np.sqrt(2 * sigmatel ** 2))
    V1 = part1 + part2

    # mu_V2 = airynormoff(r, r_zero, 1, 0) - 1
    # part1 = sigmatel * np.sqrt(2 / np.pi) * np.exp(-(mu_V2 ** 2) / (2 * sigmatel ** 2))
    # part2 = mu_V2 * special.erf(mu_V2 / np.sqrt(2 * sigmatel ** 2))
    # V2 = -(part1 + part2 - 1)

    # return ((V1 + V2) / 2 -  mu_V1)*2 + mu_V1
    return V1

def radio_clean(dat, freqfilter=[79,107.5], widthg=2):
    # freqfilter = [79, 107.5]
    # widthg = 2
    radioindexfft = np.array([np.fft.fft(g - g.mean()) for g in dat])
    nfreq = len((radioindexfft.mean(axis=0)))
    freq = (np.arange(nfreq) - nfreq / 2) * 250e6 / (nfreq * 1e6)
    gauwin = 1
    for freqf in freqfilter:
        gau1 = 1 - gaussian(freq, freqf, widthg, 1)
        gau2 = 1 - gaussian(freq, -freqf, widthg, 1)
        gauwin = gauwin * gau1 * gau2
    gauwin = np.fft.fftshift(gauwin)
    cleang2 = np.array([np.fft.ifft(g * gauwin) for g in radioindexfft]).real
    # plt.subplot(2, 2, 1)
    # plt.imshow(cleang2)
    # plt.colorbar()
    # plt.subplot(2, 2, 2)
    # plt.imshow(dat)
    # plt.colorbar()
    # plt.subplot(2, 2, 3)
    # plt.plot(freq, np.abs(radioindexfft.mean(axis=0)))
    # plt.plot(freq, gauwin * np.abs(radioindexfft.mean(axis=0)))
    # plt.plot(freq, gauwin)
    # plt.subplot(2, 2, 4)
    # plt.plot(dat.mean(axis=0))
    # plt.plot(cleang2.mean(axis=0))

    # plt.figure(figsize=(10, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(freq, np.fft.fftshift(np.abs(radioindexfft.mean(axis=0))), label="Raw Frequency Amplitude")
    # plt.plot(freq, np.fft.fftshift((gauwin * np.abs(radioindexfft.mean(axis=0)))),
    #          label="Gaussian Filter Convolved Amplitude")
    # plt.ylabel("Absolute Amplitude", fontsize=15)
    # plt.legend(fontsize=12)
    #
    # plt.subplot(2, 1, 2)
    # plt.xlabel("Frequency (MHz)", fontsize=15)
    # plt.ylabel("Amplitude Multiplier", fontsize=15)
    # plt.plot(freq, np.fft.fftshift(gauwin))
    return cleang2, radioindexfft, [np.fft.fftshift(freq),gauwin]

def fourier_radio_clean(g2data, p0=[2,1]):
    timd = np.arange(g2data.shape[1])
    cleanframe = np.zeros(g2data.shape)
    cleanpars = np.zeros((g2data.shape[0],3))
    for i,g2sig in enumerate(g2data):
        meansubg2 = g2sig - g2sig.mean()
        popt, pcov = curve_fit(fourier, timd, meansubg2, p0=p0 + 1 * [np.ptp(g2sig)], maxfev=10000)
        formod = fourier(timd, *popt)
        meanfoursub = meansubg2 -formod
        cleanframe[i] = meanfoursub - meanfoursub.mean()
        cleanpars[i] = popt
    return cleanframe, cleanpars
def tri_sig_cov(s2,s3,s4):
    covs = np.zeros((len(s2),3,3))
    incovs = np.zeros((len(s2),3,3))
    c=0

    for sig2,sig3,sig4 in zip(s2,s3,s4):
        seedsize = .00001
        covmat = np.array([[sig2 * sig4 + np.random.normal(0, seedsize), sig3 * sig4 + np.random.normal(0, seedsize), sig4 * sig4 + np.random.normal(0,
                                                                                                                                       seedsize)],
                           [sig2 * sig2 + np.random.normal(0, seedsize), sig2 * sig3 + np.random.normal(0, seedsize), sig2 * sig4 + np.random.normal(0,
                                                                                                                                       seedsize)],
                           [sig2 * sig3 + np.random.normal(0, seedsize), sig3 * sig3 + np.random.normal(0, seedsize), sig3 * sig4 + np.random.normal(0,
                                                                                                                                       seedsize)]])
        covs[c] = covmat
        invcov = inv(covmat)
        incovs[c] = invcov
        c=c+1

    return covs, incovs

def tri_var_solver(v_ab, v_ac, v_bc):
    v_aa = v_ab * v_ac/v_bc
    v_bb = v_ab * v_bc/v_ac
    v_cc = v_ac*v_bc/v_ab
    return v_aa, v_bb, v_cc

def g2_shifter(g2_surface, opd):
    meanopd = np.array(opd - opd.mean())
    minopd = meanopd - meanopd.max()
    length, width = g2_surface.shape
    shiftsig = np.zeros((length, width))
    x = np.arange(width, dtype=int)
    for i in range(0, length):
        xg =np.array(np.ceil(x + minopd[i]), dtype=int)
        shiftsig[i] = g2_surface[i][xg]
    return shiftsig
def surface_shifter(surface, values_to_shift_by):
    minshift = mean - values_to_shift_by.max()
    length, width = surface.shape
    shiftsig = np.zeros((length, width))
    x = np.arange(width, dtype=int)
    for i in range(0, length):
        xg =np.array(np.ceil(x + minshift[i]), dtype=int)
        shiftsig[i] = surface[i][xg]
    return shiftsig

def g2_minimapper(g2_waterfall, opd, binwidth=6):
    shifted_g2 = IItools.g2_shifter(g2_waterfall, opd)
    minopd = np.ceil(opd - opd.max())
    g2loc = opd - minopd
    opdmiddle = int(np.floor(g2loc.mean()))
    cutg2 = shifted_g2.T[opdmiddle - binwidth:opdmiddle + binwidth].T
    cutopd = g2loc - opdmiddle + binwidth
    return cutg2, cutopd


def background_matcher(backgrnds, dattomatch, telnm):
    timediffs = []
    unixtimes = []
    volts = []
    volerrs = []

    for bckg in backgrnds:
        if bckg['col3'] == telnm:
            datim = \
                datetime.datetime.strptime(bckg['col1'] + " " + bckg['col2'], '%Y-%m-%d %H:%M:%S')
            timedif = dattomatch - datim
            timediffs.append(np.abs(timedif))
            unixtimes.append(time.mktime(datim.timetuple()))
            volts.append(bckg['col4'])
            volerrs.append(bckg['col5'])
    timesort = np.argsort(unixtimes)
    unixtimes_srt = np.array(unixtimes)[timesort]
    volts_srt = np.array(volts)[timesort]
    errs_srt = np.array(volerrs)[timesort]
    return unixtimes_srt, volts_srt, errs_srt

def data_getter(fuldir, red_dir, backdir, utc, doback=True):
    telnames = []
    unclean_g2surfaces = []
    voltcombs = []
    rawvolts1 = []
    rawvolts2 = []
    mintimes = []
    maxtimes = []
    datadates = []
    cleang2s = []
    radiofilt_g2s =[]
    shifted_g2s = []
    telbaselines = []
    telopdcorrs = []
    telOPDS = []
    measstds = []
    polystds = []
    skytimes_arrs = []
    utracks = []
    vtracks = []
    vsquared = []
    times = []
    origopds_ls = []
    bacratios = []
    airmasses = []
    comvolts = []
    origerr= []
    siifiledirs = []
    betdirs = sorted([bd for bd in os.listdir(backdir) if 'bet uma' in bd])
    taudirb = [bd for bd in os.listdir(backdir) if 'tau sco' in bd]
    backdirs = betdirs + taudirb

    ascii.read(os.path.join(backdir, backdirs[0]))

    for root1, dirs1, files1 in os.walk(fuldir, topdown=False):
        for file1 in files1:
            if "corrOut" in file1 and '.txt' in file1:
                times.append(datetime.datetime.strptime(file1.split('_')[2], 'y%Ym%md%dh%Hm%Ms%S'))
    for root, dirs, files in os.walk(red_dir, topdown=False):
        for file in files:
            if 'siidat' in file:
                date = datetime.datetime.strptime(os.path.split(root)[1], '%Yx%mx%dX%Hx%Mx%S')
                if date in times:
                    if "T0" not in root:
                        dat_time_start = time.mktime(date.timetuple())

                        truncdir = root.split(os.sep)
                        telname = truncdir[-2]
                        targname = truncdir[1]


                        telbaseline = np.loadtxt(os.path.join(root, 'baseline.siidat'))
                        g2surf_radfilt = np.loadtxt(os.path.join(root, 'g2surfaceRadioFiltered.siidat'))
                        telOPDcorr = np.loadtxt(os.path.join(root, 'telOPDcorr.siidat'))
                        skytime = np.loadtxt(os.path.join(root, 'skytime.siidat'))

                        # unixskytime = dat_time_start + (skytime-skytime.min())*60*60



                        utrack = np.loadtxt(os.path.join(root, 'utrack.siidat'))
                        vtrack = np.loadtxt(os.path.join(root, 'vtrack.siidat'))
                        v2 = np.loadtxt(os.path.join(root, 'V2rowbyrow.siidat'))
                        measstd = np.loadtxt(os.path.join(root, 'errorRowbyRow.siidat'))
                        polystd = np.loadtxt(os.path.join(root, 'polyerrorRowbyRow.siidat'))
                        origopd = np.loadtxt(os.path.join(root, 'telOPD.siidat'))
                        # volt1 = np.loadtxt(os.path.join(root, "full%sVoltage.siidat" % (telname[:-2])))/((250e6/8))
                        # volt2 = np.loadtxt(os.path.join(root, "full%sVoltage.siidat" % (telname[-2:])))/((250e6/8))
                        airmass = np.loadtxt(os.path.join(root, "airmass.siidat"))
                        voltAB = np.loadtxt(os.path.join(root, "voltAB.siidat"))
                        rawg2count = np.loadtxt(os.path.join(root, 'rawg2counts.siidat'))*(2**10 / (250e6/8))
                        orerr = np.loadtxt(os.path.join(root, 'compresserror.siidat'))
                        # chunk = 60*8
                        # chunkavail = int(np.floor(len(volt1) / chunk))
                        # comvol = volt1*volt2
                        # comvolt1 = np.array([(volt1[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
                        # comvolt2 = np.array([(volt2[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
                        # fronbacspline = array_spliner([unixskytime[0], unixskytime[-1]],
                        #                 [comvol[:20].mean(), comvol[-20:].mean()], unixskytime)/(250e6*128)
                        # g2raw = np.loadtxt(os.path.join(root, 'g2surfaceraw.siidat'))
                        # # (((errs_a * errs_b) / amp_per_adc_8bit ** 2) / 128
                        # onskytime_unix = np.linspace(dat_time_start, dat_time_start+len(volt1)/8, len(volt1))
                        # splinedampback = array_spliner((unixtimes_b + unixtimes_a) / 2, errs_b * errs_a * volts_a * volts_b, unixskytime)*preamp_gain**2
                        # backcounts = ((splinedampback) / amp_per_adc_8bit ** 2)
                        unixskytime = dat_time_start + (skytime - skytime.min()) * 60 * 60

                        if doback:
                            backgrounds = ascii.read(os.path.join(backdir, [bd for bd in os.listdir(backdir) if utc.lower() in bd.lower() and targname.lower() in bd.replace(" ", "").lower()][0]))
                            unixtimes_a, volts_a, errs_a = background_matcher(backgrounds, date, telname[:2])
                            unixtimes_b, volts_b, errs_b = background_matcher(backgrounds, date, telname[-2:])
                            # splinedbacktime = array_spliner((unixtimes_b+unixtimes_a)/2, errs_b*errs_a, unixskytime)
                            unittimeav = (unixtimes_b + unixtimes_a) / 2

                            # splinedampback = array_spliner(unittimeav,
                            #                                (((volts_a * volts_b) ** 1)),
                            #                                unixskytime) * preamp_gain*1.23**2
                            # splinabacka = np.poly1d(np.polyfit(unittimeav, 1.1*volts_a * volts_b* preamp_gain*1.23**2, 2))(unixskytime)
                            # splinabackb = np.poly1d(np.polyfit(unittimeav, 1.1*volts_a * volts_b* preamp_gain*1.23**2, 2))(unixskytime)
                            vcomb = (volts_a * volts_b)
                            # splinedampback = np.poly1d(np.polyfit(unittimeav, volts_a * volts_b* preamp_gain*1.23**2, 2))(unixskytime)
                            interp = PchipInterpolator(unittimeav, vcomb)
                            splinedampback = interp(unixskytime)


                            bacratio_maybe = ((splinedampback**.5 / ((rawg2count).mean(axis=1)**.5-splinedampback**.5)))**2

                            volt1 = np.loadtxt(os.path.join(root, "full%sVoltage.siidat" % (telname[:-2]))) / (
                            (250e6 / 8))
                            volt2 = np.loadtxt(os.path.join(root, "full%sVoltage.siidat" % (telname[-2:]))) / (
                            (250e6 / 8))
                            rawg2count = np.loadtxt(os.path.join(root, 'rawg2counts.siidat')) * (2 ** 10 / (250e6 / 8))

                            chunk = 60 * 8
                            chunkavail = int(np.floor(len(volt1) / chunk))
                            comvol = volt1 * volt2
                            comvolt1 = np.array(
                                [(volt1[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
                            comvolt2 = np.array(
                                [(volt2[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
                            interpa = PchipInterpolator(unittimeav, volts_a)
                            interpb = PchipInterpolator(unittimeav, volts_b)
                            splinedampbacka = interpa(unixskytime)
                            splinedampbackb = interpb(unixskytime)
                            top = rawg2count - (comvolt1 * splinedampbackb)[:, None] - (comvolt2 * splinedampbackb)[:,
                                                                                       None] + (comvolt1 * comvolt2)[:,
                                                                                               None]
                            bot = (comvolt1 - splinedampbacka) * (comvolt2 - splinedampbackb)

                            bacmult = (top/bot[:,None]).mean(axis=1)-1
                            bacratio_maybe = (bacmult)
                            if (np.isnan(bacratio_maybe).any()) or np.any(bacratio_maybe < 0):
                                dfdf=3
                                if "T1" not in telname:
                                    asdf = 34
                            bacratios.append(bacratio_maybe)
                        else:
                            bacratios.append(None)
                        # interp = PchipInterpolator(unittimeav, vcomb)
                        # splinedampback = interp(unixskytime)
                        # bacratio_maybe = ((splinedampback / ((rawg2count).mean(axis=1))))
                        # plt.plot(bacratio_maybe)
                        # backratio = splinedampback / (rawg2count.sum(axis=1) - splinedampback)
                        # splinedairmass = array_spliner(np.linspace(unixskytime.min(), unixskytime.max(), len(airmass)),
                        #                                airmass, unixskytime)

                        # bacratio_maybe = ((splinedampback/((rawg2count).mean(axis=1)-splinedampback)))**.5


                        # plt.plot(unixskytime, (1. / (splinedairmass / splinedairmass.mean())) ** 2)
                        # plt.plot(unixskytime, (rawg2count / rawg2count.mean()).mean(axis=1))
                        # plt.plot(((array_spliner((unixtimes_b + unixtimes_a) / 2, errs_b * errs_a * volts_a * volts_b,
                        #                          unixskytime) / amp_per_adc_8bit) / (comvolt1 * comvolt2 / 250e6)))
                        # plt.plot(((splinedampback) / amp_per_adc_8bit ** 2) / rawg2count.sum(axis=1),
                        #          label="background light ratio")
                        # plt.plot(backratio, label="background light ratio")
                        # plt.plot((unixtimes_b + unixtimes_a) / 2, errs_b * errs_a, '.')
                        # plt.plot(unixskytime, splinedampback)
                        # plt.plot(unixtimes_a - unixtimes_a.min(), volts_a * errs_a)
                        # plt.plot(unixtimes_b - unixtimes_a.min(), volts_b * errs_b)
                        # plt.plot(onskytime_unix, volt1*volt2)
                        g2conscorr = .5e-16
                        # telOPD = np.loadtxt(os.path.join(root,'telOPD.siidat'))
                        # if (1/orerr> 2.5).any():
                        stdfilt = np.where(1/orerr < 22222.5)
                        vsquared.append(v2[stdfilt])
                        skytimes_arrs.append(skytime[stdfilt])
                        utracks.append(utrack[stdfilt])
                        vtracks.append(vtrack[stdfilt])
                        telnames.append(telname)
                        telbaselines.append(telbaseline[stdfilt])
                        radiofilt_g2s.append(g2surf_radfilt[stdfilt])
                        telopdcorrs.append(telOPDcorr[stdfilt])
                        polystds.append(polystd[stdfilt])
                        measstds.append(measstd[stdfilt])
                        origopds_ls.append(origopd[stdfilt])
                        airmasses.append(airmass[stdfilt])
                        comvolts.append(voltAB[stdfilt])
                        origerr.append(orerr[stdfilt])
                        siifiledirs.append(root)
                        adf=444
                        break
    return vsquared, skytimes_arrs, utracks, vtracks, telnames, telbaselines, radiofilt_g2s, telopdcorrs, polystds, \
           measstds, origopds_ls,bacratios,airmasses,comvolts, origerr, siifiledirs

def amp_anal_airy_FG(data, odp_corr, baseline, start, end, guessr, guessnorm, bounds=[[60,.4],[400,1.2]]):

    cut_opd_correction = odp_corr - start
    cutdata = data[:, start:end]
    g2shape = cutdata.shape

    def g2_sig_amp_ravel(x, *args):
        airyarg = [args[0], args[1]]
        g2amp_model = fold_gaus_airy(baseline, args[0], .1, args[1])
        ravelg2 = g2_sig_surface(0.85, g2amp_model, cut_opd_correction, g2shape).ravel()
        return ravelg2

    guess_par = [guessr, guessnorm]
    g2fitpar, g2fiterr = curve_fit(f=g2_sig_amp_ravel,
                                   xdata=baseline,
                                   ydata=cutdata.ravel(),
                                   p0=guess_par,
                                   bounds=bounds)
    amps = fold_gaus_airy(baseline, g2fitpar[0], .1, g2fitpar[1])
    return amps, [g2fitpar[0], g2fitpar[1]], g2fiterr

fgsig=0.1
def multi_fit_graph(targdata, back, chunk=2, p0=[90,1], bounds = [[60, .1, .0, 0.867],[4000, 11.0001, .0001,0.867000111]], order=6,cforder=6, doplots=True, do_back=True):
    cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
    cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata = targdata
    all_g2surf = np.concatenate(cutg2s)
    all_opd = np.concatenate(cutopds)
    all_baseline = np.concatenate(cutbaselines)
    all_err = np.concatenate(cuterrs)
    if do_back:
        # backcorr_arrs = (np.array(
        #     [(((cutbacratios[i]-cutbacratios[i].min())+back**2) ** .5 + 1)**2 for i in
        #      range(len(cutbacratios))]))
        backcorrun=np.concatenate(backcorr_arrs)
        backcorr = (1+backcorrun)**2

    else:
        backcorr=np.ones(len(all_err))
        backcorr_arrs = 1# (np.array(
            # [IItools.calc_back_norm(((cutbacratios[i]-0*cutbacratios[i].min()+back**2).mean()) ** .5, cutcomperr[i] ** 1)**0 for i in
            #  range(len(cutbacratios))]))
    # backcorr = IItools.calc_back_norm(back, all_err**2)
    # backcorr = np.array([IItools.calc_back_norm(cutbacratios[i][0]**.5, cuterrs[i]**2) for i in range(len(cutbacratios))])
    all_back = np.concatenate(cutbacratios)
    # backcorr = (all_back+back**2)**.5+1
    # backcorr = (1+(all_back-all_back.min()+.00)**.5)
    allamps, allopdcut, alldatacut, polypars = IItools.amp_anal(all_g2surf*backcorr[:,None], all_opd, all_baseline, 0, 800, order)
    airyallamps, fitpar, g2fiterr = amp_anal_airy_FG(data=all_g2surf*backcorr[:,None],
                                                                              odp_corr=all_opd,
                                                                              baseline=all_baseline,
                                                                              start=0,
                                                                              end=800,
                                                                              guessr=120,
                                                                              guessnorm=1)
    gamps = []
    bass = []
    chunerr = []
    for j in range(len(cutg2s)):
        print(np.mean(backcorr_arrs[j]))
        bac = (backcorr_arrs[j] + 1) ** 2
        backg2 = cutg2s[j] * bac[:,None]
        chunkavail = int(np.floor(cutg2s[j].shape[0] / chunk))
        chcutg2 = np.array([(backg2[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
        chopd = np.array([(cutopds[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
        chbase = np.array([(cutbaselines[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
        cherr_poly = np.array([(cuterrs[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) / np.sqrt(
            len(cuterrs[j][ii * chunk:(ii + 1) * chunk])) for ii in range(chunkavail)])
        cherr_err = np.array([(cutcomperr[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) / np.sqrt(
            len(cuterrs[j][ii * chunk:(ii + 1) * chunk])) for ii in range(chunkavail)])

        startr = (cutbacratios[j][:5].mean() + back**2).mean() ** .5
        cherr_backcorr = IItools.calc_back_norm(startr, cherr_err)

        tim = np.arange(len(chcutg2[0]))
        for i, rw in enumerate(chcutg2):
            fitg, fiterr = curve_fit(IItools.gaussian_off, tim, rw,
                                     [chopd[i], 0.867000111, .2, 0],
                                     bounds=[[(chopd[i]) * .99999, 0.867000111, -10, -0.00001],
                                             [(chopd[i]) * 1.00001, 0.8670001111, 10, 0.00001]])
            gamps.append(fitg[2])
            bass.append(chbase[i])
            chunerr.append(cherr_poly[i])
    chunerr = np.array(chunerr)
    # backcorr_cfit = IItools.calc_back_norm(back, chunerr**2)
    # chunerr=chunerr*backcorr_cfit
    gamps = np.array(gamps)
    bass = np.array(bass)
    if doplots:
        plt.figure(num=0,figsize=(10, 5))
        plt.plot(all_baseline, allamps, '.', label="Poly fit to VERITAS data")
        modrad = np.linspace(all_baseline.min(), all_baseline.max(), 400)
        # modamp = IItools.airynormLimb(modrad, *fitpar[:-1])
        modamp = fold_gaus_airy(modrad, fitpar[0], .1, fitpar[1])
        mas = 1.22*416e-9 / fitpar[0] * 2.063e8
        plt.plot(modrad, modamp, label="Airy fit diam: %.4f mas, %.4f norm" % (mas, fitpar[1]))
        plt.xlabel("projected baseline (meter)", fontsize=15)
        plt.ylabel("|V|^2", fontsize=15)
        plt.legend(fontsize=14)
        # savetitle = "Norm Fixed Projected Baseline\n %s" % (savename)
        # plt.title(savetitle, fontsize=18)
        plt.show()
        plt.figure(1)
        bsrt = np.argsort(bass)
        pfitgamp = np.poly1d(np.polyfit(bass[bsrt], gamps[bsrt], cforder))
        rms = (gamps[bsrt] - pfitgamp(bass[bsrt])).std()
        plt.title("RMS:%.4f, polyerrmean:%.4f" % (rms, chunerr.mean()), fontsize=16)
        plt.errorbar(bass[bsrt], gamps[bsrt], chunerr[bsrt], fmt='.', label="data")
        # pf, per = curve_fit(airynormoff, bass[bsrt], gamps[bsrt], p0=[110,fitpar[1]], bounds=[[50,fitpar[1]*.999],[1000,fitpar[1]*1.001]], sigma=chunerr[bsrt])
        pf, per = curve_fit(fold_gaus_airy, bass[bsrt], gamps[bsrt], p0=[110,fgsig,fitpar[1]], bounds=[[50,fgsig*.99999,fitpar[1]*.999],[1000,fgsig*1.000001,fitpar[1]*1.001]], sigma=chunerr[bsrt])
        rb = np.linspace(bass.min(),bass.max(),90)
        mas1 = 1.22*416e-9 / pf[0] * 2.063e8
        plt.plot(rb, pfitgamp(rb), '--',linewidth=3)
        plt.plot(rb, fold_gaus_airy(rb, *pf), label="mas = %.4f, norm = %.4f"%(mas1, pf[2]))
        plt.xlabel("projected baseline (meter)", fontsize=14)
        plt.ylabel("Time Averaged |V|^2", fontsize=14)
        plt.legend(fontsize=14)
        # plt.figure()
        # plt.hist((gamps[bsrt] - pfitgamp(bass[bsrt])), bins=40)
        # plt.title((gamps[bsrt] - pfitgamp(bass[bsrt])).mean())
        plt.show()
    mas = 1.22 * 416e-9 / fitpar[0] * 2.063e8
    return mas

def multi_fit_graph2(targdata, back, chunk=2, p0=[90,1], bounds = [[60, .1, .0, 0.867],[4000, 11.0001, .0001,0.867000111]], order=6,cforder=6, doplots=True, do_back=True):
    cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
    cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata = targdata
    all_g2surf = np.concatenate(cutg2s)
    all_opd = np.concatenate(cutopds)
    all_baseline = np.concatenate(cutbaselines)
    all_err = np.concatenate(cuterrs)
    if do_back:
        backcorr_arrs = (np.array(
            [IItools.calc_back_norm(((cutbacratios[i]-cutbacratios[i].min()+back**2).mean()) ** .5, cutcomperr[i] ** 1) for i in
             range(len(cutbacratios))]))
        backcorrun=np.concatenate(backcorr_arrs)
        backcorr = backcorrun/backcorrun.mean()

    else:
        backcorr=np.ones(len(all_err))
        backcorr_arrs = (np.array(
            [IItools.calc_back_norm(((cutbacratios[i]-0*cutbacratios[i].min()+back**2).mean()) ** .5, cutcomperr[i] ** 1)**0 for i in
             range(len(cutbacratios))]))
    # backcorr = IItools.calc_back_norm(back, all_err**2)
    # backcorr = np.array([IItools.calc_back_norm(cutbacratios[i][0]**.5, cuterrs[i]**2) for i in range(len(cutbacratios))])
    all_back = np.concatenate(cutbacratios)
    # backcorr = (all_back+back**2)**.5+1
    # backcorr = (1+(all_back-all_back.min()+.00)**.5)
    allamps, allopdcut, alldatacut, polypars = IItools.amp_anal(all_g2surf*backcorr[:,None], all_opd, all_baseline, 0, 800, order)
    airyallamps, fitpar, g2fiterr = IItools.amp_anal_airy_limb(data=all_g2surf*backcorr[:,None],
                                                                              odp_corr=all_opd,
                                                                              baseline=all_baseline,
                                                                              start=0,
                                                                              end=800,
                                                                              guess=[120, 1, .0, 0.867],
                                                                              bounds=bounds)
    gamps = []
    bass = []
    chunerr = []
    for j in range(len(cutg2s)):
        backg2 = cutg2s[j] * backcorr_arrs[j][:, None]/backcorr.mean()
        chunkavail = int(np.floor(cutg2s[j].shape[0] / chunk))
        chcutg2 = np.array([(backg2[ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
        chopd = np.array([(cutopds[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
        chbase = np.array([(cutbaselines[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) for ii in range(chunkavail)])
        cherr_poly = np.array([(cuterrs[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) / np.sqrt(
            len(cuterrs[j][ii * chunk:(ii + 1) * chunk])) for ii in range(chunkavail)])
        cherr_err = np.array([(cutcomperr[j][ii * chunk:(ii + 1) * chunk]).mean(axis=0) / np.sqrt(
            len(cuterrs[j][ii * chunk:(ii + 1) * chunk])) for ii in range(chunkavail)])

        startr = (cutbacratios[j][:5].mean() + back**2).mean() ** .5
        cherr_backcorr = IItools.calc_back_norm(startr, cherr_err)

        tim = np.arange(len(chcutg2[0]))
        for i, rw in enumerate(chcutg2):
            fitg, fiterr = curve_fit(IItools.gaussian_off, tim, rw,
                                     [chopd[i], 0.867000111, .2, 0],
                                     bounds=[[(chopd[i]) * .99999, 0.867000111, -10, -0.00001],
                                             [(chopd[i]) * 1.00001, 0.8670001111, 10, 0.00001]])
            gamps.append(fitg[2])
            bass.append(chbase[i])
            chunerr.append(cherr_poly[i])
    chunerr = np.array(chunerr)
    # backcorr_cfit = IItools.calc_back_norm(back, chunerr**2)
    # chunerr=chunerr*backcorr_cfit
    gamps = np.array(gamps)
    bass = np.array(bass)
    if doplots:
        plt.figure(figsize=(10, 5))
        plt.plot(all_baseline, allamps, '.', label="Poly fit to VERITAS data")
        modrad = np.linspace(all_baseline.min(), all_baseline.max(), 400)
        modamp = IItools.airynormLimb(modrad, *fitpar[:-1])
        mas = 1.22*416e-9 / fitpar[0] * 2.063e8
        plt.plot(modrad, modamp, label="Airy fit diam: %.4f mas, %.4f norm" % (mas, fitpar[1]))
        plt.xlabel("projected baseline (meter)", fontsize=15)
        plt.ylabel("|V|^2", fontsize=15)
        plt.legend(fontsize=14)
        # savetitle = "Norm Fixed Projected Baseline\n %s" % (savename)
        # plt.title(savetitle, fontsize=18)
        plt.show()
        plt.figure()
        bsrt = np.argsort(bass)
        pfitgamp = np.poly1d(np.polyfit(bass[bsrt], gamps[bsrt], cforder))
        rms = (gamps[bsrt] - pfitgamp(bass[bsrt])).std()
        plt.title("RMS:%.4f, polyerrmean:%.4f" % (rms, chunerr.mean()), fontsize=16)
        plt.errorbar(bass[bsrt], gamps[bsrt], chunerr[bsrt], fmt='.', label="data")
        pf, per = curve_fit(airynormoff, bass[bsrt], gamps[bsrt], p0=[110,fitpar[1]], bounds=[[50,fitpar[1]*.999],[1000,fitpar[1]*1.001]], sigma=chunerr[bsrt])
        rb = np.linspace(bass.min(),bass.max(),90)
        mas1 = 1.22*416e-9 / pf[0] * 2.063e8
        plt.plot(rb, pfitgamp(rb), '--',linewidth=3)
        plt.plot(rb, airynormoff(rb, *pf), label="mas = %.4f, norm = %.4f"%(mas1, pf[1]))
        plt.xlabel("projected baseline (meter)", fontsize=14)
        plt.ylabel("Time Averaged |V|^2", fontsize=14)
        plt.legend(fontsize=14)
        # plt.figure()
        # plt.hist((gamps[bsrt] - pfitgamp(bass[bsrt])), bins=40)
        # plt.title((gamps[bsrt] - pfitgamp(bass[bsrt])).mean())
        plt.show()
    mas = 1.22 * 416e-9 / fitpar[0] * 2.063e8
    return mas


# def maxpdf(x,mu,sigs, n):
#     sig = sigs
#     xx = (x-mu)/sig
#     part1 =  (x*n / np.sqrt(2 * np.pi*sig**2)) * np.exp(-np.power(xx, 2) / 2)
#     part2 = np.power(0.5 * (1 + special.erf(xx / np.sqrt(2))), n - 1)
#     return part1*part2
# def stannorm(x):
#     return np.exp(-x**2/np.sqrt(2*np.pi))
# def stancdf(x):
#     return .5*(1+special.erf(x/np.sqrt(2)))
# def maxpdfstd(x,mu,sig, n):
#     xx = (x-mu)/sig
#     part1 = (x**2)*n * (1 / np.sqrt(2 * np.pi)) * np.exp(-np.power(xx, 2) / 2)/sig
#     part2 = np.power(0.5 * (1 + special.erf(xx / np.sqrt(2))), n - 1)
#     return part1*part2
#
#
# def dualmaxpdf(x, mu1, mu2, sig1, sig2):
#     p = .0
#     theta = np.sqrt((sig1 * 0) ** 2 + sig2 ** 2 + p * sig2 * sig1)
#     term1 = mu1 * stancdf((mu1 - mu2) / theta)
#     term2 = mu2 * stancdf((mu2 - mu1) / theta)
#     term3 = theta * stannorm((mu1 - mu2) / theta)
#     return (-term1 + -term2 + -term3) / 2
#
#
# def max_gaus_airymod(r, r_zero, sig, norm=1, n=100):
#     sigmatel = sig
#     bb=200
#     integral_range = 2
#     mu_V1 = IImodels.airynormoff(r, r_zero, norm)
#     zero_exp = quad(maxpdf, -integral_range, integral_range, args=(0, sigmatel, n))[0]
#     zero_exp_vari = quad(maxpdfstd, -integral_range, integral_range, args=(0, sigmatel, n))[0]
#     zero_std = (zero_exp_vari-zero_exp**2)**.5
#     # expected_V1 = np.array([quad(dualmaxpdf, -np.inf, np.inf, args=(muv, zero_exp, sig, zero_std))[0] for muv in mu_V1])
#
#     # expected_V1 = np.array([quad(dualmaxpdf, bb, -bb, args=(muv, zero_exp, sigmatel, zero_std))[0] for muv in mu_V1])/bb
#     expected_V1 = dualmaxpdf(0,mu_V1, zero_exp, sigmatel, zero_std)*-2
#
#     # for muv in mu_V1:
#     #     expected_dualmax = quad(dualmaxpdf, -np.inf, np.inf, args=(muv, zero_exp, sig, zero_std))[0]
#     #     expected_V1.append(expected_dualmax)
#     # expected_V1 = np.array(expected_V1)
#     return expected_V1
#
#
# def max_gaus_airyLimb_mod(r, r_zero, sig, norm=1, n=100):
#     sigmatel = sig
#     bb=200
#     integral_range = 2
#     mu_V1 = IImodels.airynormLimb(r, r_zero, norm, 0.5)
#     zero_exp = quad(maxpdf, -integral_range, integral_range, args=(0, sigmatel, n))[0]
#     zero_exp_vari = quad(maxpdfstd, -integral_range, integral_range, args=(0, sigmatel, n))[0]
#     zero_std = (zero_exp_vari-zero_exp**2)**.5
#     # expected_V1 = np.array([quad(dualmaxpdf, -np.inf, np.inf, args=(muv, zero_exp, sig, zero_std))[0] for muv in mu_V1])
#
#     # expected_V1 = np.array([quad(dualmaxpdf, bb, -bb, args=(muv, zero_exp, sigmatel, zero_std))[0] for muv in mu_V1])/bb
#     expected_V1 = dualmaxpdf(0,mu_V1, zero_exp, sigmatel, zero_std)*-2
#
#     # for muv in mu_V1:
#     #     expected_dualmax = quad(dualmaxpdf, -np.inf, np.inf, args=(muv, zero_exp, sig, zero_std))[0]
#     #     expected_V1.append(expected_dualmax)
#     # expected_V1 = np.array(expected_V1)
#     return expected_V1


def stannorm(x):
    return np.exp(-x**2/2)
def stancdf(x):
    return .5*(1+special.erf(x/np.sqrt(2)))
def maxpdf(x,mu,sigs, n):
    sig = sigs
    xx = (x-mu)/sig
    gaus_norm = (1 / np.sqrt(2 * np.pi))
    return x * n * gaus_norm * (stancdf(xx)**(n - 1)) * stannorm(xx)/sig
def maxpdfstd(x,mu,sig, n):
    xx = (x-mu)/sig
    gaus_norm = (1 / np.sqrt(2 * np.pi))
    return (x**2) * n * gaus_norm * stancdf(xx)**(n - 1) * stannorm(xx)/sig
def dualmaxpdf(x, mu1, mu2, sig1, sig2):
    p = .0
    theta = np.sqrt((sig1 * 0) ** 2 + sig2 ** 2 + p * sig2 * sig1)
    term1 = mu1 * stancdf((mu1 - mu2) / theta)
    term2 = mu2 * stancdf((mu2 - mu1) / theta)
    term3 = theta * stannorm((mu1 - mu2) / (theta)/1.5)
    return (-term1 + -term2 + -term3) / 2
def max_gaus_airymod(r, r_zero, sig, norm=1, n=128):
    sigmatel = sig
    bb=200
    integral_range = 2
    mu_V1 = IImodels.airynormoff(r, r_zero, norm)
    zero_exp = quad(maxpdf, -integral_range, integral_range, args=(0, sigmatel, n))[0]
    zero_exp_vari = quad(maxpdfstd, -integral_range, integral_range, args=(0, sigmatel, n))[0]
    zero_std = (zero_exp_vari-zero_exp**2)**.5
    expected_V1 = dualmaxpdf(0,mu_V1, zero_exp, sigmatel, zero_std)*-2
    return expected_V1
def max_gaus_airyLimb_mod(r, r_zero, sig, norm=1, n=128):
    sigmatel = sig
    bb=200
    integral_range = 2
    mu_V1 = IImodels.airynormoff(r, r_zero, norm)
    zero_exp = quad(maxpdf, -integral_range, integral_range, args=(0, sigmatel, n))[0]
    zero_exp_vari = quad(maxpdfstd, -integral_range, integral_range, args=(0, sigmatel, n))[0]
    zero_std = (zero_exp_vari-zero_exp**2)**.5
    expected_V1 = dualmaxpdf(0,mu_V1, zero_exp, sigmatel, zero_std)*-2
    return expected_V1


def data_cutter(targdata,tel_todo,cut=8, renorm=1, tcorlim = 10, SNcut = 4,gwidth = 1.2,g2guess_opd = 64,opd_range = 5,
                doback=True):
    vsquared, skytimes_arrs, utracks, vtracks, telnames, telbaselines, radiofilt_g2s, telopdcorrs, polystds, \
    measstds, origopds_ls, bacratios, airmasses, comvolts, origerr, siifiledirs = targdata
    cutopds = []
    cutopdorig = []
    cutg2s = []
    cutg2sorig = []
    fullopdcorr = []
    cutbaselines = []
    cututracks = []
    cutvtracks = []
    cuttimes = []
    cutv2s = []
    invcovs = []
    raveldats = []
    cuterrs = []
    cutbacratios = []
    cutamasses = []
    cutvolts = []
    cuttimes = []
    cutcomperr = []
    cordata = []
    ravelcordata = []
    mid_shifted_g2 = []
    mid_shift_opd = []
    mid_shifted_g2_orig = []
    mid_shift_opd_orig = []
    opd_toffs = []
    bcors = []
    orig_meth_opd = []
    tel_ids = []
    cut_dirs = []
    # backcorr_arrs = (np.array(
    #     [IItools.calc_back_norm(((bacratios[i] - bacratios[i].min() + back ** 2).mean()) ** .5,
    #                                      origerr[i] ** 1) for i in range(len(bacratios))]))
    # backcorr_arrs = (np.array([(((bacratios[i]-bacratios[i].min())+back**2) ** .5 + 1)**2 for i in range(len(cutbacratios))]))
    backcorr_arrs = []
    minratios = []
    for telnam in tel_todo:
        telind = np.where(np.array(telnames) == telnam)
        minrat = np.concatenate(np.array(bacratios)[telind]).min()
        minratios.append(minrat)
    minratios = np.array(minratios)
    minratiocorr = minratios - minratios.max()

    for i, tel in enumerate(telnames):
        tcor = (telopdcorrs[i] - origopds_ls[i]).mean()
        SN = vsquared[i].mean()/(radiofilt_g2s[i].mean(axis=0).std())
        if np.abs(tcor) < tcorlim and np.abs(SN) > SNcut:
            opd = telopdcorrs[i]
            noise_multiplier = 1.5
        else:
            opd = origopds_ls[i]
            noise_multiplier = 1.5#np.sqrt(2)
        if tel in tel_todo:
            telind = np.where(np.array(tel_todo)==tel)
            cutg2, cutopd = g2_minimapper(radiofilt_g2s[i], opd)
            if len(cutg2[0]) < 10:
                continue

            cutbackrat = (bacratios[i] -minratiocorr[telind])**.5
            bcor = bacratios[i]#(1+cutbackrat)**2
            cutbackrat = (bcor-1)**.5
            cutopds.append(cutopd)
            cutg2s.append(cutg2/renorm)
            cutbaselines.append(telbaselines[i])
            cutvtracks.append(vtracks[i])
            cututracks.append(utracks[i])
            cuttimes.append(skytimes_arrs[i])
            cutv2s.append(vsquared[i]/renorm)
            ravel_g2 = (cutg2/renorm).ravel()
            covmatrix = np.zeros((len(ravel_g2), len(ravel_g2)))
            np.fill_diagonal(covmatrix, (bacratios[i]*noise_multiplier*origerr[i]/renorm) ** 2)
            g2_inv_cov = np.linalg.inv(covmatrix)
            invcovs.append(g2_inv_cov)
            raveldats.append(ravel_g2)
            cuterrs.append(polystds[i]/renorm)
            cutbacratios.append(bacratios[i])
            cutamasses.append(airmasses[i])
            cutvolts.append(comvolts[i])
            cutcomperr.append(origerr[i])
            cordata.append(bcor[:,None]*cutg2/renorm)
            ravelcordata.append((bcor[:,None]*cutg2/renorm).ravel())
            backcorr_arrs.append(cutbackrat)
            cutopdorig.append(origopds_ls[i])
            cutg2sorig.append(radiofilt_g2s[i])
            fullopdcorr.append(telopdcorrs[i])
            bcors.append(bcor)

            shift_g2_mid, round_opd = IItools.g2_shifter_mid(radiofilt_g2s[i], origopds_ls[i])
            amps, cut_opd_correction, cutdata, g2fitpar=\
                IItools.g2_opd_fitter(shift_g2_mid, gwidth,round_opd,telbaselines[i],2)

            shift_g2_mid_cut, round_opd_cut = IItools.g2_shifter_mid(radiofilt_g2s[i], origopds_ls[i]+ g2fitpar[0],cut)
            mid_shifted_g2.append(shift_g2_mid_cut)
            mid_shift_opd.append(round_opd_cut)
            opd_toffs.append(g2fitpar[0])

            shift_g2_mid_mean = shift_g2_mid.mean(axis=0)
            g2opdguess = round_opd.mean()
            xdt = np.arange(len(shift_g2_mid_mean))
            gpar, gerr = curve_fit(gaussian_msub, xdt, shift_g2_mid_mean, p0=[g2opdguess, g2width, 0.1],
                          bounds=[[g2opdguess - 3, g2width, -2], [g2opdguess + 3, g2width * 1.0001, 2]])
            corrected_g2pos = origopds_ls[i] + gpar[0] - round_opd.mean()

            shift_g2_mid_cut_orig, round_opd_cut_orig = IItools.g2_shifter_mid(radiofilt_g2s[i], corrected_g2pos, cut)
            mid_shifted_g2_orig.append(shift_g2_mid_cut_orig)
            mid_shift_opd_orig.append(round_opd_cut_orig)
            orig_meth_opd.append(corrected_g2pos)
            tel_ids.append(tel)
            cut_dirs.append(siifiledirs[i])

            # g2_mean = shift_g2_mid.mean(axis=0)
            # xdt = np.arange(len(g2_mean))
            # gpar, gerr = curve_fit(gaussian_msub, xdt, g2_mean, p0=[g2guess_opd, gwidth, 0.1],
            #                        bounds=[[g2guess_opd - opd_range, gwidth*.999, -2],
            #                                [g2guess_opd + opd_range, gwidth*1.001, 2]])




    return cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
           cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs,ravelcordata, cutv2s, \
           cutopdorig, cutg2sorig, fullopdcorr, mid_shifted_g2, mid_shift_opd, opd_toffs, bcors, orig_meth_opd, \
           mid_shift_opd_orig,mid_shifted_g2_orig, cututracks, cutvtracks, tel_ids, cut_dirs

# plt.imshow(shift_g2_mid)
# corrected_midshift_opd = round_opd + (round_opd.mean() - gpar[0])
# plt.vlines(corrected_midshift_opd.mean(),0,80)
# plt.vlines(round_opd.mean(),0,80,colors='orange')
# plt.vlines(round_opd.mean()+g2fitpar[0],0,80,colors='red')
# plt.plot(np.concatenate(cuttimes),np.concatenate(cutbacratios)**.5, '.', label="uncorrected")
# plt.plot(np.concatenate(cuttimes),np.concatenate(backcorr_arrs), '.', label="corrected")
# plt.ylabel("Background Ratio (I_star / I_background")
# plt.xlabel("Time from midnight (hours)")
# plt.legend(fontsize=14)

if __name__ == "__main__":

    Ts = []
    n = 5
    for ii in range(n):
        for j in range(1, n - ii):
            # pair_delays.append((mes_delays[j + ii] - mes_delays[ii]) / 4)
            Ts.append("T%sT%s" % (ii + 0, ii + j + 0))

    merak_origdir_20211216 = 'D:datasii\\betuma\\20211216'
    merak_origdir_20220211 = 'D:datasii\\betuma\\20220211'
    merak_origdir_20220213 = 'D:datasii\\betuma\\20220213'
    merak_origdir_20220312 = 'D:datasii\\betuma\\20220312'
    merak_origdir_20220510 = 'D:datasii\\betuma\\20220510'
    # merak_reducedir= 'reducedTargets\\OLDbetOLDuma'
    merak_reducedir= 'reducedTargets\\betuma'
    backdir = "D:\\dataSii\\OFFRunsAug2022"

    tausco_20220509 = 'D:datasii\\tausco\\20220509'
    tausco_reduceddir = 'reducedTargets\\tausco'

    etauma_20220516 = "D:datasii\\etauma\\20220516"
    etauma_reduceddir = 'reducedTargets\\etauma'

    betcma_20211119 = "D:\\dataSii\\betcma\\20211119"
    betcma_20220214= "D:\\dataSii\\betcma\\20220214"
    betcma_reduceddir = 'reducedTargets\\betcma'


    merak_data_20220213 = data_getter(merak_origdir_20220213, merak_reducedir, backdir, utc="UTC20220214")
    merak_data_20220211 = data_getter(merak_origdir_20220211, merak_reducedir, backdir, utc="UTC20220212")
    merak_data_20211216 = data_getter(merak_origdir_20211216, merak_reducedir, backdir, utc="UTC20211217")
    merak_data_20220312 = data_getter(merak_origdir_20220312, merak_reducedir, backdir, utc="UTC20220313")
    merak_data_20220510 = data_getter(merak_origdir_20220510, merak_reducedir,backdir, utc="UTC20220511")
    # tausco_data_20220509 = data_getter(tausco_20220509, tausco_reduceddir,backdir, utc="UTC20220510")
    #cutg2s0, cutopds1, cutbaselines2, invcovs3
    # nrm =.65
    nrm =1
    tcorlim, SNcut = 64, .01
    g2width=1.2
    datcut=6

    # cut_merak_data_20220213 = data_cutter(merak_data_20220213,["T2T3","T2T4","T3T4"], renorm=nrm, tcorlim=tcorlim, SNcut=SNcut)

    # cut_merak_data_20220211 = data_cutter(merak_data_20220211,["T3T4"], renorm=nrm, tcorlim=tcorlim, SNcut=SNcut)
    cut_merak_data_20220211 = data_cutter(merak_data_20220211,["T2T3","T2T4","T3T4"], datcut,
                                          renorm=nrm, tcorlim=tcorlim, SNcut=SNcut,gwidth=g2width)
    cut_merak_data_20211216 = data_cutter(merak_data_20211216,["T2T3","T2T4","T3T4"], datcut,
                                          renorm=nrm, tcorlim=tcorlim, SNcut=SNcut,gwidth=g2width)
    cut_merak_data_20220213 = data_cutter(merak_data_20220213,["T2T3","T2T4","T3T4", "T1T2", "T1T3", "T1T4"], datcut,
                                          renorm=nrm, tcorlim=tcorlim, SNcut=SNcut,gwidth=g2width)
    cut_merak_data_20220312 = data_cutter(merak_data_20220312,["T2T3","T2T4","T3T4", "T1T2", "T1T3", "T1T4"], datcut,
                                          renorm=nrm, tcorlim=tcorlim, SNcut=SNcut,gwidth=g2width)
    cut_merak_data_20220510 = data_cutter(merak_data_20220510,["T2T3","T2T4","T3T4", "T1T2", "T1T3", "T1T4"], datcut,
                                          renorm=nrm, tcorlim=tcorlim, SNcut=SNcut,gwidth=g2width)


    asdf=4

    def gaussian(x, mu, sig, amp):
        gausmod = amp * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)
        return gausmod - gausmod.mean()


    def g2_amps_rbr(g2surf, opd):
        gamps = []
        tim = np.arange(len(g2surf[0]))
        for i, rw in enumerate(g2surf):
            fitg, fiterr = curve_fit(gaussian, tim, rw, [opd[i], .85, .2],
                                     bounds=[[(opd[i]) * .999, .85, -10],
                                             [(opd[i]) * 1.0001, .8501, 10]])
            gamps.append(fitg[-1])
        gamps = np.array(gamps)
        return gamps


    nights_todo = [cut_merak_data_20211216, cut_merak_data_20220211, cut_merak_data_20220213]
    nights_todo = [cut_merak_data_20211216, cut_merak_data_20220211, cut_merak_data_20220213, cut_merak_data_20220312, cut_merak_data_20220510]
    allb = []
    allv2 = []
    comp_fact = 1
    doback = True
    allg2s = []
    allv2g2snap = []
    comerrs = []
    allback = []
    alltoffs = []
    for night in nights_todo:
        cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
        cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata, cutv2s, \
        cutopdorig, cutg2sorig, fullopdcorr, mid_shifted_g2, mid_shift_opd, opd_toffs, bcors, orig_meth_opd, \
        mid_shift_opd_orig, mid_shifted_g2_orig, cututracks, cutvtracks, tel_ids, cut_dirs = night
        for i in range(len(cutv2s)):
            if doback:
                backcor = bcors[i]#(1 + backcorr_arrs[i]) ** 2
            else:
                backcor = np.ones(len(cutv2s[i]))
            if comp_fact > 1:
                if comp_fact > len(cutv2s[i]): comp_fact = len(cutv2s[i])
                v2comp = IItools.datcompress(cutv2s[i], comp_fact)
                bascomp = IItools.datcompress(cutbaselines[i], comp_fact)
                backcomp = IItools.datcompress(backcor, comp_fact)
                # allv2.append(g2_amps_rbr(cutg2s[i], cutopds[i])*backcomp)
                shift_g2 = IItools.g2_shifter(cutg2sorig[i], cutopdorig[i])

                g2_mean = shift_g2.mean(axis=0)
                xdt = np.arange(len(g2_mean))
                g2guess = cutopdorig[i].max()
                gpar, gerr = curve_fit(gaussian_msub, xdt, g2_mean, p0=[g2guess, 0.9,0.1], bounds=[[g2guess-3,.9,-2],[g2guess+3,.901,2]])
                corrected_g2pos = (cutopdorig[i]) - (cutopdorig[i]).max() + gpar[0]
                corg2opd_comp = IItools.datcompress(corrected_g2pos, comp_fact)
                g2mapcomp = IItools.datcompress(cutg2sorig[i], comp_fact)
                v2g2_rbr = IItools.g2_amps_rbr(g2mapcomp, corg2opd_comp)

                allv2g2snap.append(v2g2_rbr * backcomp)
                comerrs.append(g2mapcomp.std(axis=1))
                allb.append(bascomp)
                allv2.append(v2comp * backcomp)
                allg2s.append(g2mapcomp)
                allback.append(backcomp)
            else:
                # backcomp = IItools.datcompress(backcor, comp_fact)
                # allv2.append(g2_amps_rbr(cutg2s[i], cutopds[i])*backcomp)
                allv2.append(cutv2s[i] * backcor)
                allb.append(cutbaselines[i])
                allg2s.append(cutg2s[i])
                shift_g2 = IItools.g2_shifter(cutg2sorig[i], cutopdorig[i])
                g2guess = cutopdorig[i].max()
                g2_mean = shift_g2.mean(axis=0)
                xdt = np.arange(len(g2_mean))
                gpar, gerr = \
                    curve_fit(gaussian_msub, xdt, g2_mean, p0=[g2guess, g2width,0.1],
                              bounds=[[g2guess-3,g2width,-2],[g2guess+3,g2width*1.0001,2]])
                corrected_g2pos = (cutopdorig[i]) - (cutopdorig[i]).max() + gpar[0]
                v2g2_rbr = IItools.g2_amps_rbr(cutg2sorig[i], corrected_g2pos)
                orig_meth_opd.append(corrected_g2pos)
                # v2g2_rbr = IItools.g2_amps_rbr(mid_shifted_g2[i], mid_shift_opd[i])
                v2g2_rbr = IItools.g2_amps_rbr(mid_shifted_g2_orig[i], mid_shift_opd_orig[i])
                allv2g2snap.append(v2g2_rbr * backcor)
                comerrs.append(cutg2sorig[i].std(axis=1))
                allback.append(backcor)
                alltoffs.append(opd_toffs[i])


    allcomerrs = np.concatenate(comerrs)
    allbnp = np.concatenate(allb)
    allv2np = np.concatenate(allv2)
    allv2g2snapnp = np.concatenate(allv2g2snap)
    allbacknp = np.concatenate(allback)
    order = 8
    pfitdat = np.poly1d(np.polyfit(allbnp, allv2np, order))
    dat_polymod = pfitdat(allbnp)
    guessstd = 0.05
    guessnorm = 1.2
    guessr0 = 107
    pfmg, permg = curve_fit(max_gaus_airymod, allbnp, allv2np, p0=[guessr0, guessstd, guessnorm],
                            bounds=[[10, 0.001, .0999], [10000, 10, 11.0001]], sigma=allcomerrs)
    mod_mg = max_gaus_airymod(allbnp, *pfmg)
    meas_mas_mg = 1.22 * 416e-9 / pfmg[0] * 2.063e8
    bsort = np.argsort(allbnp)
    fit_per_err = 100 * ((np.diag(permg) ** .5) / pfmg)
    # airymod = IItools.airynormoff(allbnp, pfmg[0],pfmg[1])
    perrs = (np.diag(permg) ** .5)
    highy = max_gaus_airymod(allbnp[bsort], *(pfmg + perrs))
    lowy = IImodels.airynormoff(allbnp[bsort], pfmg[0] - (np.diag(permg) ** .5)[0],
                                        pfmg[2] - (np.diag(permg) ** .5)[2])
    pfsnap, persnap = curve_fit(IImodels.airynormoff, allbnp[bsort], allv2g2snapnp[bsort],
                                p0=[guessr0, guessnorm],
                                bounds=[[10, .0999], [10000, 11.0001]], sigma=allcomerrs[bsort])
    gausSnapAiry = IImodels.airynormoff(allbnp[bsort], *pfsnap)

    meas_mas_snap = 1.22 * 416e-9 / pfsnap[0] * 2.063e8
    fit_per_snap_err = 100 * ((np.diag(persnap) ** .5) / pfsnap)


    # plt.figure(figsize=(14,6))
    # plt.fill_between(allbnp[bsort], lowy, highy, color='grey', alpha=0.5)
    # plt.plot(allbnp[bsort], dat_polymod[bsort], "--", linewidth=3,
    #          label="Secondary Methodology Polynomial fit to data")
    #
    # plt.plot(allbnp[bsort], mod_mg[bsort], linewidth=2,
    #          label='Statistical Model fit to data\nAngular diameter=%.4f +- %.4f%%, \nnorm=%.4f +- %.4f%%, \nsig=%.4f +- %.4f%%'
    #                % (meas_mas_mg, fit_per_err[0], pfmg[2], fit_per_err[1], pfmg[1], fit_per_err[2]))
    # plt.xlabel("Projected Baseline (meter)", fontsize=16)
    # plt.ylabel("VSII [g^2 X 10^-6]", fontsize=16)
    #
    # # plt.plot(allbnp, allv2g2snapnp, '.')
    # pfitdatSNAP = np.poly1d(np.polyfit(allbnp, allv2g2snapnp, order))
    # dat_polymodSNAP = pfitdat(allbnp)
    #
    # pfitdatSNAP = np.poly1d(np.polyfit(allbnp, allv2g2snapnp, order))
    # dat_polymodSNAP = pfitdatSNAP(allbnp)
    # plt.plot(allbnp[bsort], dat_polymodSNAP[bsort], "--", linewidth=3,
    #          label='Original Methodology Polynomial fit to Data')
    # plt.plot(allbnp[bsort], gausSnapAiry, linewidth=2,
    #          label='Uniform disk Fit to data \nAngular diameter=%.4f +- %.4f%% (under estimated), \nN=%.4f +- %.4f%%'
    #                % (meas_mas_snap, fit_per_snap_err[0], pfsnap[1], fit_per_snap_err[1]))
    #
    #
    # plt.legend(fontsize=14)
    # plt.show()


    from cobaya.run import run
    def airy_norm_opdoff_avg(r, norm, opdoff):
        totsprob = 0
        for ni, night in enumerate(nights_todo):
            cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
            cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata, cutv2s, \
            cutopdorig, cutg2sorig, fullopdcorr, mid_shifted_g2, mid_shift_opd, opd_toffs, bcors, orig_meth_opd, \
            mid_shift_opd_orig, mid_shifted_g2_orig, cututracks, cutvtracks, tel_ids, cut_dirs = night
            for i in range(len(cutg2s)):
                ravdata = (mid_shifted_g2_orig[i]*bcors[i][:,None]).ravel()
                g2amps_model = IItools.airynormoff(cutbaselines[i], r, norm)
                g2_surface_model = g2_sig_surface(g2width, g2amps_model, mid_shift_opd_orig[i], mid_shifted_g2_orig[i].shape)
                diff = ravdata - g2_surface_model.ravel()
                dotprob = np.dot(diff, np.dot(invcovs[i], diff)) / 2.
                totsprob = dotprob + totsprob
        return -totsprob

    def airyMG_mmean_multinight_onlyMerak_onenorm_maxgaus_prob(r, norm, sigflr):
        totsprob = 0
        back=0.02
        # norms = [N_20211216, N_20220211, N_20220213]
        for ni,night in enumerate(nights_todo):
            cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
            cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata, cutv2s, \
            cutopdorig, cutg2sorig, fullopdcorr, mid_shifted_g2, mid_shift_opd, opd_toffs, bcors, orig_meth_opd, \
            mid_shift_opd_orig, mid_shifted_g2_orig, cututracks, cutvtracks, tel_ids, cut_dirs = night
            for i in range(len(cutg2s)):
                # sigpred = norm * cuterrs[0].mean() / len(cuterrs[0]) ** .5
                g2amps_model = max_gaus_airymod(cutbaselines[i], r_zero=r, sig=sigflr, norm=norm)
                g2_surface_model = g2_sig_surface_gsub(g2width, g2amps_model, cutopds[i], cutg2s[i].shape)
                diff = ravelcordata[i] - g2_surface_model.ravel()
                dotprob = np.dot(diff, np.dot(invcovs[i], diff)) / 2.
                totsprob = dotprob + totsprob
        print("|",end="")
        # print("%.4f"%(sigflr))
        return -totsprob

    def airyMG_mmean_multinight_limb_maxgaus_prob(r, norm, sigflr):
        totsprob = 0
        for ni,night in enumerate(nights_todo):
            cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
            cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata, cutv2s, \
            cutopdorig, cutg2sorig, fullopdcorr, mid_shifted_g2, mid_shift_opd, opd_toffs, bcors, orig_meth_opd, \
            mid_shift_opd_orig, mid_shifted_g2_orig, cututracks, cutvtracks, tel_ids, cut_dirs = night
            for i in range(len(cutg2s)):
                # sigpred = norm * cuterrs[0].mean() / len(cuterrs[0]) ** .5
                g2amps_model = max_gaus_airyLimb_mod(cutbaselines[i], r_zero=r, sig=sigflr, norm=norm)
                g2_surface_model = g2_sig_surface_gsub(g2width, g2amps_model, cutopds[i], cutg2s[i].shape)
                diff = ravelcordata[i] - g2_surface_model.ravel()
                dotprob = np.dot(diff, np.dot(invcovs[i], diff)) / 2.
                totsprob = dotprob + totsprob
        print("|",end="")
        # print("%.4f"%(sigflr))
        return -totsprob


    target_name ="AAAAbetuma_t2t3t4_limb"
    info_limb = {"likelihood": {"external": airyMG_mmean_multinight_limb_maxgaus_prob}}
    info_limb["params"] = {
        "norm": {"prior": {"min": .01, "max": 4}, "ref": pfmg[2], "proposal": .01},
        "r": {"prior": {"min": 50, "max": 1000}, "ref": pfmg[0], "proposal": 5},
        "sigflr": {"prior": {"min": .01, "max": 1}, "ref": pfmg[1], "proposal": .01},
        "AngD": {"derived": get_angular_diameter, "latex": r"\theta"},
    }
    opdoffset = 0
    save_dir = "cobayaChains\\%s\\gaussianFit_OrigOPDCorrMethod2%.4f_Norm" % (target_name, opdoffset)


    info_limb['output'] = save_dir
    info_limb['sampler'] = {'mcmc': {"Rminus1_stop": 0.005, "max_tries": 10000}}
    # info_limb['force'] = True
    info_limb['resume'] = True
    update_info_airy_anal_limb, sampler_airy_anal_limb = run(info_limb)
    affffsdf=4040


    # testr = np.linspace(90, 120, 10)
    # probtest = np.array([airy_norm_opdoff_avg(r, 1.1,0) for r in testr])
    target_name ="AAAA_Fin_betuma_multinight_t2t3t4"
    info = {"likelihood": {"external": airy_norm_opdoff_avg}}
    # info["params"] = {
    #     "r": {"prior": {"min": 50, "max": 1000}, "ref": pfsnap[0], "proposal": 5},
    #     "norm": {"prior": {"min": .01, "max": 4}, "ref": pfsnap[1], "proposal": .05},
    #     "opdoff": {"prior": {"min": -3, "max": 3}, "ref": 0, "proposal": .2},
    #     "AngD": {"derived": get_angular_diameter},
    # }
    # save_dir = "cobayaChains\\%s\\gaussianFit_OpdOff_Norm" % (target_name)
    info["params"] = {
        "r": {"prior": {"min": 50, "max": 1000}, "ref": pfsnap[0], "proposal": 5},
        "norm": {"prior": {"min": .01, "max": 4}, "ref": pfsnap[1], "proposal": .05},
        "opdoff": opdoffset,
        "AngD": {"derived": get_angular_diameter, "latex": r"\theta"},
    }
    save_dir = "cobayaChains\\%s\\gaussianFit_OrigOPDCorrMethod2%.4f_Norm" % (target_name, opdoffset)


    info['output'] = save_dir
    info['sampler'] = {'mcmc': {"Rminus1_stop": 0.005, "max_tries": 10000}}
    # info['force'] = True
    info['resume'] = True
    update_info_airy_anal, sampler_airy_anal = run(info)
    asdf=4



    infom = {"likelihood": {"external": airyMG_mmean_multinight_onlyMerak_onenorm_maxgaus_prob}}
    infom["params"] = {
        "norm": {"prior": {"min": .01, "max": 4}, "ref": pfmg[2], "proposal": .01},
        "r": {"prior": {"min": 50, "max": 1000}, "ref": pfmg[0], "proposal": 5},
        "sigflr": {"prior": {"min": .01, "max": 1}, "ref": pfmg[1], "proposal": .01},
        "AngD": {"derived": get_angular_diameter},
    }

    save_dir = "cobayaChains\\%s\\MaxOPDCorrMethod2%.4f_Norm" % (target_name, opdoffset)
    infom['output'] = save_dir
    infom['sampler'] = {'mcmc': {"Rminus1_stop": 0.005, "max_tries": 10000}}
    # infom['force'] = True
    info['resume'] = True
    update_info_airy_analm, sampler_airy_analm = run(infom)
    asdf=3




osu_dirs=\
    ["/betUMa/UTC20211217/m12d16h23/T1T2",
"/betUMa/UTC20211217/m12d16h23/T2T3",
"/betUMa/UTC20211217/m12d16h23/T3T4",
"/betUMa/UTC20211217/m12d16h23/T3T4",
"/betUMa/UTC20211217/m12d16h23/T3T4",
"/betUMa/UTC20211217/m12d17h01/T3T4",
"/betUMa/UTC20211217/m12d17h03/T3T4",
"/betUMa/UTC20220212/m02d11h19/T3T4",
"/betUMa/UTC20220212/m02d11h20/T3T4",
"/betUMa/UTC20220212/m02d11h20/T3T4",
"/betUMa/UTC20220212/m02d11h23/T3T4",
"/betUMa/UTC20220214/m02d13h19/T1T2",
"/betUMa/UTC20220214/m02d13h19/T2T3",
"/betUMa/UTC20220214/m02d13h19/T3T4",
"/betUMa/UTC20220214/m02d13h19/T3T4",
"/betUMa/UTC20220214/m02d13h21/T1T2",
"/betUMa/UTC20220214/m02d13h21/T3T4",
"/betUMa/UTC20220214/m02d13h21/T3T4",
"/betUMa/UTC20220214/m02d14h03/T2T4",
"/betUMa/UTC20220313/m03d12h22/T3T4",
"/betUMa/UTC20220313/m03d12h22/T3T4"]


from datetime import datetime
osu_dirs = np.loadtxt("UVPlaneFiles.txt", dtype=np.str,delimiter='\n')
osudatimes = []
for d in osu_dirs:
    format_str = 'UTC%Y%m%dXm%Hd%Mh%S'
    format_str = "y%Ym%md%dh%Hm%Ms%S"
    dandt = d.split('_')[1:3][::-1]
    datetime_obj = datetime.strptime(dandt[0], format_str)
    if dandt[1] == "T4T3":
        dandt[1] = "T3T4"
    osudatimes.append([datetime_obj, dandt[1]])
import pandas as pd
pdates = pd.Series(osudatimes)
counts = pdates.value_counts()
alldirs = ['/'.join((d*1).split('\\')[2:][::-1]).split('/') for d in np.concatenate([d[-1] for d in nights_todo])]
color_lines = np.array(["red", "green", "blue", "orange", "magenta", "navy"])
telpairs = np.array(["T1T2", "T1T3", "T1T4", "T2T3", "T2T4", "T3T4"])
mark_shapes = ['o','d','s','X','P','*']
plt.figure(figsize=(10, 10))
for i in range(len(telpairs)):
    plt.plot(1000, 1000, linestyle = "-", marker=mark_shapes[i], markersize=10,linewidth=3, color=color_lines[i], label=telpairs[i])
# plt.plot(0, 0, linestyle = "-", marker="+", markersize=15,markeredgewidth=10,linewidth=3, color="black")
plt.legend(fontsize=14)
plt.xlim([-180, 180])
plt.ylim([-180, 180])
from datetime import datetime
for night in nights_todo:
    cutg2s, cutopds, cutbaselines, invcovs, raveldats, cuterrs, cutbacratios, \
    cutamasses, cutvolts, cuttimes, cutcomperr, cordata, backcorr_arrs, ravelcordata, cutv2s, \
    cutopdorig, cutg2sorig, fullopdcorr, mid_shifted_g2, mid_shift_opd, opd_toffs, bcors, orig_meth_opd, \
    mid_shift_opd_orig, mid_shifted_g2_orig, cututracks, cutvtracks, tel_ids, cut_dirs = night
    for i, tel in enumerate(tel_ids):
        format_str = '%Yx%mx%dX%Hx%Mx%S'
        datandtel = cut_dirs[i].split("\\")[-2:][::-1]
        datetime_obj = datetime.strptime(datandtel[0], format_str)
        if datandtel[1] == "T4T3":
            datandtel[1] = "T3T4"
        fulldandtel = [datetime_obj, datandtel[1]]
        if fulldandtel in osudatimes:
            dupcount = osudatimes.count(fulldandtel)
            if dupcount > 1:
                utrack = np.array_split(cututracks[i], dupcount)
                vtrack = np.array_split(cutvtracks[i], dupcount)
                for j in range(dupcount):
                    colorind = np.argwhere(telpairs == tel)[0][0]
                    plt.plot(utrack[j], vtrack[j], color=color_lines[colorind], linewidth=3)
                    plt.plot(utrack[j][int(len(vtrack[j]) / 2)], vtrack[j][int(len(vtrack[j]) / 2)],
                             marker=mark_shapes[colorind], markersize=14, color=color_lines[colorind], linewidth=3)
                    plt.plot(-utrack[j], -vtrack[j], color=color_lines[colorind], linewidth=3)
                    plt.plot(-utrack[j][int(len(utrack[j]) / 2)], -vtrack[j][int(len(vtrack[j]) / 2)],
                             marker=mark_shapes[colorind], markersize=14, color=color_lines[colorind], linewidth=3)
            else:
                colorind = np.argwhere(telpairs == tel)[0][0]
                plt.plot(cututracks[i], cutvtracks[i], color=color_lines[colorind], linewidth=3)
                plt.plot(cututracks[i][int(len(cututracks[i]) / 2)], cutvtracks[i][int(len(cututracks[i]) / 2)], marker=mark_shapes[colorind], markersize=14,
                         color=color_lines[colorind], linewidth=3)
                plt.plot(-cututracks[i], -cutvtracks[i], color=color_lines[colorind], linewidth=3)
                plt.plot(-cututracks[i][int(len(cututracks[i]) / 2)], -cutvtracks[i][int(len(cututracks[i]) / 2)], marker=mark_shapes[colorind], markersize=14,
                         color=color_lines[colorind], linewidth=3)
        else:
            colorind = np.argwhere(telpairs == tel)[0][0]
            plt.plot(cututracks[i], cutvtracks[i], color='gray', linewidth=3, zorder=0)
            plt.plot(cututracks[i][int(len(cututracks[i]) / 2)], cutvtracks[i][int(len(cututracks[i]) / 2)],
                     marker=mark_shapes[colorind], markersize=14,
                     color='gray', linewidth=3, zorder=0)
            plt.plot(-cututracks[i], -cutvtracks[i], color='gray', linewidth=3, zorder=0)
            plt.plot(-cututracks[i][int(len(cututracks[i]) / 2)], -cutvtracks[i][int(len(cututracks[i]) / 2)],
                     marker=mark_shapes[colorind], markersize=14,
                     color='gray', linewidth=3, zorder=0)
    # plt.legend()
    # break
plt.axis('equal')
plt.xlabel("U (meter)", fontsize=20)
plt.ylabel("V (meter)", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16, width=4, length=8)
plt.grid(which='both', linewidth=3)
# rcut = -95
# theta = np.linspace(0, 2*np.pi, 100)
# xc = rcut*np.cos(theta)
# yc = rcut*np.sin(theta)
inner = 82
outer = 1000
xc = np.linspace(-outer, outer, 1000, endpoint=True)
y0 = outer*np.sin(np.arccos(xc/outer)) # x-axis values -> outer circle
yI = inner*np.sin(np.arccos(xc/inner))
yI[np.isnan(yI)] = 0.
# plt.fill_between(xc, yc, color='gray', alpha=0.3)
plt.fill_between(xc, yI, y0, color='gray', alpha=0.3)
plt.fill_between(xc, -y0, -yI, color='gray', alpha=0.3)
# plt.fill(xc, -yc, color='gray', alpha=0.3)
# plt.fill_between(xc, yc, color='gray', alpha=0.3)
plt.xlim([-180,180])
plt.ylim([-180,180])
graphdir = "analysisPlots\\paperPlots"

