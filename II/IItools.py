import astropy
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import astropy.units as u
from astropy.time import Time
from II import IImodels
from astroquery.vizier import Vizier




def radial_profile(data, center=None):
    if center == None:
        center = (data.shape[0] / 2, data.shape[1] / 2)

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def proj_baseline(lat, dec, hour):

    ar = np.array
    Bp1 = ar([ar([-np.sin(lat) * np.sin(hour), np.cos(hour), np.cos(lat) * np.sin(hour)]),

              ar([np.sin(lat) * np.cos(hour) * np.sin(dec) + np.cos(lat) * np.cos(dec), np.sin(hour) * np.sin(dec),
                  -np.cos(lat) * np.cos(hour) * np.sin(dec) + np.sin(lat) * np.cos(dec)]),

              ar([-np.sin(lat) * np.cos(hour) * np.cos(dec) + np.cos(lat) * np.sin(dec),
                  -np.sin(hour) * np.cos(dec),
                  np.cos(lat) * np.cos(hour) * np.cos(dec) + np.sin(lat) * np.sin(dec)])])

    return Bp1


def uv_tracks(lat, dec, hours, Bn, Be, Bu):

    baselines = np.transpose(np.array([Bn, Be, Bu]))
    track = np.array([np.dot(proj_baseline(lat,dec,hour), baselines) for hour in hours])

    ref_track = np.array([np.dot(proj_baseline(lat, dec, hour), -baselines) for hour in hours])
    return track, ref_track

def array_baselines(tel_locs):
    n = np.alen(tel_locs)
    N = n*(n-1)/2
    baselines = []

    for i in range(n):
        for j in range(1,n-i):
            baselines.append(tel_locs[i] - tel_locs[i+j])
    return baselines


def airy_disk2D(shape, xpos, ypos, r):
    from astropy.modeling.functional_models import AiryDisk2D
    from astropy.modeling import models, fitting

    y, x = np.mgrid[:shape[0], :shape[1]]
    # Fit the data using astropy.modeling
    p_init = AiryDisk2D(x_0=xpos, y_0=ypos, radius=r)
    fit_p = fitting.LevMarLSQFitter()
    return p_init(x,y), p_init


def gaussian_disk2D():
    from astropy.modeling import models
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.modeling import models, fitting

    np.random.seed(0)
    x = np.linspace(-5., 5., 200)
    y = 3 * np.exp(-0.5 * (x - 1.3) ** 2 / 0.8 ** 2)
    y += np.random.normal(0., 0.2, x.shape)

    # Generate fake data
    np.random.seed(0)
    y, x = np.mgrid[:128, :128]
    z = 3 * np.exp(-.0001 * (y-50)**2 * (x - 50) ** 2 / 0.8 ** 2)
    z += np.random.normal(0., 0.1, z.shape) * 3

    # Fit the data using astropy.modeling
    p_init = models.Gaussian2D(amplitude=1, x_mean=1, y_mean=1)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)

    # Plot the data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(z, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
    plt.title("Residual")



def track_coverage(tel_tracks, airy_func):
    x_0 = airy_func.x_0.value
    y_0 = airy_func.y_0.value
    ranges = []
    amps = []
    r_0 = airy_func.radius.value
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        airy_amp = airy_func(utrack, vtrack)
        airy_radius = np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2)
        amps.append(airy_amp)
        ranges.append([np.min(airy_radius), np.max(airy_radius)])

    merged_ranges = interval_merger(ranges)
    r0_cov = 0
    r1_cov = 0
    r2_cov = 0
    r0_amp = 0
    r1_amp = 0
    for ran in merged_ranges:
        r0_cov = r0_cov + np.ptp(getIntersection(ran,[0,r_0]))
        r1_cov = r1_cov + np.ptp(getIntersection(ran,[r_0,2*r_0]))
        r2_cov = r2_cov + np.ptp(getIntersection(ran,[2*r_0,3*r_0]))
    if r0_cov > 0:
        r0_amp = curve_amplitude(merged_ranges,0,r_0,airy_func,x_0,y_0)
    if r1_cov > 0:
        r1_amp = curve_amplitude(merged_ranges,r_0,r_0*2,airy_func,x_0,y_0)
    r_amp = np.ptp(amps)

    return r0_cov/r_0, r1_cov/r_0, r2_cov/r_0, r0_amp, r1_amp, r_amp

def curve_amplitude(ranges, st, end, airy_func, x_0, y_0):
    r0_range = [getIntersection(rang, [st, end]) for rang in ranges if getIntersection(rang, [st, end]) is not 0]
    minr = np.min(r0_range)
    maxr = np.max(r0_range)
    high = airy_func(y_0, minr + x_0)
    low = airy_func(y_0, maxr + x_0)
    return high - low

def track_error(sig0, m0, m, t0, t):
    return sig0 * (2.512) ** (m - m0) * (t0 / t) ** .5

def trap_w_err(numerical_f, r, erra, errb):
    fa = numerical_f[:-1]
    fb = numerical_f[1:]
    ra = r[:-1]
    rb = r[1:]
    C = (fa + fb)/2
    dr = rb-ra
    I = dr * C
    Ierr = dr*np.sqrt(erra**2 + errb*2)*C
    return I, Ierr, dr

def trapezoidal_average(num_f):
    if np.alen(num_f)>1:
        fa = num_f[:-1]
        fb = num_f[1:]
        func_avg = (fa + fb)/2
    else:
        func_avg = np.mean(num_f)
    return func_avg


def interval_merger(intervals):
    sint = sorted(intervals, key=lambda i: i[0])
    out = [sint[0]]
    for current in sorted(intervals, key=lambda i: i[0]):
        previous = out[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            out.append(current)
    return out

def getIntersection(interval_1, interval_2):
    st = np.max([interval_1[0], interval_2[0]])
    end = np.min([interval_1[1], interval_2[1]])
    if st < end:
        return [st, end]
    return 0

def IIbootstrap_analysis(tel_tracks, airy_func, star_err, guess_r, wavelength, runs):
    fitdiams = []
    fiterrs = []
    dfs = []
    derrs = []
    failed_fits = 0
    for i in range(0, runs):
        rads, amps, avgrad, avgamp = IImodels.avg_air1D(tel_tracks=tel_tracks,
                                                          airy_func=airy_func,
                                                          err=star_err)


        airy_fitr, airy_fiterr, df,der, sigmas = IImodels.fit_airy_avg(rads=rads,
                                     amps=amps,
                                     avg_rads=avgrad,
                                     avg_amps=avgamp + np.random.normal(0,star_err, avgamp.shape),
                                     func=airy_func,
                                     err=star_err,
                                     guess_r=guess_r+ np.random.normal(0,guess_r/5),
                                     real_r=guess_r)
        # airy_fitr, airy_fiterr = IImodels.fit_airy1D(rx=tr_Irad[rsort],
        #                                              airy_amp=tr_Ints[rsort] + tr_aerrs[rsort],
        #                                              guess_r=guess_r + np.random.normal(0,guess_r/5),
        #                                              errs=np.full(np.alen(tr_Ints), star_err))

        if np.abs(airy_fitr[0]) > guess_r * 15 or airy_fiterr > guess_r * 15:
            failed_fits += 1
            continue
        else:
            fit_diam = (wavelength.to('m').value / airy_fitr[0] * u.rad).to('mas')
            fit_err = np.sqrt(np.diag(airy_fiterr))[0] / airy_fitr[0] * fit_diam
            fitdiams.append(fit_diam.value)
            fiterrs.append(fit_err.value)
            dfs.append(((wavelength.to('m').value / df[0] * u.rad).to('mas')).value)

    try:
        smartfiterr = np.mean(fitdiams)/((wavelength.to('m').value / guess_r * u.rad).to('mas')).value
        airyfiterr = np.mean(dfs)/((wavelength.to('m').value / guess_r * u.rad).to('mas')).value
        print("smartfiterr %s, airyfit %s"%(smartfiterr, airyfiterr))
    except:
        print('everything failed')
    return np.array(fitdiams), np.array(fiterrs), failed_fits


